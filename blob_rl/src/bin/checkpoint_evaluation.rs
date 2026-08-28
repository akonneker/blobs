#![recursion_limit = "512"]
//! Publish held-out retention, contact, and micro-combat evidence for a frozen PPO checkpoint.

use std::path::PathBuf;

use blob_rl::artifact::{load_policy_snapshot, verify_checkpoint_metadata};
use blob_rl::checkpoint_evaluation::{
    publish_checkpoint_evaluation, CheckpointEvaluationArtifact, CombatEvaluationHorizons,
};
use blob_rl::contact_evaluation::evaluate_contact;
use blob_rl::feeding_curriculum::evaluate_feeding_promotion;
use blob_rl::micro_combat::evaluate_micro_combat;
use burn::prelude::*;
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "checkpoint-evaluation",
    about = "Publish hash-bound feeding-retention, contact, and micro-combat evidence for a PPO checkpoint"
)]
struct Args {
    /// Immutable training checkpoint directory.
    checkpoint: PathBuf,

    /// Held-out episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// New immutable evaluation JSON file.
    #[arg(long)]
    output: PathBuf,

    /// Exit unsuccessfully after publishing valid evidence that fails any active gate.
    #[arg(long)]
    require_pass: bool,

    /// Override only the held-out direct-contact simulation horizon.
    #[arg(long)]
    contact_sim_time_limit_quanta: Option<u64>,

    /// Override only the held-out skirmish simulation horizon.
    #[arg(long)]
    skirmish_sim_time_limit_quanta: Option<u64>,
}

fn run<B: Backend>(args: &Args, device: B::Device) -> bool
where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    let metadata = verify_checkpoint_metadata(&args.checkpoint)
        .unwrap_or_else(|error| panic!("invalid checkpoint: {error}"));
    let snapshot = load_policy_snapshot::<B>(&args.checkpoint, &device)
        .unwrap_or_else(|error| panic!("failed to load checkpoint policy: {error}"));
    let mut evaluation_config = metadata.config.clone();
    let mut combat_evaluation_horizons = CombatEvaluationHorizons::from_config(&metadata.config);
    if let Some(horizon) = args.contact_sim_time_limit_quanta {
        combat_evaluation_horizons.contact_sim_time_limit_quanta = horizon;
    }
    if let Some(horizon) = args.skirmish_sim_time_limit_quanta {
        combat_evaluation_horizons.skirmish_sim_time_limit_quanta = horizon;
    }
    combat_evaluation_horizons
        .apply_to(&mut evaluation_config)
        .unwrap_or_else(|error| panic!("invalid combat evaluation horizons: {error}"));
    let metadata_bytes = std::fs::read(args.checkpoint.join("metadata.json"))
        .unwrap_or_else(|error| panic!("failed to read checkpoint metadata: {error}"));
    let metadata_sha256 = format!("{:x}", Sha256::digest(&metadata_bytes));
    let feeding = evaluate_feeding_promotion(
        &snapshot.model,
        &metadata.config.env,
        &metadata.config.reward,
        &metadata.config.feeding_curriculum,
        &args.seeds,
        &device,
    );
    let contact = evaluate_contact(
        &snapshot.model,
        &metadata.config.env,
        &metadata.config.reward,
        &metadata.config.feeding_curriculum,
        &evaluation_config.combat_curriculum,
        &args.seeds,
        &device,
    );
    let micro_combat = metadata
        .config
        .combat_curriculum
        .micro_combat
        .enabled
        .then(|| {
            evaluate_micro_combat(
                &snapshot.model,
                &metadata.config.env,
                &metadata.config.reward,
                metadata
                    .config
                    .combat_curriculum
                    .micro_combat
                    .suite
                    .as_ref()
                    .expect("validated micro-combat suite exists"),
                &args.seeds,
                &device,
            )
        });
    let artifact = CheckpointEvaluationArtifact::new(
        metadata_sha256,
        snapshot.model_sha256,
        snapshot.update,
        snapshot.actions,
        combat_evaluation_horizons,
        metadata.config,
        feeding,
        contact,
        micro_combat,
    )
    .unwrap_or_else(|error| panic!("invalid checkpoint evaluation: {error}"));
    publish_checkpoint_evaluation(&args.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish checkpoint evaluation: {error}"));
    println!(
        "Published checkpoint u{} evaluation {} to {} (artifact {}, contact/skirmish horizons {}/{}):",
        artifact.checkpoint_update,
        if artifact.passed() { "PASS" } else { "FAIL" },
        args.output.display(),
        artifact.artifact_hash,
        artifact
            .combat_evaluation_horizons
            .contact_sim_time_limit_quanta,
        artifact
            .combat_evaluation_horizons
            .skirmish_sim_time_limit_quanta,
    );
    if let Some(micro) = &artifact.micro_combat {
        let gate = micro.gate_summary();
        println!(
            "  micro: survival {:.1}%, elimination {:.1}%",
            gate.survival_success_rate * 100.0,
            gate.elimination_success_rate * 100.0,
        );
    }
    for stage in &artifact.feeding.stages {
        println!(
            "  {}: success {:.1}%, survival {:.1}%, intake {:.3}/initial cell",
            stage.stage,
            stage.episode_success_rate * 100.0,
            stage.survival_rate * 100.0,
            stage.consumed_energy_per_initial_cell,
        );
    }
    println!(
        "  contact: attacks {} / {}, damage {}, kills {}, damaging episodes {:.1}%",
        artifact.contact.attacks_succeeded,
        artifact.contact.attacks_committed,
        artifact.contact.damage_dealt,
        artifact.contact.kills,
        artifact.contact.damaging_episode_rate * 100.0,
    );
    artifact.passed()
}

fn main() {
    let args = Args::parse();
    if args.seeds.is_empty() {
        panic!("checkpoint evaluation requires at least one seed");
    }
    let require_pass = args.require_pass;

    #[cfg(feature = "wgpu")]
    let passed = run::<burn::backend::Wgpu>(&args, burn::backend::wgpu::WgpuDevice::default());

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let passed = run::<burn::backend::NdArray<f32>>(&args, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("checkpoint-evaluation requires the wgpu or ndarray feature");

    if require_pass && !passed {
        std::process::exit(2);
    }
}
