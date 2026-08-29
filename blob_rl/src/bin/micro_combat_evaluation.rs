#![recursion_limit = "512"]
//! Publish hash-bound 1v1/1vN combat evidence for a behavior clone.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, verify_behavior_clone_artifact, BehaviorCloningArtifact,
};
use blob_rl::config::TrainingConfig;
use blob_rl::micro_combat::{
    evaluate_micro_combat, MicroCombatEvaluationReport, MicroCombatSuiteConfig,
    MICRO_COMBAT_EVIDENCE_SCHEMA_VERSION,
};
use blob_rl::model::PolicyValueNetConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "micro-combat-evaluation",
    about = "Publish held-out asymmetric micro-combat evidence for a behavior clone"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    scenarios: PathBuf,

    #[arg(long)]
    behavior_clone: PathBuf,

    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    #[arg(long)]
    output: PathBuf,

    /// Exit with status 2 after publishing when any configured objective has
    /// zero successful episodes.
    #[arg(long)]
    require_all_objectives: bool,

    /// Exit with status 2 after publishing when the greedy learned policy
    /// commits fewer attacks per evaluated scenario episode than this floor.
    #[arg(long)]
    require_attack_commitments_per_episode: Option<f64>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct MicroCombatEvidence {
    schema_version: u32,
    package_version: String,
    source_config_sha256: String,
    scenario_config_sha256: String,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    report: MicroCombatEvaluationReport,
}

struct EvaluationInputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    suite: MicroCombatSuiteConfig,
    scenario_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    seeds: Vec<u64>,
    output: &'a Path,
}

fn run<B: Backend>(inputs: EvaluationInputs<'_>, device: B::Device) -> (bool, f64)
where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    let model_path = verify_behavior_clone_artifact(
        inputs.behavior_clone,
        &inputs.behavior_clone_metadata_sha256,
        &inputs.config.model,
    )
    .unwrap_or_else(|error| panic!("invalid behavior clone: {error}"));
    let model = PolicyValueNetConfig {
        hidden1: inputs.config.model.hidden1,
        hidden2: inputs.config.model.hidden2,
        recurrent_size: inputs.config.model.recurrent_size,
    }
    .init::<B>(&device)
    .load_file(model_path, &CompactRecorder::new(), &device)
    .unwrap_or_else(|error| panic!("failed to load behavior clone: {error}"));
    let report = evaluate_micro_combat(
        &model,
        &inputs.config.env,
        &inputs.config.reward,
        &inputs.suite,
        &inputs.seeds,
        &device,
    );
    report
        .validate_against(&report.ruleset_hash, &inputs.suite, &inputs.seeds)
        .unwrap_or_else(|error| panic!("invalid generated micro-combat evidence: {error}"));
    let objectives_passed = report
        .scenarios
        .iter()
        .all(|metrics| metrics.objective_successes > 0);
    let attack_commitments_per_episode = report.gate_summary().attack_commitments_per_episode;
    let evidence = MicroCombatEvidence {
        schema_version: MICRO_COMBAT_EVIDENCE_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        source_config_sha256: inputs.source_config_sha256,
        scenario_config_sha256: inputs.scenario_config_sha256,
        behavior_clone_metadata_sha256: inputs.behavior_clone_metadata_sha256,
        behavior_clone_model_sha256: inputs.behavior_clone_model_sha256,
        report,
    };
    if let Some(parent) = inputs.output.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    let mut bytes = serde_json::to_vec_pretty(&evidence)
        .unwrap_or_else(|error| panic!("failed to encode micro-combat evidence: {error}"));
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(inputs.output)
        .unwrap_or_else(|error| {
            panic!(
                "refusing to replace micro-combat evidence {}: {error}",
                inputs.output.display()
            )
        });
    file.write_all(&bytes)
        .unwrap_or_else(|error| panic!("failed to write micro-combat evidence: {error}"));
    file.sync_all()
        .unwrap_or_else(|error| panic!("failed to sync micro-combat evidence: {error}"));
    println!(
        "Published {} micro-combat scenarios to {} (SHA-256 {:x}); greedy attacks {:.3} per episode",
        evidence.report.scenarios.len(),
        inputs.output.display(),
        Sha256::digest(&bytes),
        attack_commitments_per_episode,
    );
    for metrics in &evidence.report.scenarios {
        println!(
            "  {} ({:?}): objective {:.1}%, survival {:.1}%, W/L/T {}/{}/{}, damage dealt/received {}/{}, kills {}",
            metrics.scenario,
            metrics.objective,
            metrics.objective_success_rate * 100.0,
            metrics.scientific_survival_rate * 100.0,
            metrics.wins,
            metrics.losses,
            metrics.timeouts,
            metrics.training_damage_dealt,
            metrics.training_damage_received,
            metrics.training_kills,
        );
    }
    (objectives_passed, attack_commitments_per_episode)
}

fn main() {
    let args = Args::parse();
    if args.seeds.is_empty() {
        panic!("micro-combat evaluation requires at least one seed");
    }
    if args
        .require_attack_commitments_per_episode
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        panic!("required attack commitments per episode must be finite and nonnegative");
    }
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid micro-combat base config: {error}"));

    let scenario_bytes = fs::read(&args.scenarios)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.scenarios.display()));
    let scenario_text = std::str::from_utf8(&scenario_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.scenarios.display()));
    let suite = MicroCombatSuiteConfig::from_toml_str(scenario_text)
        .unwrap_or_else(|error| panic!("invalid micro-combat scenarios: {error}"));
    suite
        .validate_against(&config.env)
        .unwrap_or_else(|error| panic!("micro-combat scenarios do not fit base config: {error}"));

    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));
    let inputs = EvaluationInputs {
        config,
        source_config_sha256: format!("{:x}", Sha256::digest(&config_bytes)),
        suite,
        scenario_config_sha256: format!("{:x}", Sha256::digest(&scenario_bytes)),
        behavior_clone: &args.behavior_clone,
        behavior_clone_metadata_sha256: metadata_sha256,
        behavior_clone_model_sha256: metadata.model_sha256,
        seeds: args.seeds,
        output: &args.output,
    };
    let require_all_objectives = args.require_all_objectives;

    #[cfg(feature = "wgpu")]
    let (objectives_passed, attack_commitments_per_episode) =
        run::<burn::backend::Autodiff<burn::backend::Wgpu>>(
            inputs,
            burn::backend::wgpu::WgpuDevice::default(),
        );

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let (objectives_passed, attack_commitments_per_episode) =
        run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(inputs, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("micro-combat-evaluation requires the wgpu or ndarray feature");

    let attacks_passed = args
        .require_attack_commitments_per_episode
        .is_none_or(|minimum| attack_commitments_per_episode >= minimum);
    if require_all_objectives && !objectives_passed || !attacks_passed {
        std::process::exit(2);
    }
}
