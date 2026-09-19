#![recursion_limit = "512"]
//! Publish held-out local-combat characterization for a behavior clone.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, BehaviorCloningArtifact};
use blob_rl::config::TrainingConfig;
use blob_rl::contact_evaluation::{evaluate_contact, ContactEvaluationReport};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::prelude::*;
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "contact-evaluation",
    about = "Publish hash-bound direct-contact and skirmish behavior for a behavior clone"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    behavior_clone: PathBuf,

    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    #[arg(long)]
    output: PathBuf,

    /// Exit unsuccessfully after publishing evidence with no successful damage.
    #[arg(long)]
    require_damage: bool,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ContactEvidence {
    schema_version: u32,
    source_config_sha256: String,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    report: ContactEvaluationReport,
}

struct EvaluationInputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    seeds: Vec<u64>,
    output: &'a Path,
}

fn run<B: Backend>(inputs: EvaluationInputs<'_>, device: B::Device) -> bool
where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    let model = load_behavior_clone::<B>(
        inputs.behavior_clone,
        &inputs.behavior_clone_metadata_sha256,
        &inputs.config.model,
        &device,
    )
    .unwrap_or_else(|error| panic!("failed to load behavior clone: {error}"));
    let report = evaluate_contact(
        &model,
        &inputs.config.env,
        &inputs.config.reward,
        &inputs.config.feeding_curriculum,
        &inputs.config.combat_curriculum,
        &inputs.seeds,
        &device,
    );
    let active = report.is_active();
    let evidence = ContactEvidence {
        schema_version: 2,
        source_config_sha256: inputs.source_config_sha256,
        behavior_clone_metadata_sha256: inputs.behavior_clone_metadata_sha256,
        behavior_clone_model_sha256: inputs.behavior_clone_model_sha256,
        report,
    };
    if let Some(parent) = inputs.output.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    let mut bytes = serde_json::to_vec_pretty(&evidence)
        .unwrap_or_else(|error| panic!("failed to encode contact evidence: {error}"));
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(inputs.output)
        .unwrap_or_else(|error| {
            panic!(
                "refusing to replace contact evidence {}: {error}",
                inputs.output.display()
            )
        });
    file.write_all(&bytes)
        .unwrap_or_else(|error| panic!("failed to write contact evidence: {error}"));
    file.sync_all()
        .unwrap_or_else(|error| panic!("failed to sync contact evidence: {error}"));
    let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
    println!(
        "Published {} contact episodes to {} (SHA-256 {}): attacks {} committed / {} succeeded, {:.1}% attacking episodes, {:.1}% damaging episodes, {} damage, {} kills",
        evidence.report.episodes,
        inputs.output.display(),
        file_sha256,
        evidence.report.attacks_committed,
        evidence.report.attacks_succeeded,
        evidence.report.attacking_episode_rate * 100.0,
        evidence.report.damaging_episode_rate * 100.0,
        evidence.report.damage_dealt,
        evidence.report.kills,
    );
    for variant in &evidence.report.variants {
        println!(
            "  {} {}v{} energy {} vs {}: attack {:.1}%, damage {:.1}%, success {:.1}%, damage {}, kills {}, W/L/T {}/{}/{}",
            variant.stage,
            variant.cells_per_team,
            variant.cells_per_team,
            variant.initial_energy,
            variant.opponent,
            variant.attacking_episode_rate * 100.0,
            variant.damaging_episode_rate * 100.0,
            variant.attack_success_rate * 100.0,
            variant.damage_dealt,
            variant.kills,
            variant.wins,
            variant.losses,
            variant.timeouts,
        );
    }
    active
}

fn main() {
    let args = Args::parse();
    if args.seeds.is_empty() {
        panic!("contact evaluation requires at least one seed");
    }
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid contact-evaluation config: {error}"));
    let source_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));
    let require_damage = args.require_damage;

    #[cfg(feature = "wgpu")]
    {
        let active = run::<burn::backend::Autodiff<burn::backend::Wgpu>>(
            EvaluationInputs {
                config,
                source_config_sha256,
                behavior_clone: &args.behavior_clone,
                behavior_clone_metadata_sha256: metadata_sha256,
                behavior_clone_model_sha256: metadata.model_sha256,
                seeds: args.seeds,
                output: &args.output,
            },
            burn::backend::wgpu::WgpuDevice::default(),
        );
        if require_damage && !active {
            std::process::exit(2);
        }
    }

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    {
        let active = run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
            EvaluationInputs {
                config,
                source_config_sha256,
                behavior_clone: &args.behavior_clone,
                behavior_clone_metadata_sha256: metadata_sha256,
                behavior_clone_model_sha256: metadata.model_sha256,
                seeds: args.seeds,
                output: &args.output,
            },
            Default::default(),
        );
        if require_damage && !active {
            std::process::exit(2);
        }
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("contact-evaluation requires the wgpu or ndarray feature");
}
