#![recursion_limit = "512"]
//! Evaluate counterfactual teacher recovery from frozen-policy feeding states.

use std::fs;
use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, BehaviorCloningArtifact};
use blob_rl::config::TrainingConfig;
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::feeding_recovery::{
    evaluate_feeding_recovery, publish_feeding_recovery, FeedingRecoveryArtifact,
    FeedingRecoveryOptions,
};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::prelude::*;
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "feeding-recovery",
    about = "Fork frozen-policy feeding states and test whether a legal teacher can recover"
)]
struct Args {
    /// Exact training TOML supplying rules, model shape, and feeding horizon.
    #[arg(long)]
    config: PathBuf,

    /// Immutable behavior-cloning artifact whose greedy trajectory is forked.
    #[arg(long)]
    behavior_clone: PathBuf,

    /// Observation-legal maintained Mind used for counterfactual continuation.
    #[arg(long, value_enum, default_value = "collision-aware-forager")]
    teacher: MaintainedMindProfile,

    /// Disjoint episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// Canonical simulation-time spacing after the first policy-induced fork.
    #[arg(long, default_value_t = 4096)]
    checkpoint_interval_quanta: u64,

    /// Maximum counterfactual forks retained per stage and seed.
    #[arg(long, default_value_t = 8)]
    max_checkpoints_per_seed: usize,

    /// Minimum forks required for every stage and seed.
    #[arg(long, default_value_t = 1)]
    min_checkpoints_per_seed: usize,

    /// Exclude frontiers with less canonical time than this before deadline.
    #[arg(long, default_value_t = 4096)]
    min_remaining_time_quanta: u64,

    /// Minimum fraction of checkpoint branches the teacher must recover.
    #[arg(long, default_value_t = 0.8)]
    min_recovery_rate: f64,

    /// New immutable recovery JSON.
    #[arg(long)]
    output: PathBuf,

    /// Exit with status 2 after publishing valid evidence that misses its gate.
    #[arg(long)]
    require_pass: bool,
}

struct Inputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    options: FeedingRecoveryOptions,
    output: &'a Path,
}

fn run<B: Backend>(inputs: Inputs<'_>, device: B::Device) -> bool
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
    let report = evaluate_feeding_recovery(&model, &inputs.config, &inputs.options, &device)
        .unwrap_or_else(|error| panic!("feeding recovery failed: {error}"));
    let artifact = FeedingRecoveryArtifact::new(
        inputs.source_config_sha256,
        inputs.behavior_clone_metadata_sha256,
        inputs.behavior_clone_model_sha256,
        inputs.config,
        inputs.options,
        report,
    )
    .unwrap_or_else(|error| panic!("invalid feeding recovery artifact: {error}"));
    publish_feeding_recovery(inputs.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish feeding recovery: {error}"));
    println!(
        "Published feeding recovery {} to {} (artifact {}, model {}):",
        if artifact.report.passed {
            "PASS"
        } else {
            "FAIL"
        },
        inputs.output.display(),
        artifact.artifact_hash,
        artifact.behavior_clone_model_sha256,
    );
    for stage in &artifact.report.stages {
        println!(
            "  {}: teacher recovered {}/{} ({:.1}%); policy solved {}/{} and survived {}/{}",
            stage.stage,
            stage.recoverable_checkpoints,
            stage.checkpoints,
            stage.recovery_rate * 100.0,
            stage.policy_objective_successes,
            stage.episodes,
            stage.policy_surviving_episodes,
            stage.episodes,
        );
    }
    artifact.report.passed
}

fn main() {
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid feeding-recovery config: {error}"));
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));
    let inputs = Inputs {
        config,
        source_config_sha256: format!("{:x}", Sha256::digest(&config_bytes)),
        behavior_clone: &args.behavior_clone,
        behavior_clone_metadata_sha256: metadata_sha256,
        behavior_clone_model_sha256: metadata.model_sha256,
        options: FeedingRecoveryOptions {
            teacher: args.teacher,
            seeds: args.seeds,
            checkpoint_interval_quanta: args.checkpoint_interval_quanta,
            max_checkpoints_per_seed: args.max_checkpoints_per_seed,
            min_checkpoints_per_seed: args.min_checkpoints_per_seed,
            min_remaining_time_quanta: args.min_remaining_time_quanta,
            min_recovery_rate: args.min_recovery_rate,
        },
        output: &args.output,
    };
    let require_pass = args.require_pass;

    #[cfg(feature = "wgpu")]
    let passed = run::<burn::backend::Autodiff<burn::backend::Wgpu>>(
        inputs,
        burn::backend::wgpu::WgpuDevice::default(),
    );

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let passed =
        run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(inputs, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("feeding-recovery requires the wgpu or ndarray feature");

    if require_pass && !passed {
        std::process::exit(2);
    }
}
