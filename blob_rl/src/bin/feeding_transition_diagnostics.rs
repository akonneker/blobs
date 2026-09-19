#![recursion_limit = "512"]
//! Attribute adjacent-food retention failures along the move/consume sequence.

use std::fs;
use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, BehaviorCloningArtifact};
use blob_rl::config::TrainingConfig;
use blob_rl::feeding_layout_evaluation::FeedingQualificationLayout;
use blob_rl::feeding_transition_diagnostics::{
    evaluate_adjacent_transitions, load_feeding_transition, publish_feeding_transition,
    verify_feeding_transition_request, FeedingTransitionArtifact, PolicyRandomnessMode,
};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::prelude::*;
use clap::{Parser, ValueEnum};
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "feeding-transition-diagnostics",
    about = "Publish cell-level adjacent-food move/consume/death attribution"
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
    #[arg(long)]
    resume: bool,
    /// Override only the founder geometry while retaining the source config
    /// hash and every other evaluation setting.
    #[arg(long, value_enum)]
    layout: Option<FeedingQualificationLayout>,
    /// Random block projected into learned-policy input. Canonical remains
    /// deterministic for a fixed match seed while varying independently by cell.
    #[arg(long, value_enum, default_value = "canonical")]
    policy_randomness: PolicyRandomnessArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PolicyRandomnessArg {
    Zero,
    Canonical,
}

impl From<PolicyRandomnessArg> for PolicyRandomnessMode {
    fn from(value: PolicyRandomnessArg) -> Self {
        match value {
            PolicyRandomnessArg::Zero => Self::Zero,
            PolicyRandomnessArg::Canonical => Self::Canonical,
        }
    }
}

struct Inputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    seeds: Vec<u64>,
    policy_randomness: PolicyRandomnessMode,
    output: &'a Path,
}

fn run<B: Backend>(inputs: Inputs<'_>, device: B::Device)
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
    let report = evaluate_adjacent_transitions(
        &model,
        &inputs.config,
        &inputs.seeds,
        inputs.policy_randomness,
        &device,
        |episode| {
            eprintln!(
                "  seed {}: survivors {}/{}, reached {}, consumed {}, left {}, deaths pre-reach/arrival/post-consume {}/{}/{}",
                episode.seed,
                episode.counts.terminal_cells,
                episode.counts.initial_cells,
                episode.counts.cells_reaching_plant,
                episode.counts.cells_successfully_consuming,
                episode.counts.cells_leaving_plant,
                episode.counts.deaths_before_reaching_plant,
                episode.counts.deaths_after_reaching_before_consume,
                episode.counts.deaths_after_successful_consume,
            );
        },
    )
    .unwrap_or_else(|error| panic!("feeding transition diagnostic failed: {error}"));
    let artifact = FeedingTransitionArtifact::new(
        inputs.source_config_sha256,
        inputs.behavior_clone_metadata_sha256,
        inputs.behavior_clone_model_sha256,
        inputs.config,
        report,
    )
    .unwrap_or_else(|error| panic!("invalid feeding transition artifact: {error}"));
    publish_feeding_transition(inputs.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish transition diagnostic: {error}"));
    let totals = &artifact.report.totals;
    println!(
        "Published adjacent transition diagnostic {} ({}) to {}",
        artifact.artifact_hash,
        artifact.report.policy_randomness,
        inputs.output.display()
    );
    println!(
        "  survivors {}/{}, reached {}, consumed {}, left {}",
        totals.terminal_cells,
        totals.initial_cells,
        totals.cells_reaching_plant,
        totals.cells_successfully_consuming,
        totals.cells_leaving_plant,
    );
    println!(
        "  visible plant: {} correct moves, {} wrong-target moves, {} non-moves",
        totals.correct_move_to_plant_decisions,
        totals.wrong_target_move_decisions,
        totals.non_move_with_visible_plant_decisions,
    );
    println!(
        "  correct-move outcomes: {} success, {} frustrated, {} contested, {} interrupted, {} rejected, {} died pending, {} pending at terminal",
        totals.correct_move_success_outcomes,
        totals.correct_move_frustrated_outcomes,
        totals.correct_move_contested_outcomes,
        totals.correct_move_interrupted_outcomes,
        totals.correct_move_rejected_outcomes,
        totals.correct_move_unobserved_deaths,
        totals.correct_move_pending_at_terminal,
    );
    println!(
        "  on plant: {} consumes, {} moves away, {} other actions",
        totals.consume_on_plant_decisions,
        totals.move_off_plant_decisions,
        totals.other_on_plant_decisions,
    );
    println!(
        "  deaths: {} before reach, {} after reach before consume, {} after consume",
        totals.deaths_before_reaching_plant,
        totals.deaths_after_reaching_before_consume,
        totals.deaths_after_successful_consume,
    );
}

fn main() {
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let mut config = TrainingConfig::from_toml_str(
        std::str::from_utf8(&config_bytes)
            .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display())),
    )
    .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    if let Some(layout) = args.layout {
        config.env.starting_cell_layout = layout.starting_layout();
    }
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid diagnostic config: {error}"));
    let source_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));

    if args.output.exists() {
        if !args.resume {
            panic!(
                "{} exists; use --resume to verify and reuse it",
                args.output.display()
            );
        }
        let artifact = load_feeding_transition(&args.output)
            .unwrap_or_else(|error| panic!("invalid existing diagnostic: {error}"));
        verify_feeding_transition_request(
            &artifact,
            &source_config_sha256,
            &metadata_sha256,
            &metadata.model_sha256,
            &config,
            &args.seeds,
            args.policy_randomness.into(),
        )
        .unwrap_or_else(|error| panic!("existing diagnostic cannot be resumed: {error}"));
        println!(
            "Reused verified {}-seed adjacent transition diagnostic {}",
            artifact.report.seeds.len(),
            artifact.artifact_hash
        );
        return;
    }

    let inputs = Inputs {
        config,
        source_config_sha256,
        behavior_clone: &args.behavior_clone,
        behavior_clone_metadata_sha256: metadata_sha256,
        behavior_clone_model_sha256: metadata.model_sha256,
        seeds: args.seeds,
        policy_randomness: args.policy_randomness.into(),
        output: &args.output,
    };

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu>(inputs, burn::backend::wgpu::WgpuDevice::default());

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    run::<burn::backend::NdArray>(inputs, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("feeding-transition-diagnostics requires wgpu or ndarray");
}
