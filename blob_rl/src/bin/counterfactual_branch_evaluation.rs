#![recursion_limit = "512"]
//! Branch legal Guard/Move interventions from exact frozen-policy frontiers.

use std::fs;
use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, BehaviorCloningArtifact};
use blob_rl::config::{FeedingCurriculumStage, OpponentProfile, TrainingConfig};
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::counterfactual_branch::{
    evaluate_counterfactual_branches, publish_counterfactual_branch_artifact, BranchCandidateScope,
    BranchContinuation, CounterfactualBranchArtifact, CounterfactualBranchOptions,
};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::prelude::*;
use clap::{Parser, ValueEnum};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FeedingStage {
    OnFood,
    AdjacentFood,
}

impl From<FeedingStage> for FeedingCurriculumStage {
    fn from(value: FeedingStage) -> Self {
        match value {
            FeedingStage::OnFood => Self::OnFood,
            FeedingStage::AdjacentFood => Self::AdjacentFood,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ContinuationArg {
    FrozenPolicy,
    Teacher,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CandidateScopeArg {
    AllLegal,
    TeacherPolicy,
    Explicit,
}

impl From<CandidateScopeArg> for BranchCandidateScope {
    fn from(value: CandidateScopeArg) -> Self {
        match value {
            CandidateScopeArg::AllLegal => Self::AllLegal,
            CandidateScopeArg::TeacherPolicy => Self::TeacherPolicy,
            CandidateScopeArg::Explicit => Self::Explicit,
        }
    }
}

impl From<ContinuationArg> for BranchContinuation {
    fn from(value: ContinuationArg) -> Self {
        match value {
            ContinuationArg::FrozenPolicy => Self::FrozenPolicy,
            ContinuationArg::Teacher => Self::Teacher,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "counterfactual-branch-evaluation",
    about = "Evaluate paired Guard/Move interventions on exact frozen-policy feeding states"
)]
struct Args {
    /// Training TOML supplying the model shape, physics, and feeding horizon.
    #[arg(long)]
    config: PathBuf,

    /// Immutable behavior clone whose greedy trajectory supplies source states.
    #[arg(long)]
    behavior_clone: PathBuf,

    /// Feeding prerequisite used to construct the effective environment.
    #[arg(long, value_enum)]
    stage: FeedingStage,

    /// Source trajectory seeds. These become evaluation data and must remain
    /// disjoint from later training and final qualification seeds.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// Maximum selected Guard-to-Move states across all seeds.
    #[arg(long, default_value_t = 16)]
    max_states: usize,

    /// Maximum selected states contributed by one trajectory seed.
    #[arg(long, default_value_t = 4)]
    max_states_per_seed: usize,

    /// Maximum selected cells from one simultaneous decision frontier.
    #[arg(long, default_value_t = 1)]
    max_states_per_frontier: usize,

    /// Maximum source decision frontiers searched per seed.
    #[arg(long, default_value_t = 512)]
    max_source_frontiers_per_seed: u64,

    /// Elapsed simulation-clock horizons after each intervention.
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384")]
    horizon_quanta: Vec<u64>,

    /// Controllers used after the first counterfactual action.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "frozen-policy,teacher"
    )]
    continuations: Vec<ContinuationArg>,

    /// Candidate action surface. Use all-legal for short discovery probes and
    /// teacher-policy for deeper confirmation of the observed disagreement.
    #[arg(long, value_enum, default_value = "all-legal")]
    candidate_scope: CandidateScopeArg,

    /// Physical action ids used by explicit candidate scope. Teacher and
    /// policy choices are included automatically as audit baselines.
    #[arg(long, value_delimiter = ',')]
    candidate_actions: Vec<usize>,

    /// Include Guard-to-Move disagreements outside the anonymous exploration
    /// routing context.
    #[arg(long)]
    all_contexts: bool,

    /// Host-only protection against a branch that stops advancing useful world
    /// time. This is not a scientific match deadline.
    #[arg(long, default_value_t = 4096)]
    max_frontiers_per_branch: u64,

    /// New immutable JSON artifact path.
    #[arg(long)]
    output: PathBuf,
}

struct Inputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    options: CounterfactualBranchOptions,
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
    let report = evaluate_counterfactual_branches(&model, &inputs.config, &inputs.options, &device)
        .unwrap_or_else(|error| panic!("counterfactual evaluation failed: {error}"));
    let artifact = CounterfactualBranchArtifact::new(
        inputs.source_config_sha256,
        inputs.behavior_clone_metadata_sha256,
        inputs.behavior_clone_model_sha256,
        inputs.config,
        inputs.options,
        report,
    )
    .unwrap_or_else(|error| panic!("failed to construct counterfactual artifact: {error}"));
    publish_counterfactual_branch_artifact(inputs.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish counterfactual artifact: {error}"));

    println!(
        "Published {} selected states, {} candidate actions, and {} exact branches to {}",
        artifact.report.selected_states,
        artifact.report.candidate_actions,
        artifact.report.evaluated_branches,
        inputs.output.display(),
    );
    let covered_seeds = artifact
        .report
        .seed_selection
        .iter()
        .filter(|seed| seed.selected_states > 0)
        .count();
    println!(
        "  source-seed coverage: {covered_seeds}/{}",
        artifact.report.seed_selection.len()
    );
    for summary in &artifact.report.pairwise_teacher_policy {
        println!(
            "  state {:?} +{}q: survival teacher/policy/tie {}/{}/{}, team energy {}/{}/{}",
            summary.continuation,
            summary.requested_elapsed_quanta,
            summary.teacher_target_survival_better,
            summary.policy_target_survival_better,
            summary.target_survival_ties,
            summary.teacher_team_stored_energy_better,
            summary.policy_team_stored_energy_better,
            summary.team_stored_energy_ties,
        );
    }
    for summary in &artifact.report.pairwise_seed_teacher_policy {
        println!(
            "  seed  {:?} +{}q: survival teacher/policy/tie {}/{}/{}, team energy {}/{}/{}",
            summary.continuation,
            summary.requested_elapsed_quanta,
            summary.teacher_target_survival_better,
            summary.policy_target_survival_better,
            summary.target_survival_ties,
            summary.teacher_team_stored_energy_better,
            summary.policy_team_stored_energy_better,
            summary.team_stored_energy_ties,
        );
    }
    println!("Artifact hash: {}", artifact.artifact_hash);
}

fn main() {
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let mut config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    if !config.feeding_curriculum.enabled {
        panic!("counterfactual feeding evaluation requires an enabled feeding curriculum");
    }
    config.env = config
        .feeding_curriculum
        .environment_for_stage(&config.env, args.stage.into());
    config.env.opponent = OpponentProfile::Wait;
    config.env.max_episode_len = config
        .feeding_curriculum
        .promotion
        .evaluation_max_episode_len;
    config.env.victory.sim_time_limit_quanta = config
        .feeding_curriculum
        .promotion
        .evaluation_sim_time_limit_quanta;
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid effective counterfactual config: {error}"));

    let behavior_clone_metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));
    let options = CounterfactualBranchOptions {
        teacher: MaintainedMindProfile::CollisionAwareForager,
        seeds: args.seeds,
        max_states: args.max_states,
        max_states_per_seed: args.max_states_per_seed,
        max_states_per_frontier: args.max_states_per_frontier,
        max_source_frontiers_per_seed: args.max_source_frontiers_per_seed,
        horizon_quanta: args.horizon_quanta,
        continuations: args.continuations.into_iter().map(Into::into).collect(),
        candidate_scope: args.candidate_scope.into(),
        explicit_candidate_actions: args.candidate_actions,
        exploration_only: !args.all_contexts,
        max_frontiers_per_branch: args.max_frontiers_per_branch,
    };
    options
        .validate()
        .unwrap_or_else(|error| panic!("invalid counterfactual options: {error}"));
    let inputs = Inputs {
        config,
        source_config_sha256: format!("{:x}", Sha256::digest(&config_bytes)),
        behavior_clone: &args.behavior_clone,
        behavior_clone_metadata_sha256,
        behavior_clone_model_sha256: metadata.model_sha256,
        options,
        output: &args.output,
    };

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Autodiff<burn::backend::Wgpu>>(
        inputs,
        burn::backend::wgpu::WgpuDevice::default(),
    );

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(inputs, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("counterfactual-branch-evaluation requires the wgpu or ndarray feature");
}
