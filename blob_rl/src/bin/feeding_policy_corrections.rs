#![recursion_limit = "512"]
//! Collect exact-input teacher labels on frozen-policy feeding trajectories.

use std::fs;
use std::path::{Path, PathBuf};

use blob_rl::action::{decompose_policy_action, PolicyActionFamily};
use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, expert_routing_phase, verify_behavior_clone_artifact,
    BehaviorCloningArtifact, ExpertRoutingStrategy, SupervisionPhase,
};
use blob_rl::config::{FeedingCurriculumStage, OpponentProfile, TrainingConfig};
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::demonstration::{
    generate_policy_correction_demonstrations, publish_demonstrations, DemonstrationCollection,
    PolicyCorrectionLabelMode, PolicyCorrectionOptions, POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS,
};
use blob_rl::feeding_layout_evaluation::FeedingQualificationLayout;
use blob_rl::model::PolicyValueNetConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
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
enum CorrectionLabels {
    FullTeacher,
    AdjacentConsumeControl,
    AdjacentConsumeTreatment,
    MoveTargetControl,
    MoveTargetTreatment,
}

impl From<CorrectionLabels> for PolicyCorrectionLabelMode {
    fn from(value: CorrectionLabels) -> Self {
        match value {
            CorrectionLabels::FullTeacher => Self::FullTeacher,
            CorrectionLabels::AdjacentConsumeControl => Self::AdjacentConsumeControl,
            CorrectionLabels::AdjacentConsumeTreatment => Self::AdjacentConsumeTreatment,
            CorrectionLabels::MoveTargetControl => Self::MoveTargetControl,
            CorrectionLabels::MoveTargetTreatment => Self::MoveTargetTreatment,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "feeding-policy-corrections",
    about = "Collect DAgger labels and exact-input agreement on frozen-policy feeding states"
)]
struct Args {
    /// Exact training TOML supplying rules, model shape, and feeding horizon.
    #[arg(long)]
    config: PathBuf,

    /// Immutable behavior clone whose greedy trajectory supplies states.
    #[arg(long)]
    behavior_clone: PathBuf,

    /// Feeding prerequisite to induce and label.
    #[arg(long, value_enum)]
    stage: FeedingStage,

    /// Optional founder geometry override for cross-layout correction data.
    #[arg(long, value_enum)]
    layout: Option<FeedingQualificationLayout>,

    /// Full DAgger labels or a matched, single-component control/treatment.
    #[arg(long, value_enum, default_value_t = CorrectionLabels::FullTeacher)]
    labels: CorrectionLabels,

    /// Disjoint collection seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// Maximum complete-prefix decisions retained across the seed suite.
    #[arg(long)]
    max_samples: usize,

    /// Minimum exact discrete-choice agreement for the diagnostic verdict.
    #[arg(long, default_value_t = 0.8)]
    minimum_policy_agreement_rate: f64,

    /// New immutable correction dataset directory.
    #[arg(long)]
    output: PathBuf,

    /// Exit with status 2 after publishing a valid failing dataset.
    #[arg(long)]
    require_pass: bool,
}

struct Inputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    seeds: Vec<u64>,
    max_samples: usize,
    minimum_policy_agreement_rate: f64,
    label_mode: PolicyCorrectionLabelMode,
    output: &'a Path,
}

fn run<B: Backend>(inputs: Inputs<'_>, device: B::Device) -> bool
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
    let (manifest, payload) = generate_policy_correction_demonstrations(
        &inputs.config,
        inputs.source_config_sha256,
        &PolicyCorrectionOptions {
            teacher: MaintainedMindProfile::CollisionAwareForager,
            seeds: inputs.seeds,
            max_samples: inputs.max_samples,
            behavior_clone_metadata_sha256: inputs.behavior_clone_metadata_sha256,
            behavior_clone_model_sha256: inputs.behavior_clone_model_sha256,
            minimum_policy_agreement_rate: inputs.minimum_policy_agreement_rate,
            label_mode: inputs.label_mode,
        },
        &model,
        &device,
    )
    .unwrap_or_else(|error| panic!("failed to collect feeding corrections: {error}"));
    publish_demonstrations(inputs.output, &manifest, &payload)
        .unwrap_or_else(|error| panic!("failed to publish feeding corrections: {error}"));
    let DemonstrationCollection::GreedyPolicyCorrection {
        policy_disagreement_samples,
        exact_policy_agreement_samples,
        policy_agreement_ppm,
        agreement_passed,
        policy_action_family_samples,
        action_family_confusion,
        action_kind_disagreement_samples,
        policy_kind_advantage_micrologit_sum,
        policy_kind_advantage_micrologit_max,
        policy_kind_advantage_within_threshold_samples,
        action_family_policy_kind_advantage_micrologit_sums,
        action_family_kind_disagreement_samples,
        label_mode,
        eligible_correction_samples,
        active_correction_samples,
        ..
    } = &manifest.collection
    else {
        unreachable!("feeding correction generator emitted another collection kind")
    };
    println!(
        "Published feeding corrections {} to {}: {}/{} exact choices ({:.1}%), {} disagreements",
        if *agreement_passed { "PASS" } else { "FAIL" },
        inputs.output.display(),
        exact_policy_agreement_samples,
        manifest.samples,
        f64::from(*policy_agreement_ppm) / 10_000.0,
        policy_disagreement_samples,
    );
    println!(
        "  labels {label_mode:?}: {eligible_correction_samples} eligible, {active_correction_samples} active"
    );
    for teacher in PolicyActionFamily::ALL {
        let row_start = teacher.index() * PolicyActionFamily::COUNT;
        for policy in PolicyActionFamily::ALL {
            let count = action_family_confusion[row_start + policy.index()];
            let kind_errors = action_family_kind_disagreement_samples[row_start + policy.index()];
            if kind_errors > 0 {
                let average_advantage = action_family_policy_kind_advantage_micrologit_sums
                    [row_start + policy.index()] as f64
                    / kind_errors as f64
                    / 1_000_000.0;
                println!(
                    "  teacher {:>11} -> policy {:>11}: {} choices, {} kind errors (mean policy advantage {:.4} logits)",
                    teacher.name(),
                    policy.name(),
                    count,
                    kind_errors,
                    average_advantage,
                );
            }
        }
    }
    let mean_advantage = if *action_kind_disagreement_samples == 0 {
        0.0
    } else {
        *policy_kind_advantage_micrologit_sum as f64
            / *action_kind_disagreement_samples as f64
            / 1_000_000.0
    };
    println!(
        "  kind-margin: {} errors, mean {:.4}, max {:.4} logits",
        action_kind_disagreement_samples,
        mean_advantage,
        f64::from(*policy_kind_advantage_micrologit_max) / 1_000_000.0,
    );
    for ((threshold, count), label) in POLICY_KIND_MARGIN_THRESHOLDS_MICROLOGITS
        .iter()
        .zip(policy_kind_advantage_within_threshold_samples)
        .zip(["0.01", "0.05", "0.10", "0.25", "0.50", "1.00"])
    {
        println!("    within {label} logits: {count}/{action_kind_disagreement_samples} (threshold {threshold} micrologits)");
    }
    println!("  policy family marginals: {policy_action_family_samples:?}");
    let mut kind_mismatches = 0usize;
    let mut target_mismatches = 0usize;
    let mut effort_mismatches = 0usize;
    let mut target_and_effort_mismatches = 0usize;
    let mut target_confusion = vec![0usize; blob_rl::action::NUM_POLICY_TARGETS.pow(2)];
    let mut teacher_target_samples = vec![0usize; blob_rl::action::NUM_POLICY_TARGETS];
    let mut policy_target_samples = vec![0usize; blob_rl::action::NUM_POLICY_TARGETS];
    let kinds = blob_rl::action::NUM_POLICY_ACTION_KINDS;
    let mut phase_kind_confusion = vec![0usize; SupervisionPhase::ALL.len() * kinds * kinds];
    let mut phase_kind_margin_sums = vec![0u64; SupervisionPhase::ALL.len() * kinds * kinds];
    for sample in &payload.samples {
        let Some(policy_action) = sample.correction_policy_action else {
            continue;
        };
        let teacher = decompose_policy_action(usize::from(
            sample
                .correction_teacher_action
                .expect("validated correction sample carries its teacher action"),
        ))
        .expect("validated teacher action is in the policy catalog");
        let policy = decompose_policy_action(usize::from(policy_action))
            .expect("validated correction action is in the policy catalog");
        if blob_rl::action::PolicyActionKind::from_index(teacher.kind)
            .is_some_and(|kind| kind.uses_target())
        {
            teacher_target_samples[teacher.target] += 1;
        }
        if blob_rl::action::PolicyActionKind::from_index(policy.kind)
            .is_some_and(|kind| kind.uses_target())
        {
            policy_target_samples[policy.target] += 1;
        }
        if teacher.kind != policy.kind {
            kind_mismatches += 1;
            let phase = expert_routing_phase(
                sample,
                ExpertRoutingStrategy::ForagingInteractionExploration,
            );
            let index = (phase.index() * kinds + teacher.kind) * kinds + policy.kind;
            phase_kind_confusion[index] += 1;
            phase_kind_margin_sums[index] += u64::from(
                sample
                    .correction_policy_kind_advantage_micrologits
                    .expect("validated correction kind error carries a margin"),
            );
            continue;
        }
        match (
            teacher.target != policy.target,
            teacher.effort != policy.effort,
        ) {
            (true, true) => {
                target_and_effort_mismatches += 1;
                target_confusion
                    [teacher.target * blob_rl::action::NUM_POLICY_TARGETS + policy.target] += 1;
            }
            (true, false) => {
                target_mismatches += 1;
                target_confusion
                    [teacher.target * blob_rl::action::NUM_POLICY_TARGETS + policy.target] += 1;
            }
            (false, true) => effort_mismatches += 1,
            (false, false) => {}
        }
    }
    println!(
        "  physical disagreement: {kind_mismatches} kind, {target_mismatches} target-only, \
         {effort_mismatches} effort-only, {target_and_effort_mismatches} target+effort"
    );
    for phase in SupervisionPhase::ALL {
        for teacher in 0..kinds {
            for policy in 0..kinds {
                let index = (phase.index() * kinds + teacher) * kinds + policy;
                let count = phase_kind_confusion[index];
                if count > 0 {
                    println!(
                        "    {:>11} teacher kind {teacher} -> policy kind {policy}: {count} errors (mean advantage {:.4} logits)",
                        phase.name(),
                        phase_kind_margin_sums[index] as f64 / count as f64 / 1_000_000.0,
                    );
                }
            }
        }
    }
    println!("  teacher target marginals: {teacher_target_samples:?}");
    println!("  policy target marginals:  {policy_target_samples:?}");
    for teacher in 0..blob_rl::action::NUM_POLICY_TARGETS {
        for policy in 0..blob_rl::action::NUM_POLICY_TARGETS {
            let count = target_confusion[teacher * blob_rl::action::NUM_POLICY_TARGETS + policy];
            if count > 0 {
                println!("    target {teacher} -> {policy}: {count} same-kind disagreements");
            }
        }
    }
    *agreement_passed
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
        panic!("feeding policy corrections require an enabled feeding curriculum");
    }
    let stage: FeedingCurriculumStage = args.stage.into();
    config.env = config
        .feeding_curriculum
        .environment_for_stage(&config.env, stage);
    if let Some(layout) = args.layout {
        config.env.starting_cell_layout = layout.starting_layout();
    }
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
        .unwrap_or_else(|error| panic!("invalid effective feeding-correction config: {error}"));

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
        seeds: args.seeds,
        max_samples: args.max_samples,
        minimum_policy_agreement_rate: args.minimum_policy_agreement_rate,
        label_mode: args.labels.into(),
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
    compile_error!("feeding-policy-corrections requires the wgpu or ndarray feature");

    if require_pass && !passed {
        std::process::exit(2);
    }
}
