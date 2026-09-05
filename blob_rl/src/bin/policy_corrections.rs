#![recursion_limit = "512"]
//! Collect deterministic teacher labels on greedy policy-induced trajectories.

use std::fs;
use std::path::PathBuf;

use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, verify_behavior_clone_artifact, BehaviorCloningArtifact,
};
use blob_rl::config::{FeedingCurriculumStage, OpponentProfile, TrainingConfig};
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::demonstration::{
    generate_policy_correction_demonstrations, publish_demonstrations, DemonstrationCollection,
    PolicyCorrectionLabelMode, PolicyCorrectionOptions,
};
use blob_rl::model::PolicyValueNetConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "policy-corrections",
    about = "Label deterministic greedy-policy contact trajectories with a maintained Mind"
)]
struct Args {
    /// Exact training TOML supplying rules and combat curriculum.
    #[arg(long)]
    config: PathBuf,

    /// Verified behavior clone whose induced states will be collected.
    #[arg(long)]
    behavior_clone: PathBuf,

    /// Maintained native Mind that labels every visited state.
    #[arg(long, value_enum)]
    teacher: MaintainedMindProfile,

    /// Unique episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// Maximum ready-cell decisions retained across complete trajectory prefixes.
    #[arg(long)]
    max_samples: usize,

    /// Initial energy for the paired one-on-one contact rollout.
    #[arg(long)]
    contact_energy: u32,

    /// Anonymous baseline inhabiting the opposing contact cell.
    #[arg(long, value_enum)]
    contact_opponent: OpponentProfile,

    /// New immutable correction dataset directory.
    #[arg(long)]
    output: PathBuf,

    /// Minimum exact teacher/policy agreement recorded in the dataset verdict.
    #[arg(long, default_value_t = 0.8)]
    minimum_policy_agreement_rate: f64,
}

struct Inputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a PathBuf,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    teacher: MaintainedMindProfile,
    seeds: Vec<u64>,
    max_samples: usize,
    minimum_policy_agreement_rate: f64,
    output: &'a PathBuf,
}

fn run<B: Backend>(inputs: Inputs<'_>, device: B::Device)
where
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
            teacher: inputs.teacher,
            seeds: inputs.seeds,
            max_samples: inputs.max_samples,
            behavior_clone_metadata_sha256: inputs.behavior_clone_metadata_sha256,
            behavior_clone_model_sha256: inputs.behavior_clone_model_sha256,
            minimum_policy_agreement_rate: inputs.minimum_policy_agreement_rate,
            label_mode: PolicyCorrectionLabelMode::FullTeacher,
        },
        &model,
        &device,
    )
    .unwrap_or_else(|error| panic!("failed to collect policy corrections: {error}"));
    publish_demonstrations(inputs.output, &manifest, &payload)
        .unwrap_or_else(|error| panic!("failed to publish policy corrections: {error}"));
    let DemonstrationCollection::GreedyPolicyCorrection {
        policy_disagreement_samples,
        teacher_attack_policy_non_attack_samples,
        projected_teacher_samples,
        exact_policy_agreement_samples,
        policy_agreement_ppm,
        agreement_passed,
        action_kind_disagreement_samples,
        policy_kind_advantage_micrologit_sum,
        policy_kind_advantage_micrologit_max,
        ..
    } = manifest.collection
    else {
        unreachable!("policy correction generator emitted a teacher-rollout identity")
    };
    println!(
        "Published {} correction samples to {} ({} exact agreements, {:.1}% agreement, gate {}; {} policy disagreements, {} missed teacher attacks, {} teacher attacks, {} projected labels)",
        manifest.samples,
        inputs.output.display(),
        exact_policy_agreement_samples,
        f64::from(policy_agreement_ppm) / 10_000.0,
        if agreement_passed { "PASS" } else { "FAIL" },
        policy_disagreement_samples,
        teacher_attack_policy_non_attack_samples,
        manifest.action_family_samples[blob_rl::action::PolicyActionFamily::Attack.index()],
        projected_teacher_samples,
    );
    let mean_advantage = if action_kind_disagreement_samples == 0 {
        0.0
    } else {
        policy_kind_advantage_micrologit_sum as f64
            / action_kind_disagreement_samples as f64
            / 1_000_000.0
    };
    println!(
        "  action-kind errors: {action_kind_disagreement_samples}, mean selected-kind advantage {mean_advantage:.4}, max {:.4} logits",
        f64::from(policy_kind_advantage_micrologit_max) / 1_000_000.0,
    );
}

fn main() {
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let mut config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    if !config.combat_curriculum.enabled
        || !matches!(
            args.contact_opponent,
            OpponentProfile::Aggressive | OpponentProfile::Defensive
        )
    {
        panic!(
            "policy corrections require an enabled combat curriculum and an aggressive or defensive contact opponent"
        );
    }
    let contact_start = config
        .combat_curriculum
        .on_food_sim_time_quanta_per_cycle
        .saturating_add(
            config
                .combat_curriculum
                .adjacent_food_sim_time_quanta_per_cycle,
        );
    config.env = config.rollout_environment(FeedingCurriculumStage::Contact, contact_start);
    config.env.initial_energy = args.contact_energy;
    config.env.opponent = args.contact_opponent;
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid effective correction config: {error}"));

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
        teacher: args.teacher,
        seeds: args.seeds,
        max_samples: args.max_samples,
        minimum_policy_agreement_rate: args.minimum_policy_agreement_rate,
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
    compile_error!("policy-corrections requires the wgpu or ndarray feature");
}
