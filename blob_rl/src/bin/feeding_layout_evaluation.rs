#![recursion_limit = "512"]
//! Evaluate one immutable learned policy across distinct founder geometries.

use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, BehaviorCloningArtifact};
use blob_rl::config::TrainingConfig;
use blob_rl::feeding_curriculum::evaluate_feeding_promotion;
use blob_rl::feeding_layout_evaluation::{
    load_feeding_layout_evaluation, publish_feeding_layout_evaluation,
    verify_feeding_layout_evaluation_request, FeedingLayoutEvaluationArtifact,
    FeedingLayoutPolicyIdentity, FeedingLayoutTrial, FeedingQualificationLayout,
};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::prelude::*;
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "feeding-layout-evaluation",
    about = "Evaluate a behavior clone across distinct 256-cell founder geometries"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    behavior_clone: PathBuf,

    #[arg(long, value_delimiter = ',', required = true)]
    layouts: Vec<FeedingQualificationLayout>,

    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    require_pass: bool,

    /// Reuse an existing artifact only after exact identity validation.
    #[arg(long)]
    resume: bool,
}

struct EvaluationInputs<'a> {
    config: TrainingConfig,
    source_config_sha256: String,
    behavior_clone: &'a Path,
    behavior_clone_metadata_sha256: String,
    behavior_clone_model_sha256: String,
    layouts: Vec<FeedingQualificationLayout>,
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

    let mut trials = Vec::with_capacity(inputs.layouts.len() * inputs.seeds.len());
    for layout in &inputs.layouts {
        let mut effective = inputs.config.clone();
        effective.env.starting_cell_layout = layout.starting_layout();
        effective
            .validate()
            .unwrap_or_else(|error| panic!("invalid {layout:?} evaluation config: {error}"));
        for seed in &inputs.seeds {
            let report = evaluate_feeding_promotion(
                &model,
                &effective.env,
                &effective.reward,
                &effective.feeding_curriculum,
                &[*seed],
                &device,
            );
            println!(
                "  {layout:?} seed {seed}: {} (on-food {:.1}%, adjacent {:.1}%)",
                if report.passed { "PASS" } else { "FAIL" },
                report.stages[0].survival_rate * 100.0,
                report.stages[1].survival_rate * 100.0,
            );
            trials.push(FeedingLayoutTrial {
                layout: *layout,
                seed: *seed,
                report,
            });
        }
    }
    let artifact = FeedingLayoutEvaluationArtifact::new(
        inputs.source_config_sha256,
        FeedingLayoutPolicyIdentity::BehaviorClone {
            metadata_sha256: inputs.behavior_clone_metadata_sha256,
            model_sha256: inputs.behavior_clone_model_sha256,
        },
        inputs.config,
        inputs.layouts,
        inputs.seeds,
        trials,
    )
    .unwrap_or_else(|error| panic!("invalid feeding-layout evaluation: {error}"));
    publish_feeding_layout_evaluation(inputs.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish feeding-layout evaluation: {error}"));
    println!(
        "Published feeding-layout evaluation {} to {} (artifact {})",
        if artifact.passed { "PASS" } else { "FAIL" },
        inputs.output.display(),
        artifact.artifact_hash,
    );
    artifact.passed
}

fn main() {
    let args = Args::parse();
    let config_bytes = std::fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid feeding-layout config: {error}"));
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &std::fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));
    let source_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    let require_pass = args.require_pass;
    let policy = FeedingLayoutPolicyIdentity::BehaviorClone {
        metadata_sha256: metadata_sha256.clone(),
        model_sha256: metadata.model_sha256.clone(),
    };

    if args.output.exists() {
        if !args.resume {
            panic!(
                "feeding-layout output {} exists; pass --resume to verify and reuse it",
                args.output.display()
            );
        }
        let artifact = load_feeding_layout_evaluation(&args.output)
            .unwrap_or_else(|error| panic!("invalid existing feeding-layout artifact: {error}"));
        verify_feeding_layout_evaluation_request(
            &artifact,
            &source_config_sha256,
            &policy,
            &config,
            &args.layouts,
            &args.seeds,
        )
        .unwrap_or_else(|error| panic!("feeding-layout resume rejected: {error}"));
        println!(
            "Reused verified {}x{} feeding-layout evaluation {} at {}",
            artifact.layouts.len(),
            artifact.seeds.len(),
            if artifact.passed { "PASS" } else { "FAIL" },
            args.output.display(),
        );
        if require_pass && !artifact.passed {
            std::process::exit(2);
        }
        return;
    }

    #[cfg(feature = "wgpu")]
    let passed = run::<burn::backend::Autodiff<burn::backend::Wgpu>>(
        EvaluationInputs {
            config,
            source_config_sha256,
            behavior_clone: &args.behavior_clone,
            behavior_clone_metadata_sha256: metadata_sha256,
            behavior_clone_model_sha256: metadata.model_sha256,
            layouts: args.layouts,
            seeds: args.seeds,
            output: &args.output,
        },
        burn::backend::wgpu::WgpuDevice::default(),
    );

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let passed = run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
        EvaluationInputs {
            config,
            source_config_sha256,
            behavior_clone: &args.behavior_clone,
            behavior_clone_metadata_sha256: metadata_sha256,
            behavior_clone_model_sha256: metadata.model_sha256,
            layouts: args.layouts,
            seeds: args.seeds,
            output: &args.output,
        },
        Default::default(),
    );

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("feeding-layout-evaluation requires the wgpu or ndarray feature");

    if require_pass && !passed {
        std::process::exit(2);
    }
}
