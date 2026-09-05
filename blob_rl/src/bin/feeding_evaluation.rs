#![recursion_limit = "512"]
//! Evaluate an immutable behavior-cloned policy on the feeding prerequisites.

use std::path::{Path, PathBuf};

use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, verify_behavior_clone_artifact, BehaviorCloningArtifact,
};
use blob_rl::config::TrainingConfig;
use blob_rl::feeding_curriculum::evaluate_feeding_promotion_with_progress;
use blob_rl::feeding_evaluation_artifact::{
    load_feeding_evaluation, publish_feeding_evaluation, verify_feeding_evaluation_request,
    FeedingEvaluationArtifact,
};
use blob_rl::model::PolicyValueNetConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use clap::Parser;
use sha2::{Digest, Sha256};
use std::io::Write;

#[derive(Debug, Parser)]
#[command(
    name = "feeding-evaluation",
    about = "Publish hash-bound feeding-competency evidence for a behavior clone"
)]
struct Args {
    /// Exact training TOML supplying the scenario, rules, gate, and model shape.
    #[arg(long)]
    config: PathBuf,

    /// Immutable behavior-cloning artifact directory.
    #[arg(long)]
    behavior_clone: PathBuf,

    /// Held-out episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// New immutable evaluation JSON file.
    #[arg(long)]
    output: PathBuf,

    /// Exit unsuccessfully after publishing a valid failing report.
    #[arg(long)]
    require_pass: bool,

    /// Reuse an existing output only after validating its exact config, model,
    /// and ordered seed suite. Invalid or unrelated outputs fail closed.
    #[arg(long)]
    resume: bool,
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
    let EvaluationInputs {
        config,
        source_config_sha256,
        behavior_clone,
        behavior_clone_metadata_sha256,
        behavior_clone_model_sha256,
        seeds,
        output,
    } = inputs;
    let model_path = verify_behavior_clone_artifact(
        behavior_clone,
        &behavior_clone_metadata_sha256,
        &config.model,
    )
    .unwrap_or_else(|error| panic!("invalid behavior clone: {error}"));
    let model = PolicyValueNetConfig {
        hidden1: config.model.hidden1,
        hidden2: config.model.hidden2,
        recurrent_size: config.model.recurrent_size,
    }
    .init::<B>(&device)
    .load_file(model_path, &CompactRecorder::new(), &device)
    .unwrap_or_else(|error| panic!("failed to load behavior-cloned model: {error}"));
    let report = evaluate_feeding_promotion_with_progress(
        &model,
        &config.env,
        &config.reward,
        &config.feeding_curriculum,
        &seeds,
        &device,
        |progress| {
            eprintln!(
                "  progress {} {}/{} seed {}: {} (moves {}, consumes {}, energy {}, survivors {}, safety_abort {})",
                progress.stage,
                progress.completed_in_stage,
                progress.total_in_stage,
                progress.seed,
                if progress.succeeded { "success" } else { "failure" },
                progress.movement_successes,
                progress.consume_successes,
                progress.consumed_energy,
                progress.surviving_cells,
                progress.safety_abort,
            );
            let _ = std::io::stderr().flush();
        },
    );
    let artifact = FeedingEvaluationArtifact::new(
        source_config_sha256,
        behavior_clone_metadata_sha256,
        behavior_clone_model_sha256,
        config,
        report,
    )
    .unwrap_or_else(|error| panic!("invalid feeding evaluation: {error}"));
    publish_feeding_evaluation(output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish feeding evaluation: {error}"));
    println!(
        "Published feeding evaluation {} to {} (artifact {}, {}):",
        if artifact.report.passed {
            "PASS"
        } else {
            "FAIL"
        },
        output.display(),
        artifact.artifact_hash,
        artifact.behavior_clone_model_sha256,
    );
    for stage in &artifact.report.stages {
        println!(
            "  {}: success {:.1}%, survival {:.1}%, intake {:.3}/initial cell (moves {}, consumes {}, safety aborts {})",
            stage.stage,
            stage.episode_success_rate * 100.0,
            stage.survival_rate * 100.0,
            stage.consumed_energy_per_initial_cell,
            stage.movement_successes,
            stage.consume_successes,
            stage.safety_aborts,
        );
    }
    artifact.report.passed
}

fn main() {
    let args = Args::parse();
    if args.seeds.is_empty() {
        panic!("feeding evaluation requires at least one seed");
    }
    let config_bytes = std::fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid feeding-evaluation config: {error}"));
    let source_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    let metadata_path = args.behavior_clone.join("behavior-cloning.json");
    let metadata: BehaviorCloningArtifact = serde_json::from_slice(
        &std::fs::read(&metadata_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", metadata_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", metadata_path.display()));
    let require_pass = args.require_pass;
    if args.output.exists() {
        if !args.resume {
            panic!(
                "feeding-evaluation output {} already exists; use --resume to validate and reuse it",
                args.output.display()
            );
        }
        let artifact = load_feeding_evaluation(&args.output)
            .unwrap_or_else(|error| panic!("invalid existing output: {error}"));
        verify_feeding_evaluation_request(
            &artifact,
            &source_config_sha256,
            &metadata_sha256,
            &metadata.model_sha256,
            &config,
            &args.seeds,
        )
        .unwrap_or_else(|error| panic!("existing output cannot be resumed: {error}"));
        println!(
            "Reused verified {}-seed feeding evaluation {} at {}",
            artifact.report.seeds.len(),
            if artifact.report.passed {
                "PASS"
            } else {
                "FAIL"
            },
            args.output.display(),
        );
        if require_pass && !artifact.report.passed {
            std::process::exit(2);
        }
        return;
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        type Backend = Autodiff<Wgpu>;
        let passed = run::<Backend>(
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
        if require_pass && !passed {
            std::process::exit(2);
        }
    }

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    {
        use burn::backend::{Autodiff, NdArray};
        type Backend = Autodiff<NdArray<f32>>;
        let passed = run::<Backend>(
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
        if require_pass && !passed {
            std::process::exit(2);
        }
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("feeding-evaluation requires the wgpu or ndarray feature");
}
