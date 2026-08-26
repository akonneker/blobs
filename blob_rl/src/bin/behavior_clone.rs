#![recursion_limit = "512"]
//! Supervised warm-start training from immutable demonstration datasets.

use std::path::PathBuf;

use blob_rl::behavior_cloning::{
    behavior_clone, behavior_clone_artifact_sha256, load_dataset_directories,
    publish_behavior_clone, BehaviorCloningConfig, DatasetSamplingStrategy,
};
use blob_rl::config::TrainingConfig;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Sampling {
    Balanced,
    Proportional,
}

impl From<Sampling> for DatasetSamplingStrategy {
    fn from(value: Sampling) -> Self {
        match value {
            Sampling::Balanced => Self::Balanced,
            Sampling::Proportional => Self::Proportional,
        }
    }
}

fn format_optional_metric(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "disabled".into())
}

#[derive(Debug, Parser)]
#[command(
    name = "blob_behavior_clone",
    about = "Pretrain the RL policy from hash-verified maintained-Mind demonstrations"
)]
struct Args {
    /// Training TOML supplying the neural architecture.
    #[arg(long)]
    config: PathBuf,

    /// Immutable demonstration directories. Repeat for a field/population curriculum.
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,

    /// New immutable pretrained-model directory.
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    #[arg(long, default_value_t = 256)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Fraction of source seeds held out whole for validation.
    #[arg(long, default_value_t = 0.1)]
    validation_fraction: f64,

    /// How each dataset contributes training samples to an epoch.
    #[arg(long, value_enum, default_value_t = Sampling::Balanced)]
    dataset_sampling: Sampling,

    /// Maximum consecutive decisions per cell in one recurrent graph.
    #[arg(long, default_value_t = 16)]
    recurrent_unroll_steps: usize,

    /// Exclude abstract labels whose full action/memory decision does not round-trip.
    #[arg(long)]
    exact_round_trip_only: bool,
}

fn main() {
    let args = Args::parse();
    let config_path = args
        .config
        .to_str()
        .unwrap_or_else(|| panic!("config path is not UTF-8: {}", args.config.display()));
    let training = TrainingConfig::from_file(config_path)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", args.config.display()));
    let datasets = load_dataset_directories(&args.dataset)
        .unwrap_or_else(|error| panic!("failed to load demonstrations: {error}"));
    let cloning = BehaviorCloningConfig {
        seed: args.seed,
        epochs: args.epochs,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        validation_fraction: args.validation_fraction,
        dataset_sampling: args.dataset_sampling.into(),
        recurrent_unroll_steps: args.recurrent_unroll_steps,
        exact_round_trip_only: args.exact_round_trip_only,
    };

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        type Backend = Autodiff<Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let (model, metrics) =
            behavior_clone::<Backend>(&datasets, &training.model, &cloning, device)
                .unwrap_or_else(|error| panic!("behavior cloning failed: {error}"));
        publish_behavior_clone(
            &args.output,
            &model,
            &training.model,
            &cloning,
            &datasets,
            &metrics,
        )
        .unwrap_or_else(|error| panic!("failed to publish behavior clone: {error}"));
        let artifact_sha256 = behavior_clone_artifact_sha256(&args.output)
            .expect("published behavior-cloning metadata should be hashable");
        println!(
            "Published {} training and {} validation samples to {}: train loss {:.4} -> {:.4}, validation loss {} -> {}; artifact SHA-256 {}",
            metrics.training_samples,
            metrics.validation_samples,
            args.output.display(),
            metrics.initial_training_loss,
            metrics.final_training_loss,
            format_optional_metric(metrics.initial_validation_loss),
            format_optional_metric(metrics.final_validation_loss),
            artifact_sha256,
        );
    }

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    {
        use burn::backend::{Autodiff, NdArray};
        type Backend = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let (model, metrics) =
            behavior_clone::<Backend>(&datasets, &training.model, &cloning, device)
                .unwrap_or_else(|error| panic!("behavior cloning failed: {error}"));
        publish_behavior_clone(
            &args.output,
            &model,
            &training.model,
            &cloning,
            &datasets,
            &metrics,
        )
        .unwrap_or_else(|error| panic!("failed to publish behavior clone: {error}"));
        let artifact_sha256 = behavior_clone_artifact_sha256(&args.output)
            .expect("published behavior-cloning metadata should be hashable");
        println!(
            "Published {} training and {} validation samples to {}: train loss {:.4} -> {:.4}, validation loss {} -> {}; artifact SHA-256 {}",
            metrics.training_samples,
            metrics.validation_samples,
            args.output.display(),
            metrics.initial_training_loss,
            metrics.final_training_loss,
            format_optional_metric(metrics.initial_validation_loss),
            format_optional_metric(metrics.final_validation_loss),
            artifact_sha256,
        );
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("behavior-clone requires the wgpu or ndarray feature");
}
