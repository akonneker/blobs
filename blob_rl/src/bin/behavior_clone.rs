#![recursion_limit = "512"]
//! Supervised warm-start training from immutable demonstration datasets.

use std::path::PathBuf;

use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, behavior_clone_from_model, load_dataset_directories,
    publish_behavior_clone, verify_behavior_clone_artifact, ActionBalancingStrategy,
    BehaviorCloningConfig, DatasetSamplingStrategy,
};
use blob_rl::config::TrainingConfig;
use blob_rl::model::PolicyValueNetConfig;
use burn::module::Module;
use burn::record::CompactRecorder;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Sampling {
    Balanced,
    Proportional,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ActionBalancing {
    None,
    Family,
    Label,
}

impl From<ActionBalancing> for ActionBalancingStrategy {
    fn from(value: ActionBalancing) -> Self {
        match value {
            ActionBalancing::None => Self::None,
            ActionBalancing::Family => Self::Family,
            ActionBalancing::Label => Self::Label,
        }
    }
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

    /// Verified behavior-cloning artifact used to initialize this stage.
    #[arg(long)]
    initial_behavior_clone: Option<PathBuf>,

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

    /// Explicit positive dataset sampling weights, one per --dataset in order.
    #[arg(long, value_delimiter = ',')]
    dataset_weight: Vec<f64>,

    /// Maximum consecutive decisions per cell in one recurrent graph.
    #[arg(long, default_value_t = 16)]
    recurrent_unroll_steps: usize,

    /// Exclude abstract labels whose full action/memory decision does not round-trip.
    #[arg(long)]
    exact_round_trip_only: bool,

    /// Reweight the supervised action loss to counter catalog imbalance.
    #[arg(long, value_enum, default_value_t = ActionBalancing::None)]
    action_balancing: ActionBalancing,

    /// Inverse-frequency weighting exponent in [0, 1].
    #[arg(long, default_value_t = 1.0)]
    action_balance_exponent: f64,

    /// Cap the largest represented action weight to this multiple of the smallest.
    #[arg(long)]
    action_balance_max_ratio: Option<f64>,
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
    let initial_artifact_sha256 = args.initial_behavior_clone.as_ref().map(|directory| {
        behavior_clone_artifact_sha256(directory).unwrap_or_else(|error| {
            panic!(
                "failed to hash initial behavior clone {}: {error}",
                directory.display()
            )
        })
    });
    let cloning = BehaviorCloningConfig {
        seed: args.seed,
        initial_artifact_sha256: initial_artifact_sha256.clone(),
        epochs: args.epochs,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        validation_fraction: args.validation_fraction,
        dataset_sampling: args.dataset_sampling.into(),
        dataset_sampling_weights: args.dataset_weight,
        recurrent_unroll_steps: args.recurrent_unroll_steps,
        exact_round_trip_only: args.exact_round_trip_only,
        action_balancing: args.action_balancing.into(),
        action_balance_exponent: args.action_balance_exponent,
        action_balance_max_ratio: args.action_balance_max_ratio,
    };

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, Wgpu};
        type Backend = Autodiff<Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let initial_model = args.initial_behavior_clone.as_ref().map(|directory| {
            let model_path = verify_behavior_clone_artifact(
                directory,
                initial_artifact_sha256
                    .as_deref()
                    .expect("initial artifact hash was computed"),
                &training.model,
            )
            .unwrap_or_else(|error| panic!("invalid initial behavior clone: {error}"));
            PolicyValueNetConfig {
                hidden1: training.model.hidden1,
                hidden2: training.model.hidden2,
                recurrent_size: training.model.recurrent_size,
            }
            .init::<Backend>(&device)
            .load_file(model_path, &CompactRecorder::new(), &device)
            .unwrap_or_else(|error| panic!("failed to load initial behavior clone: {error}"))
        });
        let (model, metrics) = behavior_clone_from_model::<Backend>(
            &datasets,
            &training.model,
            &cloning,
            initial_model,
            device,
        )
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
        let initial_model = args.initial_behavior_clone.as_ref().map(|directory| {
            let model_path = verify_behavior_clone_artifact(
                directory,
                initial_artifact_sha256
                    .as_deref()
                    .expect("initial artifact hash was computed"),
                &training.model,
            )
            .unwrap_or_else(|error| panic!("invalid initial behavior clone: {error}"));
            PolicyValueNetConfig {
                hidden1: training.model.hidden1,
                hidden2: training.model.hidden2,
                recurrent_size: training.model.recurrent_size,
            }
            .init::<Backend>(&device)
            .load_file(model_path, &CompactRecorder::new(), &device)
            .unwrap_or_else(|error| panic!("failed to load initial behavior clone: {error}"))
        });
        let (model, metrics) = behavior_clone_from_model::<Backend>(
            &datasets,
            &training.model,
            &cloning,
            initial_model,
            device,
        )
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
