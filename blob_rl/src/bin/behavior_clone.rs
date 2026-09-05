#![recursion_limit = "512"]
//! Supervised warm-start training from immutable demonstration datasets.

use std::path::PathBuf;

use blob_rl::behavior_cloning::{
    behavior_clone_artifact_sha256, behavior_clone_from_model, load_dataset_directories,
    publish_behavior_clone, verify_behavior_clone_artifact_with_schema, ActionBalancingStrategy,
    BehaviorCloningConfig, BehaviorCloningFamilyMetrics, BehaviorCloningPhaseMetrics,
    DatasetSamplingStrategy, ExpertRoutingStrategy, SupervisionPhase,
};
use blob_rl::config::{ModelConfig, TrainingConfig};
use blob_rl::model::{PolicyValueNet, PolicyValueNetConfig};
use burn::module::Module;
use burn::prelude::Backend;
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExpertRouting {
    ForagingInteractionExploration,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ActionKindExpert {
    Foraging,
    Interaction,
    Exploration,
}

impl From<ActionKindExpert> for SupervisionPhase {
    fn from(value: ActionKindExpert) -> Self {
        match value {
            ActionKindExpert::Foraging => Self::Feeding,
            ActionKindExpert::Interaction => Self::Combat,
            ActionKindExpert::Exploration => Self::Exploration,
        }
    }
}

impl From<ExpertRouting> for ExpertRoutingStrategy {
    fn from(value: ExpertRouting) -> Self {
        match value {
            ExpertRouting::ForagingInteractionExploration => Self::ForagingInteractionExploration,
        }
    }
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

fn format_family_metrics(metrics: &[BehaviorCloningFamilyMetrics]) -> String {
    metrics
        .iter()
        .filter(|metrics| metrics.samples > 0)
        .map(|metrics| {
            format!(
                "{}={:.1}% combined/{:.1}% expert/{:.1}% gate/{:.1}% exact (n={})",
                metrics.family,
                metrics.action_kind_accuracy.unwrap_or(0.0) * 100.0,
                metrics.routed_expert_action_kind_accuracy.unwrap_or(0.0) * 100.0,
                metrics.phase_gate_accuracy.unwrap_or(0.0) * 100.0,
                metrics.exact_accuracy.unwrap_or(0.0) * 100.0,
                metrics.samples,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_phase_metrics(metrics: &[BehaviorCloningPhaseMetrics]) -> String {
    metrics
        .iter()
        .filter(|metrics| metrics.samples > 0)
        .map(|metrics| {
            format!(
                "{}={:.1}% gate (n={})",
                metrics.phase,
                metrics.gate_accuracy.unwrap_or(0.0) * 100.0,
                metrics.samples,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn load_initial_model<B: Backend>(
    directory: &std::path::Path,
    artifact_sha256: &str,
    model: &ModelConfig,
    device: &B::Device,
) -> Result<PolicyValueNet<B>, String> {
    let (model_path, schema_version) =
        verify_behavior_clone_artifact_with_schema(directory, artifact_sha256, model)?;
    let config = PolicyValueNetConfig {
        hidden1: model.hidden1,
        hidden2: model.hidden2,
        recurrent_size: model.recurrent_size,
    };
    if schema_version < 25 {
        let legacy = config
            .init_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load legacy behavior clone: {error}"))?;
        Ok(config.migrate_legacy(legacy, device))
    } else if schema_version < 28 {
        let legacy = config
            .init_foraging_adapter_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load foraging-adapter behavior clone: {error}"))?;
        Ok(config.migrate_foraging_adapter(legacy, device))
    } else if schema_version < 30 {
        let legacy = config
            .init_header_context_adapter_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load header-context-adapter clone: {error}"))?;
        Ok(config.migrate_header_context_adapter(legacy, device))
    } else if schema_version < 31 {
        Err("experimental schema-30 slot-only clones are not migratable; retrain the unpromoted treatment from its verified schema-28/29 parent".into())
    } else if schema_version < 34 {
        let legacy = config
            .init_slot_context_adapter_legacy::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load slot-context-adapter clone: {error}"))?;
        Ok(config.migrate_slot_context_adapter(legacy, device))
    } else {
        config
            .init::<B>(device)
            .load_file(model_path, &CompactRecorder::new(), device)
            .map_err(|error| format!("failed to load behavior clone: {error}"))
    }
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

    /// Exact sample presentations per epoch, independent of corpus size.
    #[arg(long)]
    epoch_samples: Option<usize>,

    /// Exact optimizer updates per epoch, independent of recurrent chunk mix.
    #[arg(long)]
    optimizer_steps_per_epoch: Option<usize>,

    /// Stop after this exact total number of optimizer updates.
    #[arg(long)]
    optimizer_step_budget: Option<usize>,

    /// Learning-rate multiplier for only the final retained optimizer update.
    #[arg(long, default_value_t = 1.0)]
    terminal_optimizer_step_scale: f64,

    /// Strictly local rule used to supervise the three-expert gate.
    #[arg(
        long,
        value_enum,
        default_value_t = ExpertRouting::ForagingInteractionExploration
    )]
    expert_routing: ExpertRouting,

    /// Relative auxiliary loss for the local per-decision expert gate.
    #[arg(long, default_value_t = 1.0)]
    phase_gate_loss_weight: f64,

    /// Update only this action-kind expert, freezing every other parent parameter.
    #[arg(long, value_enum)]
    action_kind_expert_only: Option<ActionKindExpert>,

    /// Update only the observation-local foraging residual.
    #[arg(long)]
    foraging_adapter_only: bool,

    /// Update only this interaction/exploration action-kind residual.
    #[arg(long, value_enum)]
    context_adapter_only: Option<ActionKindExpert>,

    /// Update only this context's non-random local-header/raw-slot residual.
    #[arg(long, value_enum)]
    context_slot_adapter_only: Option<ActionKindExpert>,

    /// Update only the exploration Guard readiness residual over local energy.
    #[arg(long)]
    exploration_guard_readiness_only: bool,

    /// Desired teacher-kind lead over the strongest other legal kind.
    #[arg(long)]
    action_kind_margin: Option<f64>,

    /// Relative weight of the action-kind margin auxiliary.
    #[arg(long, default_value_t = 0.0)]
    action_kind_margin_loss_weight: f64,

    /// Update only the action-kind-conditioned target query head.
    #[arg(long)]
    target_query_head_only: bool,

    /// Update only the action-kind-conditioned effort head.
    #[arg(long)]
    effort_head_only: bool,

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
        epoch_sample_budget: args.epoch_samples,
        optimizer_steps_per_epoch: args.optimizer_steps_per_epoch,
        optimizer_step_budget: args.optimizer_step_budget,
        terminal_optimizer_step_scale: args.terminal_optimizer_step_scale,
        expert_routing: args.expert_routing.into(),
        phase_gate_loss_weight: args.phase_gate_loss_weight,
        action_kind_expert_only: args.action_kind_expert_only.map(Into::into),
        foraging_adapter_only: args.foraging_adapter_only,
        context_adapter_only: args.context_adapter_only.map(Into::into),
        context_slot_adapter_only: args.context_slot_adapter_only.map(Into::into),
        exploration_guard_readiness_only: args.exploration_guard_readiness_only,
        action_kind_margin: args.action_kind_margin,
        action_kind_margin_loss_weight: args.action_kind_margin_loss_weight,
        target_query_head_only: args.target_query_head_only,
        effort_head_only: args.effort_head_only,
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
            load_initial_model::<Backend>(
                directory,
                initial_artifact_sha256
                    .as_deref()
                    .expect("initial artifact hash was computed"),
                &training.model,
                &device,
            )
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
        println!(
            "Held-out family accuracy: {}",
            format_family_metrics(&metrics.final_validation_family_metrics)
        );
        println!(
            "Held-out phase accuracy: {}",
            format_phase_metrics(&metrics.final_validation_phase_metrics)
        );
    }

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    {
        use burn::backend::{Autodiff, NdArray};
        type Backend = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let initial_model = args.initial_behavior_clone.as_ref().map(|directory| {
            load_initial_model::<Backend>(
                directory,
                initial_artifact_sha256
                    .as_deref()
                    .expect("initial artifact hash was computed"),
                &training.model,
                &device,
            )
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
        println!(
            "Held-out family accuracy: {}",
            format_family_metrics(&metrics.final_validation_family_metrics)
        );
        println!(
            "Held-out phase accuracy: {}",
            format_phase_metrics(&metrics.final_validation_phase_metrics)
        );
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("behavior-clone requires the wgpu or ndarray feature");
}
