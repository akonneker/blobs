//! Masked supervised pretraining from verified demonstration datasets.

use std::collections::{BTreeMap, HashMap};

mod dataset_partition;
pub mod seed_ledger;
use dataset_partition::{partition_datasets, DatasetPartition};
use seed_ledger::{InheritedSeedLedger, SeedLedger};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use burn::optim::grad_clipping::GradientClipping;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::action::{
    decompose_policy_action, policy_action_family, policy_action_kind_mask, policy_effort_mask,
    policy_target_mask, PolicyActionFamily, NUM_ACTIONS, NUM_AMOUNT_CHOICES,
    NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS, NUM_POLICY_EFFORTS,
    NUM_POLICY_EFFORT_LOGITS, NUM_POLICY_TARGETS, NUM_POLICY_TARGET_LOGITS, NUM_SIGNAL_CHOICES,
    NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::artifact::training_backend_id;
use crate::config::ModelConfig;
use crate::demonstration::{load_demonstrations, DemonstrationSample, LoadedDemonstrations};
use crate::model::{
    decode_policy_memory, encode_policy_memory, PolicyValueNet, PolicyValueNetConfig,
    NUM_ACTION_KIND_EXPERTS,
};
use crate::observation::{observation_expert_context, ObservationExpertContext, OBS_DIM};

pub const BEHAVIOR_CLONING_SCHEMA_VERSION: u32 = 37;
const MIN_SUPPORTED_BEHAVIOR_CLONING_SCHEMA_VERSION: u32 = 22;
static CLONING_NONCE: AtomicU64 = AtomicU64::new(0);

const fn unit_learning_rate_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetSamplingStrategy {
    /// Present every training sample once per epoch.
    Proportional,
    /// Give every dataset the same number of presentations per epoch while
    /// staying within the overall proportional epoch sample budget.
    Balanced,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ActionBalancingStrategy {
    None,
    /// Equalize aggregate loss across coarse physical families.
    Family,
    /// Equalize every represented flat catalog label. This is useful for a
    /// targeted warm start where direction/slot labels would otherwise be
    /// overwhelmed by a single non-targeted action such as Consume.
    Label,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SupervisionPhase {
    Feeding,
    Combat,
    Exploration,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ExpertRoutingStrategy {
    /// Attack and Guard use the combat expert; every other physical action
    /// uses the general/feeding expert. The assignment is per decision and
    /// contains no host scenario phase.
    LocalActionFamily,
    /// Route every observation with a visible neighboring cell through the
    /// combat expert. Unlike action-family routing, this labels Move, Wait,
    /// Consume, and other decisions from an interaction sequence using only
    /// features present in the anonymous Mind observation.
    VisibleNeighborContext,
    /// Route from anonymous local evidence with strict precedence: a visible
    /// attack windup or guard selects interaction, otherwise directly
    /// a plant tile or consumable loose energy under the cell selects foraging,
    /// otherwise any visible neighboring cell selects interaction, and all
    /// remaining observations select exploration. Diffuse energy is not food
    /// for a cell and does not select foraging.
    ForagingInteractionExploration,
}

impl SupervisionPhase {
    pub const ALL: [Self; NUM_ACTION_KIND_EXPERTS] =
        [Self::Feeding, Self::Combat, Self::Exploration];

    pub const fn index(self) -> usize {
        match self {
            Self::Feeding => 0,
            Self::Combat => 1,
            Self::Exploration => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Feeding => "foraging",
            Self::Combat => "interaction",
            Self::Exploration => "exploration",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningConfig {
    pub seed: u64,
    /// Hash of the verified behavior-cloning artifact used to initialize this
    /// stage, or `None` for a fresh model. The path is deliberately excluded
    /// so artifact identity remains machine-independent.
    pub initial_artifact_sha256: Option<String>,
    #[serde(default)]
    pub inherited_seed_ledger: Option<InheritedSeedLedger>,
    /// Reserved final confirmation seeds, excluded from all demonstration input.
    #[serde(default)]
    pub confirmation_seeds: Vec<u64>,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    /// Fraction of distinct source seeds held out as complete trajectories.
    /// Existing lineage roles are fixed; this fraction guides assignment of new seeds.
    pub validation_fraction: f64,
    pub dataset_sampling: DatasetSamplingStrategy,
    /// Optional explicit sampling mass for each dataset, in CLI order. When
    /// present this overrides `dataset_sampling` while preserving the same
    /// total presentations per epoch.
    #[serde(default)]
    pub dataset_sampling_weights: Vec<f64>,
    /// Optional exact number of sample presentations in each epoch. This
    /// makes paired experiments comparable even when one arm adds a dataset.
    #[serde(default)]
    pub epoch_sample_budget: Option<usize>,
    /// Optional exact optimizer updates in each epoch. Recurrent chunks are
    /// still kept intact and grouped by length, but each length group is split
    /// into a deterministic number of batches. This closes the update-count
    /// confound in paired mixtures with different trajectory-length profiles.
    #[serde(default)]
    pub optimizer_steps_per_epoch: Option<usize>,
    /// Optional exact total optimizer-update prefix to retain. This requires
    /// `optimizer_steps_per_epoch`, making mid-epoch checkpoints reproducible
    /// prefixes of the corresponding uninterrupted run.
    #[serde(default)]
    pub optimizer_step_budget: Option<usize>,
    /// Learning-rate multiplier applied only to the final retained optimizer
    /// update. Values below one require an exact total step budget so the
    /// partially dosed update is unambiguous and reproducible.
    #[serde(default = "unit_learning_rate_scale")]
    pub terminal_optimizer_step_scale: f64,
    /// Per-decision expert assignment, bound into the artifact.
    pub expert_routing: ExpertRoutingStrategy,
    /// Relative auxiliary loss used to teach the observation-driven phase
    /// gate. Expert action loss retains unit weight.
    pub phase_gate_loss_weight: f64,
    /// Optional stage-local adaptation mode. Only the selected action-kind
    /// expert is retained after each optimizer update; the shared recurrent
    /// trunk and every other head remain exactly at their parent values.
    /// This requires a verified initial artifact.
    #[serde(default)]
    pub action_kind_expert_only: Option<SupervisionPhase>,
    /// Update only the observation-local foraging residual while preserving
    /// every parameter inherited from the verified parent.
    #[serde(default)]
    pub foraging_adapter_only: bool,
    /// Update only one observation-local interaction or exploration residual
    /// while preserving every parameter inherited from the verified parent.
    #[serde(default)]
    pub context_adapter_only: Option<SupervisionPhase>,
    /// Update only one non-random local-header plus featurewise pooled
    /// raw-slot residual while preserving the context's inherited recurrent
    /// residual and every other parameter.
    #[serde(default)]
    pub context_slot_adapter_only: Option<SupervisionPhase>,
    /// Update only the scalar exploration Guard readiness residual over local
    /// assimilated and gut energy, preserving every inherited parameter.
    #[serde(default)]
    pub exploration_guard_readiness_only: bool,
    /// Optional desired teacher-kind logit lead over the strongest other
    /// legal kind. When enabled, this hinge objective replaces action-kind
    /// cross-entropy while leaving every conditional head and gate supervised.
    /// Zero targets only currently tied or misclassified samples.
    #[serde(default)]
    pub action_kind_margin: Option<f64>,
    /// Relative weight of the multiclass hinge margin auxiliary. This must be
    /// positive exactly when `action_kind_margin` is enabled.
    #[serde(default)]
    pub action_kind_margin_loss_weight: f64,
    /// Update only the action-kind-conditioned target query head while
    /// preserving every parameter inherited from the verified parent.
    #[serde(default)]
    pub target_query_head_only: bool,
    /// Update only the cell-private randomness/local-slot target residual.
    #[serde(default)]
    pub target_residual_only: bool,
    /// Update only the action-kind-conditioned effort head while preserving
    /// every parameter inherited from the verified parent.
    #[serde(default)]
    pub effort_head_only: bool,
    /// Maximum number of consecutive decisions from one cell kept in a single
    /// recurrent autodiff graph.
    pub recurrent_unroll_steps: usize,
    /// Restrict training to decisions exactly reproduced by the current
    /// decoder. This normally excludes stateful colony demonstrations.
    pub exact_round_trip_only: bool,
    pub action_balancing: ActionBalancingStrategy,
    /// Exponent applied to inverse-frequency action weights. Zero disables
    /// their effect and one applies full balancing.
    pub action_balance_exponent: f64,
    /// Optional upper bound on the ratio between the largest and smallest
    /// represented per-sample action weights. This prevents a newly introduced
    /// rare family from overwhelming established behavior while retaining
    /// useful balancing among common families.
    pub action_balance_max_ratio: Option<f64>,
}

impl BehaviorCloningConfig {
    pub fn validate(&self) -> Result<(), String> {
        seed_ledger::inherited_roles(self)?;
        let valid_initial_artifact = self.initial_artifact_sha256.as_ref().is_none_or(|hash| {
            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        });
        let adaptation_modes = usize::from(self.action_kind_expert_only.is_some())
            + usize::from(self.foraging_adapter_only)
            + usize::from(self.context_adapter_only.is_some())
            + usize::from(self.context_slot_adapter_only.is_some())
            + usize::from(self.exploration_guard_readiness_only)
            + usize::from(self.target_query_head_only)
            + usize::from(self.target_residual_only)
            + usize::from(self.effort_head_only);
        if !valid_initial_artifact
            || self.epochs == 0
            || self.minibatch_size == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.validation_fraction.is_finite()
            || !(0.0..1.0).contains(&self.validation_fraction)
            || self
                .dataset_sampling_weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || self.epoch_sample_budget == Some(0)
            || self.optimizer_steps_per_epoch == Some(0)
            || self.optimizer_step_budget == Some(0)
            || self.optimizer_step_budget.is_some() && self.optimizer_steps_per_epoch.is_none()
            || self.optimizer_step_budget.is_some_and(|budget| {
                self.optimizer_steps_per_epoch
                    .and_then(|steps| steps.checked_mul(self.epochs))
                    .is_none_or(|maximum| budget > maximum)
            })
            || !self.terminal_optimizer_step_scale.is_finite()
            || !(self.terminal_optimizer_step_scale > 0.0
                && self.terminal_optimizer_step_scale <= 1.0)
            || (self.terminal_optimizer_step_scale != 1.0 && self.optimizer_step_budget.is_none())
            || self.expert_routing != ExpertRoutingStrategy::ForagingInteractionExploration
            || !self.phase_gate_loss_weight.is_finite()
            || self.phase_gate_loss_weight <= 0.0
            || (self.action_kind_expert_only.is_some() && self.initial_artifact_sha256.is_none())
            || (self.foraging_adapter_only && self.initial_artifact_sha256.is_none())
            || (self.context_adapter_only.is_some() && self.initial_artifact_sha256.is_none())
            || (self.context_slot_adapter_only.is_some() && self.initial_artifact_sha256.is_none())
            || (self.exploration_guard_readiness_only && self.initial_artifact_sha256.is_none())
            || adaptation_modes > 1
            || self.context_adapter_only == Some(SupervisionPhase::Feeding)
            || self.context_slot_adapter_only == Some(SupervisionPhase::Feeding)
            || self
                .action_kind_margin
                .is_some_and(|margin| !margin.is_finite() || !(0.0..=100.0).contains(&margin))
            || !self.action_kind_margin_loss_weight.is_finite()
            || !(0.0..=1_000.0).contains(&self.action_kind_margin_loss_weight)
            || (self.action_kind_margin.is_some() != (self.action_kind_margin_loss_weight > 0.0))
            || (self.target_query_head_only && self.initial_artifact_sha256.is_none())
            || (self.target_residual_only && self.initial_artifact_sha256.is_none())
            || (self.effort_head_only && self.initial_artifact_sha256.is_none())
            || (self.foraging_adapter_only && self.action_kind_expert_only.is_some())
            || (self.context_adapter_only.is_some()
                && (self.foraging_adapter_only || self.action_kind_expert_only.is_some()))
            || (self.context_slot_adapter_only.is_some()
                && (self.foraging_adapter_only
                    || self.context_adapter_only.is_some()
                    || self.action_kind_expert_only.is_some()))
            || (self.target_query_head_only
                && (self.foraging_adapter_only
                    || self.context_adapter_only.is_some()
                    || self.context_slot_adapter_only.is_some()
                    || self.action_kind_expert_only.is_some()))
            || (self.effort_head_only
                && (self.foraging_adapter_only
                    || self.context_adapter_only.is_some()
                    || self.context_slot_adapter_only.is_some()
                    || self.target_query_head_only
                    || self.action_kind_expert_only.is_some()))
            || self.recurrent_unroll_steps == 0
            || self.recurrent_unroll_steps > 256
            || !self.action_balance_exponent.is_finite()
            || !(0.0..=1.0).contains(&self.action_balance_exponent)
            || self
                .action_balance_max_ratio
                .is_some_and(|ratio| !ratio.is_finite() || ratio < 1.0)
        {
            return Err(
                "behavior-cloning initial artifact must be a SHA-256 hash; expert-head-only, foraging-adapter-only, interaction/exploration context-adapter-only and context-slot-adapter-only, exploration-Guard-readiness-only, target-query-head-only, target-residual-only, and effort-head-only adaptation require an initial artifact and are mutually exclusive; authoritative three-context expert routing is required; epochs, batch size, learning rate, explicit dataset weights, epoch sample and optimizer-step budgets, and phase-gate loss weight must be positive; a total optimizer-step budget requires a fixed per-epoch budget and cannot exceed the configured run; a fractional terminal optimizer-step scale must be in (0, 1] and requires an exact total step budget; action-kind margin and its positive loss weight must be enabled together and bounded; validation fraction and action-balance exponent must be in [0, 1]; action-balance max ratio must be finite and at least 1; and recurrent unroll steps must be in 1..=256".into(),
            );
        }
        Ok(())
    }
}

impl Default for BehaviorCloningConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            initial_artifact_sha256: None,
            inherited_seed_ledger: None,
            confirmation_seeds: Vec::new(),
            epochs: 10,
            minibatch_size: 256,
            learning_rate: 3e-4,
            validation_fraction: 0.1,
            dataset_sampling: DatasetSamplingStrategy::Balanced,
            dataset_sampling_weights: Vec::new(),
            epoch_sample_budget: None,
            optimizer_steps_per_epoch: None,
            optimizer_step_budget: None,
            terminal_optimizer_step_scale: 1.0,
            expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
            phase_gate_loss_weight: 1.0,
            action_kind_expert_only: None,
            foraging_adapter_only: false,
            context_adapter_only: None,
            context_slot_adapter_only: None,
            exploration_guard_readiness_only: false,
            action_kind_margin: None,
            action_kind_margin_loss_weight: 0.0,
            target_query_head_only: false,
            target_residual_only: false,
            effort_head_only: false,
            recurrent_unroll_steps: 16,
            exact_round_trip_only: false,
            action_balancing: ActionBalancingStrategy::None,
            action_balance_exponent: 1.0,
            action_balance_max_ratio: None,
        }
    }
}

/// Choose an exact number of nonempty, equal-length recurrent batches while
/// respecting the sample minibatch ceiling. Each tuple is `(chunk_len,
/// chunk_count)`. The minimum is the ordinary ceiling-based batching count;
/// the maximum is one chunk per optimizer update.
fn exact_recurrent_batch_counts(
    groups: &[(usize, usize)],
    minibatch_size: usize,
    target_steps: usize,
) -> Result<Vec<usize>, String> {
    let mut counts = groups
        .iter()
        .map(|(length, chunks)| {
            let capacity = (minibatch_size / *length).max(1);
            chunks.div_ceil(capacity)
        })
        .collect::<Vec<_>>();
    let minimum = counts.iter().sum::<usize>();
    let maximum = groups.iter().map(|(_, chunks)| chunks).sum::<usize>();
    if !(minimum..=maximum).contains(&target_steps) {
        return Err(format!(
            "optimizer-step budget {target_steps} is infeasible for recurrent epoch; valid range is {minimum}..={maximum}"
        ));
    }

    let mut assigned = minimum;
    while assigned < target_steps {
        let selected = groups
            .iter()
            .enumerate()
            .filter(|(index, (_, chunks))| counts[*index] < *chunks)
            .max_by(
                |(left_index, (left_length, left_chunks)),
                 (right_index, (right_length, right_chunks))| {
                    let left_score =
                        *left_chunks as u128 * *left_length as u128 * counts[*right_index] as u128;
                    let right_score =
                        *right_chunks as u128 * *right_length as u128 * counts[*left_index] as u128;
                    left_score
                        .cmp(&right_score)
                        .then_with(|| right_index.cmp(left_index))
                },
            )
            .map(|(index, _)| index)
            .ok_or("optimizer-step allocation exhausted recurrent chunks")?;
        counts[selected] += 1;
        assigned += 1;
    }
    Ok(counts)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningMetrics {
    pub datasets: usize,
    pub available_samples: usize,
    pub eligible_samples: usize,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub samples_per_epoch: usize,
    pub sample_presentations: usize,
    /// Actual supervised presentations by physical action family over every
    /// epoch, ordered as in `PolicyActionFamily`.
    pub action_family_presentations: Vec<usize>,
    /// Sum of per-sample loss weights by action family over every epoch. This
    /// makes the effective, post-resampling supervision mixture auditable.
    pub action_family_weighted_loss_mass: Vec<f64>,
    pub supervision_phase_presentations: Vec<usize>,
    pub supervision_phase_weighted_gate_loss_mass: Vec<f64>,
    pub recurrent_unroll_steps: usize,
    pub training_trajectories: usize,
    pub validation_trajectories: usize,
    pub exact_round_trip_samples: usize,
    /// Number of completely processed epochs. A final partial epoch is
    /// represented by `optimizer_steps` and actual presentation counters.
    #[serde(default)]
    pub completed_epochs: usize,
    pub epochs: usize,
    pub optimizer_steps: usize,
    pub initial_training_loss: f64,
    pub final_training_loss: f64,
    pub initial_training_accuracy: f64,
    pub final_training_accuracy: f64,
    pub initial_validation_loss: Option<f64>,
    pub final_validation_loss: Option<f64>,
    pub initial_validation_accuracy: Option<f64>,
    pub final_validation_accuracy: Option<f64>,
    pub initial_training_family_metrics: Vec<BehaviorCloningFamilyMetrics>,
    pub final_training_family_metrics: Vec<BehaviorCloningFamilyMetrics>,
    pub initial_validation_family_metrics: Vec<BehaviorCloningFamilyMetrics>,
    pub final_validation_family_metrics: Vec<BehaviorCloningFamilyMetrics>,
    pub initial_training_phase_metrics: Vec<BehaviorCloningPhaseMetrics>,
    pub final_training_phase_metrics: Vec<BehaviorCloningPhaseMetrics>,
    pub initial_validation_phase_metrics: Vec<BehaviorCloningPhaseMetrics>,
    pub final_validation_phase_metrics: Vec<BehaviorCloningPhaseMetrics>,
    pub partitions: Vec<BehaviorCloningDatasetPartition>,
}

/// Exact-label and action-kind accuracy for one physical action family. The
/// vector containing this record is ordered as Wait, Guard, Consume, Move,
/// Attack, Split, Regurgitate, Terrain, Signal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningFamilyMetrics {
    pub family: String,
    pub samples: usize,
    pub action_kind_accuracy: Option<f64>,
    pub routed_expert_action_kind_accuracy: Option<f64>,
    pub phase_gate_accuracy: Option<f64>,
    pub exact_accuracy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningPhaseMetrics {
    pub phase: String,
    pub samples: usize,
    pub gate_accuracy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningDatasetPartition {
    pub manifest_sha256: String,
    pub eligible_phase_samples: Vec<usize>,
    pub training_phase_samples: Vec<usize>,
    pub validation_phase_samples: Vec<usize>,
    pub eligible_samples: usize,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub training_trajectories: usize,
    pub validation_trajectories: usize,
    pub samples_per_epoch: usize,
    pub held_out_seeds: Vec<u64>,
    #[serde(default)]
    pub training_seeds: Vec<u64>,
    /// Exact policy-catalog label counts, indexed by action ID. These make
    /// target-slot fragmentation and dominant fallback actions visible before
    /// a warm start is trusted.
    pub eligible_action_histogram: Vec<usize>,
    pub training_action_histogram: Vec<usize>,
    pub validation_action_histogram: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningDatasetIdentity {
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub teacher: String,
    pub samples: usize,
    pub exact_round_trip_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningArtifact {
    #[serde(default)]
    pub seed_ledger: Option<SeedLedger>,
    #[serde(default)]
    pub execution: Option<crate::policy_artifact::PolicyExecutionIdentity>,
    #[serde(default)]
    pub code_revision: Option<String>,
    pub schema_version: u32,
    pub package_version: String,
    pub training_backend: String,
    pub config: BehaviorCloningConfig,
    pub model: ModelConfig,
    pub datasets: Vec<BehaviorCloningDatasetIdentity>,
    pub metrics: BehaviorCloningMetrics,
    pub model_file: String,
    pub model_sha256: String,
}

fn action_histogram<'a>(samples: impl IntoIterator<Item = &'a DemonstrationSample>) -> Vec<usize> {
    let mut counts = vec![0; NUM_ACTIONS];
    for sample in samples {
        counts[usize::from(sample.action)] += 1;
    }
    counts
}

fn cap_weight_ratio(weights: &mut [f32], represented: &[bool], max_ratio: Option<f64>) {
    let Some(max_ratio) = max_ratio else {
        return;
    };
    let minimum = weights
        .iter()
        .zip(represented)
        .filter_map(|(weight, represented)| represented.then_some(*weight))
        .fold(f32::INFINITY, f32::min);
    let maximum = minimum * max_ratio as f32;
    for (weight, represented) in weights.iter_mut().zip(represented) {
        if *represented {
            *weight = weight.min(maximum);
        }
    }
}

fn normalize_presented_weight_mass(weights: &mut [f32], action_counts: &[usize]) {
    let presentations = action_counts.iter().sum::<usize>() as f64;
    let weighted_mass = weights
        .iter()
        .zip(action_counts)
        .map(|(weight, count)| f64::from(*weight) * *count as f64)
        .sum::<f64>();
    if presentations == 0.0 || weighted_mass == 0.0 {
        return;
    }
    let scale = (presentations / weighted_mass) as f32;
    for weight in weights {
        *weight *= scale;
    }
}

fn action_weights_for_samples<'a>(
    samples: impl IntoIterator<Item = &'a DemonstrationSample>,
    strategy: ActionBalancingStrategy,
    exponent: f64,
    max_ratio: Option<f64>,
) -> Vec<f32> {
    if strategy == ActionBalancingStrategy::None {
        return vec![1.0; NUM_ACTIONS];
    }
    let mut action_counts = vec![0usize; NUM_ACTIONS];
    for sample in samples {
        action_counts[usize::from(sample.action)] += 1;
    }
    let mut represented_actions = vec![false; NUM_ACTIONS];
    let mut weights: Vec<f32> = match strategy {
        ActionBalancingStrategy::None => unreachable!(),
        ActionBalancingStrategy::Label => {
            let represented = action_counts.iter().filter(|count| **count > 0).count();
            let total = action_counts.iter().sum::<usize>();
            represented_actions
                .iter_mut()
                .zip(&action_counts)
                .for_each(|(represented, count)| *represented = *count > 0);
            action_counts
                .iter()
                .copied()
                .map(|count| {
                    if count == 0 {
                        1.0
                    } else {
                        (total as f64 / (represented * count) as f64).powf(exponent) as f32
                    }
                })
                .collect()
        }
        ActionBalancingStrategy::Family => {
            let mut family_counts = [0usize; PolicyActionFamily::COUNT];
            for (action, count) in action_counts.iter().copied().enumerate() {
                if count > 0 {
                    let family = policy_action_family(action)
                        .expect("policy action histogram index is in the catalog");
                    family_counts[family.index()] += count;
                }
            }
            let represented = family_counts.iter().filter(|count| **count > 0).count();
            let total = family_counts.iter().sum::<usize>();
            for (action, represented_action) in represented_actions.iter_mut().enumerate() {
                let family =
                    policy_action_family(action).expect("policy action index is in the catalog");
                *represented_action = family_counts[family.index()] > 0;
            }
            (0..NUM_ACTIONS)
                .map(|action| {
                    let family = policy_action_family(action)
                        .expect("policy action index is in the catalog");
                    let count = family_counts[family.index()];
                    if count == 0 {
                        1.0
                    } else {
                        (total as f64 / (represented * count) as f64).powf(exponent) as f32
                    }
                })
                .collect()
        }
    };
    cap_weight_ratio(&mut weights, &represented_actions, max_ratio);
    normalize_presented_weight_mass(&mut weights, &action_counts);
    weights
}

fn balanced_phase_weights(
    counts: [usize; NUM_ACTION_KIND_EXPERTS],
) -> [f32; NUM_ACTION_KIND_EXPERTS] {
    let represented = counts.iter().filter(|count| **count > 0).count();
    let total = counts.iter().sum::<usize>();
    std::array::from_fn(|phase| {
        if counts[phase] == 0 {
            1.0
        } else {
            total as f32 / (represented * counts[phase]) as f32
        }
    })
}

pub fn expert_routing_phase(
    sample: &DemonstrationSample,
    strategy: ExpertRoutingStrategy,
) -> SupervisionPhase {
    match strategy {
        ExpertRoutingStrategy::LocalActionFamily => {
            match policy_action_family(usize::from(sample.action))
                .expect("validated demonstration action belongs to a policy family")
            {
                PolicyActionFamily::Attack | PolicyActionFamily::Guard => SupervisionPhase::Combat,
                _ => SupervisionPhase::Feeding,
            }
        }
        ExpertRoutingStrategy::VisibleNeighborContext => {
            if sample
                .observation
                .get(crate::observation::HEADER_FEATURES..)
                .is_some_and(|slots| {
                    slots
                        .chunks_exact(crate::observation::SLOT_FEATURES)
                        .any(|slot| slot[crate::observation::SLOT_NEIGHBOR_PRESENT_FEATURE] > 0.5)
                })
            {
                SupervisionPhase::Combat
            } else {
                SupervisionPhase::Feeding
            }
        }
        ExpertRoutingStrategy::ForagingInteractionExploration => {
            match observation_expert_context(&sample.observation) {
                ObservationExpertContext::Foraging => SupervisionPhase::Feeding,
                ObservationExpertContext::Interaction => SupervisionPhase::Combat,
                ObservationExpertContext::Exploration => SupervisionPhase::Exploration,
            }
        }
    }
}

fn phase_histogram<'a>(
    samples: impl IntoIterator<Item = &'a DemonstrationSample>,
    strategy: ExpertRoutingStrategy,
) -> Vec<usize> {
    let mut counts = vec![0usize; NUM_ACTION_KIND_EXPERTS];
    for sample in samples {
        counts[expert_routing_phase(sample, strategy).index()] += 1;
    }
    counts
}

fn phase_weights_for_chunks(
    chunks: &[SequenceChunk<'_>],
    strategy: ExpertRoutingStrategy,
) -> [f32; NUM_ACTION_KIND_EXPERTS] {
    let mut counts = [0usize; NUM_ACTION_KIND_EXPERTS];
    for chunk in chunks {
        for sample in &chunk.samples {
            counts[expert_routing_phase(sample, strategy).index()] += 1;
        }
    }
    balanced_phase_weights(counts)
}

/// The gate does not receive recurrent memory or host scenario metadata. Fail
/// closed if the immutable corpus asks it to route an identical anonymous
/// observation to both experts.
fn validate_gate_label_consistency(
    partitions: &[DatasetPartition<'_>],
    strategy: ExpertRoutingStrategy,
) -> Result<(), String> {
    let mut labels = HashMap::<[u8; 32], SupervisionPhase>::new();
    for sample in partitions
        .iter()
        .flat_map(|partition| partition.training.iter().chain(&partition.validation))
    {
        let mut hasher = Sha256::new();
        hasher.update(b"blob-expert-gate-observation-v1");
        for value in &sample.observation {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let key = hasher.finalize().into();
        let phase = expert_routing_phase(sample, strategy);
        if labels
            .insert(key, phase)
            .is_some_and(|prior| prior != phase)
        {
            return Err(
                "identical anonymous observations have contradictory expert-routing labels".into(),
            );
        }
    }
    Ok(())
}

fn samples_per_dataset_per_epoch(
    partitions: &[DatasetPartition<'_>],
    strategy: DatasetSamplingStrategy,
    explicit_weights: &[f64],
    epoch_sample_budget: Option<usize>,
) -> Result<Vec<usize>, String> {
    fn apportion(total: usize, weights: &[f64]) -> Vec<usize> {
        let weight_total = weights.iter().sum::<f64>();
        let raw = weights
            .iter()
            .map(|weight| total as f64 * weight / weight_total)
            .collect::<Vec<_>>();
        let mut samples = raw
            .iter()
            .map(|value| value.floor() as usize)
            .collect::<Vec<_>>();
        let remainder = total.saturating_sub(samples.iter().sum());
        let mut fractional_order = raw
            .iter()
            .enumerate()
            .map(|(index, value)| (index, value.fract()))
            .collect::<Vec<_>>();
        fractional_order.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (index, _) in fractional_order.into_iter().take(remainder) {
            samples[index] += 1;
        }
        samples
    }

    let natural_total = partitions
        .iter()
        .map(|partition| partition.training.len())
        .sum::<usize>();
    let total = epoch_sample_budget.unwrap_or(natural_total);
    if !explicit_weights.is_empty() {
        if explicit_weights.len() != partitions.len() {
            return Err(format!(
                "received {} dataset sampling weights for {} datasets",
                explicit_weights.len(),
                partitions.len()
            ));
        }
        return Ok(apportion(total, explicit_weights));
    }
    Ok(match strategy {
        DatasetSamplingStrategy::Proportional if epoch_sample_budget.is_none() => partitions
            .iter()
            .map(|partition| partition.training.len())
            .collect(),
        DatasetSamplingStrategy::Proportional => apportion(
            total,
            &partitions
                .iter()
                .map(|partition| partition.training.len() as f64)
                .collect::<Vec<_>>(),
        ),
        DatasetSamplingStrategy::Balanced => apportion(total, &vec![1.0; partitions.len()]),
    })
}

fn kind_mask_bias(sample: &DemonstrationSample) -> impl Iterator<Item = f32> {
    policy_action_kind_mask(&sample.action_mask)
        .into_iter()
        .map(|allowed| if allowed { 0.0 } else { -1.0e9 })
}

fn expert_kind_mask_bias(
    sample: &DemonstrationSample,
    supervision_phase: SupervisionPhase,
) -> Vec<f32> {
    let mask = kind_mask_bias(sample).collect::<Vec<_>>();
    (0..NUM_ACTION_KIND_EXPERTS)
        .flat_map(|expert| {
            (expert == supervision_phase.index())
                .then_some(mask.iter().copied())
                .into_iter()
                .flatten()
                .chain(
                    (expert != supervision_phase.index())
                        .then_some(std::iter::repeat_n(-1.0e9, NUM_POLICY_ACTION_KINDS))
                        .into_iter()
                        .flatten(),
                )
        })
        .collect()
}

fn target_mask_bias(sample: &DemonstrationSample) -> Vec<f32> {
    let choice = decompose_policy_action(usize::from(sample.action))
        .expect("validated demonstration action is in the policy catalog");
    let mut bias = vec![-1.0e9; NUM_POLICY_TARGET_LOGITS];
    let start = choice.kind * NUM_POLICY_TARGETS;
    for (target, allowed) in policy_target_mask(&sample.action_mask, choice.kind)
        .into_iter()
        .enumerate()
    {
        if allowed {
            bias[start + target] = 0.0;
        }
    }
    bias
}

fn effort_mask_bias(sample: &DemonstrationSample) -> Vec<f32> {
    let choice = decompose_policy_action(usize::from(sample.action))
        .expect("validated demonstration action is in the policy catalog");
    let mut bias = vec![-1.0e9; NUM_POLICY_EFFORT_LOGITS];
    let start = choice.kind * NUM_POLICY_EFFORTS;
    for (effort, allowed) in policy_effort_mask(&sample.action_mask, choice.kind, choice.target)
        .into_iter()
        .enumerate()
    {
        if allowed {
            bias[start + effort] = 0.0;
        }
    }
    bias
}

fn signal_mask_bias(sample: &DemonstrationSample) -> impl Iterator<Item = f32> + '_ {
    sample
        .signal_mask
        .iter()
        .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
}

fn signal_strength_mask_bias(sample: &DemonstrationSample) -> impl Iterator<Item = f32> + '_ {
    sample
        .signal_strength_mask
        .iter()
        .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
}

fn amount_mask_bias(sample: &DemonstrationSample) -> Vec<f32> {
    let choice = decompose_policy_action(usize::from(sample.action))
        .expect("validated demonstration action is in the policy catalog");
    let mut bias = vec![-1.0e9; NUM_POLICY_AMOUNT_LOGITS];
    let start = choice.kind * NUM_AMOUNT_CHOICES;
    for (amount, allowed) in sample.amount_mask.iter().enumerate() {
        if *allowed {
            bias[start + amount] = 0.0;
        }
    }
    bias
}

#[derive(Clone)]
struct DemonstrationSequence<'a> {
    samples: Vec<&'a DemonstrationSample>,
    initial_memory: Vec<f32>,
}

#[derive(Clone)]
struct SequenceChunk<'a> {
    samples: Vec<&'a DemonstrationSample>,
    initial_memory: Vec<f32>,
}

fn build_sequences<'a>(samples: &[&'a DemonstrationSample]) -> Vec<DemonstrationSequence<'a>> {
    let mut sequences = BTreeMap::<(u64, u64), Vec<&'a DemonstrationSample>>::new();
    for sample in samples {
        sequences
            .entry((sample.source_seed, sample.source_cell))
            .or_default()
            .push(*sample);
    }
    sequences
        .into_values()
        .map(|samples| DemonstrationSequence {
            initial_memory: samples
                .first()
                .map(|sample| sample.initial_policy_memory.clone())
                .unwrap_or_default(),
            samples,
        })
        .collect()
}

fn canonicalize_memory(memory: &[f32]) -> Vec<f32> {
    decode_policy_memory(&encode_policy_memory(memory), memory.len())
}

fn prepare_sequence_chunks<'a, B: AutodiffBackend>(
    model: &PolicyValueNet<B>,
    sequences: &[DemonstrationSequence<'a>],
    unroll_steps: usize,
    sequence_batch_size: usize,
    device: &B::Device,
) -> Vec<SequenceChunk<'a>>
where
    f32: From<B::FloatElem>,
{
    let recurrent_size = model.recurrent_size();
    let mut chunks = Vec::new();
    for sequence_batch in sequences.chunks(sequence_batch_size.max(1)) {
        let mut positions = vec![0usize; sequence_batch.len()];
        let mut memories = sequence_batch
            .iter()
            .map(|sequence| {
                if sequence.initial_memory.is_empty() {
                    vec![0.0; recurrent_size]
                } else {
                    sequence.initial_memory.clone()
                }
            })
            .collect::<Vec<_>>();
        while positions
            .iter()
            .zip(sequence_batch)
            .any(|(position, sequence)| *position < sequence.samples.len())
        {
            let active = positions
                .iter()
                .zip(sequence_batch)
                .enumerate()
                .filter_map(|(index, (position, sequence))| {
                    (*position < sequence.samples.len()).then_some(index)
                })
                .collect::<Vec<_>>();
            for index in &active {
                let position = positions[*index];
                if position.is_multiple_of(unroll_steps) {
                    let end = (position + unroll_steps).min(sequence_batch[*index].samples.len());
                    chunks.push(SequenceChunk {
                        samples: sequence_batch[*index].samples[position..end].to_vec(),
                        initial_memory: memories[*index].clone(),
                    });
                }
            }
            let observations = active
                .iter()
                .flat_map(|index| {
                    sequence_batch[*index].samples[positions[*index]]
                        .observation
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let memory = active
                .iter()
                .flat_map(|index| memories[*index].iter().copied())
                .collect::<Vec<_>>();
            let next_memory = model
                .forward_with_memory(
                    Tensor::<B, 2>::from_data(
                        TensorData::new(observations, [active.len(), OBS_DIM]),
                        device,
                    ),
                    Tensor::<B, 2>::from_data(
                        TensorData::new(memory, [active.len(), recurrent_size]),
                        device,
                    ),
                )
                .next_memory
                .into_data()
                .to_vec::<f32>()
                .expect("behavior-cloning recurrent state uses f32");
            for (row, index) in active.into_iter().enumerate() {
                memories[index] = canonicalize_memory(
                    &next_memory[row * recurrent_size..(row + 1) * recurrent_size],
                );
                positions[index] += 1;
            }
        }
    }
    chunks
}

fn sample_chunk_epoch<'a>(
    chunks_by_dataset: &[Vec<SequenceChunk<'a>>],
    per_dataset: &[usize],
    rng: &mut ChaCha12Rng,
) -> Vec<SequenceChunk<'a>> {
    let mut epoch = Vec::with_capacity(per_dataset.iter().sum());
    for (chunks, target) in chunks_by_dataset.iter().zip(per_dataset) {
        let mut remaining = *target;
        while remaining > 0 {
            let mut cycle = chunks.clone();
            cycle.shuffle(rng);
            for mut chunk in cycle {
                if remaining == 0 {
                    break;
                }
                if chunk.samples.len() > remaining {
                    chunk.samples.truncate(remaining);
                }
                remaining -= chunk.samples.len();
                epoch.push(chunk);
            }
        }
    }
    epoch.shuffle(rng);
    epoch
}

struct SequenceEvaluation {
    loss: f64,
    exact_accuracy: f64,
    family_metrics: Vec<BehaviorCloningFamilyMetrics>,
    phase_metrics: Vec<BehaviorCloningPhaseMetrics>,
}

fn empty_family_metrics() -> Vec<BehaviorCloningFamilyMetrics> {
    PolicyActionFamily::ALL
        .into_iter()
        .map(|family| BehaviorCloningFamilyMetrics {
            family: family.name().into(),
            samples: 0,
            action_kind_accuracy: None,
            routed_expert_action_kind_accuracy: None,
            phase_gate_accuracy: None,
            exact_accuracy: None,
        })
        .collect()
}

fn empty_phase_metrics() -> Vec<BehaviorCloningPhaseMetrics> {
    SupervisionPhase::ALL
        .into_iter()
        .map(|phase| BehaviorCloningPhaseMetrics {
            phase: phase.name().into(),
            samples: 0,
            gate_accuracy: None,
        })
        .collect()
}

#[derive(Clone, Copy)]
struct EvaluationLossConfig {
    phase_gate_weight: f64,
    action_kind_margin: Option<f64>,
    action_kind_margin_weight: f64,
}

fn evaluate_sequences<B: AutodiffBackend>(
    model: &PolicyValueNet<B>,
    sequences: &[DemonstrationSequence<'_>],
    sequence_batch_size: usize,
    expert_routing: ExpertRoutingStrategy,
    loss_config: EvaluationLossConfig,
    device: &B::Device,
) -> Option<SequenceEvaluation>
where
    f32: From<B::FloatElem>,
{
    if sequences.is_empty() {
        return None;
    }
    let recurrent_size = model.recurrent_size();
    let mut evaluation_phase_counts = [0usize; NUM_ACTION_KIND_EXPERTS];
    for sequence in sequences {
        for sample in &sequence.samples {
            evaluation_phase_counts[expert_routing_phase(sample, expert_routing).index()] += 1;
        }
    }
    let evaluation_phase_weights = balanced_phase_weights(evaluation_phase_counts);
    let mut total_loss = 0.0_f64;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut family_samples = [0usize; PolicyActionFamily::COUNT];
    let mut family_kind_correct = [0usize; PolicyActionFamily::COUNT];
    let mut family_expert_kind_correct = [0usize; PolicyActionFamily::COUNT];
    let mut family_gate_correct = [0usize; PolicyActionFamily::COUNT];
    let mut family_exact_correct = [0usize; PolicyActionFamily::COUNT];
    let mut phase_samples = [0usize; NUM_ACTION_KIND_EXPERTS];
    let mut phase_gate_correct = [0usize; NUM_ACTION_KIND_EXPERTS];
    for sequence_batch in sequences.chunks(sequence_batch_size.max(1)) {
        let mut positions = vec![0usize; sequence_batch.len()];
        let mut memories = sequence_batch
            .iter()
            .map(|sequence| {
                if sequence.initial_memory.is_empty() {
                    vec![0.0; recurrent_size]
                } else {
                    sequence.initial_memory.clone()
                }
            })
            .collect::<Vec<_>>();
        while positions
            .iter()
            .zip(sequence_batch)
            .any(|(position, sequence)| *position < sequence.samples.len())
        {
            let active = positions
                .iter()
                .zip(sequence_batch)
                .enumerate()
                .filter_map(|(index, (position, sequence))| {
                    (*position < sequence.samples.len()).then_some(index)
                })
                .collect::<Vec<_>>();
            let samples = active
                .iter()
                .map(|index| sequence_batch[*index].samples[positions[*index]])
                .collect::<Vec<_>>();
            let phases = active
                .iter()
                .map(|index| {
                    expert_routing_phase(
                        sequence_batch[*index].samples[positions[*index]],
                        expert_routing,
                    )
                })
                .collect::<Vec<_>>();
            let observations = samples
                .iter()
                .flat_map(|sample| sample.observation.iter().copied())
                .collect::<Vec<_>>();
            let kind_masks = samples
                .iter()
                .flat_map(|sample| kind_mask_bias(sample))
                .collect::<Vec<_>>();
            let expert_kind_masks = samples
                .iter()
                .zip(&phases)
                .flat_map(|(sample, phase)| expert_kind_mask_bias(sample, *phase))
                .collect::<Vec<_>>();
            let target_masks = samples
                .iter()
                .flat_map(|sample| target_mask_bias(sample))
                .collect::<Vec<_>>();
            let effort_masks = samples
                .iter()
                .flat_map(|sample| effort_mask_bias(sample))
                .collect::<Vec<_>>();
            let signal_masks = samples
                .iter()
                .flat_map(|sample| signal_mask_bias(sample))
                .collect::<Vec<_>>();
            let signal_strength_masks = samples
                .iter()
                .flat_map(|sample| signal_strength_mask_bias(sample))
                .collect::<Vec<_>>();
            let amount_masks = samples
                .iter()
                .flat_map(|sample| amount_mask_bias(sample))
                .collect::<Vec<_>>();
            let memory = active
                .iter()
                .flat_map(|index| memories[*index].iter().copied())
                .collect::<Vec<_>>();
            let output = model.forward_with_memory(
                Tensor::<B, 2>::from_data(
                    TensorData::new(observations, [active.len(), OBS_DIM]),
                    device,
                ),
                Tensor::<B, 2>::from_data(
                    TensorData::new(memory, [active.len(), recurrent_size]),
                    device,
                ),
            );
            let kind_logits = output.action_kind_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(kind_masks, [active.len(), NUM_POLICY_ACTION_KINDS]),
                    device,
                );
            let expert_kind_logits = output.action_kind_expert_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(
                        expert_kind_masks,
                        [
                            active.len(),
                            NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS,
                        ],
                    ),
                    device,
                );
            let target_logits = output.target_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(target_masks, [active.len(), NUM_POLICY_TARGET_LOGITS]),
                    device,
                );
            let effort_logits = output.effort_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(effort_masks, [active.len(), NUM_POLICY_EFFORT_LOGITS]),
                    device,
                );
            let signal_logits = output.signal_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(signal_masks, [active.len(), NUM_SIGNAL_CHOICES]),
                    device,
                );
            let signal_strength_logits = output.signal_strength_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(
                        signal_strength_masks,
                        [active.len(), NUM_SIGNAL_STRENGTH_CHOICES],
                    ),
                    device,
                );
            let amount_logits = output.amount_logits
                + Tensor::<B, 2>::from_data(
                    TensorData::new(amount_masks, [active.len(), NUM_POLICY_AMOUNT_LOGITS]),
                    device,
                );
            let width = NUM_POLICY_ACTION_KINDS
                + NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS
                + NUM_POLICY_TARGET_LOGITS
                + NUM_POLICY_EFFORT_LOGITS
                + NUM_POLICY_AMOUNT_LOGITS
                + NUM_SIGNAL_CHOICES
                + NUM_SIGNAL_STRENGTH_CHOICES
                + NUM_ACTION_KIND_EXPERTS
                + recurrent_size;
            let data = Tensor::cat(
                vec![
                    burn::tensor::activation::log_softmax(kind_logits, 1),
                    burn::tensor::activation::log_softmax(expert_kind_logits, 1),
                    burn::tensor::activation::log_softmax(target_logits, 1),
                    burn::tensor::activation::log_softmax(effort_logits, 1),
                    burn::tensor::activation::log_softmax(amount_logits, 1),
                    burn::tensor::activation::log_softmax(signal_logits, 1),
                    burn::tensor::activation::log_softmax(signal_strength_logits, 1),
                    burn::tensor::activation::log_softmax(output.phase_gate_logits, 1),
                    output.next_memory,
                ],
                1,
            )
            .into_data()
            .to_vec::<f32>()
            .expect("behavior-cloning evaluation uses f32");
            for (row, ((index, sample), phase)) in
                active.into_iter().zip(samples).zip(phases).enumerate()
            {
                let start = row * width;
                let action = usize::from(sample.action);
                let hierarchical = decompose_policy_action(action)
                    .expect("validated demonstration action is in the policy catalog");
                let amount = usize::from(sample.amount);
                let signal = usize::from(sample.signal);
                let signal_strength = usize::from(sample.signal_strength);
                let expert_kind_start = start + NUM_POLICY_ACTION_KINDS;
                let target_logits_start =
                    expert_kind_start + NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS;
                let effort_logits_start = target_logits_start + NUM_POLICY_TARGET_LOGITS;
                let amount_logits_start = effort_logits_start + NUM_POLICY_EFFORT_LOGITS;
                let signal_start = amount_logits_start + NUM_POLICY_AMOUNT_LOGITS;
                let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
                let phase_start = signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES;
                let target_index = hierarchical.kind * NUM_POLICY_TARGETS + hierarchical.target;
                let effort_index = hierarchical.kind * NUM_POLICY_EFFORTS + hierarchical.effort;
                let amount_index = hierarchical.kind * NUM_AMOUNT_CHOICES + amount;
                let expert_start = expert_kind_start + phase.index() * NUM_POLICY_ACTION_KINDS;
                let conditional_and_gate_log_likelihood = data[target_logits_start + target_index]
                    + data[effort_logits_start + effort_index]
                    + data[amount_logits_start + amount_index]
                    + data[signal_start + signal]
                    + data[signal_strength_start + signal_strength]
                    + evaluation_phase_weights[phase.index()]
                        * loss_config.phase_gate_weight as f32
                        * data[phase_start + phase.index()];
                total_loss -= f64::from(conditional_and_gate_log_likelihood);
                if let Some(margin) = loss_config.action_kind_margin {
                    let teacher_logit = data[expert_start + hierarchical.kind];
                    let strongest_competitor = (0..NUM_POLICY_ACTION_KINDS)
                        .filter(|kind| *kind != hierarchical.kind)
                        .map(|kind| data[expert_start + kind])
                        .fold(f32::NEG_INFINITY, f32::max);
                    total_loss += f64::from(
                        (strongest_competitor - teacher_logit + margin as f32).max(0.0)
                            * loss_config.action_kind_margin_weight as f32,
                    );
                } else {
                    total_loss -= f64::from(data[expert_start + hierarchical.kind]);
                }
                let predicted_kind = (0..NUM_POLICY_ACTION_KINDS)
                    .max_by(|left, right| {
                        data[start + *left]
                            .total_cmp(&data[start + *right])
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let predicted_expert_kind = (0..NUM_POLICY_ACTION_KINDS)
                    .max_by(|left, right| {
                        data[expert_kind_start + phase.index() * NUM_POLICY_ACTION_KINDS + *left]
                            .total_cmp(
                                &data[expert_kind_start
                                    + phase.index() * NUM_POLICY_ACTION_KINDS
                                    + *right],
                            )
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let predicted_target = (0..NUM_POLICY_TARGETS)
                    .max_by(|left, right| {
                        data[target_logits_start + hierarchical.kind * NUM_POLICY_TARGETS + *left]
                            .total_cmp(
                                &data[target_logits_start
                                    + hierarchical.kind * NUM_POLICY_TARGETS
                                    + *right],
                            )
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let predicted_effort = (0..NUM_POLICY_EFFORTS)
                    .max_by(|left, right| {
                        data[effort_logits_start + hierarchical.kind * NUM_POLICY_EFFORTS + *left]
                            .total_cmp(
                                &data[effort_logits_start
                                    + hierarchical.kind * NUM_POLICY_EFFORTS
                                    + *right],
                            )
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let predicted_signal = (0..NUM_SIGNAL_CHOICES)
                    .max_by(|left, right| {
                        data[signal_start + *left]
                            .total_cmp(&data[signal_start + *right])
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let predicted_amount = (0..NUM_AMOUNT_CHOICES)
                    .max_by(|left, right| {
                        data[amount_logits_start + hierarchical.kind * NUM_AMOUNT_CHOICES + *left]
                            .total_cmp(
                                &data[amount_logits_start
                                    + hierarchical.kind * NUM_AMOUNT_CHOICES
                                    + *right],
                            )
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let predicted_signal_strength = (0..NUM_SIGNAL_STRENGTH_CHOICES)
                    .max_by(|left, right| {
                        data[signal_strength_start + *left]
                            .total_cmp(&data[signal_strength_start + *right])
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                let kind_correct = predicted_kind == hierarchical.kind;
                let exact = kind_correct
                    && predicted_target == hierarchical.target
                    && predicted_effort == hierarchical.effort
                    && predicted_amount == amount
                    && predicted_signal == signal
                    && predicted_signal_strength == signal_strength;
                correct += usize::from(exact);
                let family = policy_action_family(action)
                    .expect("validated demonstration action belongs to a policy family")
                    .index();
                family_samples[family] += 1;
                family_kind_correct[family] += usize::from(kind_correct);
                family_expert_kind_correct[family] +=
                    usize::from(predicted_expert_kind == hierarchical.kind);
                family_exact_correct[family] += usize::from(exact);
                let predicted_phase = (0..NUM_ACTION_KIND_EXPERTS)
                    .max_by(|left, right| {
                        data[phase_start + *left]
                            .total_cmp(&data[phase_start + *right])
                            .then_with(|| right.cmp(left))
                    })
                    .unwrap_or(0);
                phase_samples[phase.index()] += 1;
                phase_gate_correct[phase.index()] += usize::from(predicted_phase == phase.index());
                family_gate_correct[family] += usize::from(predicted_phase == phase.index());
                memories[index] = canonicalize_memory(
                    &data[phase_start + NUM_ACTION_KIND_EXPERTS..start + width],
                );
                positions[index] += 1;
                total += 1;
            }
        }
    }
    Some(SequenceEvaluation {
        loss: total_loss / total.max(1) as f64,
        exact_accuracy: correct as f64 / total.max(1) as f64,
        family_metrics: PolicyActionFamily::ALL
            .into_iter()
            .map(|family| {
                let family_index = family.index();
                let samples = family_samples[family_index];
                BehaviorCloningFamilyMetrics {
                    family: family.name().into(),
                    samples,
                    action_kind_accuracy: (samples > 0)
                        .then_some(family_kind_correct[family_index] as f64 / samples as f64),
                    routed_expert_action_kind_accuracy: (samples > 0).then_some(
                        family_expert_kind_correct[family_index] as f64 / samples as f64,
                    ),
                    phase_gate_accuracy: (samples > 0)
                        .then_some(family_gate_correct[family_index] as f64 / samples as f64),
                    exact_accuracy: (samples > 0)
                        .then_some(family_exact_correct[family_index] as f64 / samples as f64),
                }
            })
            .collect(),
        phase_metrics: SupervisionPhase::ALL
            .into_iter()
            .map(|phase| {
                let samples = phase_samples[phase.index()];
                BehaviorCloningPhaseMetrics {
                    phase: phase.name().into(),
                    samples,
                    gate_accuracy: (samples > 0)
                        .then_some(phase_gate_correct[phase.index()] as f64 / samples as f64),
                }
            })
            .collect(),
    })
}

fn quantize_straight_through<B: Backend>(memory: Tensor<B, 2>) -> Tensor<B, 2> {
    let quantized = (memory.clone().detach().clamp(-1.0, 1.0) * f32::from(i16::MAX)).round()
        / f32::from(i16::MAX);
    memory.clone() + (quantized - memory).detach()
}

struct SupervisedLossWeights<'a> {
    actions: &'a [f32],
    phases: &'a [f32; NUM_ACTION_KIND_EXPERTS],
    phase_gate: f64,
}

struct SupervisedStep<'a, B: Backend> {
    loss_weights: SupervisedLossWeights<'a>,
    expert_routing: ExpertRoutingStrategy,
    action_kind_expert_only: Option<SupervisionPhase>,
    foraging_adapter_only: bool,
    context_adapter_only: Option<SupervisionPhase>,
    context_slot_adapter_only: Option<SupervisionPhase>,
    exploration_guard_readiness_only: bool,
    action_kind_margin: Option<f64>,
    action_kind_margin_loss_weight: f64,
    target_query_head_only: bool,
    target_residual_only: bool,
    effort_head_only: bool,
    frozen_template: Option<&'a PolicyValueNet<B>>,
    learning_rate: f64,
    device: &'a B::Device,
}

fn train_chunk_batch<B: AutodiffBackend>(
    mut model: PolicyValueNet<B>,
    optimizer: &mut impl Optimizer<PolicyValueNet<B>, B>,
    chunks: &[SequenceChunk<'_>],
    step: SupervisedStep<'_, B>,
) -> PolicyValueNet<B> {
    let steps = chunks[0].samples.len();
    debug_assert!(chunks.iter().all(|chunk| chunk.samples.len() == steps));
    let recurrent_size = model.recurrent_size();
    let mut memory = Tensor::<B, 2>::from_data(
        TensorData::new(
            chunks
                .iter()
                .flat_map(|chunk| chunk.initial_memory.iter().copied())
                .collect::<Vec<_>>(),
            [chunks.len(), recurrent_size],
        ),
        step.device,
    );
    let mut loss = None;
    for time in 0..steps {
        let samples = chunks
            .iter()
            .map(|chunk| chunk.samples[time])
            .collect::<Vec<_>>();
        let observations = samples
            .iter()
            .flat_map(|sample| sample.observation.iter().copied())
            .collect::<Vec<_>>();
        let expert_kind_masks = samples
            .iter()
            .flat_map(|sample| {
                expert_kind_mask_bias(sample, expert_routing_phase(sample, step.expert_routing))
            })
            .collect::<Vec<_>>();
        let target_masks = samples
            .iter()
            .flat_map(|sample| target_mask_bias(sample))
            .collect::<Vec<_>>();
        let effort_masks = samples
            .iter()
            .flat_map(|sample| effort_mask_bias(sample))
            .collect::<Vec<_>>();
        let signal_masks = samples
            .iter()
            .flat_map(|sample| signal_mask_bias(sample))
            .collect::<Vec<_>>();
        let signal_strength_masks = samples
            .iter()
            .flat_map(|sample| signal_strength_mask_bias(sample))
            .collect::<Vec<_>>();
        let amount_masks = samples
            .iter()
            .flat_map(|sample| amount_mask_bias(sample))
            .collect::<Vec<_>>();
        let hierarchical = samples
            .iter()
            .map(|sample| {
                decompose_policy_action(usize::from(sample.action))
                    .expect("validated demonstration action is in the policy catalog")
            })
            .collect::<Vec<_>>();
        let expert_kinds = hierarchical
            .iter()
            .zip(&samples)
            .map(|(choice, sample)| {
                (expert_routing_phase(sample, step.expert_routing).index()
                    * NUM_POLICY_ACTION_KINDS
                    + choice.kind) as i32
            })
            .collect::<Vec<_>>();
        let phases = samples
            .iter()
            .map(|sample| expert_routing_phase(sample, step.expert_routing).index() as i32)
            .collect::<Vec<_>>();
        let targets = hierarchical
            .iter()
            .map(|choice| (choice.kind * NUM_POLICY_TARGETS + choice.target) as i32)
            .collect::<Vec<_>>();
        let efforts = hierarchical
            .iter()
            .map(|choice| (choice.kind * NUM_POLICY_EFFORTS + choice.effort) as i32)
            .collect::<Vec<_>>();
        let sample_weights = samples
            .iter()
            .map(|sample| step.loss_weights.actions[usize::from(sample.action)])
            .collect::<Vec<_>>();
        let phase_sample_weights = samples
            .iter()
            .map(|sample| {
                step.loss_weights.phases[expert_routing_phase(sample, step.expert_routing).index()]
            })
            .collect::<Vec<_>>();
        let signals = samples
            .iter()
            .map(|sample| i32::from(sample.signal))
            .collect::<Vec<_>>();
        let signal_strengths = samples
            .iter()
            .map(|sample| i32::from(sample.signal_strength))
            .collect::<Vec<_>>();
        let amounts = samples
            .iter()
            .zip(&hierarchical)
            .map(|(sample, choice)| {
                (choice.kind * NUM_AMOUNT_CHOICES + usize::from(sample.amount)) as i32
            })
            .collect::<Vec<_>>();
        let output = model.forward_with_memory(
            Tensor::<B, 2>::from_data(
                TensorData::new(observations, [chunks.len(), OBS_DIM]),
                step.device,
            ),
            memory,
        );
        let expert_kind_logits = output.action_kind_expert_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(
                    expert_kind_masks.clone(),
                    [
                        chunks.len(),
                        NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS,
                    ],
                ),
                step.device,
            );
        let target_logits = output.target_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(target_masks, [chunks.len(), NUM_POLICY_TARGET_LOGITS]),
                step.device,
            );
        let effort_logits = output.effort_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(effort_masks, [chunks.len(), NUM_POLICY_EFFORT_LOGITS]),
                step.device,
            );
        let signal_logits = output.signal_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(signal_masks, [chunks.len(), NUM_SIGNAL_CHOICES]),
                step.device,
            );
        let signal_strength_logits = output.signal_strength_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(
                    signal_strength_masks,
                    [chunks.len(), NUM_SIGNAL_STRENGTH_CHOICES],
                ),
                step.device,
            );
        let amount_logits = output.amount_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(amount_masks, [chunks.len(), NUM_POLICY_AMOUNT_LOGITS]),
                step.device,
            );
        let expert_kind_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(expert_kinds.clone(), [chunks.len()]),
            step.device,
        );
        let phase_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(phases, [chunks.len()]), step.device);
        let target_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(targets, [chunks.len()]), step.device);
        let effort_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(efforts, [chunks.len()]), step.device);
        let signal_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(signals, [chunks.len()]), step.device);
        let signal_strength_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(signal_strengths, [chunks.len()]),
            step.device,
        );
        let amount_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(amounts, [chunks.len()]), step.device);
        let sample_weight_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(sample_weights, [chunks.len(), 1]),
            step.device,
        );
        let phase_weight_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(phase_sample_weights, [chunks.len(), 1]),
            step.device,
        );
        let action_kind_margin_loss = step.action_kind_margin.map(|margin| {
            let mut competitor_masks = expert_kind_masks;
            for (row, teacher) in competitor_masks
                .chunks_mut(NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS)
                .zip(expert_kinds.iter().copied())
            {
                row[teacher as usize] = -1.0e9;
            }
            let strongest_competitor = (expert_kind_logits.clone()
                + Tensor::<B, 2>::from_data(
                    TensorData::new(
                        competitor_masks,
                        [
                            chunks.len(),
                            NUM_ACTION_KIND_EXPERTS * NUM_POLICY_ACTION_KINDS,
                        ],
                    ),
                    step.device,
                ))
            .max_dim(1);
            let teacher_logits = expert_kind_logits
                .clone()
                .gather(1, expert_kind_tensor.clone().unsqueeze_dim(1));
            (burn::tensor::activation::relu(strongest_competitor - teacher_logits + margin as f32)
                * sample_weight_tensor.clone())
            .mean()
                * step.action_kind_margin_loss_weight as f32
        });
        let conditional_log_likelihood = burn::tensor::activation::log_softmax(target_logits, 1)
            .gather(1, target_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(effort_logits, 1)
                .gather(1, effort_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(amount_logits, 1)
                .gather(1, amount_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(signal_logits, 1)
                .gather(1, signal_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(signal_strength_logits, 1)
                .gather(1, signal_strength_tensor.unsqueeze_dim(1));
        let action_log_likelihood = if step.action_kind_margin.is_some() {
            conditional_log_likelihood
        } else {
            conditional_log_likelihood
                + burn::tensor::activation::log_softmax(expert_kind_logits, 1)
                    .gather(1, expert_kind_tensor.unsqueeze_dim(1))
        };
        let phase_log_likelihood =
            burn::tensor::activation::log_softmax(output.phase_gate_logits, 1)
                .gather(1, phase_tensor.unsqueeze_dim(1));
        let mut step_loss = (action_log_likelihood * sample_weight_tensor
            + phase_log_likelihood * phase_weight_tensor * step.loss_weights.phase_gate as f32)
            .mean()
            .neg();
        if let Some(action_kind_margin_loss) = action_kind_margin_loss {
            step_loss = step_loss + action_kind_margin_loss;
        }
        loss = Some(match loss {
            Some(loss) => loss + step_loss,
            None => step_loss,
        });
        memory = quantize_straight_through(output.next_memory);
    }
    let loss = loss.expect("a sequence chunk always contains a sample") / steps as f32;
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    model = optimizer.step(step.learning_rate, model, gradients);
    if step.exploration_guard_readiness_only {
        model = step
            .frozen_template
            .expect("exploration-Guard-readiness-only adaptation has a frozen parent template")
            .clone()
            .with_exploration_guard_readiness_from(model);
    } else if let Some(context) = step.context_slot_adapter_only {
        let context = match context {
            SupervisionPhase::Combat => ObservationExpertContext::Interaction,
            SupervisionPhase::Exploration => ObservationExpertContext::Exploration,
            SupervisionPhase::Feeding => {
                unreachable!("foraging uses its dedicated adapter-only mode")
            }
        };
        model = step
            .frozen_template
            .expect("context-slot-adapter-only adaptation has a frozen parent template")
            .clone()
            .with_context_slot_adapter_from(context, model);
    } else if let Some(context) = step.context_adapter_only {
        let context = match context {
            SupervisionPhase::Combat => ObservationExpertContext::Interaction,
            SupervisionPhase::Exploration => ObservationExpertContext::Exploration,
            SupervisionPhase::Feeding => {
                unreachable!("foraging uses its dedicated adapter-only mode")
            }
        };
        model = step
            .frozen_template
            .expect("context-adapter-only adaptation has a frozen parent template")
            .clone()
            .with_context_adapter_from(context, model);
    } else if step.effort_head_only {
        model = step
            .frozen_template
            .expect("effort-head-only adaptation has a frozen parent template")
            .clone()
            .with_effort_head_from(model);
    } else if step.target_residual_only {
        model = step
            .frozen_template
            .expect("target-residual-only adaptation has a frozen parent template")
            .clone()
            .with_target_residual_from(model);
    } else if step.target_query_head_only {
        model = step
            .frozen_template
            .expect("target-head-only adaptation has a frozen parent template")
            .clone()
            .with_target_query_head_from(model);
    } else if step.foraging_adapter_only {
        model = step
            .frozen_template
            .expect("adapter-only adaptation has a frozen parent template")
            .clone()
            .with_foraging_adapter_from(model);
    } else if let Some(expert) = step.action_kind_expert_only {
        let context = match expert {
            SupervisionPhase::Feeding => ObservationExpertContext::Foraging,
            SupervisionPhase::Combat => ObservationExpertContext::Interaction,
            SupervisionPhase::Exploration => ObservationExpertContext::Exploration,
        };
        model = step
            .frozen_template
            .expect("expert-only adaptation has a frozen parent template")
            .clone()
            .with_action_kind_expert_from(model, context);
    }
    model
}

pub fn behavior_clone<B: AutodiffBackend>(
    datasets: &[LoadedDemonstrations],
    model_config: &ModelConfig,
    config: &BehaviorCloningConfig,
    device: B::Device,
) -> Result<(PolicyValueNet<B>, BehaviorCloningMetrics), String>
where
    f32: From<B::FloatElem>,
{
    behavior_clone_from_model(datasets, model_config, config, None, device)
}

/// Train a behavior-cloning stage from either a fresh model or a verified
/// parent model. Callers must put the parent's artifact hash in `config` when
/// `initial_model` is present, and attach its verified cumulative seed ledger.
/// Use `InheritedSeedLedger::from_artifact` (or an explicit historical audit).
pub fn behavior_clone_from_model<B: AutodiffBackend>(
    datasets: &[LoadedDemonstrations],
    model_config: &ModelConfig,
    config: &BehaviorCloningConfig,
    initial_model: Option<PolicyValueNet<B>>,
    device: B::Device,
) -> Result<(PolicyValueNet<B>, BehaviorCloningMetrics), String>
where
    f32: From<B::FloatElem>,
{
    config.validate()?;
    if initial_model.is_some() != config.initial_artifact_sha256.is_some() {
        return Err(
            "initial model and initial behavior-cloning artifact hash must be supplied together"
                .into(),
        );
    }
    if datasets.is_empty() {
        return Err("behavior cloning requires at least one dataset".into());
    }
    if datasets
        .iter()
        .flat_map(|dataset| &dataset.payload.samples)
        .any(|sample| {
            !sample.initial_policy_memory.is_empty()
                && sample.initial_policy_memory.len() != model_config.recurrent_size
        })
    {
        return Err("demonstration initial policy memory does not match the model".into());
    }
    let available_samples = datasets
        .iter()
        .map(|dataset| dataset.payload.samples.len())
        .sum();
    let exact_round_trip_samples = datasets
        .iter()
        .flat_map(|dataset| &dataset.payload.samples)
        .filter(|sample| sample.exact_round_trip)
        .count();
    let partitions = partition_datasets(datasets, config)?;
    validate_gate_label_consistency(&partitions, config.expert_routing)?;
    let training_sequences_by_dataset = partitions
        .iter()
        .map(|partition| build_sequences(&partition.training))
        .collect::<Vec<_>>();
    let validation_sequences_by_dataset = partitions
        .iter()
        .map(|partition| build_sequences(&partition.validation))
        .collect::<Vec<_>>();
    let training_sequences = training_sequences_by_dataset
        .iter()
        .flat_map(|sequences| sequences.iter().cloned())
        .collect::<Vec<_>>();
    let validation_sequences = validation_sequences_by_dataset
        .iter()
        .flat_map(|sequences| sequences.iter().cloned())
        .collect::<Vec<_>>();
    let training_samples = partitions
        .iter()
        .map(|partition| partition.training.len())
        .sum::<usize>();
    let validation_samples = partitions
        .iter()
        .map(|partition| partition.validation.len())
        .sum::<usize>();
    let eligible_samples = training_samples + validation_samples;
    let per_dataset = samples_per_dataset_per_epoch(
        &partitions,
        config.dataset_sampling,
        &config.dataset_sampling_weights,
        config.epoch_sample_budget,
    )?;
    let samples_per_epoch = per_dataset.iter().sum::<usize>();
    let partition_metrics = partitions
        .iter()
        .zip(&per_dataset)
        .enumerate()
        .map(
            |(index, (partition, samples_per_epoch))| BehaviorCloningDatasetPartition {
                manifest_sha256: partition.manifest_sha256.clone(),
                eligible_phase_samples: phase_histogram(
                    partition
                        .training
                        .iter()
                        .chain(&partition.validation)
                        .copied(),
                    config.expert_routing,
                ),
                training_phase_samples: phase_histogram(
                    partition.training.iter().copied(),
                    config.expert_routing,
                ),
                validation_phase_samples: phase_histogram(
                    partition.validation.iter().copied(),
                    config.expert_routing,
                ),
                eligible_samples: partition.training.len() + partition.validation.len(),
                training_samples: partition.training.len(),
                validation_samples: partition.validation.len(),
                training_trajectories: training_sequences_by_dataset[index].len(),
                validation_trajectories: validation_sequences_by_dataset[index].len(),
                samples_per_epoch: *samples_per_epoch,
                held_out_seeds: partition.held_out_seeds.clone(),
                training_seeds: partition
                    .training
                    .iter()
                    .map(|sample| sample.source_seed)
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                eligible_action_histogram: action_histogram(
                    partition
                        .training
                        .iter()
                        .chain(&partition.validation)
                        .copied(),
                ),
                training_action_histogram: action_histogram(partition.training.iter().copied()),
                validation_action_histogram: action_histogram(partition.validation.iter().copied()),
            },
        )
        .collect::<Vec<_>>();

    B::seed(&device, config.seed);
    let mut model = initial_model.unwrap_or_else(|| {
        PolicyValueNetConfig {
            hidden1: model_config.hidden1,
            hidden2: model_config.hidden2,
            recurrent_size: model_config.recurrent_size,
        }
        .init::<B>(&device)
    });
    // Fork once before creating any autodiff graph. Cloning the active model
    // inside an update would duplicate parameter IDs in that graph and can
    // consume the selected expert's gradients. The fork is an independent,
    // immutable source for every non-selected parameter instead.
    let frozen_template = config
        .action_kind_expert_only
        .is_some()
        .then_some(())
        .or(config.foraging_adapter_only.then_some(()))
        .or(config.context_adapter_only.map(|_| ()))
        .or(config.context_slot_adapter_only.map(|_| ()))
        .or(config.exploration_guard_readiness_only.then_some(()))
        .or(config.target_query_head_only.then_some(()))
        .or(config.target_residual_only.then_some(()))
        .or(config.effort_head_only.then_some(()))
        .map(|()| model.clone().fork(&device));
    let mut optimizer = AdamWConfig::new()
        .init()
        .with_grad_clipping(GradientClipping::Norm(1.0));
    let initial_training = evaluate_sequences(
        &model,
        &training_sequences,
        config.minibatch_size,
        config.expert_routing,
        EvaluationLossConfig {
            phase_gate_weight: config.phase_gate_loss_weight,
            action_kind_margin: config.action_kind_margin,
            action_kind_margin_weight: config.action_kind_margin_loss_weight,
        },
        &device,
    )
    .expect("a nonempty training partition has a trajectory");
    let initial_validation = evaluate_sequences(
        &model,
        &validation_sequences,
        config.minibatch_size,
        config.expert_routing,
        EvaluationLossConfig {
            phase_gate_weight: config.phase_gate_loss_weight,
            action_kind_margin: config.action_kind_margin,
            action_kind_margin_weight: config.action_kind_margin_loss_weight,
        },
        &device,
    );
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed ^ 0x4245_4841_5649_4f52);
    let mut optimizer_steps = 0usize;
    let mut completed_epochs = 0usize;
    let mut sample_presentations = 0usize;
    let mut action_family_presentations = vec![0usize; PolicyActionFamily::COUNT];
    let mut action_family_weighted_loss_mass = vec![0.0_f64; PolicyActionFamily::COUNT];
    let mut supervision_phase_presentations = vec![0usize; NUM_ACTION_KIND_EXPERTS];
    let mut supervision_phase_weighted_gate_loss_mass = vec![0.0_f64; NUM_ACTION_KIND_EXPERTS];
    'epochs: for _ in 0..config.epochs {
        let chunks_by_dataset = training_sequences_by_dataset
            .iter()
            .map(|sequences| {
                prepare_sequence_chunks(
                    &model,
                    sequences,
                    config.recurrent_unroll_steps,
                    config.minibatch_size,
                    &device,
                )
            })
            .collect::<Vec<_>>();
        let epoch = sample_chunk_epoch(&chunks_by_dataset, &per_dataset, &mut rng);
        let action_weights = action_weights_for_samples(
            epoch.iter().flat_map(|chunk| chunk.samples.iter().copied()),
            config.action_balancing,
            config.action_balance_exponent,
            config.action_balance_max_ratio,
        );
        let phase_weights = phase_weights_for_chunks(&epoch, config.expert_routing);
        let mut by_length = BTreeMap::<usize, Vec<SequenceChunk<'_>>>::new();
        for chunk in epoch {
            by_length
                .entry(chunk.samples.len())
                .or_default()
                .push(chunk);
        }
        let group_shapes = by_length
            .iter()
            .map(|(length, chunks)| (*length, chunks.len()))
            .collect::<Vec<_>>();
        let exact_batch_counts = config
            .optimizer_steps_per_epoch
            .map(|steps| exact_recurrent_batch_counts(&group_shapes, config.minibatch_size, steps))
            .transpose()?;
        let mut chunk_batches = Vec::new();
        for (group_index, (length, chunks)) in by_length.iter_mut().enumerate() {
            chunks.shuffle(&mut rng);
            if let Some(batch_counts) = &exact_batch_counts {
                let batches = batch_counts[group_index];
                let base = chunks.len() / batches;
                let remainder = chunks.len() % batches;
                let mut start = 0usize;
                for batch_index in 0..batches {
                    let size = base + usize::from(batch_index < remainder);
                    chunk_batches.push(chunks[start..start + size].to_vec());
                    start += size;
                }
                debug_assert_eq!(start, chunks.len());
            } else {
                let chunk_batch_size = (config.minibatch_size / *length).max(1);
                for batch in chunks.chunks(chunk_batch_size) {
                    chunk_batches.push(batch.to_vec());
                }
            }
        }
        debug_assert_eq!(
            config
                .optimizer_steps_per_epoch
                .unwrap_or(chunk_batches.len()),
            chunk_batches.len()
        );
        chunk_batches.shuffle(&mut rng);
        let batches_in_epoch = chunk_batches.len();
        let remaining_steps = config
            .optimizer_step_budget
            .map_or(batches_in_epoch, |budget| budget - optimizer_steps);
        let batches_to_run = batches_in_epoch.min(remaining_steps);
        for batch in chunk_batches.into_iter().take(batches_to_run) {
            for sample in batch.iter().flat_map(|chunk| chunk.samples.iter().copied()) {
                let family = policy_action_family(usize::from(sample.action))
                    .expect("validated demonstration action belongs to a policy family")
                    .index();
                sample_presentations += 1;
                action_family_presentations[family] += 1;
                action_family_weighted_loss_mass[family] +=
                    f64::from(action_weights[usize::from(sample.action)]);
                let phase = expert_routing_phase(sample, config.expert_routing).index();
                supervision_phase_presentations[phase] += 1;
                supervision_phase_weighted_gate_loss_mass[phase] += f64::from(phase_weights[phase]);
            }
            let learning_rate = if config.optimizer_step_budget == Some(optimizer_steps + 1) {
                config.learning_rate * config.terminal_optimizer_step_scale
            } else {
                config.learning_rate
            };
            model = train_chunk_batch(
                model,
                &mut optimizer,
                &batch,
                SupervisedStep {
                    loss_weights: SupervisedLossWeights {
                        actions: &action_weights,
                        phases: &phase_weights,
                        phase_gate: config.phase_gate_loss_weight,
                    },
                    expert_routing: config.expert_routing,
                    action_kind_expert_only: config.action_kind_expert_only,
                    foraging_adapter_only: config.foraging_adapter_only,
                    context_adapter_only: config.context_adapter_only,
                    context_slot_adapter_only: config.context_slot_adapter_only,
                    exploration_guard_readiness_only: config.exploration_guard_readiness_only,
                    action_kind_margin: config.action_kind_margin,
                    action_kind_margin_loss_weight: config.action_kind_margin_loss_weight,
                    target_query_head_only: config.target_query_head_only,
                    target_residual_only: config.target_residual_only,
                    effort_head_only: config.effort_head_only,
                    frozen_template: frozen_template.as_ref(),
                    learning_rate,
                    device: &device,
                },
            );
            optimizer_steps += 1;
        }
        if batches_to_run == batches_in_epoch {
            completed_epochs += 1;
        }
        if config
            .optimizer_step_budget
            .is_some_and(|budget| optimizer_steps == budget)
        {
            break 'epochs;
        }
    }
    debug_assert_eq!(
        config.optimizer_step_budget.unwrap_or(optimizer_steps),
        optimizer_steps
    );
    let final_training = evaluate_sequences(
        &model,
        &training_sequences,
        config.minibatch_size,
        config.expert_routing,
        EvaluationLossConfig {
            phase_gate_weight: config.phase_gate_loss_weight,
            action_kind_margin: config.action_kind_margin,
            action_kind_margin_weight: config.action_kind_margin_loss_weight,
        },
        &device,
    )
    .expect("a nonempty training partition has a trajectory");
    let final_validation = evaluate_sequences(
        &model,
        &validation_sequences,
        config.minibatch_size,
        config.expert_routing,
        EvaluationLossConfig {
            phase_gate_weight: config.phase_gate_loss_weight,
            action_kind_margin: config.action_kind_margin,
            action_kind_margin_weight: config.action_kind_margin_loss_weight,
        },
        &device,
    );
    Ok((
        model,
        BehaviorCloningMetrics {
            datasets: datasets.len(),
            available_samples,
            eligible_samples,
            training_samples,
            validation_samples,
            samples_per_epoch,
            sample_presentations,
            action_family_presentations,
            action_family_weighted_loss_mass,
            supervision_phase_presentations,
            supervision_phase_weighted_gate_loss_mass,
            recurrent_unroll_steps: config.recurrent_unroll_steps,
            training_trajectories: training_sequences.len(),
            validation_trajectories: validation_sequences.len(),
            exact_round_trip_samples,
            completed_epochs,
            epochs: config.epochs,
            optimizer_steps,
            initial_training_loss: initial_training.loss,
            final_training_loss: final_training.loss,
            initial_training_accuracy: initial_training.exact_accuracy,
            final_training_accuracy: final_training.exact_accuracy,
            initial_validation_loss: initial_validation.as_ref().map(|metrics| metrics.loss),
            final_validation_loss: final_validation.as_ref().map(|metrics| metrics.loss),
            initial_validation_accuracy: initial_validation
                .as_ref()
                .map(|metrics| metrics.exact_accuracy),
            final_validation_accuracy: final_validation
                .as_ref()
                .map(|metrics| metrics.exact_accuracy),
            initial_training_family_metrics: initial_training.family_metrics,
            final_training_family_metrics: final_training.family_metrics,
            initial_validation_family_metrics: initial_validation
                .as_ref()
                .map(|metrics| metrics.family_metrics.clone())
                .unwrap_or_else(empty_family_metrics),
            final_validation_family_metrics: final_validation
                .as_ref()
                .map(|metrics| metrics.family_metrics.clone())
                .unwrap_or_else(empty_family_metrics),
            initial_training_phase_metrics: initial_training.phase_metrics,
            final_training_phase_metrics: final_training.phase_metrics,
            initial_validation_phase_metrics: initial_validation
                .as_ref()
                .map(|metrics| metrics.phase_metrics.clone())
                .unwrap_or_else(empty_phase_metrics),
            final_validation_phase_metrics: final_validation
                .as_ref()
                .map(|metrics| metrics.phase_metrics.clone())
                .unwrap_or_else(empty_phase_metrics),
            partitions: partition_metrics,
        },
    ))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn behavior_clone_artifact_sha256(directory: &Path) -> Result<String, String> {
    sha256_file(&directory.join("behavior-cloning.json"))
}

/// Verify a configured warm start and return the recorder base path (`model`,
/// without the `.mpk` suffix) expected by Burn.
pub fn verify_behavior_clone_artifact(
    directory: &Path,
    expected_artifact_sha256: &str,
    expected_model: &ModelConfig,
) -> Result<PathBuf, String> {
    verify_behavior_clone_artifact_with_schema(directory, expected_artifact_sha256, expected_model)
        .map(|(path, _)| path)
}

/// Verify a configured warm start and also return its schema so the sole
/// behavior-cloning lineage loader can perform one-way model migrations.
pub fn verify_behavior_clone_artifact_with_schema(
    directory: &Path,
    expected_artifact_sha256: &str,
    expected_model: &ModelConfig,
) -> Result<(PathBuf, u32), String> {
    let artifact =
        verify_behavior_clone_metadata(directory, expected_artifact_sha256, expected_model)?;
    Ok((directory.join("model"), artifact.schema_version))
}

pub fn verify_behavior_clone_metadata(
    directory: &Path,
    expected_artifact_sha256: &str,
    expected_model: &ModelConfig,
) -> Result<BehaviorCloningArtifact, String> {
    let metadata_path = directory.join("behavior-cloning.json");
    let metadata = crate::artifact_io::read_bounded(&metadata_path, 1024 * 1024)?;
    if format!("{:x}", Sha256::digest(&metadata)) != expected_artifact_sha256 {
        return Err("behavior-cloning artifact SHA-256 mismatch".into());
    }
    let artifact: BehaviorCloningArtifact = serde_json::from_slice(&metadata)
        .map_err(|error| format!("failed to decode {}: {error}", metadata_path.display()))?;
    if !(MIN_SUPPORTED_BEHAVIOR_CLONING_SCHEMA_VERSION..=BEHAVIOR_CLONING_SCHEMA_VERSION)
        .contains(&artifact.schema_version)
        || artifact.schema_version == 30
        || (artifact.schema_version < 37 && artifact.config.target_residual_only)
        || artifact.model != *expected_model
        || artifact.model_file != "model.mpk"
    {
        return Err("behavior-cloning artifact schema or model architecture mismatch".into());
    }
    if artifact.execution.as_ref().is_some_and(|identity| {
        *identity != crate::policy_artifact::PolicyExecutionIdentity::current()
    }) || (artifact.schema_version >= 35
        && (artifact.execution.is_none()
            || artifact.code_revision.as_deref().is_none_or(str::is_empty)))
    {
        return Err("behavior-cloning policy execution identity is missing or incompatible".into());
    }
    let model_file = directory.join(&artifact.model_file);
    if sha256_file(&model_file)? != artifact.model_sha256 {
        return Err("behavior-cloned model SHA-256 mismatch".into());
    }
    if artifact.schema_version >= 36 {
        artifact.config.validate()?;
        let expected =
            seed_ledger::ledger_from_metrics(&artifact.config, &artifact.metrics.partitions)?;
        if artifact.seed_ledger.as_ref() != Some(&expected)
            || artifact.metrics.partitions.is_empty()
            || artifact.datasets.len() != artifact.metrics.partitions.len()
            || artifact
                .datasets
                .iter()
                .zip(&artifact.metrics.partitions)
                .any(|(dataset, partition)| dataset.manifest_sha256 != partition.manifest_sha256)
        {
            return Err(
                "behavior-cloning cumulative seed ledger is missing or inconsistent".into(),
            );
        }
    }
    Ok(artifact)
}

pub fn publish_behavior_clone<B: AutodiffBackend>(
    output: &Path,
    model: &PolicyValueNet<B>,
    model_config: &ModelConfig,
    config: &BehaviorCloningConfig,
    datasets: &[LoadedDemonstrations],
    metrics: &BehaviorCloningMetrics,
) -> Result<PathBuf, String> {
    config.validate()?;
    let partitions = partition_datasets(datasets, config)?;
    if partitions.is_empty()
        || partitions.len() != metrics.partitions.len()
        || partitions
            .iter()
            .zip(&metrics.partitions)
            .any(|(actual, reported)| {
                actual.manifest_sha256 != reported.manifest_sha256
                    || actual.training.len() != reported.training_samples
                    || actual.validation.len() != reported.validation_samples
                    || actual.held_out_seeds != reported.held_out_seeds
                    || actual
                        .training
                        .iter()
                        .map(|sample| sample.source_seed)
                        .collect::<std::collections::BTreeSet<_>>()
                        != reported.training_seeds.iter().copied().collect()
            })
    {
        return Err("published seed partitions do not match the training datasets".into());
    }
    let seed_ledger = seed_ledger::ledger_from_metrics(config, &metrics.partitions)?;
    if output.exists() {
        return Err(format!(
            "refusing to replace behavior-cloning artifact {}",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "behavior-cloning output needs a UTF-8 name".to_string())?;
    let nonce = CLONING_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&staging)
        .map_err(|error| format!("failed to create {}: {error}", staging.display()))?;
    let result = (|| {
        model
            .clone()
            .save_file(staging.join("model"), &CompactRecorder::new())
            .map_err(|error| format!("failed to save behavior-cloned model: {error}"))?;
        let model_file = staging.join("model.mpk");
        let artifact = BehaviorCloningArtifact {
            seed_ledger: Some(seed_ledger),
            execution: Some(crate::policy_artifact::PolicyExecutionIdentity::current()),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_owned),
            schema_version: BEHAVIOR_CLONING_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            training_backend: training_backend_id::<B>().into(),
            config: config.clone(),
            model: model_config.clone(),
            datasets: datasets
                .iter()
                .map(|dataset| BehaviorCloningDatasetIdentity {
                    manifest_sha256: dataset.manifest_sha256.clone(),
                    payload_sha256: dataset.manifest.payload_sha256.clone(),
                    teacher: dataset.manifest.teacher.clone(),
                    samples: dataset.manifest.samples,
                    exact_round_trip_samples: dataset.manifest.exact_round_trip_samples,
                })
                .collect(),
            metrics: metrics.clone(),
            model_file: "model.mpk".into(),
            model_sha256: sha256_file(&model_file)?,
        };
        let mut metadata = serde_json::to_vec_pretty(&artifact)
            .map_err(|error| format!("failed to encode behavior-cloning metadata: {error}"))?;
        metadata.push(b'\n');
        if metadata.len() > 1024 * 1024 {
            return Err("behavior-cloning metadata exceeds 1 MiB".into());
        }
        let metadata_path = staging.join("behavior-cloning.json");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&metadata_path)
            .map_err(|error| format!("failed to create {}: {error}", metadata_path.display()))?;
        file.write_all(&metadata)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", metadata_path.display()))?;
        File::open(&model_file)
            .and_then(|file| file.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", model_file.display()))?;
        File::open(&staging)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", staging.display()))?;
        fs::rename(&staging, output)
            .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result.map(|()| output.to_path_buf())
}

pub fn load_dataset_directories(paths: &[PathBuf]) -> Result<Vec<LoadedDemonstrations>, String> {
    paths.iter().map(|path| load_demonstrations(path)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_matrix::MaintainedMindProfile;
    use crate::demonstration::{generate_demonstrations, DemonstrationOptions};
    use crate::observation::{
        CURRENT_TILE_DIFFUSE_ENERGY_FEATURE, CURRENT_TILE_LOOSE_ENERGY_FEATURE,
        CURRENT_TILE_PLANT_CAPACITY_FEATURE, CURRENT_TILE_PLANT_ENERGY_FEATURE, HEADER_FEATURES,
        SLOT_DIFFUSE_ENERGY_FEATURE, SLOT_FEATURES, SLOT_LOOSE_ENERGY_FEATURE,
        SLOT_NEIGHBOR_ACTIVITY_FEATURE, SLOT_NEIGHBOR_PRESENT_FEATURE, SLOT_PLANT_ENERGY_FEATURE,
    };
    use burn::backend::{Autodiff, NdArray};
    use std::collections::HashSet;

    type TestBackend = Autodiff<NdArray<f32>>;

    fn small_training_config() -> crate::config::TrainingConfig {
        let mut training = crate::config::TrainingConfig::default();
        training.env.world_size = 4;
        training.env.cells_per_team = 1;
        training.env.max_episode_len = 1_024;
        training.env.victory.sim_time_limit_quanta = 16_384;
        training.env.num_scattered_energy = 2;
        training.env.num_plants = 1;
        training
    }

    fn demonstration_samples(count: usize) -> Vec<DemonstrationSample> {
        let training = small_training_config();
        let (_, payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![77],
                max_samples: count,
            },
        )
        .unwrap();
        payload.samples
    }

    #[test]
    fn recurrent_trajectory_keys_include_the_source_seed() {
        let mut samples = demonstration_samples(2);
        assert_eq!(samples.len(), 2);
        samples[0].source_seed = 10;
        samples[0].source_cell = 5;
        samples[1].source_seed = 11;
        samples[1].source_cell = 5;
        let references = samples.iter().collect::<Vec<_>>();

        let sequences = build_sequences(&references);

        assert_eq!(sequences.len(), 2);
        assert!(sequences.iter().all(|sequence| sequence.samples.len() == 1));
    }

    #[test]
    fn recurrent_chunks_are_bounded_and_start_from_canonical_memory() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let mut samples = demonstration_samples(3);
        assert_eq!(samples.len(), 3);
        for sample in &mut samples {
            sample.source_seed = 10;
            sample.source_cell = 5;
        }
        let references = samples.iter().collect::<Vec<_>>();
        let sequences = build_sequences(&references);
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 123);
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);

        let chunks = prepare_sequence_chunks(&model, &sequences, 2, 4, &device);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].samples.len(), 2);
        assert_eq!(chunks[1].samples.len(), 1);
        assert_eq!(chunks[0].initial_memory, vec![0.0; 8]);
        assert!(chunks
            .iter()
            .flat_map(|chunk| &chunk.samples)
            .all(|sample| {
                expert_routing_phase(sample, ExpertRoutingStrategy::LocalActionFamily)
                    == SupervisionPhase::Feeding
            }));
        assert!(chunks[1].initial_memory.iter().any(|value| *value != 0.0));
        assert_eq!(
            chunks[1].initial_memory,
            canonicalize_memory(&chunks[1].initial_memory)
        );
    }

    #[test]
    fn isolated_correction_uses_its_exact_private_memory() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let mut sample = demonstration_samples(1).remove(0);
        sample.initial_policy_memory = vec![0.25, -0.5, 0.75, -1.0];
        let references = vec![&sample];
        let sequences = build_sequences(&references);
        let device = Default::default();
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 4,
        }
        .init::<TestBackend>(&device);

        let chunks = prepare_sequence_chunks(&model, &sequences, 1, 1, &device);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].initial_memory, sample.initial_policy_memory);
    }

    #[test]
    fn recurrent_unroll_must_be_bounded() {
        let zero = BehaviorCloningConfig {
            recurrent_unroll_steps: 0,
            ..BehaviorCloningConfig::default()
        };
        assert!(zero.validate().is_err());
        let oversized = BehaviorCloningConfig {
            recurrent_unroll_steps: 257,
            ..BehaviorCloningConfig::default()
        };
        assert!(oversized.validate().is_err());
        let zero_epoch_budget = BehaviorCloningConfig {
            epoch_sample_budget: Some(0),
            ..BehaviorCloningConfig::default()
        };
        assert!(zero_epoch_budget.validate().is_err());
        let zero_optimizer_budget = BehaviorCloningConfig {
            optimizer_steps_per_epoch: Some(0),
            ..BehaviorCloningConfig::default()
        };
        assert!(zero_optimizer_budget.validate().is_err());
        let zero_total_optimizer_budget = BehaviorCloningConfig {
            optimizer_steps_per_epoch: Some(1),
            optimizer_step_budget: Some(0),
            ..BehaviorCloningConfig::default()
        };
        assert!(zero_total_optimizer_budget.validate().is_err());
        let unbounded_total_optimizer_budget = BehaviorCloningConfig {
            optimizer_step_budget: Some(1),
            ..BehaviorCloningConfig::default()
        };
        assert!(unbounded_total_optimizer_budget.validate().is_err());
        let excessive_total_optimizer_budget = BehaviorCloningConfig {
            epochs: 2,
            optimizer_steps_per_epoch: Some(3),
            optimizer_step_budget: Some(7),
            ..BehaviorCloningConfig::default()
        };
        assert!(excessive_total_optimizer_budget.validate().is_err());
        let zero_terminal_scale = BehaviorCloningConfig {
            optimizer_steps_per_epoch: Some(1),
            optimizer_step_budget: Some(1),
            terminal_optimizer_step_scale: 0.0,
            ..BehaviorCloningConfig::default()
        };
        assert!(zero_terminal_scale.validate().is_err());
        let excessive_terminal_scale = BehaviorCloningConfig {
            optimizer_steps_per_epoch: Some(1),
            optimizer_step_budget: Some(1),
            terminal_optimizer_step_scale: 1.01,
            ..BehaviorCloningConfig::default()
        };
        assert!(excessive_terminal_scale.validate().is_err());
        let unbounded_terminal_scale = BehaviorCloningConfig {
            terminal_optimizer_step_scale: 0.5,
            ..BehaviorCloningConfig::default()
        };
        assert!(unbounded_terminal_scale.validate().is_err());
        let invalid_balance = BehaviorCloningConfig {
            action_balance_exponent: 1.01,
            ..BehaviorCloningConfig::default()
        };
        assert!(invalid_balance.validate().is_err());
        let invalid_balance_ratio = BehaviorCloningConfig {
            action_balance_max_ratio: Some(0.99),
            ..BehaviorCloningConfig::default()
        };
        assert!(invalid_balance_ratio.validate().is_err());
        let invalid_initial_artifact = BehaviorCloningConfig {
            initial_artifact_sha256: Some("not-a-hash".into()),
            ..BehaviorCloningConfig::default()
        };
        assert!(invalid_initial_artifact.validate().is_err());
        let invalid_phase_gate_weight = BehaviorCloningConfig {
            phase_gate_loss_weight: 0.0,
            ..BehaviorCloningConfig::default()
        };
        assert!(invalid_phase_gate_weight.validate().is_err());
        let unweighted_margin = BehaviorCloningConfig {
            action_kind_margin: Some(0.0),
            ..BehaviorCloningConfig::default()
        };
        assert!(unweighted_margin.validate().is_err());
        let weight_without_margin = BehaviorCloningConfig {
            action_kind_margin_loss_weight: 1.0,
            ..BehaviorCloningConfig::default()
        };
        assert!(weight_without_margin.validate().is_err());
        let excessive_margin = BehaviorCloningConfig {
            action_kind_margin: Some(100.01),
            action_kind_margin_loss_weight: 1.0,
            ..BehaviorCloningConfig::default()
        };
        assert!(excessive_margin.validate().is_err());
        let unparented_expert_only = BehaviorCloningConfig {
            action_kind_expert_only: Some(SupervisionPhase::Exploration),
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_expert_only.validate().is_err());
        let unparented_adapter_only = BehaviorCloningConfig {
            foraging_adapter_only: true,
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_adapter_only.validate().is_err());
        let unparented_context_adapter = BehaviorCloningConfig {
            context_adapter_only: Some(SupervisionPhase::Combat),
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_context_adapter.validate().is_err());
        let invalid_foraging_context_adapter = BehaviorCloningConfig {
            initial_artifact_sha256: Some("a".repeat(64)),
            context_adapter_only: Some(SupervisionPhase::Feeding),
            ..BehaviorCloningConfig::default()
        };
        assert!(invalid_foraging_context_adapter.validate().is_err());
        let unparented_context_slot_adapter = BehaviorCloningConfig {
            context_slot_adapter_only: Some(SupervisionPhase::Combat),
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_context_slot_adapter.validate().is_err());
        let unparented_guard_readiness = BehaviorCloningConfig {
            exploration_guard_readiness_only: true,
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_guard_readiness.validate().is_err());
        let conflicting_guard_readiness = BehaviorCloningConfig {
            initial_artifact_sha256: Some("a".repeat(64)),
            context_slot_adapter_only: Some(SupervisionPhase::Exploration),
            exploration_guard_readiness_only: true,
            ..BehaviorCloningConfig::default()
        };
        assert!(conflicting_guard_readiness.validate().is_err());
        let conflicting_context_adapters = BehaviorCloningConfig {
            initial_artifact_sha256: Some("a".repeat(64)),
            context_adapter_only: Some(SupervisionPhase::Combat),
            context_slot_adapter_only: Some(SupervisionPhase::Combat),
            ..BehaviorCloningConfig::default()
        };
        assert!(conflicting_context_adapters.validate().is_err());
        let unparented_target_only = BehaviorCloningConfig {
            target_query_head_only: true,
            target_residual_only: false,
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_target_only.validate().is_err());
        let unparented_residual = BehaviorCloningConfig {
            target_residual_only: true,
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_residual.validate().is_err());
        let unparented_effort_only = BehaviorCloningConfig {
            effort_head_only: true,
            ..BehaviorCloningConfig::default()
        };
        assert!(unparented_effort_only.validate().is_err());
        let non_authoritative_router = BehaviorCloningConfig {
            expert_routing: ExpertRoutingStrategy::VisibleNeighborContext,
            ..BehaviorCloningConfig::default()
        };
        assert!(non_authoritative_router.validate().is_err());
    }

    #[test]
    fn expert_only_training_updates_only_the_selected_head() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 124);
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let mut sample = demonstration_samples(1).remove(0);
        sample.observation.fill(0.0);
        assert_eq!(
            expert_routing_phase(
                &sample,
                ExpertRoutingStrategy::ForagingInteractionExploration
            ),
            SupervisionPhase::Exploration
        );
        let chunk = SequenceChunk {
            samples: vec![&sample],
            initial_memory: vec![0.0; 8],
        };
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sample.observation.to_vec(), [1, OBS_DIM]),
            &device,
        );
        let before = model
            .forward(observation.clone())
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let frozen_template = model.clone().fork(&device);
        let mut optimizer = AdamWConfig::new().init();
        let action_weights = vec![1.0; NUM_ACTIONS];
        let phase_weights = [1.0; NUM_ACTION_KIND_EXPERTS];
        let trained = train_chunk_batch(
            model,
            &mut optimizer,
            &[chunk],
            SupervisedStep {
                loss_weights: SupervisedLossWeights {
                    actions: &action_weights,
                    phases: &phase_weights,
                    phase_gate: 1.0,
                },
                expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
                action_kind_expert_only: Some(SupervisionPhase::Exploration),
                foraging_adapter_only: false,
                context_adapter_only: None,
                context_slot_adapter_only: None,
                exploration_guard_readiness_only: false,
                action_kind_margin: None,
                action_kind_margin_loss_weight: 0.0,
                target_query_head_only: false,
                target_residual_only: false,
                effort_head_only: false,
                frozen_template: Some(&frozen_template),
                learning_rate: 1e-2,
                device: &device,
            },
        );
        let after = trained
            .forward(observation)
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let split = NUM_POLICY_ACTION_KINDS * 2;
        assert_eq!(&after[..split], &before[..split]);
        assert_ne!(&after[split..], &before[split..]);
    }

    #[test]
    fn adapter_only_training_preserves_trunk_and_other_experts() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 125);
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let mut sample = demonstration_samples(1).remove(0);
        sample.observation.fill(0.0);
        sample.observation[CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.25;
        sample.observation[CURRENT_TILE_PLANT_ENERGY_FEATURE] = 0.25;
        assert_eq!(
            expert_routing_phase(
                &sample,
                ExpertRoutingStrategy::ForagingInteractionExploration
            ),
            SupervisionPhase::Feeding
        );
        let chunk = SequenceChunk {
            samples: vec![&sample],
            initial_memory: vec![0.0; 8],
        };
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sample.observation.to_vec(), [1, OBS_DIM]),
            &device,
        );
        let before = model.forward(observation.clone());
        let before_experts = before
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let before_memory = before.next_memory.into_data().to_vec::<f32>().unwrap();
        let frozen_template = model.clone().fork(&device);
        let mut optimizer = AdamWConfig::new().init();
        let action_weights = vec![1.0; NUM_ACTIONS];
        let phase_weights = [1.0; NUM_ACTION_KIND_EXPERTS];
        let trained = train_chunk_batch(
            model,
            &mut optimizer,
            &[chunk],
            SupervisedStep {
                loss_weights: SupervisedLossWeights {
                    actions: &action_weights,
                    phases: &phase_weights,
                    phase_gate: 1.0,
                },
                expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
                action_kind_expert_only: None,
                foraging_adapter_only: true,
                context_adapter_only: None,
                context_slot_adapter_only: None,
                exploration_guard_readiness_only: false,
                action_kind_margin: None,
                action_kind_margin_loss_weight: 0.0,
                target_query_head_only: false,
                target_residual_only: false,
                effort_head_only: false,
                frozen_template: Some(&frozen_template),
                learning_rate: 1e-2,
                device: &device,
            },
        );
        let after = trained.forward(observation);
        let after_experts = after
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_ne!(
            &after_experts[..NUM_POLICY_ACTION_KINDS],
            &before_experts[..NUM_POLICY_ACTION_KINDS]
        );
        assert_eq!(
            &after_experts[NUM_POLICY_ACTION_KINDS..],
            &before_experts[NUM_POLICY_ACTION_KINDS..]
        );
        assert_eq!(
            after.next_memory.into_data().to_vec::<f32>().unwrap(),
            before_memory
        );
    }

    #[test]
    fn effort_only_training_preserves_every_other_output() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 126);
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let mut sample = demonstration_samples(1).remove(0);
        sample.action = u16::try_from(
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Move.index(),
                target: 0,
                effort: 2,
            })
            .unwrap(),
        )
        .unwrap();
        let chunk = SequenceChunk {
            samples: vec![&sample],
            initial_memory: vec![0.0; 8],
        };
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sample.observation.to_vec(), [1, OBS_DIM]),
            &device,
        );
        let before = model.forward(observation.clone());
        let before_kinds = before
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let before_targets = before.target_logits.into_data().to_vec::<f32>().unwrap();
        let before_efforts = before.effort_logits.into_data().to_vec::<f32>().unwrap();
        let before_memory = before.next_memory.into_data().to_vec::<f32>().unwrap();
        let frozen_template = model.clone().fork(&device);
        let mut optimizer = AdamWConfig::new().init();
        let action_weights = vec![1.0; NUM_ACTIONS];
        let phase_weights = [1.0; NUM_ACTION_KIND_EXPERTS];
        let trained = train_chunk_batch(
            model,
            &mut optimizer,
            &[chunk],
            SupervisedStep {
                loss_weights: SupervisedLossWeights {
                    actions: &action_weights,
                    phases: &phase_weights,
                    phase_gate: 1.0,
                },
                expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
                action_kind_expert_only: None,
                foraging_adapter_only: false,
                context_adapter_only: None,
                context_slot_adapter_only: None,
                exploration_guard_readiness_only: false,
                action_kind_margin: None,
                action_kind_margin_loss_weight: 0.0,
                target_query_head_only: false,
                target_residual_only: false,
                effort_head_only: true,
                frozen_template: Some(&frozen_template),
                learning_rate: 1e-2,
                device: &device,
            },
        );
        let after = trained.forward(observation);
        assert_eq!(
            after
                .action_kind_expert_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            before_kinds
        );
        assert_eq!(
            after.target_logits.into_data().to_vec::<f32>().unwrap(),
            before_targets
        );
        assert_ne!(
            after.effort_logits.into_data().to_vec::<f32>().unwrap(),
            before_efforts
        );
        assert_eq!(
            after.next_memory.into_data().to_vec::<f32>().unwrap(),
            before_memory
        );
    }

    #[test]
    fn context_slot_adapter_training_updates_only_the_selected_residual() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 127);
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let mut sample = demonstration_samples(1).remove(0);
        sample.observation.fill(0.0);
        sample.observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        sample.action = u16::try_from(
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Guard.index(),
                target: 0,
                effort: 0,
            })
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            expert_routing_phase(
                &sample,
                ExpertRoutingStrategy::ForagingInteractionExploration
            ),
            SupervisionPhase::Combat
        );
        let chunk = SequenceChunk {
            samples: vec![&sample],
            initial_memory: vec![0.0; 8],
        };
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sample.observation.to_vec(), [1, OBS_DIM]),
            &device,
        );
        let before = model.forward(observation.clone());
        let before_experts = before
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let before_targets = before.target_logits.into_data().to_vec::<f32>().unwrap();
        let before_memory = before.next_memory.into_data().to_vec::<f32>().unwrap();
        let frozen_template = model.clone().fork(&device);
        let mut optimizer = AdamWConfig::new().init();
        let action_weights = vec![1.0; NUM_ACTIONS];
        let phase_weights = [1.0; NUM_ACTION_KIND_EXPERTS];
        let trained = train_chunk_batch(
            model,
            &mut optimizer,
            &[chunk],
            SupervisedStep {
                loss_weights: SupervisedLossWeights {
                    actions: &action_weights,
                    phases: &phase_weights,
                    phase_gate: 1.0,
                },
                expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
                action_kind_expert_only: None,
                foraging_adapter_only: false,
                context_adapter_only: None,
                context_slot_adapter_only: Some(SupervisionPhase::Combat),
                exploration_guard_readiness_only: false,
                action_kind_margin: Some(0.0),
                action_kind_margin_loss_weight: 10.0,
                target_query_head_only: false,
                target_residual_only: false,
                effort_head_only: false,
                frozen_template: Some(&frozen_template),
                learning_rate: 1e-2,
                device: &device,
            },
        );
        let after = trained.forward(observation);
        let after_experts = after
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let interaction = NUM_POLICY_ACTION_KINDS;
        let guard = crate::action::PolicyActionKind::Guard.index();
        let movement = crate::action::PolicyActionKind::Move.index();
        assert!(
            after_experts[interaction + guard] - after_experts[interaction + movement]
                > before_experts[interaction + guard] - before_experts[interaction + movement]
        );
        assert_eq!(
            &after_experts[..NUM_POLICY_ACTION_KINDS],
            &before_experts[..NUM_POLICY_ACTION_KINDS]
        );
        assert_ne!(
            &after_experts[NUM_POLICY_ACTION_KINDS..2 * NUM_POLICY_ACTION_KINDS],
            &before_experts[NUM_POLICY_ACTION_KINDS..2 * NUM_POLICY_ACTION_KINDS]
        );
        assert_eq!(
            &after_experts[2 * NUM_POLICY_ACTION_KINDS..],
            &before_experts[2 * NUM_POLICY_ACTION_KINDS..]
        );
        assert_eq!(
            after.target_logits.into_data().to_vec::<f32>().unwrap(),
            before_targets
        );
        assert_eq!(
            after.next_memory.into_data().to_vec::<f32>().unwrap(),
            before_memory
        );
    }

    #[test]
    fn exploration_guard_readiness_training_updates_only_the_guard_residual() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 128);
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let mut sample = demonstration_samples(1).remove(0);
        sample.observation.fill(0.0);
        sample.observation[1] = 0.1;
        sample.action = u16::try_from(
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Guard.index(),
                target: 0,
                effort: 0,
            })
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            expert_routing_phase(
                &sample,
                ExpertRoutingStrategy::ForagingInteractionExploration
            ),
            SupervisionPhase::Exploration
        );
        let chunk = SequenceChunk {
            samples: vec![&sample],
            initial_memory: vec![0.0; 8],
        };
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sample.observation.to_vec(), [1, OBS_DIM]),
            &device,
        );
        let before = model.forward(observation.clone());
        let before_experts = before
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let before_targets = before.target_logits.into_data().to_vec::<f32>().unwrap();
        let before_memory = before.next_memory.into_data().to_vec::<f32>().unwrap();
        let frozen_template = model.clone().fork(&device);
        let mut optimizer = AdamWConfig::new().init();
        let action_weights = vec![1.0; NUM_ACTIONS];
        let phase_weights = [1.0; NUM_ACTION_KIND_EXPERTS];
        let trained = train_chunk_batch(
            model,
            &mut optimizer,
            &[chunk],
            SupervisedStep {
                loss_weights: SupervisedLossWeights {
                    actions: &action_weights,
                    phases: &phase_weights,
                    phase_gate: 1.0,
                },
                expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
                action_kind_expert_only: None,
                foraging_adapter_only: false,
                context_adapter_only: None,
                context_slot_adapter_only: None,
                exploration_guard_readiness_only: true,
                action_kind_margin: Some(0.0),
                action_kind_margin_loss_weight: 10.0,
                target_query_head_only: false,
                target_residual_only: false,
                effort_head_only: false,
                frozen_template: Some(&frozen_template),
                learning_rate: 1e-2,
                device: &device,
            },
        );
        let after = trained.forward(observation);
        let after_experts = after
            .action_kind_expert_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let exploration = 2 * NUM_POLICY_ACTION_KINDS;
        let guard = crate::action::PolicyActionKind::Guard.index();
        for index in 0..after_experts.len() {
            if index == exploration + guard {
                assert_ne!(after_experts[index], before_experts[index]);
            } else {
                assert_eq!(after_experts[index], before_experts[index]);
            }
        }
        assert_eq!(
            after.target_logits.into_data().to_vec::<f32>().unwrap(),
            before_targets
        );
        assert_eq!(
            after.next_memory.into_data().to_vec::<f32>().unwrap(),
            before_memory
        );
    }

    #[test]
    fn target_residual_training_changes_targets_and_freezes_every_inherited_parameter() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 720);
        let mut model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let mut sample = demonstration_samples(1).remove(0);
        for (i, value) in sample.observation.iter_mut().enumerate() {
            *value = (i % 37) as f32 / 37.0;
        }
        sample.action_mask.fill(true);
        let kind = crate::action::PolicyActionKind::Move.index();
        sample.action = u16::try_from(
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind,
                target: 1,
                effort: 0,
            })
            .unwrap(),
        )
        .unwrap();
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sample.observation.to_vec(), [1, OBS_DIM]),
            &device,
        );
        let before = model.forward(observation.clone());
        let before_targets = before.target_logits.into_data().to_vec::<f32>().unwrap();
        let before_memory = before.next_memory.into_data().to_vec::<f32>().unwrap();
        let before_kinds = before
            .action_kind_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let inherited_bits = model.inherited_target_parameter_bits();
        let frozen_template = model.clone().fork(&device);
        let mut optimizer = AdamWConfig::new().init();
        let action_weights = vec![1.0; NUM_ACTIONS];
        let phase_weights = [1.0; NUM_ACTION_KIND_EXPERTS];
        for _ in 0..8 {
            let chunk = SequenceChunk {
                samples: vec![&sample],
                initial_memory: vec![0.0; 8],
            };
            model = train_chunk_batch(
                model,
                &mut optimizer,
                &[chunk],
                SupervisedStep {
                    loss_weights: SupervisedLossWeights {
                        actions: &action_weights,
                        phases: &phase_weights,
                        phase_gate: 1.0,
                    },
                    expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
                    action_kind_expert_only: None,
                    foraging_adapter_only: false,
                    context_adapter_only: None,
                    context_slot_adapter_only: None,
                    exploration_guard_readiness_only: false,
                    action_kind_margin: None,
                    action_kind_margin_loss_weight: 0.0,
                    target_query_head_only: false,
                    target_residual_only: true,
                    effort_head_only: false,
                    frozen_template: Some(&frozen_template),
                    learning_rate: 1e-2,
                    device: &device,
                },
            );
            assert_eq!(model.inherited_target_parameter_bits(), inherited_bits);
        }
        let after = model.forward(observation);
        let targets = after.target_logits.into_data().to_vec::<f32>().unwrap();
        let offset = kind * NUM_POLICY_TARGETS;
        assert!(
            targets[offset + 1] - targets[offset]
                > before_targets[offset + 1] - before_targets[offset]
        );
        assert_eq!(
            after.next_memory.into_data().to_vec::<f32>().unwrap(),
            before_memory
        );
        assert_eq!(
            after
                .action_kind_logits
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            before_kinds
        );
    }

    #[test]
    fn initial_model_and_artifact_identity_must_be_paired() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let model_config = ModelConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        };
        let config_with_identity = BehaviorCloningConfig {
            initial_artifact_sha256: Some("a".repeat(64)),
            ..BehaviorCloningConfig::default()
        };
        let missing_model = behavior_clone::<TestBackend>(
            &[],
            &model_config,
            &config_with_identity,
            Default::default(),
        );
        assert!(missing_model.is_err());

        let device = Default::default();
        let initial_model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<TestBackend>(&device);
        let missing_identity = behavior_clone_from_model::<TestBackend>(
            &[],
            &model_config,
            &BehaviorCloningConfig::default(),
            Some(initial_model),
            device,
        );
        assert!(missing_identity.is_err());
    }

    #[test]
    fn action_weight_cap_only_limits_represented_labels() {
        let mut weights = vec![0.5, 4.0, 7.0];
        cap_weight_ratio(&mut weights, &[true, true, false], Some(2.0));
        assert_eq!(weights, vec![0.5, 1.0, 7.0]);
    }

    #[test]
    fn capped_action_weights_preserve_total_presentation_mass() {
        let mut samples = demonstration_samples(4);
        let consume =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Consume.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        let attack =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Attack.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        for sample in &mut samples[..3] {
            sample.action = consume as u16;
        }
        samples[3].action = attack as u16;

        let weights = action_weights_for_samples(
            samples.iter(),
            ActionBalancingStrategy::Family,
            1.0,
            Some(2.0),
        );
        let total_mass = 3.0 * weights[consume] + weights[attack];
        assert!((total_mass - samples.len() as f32).abs() < 1.0e-6);
        assert!((weights[attack] / weights[consume] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn family_balancing_uses_the_effective_presented_sample_mix() {
        let mut samples = demonstration_samples(4);
        let consume =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Consume.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        let attack =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Attack.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        for sample in &mut samples[..3] {
            sample.action = consume as u16;
        }
        samples[3].action = attack as u16;

        let weights =
            action_weights_for_samples(samples.iter(), ActionBalancingStrategy::Family, 1.0, None);

        let consume_mass = 3.0 * weights[consume];
        let attack_mass = weights[attack];
        assert!((consume_mass - attack_mass).abs() < 1.0e-6);
        assert!(weights[attack] > weights[consume]);
    }

    #[test]
    fn expert_kind_mask_routes_loss_to_only_the_supervised_phase() {
        let samples = demonstration_samples(1);
        let sample = &samples[0];
        let feeding = expert_kind_mask_bias(sample, SupervisionPhase::Feeding);
        let combat = expert_kind_mask_bias(sample, SupervisionPhase::Combat);
        let exploration = expert_kind_mask_bias(sample, SupervisionPhase::Exploration);
        assert!(feeding[..NUM_POLICY_ACTION_KINDS].contains(&0.0));
        assert!(feeding[NUM_POLICY_ACTION_KINDS..]
            .iter()
            .all(|bias| *bias == -1.0e9));
        assert!(combat[..NUM_POLICY_ACTION_KINDS]
            .iter()
            .all(|bias| *bias == -1.0e9));
        assert!(combat[NUM_POLICY_ACTION_KINDS..2 * NUM_POLICY_ACTION_KINDS].contains(&0.0));
        assert!(combat[2 * NUM_POLICY_ACTION_KINDS..]
            .iter()
            .all(|bias| *bias == -1.0e9));
        assert!(exploration[..2 * NUM_POLICY_ACTION_KINDS]
            .iter()
            .all(|bias| *bias == -1.0e9));
        assert!(exploration[2 * NUM_POLICY_ACTION_KINDS..].contains(&0.0));
    }

    #[test]
    fn identical_gate_observations_cannot_receive_conflicting_local_routes() {
        let mut samples = demonstration_samples(2);
        samples[1].observation = samples[0].observation.clone();
        let consume =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Consume.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        let attack =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Attack.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        samples[0].action = consume as u16;
        samples[1].action = attack as u16;
        let partition = DatasetPartition {
            manifest_sha256: "manifest".into(),
            training: samples.iter().collect(),
            validation: Vec::new(),
            held_out_seeds: Vec::new(),
        };

        assert!(validate_gate_label_consistency(
            &[partition],
            ExpertRoutingStrategy::LocalActionFamily,
        )
        .unwrap_err()
        .contains("contradictory"));
    }

    #[test]
    fn visible_neighbor_routing_is_observation_only_across_action_families() {
        let mut samples = demonstration_samples(2);
        samples[1].observation = samples[0].observation.clone();
        samples[0].observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        samples[1].observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        let attack =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Attack.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        samples[1].action = attack as u16;
        let partition = DatasetPartition {
            manifest_sha256: "manifest".into(),
            training: samples.iter().collect(),
            validation: Vec::new(),
            held_out_seeds: Vec::new(),
        };

        assert!(samples.iter().all(|sample| {
            expert_routing_phase(sample, ExpertRoutingStrategy::VisibleNeighborContext)
                == SupervisionPhase::Combat
        }));
        validate_gate_label_consistency(
            &[partition],
            ExpertRoutingStrategy::VisibleNeighborContext,
        )
        .unwrap();
    }

    #[test]
    fn three_way_routing_prioritizes_threat_then_current_food() {
        let mut samples = demonstration_samples(6);
        for sample in &mut samples {
            for feature in [
                CURRENT_TILE_PLANT_ENERGY_FEATURE,
                CURRENT_TILE_PLANT_CAPACITY_FEATURE,
                CURRENT_TILE_LOOSE_ENERGY_FEATURE,
                CURRENT_TILE_DIFFUSE_ENERGY_FEATURE,
            ] {
                sample.observation[feature] = 0.0;
            }
            for slot in sample.observation[HEADER_FEATURES..].chunks_exact_mut(SLOT_FEATURES) {
                for feature in [
                    SLOT_PLANT_ENERGY_FEATURE,
                    SLOT_LOOSE_ENERGY_FEATURE,
                    SLOT_DIFFUSE_ENERGY_FEATURE,
                    SLOT_NEIGHBOR_PRESENT_FEATURE,
                    SLOT_NEIGHBOR_ACTIVITY_FEATURE,
                ] {
                    slot[feature] = 0.0;
                }
            }
        }
        samples[0].observation[CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.25;
        samples[1].observation[HEADER_FEATURES + SLOT_LOOSE_ENERGY_FEATURE] = 0.5;
        samples[1].observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        samples[2].observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        samples[3].observation[HEADER_FEATURES + SLOT_LOOSE_ENERGY_FEATURE] = 0.5;
        samples[4].observation[CURRENT_TILE_PLANT_CAPACITY_FEATURE] = 0.25;
        samples[4].observation[HEADER_FEATURES + SLOT_NEIGHBOR_PRESENT_FEATURE] = 1.0;
        samples[4].observation[HEADER_FEATURES + SLOT_NEIGHBOR_ACTIVITY_FEATURE] = 3.0 / 8.0;
        samples[5].observation[CURRENT_TILE_DIFFUSE_ENERGY_FEATURE] = 0.25;

        let strategy = ExpertRoutingStrategy::ForagingInteractionExploration;
        assert_eq!(
            expert_routing_phase(&samples[0], strategy),
            SupervisionPhase::Feeding
        );
        assert_eq!(
            expert_routing_phase(&samples[1], strategy),
            SupervisionPhase::Combat,
            "a visible neighbor must take precedence over merely nearby food"
        );
        assert_eq!(
            expert_routing_phase(&samples[2], strategy),
            SupervisionPhase::Combat
        );
        assert_eq!(
            expert_routing_phase(&samples[3], strategy),
            SupervisionPhase::Exploration,
            "nearby food is a destination for the exploration expert"
        );
        assert_eq!(
            expert_routing_phase(&samples[4], strategy),
            SupervisionPhase::Combat,
            "visible attack windup must take precedence over current food"
        );
        assert_eq!(
            expert_routing_phase(&samples[5], strategy),
            SupervisionPhase::Exploration,
            "diffuse energy feeds plants but is not directly consumable"
        );
    }

    #[test]
    fn recurrent_chunk_sampling_preserves_the_exact_sample_budget() {
        let mut samples = demonstration_samples(5);
        let attack =
            crate::action::compose_policy_action(crate::action::HierarchicalActionChoice {
                kind: crate::action::PolicyActionKind::Attack.index(),
                target: 0,
                effort: 0,
            })
            .unwrap();
        for sample in &mut samples[3..] {
            sample.action = attack as u16;
        }
        let first = SequenceChunk {
            samples: samples[..3].iter().collect(),
            initial_memory: vec![0.0; 8],
        };
        let second = SequenceChunk {
            samples: samples[3..].iter().collect(),
            initial_memory: vec![0.0; 8],
        };
        let mut rng = ChaCha12Rng::seed_from_u64(3);

        let phase_weights = phase_weights_for_chunks(
            &[first.clone(), second.clone()],
            ExpertRoutingStrategy::LocalActionFamily,
        );
        assert!(
            (3.0 * phase_weights[SupervisionPhase::Feeding.index()]
                - 2.0 * phase_weights[SupervisionPhase::Combat.index()])
            .abs()
                < 1.0e-6
        );

        let epoch = sample_chunk_epoch(&[vec![first, second]], &[7], &mut rng);

        assert_eq!(
            epoch.iter().map(|chunk| chunk.samples.len()).sum::<usize>(),
            7
        );
        assert!(epoch.iter().all(|chunk| !chunk.samples.is_empty()));
    }

    #[test]
    fn masked_supervision_learns_a_small_maintained_mind_dataset() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let training = small_training_config();
        let (manifest, mut payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![20, 21],
                max_samples: 32,
            },
        )
        .unwrap();
        for sample in &mut payload.samples {
            sample.observation[CURRENT_TILE_PLANT_ENERGY_FEATURE] = 0.25;
            for slot in sample.observation[HEADER_FEATURES..].chunks_exact_mut(SLOT_FEATURES) {
                slot[SLOT_NEIGHBOR_PRESENT_FEATURE] = 0.0;
                slot[SLOT_NEIGHBOR_ACTIVITY_FEATURE] = 0.0;
            }
        }
        let dataset = LoadedDemonstrations {
            directory: PathBuf::from("in-memory"),
            manifest_sha256: "manifest".into(),
            manifest,
            payload,
        };
        let model = ModelConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        };
        let config = BehaviorCloningConfig {
            seed: 9,
            initial_artifact_sha256: None,
            inherited_seed_ledger: None,
            confirmation_seeds: Vec::new(),
            epochs: 20,
            minibatch_size: 16,
            learning_rate: 1e-2,
            validation_fraction: 0.5,
            dataset_sampling: DatasetSamplingStrategy::Balanced,
            dataset_sampling_weights: Vec::new(),
            epoch_sample_budget: None,
            optimizer_steps_per_epoch: None,
            optimizer_step_budget: None,
            terminal_optimizer_step_scale: 1.0,
            expert_routing: ExpertRoutingStrategy::ForagingInteractionExploration,
            phase_gate_loss_weight: 1.0,
            action_kind_expert_only: None,
            foraging_adapter_only: false,
            context_adapter_only: None,
            context_slot_adapter_only: None,
            exploration_guard_readiness_only: false,
            action_kind_margin: None,
            action_kind_margin_loss_weight: 0.0,
            target_query_head_only: false,
            target_residual_only: false,
            effort_head_only: false,
            recurrent_unroll_steps: 4,
            exact_round_trip_only: false,
            action_balancing: ActionBalancingStrategy::None,
            action_balance_exponent: 1.0,
            action_balance_max_ratio: None,
        };
        let (trained, metrics) = behavior_clone::<TestBackend>(
            std::slice::from_ref(&dataset),
            &model,
            &config,
            Default::default(),
        )
        .unwrap();
        assert!(metrics.final_training_loss < metrics.initial_training_loss);
        assert!(metrics.final_training_accuracy >= metrics.initial_training_accuracy);
        assert!(metrics.validation_samples > 0);
        assert!(metrics.final_validation_loss.is_some());
        assert_eq!(metrics.recurrent_unroll_steps, 4);
        assert_eq!(metrics.partitions[0].eligible_phase_samples[1], 0);
        assert_eq!(
            metrics.final_validation_phase_metrics,
            vec![
                BehaviorCloningPhaseMetrics {
                    phase: "foraging".into(),
                    samples: metrics.validation_samples,
                    gate_accuracy: metrics.final_validation_phase_metrics[0].gate_accuracy,
                },
                BehaviorCloningPhaseMetrics {
                    phase: "interaction".into(),
                    samples: 0,
                    gate_accuracy: None,
                },
                BehaviorCloningPhaseMetrics {
                    phase: "exploration".into(),
                    samples: 0,
                    gate_accuracy: None,
                },
            ]
        );
        assert_eq!(
            metrics.action_family_presentations.iter().sum::<usize>(),
            metrics.sample_presentations
        );
        assert_eq!(
            metrics
                .supervision_phase_presentations
                .iter()
                .sum::<usize>(),
            metrics.sample_presentations
        );
        assert_eq!(
            metrics.final_validation_family_metrics.len(),
            PolicyActionFamily::COUNT
        );
        assert_eq!(
            metrics
                .final_validation_family_metrics
                .iter()
                .map(|family| family.family.as_str())
                .collect::<Vec<_>>(),
            PolicyActionFamily::ALL
                .into_iter()
                .map(PolicyActionFamily::name)
                .collect::<Vec<_>>()
        );
        assert!(metrics.training_trajectories > 0);
        assert!(metrics.validation_trajectories > 0);
        assert_eq!(metrics.completed_epochs, config.epochs);

        let (retrained, retrained_metrics) = behavior_clone::<TestBackend>(
            std::slice::from_ref(&dataset),
            &model,
            &config,
            Default::default(),
        )
        .unwrap();
        assert_eq!(retrained_metrics, metrics);
        let observations = dataset
            .payload
            .samples
            .iter()
            .flat_map(|sample| sample.observation.iter().copied())
            .collect::<Vec<_>>();
        let device = Default::default();
        let shape = [dataset.payload.samples.len(), OBS_DIM];
        let trained_logits = trained
            .forward(Tensor::<TestBackend, 2>::from_data(
                TensorData::new(observations.clone(), shape),
                &device,
            ))
            .action_kind_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let retrained_logits = retrained
            .forward(Tensor::<TestBackend, 2>::from_data(
                TensorData::new(observations, shape),
                &device,
            ))
            .action_kind_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(retrained_logits, trained_logits);

        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("pretrained");
        publish_behavior_clone(
            &output,
            &trained,
            &model,
            &config,
            std::slice::from_ref(&dataset),
            &metrics,
        )
        .unwrap();
        assert!(output.join("model.mpk").is_file());
        let artifact: BehaviorCloningArtifact =
            serde_json::from_slice(&fs::read(output.join("behavior-cloning.json")).unwrap())
                .unwrap();
        assert_eq!(artifact.metrics, metrics);
        assert_eq!(
            artifact.model_sha256,
            sha256_file(&output.join("model.mpk")).unwrap()
        );
        let artifact_sha256 = behavior_clone_artifact_sha256(&output).unwrap();
        assert_eq!(
            verify_behavior_clone_artifact(&output, &artifact_sha256, &model).unwrap(),
            output.join("model")
        );
        assert!(
            verify_behavior_clone_artifact(&output, &"0".repeat(64), &model)
                .unwrap_err()
                .contains("artifact SHA-256")
        );
    }

    #[test]
    fn total_optimizer_step_budget_stops_mid_epoch_and_counts_actual_presentations() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let training = small_training_config();
        let (manifest, payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![90, 91],
                max_samples: 32,
            },
        )
        .unwrap();
        let dataset = LoadedDemonstrations {
            directory: PathBuf::from("step-budget"),
            manifest_sha256: "step-budget-manifest".into(),
            manifest,
            payload,
        };
        let model = ModelConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        };
        let config = BehaviorCloningConfig {
            epochs: 2,
            minibatch_size: 16,
            validation_fraction: 0.5,
            epoch_sample_budget: Some(16),
            optimizer_steps_per_epoch: Some(2),
            optimizer_step_budget: Some(3),
            recurrent_unroll_steps: 1,
            ..BehaviorCloningConfig::default()
        };

        let (full_terminal_step, metrics) = behavior_clone::<TestBackend>(
            std::slice::from_ref(&dataset),
            &model,
            &config,
            Default::default(),
        )
        .unwrap();

        assert_eq!(metrics.optimizer_steps, 3);
        assert_eq!(metrics.completed_epochs, 1);
        assert_eq!(metrics.samples_per_epoch, 16);
        assert_eq!(metrics.sample_presentations, 24);
        assert_eq!(
            metrics.action_family_presentations.iter().sum::<usize>(),
            metrics.sample_presentations
        );
        assert_eq!(
            metrics
                .supervision_phase_presentations
                .iter()
                .sum::<usize>(),
            metrics.sample_presentations
        );

        let fractional_config = BehaviorCloningConfig {
            terminal_optimizer_step_scale: 0.5,
            ..config.clone()
        };
        let (fractional_terminal_step, fractional_metrics) = behavior_clone::<TestBackend>(
            std::slice::from_ref(&dataset),
            &model,
            &fractional_config,
            Default::default(),
        )
        .unwrap();
        assert_eq!(fractional_metrics.optimizer_steps, metrics.optimizer_steps);
        assert_eq!(
            fractional_metrics.sample_presentations,
            metrics.sample_presentations
        );
        let observation = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                dataset.payload.samples[0].observation.to_vec(),
                [1, OBS_DIM],
            ),
            &Default::default(),
        );
        let full_logits = full_terminal_step
            .forward(observation.clone())
            .action_kind_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let fractional_logits = fractional_terminal_step
            .forward(observation)
            .action_kind_logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_ne!(fractional_logits, full_logits);
    }

    #[test]
    fn partially_overlapping_corpora_share_one_seed_partition() {
        let training = small_training_config();
        let (manifest, payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![3, 2, 4, 1],
                max_samples: 16,
            },
        )
        .unwrap();
        let dataset = |name: &str, seeds: &[u64]| {
            let mut payload = payload.clone();
            payload.samples.retain(|s| seeds.contains(&s.source_seed));
            LoadedDemonstrations {
                directory: name.into(),
                manifest_sha256: name.into(),
                manifest: manifest.clone(),
                payload,
            }
        };
        let config = BehaviorCloningConfig {
            seed: 42,
            validation_fraction: 0.5,
            ..Default::default()
        };
        let datasets = vec![dataset("first", &[3, 2, 4]), dataset("second", &[2, 4, 1])];
        let partitions = partition_datasets(&datasets, &config).unwrap();
        let training: HashSet<_> = partitions
            .iter()
            .flat_map(|p| &p.training)
            .map(|s| s.source_seed)
            .collect();
        let validation: HashSet<_> = partitions
            .iter()
            .flat_map(|p| &p.validation)
            .map(|s| s.source_seed)
            .collect();
        assert!(training.is_disjoint(&validation));
        assert_eq!(validation, HashSet::from([2, 3]));
        let incompatible = vec![dataset("first", &[3, 2]), dataset("second", &[2, 4])];
        assert!(
            partition_datasets(&incompatible, &config).is_err(),
            "never switch a seed's role to fill an empty partition"
        );
        let reversed = vec![datasets[1].clone(), datasets[0].clone()];
        let reverse = partition_datasets(&reversed, &config).unwrap();
        assert_eq!(partitions[0].held_out_seeds, reverse[1].held_out_seeds);
    }

    #[test]
    fn partitions_hold_out_whole_seeds_and_balance_uneven_datasets() {
        let training = small_training_config();
        let (manifest, payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![30, 31, 32, 33],
                max_samples: 32,
            },
        )
        .unwrap();
        let first = LoadedDemonstrations {
            directory: PathBuf::from("first"),
            manifest_sha256: "first-manifest".into(),
            manifest: manifest.clone(),
            payload: payload.clone(),
        };
        let mut second = LoadedDemonstrations {
            directory: PathBuf::from("second"),
            manifest_sha256: "second-manifest".into(),
            manifest,
            payload,
        };
        let repeated = second.payload.samples.clone();
        second.payload.samples.extend(repeated);
        let datasets = vec![first, second];
        let config = BehaviorCloningConfig {
            validation_fraction: 0.25,
            ..BehaviorCloningConfig::default()
        };

        let partitions = partition_datasets(&datasets, &config).unwrap();
        assert_eq!(partitions[0].held_out_seeds, partitions[1].held_out_seeds);
        for partition in &partitions {
            let held_out = partition
                .held_out_seeds
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            assert!(partition
                .training
                .iter()
                .all(|sample| !held_out.contains(&sample.source_seed)));
            assert!(partition
                .validation
                .iter()
                .all(|sample| held_out.contains(&sample.source_seed)));
        }
        let presentations = samples_per_dataset_per_epoch(
            &partitions,
            DatasetSamplingStrategy::Balanced,
            &[],
            None,
        )
        .unwrap();
        assert_eq!(presentations[0], presentations[1]);
        assert!(partitions[0].training.len() < partitions[1].training.len());

        let mut invalid = config;
        invalid.validation_fraction = 1.0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn explicit_dataset_weights_preserve_budget_and_order() {
        let training = small_training_config();
        let (manifest, payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![40, 41, 42, 43],
                max_samples: 32,
            },
        )
        .unwrap();
        let datasets = vec![
            LoadedDemonstrations {
                directory: PathBuf::from("first"),
                manifest_sha256: "weighted-first".into(),
                manifest: manifest.clone(),
                payload: payload.clone(),
            },
            LoadedDemonstrations {
                directory: PathBuf::from("second"),
                manifest_sha256: "weighted-second".into(),
                manifest,
                payload,
            },
        ];
        let config = BehaviorCloningConfig {
            validation_fraction: 0.25,
            ..BehaviorCloningConfig::default()
        };
        let partitions = partition_datasets(&datasets, &config).unwrap();
        let presentations = samples_per_dataset_per_epoch(
            &partitions,
            DatasetSamplingStrategy::Proportional,
            &[3.0, 1.0],
            None,
        )
        .unwrap();
        assert_eq!(presentations.iter().sum::<usize>(), 48);
        assert_eq!(presentations, vec![36, 12]);
        let fixed_presentations = samples_per_dataset_per_epoch(
            &partitions,
            DatasetSamplingStrategy::Proportional,
            &[3.0, 1.0],
            Some(47),
        )
        .unwrap();
        assert_eq!(fixed_presentations, vec![35, 12]);
        assert_eq!(fixed_presentations.iter().sum::<usize>(), 47);
        assert!(samples_per_dataset_per_epoch(
            &partitions,
            DatasetSamplingStrategy::Proportional,
            &[1.0],
            None,
        )
        .is_err());
    }

    #[test]
    fn exact_optimizer_step_budget_respects_recurrent_batch_capacity() {
        let groups = vec![(1, 20), (4, 18), (16, 9)];
        // Minimum: ceil(20/32) + ceil(18/8) + ceil(9/2) = 9.
        let counts = exact_recurrent_batch_counts(&groups, 32, 12).unwrap();
        assert_eq!(counts.iter().sum::<usize>(), 12);
        for ((length, chunks), batches) in groups.iter().zip(&counts) {
            assert!(*batches <= *chunks);
            assert!(chunks.div_ceil(*batches) * length <= 32 || *length > 32);
        }
        assert!(exact_recurrent_batch_counts(&groups, 32, 8).is_err());
        assert!(exact_recurrent_batch_counts(&groups, 32, 48).is_err());
        assert_eq!(
            counts,
            exact_recurrent_batch_counts(&groups, 32, 12).unwrap()
        );
    }
}
