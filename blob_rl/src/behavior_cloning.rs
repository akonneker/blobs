//! Masked supervised pretraining from verified demonstration datasets.

use std::collections::{BTreeMap, BTreeSet, HashSet};
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
};
use crate::observation::OBS_DIM;

pub const BEHAVIOR_CLONING_SCHEMA_VERSION: u32 = 15;
static CLONING_NONCE: AtomicU64 = AtomicU64::new(0);

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningConfig {
    pub seed: u64,
    /// Hash of the verified behavior-cloning artifact used to initialize this
    /// stage, or `None` for a fresh model. The path is deliberately excluded
    /// so artifact identity remains machine-independent.
    pub initial_artifact_sha256: Option<String>,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    /// Fraction of distinct source seeds held out as complete trajectories.
    /// The split is deterministic and shared across datasets with the same
    /// seed suite.
    pub validation_fraction: f64,
    pub dataset_sampling: DatasetSamplingStrategy,
    /// Optional explicit sampling mass for each dataset, in CLI order. When
    /// present this overrides `dataset_sampling` while preserving the same
    /// total presentations per epoch.
    #[serde(default)]
    pub dataset_sampling_weights: Vec<f64>,
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
        let valid_initial_artifact = self.initial_artifact_sha256.as_ref().is_none_or(|hash| {
            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        });
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
            || self.recurrent_unroll_steps == 0
            || self.recurrent_unroll_steps > 256
            || !self.action_balance_exponent.is_finite()
            || !(0.0..=1.0).contains(&self.action_balance_exponent)
            || self
                .action_balance_max_ratio
                .is_some_and(|ratio| !ratio.is_finite() || ratio < 1.0)
        {
            return Err(
                "behavior-cloning initial artifact must be a SHA-256 hash, epochs, batch size, learning rate, and explicit dataset weights must be positive, validation fraction and action-balance exponent must be in [0, 1], action-balance max ratio must be finite and at least 1, and recurrent unroll steps must be in 1..=256".into(),
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
            epochs: 10,
            minibatch_size: 256,
            learning_rate: 3e-4,
            validation_fraction: 0.1,
            dataset_sampling: DatasetSamplingStrategy::Balanced,
            dataset_sampling_weights: Vec::new(),
            recurrent_unroll_steps: 16,
            exact_round_trip_only: false,
            action_balancing: ActionBalancingStrategy::None,
            action_balance_exponent: 1.0,
            action_balance_max_ratio: None,
        }
    }
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
    pub recurrent_unroll_steps: usize,
    pub training_trajectories: usize,
    pub validation_trajectories: usize,
    pub exact_round_trip_samples: usize,
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
    pub partitions: Vec<BehaviorCloningDatasetPartition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorCloningDatasetPartition {
    pub manifest_sha256: String,
    pub eligible_samples: usize,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub training_trajectories: usize,
    pub validation_trajectories: usize,
    pub samples_per_epoch: usize,
    pub held_out_seeds: Vec<u64>,
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

struct DatasetPartition<'a> {
    manifest_sha256: String,
    training: Vec<&'a DemonstrationSample>,
    validation: Vec<&'a DemonstrationSample>,
    held_out_seeds: Vec<u64>,
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

fn action_weights(
    partitions: &[DatasetPartition<'_>],
    strategy: ActionBalancingStrategy,
    exponent: f64,
    max_ratio: Option<f64>,
) -> Vec<f32> {
    if strategy == ActionBalancingStrategy::None {
        return vec![1.0; NUM_ACTIONS];
    }
    let mut action_counts = vec![0usize; NUM_ACTIONS];
    for sample in partitions.iter().flat_map(|partition| &partition.training) {
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
                .into_iter()
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
    weights
}

fn seed_rank(split_seed: u64, source_seed: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"blob-behavior-cloning-validation-seed-v1");
    hasher.update(split_seed.to_le_bytes());
    hasher.update(source_seed.to_le_bytes());
    hasher.finalize().into()
}

fn partition_datasets<'a>(
    datasets: &'a [LoadedDemonstrations],
    config: &BehaviorCloningConfig,
) -> Result<Vec<DatasetPartition<'a>>, String> {
    let mut identities = HashSet::with_capacity(datasets.len());
    let mut partitions = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        if !identities.insert(&dataset.manifest_sha256) {
            return Err(format!(
                "duplicate demonstration manifest {}",
                dataset.manifest_sha256
            ));
        }
        let eligible = dataset
            .payload
            .samples
            .iter()
            .filter(|sample| !config.exact_round_trip_only || sample.exact_round_trip)
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "behavior-cloning filters removed every sample from dataset {}",
                dataset.manifest_sha256
            ));
        }
        let source_seeds = eligible
            .iter()
            .map(|sample| sample.source_seed)
            .collect::<BTreeSet<_>>();
        let held_out_seeds = if config.validation_fraction == 0.0 {
            Vec::new()
        } else {
            if source_seeds.len() < 2 {
                return Err(format!(
                    "dataset {} needs at least two eligible source seeds for held-out validation",
                    dataset.manifest_sha256
                ));
            }
            let mut ranked = source_seeds.into_iter().collect::<Vec<_>>();
            ranked.sort_unstable_by_key(|source_seed| seed_rank(config.seed, *source_seed));
            let held_out_count = ((ranked.len() as f64 * config.validation_fraction).round()
                as usize)
                .clamp(1, ranked.len() - 1);
            let mut held_out = ranked[..held_out_count].to_vec();
            held_out.sort_unstable();
            held_out
        };
        let held_out_set = held_out_seeds.iter().copied().collect::<HashSet<_>>();
        let (validation, training): (Vec<_>, Vec<_>) = eligible
            .into_iter()
            .partition(|sample| held_out_set.contains(&sample.source_seed));
        if training.is_empty() || (config.validation_fraction > 0.0 && validation.is_empty()) {
            return Err(format!(
                "dataset {} produced an empty training or validation partition",
                dataset.manifest_sha256
            ));
        }
        partitions.push(DatasetPartition {
            manifest_sha256: dataset.manifest_sha256.clone(),
            training,
            validation,
            held_out_seeds,
        });
    }
    Ok(partitions)
}

fn samples_per_dataset_per_epoch(
    partitions: &[DatasetPartition<'_>],
    strategy: DatasetSamplingStrategy,
    explicit_weights: &[f64],
) -> Result<Vec<usize>, String> {
    if !explicit_weights.is_empty() {
        if explicit_weights.len() != partitions.len() {
            return Err(format!(
                "received {} dataset sampling weights for {} datasets",
                explicit_weights.len(),
                partitions.len()
            ));
        }
        let total = partitions
            .iter()
            .map(|partition| partition.training.len())
            .sum::<usize>();
        let weight_total = explicit_weights.iter().sum::<f64>();
        let raw = explicit_weights
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
        return Ok(samples);
    }
    Ok(match strategy {
        DatasetSamplingStrategy::Proportional => partitions
            .iter()
            .map(|partition| partition.training.len())
            .collect(),
        DatasetSamplingStrategy::Balanced => {
            let total = partitions
                .iter()
                .map(|partition| partition.training.len())
                .sum::<usize>();
            vec![total / partitions.len(); partitions.len()]
        }
    })
}

fn kind_mask_bias(sample: &DemonstrationSample) -> impl Iterator<Item = f32> {
    policy_action_kind_mask(&sample.action_mask)
        .into_iter()
        .map(|allowed| if allowed { 0.0 } else { -1.0e9 })
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

type DemonstrationSequence<'a> = Vec<&'a DemonstrationSample>;

#[derive(Clone)]
struct SequenceChunk<'a> {
    samples: Vec<&'a DemonstrationSample>,
    initial_memory: Vec<f32>,
}

fn build_sequences<'a>(samples: &[&'a DemonstrationSample]) -> Vec<DemonstrationSequence<'a>> {
    let mut sequences = BTreeMap::<(u64, u64), DemonstrationSequence<'a>>::new();
    for sample in samples {
        sequences
            .entry((sample.source_seed, sample.source_cell))
            .or_default()
            .push(*sample);
    }
    sequences.into_values().collect()
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
        let mut memories = vec![vec![0.0; recurrent_size]; sequence_batch.len()];
        while positions
            .iter()
            .zip(sequence_batch)
            .any(|(position, sequence)| *position < sequence.len())
        {
            let active = positions
                .iter()
                .zip(sequence_batch)
                .enumerate()
                .filter_map(|(index, (position, sequence))| {
                    (*position < sequence.len()).then_some(index)
                })
                .collect::<Vec<_>>();
            for index in &active {
                let position = positions[*index];
                if position.is_multiple_of(unroll_steps) {
                    let end = (position + unroll_steps).min(sequence_batch[*index].len());
                    chunks.push(SequenceChunk {
                        samples: sequence_batch[*index][position..end].to_vec(),
                        initial_memory: memories[*index].clone(),
                    });
                }
            }
            let observations = active
                .iter()
                .flat_map(|index| {
                    sequence_batch[*index][positions[*index]]
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

fn evaluate_sequences<B: AutodiffBackend>(
    model: &PolicyValueNet<B>,
    sequences: &[DemonstrationSequence<'_>],
    sequence_batch_size: usize,
    device: &B::Device,
) -> Option<(f64, f64)>
where
    f32: From<B::FloatElem>,
{
    if sequences.is_empty() {
        return None;
    }
    let recurrent_size = model.recurrent_size();
    let mut total_loss = 0.0_f64;
    let mut correct = 0usize;
    let mut total = 0usize;
    for sequence_batch in sequences.chunks(sequence_batch_size.max(1)) {
        let mut positions = vec![0usize; sequence_batch.len()];
        let mut memories = vec![vec![0.0; recurrent_size]; sequence_batch.len()];
        while positions
            .iter()
            .zip(sequence_batch)
            .any(|(position, sequence)| *position < sequence.len())
        {
            let active = positions
                .iter()
                .zip(sequence_batch)
                .enumerate()
                .filter_map(|(index, (position, sequence))| {
                    (*position < sequence.len()).then_some(index)
                })
                .collect::<Vec<_>>();
            let samples = active
                .iter()
                .map(|index| sequence_batch[*index][positions[*index]])
                .collect::<Vec<_>>();
            let observations = samples
                .iter()
                .flat_map(|sample| sample.observation.iter().copied())
                .collect::<Vec<_>>();
            let kind_masks = samples
                .iter()
                .flat_map(|sample| kind_mask_bias(sample))
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
                + NUM_POLICY_TARGET_LOGITS
                + NUM_POLICY_EFFORT_LOGITS
                + NUM_POLICY_AMOUNT_LOGITS
                + NUM_SIGNAL_CHOICES
                + NUM_SIGNAL_STRENGTH_CHOICES
                + recurrent_size;
            let data = Tensor::cat(
                vec![
                    burn::tensor::activation::log_softmax(kind_logits, 1),
                    burn::tensor::activation::log_softmax(target_logits, 1),
                    burn::tensor::activation::log_softmax(effort_logits, 1),
                    burn::tensor::activation::log_softmax(amount_logits, 1),
                    burn::tensor::activation::log_softmax(signal_logits, 1),
                    burn::tensor::activation::log_softmax(signal_strength_logits, 1),
                    output.next_memory,
                ],
                1,
            )
            .into_data()
            .to_vec::<f32>()
            .expect("behavior-cloning evaluation uses f32");
            for (row, (index, sample)) in active.into_iter().zip(samples).enumerate() {
                let start = row * width;
                let action = usize::from(sample.action);
                let hierarchical = decompose_policy_action(action)
                    .expect("validated demonstration action is in the policy catalog");
                let amount = usize::from(sample.amount);
                let signal = usize::from(sample.signal);
                let signal_strength = usize::from(sample.signal_strength);
                let target_logits_start = start + NUM_POLICY_ACTION_KINDS;
                let effort_logits_start = target_logits_start + NUM_POLICY_TARGET_LOGITS;
                let amount_logits_start = effort_logits_start + NUM_POLICY_EFFORT_LOGITS;
                let signal_start = amount_logits_start + NUM_POLICY_AMOUNT_LOGITS;
                let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
                let target_index = hierarchical.kind * NUM_POLICY_TARGETS + hierarchical.target;
                let effort_index = hierarchical.kind * NUM_POLICY_EFFORTS + hierarchical.effort;
                let amount_index = hierarchical.kind * NUM_AMOUNT_CHOICES + amount;
                total_loss -= f64::from(
                    data[start + hierarchical.kind]
                        + data[target_logits_start + target_index]
                        + data[effort_logits_start + effort_index]
                        + data[amount_logits_start + amount_index]
                        + data[signal_start + signal]
                        + data[signal_strength_start + signal_strength],
                );
                let predicted_kind = (0..NUM_POLICY_ACTION_KINDS)
                    .max_by(|left, right| {
                        data[start + *left]
                            .total_cmp(&data[start + *right])
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
                correct += usize::from(
                    predicted_kind == hierarchical.kind
                        && predicted_target == hierarchical.target
                        && predicted_effort == hierarchical.effort
                        && predicted_amount == amount
                        && predicted_signal == signal
                        && predicted_signal_strength == signal_strength,
                );
                memories[index] = canonicalize_memory(
                    &data[signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES..start + width],
                );
                positions[index] += 1;
                total += 1;
            }
        }
    }
    Some((
        total_loss / total.max(1) as f64,
        correct as f64 / total.max(1) as f64,
    ))
}

fn quantize_straight_through<B: Backend>(memory: Tensor<B, 2>) -> Tensor<B, 2> {
    let quantized = (memory.clone().detach().clamp(-1.0, 1.0) * f32::from(i16::MAX)).round()
        / f32::from(i16::MAX);
    memory.clone() + (quantized - memory).detach()
}

fn train_chunk_batch<B: AutodiffBackend>(
    mut model: PolicyValueNet<B>,
    optimizer: &mut impl Optimizer<PolicyValueNet<B>, B>,
    chunks: &[SequenceChunk<'_>],
    action_weights: &[f32],
    learning_rate: f64,
    device: &B::Device,
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
        device,
    );
    let mut loss = None;
    for step in 0..steps {
        let samples = chunks
            .iter()
            .map(|chunk| chunk.samples[step])
            .collect::<Vec<_>>();
        let observations = samples
            .iter()
            .flat_map(|sample| sample.observation.iter().copied())
            .collect::<Vec<_>>();
        let kind_masks = samples
            .iter()
            .flat_map(|sample| kind_mask_bias(sample))
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
        let kinds = hierarchical
            .iter()
            .map(|choice| choice.kind as i32)
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
            .map(|sample| action_weights[usize::from(sample.action)])
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
                device,
            ),
            memory,
        );
        let kind_logits = output.action_kind_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(kind_masks, [chunks.len(), NUM_POLICY_ACTION_KINDS]),
                device,
            );
        let target_logits = output.target_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(target_masks, [chunks.len(), NUM_POLICY_TARGET_LOGITS]),
                device,
            );
        let effort_logits = output.effort_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(effort_masks, [chunks.len(), NUM_POLICY_EFFORT_LOGITS]),
                device,
            );
        let signal_logits = output.signal_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(signal_masks, [chunks.len(), NUM_SIGNAL_CHOICES]),
                device,
            );
        let signal_strength_logits = output.signal_strength_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(
                    signal_strength_masks,
                    [chunks.len(), NUM_SIGNAL_STRENGTH_CHOICES],
                ),
                device,
            );
        let amount_logits = output.amount_logits
            + Tensor::<B, 2>::from_data(
                TensorData::new(amount_masks, [chunks.len(), NUM_POLICY_AMOUNT_LOGITS]),
                device,
            );
        let kind_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(kinds, [chunks.len()]), device);
        let target_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(targets, [chunks.len()]), device);
        let effort_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(efforts, [chunks.len()]), device);
        let signal_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(signals, [chunks.len()]), device);
        let signal_strength_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(signal_strengths, [chunks.len()]),
            device,
        );
        let amount_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(amounts, [chunks.len()]), device);
        let sample_weight_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(sample_weights, [chunks.len(), 1]), device);
        let step_loss = ((burn::tensor::activation::log_softmax(kind_logits, 1)
            .gather(1, kind_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(target_logits, 1)
                .gather(1, target_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(effort_logits, 1)
                .gather(1, effort_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(amount_logits, 1)
                .gather(1, amount_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(signal_logits, 1)
                .gather(1, signal_tensor.unsqueeze_dim(1))
            + burn::tensor::activation::log_softmax(signal_strength_logits, 1)
                .gather(1, signal_strength_tensor.unsqueeze_dim(1)))
            * sample_weight_tensor)
            .mean()
            .neg();
        loss = Some(match loss {
            Some(loss) => loss + step_loss,
            None => step_loss,
        });
        memory = quantize_straight_through(output.next_memory);
    }
    let loss = loss.expect("a sequence chunk always contains a sample") / steps as f32;
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    model = optimizer.step(learning_rate, model, gradients);
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
/// `initial_model` is present so the published lineage is self-authenticating.
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
    let action_weights = action_weights(
        &partitions,
        config.action_balancing,
        config.action_balance_exponent,
        config.action_balance_max_ratio,
    );
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
    )?;
    let samples_per_epoch = per_dataset.iter().sum::<usize>();
    let sample_presentations = samples_per_epoch
        .checked_mul(config.epochs)
        .ok_or_else(|| "behavior-cloning sample presentation count overflowed".to_string())?;
    let partition_metrics = partitions
        .iter()
        .zip(&per_dataset)
        .enumerate()
        .map(
            |(index, (partition, samples_per_epoch))| BehaviorCloningDatasetPartition {
                manifest_sha256: partition.manifest_sha256.clone(),
                eligible_samples: partition.training.len() + partition.validation.len(),
                training_samples: partition.training.len(),
                validation_samples: partition.validation.len(),
                training_trajectories: training_sequences_by_dataset[index].len(),
                validation_trajectories: validation_sequences_by_dataset[index].len(),
                samples_per_epoch: *samples_per_epoch,
                held_out_seeds: partition.held_out_seeds.clone(),
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
    let mut optimizer = AdamWConfig::new()
        .init()
        .with_grad_clipping(GradientClipping::Norm(1.0));
    let (initial_training_loss, initial_training_accuracy) =
        evaluate_sequences(&model, &training_sequences, config.minibatch_size, &device)
            .expect("a nonempty training partition has a trajectory");
    let (initial_validation_loss, initial_validation_accuracy) = match evaluate_sequences(
        &model,
        &validation_sequences,
        config.minibatch_size,
        &device,
    ) {
        Some((loss, accuracy)) => (Some(loss), Some(accuracy)),
        None => (None, None),
    };
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed ^ 0x4245_4841_5649_4f52);
    let mut optimizer_steps = 0usize;
    for _ in 0..config.epochs {
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
        let mut by_length = BTreeMap::<usize, Vec<SequenceChunk<'_>>>::new();
        for chunk in epoch {
            by_length
                .entry(chunk.samples.len())
                .or_default()
                .push(chunk);
        }
        let mut chunk_batches = Vec::new();
        for (length, chunks) in &mut by_length {
            chunks.shuffle(&mut rng);
            let chunk_batch_size = (config.minibatch_size / *length).max(1);
            for batch in chunks.chunks(chunk_batch_size) {
                chunk_batches.push(batch.to_vec());
            }
        }
        chunk_batches.shuffle(&mut rng);
        for batch in chunk_batches {
            model = train_chunk_batch(
                model,
                &mut optimizer,
                &batch,
                &action_weights,
                config.learning_rate,
                &device,
            );
            optimizer_steps += 1;
        }
    }
    let (final_training_loss, final_training_accuracy) =
        evaluate_sequences(&model, &training_sequences, config.minibatch_size, &device)
            .expect("a nonempty training partition has a trajectory");
    let (final_validation_loss, final_validation_accuracy) = match evaluate_sequences(
        &model,
        &validation_sequences,
        config.minibatch_size,
        &device,
    ) {
        Some((loss, accuracy)) => (Some(loss), Some(accuracy)),
        None => (None, None),
    };
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
            recurrent_unroll_steps: config.recurrent_unroll_steps,
            training_trajectories: training_sequences.len(),
            validation_trajectories: validation_sequences.len(),
            exact_round_trip_samples,
            epochs: config.epochs,
            optimizer_steps,
            initial_training_loss,
            final_training_loss,
            initial_training_accuracy,
            final_training_accuracy,
            initial_validation_loss,
            final_validation_loss,
            initial_validation_accuracy,
            final_validation_accuracy,
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
    let metadata_path = directory.join("behavior-cloning.json");
    let metadata = fs::read(&metadata_path)
        .map_err(|error| format!("failed to read {}: {error}", metadata_path.display()))?;
    if metadata.len() > 1024 * 1024 {
        return Err("behavior-cloning metadata exceeds 1 MiB".into());
    }
    if format!("{:x}", Sha256::digest(&metadata)) != expected_artifact_sha256 {
        return Err("behavior-cloning artifact SHA-256 mismatch".into());
    }
    let artifact: BehaviorCloningArtifact = serde_json::from_slice(&metadata)
        .map_err(|error| format!("failed to decode {}: {error}", metadata_path.display()))?;
    if artifact.schema_version != BEHAVIOR_CLONING_SCHEMA_VERSION
        || artifact.model != *expected_model
        || artifact.model_file != "model.mpk"
    {
        return Err("behavior-cloning artifact schema or model architecture mismatch".into());
    }
    let model_file = directory.join(&artifact.model_file);
    if sha256_file(&model_file)? != artifact.model_sha256 {
        return Err("behavior-cloned model SHA-256 mismatch".into());
    }
    Ok(directory.join("model"))
}

pub fn publish_behavior_clone<B: AutodiffBackend>(
    output: &Path,
    model: &PolicyValueNet<B>,
    model_config: &ModelConfig,
    config: &BehaviorCloningConfig,
    datasets: &[LoadedDemonstrations],
    metrics: &BehaviorCloningMetrics,
) -> Result<PathBuf, String> {
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
                    teacher: dataset.manifest.teacher.to_string(),
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
    use burn::backend::{Autodiff, NdArray};

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
        assert!(sequences.iter().all(|sequence| sequence.len() == 1));
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
        assert!(chunks[1].initial_memory.iter().any(|value| *value != 0.0));
        assert_eq!(
            chunks[1].initial_memory,
            canonicalize_memory(&chunks[1].initial_memory)
        );
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
    fn recurrent_chunk_sampling_preserves_the_exact_sample_budget() {
        let samples = demonstration_samples(5);
        let first = SequenceChunk {
            samples: samples[..3].iter().collect(),
            initial_memory: vec![0.0; 8],
        };
        let second = SequenceChunk {
            samples: samples[3..].iter().collect(),
            initial_memory: vec![0.0; 8],
        };
        let mut rng = ChaCha12Rng::seed_from_u64(3);

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
        let (manifest, payload) = generate_demonstrations(
            &training,
            "config".into(),
            &DemonstrationOptions {
                teacher: MaintainedMindProfile::Simple,
                seeds: vec![20, 21],
                max_samples: 32,
            },
        )
        .unwrap();
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
            epochs: 20,
            minibatch_size: 16,
            learning_rate: 1e-2,
            validation_fraction: 0.5,
            dataset_sampling: DatasetSamplingStrategy::Balanced,
            dataset_sampling_weights: Vec::new(),
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
        assert!(metrics.training_trajectories > 0);
        assert!(metrics.validation_trajectories > 0);

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
        let presentations =
            samples_per_dataset_per_epoch(&partitions, DatasetSamplingStrategy::Balanced, &[])
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
        )
        .unwrap();
        assert_eq!(presentations.iter().sum::<usize>(), 48);
        assert_eq!(presentations, vec![36, 12]);
        assert!(samples_per_dataset_per_epoch(
            &partitions,
            DatasetSamplingStrategy::Proportional,
            &[1.0],
        )
        .is_err());
    }
}
