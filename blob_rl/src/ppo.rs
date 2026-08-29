//! PPO (Proximal Policy Optimization) algorithm.

use std::collections::BTreeMap;

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::action::{
    decompose_policy_action, policy_action_kind_mask, policy_effort_mask, policy_target_mask,
    PolicyActionKind, NUM_AMOUNT_CHOICES, NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS,
    NUM_POLICY_EFFORTS, NUM_POLICY_EFFORT_LOGITS, NUM_POLICY_TARGETS, NUM_POLICY_TARGET_LOGITS,
    NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::config::PPOConfig;
use crate::model::PolicyValueNet;
use crate::observation::OBS_DIM;

/// Detached functional outputs from a verified initial or frontier policy.
#[derive(Debug, Clone)]
pub struct PolicyAnchorTarget {
    pub action_kind_logits: Vec<f32>,
    pub target_logits: Vec<f32>,
    pub effort_logits: Vec<f32>,
    pub amount_logits: Vec<f32>,
    pub signal_logits: Vec<f32>,
    pub signal_strength_logits: Vec<f32>,
    pub next_memory: Vec<f32>,
}

/// A single experience transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub observation: Vec<f32>,
    /// Cell-private recurrent state before this decision.
    pub policy_memory: Vec<f32>,
    pub action_mask: Vec<bool>,
    pub action: usize,
    pub amount_mask: Vec<bool>,
    pub amount: usize,
    pub signal_mask: Vec<bool>,
    pub signal: usize,
    pub signal_strength_mask: Vec<bool>,
    pub signal_strength: usize,
    /// Exact legal-kind behavior-policy mixture used when this action was
    /// sampled. Contact curricula may override the global PPO default.
    pub action_kind_exploration_floor: f32,
    /// Exact direct Attack mixture used after the legal-kind mixture. This is
    /// nonzero only for assigned named micro-combat training rollouts.
    pub attack_action_kind_exploration_floor: f32,
    pub reward: f32,
    pub value: f32,
    /// Value at this cell's next decision frontier. This is supplied only
    /// after the same cell becomes ready again; death/episode termination use
    /// zero.
    pub next_value: f32,
    pub log_prob: f32,
    /// Detached functional target from the active verified teacher on this
    /// cell's ordinary observation and private recurrent state.
    pub anchor: Option<PolicyAnchorTarget>,
    /// Exact coefficient applied to the functional anchor for this decision.
    /// Curriculum stages and host-side teacher selection may vary it without
    /// becoming part of the Mind input or behavior policy.
    pub initial_policy_anchor_coeff: f32,
    pub done: bool,
    /// Simulated time between this decision and its next frontier, expressed
    /// in nominal ruleset time units rather than resolver batches.
    pub elapsed_time: f32,
    /// False while the action is still pending at the end of collection.
    /// Unresolved tails are excluded from PPO instead of being treated as
    /// false terminals with a zero bootstrap.
    pub complete: bool,
    /// Trajectory identifier (env_id, cell_id) for per-trajectory GAE.
    pub trajectory_id: (usize, usize),
}

/// Collected rollout data from N environments.
pub struct RolloutBuffer {
    pub transitions: Vec<Transition>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        RolloutBuffer {
            transitions: Vec::new(),
        }
    }

    pub fn push(&mut self, t: Transition) {
        self.transitions.push(t);
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    pub fn retain_complete(&mut self) {
        self.transitions.retain(|transition| transition.complete);
    }
}

impl Default for RolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagnostics aggregated across all minibatches evaluated by a PPO update.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PpoMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    /// Functional divergence from the frozen qualified initial policy.
    pub anchor_loss: f32,
    pub clip_fraction: f32,
    pub explained_variance: f32,
    pub recurrent_unroll_steps: usize,
    pub recurrent_chunks: usize,
    pub optimizer_steps: usize,
    pub epochs_completed: usize,
    pub early_stopped: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RecurrentChunk {
    indices: Vec<usize>,
}

fn behavior_kind_fixed_and_scale(
    mask: &[bool; NUM_POLICY_ACTION_KINDS],
    uniform_floor: f32,
    attack_floor: f32,
) -> ([f32; NUM_POLICY_ACTION_KINDS], f32) {
    debug_assert!(
        attack_floor == 0.0 || mask[PolicyActionKind::Attack.index()],
        "an Attack mixture may only be recorded when Attack is legal"
    );
    let legal = mask.iter().filter(|allowed| **allowed).count().max(1) as f32;
    let mut fixed = [0.0; NUM_POLICY_ACTION_KINDS];
    for (kind, allowed) in mask.iter().enumerate() {
        let uniform = if *allowed { uniform_floor / legal } else { 0.0 };
        fixed[kind] = (1.0 - attack_floor) * uniform
            + if kind == PolicyActionKind::Attack.index() {
                attack_floor
            } else {
                0.0
            };
    }
    (fixed, (1.0 - uniform_floor) * (1.0 - attack_floor))
}

fn recurrent_chunks(transitions: &[Transition], unroll_steps: usize) -> Vec<RecurrentChunk> {
    let mut trajectories = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (index, transition) in transitions.iter().enumerate() {
        trajectories
            .entry(transition.trajectory_id)
            .or_default()
            .push(index);
    }
    let mut chunks = Vec::new();
    for indices in trajectories.into_values() {
        let mut current = Vec::with_capacity(unroll_steps);
        for index in indices {
            current.push(index);
            if transitions[index].done || current.len() == unroll_steps {
                chunks.push(RecurrentChunk { indices: current });
                current = Vec::with_capacity(unroll_steps);
            }
        }
        if !current.is_empty() {
            chunks.push(RecurrentChunk { indices: current });
        }
    }
    chunks
}

fn shuffled_recurrent_minibatches(
    chunks: &[RecurrentChunk],
    decision_budget: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<RecurrentChunk>> {
    let mut by_length = BTreeMap::<usize, Vec<RecurrentChunk>>::new();
    for chunk in chunks {
        by_length
            .entry(chunk.indices.len())
            .or_default()
            .push(chunk.clone());
    }
    let mut batches = Vec::new();
    for (length, chunks) in &mut by_length {
        chunks.shuffle(rng);
        let chunk_batch_size = (decision_budget / *length).max(1);
        batches.extend(
            chunks
                .chunks(chunk_batch_size)
                .map(<[RecurrentChunk]>::to_vec),
        );
    }
    batches.shuffle(rng);
    batches
}

fn quantize_straight_through<B: Backend>(memory: Tensor<B, 2>) -> Tensor<B, 2> {
    let quantized = (memory.clone().detach().clamp(-1.0, 1.0) * f32::from(i16::MAX)).round()
        / f32::from(i16::MAX);
    memory.clone() + (quantized - memory).detach()
}

/// KL(reference || candidate) for one masked categorical policy head. The
/// reference side is detached so gradients can update only the live policy.
fn categorical_anchor_kl<B: Backend>(
    candidate_logits: Tensor<B, 2>,
    reference_logits: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let reference_logits = reference_logits.detach();
    let reference_probs = burn::tensor::activation::softmax(reference_logits.clone(), 1);
    let reference_log_probs = burn::tensor::activation::log_softmax(reference_logits, 1);
    let candidate_log_probs = burn::tensor::activation::log_softmax(candidate_logits, 1);
    (reference_probs * (reference_log_probs - candidate_log_probs))
        .sum_dim(1)
        .squeeze_dims::<1>(&[1])
}

fn explained_variance(predictions: &[f32], targets: &[f32]) -> f32 {
    debug_assert_eq!(predictions.len(), targets.len());
    if targets.is_empty() {
        return 0.0;
    }

    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let residual_mean = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| target - prediction)
        .sum::<f32>()
        / targets.len() as f32;
    let target_variance = targets
        .iter()
        .map(|target| (target - target_mean).powi(2))
        .sum::<f32>()
        / targets.len() as f32;
    if target_variance <= f32::EPSILON {
        return 0.0;
    }
    let residual_variance = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (target - prediction - residual_mean).powi(2))
        .sum::<f32>()
        / targets.len() as f32;
    1.0 - residual_variance / target_variance
}

/// Compute Generalized Advantage Estimation (GAE) per trajectory.
/// Transitions must be tagged with trajectory_id. GAE is computed
/// independently within each (env, cell) trajectory.
pub fn compute_gae_per_trajectory(
    transitions: &[Transition],
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = transitions.len();
    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];

    // Group transition indices by trajectory_id, preserving order
    let mut trajectory_indices: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, t) in transitions.iter().enumerate() {
        trajectory_indices
            .entry(t.trajectory_id)
            .or_default()
            .push(i);
    }

    // Compute GAE within each trajectory independently
    for indices in trajectory_indices.values() {
        let traj_len = indices.len();
        let mut last_gae = 0.0f32;

        for step in (0..traj_len).rev() {
            let idx = indices[step];
            let t = &transitions[idx];

            debug_assert!(t.complete, "GAE received an unresolved transition");
            let elapsed_time = t.elapsed_time.max(f32::EPSILON);
            let discount = gamma.powf(elapsed_time);
            let trace_discount = (gamma * gae_lambda).powf(elapsed_time);
            let mask = if t.done { 0.0 } else { 1.0 };
            let delta = t.reward + discount * mask * t.next_value - t.value;
            last_gae = delta + trace_discount * mask * last_gae;
            advantages[idx] = last_gae;
            returns[idx] = last_gae + t.value;
        }
    }

    (advantages, returns)
}

#[derive(Debug, Clone, Copy, Default)]
struct RecurrentBatchMetrics {
    policy_loss: f32,
    value_loss: f32,
    entropy: f32,
    approx_kl: f32,
    anchor_loss: f32,
    clip_fraction: f32,
    samples: usize,
}

#[allow(clippy::too_many_arguments)]
fn train_recurrent_batch<B: AutodiffBackend>(
    mut model: PolicyValueNet<B>,
    optimizer: &mut impl Optimizer<PolicyValueNet<B>, B>,
    rollout: &RolloutBuffer,
    advantages: &[f32],
    returns: &[f32],
    chunks: &[RecurrentChunk],
    config: &PPOConfig,
    device: &B::Device,
) -> (PolicyValueNet<B>, RecurrentBatchMetrics, bool)
where
    f32: From<B::FloatElem>,
{
    let steps = chunks[0].indices.len();
    debug_assert!(chunks.iter().all(|chunk| chunk.indices.len() == steps));
    let batch_size = chunks.len();
    let recurrent_size = model.recurrent_size();
    let mut memory = Tensor::<B, 2>::from_data(
        TensorData::new(
            chunks
                .iter()
                .flat_map(|chunk| {
                    rollout.transitions[chunk.indices[0]]
                        .policy_memory
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>(),
            [batch_size, recurrent_size],
        ),
        device,
    );
    let mut total_loss = None;
    let mut policy_loss_sum = None;
    let mut value_loss_sum = None;
    let mut entropy_sum = None;
    let mut approx_kl_sum = None;
    let mut anchor_loss_sum = None;
    let mut clip_fraction_sum = None;

    for step in 0..steps {
        let indices = chunks
            .iter()
            .map(|chunk| chunk.indices[step])
            .collect::<Vec<_>>();
        let obs_data = indices
            .iter()
            .flat_map(|index| rollout.transitions[*index].observation.iter().copied())
            .collect::<Vec<_>>();
        let kind_masks = indices
            .iter()
            .map(|index| policy_action_kind_mask(&rollout.transitions[*index].action_mask))
            .collect::<Vec<_>>();
        let kind_mask_bias = kind_masks
            .iter()
            .flat_map(|mask| {
                mask.iter()
                    .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
            })
            .collect::<Vec<_>>();
        let kind_fixed = kind_masks
            .iter()
            .zip(&indices)
            .flat_map(|(mask, index)| {
                let transition = &rollout.transitions[*index];
                behavior_kind_fixed_and_scale(
                    mask,
                    transition.action_kind_exploration_floor,
                    transition.attack_action_kind_exploration_floor,
                )
                .0
            })
            .collect::<Vec<_>>();
        let kind_policy_scale = kind_masks
            .iter()
            .zip(&indices)
            .flat_map(|(mask, index)| {
                let transition = &rollout.transitions[*index];
                std::iter::repeat_n(
                    behavior_kind_fixed_and_scale(
                        mask,
                        transition.action_kind_exploration_floor,
                        transition.attack_action_kind_exploration_floor,
                    )
                    .1,
                    NUM_POLICY_ACTION_KINDS,
                )
            })
            .collect::<Vec<_>>();
        let batch_advantages = indices
            .iter()
            .map(|index| advantages[*index])
            .collect::<Vec<_>>();
        let batch_returns = indices
            .iter()
            .map(|index| returns[*index])
            .collect::<Vec<_>>();
        let batch_old_values = indices
            .iter()
            .map(|index| rollout.transitions[*index].value)
            .collect::<Vec<_>>();
        let old_log_probs = indices
            .iter()
            .map(|index| rollout.transitions[*index].log_prob)
            .collect::<Vec<_>>();
        let hierarchical = indices
            .iter()
            .map(|index| {
                decompose_policy_action(rollout.transitions[*index].action)
                    .expect("rollout action is in the policy catalog")
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
        let target_mask_bias = indices
            .iter()
            .zip(&hierarchical)
            .flat_map(|(index, choice)| {
                let mut bias = vec![-1.0e9; NUM_POLICY_TARGET_LOGITS];
                let start = choice.kind * NUM_POLICY_TARGETS;
                for (target, allowed) in
                    policy_target_mask(&rollout.transitions[*index].action_mask, choice.kind)
                        .into_iter()
                        .enumerate()
                {
                    if allowed {
                        bias[start + target] = 0.0;
                    }
                }
                bias
            })
            .collect::<Vec<_>>();
        let effort_mask_bias = indices
            .iter()
            .zip(&hierarchical)
            .flat_map(|(index, choice)| {
                let mut bias = vec![-1.0e9; NUM_POLICY_EFFORT_LOGITS];
                let start = choice.kind * NUM_POLICY_EFFORTS;
                for (effort, allowed) in policy_effort_mask(
                    &rollout.transitions[*index].action_mask,
                    choice.kind,
                    choice.target,
                )
                .into_iter()
                .enumerate()
                {
                    if allowed {
                        bias[start + effort] = 0.0;
                    }
                }
                bias
            })
            .collect::<Vec<_>>();
        let amounts = indices
            .iter()
            .zip(&hierarchical)
            .map(|(index, choice)| {
                (choice.kind * NUM_AMOUNT_CHOICES + rollout.transitions[*index].amount) as i32
            })
            .collect::<Vec<_>>();
        let amount_mask_bias = indices
            .iter()
            .zip(&hierarchical)
            .flat_map(|(index, choice)| {
                let mut bias = vec![-1.0e9; NUM_POLICY_AMOUNT_LOGITS];
                let start = choice.kind * NUM_AMOUNT_CHOICES;
                for (amount, allowed) in rollout.transitions[*index].amount_mask.iter().enumerate()
                {
                    if *allowed {
                        bias[start + amount] = 0.0;
                    }
                }
                bias
            })
            .collect::<Vec<_>>();
        let signals = indices
            .iter()
            .map(|index| rollout.transitions[*index].signal as i32)
            .collect::<Vec<_>>();
        let signal_mask_bias = indices
            .iter()
            .flat_map(|index| {
                rollout.transitions[*index]
                    .signal_mask
                    .iter()
                    .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
            })
            .collect::<Vec<_>>();
        let signal_strengths = indices
            .iter()
            .map(|index| rollout.transitions[*index].signal_strength as i32)
            .collect::<Vec<_>>();
        let signal_strength_mask_bias = indices
            .iter()
            .flat_map(|index| {
                rollout.transitions[*index]
                    .signal_strength_mask
                    .iter()
                    .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
            })
            .collect::<Vec<_>>();

        let observations =
            Tensor::<B, 2>::from_data(TensorData::new(obs_data, [batch_size, OBS_DIM]), device);
        let output = model.forward_with_memory(observations, memory);
        let kind_mask_bias = Tensor::<B, 2>::from_data(
            TensorData::new(kind_mask_bias, [batch_size, NUM_POLICY_ACTION_KINDS]),
            device,
        );
        let masked_kind_logits = output.action_kind_logits + kind_mask_bias.clone();
        let learned_kind_logits = masked_kind_logits.clone();
        let kind_probs = burn::tensor::activation::softmax(masked_kind_logits, 1)
            * Tensor::<B, 2>::from_data(
                TensorData::new(kind_policy_scale, [batch_size, NUM_POLICY_ACTION_KINDS]),
                device,
            )
            + Tensor::<B, 2>::from_data(
                TensorData::new(kind_fixed, [batch_size, NUM_POLICY_ACTION_KINDS]),
                device,
            );
        let kind_log_probs = kind_probs.clone().clamp_min(1.0e-20).log();
        let target_mask_bias = Tensor::<B, 2>::from_data(
            TensorData::new(target_mask_bias, [batch_size, NUM_POLICY_TARGET_LOGITS]),
            device,
        );
        let masked_target_logits = output.target_logits + target_mask_bias.clone();
        let target_log_probs =
            burn::tensor::activation::log_softmax(masked_target_logits.clone(), 1);
        let effort_mask_bias = Tensor::<B, 2>::from_data(
            TensorData::new(effort_mask_bias, [batch_size, NUM_POLICY_EFFORT_LOGITS]),
            device,
        );
        let masked_effort_logits = output.effort_logits + effort_mask_bias.clone();
        let effort_log_probs =
            burn::tensor::activation::log_softmax(masked_effort_logits.clone(), 1);
        let amount_mask_bias = Tensor::<B, 2>::from_data(
            TensorData::new(amount_mask_bias, [batch_size, NUM_POLICY_AMOUNT_LOGITS]),
            device,
        );
        let masked_amount_logits = output.amount_logits + amount_mask_bias.clone();
        let amount_log_probs =
            burn::tensor::activation::log_softmax(masked_amount_logits.clone(), 1);
        let signal_mask_bias = Tensor::<B, 2>::from_data(
            TensorData::new(signal_mask_bias, [batch_size, NUM_SIGNAL_CHOICES]),
            device,
        );
        let masked_signal_logits = output.signal_logits + signal_mask_bias.clone();
        let signal_log_probs =
            burn::tensor::activation::log_softmax(masked_signal_logits.clone(), 1);
        let signal_strength_mask_bias = Tensor::<B, 2>::from_data(
            TensorData::new(
                signal_strength_mask_bias,
                [batch_size, NUM_SIGNAL_STRENGTH_CHOICES],
            ),
            device,
        );
        let masked_signal_strength_logits =
            output.signal_strength_logits + signal_strength_mask_bias.clone();
        let signal_strength_log_probs =
            burn::tensor::activation::log_softmax(masked_signal_strength_logits.clone(), 1);
        let kinds_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(kinds, [batch_size]), device);
        let targets_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(targets, [batch_size]), device);
        let efforts_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(efforts, [batch_size]), device);
        let amounts_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(amounts, [batch_size]), device);
        let signals_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(signals, [batch_size]), device);
        let signal_strengths_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(signal_strengths, [batch_size]), device);
        let action_log_probs = kind_log_probs
            .clone()
            .gather(1, kinds_tensor.unsqueeze_dim(1))
            .squeeze_dims::<1>(&[1])
            + target_log_probs
                .clone()
                .gather(1, targets_tensor.unsqueeze_dim(1))
                .squeeze_dims::<1>(&[1])
            + effort_log_probs
                .clone()
                .gather(1, efforts_tensor.unsqueeze_dim(1))
                .squeeze_dims::<1>(&[1])
            + amount_log_probs
                .clone()
                .gather(1, amounts_tensor.unsqueeze_dim(1))
                .squeeze_dims::<1>(&[1])
            + signal_log_probs
                .clone()
                .gather(1, signals_tensor.unsqueeze_dim(1))
                .squeeze_dims::<1>(&[1])
            + signal_strength_log_probs
                .clone()
                .gather(1, signal_strengths_tensor.unsqueeze_dim(1))
                .squeeze_dims::<1>(&[1]);
        let old_log_probs_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(old_log_probs, [batch_size]), device);
        let log_ratio = action_log_probs - old_log_probs_tensor;
        let ratio = log_ratio.clone().exp();
        let advantages_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(batch_advantages, [batch_size]), device);
        let surr1 = ratio.clone() * advantages_tensor.clone();
        let surr2 = ratio
            .clone()
            .clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
            * advantages_tensor;
        let policy_loss = surr1.min_pair(surr2).mean().neg();

        let returns_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(batch_returns, [batch_size]), device);
        let old_values_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(batch_old_values, [batch_size]), device);
        let predicted_values = output.values.squeeze_dims::<1>(&[1]);
        let clipped_values = old_values_tensor.clone()
            + (predicted_values.clone() - old_values_tensor)
                .clamp(-config.value_clip_epsilon, config.value_clip_epsilon);
        let value_loss = (predicted_values - returns_tensor.clone())
            .powf_scalar(2.0)
            .max_pair((clipped_values - returns_tensor).powf_scalar(2.0))
            .mean();
        let anchor_loss = indices
            .iter()
            .any(|index| rollout.transitions[*index].initial_policy_anchor_coeff > 0.0)
            .then(|| {
                let anchors = indices
                    .iter()
                    .map(|index| {
                        rollout.transitions[*index]
                            .anchor
                            .as_ref()
                            .expect("anchored PPO transition has a frozen target")
                    })
                    .collect::<Vec<_>>();
                let reference_kind_logits = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.action_kind_logits.iter().copied())
                            .collect(),
                        [batch_size, NUM_POLICY_ACTION_KINDS],
                    ),
                    device,
                );
                let reference_target_logits = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.target_logits.iter().copied())
                            .collect(),
                        [batch_size, NUM_POLICY_TARGET_LOGITS],
                    ),
                    device,
                );
                let reference_effort_logits = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.effort_logits.iter().copied())
                            .collect(),
                        [batch_size, NUM_POLICY_EFFORT_LOGITS],
                    ),
                    device,
                );
                let reference_amount_logits = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.amount_logits.iter().copied())
                            .collect(),
                        [batch_size, NUM_POLICY_AMOUNT_LOGITS],
                    ),
                    device,
                );
                let reference_signal_logits = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.signal_logits.iter().copied())
                            .collect(),
                        [batch_size, NUM_SIGNAL_CHOICES],
                    ),
                    device,
                );
                let reference_signal_strength_logits = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.signal_strength_logits.iter().copied())
                            .collect(),
                        [batch_size, NUM_SIGNAL_STRENGTH_CHOICES],
                    ),
                    device,
                );
                let reference_next_memory = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        anchors
                            .iter()
                            .flat_map(|target| target.next_memory.iter().copied())
                            .collect(),
                        [batch_size, recurrent_size],
                    ),
                    device,
                );
                let head_kl = categorical_anchor_kl(
                    learned_kind_logits,
                    reference_kind_logits + kind_mask_bias,
                ) + categorical_anchor_kl(
                    masked_target_logits.clone(),
                    reference_target_logits + target_mask_bias,
                ) + categorical_anchor_kl(
                    masked_effort_logits.clone(),
                    reference_effort_logits + effort_mask_bias,
                ) + categorical_anchor_kl(
                    masked_amount_logits.clone(),
                    reference_amount_logits + amount_mask_bias,
                ) + categorical_anchor_kl(
                    masked_signal_logits.clone(),
                    reference_signal_logits + signal_mask_bias,
                ) + categorical_anchor_kl(
                    masked_signal_strength_logits.clone(),
                    reference_signal_strength_logits + signal_strength_mask_bias,
                );
                let memory_loss = (output.next_memory.clone() - reference_next_memory)
                    .powf_scalar(2.0)
                    .sum_dim(1)
                    .squeeze_dims::<1>(&[1])
                    / recurrent_size as f32;
                let per_sample_loss = head_kl + memory_loss;
                let coefficients = Tensor::<B, 1>::from_data(
                    TensorData::new(
                        indices
                            .iter()
                            .map(|index| rollout.transitions[*index].initial_policy_anchor_coeff)
                            .collect(),
                        [batch_size],
                    ),
                    device,
                );
                let weighted_penalty = (per_sample_loss.clone() * coefficients).mean();
                (per_sample_loss.mean(), weighted_penalty)
            });
        let target_probs = burn::tensor::activation::softmax(masked_target_logits, 1);
        let effort_probs = burn::tensor::activation::softmax(masked_effort_logits, 1);
        let amount_probs = burn::tensor::activation::softmax(masked_amount_logits, 1);
        let signal_probs = burn::tensor::activation::softmax(masked_signal_logits, 1);
        let signal_strength_probs =
            burn::tensor::activation::softmax(masked_signal_strength_logits, 1);
        let entropy = (-(kind_probs * kind_log_probs).sum_dim(1)
            - (target_probs * target_log_probs).sum_dim(1)
            - (effort_probs * effort_log_probs).sum_dim(1)
            - (amount_probs * amount_log_probs).sum_dim(1)
            - (signal_probs * signal_log_probs).sum_dim(1)
            - (signal_strength_probs * signal_strength_log_probs).sum_dim(1))
        .mean();
        let approx_kl = (ratio.clone() - log_ratio - 1.0).mean();
        let clip_fraction = (ratio - 1.0)
            .abs()
            .greater_elem(config.clip_epsilon)
            .float()
            .mean();
        let mut step_loss = policy_loss.clone() + value_loss.clone() * config.value_loss_coeff
            - entropy.clone() * config.entropy_coeff;
        if let Some((anchor_loss, anchor_penalty)) = anchor_loss {
            step_loss = step_loss + anchor_penalty;
            anchor_loss_sum = Some(match anchor_loss_sum {
                Some(value) => value + anchor_loss,
                None => anchor_loss,
            });
        }

        total_loss = Some(match total_loss {
            Some(value) => value + step_loss,
            None => step_loss,
        });
        policy_loss_sum = Some(match policy_loss_sum {
            Some(value) => value + policy_loss,
            None => policy_loss,
        });
        value_loss_sum = Some(match value_loss_sum {
            Some(value) => value + value_loss,
            None => value_loss,
        });
        entropy_sum = Some(match entropy_sum {
            Some(value) => value + entropy,
            None => entropy,
        });
        approx_kl_sum = Some(match approx_kl_sum {
            Some(value) => value + approx_kl,
            None => approx_kl,
        });
        clip_fraction_sum = Some(match clip_fraction_sum {
            Some(value) => value + clip_fraction,
            None => clip_fraction,
        });
        memory = quantize_straight_through(output.next_memory);
    }

    let divisor = steps as f32;
    let policy_loss = policy_loss_sum.expect("a recurrent chunk is nonempty") / divisor;
    let value_loss = value_loss_sum.expect("a recurrent chunk is nonempty") / divisor;
    let entropy = entropy_sum.expect("a recurrent chunk is nonempty") / divisor;
    let approx_kl = approx_kl_sum.expect("a recurrent chunk is nonempty") / divisor;
    let anchor_loss = anchor_loss_sum
        .map(|loss| f32::from((loss / divisor).into_scalar()))
        .unwrap_or(0.0);
    let clip_fraction = clip_fraction_sum.expect("a recurrent chunk is nonempty") / divisor;
    let batch_metrics = RecurrentBatchMetrics {
        policy_loss: f32::from(policy_loss.into_scalar()),
        value_loss: f32::from(value_loss.into_scalar()),
        entropy: f32::from(entropy.into_scalar()),
        approx_kl: f32::from(approx_kl.into_scalar()),
        anchor_loss,
        clip_fraction: f32::from(clip_fraction.into_scalar()),
        samples: batch_size * steps,
    };
    let early_stopped = config
        .target_kl
        .is_some_and(|target| batch_metrics.approx_kl > target);
    if !early_stopped {
        let loss = total_loss.expect("a recurrent chunk is nonempty") / divisor;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, gradients);
    }
    (model, batch_metrics, early_stopped)
}

/// PPO training step with an optional frozen qualified-policy anchor.
pub fn ppo_update_anchored<B: AutodiffBackend>(
    model: PolicyValueNet<B>,
    optimizer: &mut impl Optimizer<PolicyValueNet<B>, B>,
    rollout: &RolloutBuffer,
    config: &PPOConfig,
    rng: &mut impl Rng,
    device: &B::Device,
) -> (PolicyValueNet<B>, PpoMetrics)
where
    f32: From<B::FloatElem>,
{
    let n = rollout.len();
    if n == 0 {
        return (model, PpoMetrics::default());
    }
    let recurrent_size = model.recurrent_size();
    assert!(
        rollout.transitions.iter().all(|transition| {
            transition.initial_policy_anchor_coeff.is_finite()
                && transition.initial_policy_anchor_coeff >= 0.0
        }),
        "PPO transition anchor coefficients must be finite and nonnegative"
    );
    let has_anchor_targets = rollout
        .transitions
        .first()
        .is_some_and(|transition| transition.anchor.is_some());
    assert!(
        rollout
            .transitions
            .iter()
            .all(|transition| transition.anchor.is_some() == has_anchor_targets),
        "anchored PPO requires a frozen target on every transition"
    );
    assert!(
        has_anchor_targets
            || rollout
                .transitions
                .iter()
                .all(|transition| transition.initial_policy_anchor_coeff == 0.0),
        "positive transition anchor coefficients require frozen targets"
    );
    assert!(
        rollout
            .transitions
            .iter()
            .all(|transition| transition.policy_memory.len() == recurrent_size),
        "rollout recurrent state does not match the model architecture"
    );

    // Compute GAE per trajectory (correctly handles multi-env multi-cell rollouts)
    let (advantages, returns) =
        compute_gae_per_trajectory(&rollout.transitions, config.gamma, config.gae_lambda);

    // Normalize advantages
    let mean = advantages.iter().sum::<f32>() / n as f32;
    let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32).sqrt() + 1e-8;
    let advantages: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();

    let mut model = model;
    let old_values = rollout
        .transitions
        .iter()
        .map(|transition| transition.value)
        .collect::<Vec<_>>();
    let mut metrics = PpoMetrics {
        explained_variance: explained_variance(&old_values, &returns),
        recurrent_unroll_steps: config.recurrent_unroll_steps,
        ..PpoMetrics::default()
    };
    let chunks = recurrent_chunks(&rollout.transitions, config.recurrent_unroll_steps);
    metrics.recurrent_chunks = chunks.len();
    let mut samples_evaluated = 0usize;

    // PPO epochs
    'epochs: for epoch in 0..config.epochs_per_update {
        for batch in shuffled_recurrent_minibatches(&chunks, config.minibatch_size, rng) {
            let (updated, batch_metrics, early_stopped) = train_recurrent_batch(
                model,
                optimizer,
                rollout,
                &advantages,
                &returns,
                &batch,
                config,
                device,
            );
            model = updated;
            let weight = batch_metrics.samples as f32;
            metrics.policy_loss += batch_metrics.policy_loss * weight;
            metrics.value_loss += batch_metrics.value_loss * weight;
            metrics.entropy += batch_metrics.entropy * weight;
            metrics.approx_kl += batch_metrics.approx_kl * weight;
            metrics.anchor_loss += batch_metrics.anchor_loss * weight;
            metrics.clip_fraction += batch_metrics.clip_fraction * weight;
            samples_evaluated += batch_metrics.samples;
            if early_stopped {
                metrics.early_stopped = true;
                break 'epochs;
            }
            metrics.optimizer_steps += 1;
        }
        metrics.epochs_completed = epoch + 1;
    }

    let denominator = samples_evaluated.max(1) as f32;
    metrics.policy_loss /= denominator;
    metrics.value_loss /= denominator;
    metrics.entropy /= denominator;
    metrics.approx_kl /= denominator;
    metrics.anchor_loss /= denominator;
    metrics.clip_fraction /= denominator;
    (model, metrics)
}

/// Ordinary PPO update without an initial-policy anchor.
pub fn ppo_update<B: AutodiffBackend>(
    model: PolicyValueNet<B>,
    optimizer: &mut impl Optimizer<PolicyValueNet<B>, B>,
    rollout: &RolloutBuffer,
    config: &PPOConfig,
    rng: &mut impl Rng,
    device: &B::Device,
) -> (PolicyValueNet<B>, PpoMetrics)
where
    f32: From<B::FloatElem>,
{
    ppo_update_anchored(model, optimizer, rollout, config, rng, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::NUM_ACTIONS;
    use burn::backend::{Autodiff, NdArray};
    use burn::optim::AdamWConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn transition(trajectory_id: (usize, usize), done: bool) -> Transition {
        Transition {
            observation: vec![0.0; OBS_DIM],
            policy_memory: vec![0.0; 64],
            action_mask: vec![true; NUM_ACTIONS],
            action: 0,
            amount_mask: vec![true; NUM_AMOUNT_CHOICES],
            amount: 0,
            signal_mask: vec![true; NUM_SIGNAL_CHOICES],
            signal: 0,
            signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
            signal_strength: 0,
            action_kind_exploration_floor: 0.0,
            attack_action_kind_exploration_floor: 0.0,
            reward: 1.0,
            value: 0.0,
            next_value: 0.0,
            log_prob: 0.0,
            anchor: None,
            initial_policy_anchor_coeff: 0.0,
            done,
            elapsed_time: 1.0,
            complete: true,
            trajectory_id,
        }
    }

    #[test]
    fn recurrent_chunks_split_on_limits_terminals_and_trajectory_keys() {
        let transitions = vec![
            transition((0, 1), false),
            transition((1, 1), false),
            transition((0, 1), true),
            transition((0, 1), false),
            transition((0, 1), false),
            transition((0, 1), false),
            transition((1, 1), true),
        ];

        let chunks = recurrent_chunks(&transitions, 2);
        let indices = chunks
            .iter()
            .map(|chunk| chunk.indices.clone())
            .collect::<Vec<_>>();

        assert_eq!(indices, [vec![0, 2], vec![3, 4], vec![5], vec![1, 6]]);
    }

    #[test]
    fn recurrent_minibatches_are_seeded_and_cover_each_sample_once() {
        let chunks = (0..11)
            .map(|index| RecurrentChunk {
                indices: vec![index],
            })
            .collect::<Vec<_>>();
        let mut left_rng = StdRng::seed_from_u64(17);
        let mut right_rng = StdRng::seed_from_u64(17);
        let left = shuffled_recurrent_minibatches(&chunks, 4, &mut left_rng);
        let right = shuffled_recurrent_minibatches(&chunks, 4, &mut right_rng);

        assert_eq!(left, right);
        let mut batch_lengths = left.iter().map(Vec::len).collect::<Vec<_>>();
        batch_lengths.sort_unstable();
        assert_eq!(batch_lengths, [3, 4, 4]);
        let mut flattened = left
            .into_iter()
            .flatten()
            .flat_map(|chunk| chunk.indices)
            .collect::<Vec<_>>();
        flattened.sort_unstable();
        assert_eq!(flattened, (0..11).collect::<Vec<_>>());
    }

    #[test]
    fn explained_variance_distinguishes_perfect_and_constant_predictions() {
        let targets = [1.0, 2.0, 4.0, 8.0];
        assert!((explained_variance(&targets, &targets) - 1.0).abs() < f32::EPSILON);
        assert!(explained_variance(&[0.0; 4], &targets).abs() < f32::EPSILON);
    }

    #[test]
    fn ppo_reconstructs_uniform_then_direct_attack_mixture() {
        let mut mask = [false; NUM_POLICY_ACTION_KINDS];
        mask[PolicyActionKind::Wait.index()] = true;
        mask[PolicyActionKind::Guard.index()] = true;
        mask[PolicyActionKind::Attack.index()] = true;
        let (fixed, scale) = behavior_kind_fixed_and_scale(&mask, 0.3, 0.5);

        assert!((scale - 0.35).abs() < 1.0e-6);
        assert!((fixed[PolicyActionKind::Wait.index()] - 0.05).abs() < 1.0e-6);
        assert!((fixed[PolicyActionKind::Guard.index()] - 0.05).abs() < 1.0e-6);
        assert!((fixed[PolicyActionKind::Attack.index()] - 0.55).abs() < 1.0e-6);
        assert!((fixed.iter().sum::<f32>() + scale - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn singleton_minibatch_preserves_the_batch_axis() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        type TestBackend = Autodiff<NdArray>;

        let device = Default::default();
        let model = crate::model::PolicyValueNetConfig::new().init::<TestBackend>(&device);
        let mut optimizer = AdamWConfig::new().init();
        let rollout = RolloutBuffer {
            transitions: vec![Transition {
                observation: vec![0.0; OBS_DIM],
                policy_memory: vec![0.0; 64],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                amount_mask: vec![true; NUM_AMOUNT_CHOICES],
                amount: 0,
                signal_mask: vec![true; NUM_SIGNAL_CHOICES],
                signal: 0,
                signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
                signal_strength: 0,
                action_kind_exploration_floor: 0.1,
                attack_action_kind_exploration_floor: 0.5,
                reward: 1.0,
                value: 0.0,
                next_value: 0.0,
                log_prob: 0.0,
                anchor: None,
                initial_policy_anchor_coeff: 0.0,
                done: true,
                elapsed_time: 1.0,
                complete: true,
                trajectory_id: (0, 0),
            }],
        };
        let config = PPOConfig {
            epochs_per_update: 1,
            minibatch_size: 128,
            target_kl: None,
            action_kind_exploration_floor: 0.1,
            ..PPOConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(9);

        let (_, metrics) = ppo_update(model, &mut optimizer, &rollout, &config, &mut rng, &device);
        assert_eq!(metrics.optimizer_steps, 1);
        assert_eq!(metrics.recurrent_unroll_steps, 16);
        assert_eq!(metrics.recurrent_chunks, 1);
        assert!(metrics.policy_loss.is_finite());
        assert!(metrics.value_loss.is_finite());
    }

    #[test]
    fn identical_frozen_policy_has_zero_functional_anchor_loss() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        type TestBackend = Autodiff<NdArray>;

        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 19);
        let model = crate::model::PolicyValueNetConfig::new().init::<TestBackend>(&device);
        let reference = model.clone();
        let mut optimizer = AdamWConfig::new().init();
        let mut rollout = RolloutBuffer::new();
        let mut transition = transition((0, 0), true);
        let reference_output = reference.forward_with_memory(
            Tensor::from_data(
                TensorData::new(transition.observation.clone(), [1, OBS_DIM]),
                &device,
            ),
            Tensor::from_data(
                TensorData::new(transition.policy_memory.clone(), [1, 64]),
                &device,
            ),
        );
        transition.anchor = Some(PolicyAnchorTarget {
            action_kind_logits: reference_output
                .action_kind_logits
                .into_data()
                .to_vec()
                .unwrap(),
            target_logits: reference_output.target_logits.into_data().to_vec().unwrap(),
            effort_logits: reference_output.effort_logits.into_data().to_vec().unwrap(),
            amount_logits: reference_output.amount_logits.into_data().to_vec().unwrap(),
            signal_logits: reference_output.signal_logits.into_data().to_vec().unwrap(),
            signal_strength_logits: reference_output
                .signal_strength_logits
                .into_data()
                .to_vec()
                .unwrap(),
            next_memory: reference_output.next_memory.into_data().to_vec().unwrap(),
        });
        transition.initial_policy_anchor_coeff = 0.1;
        rollout.push(transition);
        let config = PPOConfig {
            epochs_per_update: 1,
            minibatch_size: 1,
            target_kl: None,
            initial_policy_anchor_coeff: 0.1,
            ..PPOConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(23);

        let (_, metrics) =
            ppo_update_anchored(model, &mut optimizer, &rollout, &config, &mut rng, &device);
        assert!(metrics.anchor_loss.abs() < 1e-6, "{metrics:?}");
        assert_eq!(metrics.optimizer_steps, 1);
    }

    #[test]
    fn recurrent_update_uses_recorded_memory_only_at_chunk_boundaries() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        type TestBackend = Autodiff<NdArray>;

        let mut left = RolloutBuffer::new();
        left.push(transition((0, 0), false));
        let mut second = transition((0, 0), true);
        second.action = 1;
        second.reward = -1.0;
        second.policy_memory = vec![-1.0; 64];
        left.push(second.clone());
        let mut right = RolloutBuffer::new();
        right.push(transition((0, 0), false));
        second.policy_memory = vec![1.0; 64];
        right.push(second);

        let update = |rollout: &RolloutBuffer, unroll_steps| {
            let device = Default::default();
            <TestBackend as Backend>::seed(&device, 99);
            let model = crate::model::PolicyValueNetConfig::new().init::<TestBackend>(&device);
            let mut optimizer = AdamWConfig::new().init();
            let config = PPOConfig {
                epochs_per_update: 1,
                minibatch_size: 2,
                recurrent_unroll_steps: unroll_steps,
                target_kl: None,
                ..PPOConfig::default()
            };
            let mut rng = StdRng::seed_from_u64(7);
            ppo_update(model, &mut optimizer, rollout, &config, &mut rng, &device).1
        };

        let left_unrolled = update(&left, 2);
        let right_unrolled = update(&right, 2);
        assert_eq!(left_unrolled, right_unrolled);

        let left_boundary = update(&left, 1);
        let right_boundary = update(&right, 1);
        assert_ne!(left_boundary, right_boundary);
    }

    #[test]
    fn test_gae_computation() {
        let transitions = vec![
            Transition {
                observation: vec![],
                policy_memory: vec![],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                amount_mask: vec![true; NUM_AMOUNT_CHOICES],
                amount: 0,
                signal_mask: vec![true; NUM_SIGNAL_CHOICES],
                signal: 0,
                signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
                signal_strength: 0,
                action_kind_exploration_floor: 0.0,
                attack_action_kind_exploration_floor: 0.0,
                reward: 1.0,
                value: 0.5,
                next_value: 1.0,
                log_prob: 0.0,
                anchor: None,
                initial_policy_anchor_coeff: 0.0,
                done: false,
                elapsed_time: 1.0,
                complete: true,
                trajectory_id: (0, 0),
            },
            Transition {
                observation: vec![],
                policy_memory: vec![],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                amount_mask: vec![true; NUM_AMOUNT_CHOICES],
                amount: 0,
                signal_mask: vec![true; NUM_SIGNAL_CHOICES],
                signal: 0,
                signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
                signal_strength: 0,
                action_kind_exploration_floor: 0.0,
                attack_action_kind_exploration_floor: 0.0,
                reward: 2.0,
                value: 1.0,
                next_value: 1.5,
                log_prob: 0.0,
                anchor: None,
                initial_policy_anchor_coeff: 0.0,
                done: false,
                elapsed_time: 1.0,
                complete: true,
                trajectory_id: (0, 0),
            },
            Transition {
                observation: vec![],
                policy_memory: vec![],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                amount_mask: vec![true; NUM_AMOUNT_CHOICES],
                amount: 0,
                signal_mask: vec![true; NUM_SIGNAL_CHOICES],
                signal: 0,
                signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
                signal_strength: 0,
                action_kind_exploration_floor: 0.0,
                attack_action_kind_exploration_floor: 0.0,
                reward: 3.0,
                value: 1.5,
                next_value: 0.0,
                log_prob: 0.0,
                anchor: None,
                initial_policy_anchor_coeff: 0.0,
                done: true,
                elapsed_time: 1.0,
                complete: true,
                trajectory_id: (0, 0),
            },
        ];
        let (advantages, returns) = compute_gae_per_trajectory(&transitions, 0.99, 0.95);

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);
        // Last step is terminal: advantage = reward + 0 - value = 3 - 1.5 = 1.5
        assert!((advantages[2] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_rollout_buffer() {
        let mut buffer = RolloutBuffer::new();
        assert_eq!(buffer.len(), 0);
        buffer.push(Transition {
            observation: vec![0.0; OBS_DIM],
            policy_memory: vec![0.0; 64],
            action_mask: vec![true; NUM_ACTIONS],
            action: 0,
            amount_mask: vec![true; NUM_AMOUNT_CHOICES],
            amount: 0,
            signal_mask: vec![true; NUM_SIGNAL_CHOICES],
            signal: 0,
            signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
            signal_strength: 0,
            action_kind_exploration_floor: 0.0,
            attack_action_kind_exploration_floor: 0.0,
            reward: 1.0,
            value: 0.5,
            next_value: 0.0,
            log_prob: -0.5,
            anchor: None,
            initial_policy_anchor_coeff: 0.0,
            done: false,
            elapsed_time: 1.0,
            complete: false,
            trajectory_id: (0, 0),
        });
        assert_eq!(buffer.len(), 1);
        buffer.retain_complete();
        assert!(buffer.is_empty());
        buffer.push(Transition {
            observation: vec![0.0; OBS_DIM],
            policy_memory: vec![0.0; 64],
            action_mask: vec![true; NUM_ACTIONS],
            action: 0,
            amount_mask: vec![true; NUM_AMOUNT_CHOICES],
            amount: 0,
            signal_mask: vec![true; NUM_SIGNAL_CHOICES],
            signal: 0,
            signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
            signal_strength: 0,
            action_kind_exploration_floor: 0.0,
            attack_action_kind_exploration_floor: 0.0,
            reward: 1.0,
            value: 0.5,
            next_value: 0.0,
            log_prob: -0.5,
            anchor: None,
            initial_policy_anchor_coeff: 0.0,
            done: true,
            elapsed_time: 2.0,
            complete: true,
            trajectory_id: (0, 0),
        });
        buffer.clear();
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn gae_discounts_by_simulated_time_and_uses_explicit_bootstrap() {
        let transitions = vec![Transition {
            observation: vec![],
            policy_memory: vec![],
            action_mask: vec![true; NUM_ACTIONS],
            action: 0,
            amount_mask: vec![true; NUM_AMOUNT_CHOICES],
            amount: 0,
            signal_mask: vec![true; NUM_SIGNAL_CHOICES],
            signal: 0,
            signal_strength_mask: vec![true; NUM_SIGNAL_STRENGTH_CHOICES],
            signal_strength: 0,
            action_kind_exploration_floor: 0.0,
            attack_action_kind_exploration_floor: 0.0,
            reward: 1.0,
            value: 2.0,
            next_value: 10.0,
            log_prob: 0.0,
            anchor: None,
            initial_policy_anchor_coeff: 0.0,
            done: false,
            elapsed_time: 2.0,
            complete: true,
            trajectory_id: (0, 0),
        }];
        let (advantages, returns) = compute_gae_per_trajectory(&transitions, 0.9, 0.95);
        let expected = 1.0 + 0.9_f32.powf(2.0) * 10.0 - 2.0;
        assert!((advantages[0] - expected).abs() < 1.0e-6);
        assert!((returns[0] - (expected + 2.0)).abs() < 1.0e-6);
    }
}
