//! PPO (Proximal Policy Optimization) algorithm.

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::action::NUM_ACTIONS;
use crate::config::PPOConfig;
use crate::model::PolicyValueNet;
use crate::observation::OBS_DIM;

/// A single experience transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub observation: Vec<f32>,
    pub action_mask: Vec<bool>,
    pub action: usize,
    pub reward: f32,
    pub value: f32,
    pub log_prob: f32,
    pub done: bool,
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
}

impl Default for RolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
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

            let next_value = if step + 1 < traj_len && !t.done {
                transitions[indices[step + 1]].value
            } else {
                0.0
            };

            let delta = t.reward + gamma * next_value - t.value;
            let mask = if t.done { 0.0 } else { 1.0 };
            last_gae = delta + gamma * gae_lambda * mask * last_gae;
            advantages[idx] = last_gae;
            returns[idx] = last_gae + t.value;
        }
    }

    (advantages, returns)
}

/// PPO training step: compute loss and update model.
pub fn ppo_update<B: AutodiffBackend>(
    model: PolicyValueNet<B>,
    optimizer: &mut impl Optimizer<PolicyValueNet<B>, B>,
    rollout: &RolloutBuffer,
    config: &PPOConfig,
    device: &B::Device,
) -> (PolicyValueNet<B>, f32, f32, f32)
where
    f32: From<B::FloatElem>,
{
    let n = rollout.len();
    if n == 0 {
        return (model, 0.0, 0.0, 0.0);
    }

    // Extract data from rollout
    let old_log_probs: Vec<f32> = rollout.transitions.iter().map(|t| t.log_prob).collect();
    let actions: Vec<usize> = rollout.transitions.iter().map(|t| t.action).collect();

    // Compute GAE per trajectory (correctly handles multi-env multi-cell rollouts)
    let (advantages, returns) =
        compute_gae_per_trajectory(&rollout.transitions, config.gamma, config.gae_lambda);

    // Normalize advantages
    let mean = advantages.iter().sum::<f32>() / n as f32;
    let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32).sqrt() + 1e-8;
    let advantages: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();

    // Convert to tensors
    let obs_data: Vec<f32> = rollout
        .transitions
        .iter()
        .flat_map(|t| t.observation.iter().copied())
        .collect();
    let obs_tensor = Tensor::<B, 2>::from_data(TensorData::new(obs_data, [n, OBS_DIM]), device);
    let mask_bias: Vec<f32> = rollout
        .transitions
        .iter()
        .flat_map(|transition| {
            transition
                .action_mask
                .iter()
                .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
        })
        .collect();
    let mask_bias = Tensor::<B, 2>::from_data(TensorData::new(mask_bias, [n, NUM_ACTIONS]), device);
    let advantages_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(advantages.clone(), [n]), device);
    let returns_tensor = Tensor::<B, 1>::from_data(TensorData::new(returns.clone(), [n]), device);
    let old_log_probs_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(old_log_probs.clone(), [n]), device);

    // Create action indices tensor
    let actions_i32: Vec<i32> = actions.iter().map(|&a| a as i32).collect();
    let actions_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(actions_i32, [n]), device);

    let mut model = model;
    let mut total_policy_loss = 0.0f32;
    let mut total_value_loss = 0.0f32;
    let mut total_entropy = 0.0f32;

    // PPO epochs
    for _epoch in 0..config.epochs_per_update {
        // Forward pass
        let output = model.forward(obs_tensor.clone());

        // Log probabilities via log-softmax
        let masked_logits = output.policy_logits + mask_bias.clone();
        let log_probs = burn::tensor::activation::log_softmax(masked_logits.clone(), 1);

        // Gather log probs for chosen actions
        let action_log_probs = log_probs
            .clone()
            .gather(1, actions_tensor.clone().unsqueeze_dim(1))
            .squeeze::<1>();

        // Ratio = exp(new_log_prob - old_log_prob)
        let ratio = (action_log_probs.clone() - old_log_probs_tensor.clone()).exp();

        // Clipped surrogate loss
        let surr1 = ratio.clone() * advantages_tensor.clone();
        let surr2 = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
            * advantages_tensor.clone();
        let policy_loss = surr1.min_pair(surr2).mean().neg();

        // Value loss
        let predicted_values = output.values.squeeze::<1>();
        let value_loss = (predicted_values - returns_tensor.clone())
            .powf_scalar(2.0)
            .mean();

        // Entropy bonus (encourage exploration)
        let probs = burn::tensor::activation::softmax(masked_logits, 1);
        let entropy = -(probs.clone() * log_probs).sum_dim(1).mean();

        // Total loss
        let loss = policy_loss.clone() + value_loss.clone() * config.value_loss_coeff
            - entropy.clone() * config.entropy_coeff;

        // Track metrics
        total_policy_loss += f32::from(policy_loss.clone().into_scalar());
        total_value_loss += f32::from(value_loss.clone().into_scalar());
        total_entropy += f32::from(entropy.clone().into_scalar());

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);
    }

    let epochs = config.epochs_per_update as f32;
    (
        model,
        total_policy_loss / epochs,
        total_value_loss / epochs,
        total_entropy / epochs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gae_computation() {
        let transitions = vec![
            Transition {
                observation: vec![],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                reward: 1.0,
                value: 0.5,
                log_prob: 0.0,
                done: false,
                trajectory_id: (0, 0),
            },
            Transition {
                observation: vec![],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                reward: 2.0,
                value: 1.0,
                log_prob: 0.0,
                done: false,
                trajectory_id: (0, 0),
            },
            Transition {
                observation: vec![],
                action_mask: vec![true; NUM_ACTIONS],
                action: 0,
                reward: 3.0,
                value: 1.5,
                log_prob: 0.0,
                done: true,
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
            action_mask: vec![true; NUM_ACTIONS],
            action: 0,
            reward: 1.0,
            value: 0.5,
            log_prob: -0.5,
            done: false,
            trajectory_id: (0, 0),
        });
        assert_eq!(buffer.len(), 1);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
    }
}
