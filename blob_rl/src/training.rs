use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use std::io::Write;
use std::time::Instant;

use crate::action::NUM_ACTIONS;
use crate::config::TrainingConfig;
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::model::{PolicyValueNet, PolicyValueNetConfig};
use crate::observation::{Observation, OBS_DIM};
use crate::ppo::{ppo_update, RolloutBuffer, Transition};
use blob_interface::types::CellId;

/// Aggregated episode stats for logging.
struct EpisodeStats {
    wins: u64,
    losses: u64,
    timeouts: u64,
    total_episode_len: u64,
    total_reward: f64,
    episodes_completed: u64,
}

impl EpisodeStats {
    fn new() -> Self {
        EpisodeStats {
            wins: 0,
            losses: 0,
            timeouts: 0,
            total_episode_len: 0,
            total_reward: 0.0,
            episodes_completed: 0,
        }
    }

    fn record(&mut self, outcome: EpisodeOutcome, episode_len: u64, total_reward: f32) {
        match outcome {
            EpisodeOutcome::Win => self.wins += 1,
            EpisodeOutcome::Loss => self.losses += 1,
            EpisodeOutcome::Timeout => self.timeouts += 1,
        }
        self.total_episode_len += episode_len;
        self.total_reward += total_reward as f64;
        self.episodes_completed += 1;
    }

    fn avg_episode_len(&self) -> f64 {
        if self.episodes_completed == 0 {
            0.0
        } else {
            self.total_episode_len as f64 / self.episodes_completed as f64
        }
    }

    fn avg_reward(&self) -> f64 {
        if self.episodes_completed == 0 {
            0.0
        } else {
            self.total_reward / self.episodes_completed as f64
        }
    }

    fn win_rate(&self) -> f64 {
        if self.episodes_completed == 0 {
            0.0
        } else {
            self.wins as f64 / self.episodes_completed as f64
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Run the full training loop.
///
/// - `load_model_path`: If Some, load weights from this path to continue training.
/// - `save_model_path`: If Some, save the trained model to this path at the end.
pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    device: B::Device,
    load_model_path: Option<&str>,
    save_model_path: Option<&str>,
) where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║           blob_rl Training                          ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!(
        "║  Envs: {:>4}  Timesteps: {:>10}                    ║",
        config.num_envs, config.total_timesteps
    );
    println!(
        "║  Model: {}→{}  LR: {:.0e}                       ║",
        config.model.hidden1, config.model.hidden2, config.ppo.learning_rate
    );
    println!("╚══════════════════════════════════════════════════════╝");

    // Initialize model
    let model_config = PolicyValueNetConfig {
        hidden1: config.model.hidden1,
        hidden2: config.model.hidden2,
    };
    let mut model: PolicyValueNet<B> = if let Some(path) = load_model_path {
        println!("  Loading model from: {}", path);
        model_config
            .init(&device)
            .load_file(path, &CompactRecorder::new(), &device)
            .expect("Failed to load model")
    } else {
        model_config.init(&device)
    };
    let mut optimizer = AdamWConfig::new().init();

    // Create vectorized environments
    let mut envs: Vec<BlobEnv> = (0..config.num_envs)
        .map(|i| BlobEnv::new(config.env.clone(), config.reward.clone(), 42 + i as u64))
        .collect();

    // Get initial observations
    let mut env_observations: Vec<Vec<(CellId, Observation)>> =
        envs.iter().map(BlobEnv::get_observations).collect();

    let mut total_timesteps = 0u64;
    let mut update_count = 0usize;
    let start_time = Instant::now();
    let mut episode_stats = EpisodeStats::new();
    let mut env_episode_rewards: Vec<f32> = vec![0.0; config.num_envs];

    // Create metrics log file
    std::fs::create_dir_all(&config.checkpoint_dir).ok();
    let metrics_path = format!("{}/metrics.csv", config.checkpoint_dir);
    let mut metrics_file =
        std::fs::File::create(&metrics_path).expect("Failed to create metrics file");
    writeln!(metrics_file,
        "update,timesteps,policy_loss,value_loss,entropy,episodes,wins,losses,timeouts,win_rate,avg_ep_len,avg_reward,cells_alive"
    ).unwrap();
    println!("  Logging metrics to: {}", metrics_path);

    println!("\n  Update │ Timesteps │  Policy Loss │  Value Loss │  Entropy │  Win Rate │ Avg Len │ Cells");
    println!(
        "  ──────┼───────────┼──────────────┼─────────────┼──────────┼───────────┼─────────┼──────"
    );

    while total_timesteps < config.total_timesteps {
        let mut rollout = RolloutBuffer::new();

        // Collect rollout across all environments
        for _step in 0..config.rollout_length {
            for (env_idx, env) in envs.iter_mut().enumerate() {
                let obs_list = &env_observations[env_idx];
                if obs_list.is_empty() {
                    continue;
                }

                // Batch all observations for this env
                let obs_data: Vec<f32> = obs_list
                    .iter()
                    .flat_map(|(_, obs)| obs.data.iter().copied())
                    .collect();
                let n_cells = obs_list.len();
                if n_cells == 0 {
                    continue;
                }

                let obs_tensor = Tensor::<B, 2>::from_data(
                    TensorData::new(obs_data.clone(), [n_cells, OBS_DIM]),
                    &device,
                );

                // Forward pass (no grad needed for rollout collection)
                let obs_inner = obs_tensor.inner();

                let output = model.valid().forward(obs_inner);
                let mask_bias: Vec<f32> = obs_list
                    .iter()
                    .flat_map(|(_, observation)| {
                        observation
                            .action_mask
                            .iter()
                            .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
                    })
                    .collect();
                let mask_bias = Tensor::<B::InnerBackend, 2>::from_data(
                    TensorData::new(mask_bias, [n_cells, NUM_ACTIONS]),
                    &device,
                );
                let log_probs_all =
                    burn::tensor::activation::log_softmax(output.policy_logits + mask_bias, 1);
                let probs_all = log_probs_all.clone().exp();
                let values_vec: Vec<f32> = output
                    .values
                    .reshape([n_cells])
                    .into_data()
                    .to_vec()
                    .unwrap();

                // Sample actions from policy
                let probs_data: Vec<f32> = probs_all.into_data().to_vec().unwrap();
                let log_probs_data: Vec<f32> = log_probs_all.into_data().to_vec().unwrap();

                let mut actions: Vec<(CellId, usize)> = Vec::with_capacity(n_cells);

                for (cell_idx, (cell_id, obs)) in obs_list.iter().enumerate() {
                    let cell_probs =
                        &probs_data[cell_idx * NUM_ACTIONS..(cell_idx + 1) * NUM_ACTIONS];
                    let cell_log_probs =
                        &log_probs_data[cell_idx * NUM_ACTIONS..(cell_idx + 1) * NUM_ACTIONS];

                    // Sample action
                    let action = sample_action(cell_probs);
                    let log_prob = cell_log_probs[action];
                    let value = values_vec[cell_idx];

                    actions.push((*cell_id, action));

                    rollout.push(Transition {
                        observation: obs.to_vec(),
                        action_mask: obs.action_mask.to_vec(),
                        action,
                        reward: 0.0, // Will be filled after step
                        value,
                        log_prob,
                        done: false, // Will be filled after step
                        trajectory_id: (env_idx, cell_id.0),
                    });
                }

                // Step environment
                let step_result = env.step(&actions);
                total_timesteps += actions.len() as u64;

                // Fill in rewards and done flags for the transitions we just pushed
                let start_idx = rollout.len() - actions.len();
                for (i, (cell_id, _)) in actions.iter().enumerate() {
                    let reward = step_result.rewards.get(cell_id).copied().unwrap_or(0.0);
                    rollout.transitions[start_idx + i].reward = reward;
                    rollout.transitions[start_idx + i].done = step_result.done;
                    env_episode_rewards[env_idx] += reward;
                }

                // Update observations for next step
                if step_result.done {
                    // Record episode outcome
                    if let Some(outcome) = step_result.outcome {
                        episode_stats.record(
                            outcome,
                            step_result.episode_step,
                            env_episode_rewards[env_idx],
                        );
                    }
                    env_episode_rewards[env_idx] = 0.0;

                    let new_obs = env.reset(42 + env_idx as u64 + total_timesteps);
                    env_observations[env_idx] = new_obs;
                } else {
                    env_observations[env_idx] = step_result.observations;
                }
            }
        }

        // PPO update
        if !rollout.is_empty() {
            // Diagnostic: print reward and advantage stats for first few updates
            if update_count < 5 {
                let rewards: Vec<f32> = rollout.transitions.iter().map(|t| t.reward).collect();
                let nonzero_rewards = rewards.iter().filter(|&&r| r.abs() > 1e-6).count();
                let max_reward = rewards.iter().cloned().fold(0.0f32, f32::max);
                let min_reward = rewards.iter().cloned().fold(0.0f32, f32::min);
                let mean_reward: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
                let unique_trajs: std::collections::HashSet<(usize, usize)> = rollout
                    .transitions
                    .iter()
                    .map(|t| t.trajectory_id)
                    .collect();
                eprintln!(
                    "  [DEBUG] Rollout: {} transitions, {} trajectories",
                    rollout.len(),
                    unique_trajs.len()
                );
                eprintln!(
                    "  [DEBUG] Rewards: mean={:.6} min={:.4} max={:.4} nonzero={}/{}",
                    mean_reward,
                    min_reward,
                    max_reward,
                    nonzero_rewards,
                    rewards.len()
                );
            }

            let (updated_model, policy_loss, value_loss, entropy) =
                ppo_update(model, &mut optimizer, &rollout, &config.ppo, &device);
            model = updated_model;
            update_count += 1;

            let total_cells: usize = envs.iter().map(|e| e.get_observations().len()).sum();

            if update_count.is_multiple_of(10) || update_count <= 5 {
                println!(
                    "  {:>5} │ {:>9} │ {:>12.6} │ {:>11.6} │ {:>8.4} │ {:>8.1}% │ {:>7.1} │ {:>4}",
                    update_count,
                    total_timesteps,
                    policy_loss,
                    value_loss,
                    entropy,
                    episode_stats.win_rate() * 100.0,
                    episode_stats.avg_episode_len(),
                    total_cells,
                );
            }

            // Write metrics CSV row
            writeln!(
                metrics_file,
                "{},{},{:.6},{:.6},{:.4},{},{},{},{},{:.4},{:.1},{:.4},{}",
                update_count,
                total_timesteps,
                policy_loss,
                value_loss,
                entropy,
                episode_stats.episodes_completed,
                episode_stats.wins,
                episode_stats.losses,
                episode_stats.timeouts,
                episode_stats.win_rate(),
                episode_stats.avg_episode_len(),
                episode_stats.avg_reward(),
                total_cells,
            )
            .unwrap();

            // Reset episode stats each update
            episode_stats.reset();
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!(
        "\n  Training complete: {} timesteps in {:.1}s ({:.0} steps/sec)",
        total_timesteps,
        total_time,
        total_timesteps as f64 / total_time
    );

    // Save the trained model
    if let Some(path) = save_model_path {
        model
            .clone()
            .save_file(path, &CompactRecorder::new())
            .expect("Failed to save model");
        println!("  Model saved to: {}", path);
    }
}

/// Sample an action from a probability distribution.
fn sample_action(probs: &[f32]) -> usize {
    let rng_val: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rng_val <= cumsum {
            return i;
        }
    }
    probs.len() - 1 // fallback
}
