//! Training configuration — TOML file with CLI overrides.

use serde::{Deserialize, Serialize};

/// Top-level training config, loadable from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_num_envs")]
    pub num_envs: usize,

    #[serde(default = "default_rollout_length")]
    pub rollout_length: u64,

    #[serde(default = "default_total_timesteps")]
    pub total_timesteps: u64,

    #[serde(default = "default_eval_interval")]
    pub eval_interval: usize,

    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,

    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: String,

    #[serde(default)]
    pub ppo: PPOConfig,

    #[serde(default)]
    pub model: ModelConfig,

    #[serde(default)]
    pub env: EnvConfig,

    #[serde(default)]
    pub reward: RewardConfig,

    #[serde(default)]
    pub self_play: SelfPlayConfig,
}

/// PPO hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    #[serde(default = "default_clip_epsilon")]
    pub clip_epsilon: f32,

    #[serde(default = "default_gamma")]
    pub gamma: f32,

    #[serde(default = "default_gae_lambda")]
    pub gae_lambda: f32,

    #[serde(default = "default_epochs_per_update")]
    pub epochs_per_update: usize,

    #[serde(default = "default_minibatch_size")]
    pub minibatch_size: usize,

    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    #[serde(default = "default_entropy_coeff")]
    pub entropy_coeff: f32,

    #[serde(default = "default_value_loss_coeff")]
    pub value_loss_coeff: f32,

    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
}

/// Model architecture config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_hidden1")]
    pub hidden1: usize,

    #[serde(default = "default_hidden2")]
    pub hidden2: usize,
}

/// Environment config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    #[serde(default = "default_world_size")]
    pub world_size: usize,

    #[serde(default = "default_cells_per_team")]
    pub cells_per_team: usize,

    #[serde(default = "default_max_episode_len")]
    pub max_episode_len: u64,

    #[serde(default = "default_num_teams")]
    pub num_teams: usize,

    // Cell config (matching blob_engine::CellConfig)
    #[serde(default = "default_min_energy")]
    pub min_energy: u32,

    #[serde(default = "default_initial_energy")]
    pub initial_energy: u32,

    #[serde(default = "default_max_cell_energy")]
    pub max_energy: u32,

    #[serde(default = "default_min_attack_power")]
    pub min_attack_power: u32,

    #[serde(default = "default_max_attack_power")]
    pub max_attack_power: u32,

    #[serde(default = "default_max_energy_for_attack_scaling")]
    pub max_energy_for_attack_scaling: u32,

    // Energy scatter config
    #[serde(default = "default_num_scattered_energy")]
    pub num_scattered_energy: usize,

    #[serde(default = "default_scattered_energy_amount")]
    pub scattered_energy_amount: u32,

    #[serde(default = "default_num_plants")]
    pub num_plants: usize,

    #[serde(default = "default_plant_rate")]
    pub plant_rate: u32,

    #[serde(default = "default_plant_max_energy")]
    pub plant_max_energy: u32,
}

/// Reward shaping config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    #[serde(default = "default_survive_tick")]
    pub survive_tick: f32,

    #[serde(default = "default_eat_energy")]
    pub eat_energy: f32,

    #[serde(default = "default_kill_enemy")]
    pub kill_enemy: f32,

    #[serde(default = "default_cell_died")]
    pub cell_died: f32,

    #[serde(default = "default_split_success")]
    pub split_success: f32,

    #[serde(default = "default_team_wins")]
    pub team_wins: f32,

    #[serde(default = "default_team_loses")]
    pub team_loses: f32,

    // Proximity-based reward shaping (dense signals)
    #[serde(default = "default_move_toward_food")]
    pub move_toward_food: f32,

    #[serde(default = "default_move_toward_enemy")]
    pub move_toward_enemy: f32,

    #[serde(default = "default_proximity_search_radius")]
    pub proximity_search_radius: usize,
}

/// Self-play config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPlayConfig {
    #[serde(default = "default_self_play_start")]
    pub start_after_timesteps: u64,

    #[serde(default = "default_opponent_update_interval")]
    pub opponent_update_interval: usize,

    #[serde(default = "default_max_opponent_pool")]
    pub max_opponent_pool: usize,
}

// ─── Defaults ───

fn default_num_envs() -> usize {
    16
}
fn default_rollout_length() -> u64 {
    128
}
fn default_total_timesteps() -> u64 {
    1_000_000
}
fn default_eval_interval() -> usize {
    50
}
fn default_checkpoint_interval() -> usize {
    100
}
fn default_checkpoint_dir() -> String {
    "./checkpoints".to_string()
}

fn default_clip_epsilon() -> f32 {
    0.2
}
fn default_gamma() -> f32 {
    0.99
}
fn default_gae_lambda() -> f32 {
    0.95
}
fn default_epochs_per_update() -> usize {
    4
}
fn default_minibatch_size() -> usize {
    64
}
fn default_learning_rate() -> f64 {
    3e-4
}
fn default_entropy_coeff() -> f32 {
    0.01
}
fn default_value_loss_coeff() -> f32 {
    0.5
}
fn default_max_grad_norm() -> f32 {
    0.5
}

fn default_hidden1() -> usize {
    128
}
fn default_hidden2() -> usize {
    64
}

fn default_world_size() -> usize {
    64
}
fn default_cells_per_team() -> usize {
    12
}
fn default_max_episode_len() -> u64 {
    1000
}
fn default_num_teams() -> usize {
    2
}
fn default_min_energy() -> u32 {
    10
}
fn default_initial_energy() -> u32 {
    100
}
fn default_max_cell_energy() -> u32 {
    500
}
fn default_min_attack_power() -> u32 {
    5
}
fn default_max_attack_power() -> u32 {
    50
}
fn default_max_energy_for_attack_scaling() -> u32 {
    200
}
fn default_num_scattered_energy() -> usize {
    200
}
fn default_scattered_energy_amount() -> u32 {
    25
}
fn default_num_plants() -> usize {
    50
}
fn default_plant_rate() -> u32 {
    3
}
fn default_plant_max_energy() -> u32 {
    100
}

fn default_survive_tick() -> f32 {
    0.01
}
fn default_eat_energy() -> f32 {
    0.002
}
fn default_kill_enemy() -> f32 {
    1.0
}
fn default_cell_died() -> f32 {
    -1.0
}
fn default_split_success() -> f32 {
    2.0
}
fn default_team_wins() -> f32 {
    10.0
}
fn default_team_loses() -> f32 {
    -10.0
}
fn default_move_toward_food() -> f32 {
    0.05
}
fn default_move_toward_enemy() -> f32 {
    0.02
}
fn default_proximity_search_radius() -> usize {
    10
}

fn default_self_play_start() -> u64 {
    500_000
}
fn default_opponent_update_interval() -> usize {
    50
}
fn default_max_opponent_pool() -> usize {
    10
}

impl Default for TrainingConfig {
    fn default() -> Self {
        toml::from_str("").unwrap()
    }
}

impl Default for PPOConfig {
    fn default() -> Self {
        PPOConfig {
            clip_epsilon: default_clip_epsilon(),
            gamma: default_gamma(),
            gae_lambda: default_gae_lambda(),
            epochs_per_update: default_epochs_per_update(),
            minibatch_size: default_minibatch_size(),
            learning_rate: default_learning_rate(),
            entropy_coeff: default_entropy_coeff(),
            value_loss_coeff: default_value_loss_coeff(),
            max_grad_norm: default_max_grad_norm(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            hidden1: default_hidden1(),
            hidden2: default_hidden2(),
        }
    }
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            world_size: default_world_size(),
            cells_per_team: default_cells_per_team(),
            max_episode_len: default_max_episode_len(),
            num_teams: default_num_teams(),
            min_energy: default_min_energy(),
            initial_energy: default_initial_energy(),
            max_energy: default_max_cell_energy(),
            min_attack_power: default_min_attack_power(),
            max_attack_power: default_max_attack_power(),
            max_energy_for_attack_scaling: default_max_energy_for_attack_scaling(),
            num_scattered_energy: default_num_scattered_energy(),
            scattered_energy_amount: default_scattered_energy_amount(),
            num_plants: default_num_plants(),
            plant_rate: default_plant_rate(),
            plant_max_energy: default_plant_max_energy(),
        }
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            survive_tick: default_survive_tick(),
            eat_energy: default_eat_energy(),
            kill_enemy: default_kill_enemy(),
            cell_died: default_cell_died(),
            split_success: default_split_success(),
            team_wins: default_team_wins(),
            team_loses: default_team_loses(),
            move_toward_food: default_move_toward_food(),
            move_toward_enemy: default_move_toward_enemy(),
            proximity_search_radius: default_proximity_search_radius(),
        }
    }
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            start_after_timesteps: default_self_play_start(),
            opponent_update_interval: default_opponent_update_interval(),
            max_opponent_pool: default_max_opponent_pool(),
        }
    }
}

impl TrainingConfig {
    /// Load config from a TOML file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file {}: {}", path, e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))
    }
}
