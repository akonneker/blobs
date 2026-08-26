//! Training configuration — TOML file with CLI overrides.

use blob_engine::resolution::{BoundaryRule, NeighborhoodSpec, ReferenceRuleset};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::telemetry::TelemetryConfig;
use std::fmt;

/// Reproducible native baseline used for training or held-out evaluation.
/// Every profile runs through the same anonymous reference Mind input as a
/// submitted policy; these names describe behavior, not extra capabilities.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum OpponentProfile {
    Wait,
    Random,
    Forager,
    #[default]
    Aggressive,
}

impl OpponentProfile {
    pub const DEFAULT_EVALUATION: [Self; 3] = [Self::Wait, Self::Random, Self::Aggressive];
    pub const ALL: [Self; 4] = [Self::Wait, Self::Random, Self::Forager, Self::Aggressive];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wait => "wait",
            Self::Random => "random",
            Self::Forager => "forager",
            Self::Aggressive => "aggressive",
        }
    }

    pub const fn can_attack(self) -> bool {
        matches!(self, Self::Random | Self::Aggressive)
    }
}

impl fmt::Display for OpponentProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ImmediateVictoryCondition {
    #[default]
    Extermination,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DeadlineAdjudication {
    #[default]
    Draw,
}

/// Training-only interpretation of a canonical deadline draw. This never
/// changes `VictoryConfig`, `EpisodeOutcome`, replay semantics, or a published
/// match result; it only selects an optional terminal reward during rollout
/// collection.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DeadlineRewardMode {
    #[default]
    None,
    WeightedCellEnergy,
}

pub const DEADLINE_REWARD_WEIGHT_SCALE: u32 = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct DeadlineRewardConfig {
    #[serde(default)]
    pub mode: DeadlineRewardMode,
    #[serde(default)]
    pub core_basis_points: u32,
    #[serde(default)]
    pub assimilated_basis_points: u32,
    #[serde(default)]
    pub gut_basis_points: u32,
    #[serde(default)]
    pub carried_material_basis_points: u32,
    #[serde(default)]
    pub payload_escrow_basis_points: u32,
    #[serde(default)]
    pub minimum_margin_mass_energy: u64,
}

impl DeadlineRewardConfig {
    pub fn weights(&self) -> [u32; 5] {
        [
            self.core_basis_points,
            self.assimilated_basis_points,
            self.gut_basis_points,
            self.carried_material_basis_points,
            self.payload_escrow_basis_points,
        ]
    }

    pub fn score(&self, compartments: [u64; 5]) -> u128 {
        compartments
            .into_iter()
            .zip(self.weights())
            .map(|(amount, weight)| u128::from(amount) * u128::from(weight))
            .sum()
    }

    pub fn minimum_margin_score(&self) -> u128 {
        u128::from(self.minimum_margin_mass_energy) * u128::from(DEADLINE_REWARD_WEIGHT_SCALE)
    }

    fn validate(&self, maximum_cells: usize) -> Result<(), String> {
        let weights = self.weights();
        if weights
            .into_iter()
            .any(|weight| weight > DEADLINE_REWARD_WEIGHT_SCALE)
        {
            return Err(format!(
                "deadline reward weights must be in 0..={DEADLINE_REWARD_WEIGHT_SCALE} basis points"
            ));
        }
        match self.mode {
            DeadlineRewardMode::None => {
                if weights.into_iter().any(|weight| weight != 0)
                    || self.minimum_margin_mass_energy != 0
                {
                    return Err(
                        "deadline reward mode none cannot retain weights or a margin".into(),
                    );
                }
            }
            DeadlineRewardMode::WeightedCellEnergy => {
                if weights.into_iter().all(|weight| weight == 0) {
                    return Err(
                        "weighted_cell_energy deadline reward requires a positive weight".into(),
                    );
                }
                let maximum_cell_score = weights.into_iter().try_fold(0u128, |sum, weight| {
                    u128::from(u64::MAX)
                        .checked_mul(u128::from(weight))
                        .and_then(|term| sum.checked_add(term))
                });
                let bounded = maximum_cell_score
                    .and_then(|score| score.checked_mul(u128::try_from(maximum_cells).ok()?));
                if bounded.is_none() {
                    return Err(
                        "world is too large for exact weighted deadline reward accumulation".into(),
                    );
                }
            }
        }
        Ok(())
    }
}

impl Default for DeadlineRewardConfig {
    fn default() -> Self {
        Self {
            mode: DeadlineRewardMode::None,
            core_basis_points: 0,
            assimilated_basis_points: 0,
            gut_basis_points: 0,
            carried_material_basis_points: 0,
            payload_escrow_basis_points: 0,
            minimum_margin_mass_energy: 0,
        }
    }
}

/// Public, hash-bound match objective. Alternative adjudications will extend
/// this boundary only after their primitive terminal metrics have been swept.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct VictoryConfig {
    #[serde(default)]
    pub immediate: ImmediateVictoryCondition,
    #[serde(default)]
    pub deadline: DeadlineAdjudication,
    /// Canonical event-time deadline. Termination is observed at the first
    /// resolved event frontier at or beyond this value, so the reported final
    /// time may overshoot the boundary by one event frontier.
    #[serde(default = "default_sim_time_limit_quanta")]
    pub sim_time_limit_quanta: u64,
}

impl Default for VictoryConfig {
    fn default() -> Self {
        Self {
            immediate: ImmediateVictoryCondition::default(),
            deadline: DeadlineAdjudication::default(),
            sim_time_limit_quanta: default_sim_time_limit_quanta(),
        }
    }
}

/// Top-level training config, loadable from TOML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingConfig {
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[serde(default = "default_num_envs")]
    pub num_envs: usize,

    #[serde(default = "default_rollout_length")]
    pub rollout_length: u64,

    #[serde(default = "default_total_timesteps")]
    /// Hard ceiling on sampled cell actions. When
    /// `total_simulation_quanta_per_env` is set, this is a compute-safety cap
    /// rather than the scientific training budget.
    pub total_timesteps: u64,

    /// Authoritative cumulative simulation-time exposure required from every
    /// parallel environment. `None` preserves the legacy action-budget mode.
    #[serde(default)]
    pub total_simulation_quanta_per_env: Option<u64>,

    #[serde(default = "default_eval_interval")]
    pub eval_interval: usize,

    #[serde(default = "default_eval_episodes")]
    pub eval_episodes: usize,

    /// Baselines evaluated independently on the same held-out seeds.
    #[serde(default = "default_evaluation_opponents")]
    pub evaluation_opponents: Vec<OpponentProfile>,

    /// Immutable checkpoint directories used as held-out policy opponents.
    /// Each artifact is integrity-checked and loaded once at startup.
    #[serde(default)]
    pub evaluation_snapshots: Vec<String>,

    /// First seed in the fixed held-out evaluation suite. Evaluation episode
    /// `i` uses `evaluation_seed + i`.
    #[serde(default = "default_evaluation_seed")]
    pub evaluation_seed: u64,

    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,

    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: String,

    /// Optional immutable behavior-cloning artifact used only to initialize a
    /// fresh run. Exact resume restores the checkpointed model instead.
    #[serde(default)]
    pub initial_policy: Option<InitialPolicyConfig>,

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

    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InitialPolicyConfig {
    pub directory: String,
    /// SHA-256 of the exact `behavior-cloning.json` bytes.
    pub artifact_sha256: String,
}

/// PPO hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PPOConfig {
    #[serde(default = "default_clip_epsilon")]
    pub clip_epsilon: f32,

    #[serde(default = "default_value_clip_epsilon")]
    pub value_clip_epsilon: f32,

    #[serde(default = "default_gamma")]
    pub gamma: f32,

    #[serde(default = "default_gae_lambda")]
    pub gae_lambda: f32,

    #[serde(default = "default_epochs_per_update")]
    pub epochs_per_update: usize,

    #[serde(default = "default_minibatch_size")]
    pub minibatch_size: usize,

    /// Maximum consecutive decisions by one cell in a recurrent PPO graph.
    /// Terminal transitions always end a sequence earlier.
    #[serde(default = "default_recurrent_unroll_steps")]
    pub recurrent_unroll_steps: usize,

    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    #[serde(default = "default_entropy_coeff")]
    pub entropy_coeff: f32,

    #[serde(default = "default_value_loss_coeff")]
    pub value_loss_coeff: f32,

    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,

    /// Stop the remaining PPO epochs when the sampled policy divergence is
    /// already too large. Set to `None` to disable early stopping.
    #[serde(default = "default_target_kl")]
    pub target_kl: Option<f32>,
}

/// Model architecture config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    #[serde(default = "default_hidden1")]
    pub hidden1: usize,

    #[serde(default = "default_hidden2")]
    pub hidden2: usize,

    /// Number of learned values stored independently in each cell's canonical
    /// private Mind memory using deterministic signed-16-bit quantization.
    #[serde(default = "default_recurrent_size")]
    pub recurrent_size: usize,
}

/// Environment config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvConfig {
    #[serde(default = "default_world_size")]
    pub world_size: usize,

    #[serde(default = "default_cells_per_team")]
    pub cells_per_team: usize,

    #[serde(default = "default_max_episode_len")]
    /// Host safety ceiling measured in team-zero decision frontiers. Scientific
    /// matches must normally terminate through `victory`, not this limit.
    pub max_episode_len: u64,

    #[serde(default)]
    pub victory: VictoryConfig,

    #[serde(default = "default_num_teams")]
    pub num_teams: usize,

    /// Fixed baseline used by non-training teams during rollout collection.
    #[serde(default)]
    pub opponent: OpponentProfile,

    /// Canonical action, timing, metabolism, and transport semantics. This is
    /// serialized independently from reward shaping and hash-bound by the
    /// reference resolver.
    #[serde(default = "default_reference_ruleset")]
    pub rules: ReferenceRuleset,

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

/// Initial world and population parameters, distinct from resolver semantics.
/// Sweep variants may change this profile while reward, optimizer, and
/// opponent-curriculum settings remain fixed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioProfile {
    pub world_size: usize,
    pub cells_per_team: usize,
    pub victory: VictoryConfig,
    pub num_teams: usize,
    pub min_energy: u32,
    pub initial_energy: u32,
    pub num_scattered_energy: usize,
    pub scattered_energy_amount: u32,
    pub num_plants: usize,
    pub plant_rate: u32,
    pub plant_max_energy: u32,
}

impl From<&EnvConfig> for ScenarioProfile {
    fn from(env: &EnvConfig) -> Self {
        Self {
            world_size: env.world_size,
            cells_per_team: env.cells_per_team,
            victory: env.victory.clone(),
            num_teams: env.num_teams,
            min_energy: env.min_energy,
            initial_energy: env.initial_energy,
            num_scattered_energy: env.num_scattered_energy,
            scattered_energy_amount: env.scattered_energy_amount,
            num_plants: env.num_plants,
            plant_rate: env.plant_rate,
            plant_max_energy: env.plant_max_energy,
        }
    }
}

impl ScenarioProfile {
    pub fn semantic_hash(&self) -> Result<String, String> {
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("Failed to encode scenario profile: {error}"))?;
        let digest = Sha256::digest(encoded);
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    fn apply_to(&self, env: &mut EnvConfig) {
        env.world_size = self.world_size;
        env.cells_per_team = self.cells_per_team;
        env.victory = self.victory.clone();
        env.num_teams = self.num_teams;
        env.min_energy = self.min_energy;
        env.initial_energy = self.initial_energy;
        env.num_scattered_energy = self.num_scattered_energy;
        env.scattered_energy_amount = self.scattered_energy_amount;
        env.num_plants = self.num_plants;
        env.plant_rate = self.plant_rate;
        env.plant_max_energy = self.plant_max_energy;
    }
}

/// Reward shaping config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

    /// Optional rollout-only reward at a canonical deadline draw. The public
    /// match objective remains `env.victory.deadline = "draw"`.
    #[serde(default)]
    pub deadline: DeadlineRewardConfig,
}

/// Self-play config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelfPlayConfig {
    /// First completed training-action count eligible for pool promotion.
    #[serde(default = "default_self_play_start")]
    pub start_after_timesteps: u64,

    /// PPO update cadence for promotion evaluation. A zero-sized pool disables
    /// self-play entirely; otherwise this interval must be positive.
    #[serde(default = "default_opponent_update_interval")]
    pub opponent_update_interval: usize,

    #[serde(default = "default_max_opponent_pool")]
    pub max_opponent_pool: usize,

    #[serde(default = "default_baseline_probability")]
    pub baseline_probability: f64,

    #[serde(default = "default_min_pool_win_rate")]
    pub min_pool_win_rate: f64,

    #[serde(default = "default_max_fixed_regression")]
    pub max_fixed_worst_case_regression: f64,

    #[serde(default = "default_initial_rating")]
    pub initial_rating: f64,

    #[serde(default = "default_elo_k_factor")]
    pub elo_k_factor: f64,

    /// Rating scale used by rollout-selection softmax.
    #[serde(default = "default_rating_temperature")]
    pub rating_temperature: f64,

    /// Strength of inverse exposure weighting. Zero selects by rating only.
    #[serde(default = "default_exposure_exponent")]
    pub exposure_exponent: f64,
}

// ─── Defaults ───

fn default_num_envs() -> usize {
    16
}
fn default_seed() -> u64 {
    42
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
fn default_eval_episodes() -> usize {
    8
}
fn default_evaluation_opponents() -> Vec<OpponentProfile> {
    OpponentProfile::DEFAULT_EVALUATION.to_vec()
}
fn default_evaluation_seed() -> u64 {
    1_000_000_000
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
fn default_value_clip_epsilon() -> f32 {
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
fn default_recurrent_unroll_steps() -> usize {
    16
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
fn default_target_kl() -> Option<f32> {
    Some(0.03)
}

fn default_hidden1() -> usize {
    128
}
fn default_hidden2() -> usize {
    64
}
fn default_recurrent_size() -> usize {
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
fn default_sim_time_limit_quanta() -> u64 {
    1_048_576
}
fn default_reference_ruleset() -> ReferenceRuleset {
    ReferenceRuleset {
        // Existing hosted games use a toroidal board. Keeping that choice here
        // makes it explicit instead of allowing Engine to rewrite callers.
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
        ..ReferenceRuleset::default()
    }
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
fn default_baseline_probability() -> f64 {
    0.2
}
fn default_min_pool_win_rate() -> f64 {
    0.4
}
fn default_max_fixed_regression() -> f64 {
    0.05
}
fn default_initial_rating() -> f64 {
    1_000.0
}
fn default_elo_k_factor() -> f64 {
    32.0
}
fn default_rating_temperature() -> f64 {
    200.0
}
fn default_exposure_exponent() -> f64 {
    0.5
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
            value_clip_epsilon: default_value_clip_epsilon(),
            gamma: default_gamma(),
            gae_lambda: default_gae_lambda(),
            epochs_per_update: default_epochs_per_update(),
            minibatch_size: default_minibatch_size(),
            recurrent_unroll_steps: default_recurrent_unroll_steps(),
            learning_rate: default_learning_rate(),
            entropy_coeff: default_entropy_coeff(),
            value_loss_coeff: default_value_loss_coeff(),
            max_grad_norm: default_max_grad_norm(),
            target_kl: default_target_kl(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            hidden1: default_hidden1(),
            hidden2: default_hidden2(),
            recurrent_size: default_recurrent_size(),
        }
    }
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            world_size: default_world_size(),
            cells_per_team: default_cells_per_team(),
            max_episode_len: default_max_episode_len(),
            victory: VictoryConfig::default(),
            num_teams: default_num_teams(),
            opponent: OpponentProfile::default(),
            rules: default_reference_ruleset(),
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
            deadline: DeadlineRewardConfig::default(),
        }
    }
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            start_after_timesteps: default_self_play_start(),
            opponent_update_interval: default_opponent_update_interval(),
            max_opponent_pool: default_max_opponent_pool(),
            baseline_probability: default_baseline_probability(),
            min_pool_win_rate: default_min_pool_win_rate(),
            max_fixed_worst_case_regression: default_max_fixed_regression(),
            initial_rating: default_initial_rating(),
            elo_k_factor: default_elo_k_factor(),
            rating_temperature: default_rating_temperature(),
            exposure_exponent: default_exposure_exponent(),
        }
    }
}

impl TrainingConfig {
    /// Parse a training config while applying the hosted rules profile as the
    /// base for partial `[env.rules]` overrides. Checkpoint JSON always stores
    /// the fully expanded profile, so this merge is needed only for TOML input.
    pub fn from_toml_str(content: &str) -> Result<Self, String> {
        let mut value = content
            .parse::<toml::Value>()
            .map_err(|error| format!("Failed to parse config: {error}"))?;
        if let Some(rules) = value
            .get_mut("env")
            .and_then(toml::Value::as_table_mut)
            .and_then(|env| env.get_mut("rules"))
        {
            let defaults = toml::Value::try_from(default_reference_ruleset())
                .map_err(|error| format!("Failed to encode default rules profile: {error}"))?;
            merge_missing_toml(rules, &defaults);
        }
        value
            .try_into()
            .map_err(|error| format!("Failed to parse config: {error}"))
    }

    /// Load config from a TOML file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file {}: {}", path, e))?;
        Self::from_toml_str(&content)
    }

    /// Clone this training configuration and apply a strict partial rules
    /// override. Sweep planners use this rather than patching rewards or other
    /// training dynamics through an untyped document merge.
    pub fn with_rules_override(&self, rules: &toml::Table) -> Result<Self, String> {
        let mut expanded = toml::Value::try_from(&self.env.rules)
            .map_err(|error| format!("Failed to encode base rules profile: {error}"))?;
        merge_toml_override(&mut expanded, &toml::Value::Table(rules.clone()));
        let mut config = self.clone();
        config.env.rules = expanded
            .try_into()
            .map_err(|error| format!("Failed to parse rules override: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Clone this training configuration and apply a strict partial scenario
    /// override. Only initial world/population fields are represented by
    /// `ScenarioProfile`, so a sweep cannot use this path to alter rewards,
    /// PPO settings, opponent selection, or resolver rules.
    pub fn with_scenario_override(&self, scenario: &toml::Table) -> Result<Self, String> {
        let base = ScenarioProfile::from(&self.env);
        let mut expanded = toml::Value::try_from(base)
            .map_err(|error| format!("Failed to encode base scenario profile: {error}"))?;
        merge_toml_override(&mut expanded, &toml::Value::Table(scenario.clone()));
        let expanded: ScenarioProfile = expanded
            .try_into()
            .map_err(|error| format!("Failed to parse scenario override: {error}"))?;
        let mut config = self.clone();
        expanded.apply_to(&mut config.env);
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.num_envs == 0 {
            return Err("num_envs must be positive".into());
        }
        self.telemetry.validate()?;
        if self.rollout_length == 0 || self.total_timesteps == 0 {
            return Err("rollout_length and total_timesteps must be positive".into());
        }
        if self.total_simulation_quanta_per_env == Some(0) {
            return Err("total_simulation_quanta_per_env must be positive when set".into());
        }
        if let Some(initial) = &self.initial_policy {
            if initial.directory.trim().is_empty()
                || initial.artifact_sha256.len() != 64
                || !initial
                    .artifact_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                return Err(
                    "initial_policy requires a directory and 64-character lowercase SHA-256".into(),
                );
            }
        }
        if self.eval_interval > 0 && self.eval_episodes == 0 {
            return Err("eval_episodes must be positive when evaluation is enabled".into());
        }
        if self.self_play.max_opponent_pool > 0 && self.eval_episodes == 0 {
            return Err("eval_episodes must be positive when self-play is enabled".into());
        }
        if self.eval_interval > 0
            && self.evaluation_opponents.is_empty()
            && self.evaluation_snapshots.is_empty()
        {
            return Err(
                "evaluation_opponents and evaluation_snapshots must not both be empty when evaluation is enabled"
                    .into(),
            );
        }
        if self.self_play.max_opponent_pool > 0
            && self.evaluation_opponents.is_empty()
            && self.evaluation_snapshots.is_empty()
        {
            return Err("a fixed evaluation opponent is required when self-play is enabled".into());
        }
        let unique_evaluation_opponents = self
            .evaluation_opponents
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        if unique_evaluation_opponents.len() != self.evaluation_opponents.len() {
            return Err("evaluation_opponents must not contain duplicates".into());
        }
        if self
            .evaluation_snapshots
            .iter()
            .any(|path| path.trim().is_empty())
        {
            return Err("evaluation_snapshots must not contain empty paths".into());
        }
        let unique_evaluation_snapshots = self
            .evaluation_snapshots
            .iter()
            .collect::<std::collections::HashSet<_>>();
        if unique_evaluation_snapshots.len() != self.evaluation_snapshots.len() {
            return Err("evaluation_snapshots must not contain duplicates".into());
        }
        if self.checkpoint_dir.trim().is_empty() {
            return Err("checkpoint_dir must not be empty".into());
        }
        if self.self_play.max_opponent_pool > 0 && self.self_play.opponent_update_interval == 0 {
            return Err(
                "self_play.opponent_update_interval must be positive when self-play is enabled"
                    .into(),
            );
        }
        if !((0.0..=1.0).contains(&self.self_play.baseline_probability)
            && (0.0..=1.0).contains(&self.self_play.min_pool_win_rate)
            && (0.0..=1.0).contains(&self.self_play.max_fixed_worst_case_regression))
        {
            return Err("self-play probabilities and evaluation gates must be in [0, 1]".into());
        }
        if !self.self_play.initial_rating.is_finite()
            || !self.self_play.elo_k_factor.is_finite()
            || self.self_play.elo_k_factor <= 0.0
            || !self.self_play.rating_temperature.is_finite()
            || self.self_play.rating_temperature <= 0.0
            || !self.self_play.exposure_exponent.is_finite()
            || self.self_play.exposure_exponent < 0.0
        {
            return Err("self-play rating and exposure parameters are invalid".into());
        }
        if self.model.hidden1 == 0 || self.model.hidden2 == 0 || self.model.recurrent_size == 0 {
            return Err("model hidden dimensions must be positive".into());
        }
        if self.model.recurrent_size > 4_096 {
            return Err("model recurrent dimension exceeds the supported limit of 4096".into());
        }
        let memory_bytes = crate::model::policy_memory_bytes(self.model.recurrent_size)
            .ok_or_else(|| "model recurrent memory byte size overflowed".to_string())?;
        if memory_bytes > self.env.rules.max_private_memory_bytes {
            return Err("model recurrent memory exceeds the ruleset private-memory limit".into());
        }
        if !(0.0 < self.ppo.gamma && self.ppo.gamma <= 1.0) {
            return Err("ppo.gamma must be in (0, 1]".into());
        }
        if !(0.0..=1.0).contains(&self.ppo.gae_lambda) {
            return Err("ppo.gae_lambda must be in [0, 1]".into());
        }
        if self.ppo.clip_epsilon <= 0.0
            || self.ppo.value_clip_epsilon <= 0.0
            || self.ppo.epochs_per_update == 0
            || self.ppo.minibatch_size == 0
            || self.ppo.recurrent_unroll_steps == 0
            || self.ppo.recurrent_unroll_steps > 256
            || !self.ppo.learning_rate.is_finite()
            || self.ppo.learning_rate <= 0.0
            || self.ppo.entropy_coeff < 0.0
            || self.ppo.value_loss_coeff < 0.0
            || self.ppo.max_grad_norm <= 0.0
            || self
                .ppo
                .target_kl
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(
                "PPO coefficients, epochs, batch size, recurrent unroll steps (1..=256), and learning rate are invalid".into(),
            );
        }
        if self.env.world_size == 0
            || self.env.cells_per_team == 0
            || self.env.num_teams < 2
            || self.env.max_episode_len == 0
            || self.env.victory.sim_time_limit_quanta == 0
        {
            return Err("environment dimensions, population, teams, host safety limit, or event-time deadline are invalid".into());
        }
        self.env
            .rules
            .validate_for_world(self.env.world_size, self.env.world_size)
            .map_err(|error| error.to_string())?;
        let starting_cells = self
            .env
            .cells_per_team
            .checked_mul(self.env.num_teams)
            .ok_or_else(|| "starting population overflows usize".to_string())?;
        let tiles = self
            .env
            .world_size
            .checked_mul(self.env.world_size)
            .ok_or_else(|| "world area overflows usize".to_string())?;
        if starting_cells > tiles {
            return Err("starting population exceeds world area".into());
        }
        let resource_sources = self
            .env
            .num_scattered_energy
            .checked_add(self.env.num_plants)
            .ok_or_else(|| "initial resource count overflows usize".to_string())?;
        if resource_sources > tiles {
            return Err("initial resource sources exceed world area".into());
        }
        let rewards = [
            self.reward.survive_tick,
            self.reward.eat_energy,
            self.reward.kill_enemy,
            self.reward.cell_died,
            self.reward.split_success,
            self.reward.team_wins,
            self.reward.team_loses,
            self.reward.move_toward_food,
            self.reward.move_toward_enemy,
        ];
        if rewards.iter().any(|value| !value.is_finite()) {
            return Err("reward coefficients must be finite".into());
        }
        self.reward.deadline.validate(tiles)?;
        Ok(())
    }
}

fn merge_missing_toml(target: &mut toml::Value, defaults: &toml::Value) {
    let (Some(target), Some(defaults)) = (target.as_table_mut(), defaults.as_table()) else {
        return;
    };
    for (key, default) in defaults {
        if let Some(value) = target.get_mut(key) {
            merge_missing_toml(value, default);
        } else {
            target.insert(key.clone(), default.clone());
        }
    }
}

fn merge_toml_override(target: &mut toml::Value, patch: &toml::Value) {
    match (target, patch) {
        (toml::Value::Table(target), toml::Value::Table(patch)) => {
            for (key, value) in patch {
                if let Some(target) = target.get_mut(key) {
                    merge_toml_override(target, value);
                } else {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
        (target, patch) => {
            *target = patch.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_config_rejects_invalid_rollout_and_environment_values() {
        let mut config = TrainingConfig::default();
        assert!(config.validate().is_ok());
        config.num_envs = 0;
        assert_eq!(config.validate().unwrap_err(), "num_envs must be positive");
        config.num_envs = 1;
        config.total_simulation_quanta_per_env = Some(0);
        assert_eq!(
            config.validate().unwrap_err(),
            "total_simulation_quanta_per_env must be positive when set"
        );
        config.total_simulation_quanta_per_env = None;
        config.initial_policy = Some(InitialPolicyConfig {
            directory: "artifact".into(),
            artifact_sha256: "not-a-hash".into(),
        });
        assert!(config
            .validate()
            .unwrap_err()
            .contains("initial_policy requires"));
        config.initial_policy = None;
        config.ppo.gamma = 1.1;
        assert_eq!(
            config.validate().unwrap_err(),
            "ppo.gamma must be in (0, 1]"
        );
        config.ppo.gamma = 0.99;
        config.ppo.target_kl = Some(0.0);
        assert_eq!(
            config.validate().unwrap_err(),
            "PPO coefficients, epochs, batch size, recurrent unroll steps (1..=256), and learning rate are invalid"
        );
        config.ppo.target_kl = None;
        config.ppo.recurrent_unroll_steps = 0;
        assert!(config.validate().is_err());
        config.ppo.recurrent_unroll_steps = 257;
        assert!(config.validate().is_err());
        config.ppo.recurrent_unroll_steps = 16;
        config.evaluation_opponents = vec![OpponentProfile::Wait, OpponentProfile::Wait];
        assert_eq!(
            config.validate().unwrap_err(),
            "evaluation_opponents must not contain duplicates"
        );
        config.evaluation_opponents = OpponentProfile::ALL.to_vec();
        config.evaluation_snapshots = vec![" ".into()];
        assert_eq!(
            config.validate().unwrap_err(),
            "evaluation_snapshots must not contain empty paths"
        );
        config.evaluation_snapshots.clear();
        config.self_play.baseline_probability = 1.1;
        assert_eq!(
            config.validate().unwrap_err(),
            "self-play probabilities and evaluation gates must be in [0, 1]"
        );
        config.self_play.baseline_probability = 0.2;
        config.self_play.opponent_update_interval = 0;
        assert_eq!(
            config.validate().unwrap_err(),
            "self_play.opponent_update_interval must be positive when self-play is enabled"
        );
        config.self_play.opponent_update_interval = 50;
        config.self_play.rating_temperature = 0.0;
        assert_eq!(
            config.validate().unwrap_err(),
            "self-play rating and exposure parameters are invalid"
        );
        config.self_play.rating_temperature = 200.0;
        config.env.world_size = 1;
        config.env.cells_per_team = 2;
        assert_eq!(
            config.validate().unwrap_err(),
            "starting population exceeds world area"
        );
        config.env.world_size = 2;
        config.env.cells_per_team = 1;
        config.env.num_scattered_energy = 4;
        config.env.num_plants = 1;
        assert_eq!(
            config.validate().unwrap_err(),
            "initial resource sources exceed world area"
        );
    }

    #[test]
    fn opponent_profiles_have_stable_human_readable_config_names() {
        let config = TrainingConfig::from_toml_str(
            r#"
            evaluation_opponents = ["wait", "random", "aggressive"]
            [env]
            opponent = "random"
            "#,
        )
        .unwrap();
        assert_eq!(
            config.evaluation_opponents,
            OpponentProfile::DEFAULT_EVALUATION
        );
        assert_eq!(config.env.opponent, OpponentProfile::Random);
        assert_eq!(OpponentProfile::Aggressive.to_string(), "aggressive");
        assert_eq!(OpponentProfile::Forager.to_string(), "forager");
    }

    #[test]
    fn rules_profile_supports_partial_overrides_and_rejects_typos() {
        let encoded_defaults = toml::Value::try_from(default_reference_ruleset()).unwrap();
        assert_eq!(
            encoded_defaults["neighborhood"]["boundary_rule"].as_str(),
            Some("wrap")
        );
        let config = TrainingConfig::from_toml_str(
            r#"
            [env.rules]
            bite_capacity = 23
            metabolism_rate_numerator = 2
            "#,
        )
        .unwrap();
        assert_eq!(config.env.rules.bite_capacity, 23);
        assert_eq!(config.env.rules.metabolism_rate_numerator, 2);
        assert_eq!(
            config.env.rules.move_duration,
            default_reference_ruleset().move_duration
        );
        assert_eq!(
            config.env.rules.neighborhood.boundary_rule,
            BoundaryRule::Wrap
        );
        assert!(config.validate().is_ok());
        let json = serde_json::to_vec(&config).unwrap();
        assert_eq!(
            serde_json::from_slice::<TrainingConfig>(&json).unwrap(),
            config
        );

        let typo = TrainingConfig::from_toml_str(
            r#"
            [env.rules]
            bite_capcity = 23
            "#,
        )
        .unwrap_err()
        .to_string();
        assert!(typo.contains("unknown field `bite_capcity`"));

        let mut invalid = config;
        invalid.env.rules.metabolism_rate_denominator = 0;
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("ruleset denominators must be positive"));
    }

    #[test]
    fn scenario_overrides_are_strict_and_cannot_change_training_dynamics() {
        let base = TrainingConfig::default();
        let scenario = toml::Table::from_iter([
            ("num_plants".into(), toml::Value::Integer(12)),
            ("initial_energy".into(), toml::Value::Integer(80)),
        ]);
        let changed = base.with_scenario_override(&scenario).unwrap();
        assert_eq!(changed.env.num_plants, 12);
        assert_eq!(changed.env.initial_energy, 80);
        assert_eq!(changed.env.rules, base.env.rules);
        assert_eq!(changed.reward, base.reward);
        assert_eq!(changed.ppo, base.ppo);
        assert_eq!(changed.env.opponent, base.env.opponent);
        assert_ne!(
            ScenarioProfile::from(&changed.env).semantic_hash().unwrap(),
            ScenarioProfile::from(&base.env).semantic_hash().unwrap()
        );

        let typo = toml::Table::from_iter([("initial_energi".into(), toml::Value::Integer(80))]);
        assert!(base
            .with_scenario_override(&typo)
            .unwrap_err()
            .contains("unknown field `initial_energi`"));

        let host_guard =
            toml::Table::from_iter([("max_episode_len".into(), toml::Value::Integer(32_768))]);
        assert!(base
            .with_scenario_override(&host_guard)
            .unwrap_err()
            .contains("unknown field `max_episode_len`"));
    }

    #[test]
    fn victory_objective_is_strict_and_scenario_hash_bound() {
        let base = TrainingConfig::default();
        let config = TrainingConfig::from_toml_str(
            r#"
            [env.victory]
            immediate = "extermination"
            deadline = "draw"
            sim_time_limit_quanta = 2048
            "#,
        )
        .unwrap();
        assert_eq!(config.env.victory.sim_time_limit_quanta, 2048);
        assert_ne!(
            ScenarioProfile::from(&config.env).semantic_hash().unwrap(),
            ScenarioProfile::from(&base.env).semantic_hash().unwrap()
        );

        let typo = TrainingConfig::from_toml_str(
            r#"
            [env.victory]
            sim_time_limit_quantum = 2048
            "#,
        )
        .unwrap_err();
        assert!(typo.contains("unknown field `sim_time_limit_quantum`"));
    }

    #[test]
    fn deadline_reward_is_training_only_exact_and_strict() {
        let base = TrainingConfig::default();
        let weighted = TrainingConfig::from_toml_str(
            r#"
            [reward.deadline]
            mode = "weighted_cell_energy"
            core_basis_points = 10000
            assimilated_basis_points = 10000
            gut_basis_points = 5000
            carried_material_basis_points = 0
            payload_escrow_basis_points = 0
            minimum_margin_mass_energy = 10
            "#,
        )
        .unwrap();
        assert_eq!(weighted.env.victory, base.env.victory);
        assert_ne!(weighted.reward, base.reward);
        assert_eq!(weighted.reward.deadline.score([10, 20, 4, 9, 7]), 320_000);
        assert_eq!(weighted.reward.deadline.minimum_margin_score(), 100_000);
        weighted.validate().unwrap();

        let inert = TrainingConfig::from_toml_str(
            r#"
            [reward.deadline]
            mode = "none"
            gut_basis_points = 5000
            "#,
        )
        .unwrap();
        assert!(inert
            .validate()
            .unwrap_err()
            .contains("mode none cannot retain weights"));

        let excessive = TrainingConfig::from_toml_str(
            r#"
            [reward.deadline]
            mode = "weighted_cell_energy"
            core_basis_points = 10001
            "#,
        )
        .unwrap();
        assert!(excessive.validate().unwrap_err().contains("0..=10000"));

        let typo = TrainingConfig::from_toml_str(
            r#"
            [reward.deadline]
            mode = "weighted_cell_energy"
            core_basis_points = 10000
            gut_basis_point = 5000
            "#,
        )
        .unwrap_err();
        assert!(typo.contains("unknown field `gut_basis_point`"));
    }

    #[test]
    fn documented_default_config_expands_to_a_valid_profile() {
        let path = format!("{}/config/default.toml", env!("CARGO_MANIFEST_DIR"));
        let config = TrainingConfig::from_file(&path).unwrap();
        assert!(config.validate().is_ok());
        assert_eq!(config.env.rules.bite_capacity, 16);
        assert_eq!(
            config.env.rules.neighborhood.boundary_rule,
            BoundaryRule::Wrap
        );
    }
}
