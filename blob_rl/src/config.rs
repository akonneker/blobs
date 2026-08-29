//! Training configuration — TOML file with CLI overrides.

use blob_engine::engine::StartingCellLayout;
use blob_engine::resolution::{BoundaryRule, NeighborhoodSpec, ReferenceRuleset};
use blob_engine::world_gen::ResourceLayout;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::micro_combat::{scenario_environment, MicroCombatScenario, MicroCombatTrainingConfig};
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
    StochasticAggressive,
    Defensive,
}

impl OpponentProfile {
    pub const DEFAULT_EVALUATION: [Self; 3] = [Self::Wait, Self::Random, Self::Aggressive];
    pub const ALL: [Self; 6] = [
        Self::Wait,
        Self::Random,
        Self::Forager,
        Self::Aggressive,
        Self::StochasticAggressive,
        Self::Defensive,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wait => "wait",
            Self::Random => "random",
            Self::Forager => "forager",
            Self::Aggressive => "aggressive",
            Self::StochasticAggressive => "stochastic_aggressive",
            Self::Defensive => "defensive",
        }
    }

    pub const fn can_attack(self) -> bool {
        matches!(
            self,
            Self::Random | Self::Aggressive | Self::StochasticAggressive
        )
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

    /// Optional staged feeding curriculum and its independent held-out
    /// promotion checks. Disabled by default so existing training semantics do
    /// not change silently.
    #[serde(default)]
    pub feeding_curriculum: FeedingCurriculumConfig,

    /// Cyclic contact practice interleaved with feeding-retention and ordinary
    /// competitive rollouts. Disabled by default.
    #[serde(default, skip_serializing_if = "combat_curriculum_is_default")]
    pub combat_curriculum: CombatCurriculumConfig,

    /// Optional host-side functional distillation from independently
    /// qualified ecology and combat checkpoints on matching curriculum
    /// stages. Disabled by default.
    #[serde(default, skip_serializing_if = "specialist_distillation_is_default")]
    pub specialist_distillation: SpecialistDistillationConfig,

    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InitialPolicyConfig {
    pub directory: String,
    /// SHA-256 of the exact `behavior-cloning.json` bytes.
    pub artifact_sha256: String,
    /// Required feeding-competency evidence when the retention gate is active.
    #[serde(default)]
    pub qualification: Option<InitialPolicyQualificationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InitialPolicyQualificationConfig {
    pub path: String,
    /// Semantic artifact hash embedded in the verified evaluation JSON.
    pub artifact_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SpecialistDistillationConfig {
    pub enabled: bool,
    /// Functional-anchor coefficient on on-food and adjacent-food decisions.
    pub ecology_coeff: f32,
    /// Functional-anchor coefficient on contact and skirmish decisions.
    pub combat_coeff: f32,
    /// When no checkpoint passes the combat gate, permit a checkpoint with at
    /// least this much resolver-attributed skirmish damage as a temporary
    /// combat teacher. `None` keeps the strict qualified-only behavior.
    pub combat_precursor_min_skirmish_damage: Option<u64>,
}

fn specialist_distillation_is_default(config: &SpecialistDistillationConfig) -> bool {
    config == &SpecialistDistillationConfig::default()
}

impl Default for SpecialistDistillationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ecology_coeff: 0.1,
            combat_coeff: 0.1,
            combat_precursor_min_skirmish_damage: None,
        }
    }
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

    /// Training-only mixture weight for a uniform draw over currently legal
    /// action kinds. Conditional target/effort distributions remain learned.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub action_kind_exploration_floor: f32,

    #[serde(default = "default_value_loss_coeff")]
    pub value_loss_coeff: f32,

    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,

    /// Stop the remaining PPO epochs when the sampled policy divergence is
    /// already too large. Set to `None` to disable early stopping.
    #[serde(default = "default_target_kl")]
    pub target_kl: Option<f32>,

    /// Functional distillation penalty against the verified initial policy.
    /// This anchors every policy head and the cell-private recurrent
    /// transition on ordinary rollout observations. Zero disables anchoring.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub initial_policy_anchor_coeff: f32,
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

    /// Pre-match spatial assembly. This affects only initial coordinates and
    /// is included in the scientific scenario identity.
    #[serde(default)]
    pub starting_cell_layout: StartingCellLayout,

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

    /// Spatial treatment for initial loose-energy sources.
    #[serde(default)]
    pub scattered_energy_layout: ResourceLayout,

    #[serde(default = "default_scattered_energy_amount")]
    pub scattered_energy_amount: u32,

    #[serde(default = "default_num_plants")]
    pub num_plants: usize,

    /// Spatial treatment for initial plant sources, independent of loose
    /// energy so their causal effects can be crossed experimentally.
    #[serde(default)]
    pub plant_layout: ResourceLayout,

    #[serde(default = "default_plant_rate")]
    pub plant_rate: u32,

    #[serde(default = "default_plant_max_energy")]
    pub plant_max_energy: u32,

    /// Initial resource placement used by explicit curriculum scenarios.
    /// This changes only episode initialization, never observations or physics.
    #[serde(default)]
    pub resource_placement: ResourcePlacement,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePlacement {
    #[default]
    Random,
    /// Symmetric scenario placement: put a plant under every team's starting
    /// cells. Useful for single-founder-on-food controls.
    OnAllCells,
    OnTrainingCells,
    AdjacentToTrainingCells,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeedingCurriculumStage {
    OnFood,
    AdjacentFood,
    Contact,
    Skirmish,
    Competitive,
}

impl fmt::Display for FeedingCurriculumStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OnFood => "on_food",
            Self::AdjacentFood => "adjacent_food",
            Self::Contact => "contact",
            Self::Skirmish => "skirmish",
            Self::Competitive => "competitive",
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FeedingPromotionGateConfig {
    pub min_on_food_episode_success_rate: f64,
    pub min_adjacent_episode_success_rate: f64,
    pub min_survival_rate: f64,
    pub min_consumed_energy_per_initial_cell: f64,
    pub evaluation_max_episode_len: u64,
    pub evaluation_sim_time_limit_quanta: u64,
}

impl Default for FeedingPromotionGateConfig {
    fn default() -> Self {
        Self {
            min_on_food_episode_success_rate: 0.95,
            min_adjacent_episode_success_rate: 0.80,
            min_survival_rate: 0.80,
            min_consumed_energy_per_initial_cell: 1.0,
            evaluation_max_episode_len: 4096,
            evaluation_sim_time_limit_quanta: 65_536,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FeedingCurriculumConfig {
    pub enabled: bool,
    /// Whether fresh rollout environments use the two prerequisite stages.
    /// Retention evaluation remains active whenever `enabled` is true.
    #[serde(default = "default_true")]
    pub rollout_stages_enabled: bool,
    /// Per-environment cumulative canonical world time at which on-food
    /// episodes stop being assigned to newly reset environments.
    pub on_food_until_sim_time_quanta_per_env: u64,
    /// Per-environment cumulative canonical world time at which adjacent-food
    /// episodes stop and ordinary competitive rollout assignment begins.
    pub adjacent_food_until_sim_time_quanta_per_env: u64,
    pub promotion: FeedingPromotionGateConfig,
}

impl Default for FeedingCurriculumConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rollout_stages_enabled: true,
            on_food_until_sim_time_quanta_per_env: 524_288,
            adjacent_food_until_sim_time_quanta_per_env: 1_572_864,
            promotion: FeedingPromotionGateConfig::default(),
        }
    }
}

impl FeedingCurriculumConfig {
    pub fn stage(&self, simulation_time_quanta: u64) -> FeedingCurriculumStage {
        if !self.enabled
            || !self.rollout_stages_enabled
            || simulation_time_quanta >= self.adjacent_food_until_sim_time_quanta_per_env
        {
            FeedingCurriculumStage::Competitive
        } else if simulation_time_quanta >= self.on_food_until_sim_time_quanta_per_env {
            FeedingCurriculumStage::AdjacentFood
        } else {
            FeedingCurriculumStage::OnFood
        }
    }

    pub fn environment_for_stage(
        &self,
        base: &EnvConfig,
        stage: FeedingCurriculumStage,
    ) -> EnvConfig {
        let mut env = base.clone();
        env.resource_placement = match stage {
            FeedingCurriculumStage::OnFood => ResourcePlacement::OnTrainingCells,
            FeedingCurriculumStage::AdjacentFood => ResourcePlacement::AdjacentToTrainingCells,
            FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish => {
                ResourcePlacement::Random
            }
            FeedingCurriculumStage::Competitive => ResourcePlacement::Random,
        };
        if stage != FeedingCurriculumStage::Competitive {
            env.opponent = OpponentProfile::Wait;
        }
        env
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CombatCurriculumConfig {
    pub enabled: bool,
    /// Repeating per-environment canonical-time cycle.
    pub cycle_sim_time_quanta_per_env: u64,
    pub on_food_sim_time_quanta_per_cycle: u64,
    pub adjacent_food_sim_time_quanta_per_cycle: u64,
    pub contact_sim_time_quanta_per_cycle: u64,
    pub skirmish_sim_time_quanta_per_cycle: u64,
    /// Rollout horizon for ecology-retention stages. Held-out feeding gates use
    /// `feeding_curriculum.promotion.evaluation_sim_time_limit_quanta`.
    pub retention_episode_sim_time_limit_quanta: u64,
    /// Rollout horizon for direct-contact curriculum episodes.
    pub contact_episode_sim_time_limit_quanta: u64,
    /// Rollout horizon for local-skirmish curriculum episodes.
    pub skirmish_episode_sim_time_limit_quanta: u64,
    /// Independent held-out direct-contact gate horizon.
    pub contact_evaluation_sim_time_limit_quanta: u64,
    /// Independent held-out local-skirmish gate horizon.
    pub skirmish_evaluation_sim_time_limit_quanta: u64,
    /// Ordered world-time offsets within each curriculum cycle that trigger a
    /// complete held-out competency evaluation. Empty preserves update-based
    /// evaluation only. A boundary is observed at the first PPO update whose
    /// minimum per-environment clock reaches or crosses the offset.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub competency_evaluation_frontiers_sim_time_quanta_per_cycle: Vec<u64>,
    /// Training-only legal-action-kind mixture used during contact episodes.
    /// Stored on each transition so PPO recomputes the exact behavior policy.
    pub contact_action_kind_exploration_floor: f32,
    /// Lower training-only legal-kind mixture for multi-cell skirmishes.
    pub skirmish_action_kind_exploration_floor: f32,
    /// Optional functional-anchor coefficient for contact-stage transitions.
    /// When omitted, the ordinary PPO initial-policy coefficient is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_initial_policy_anchor_coeff: Option<f32>,
    pub contact_cells_per_team: usize,
    pub skirmish_cells_per_team: usize,
    pub contact_initial_energies: Vec<u32>,
    pub contact_opponents: Vec<OpponentProfile>,
    /// Resolver-attributed kills required before an evaluated checkpoint may
    /// be labeled best or promoted. Scenario wins alone are insufficient.
    pub min_contact_kills_for_promotion: u64,
    pub min_skirmish_kills_for_promotion: u64,
    /// Optional balanced asymmetric 1v1/1vN practice and its independent
    /// held-out survival/elimination promotion gates.
    pub micro_combat: MicroCombatTrainingConfig,
}

fn combat_curriculum_is_default(config: &CombatCurriculumConfig) -> bool {
    config == &CombatCurriculumConfig::default()
}

impl Default for CombatCurriculumConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cycle_sim_time_quanta_per_env: 262_144,
            on_food_sim_time_quanta_per_cycle: 32_768,
            adjacent_food_sim_time_quanta_per_cycle: 32_768,
            contact_sim_time_quanta_per_cycle: 65_536,
            skirmish_sim_time_quanta_per_cycle: 65_536,
            retention_episode_sim_time_limit_quanta: 32_768,
            contact_episode_sim_time_limit_quanta: 32_768,
            skirmish_episode_sim_time_limit_quanta: 65_536,
            contact_evaluation_sim_time_limit_quanta: 32_768,
            skirmish_evaluation_sim_time_limit_quanta: 65_536,
            competency_evaluation_frontiers_sim_time_quanta_per_cycle: Vec::new(),
            contact_action_kind_exploration_floor: 0.50,
            skirmish_action_kind_exploration_floor: 0.25,
            contact_initial_policy_anchor_coeff: None,
            contact_cells_per_team: 1,
            skirmish_cells_per_team: 4,
            contact_initial_energies: vec![60, 100, 180],
            contact_opponents: vec![OpponentProfile::Aggressive, OpponentProfile::Defensive],
            min_contact_kills_for_promotion: 0,
            min_skirmish_kills_for_promotion: 0,
            micro_combat: MicroCombatTrainingConfig::default(),
        }
    }
}

impl CombatCurriculumConfig {
    /// Return the latest configured periodic world-time frontier crossed in
    /// `(previous, current]`. At most one evaluation is useful at an update
    /// boundary even when an unusually large rollout crosses several points.
    pub fn crossed_competency_evaluation_frontier(
        &self,
        previous: u64,
        current: u64,
    ) -> Option<u64> {
        if !self.enabled || current <= previous || self.cycle_sim_time_quanta_per_env == 0 {
            return None;
        }
        self.competency_evaluation_frontiers_sim_time_quanta_per_cycle
            .iter()
            .filter_map(|offset| {
                let cycle = if previous < *offset {
                    0
                } else {
                    previous
                        .saturating_sub(*offset)
                        .checked_div(self.cycle_sim_time_quanta_per_env)?
                        .saturating_add(1)
                };
                let boundary = cycle
                    .checked_mul(self.cycle_sim_time_quanta_per_env)?
                    .checked_add(*offset)?;
                (boundary > previous && boundary <= current).then_some(boundary)
            })
            .max()
    }

    pub fn stage(
        &self,
        feeding: &FeedingCurriculumConfig,
        simulation_time_quanta: u64,
    ) -> FeedingCurriculumStage {
        if !self.enabled {
            return feeding.stage(simulation_time_quanta);
        }
        let phase = simulation_time_quanta % self.cycle_sim_time_quanta_per_env;
        let adjacent_start = self.on_food_sim_time_quanta_per_cycle;
        let contact_start =
            adjacent_start.saturating_add(self.adjacent_food_sim_time_quanta_per_cycle);
        let skirmish_start = contact_start.saturating_add(self.contact_sim_time_quanta_per_cycle);
        let competitive_start =
            skirmish_start.saturating_add(self.skirmish_sim_time_quanta_per_cycle);
        if phase < adjacent_start {
            FeedingCurriculumStage::OnFood
        } else if phase < contact_start {
            FeedingCurriculumStage::AdjacentFood
        } else if phase < skirmish_start {
            FeedingCurriculumStage::Contact
        } else if phase < competitive_start {
            FeedingCurriculumStage::Skirmish
        } else {
            FeedingCurriculumStage::Competitive
        }
    }

    fn variant_index(&self, stage: FeedingCurriculumStage, simulation_time_quanta: u64) -> usize {
        let episode_limit = match stage {
            FeedingCurriculumStage::Contact => self.contact_episode_sim_time_limit_quanta,
            FeedingCurriculumStage::Skirmish => self.skirmish_episode_sim_time_limit_quanta,
            _ => return 0,
        };
        usize::try_from(simulation_time_quanta / episode_limit).unwrap_or(usize::MAX)
    }

    pub fn combat_opponent(
        &self,
        stage: FeedingCurriculumStage,
        simulation_time_quanta: u64,
    ) -> OpponentProfile {
        self.contact_opponents
            [self.variant_index(stage, simulation_time_quanta) % self.contact_opponents.len()]
    }

    pub fn combat_initial_energy(
        &self,
        stage: FeedingCurriculumStage,
        simulation_time_quanta: u64,
    ) -> u32 {
        self.contact_initial_energies[self.variant_index(stage, simulation_time_quanta)
            % self.contact_initial_energies.len()]
    }

    pub fn combat_cells_per_team(&self, stage: FeedingCurriculumStage) -> usize {
        match stage {
            FeedingCurriculumStage::Contact => self.contact_cells_per_team,
            FeedingCurriculumStage::Skirmish => self.skirmish_cells_per_team,
            _ => 0,
        }
    }

    pub fn combat_episode_limit(&self, stage: FeedingCurriculumStage) -> u64 {
        match stage {
            FeedingCurriculumStage::Contact => self.contact_episode_sim_time_limit_quanta,
            FeedingCurriculumStage::Skirmish => self.skirmish_episode_sim_time_limit_quanta,
            _ => 0,
        }
    }

    pub fn combat_evaluation_episode_limit(&self, stage: FeedingCurriculumStage) -> u64 {
        match stage {
            FeedingCurriculumStage::Contact => self.contact_evaluation_sim_time_limit_quanta,
            FeedingCurriculumStage::Skirmish => self.skirmish_evaluation_sim_time_limit_quanta,
            _ => 0,
        }
    }

    pub const fn combat_stages() -> [FeedingCurriculumStage; 2] {
        [
            FeedingCurriculumStage::Contact,
            FeedingCurriculumStage::Skirmish,
        ]
    }

    fn stage_remaining_quanta(&self, simulation_time_quanta: u64) -> u64 {
        let phase = simulation_time_quanta % self.cycle_sim_time_quanta_per_env;
        let on_food_end = self.on_food_sim_time_quanta_per_cycle;
        let adjacent_end = on_food_end.saturating_add(self.adjacent_food_sim_time_quanta_per_cycle);
        let contact_end = adjacent_end.saturating_add(self.contact_sim_time_quanta_per_cycle);
        let skirmish_end = contact_end.saturating_add(self.skirmish_sim_time_quanta_per_cycle);
        let stage_end = if phase < on_food_end {
            on_food_end
        } else if phase < adjacent_end {
            adjacent_end
        } else if phase < contact_end {
            contact_end
        } else if phase < skirmish_end {
            skirmish_end
        } else {
            self.cycle_sim_time_quanta_per_env
        };
        stage_end.saturating_sub(phase).max(1)
    }

    pub fn environment_for_stage(
        &self,
        feeding: &FeedingCurriculumConfig,
        base: &EnvConfig,
        stage: FeedingCurriculumStage,
        simulation_time_quanta: u64,
    ) -> EnvConfig {
        if !self.enabled {
            return feeding.environment_for_stage(base, stage);
        }
        let mut env = match stage {
            FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish => {
                let mut env = base.clone();
                env.cells_per_team = self.combat_cells_per_team(stage);
                env.starting_cell_layout = match stage {
                    FeedingCurriculumStage::Contact => StartingCellLayout::PairedContact,
                    FeedingCurriculumStage::Skirmish => StartingCellLayout::OpposedLines,
                    _ => unreachable!(),
                };
                env.initial_energy = self.combat_initial_energy(stage, simulation_time_quanta);
                env.num_scattered_energy = 0;
                env.num_plants = 0;
                env.resource_placement = ResourcePlacement::Random;
                env.opponent = self.combat_opponent(stage, simulation_time_quanta);
                env.victory.sim_time_limit_quanta = self.combat_episode_limit(stage);
                env
            }
            FeedingCurriculumStage::OnFood | FeedingCurriculumStage::AdjacentFood => {
                let mut env = feeding.environment_for_stage(base, stage);
                env.victory.sim_time_limit_quanta = self.retention_episode_sim_time_limit_quanta;
                env
            }
            FeedingCurriculumStage::Competitive => feeding.environment_for_stage(base, stage),
        };
        if matches!(
            stage,
            FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish
        ) {
            env.max_episode_len = env.max_episode_len.max(256);
        }
        // Stages rotate only at episode reset. Cap every new episode by the
        // remaining canonical time in its assigned stage so a long episode
        // cannot cross a boundary and silently skip later curriculum blocks.
        env.victory.sim_time_limit_quanta = env
            .victory
            .sim_time_limit_quanta
            .min(self.stage_remaining_quanta(simulation_time_quanta));
        env
    }

    pub fn evaluation_environment_for_stage(
        &self,
        feeding: &FeedingCurriculumConfig,
        base: &EnvConfig,
        stage: FeedingCurriculumStage,
        simulation_time_quanta: u64,
    ) -> EnvConfig {
        let mut env = self.environment_for_stage(feeding, base, stage, simulation_time_quanta);
        env.victory.sim_time_limit_quanta = self.combat_evaluation_episode_limit(stage);
        env
    }

    pub fn micro_combat_rollout_environment(
        &self,
        base: &EnvConfig,
        scenario: &MicroCombatScenario,
        simulation_time_quanta: u64,
    ) -> EnvConfig {
        let mut env = scenario_environment(base, scenario);
        env.victory.sim_time_limit_quanta = env
            .victory
            .sim_time_limit_quanta
            .min(self.stage_remaining_quanta(simulation_time_quanta));
        env
    }
}

/// Initial world and population parameters, distinct from resolver semantics.
/// Sweep variants may change this profile while reward, optimizer, and
/// opponent-curriculum settings remain fixed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioProfile {
    pub world_size: usize,
    pub cells_per_team: usize,
    pub starting_cell_layout: StartingCellLayout,
    pub victory: VictoryConfig,
    pub num_teams: usize,
    pub min_energy: u32,
    pub initial_energy: u32,
    pub num_scattered_energy: usize,
    pub scattered_energy_layout: ResourceLayout,
    pub scattered_energy_amount: u32,
    pub num_plants: usize,
    pub plant_layout: ResourceLayout,
    pub plant_rate: u32,
    pub plant_max_energy: u32,
    pub resource_placement: ResourcePlacement,
}

impl From<&EnvConfig> for ScenarioProfile {
    fn from(env: &EnvConfig) -> Self {
        Self {
            world_size: env.world_size,
            cells_per_team: env.cells_per_team,
            starting_cell_layout: env.starting_cell_layout,
            victory: env.victory.clone(),
            num_teams: env.num_teams,
            min_energy: env.min_energy,
            initial_energy: env.initial_energy,
            num_scattered_energy: env.num_scattered_energy,
            scattered_energy_layout: env.scattered_energy_layout.clone(),
            scattered_energy_amount: env.scattered_energy_amount,
            num_plants: env.num_plants,
            plant_layout: env.plant_layout.clone(),
            plant_rate: env.plant_rate,
            plant_max_energy: env.plant_max_energy,
            resource_placement: env.resource_placement,
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
        env.starting_cell_layout = self.starting_cell_layout;
        env.victory = self.victory.clone();
        env.num_teams = self.num_teams;
        env.min_energy = self.min_energy;
        env.initial_energy = self.initial_energy;
        env.num_scattered_energy = self.num_scattered_energy;
        env.scattered_energy_layout = self.scattered_energy_layout.clone();
        env.scattered_energy_amount = self.scattered_energy_amount;
        env.num_plants = self.num_plants;
        env.plant_layout = self.plant_layout.clone();
        env.plant_rate = self.plant_rate;
        env.plant_max_energy = self.plant_max_energy;
        env.resource_placement = self.resource_placement;
    }
}

/// Reward shaping config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewardConfig {
    #[serde(default = "default_survive_tick")]
    pub survive_tick: f32,

    #[serde(default = "default_eat_energy")]
    pub eat_energy: f32,

    /// Reward per unit of resolver-confirmed damage applied to an opposing
    /// cell. Friendly-fire and overkill damage receive no credit.
    #[serde(default)]
    pub damage_enemy: f32,

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
    /// Minimum cumulative canonical world time required in every rollout
    /// environment before snapshots may enter the opponent pool.
    #[serde(default = "default_self_play_start")]
    pub start_after_sim_time_quanta_per_env: u64,

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
fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0
}
fn default_true() -> bool {
    true
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
    2_097_152
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
            action_kind_exploration_floor: 0.0,
            value_loss_coeff: default_value_loss_coeff(),
            max_grad_norm: default_max_grad_norm(),
            target_kl: default_target_kl(),
            initial_policy_anchor_coeff: 0.0,
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
            starting_cell_layout: StartingCellLayout::default(),
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
            scattered_energy_layout: ResourceLayout::default(),
            scattered_energy_amount: default_scattered_energy_amount(),
            num_plants: default_num_plants(),
            plant_layout: ResourceLayout::default(),
            plant_rate: default_plant_rate(),
            plant_max_energy: default_plant_max_energy(),
            resource_placement: ResourcePlacement::Random,
        }
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            survive_tick: default_survive_tick(),
            eat_energy: default_eat_energy(),
            damage_enemy: 0.0,
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
            start_after_sim_time_quanta_per_env: default_self_play_start(),
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
    /// Whether this run has a configured held-out evaluation schedule. World-
    /// time competency frontiers are first-class evaluation triggers even
    /// when the legacy update-count interval is disabled.
    pub fn fixed_evaluation_enabled(&self) -> bool {
        self.eval_interval > 0
            || self.combat_curriculum.micro_combat.enabled
            || !self
                .combat_curriculum
                .competency_evaluation_frontiers_sim_time_quanta_per_cycle
                .is_empty()
    }

    pub fn rollout_stage(&self, simulation_time_quanta: u64) -> FeedingCurriculumStage {
        self.combat_curriculum
            .stage(&self.feeding_curriculum, simulation_time_quanta)
    }

    pub fn rollout_environment(
        &self,
        stage: FeedingCurriculumStage,
        simulation_time_quanta: u64,
    ) -> EnvConfig {
        self.combat_curriculum.environment_for_stage(
            &self.feeding_curriculum,
            &self.env,
            stage,
            simulation_time_quanta,
        )
    }

    pub fn rollout_baseline_opponent(
        &self,
        stage: FeedingCurriculumStage,
        simulation_time_quanta: u64,
    ) -> OpponentProfile {
        match stage {
            FeedingCurriculumStage::Competitive => self.env.opponent,
            FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish => self
                .combat_curriculum
                .combat_opponent(stage, simulation_time_quanta),
            FeedingCurriculumStage::OnFood | FeedingCurriculumStage::AdjacentFood => {
                OpponentProfile::Wait
            }
        }
    }

    pub fn rollout_action_kind_exploration_floor(&self, stage: FeedingCurriculumStage) -> f32 {
        if self.combat_curriculum.enabled {
            match stage {
                FeedingCurriculumStage::Contact => {
                    return self.combat_curriculum.contact_action_kind_exploration_floor;
                }
                FeedingCurriculumStage::Skirmish => {
                    return self
                        .combat_curriculum
                        .skirmish_action_kind_exploration_floor;
                }
                _ => {}
            }
        }
        self.ppo.action_kind_exploration_floor
    }

    /// Direct Attack mixture used only inside an explicitly assigned named
    /// micro-combat scenario. The scenario index stays host-private and is not
    /// projected into observations or canonical physics.
    pub fn rollout_attack_action_kind_exploration_floor(
        &self,
        micro_combat_scenario: Option<usize>,
    ) -> f32 {
        if micro_combat_scenario.is_some() && self.combat_curriculum.micro_combat.rollout_enabled {
            self.combat_curriculum
                .micro_combat
                .attack_action_kind_exploration_floor
        } else {
            0.0
        }
    }

    pub fn rollout_initial_policy_anchor_coeff(&self, stage: FeedingCurriculumStage) -> f32 {
        if self.combat_curriculum.enabled && stage == FeedingCurriculumStage::Contact {
            self.combat_curriculum
                .contact_initial_policy_anchor_coeff
                .unwrap_or(self.ppo.initial_policy_anchor_coeff)
        } else {
            self.ppo.initial_policy_anchor_coeff
        }
    }

    pub fn rollout_specialist_distillation_coeff(
        &self,
        stage: FeedingCurriculumStage,
    ) -> Option<f32> {
        if !self.specialist_distillation.enabled {
            return None;
        }
        match stage {
            FeedingCurriculumStage::OnFood | FeedingCurriculumStage::AdjacentFood => {
                Some(self.specialist_distillation.ecology_coeff)
            }
            FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish => {
                Some(self.specialist_distillation.combat_coeff)
            }
            FeedingCurriculumStage::Competitive => None,
        }
    }

    pub fn uses_initial_policy_anchor(&self) -> bool {
        self.ppo.initial_policy_anchor_coeff > 0.0
            || self
                .combat_curriculum
                .contact_initial_policy_anchor_coeff
                .is_some_and(|coefficient| coefficient > 0.0)
    }

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

    /// Clone this training configuration and apply a strict partial combat-
    /// curriculum override. This intentionally exposes only the curriculum
    /// profile: a cadence sweep cannot silently alter physics, rewards, PPO,
    /// model capacity, evaluation seeds, or the base competitive scenario.
    pub fn with_combat_curriculum_override(
        &self,
        combat_curriculum: &toml::Table,
    ) -> Result<Self, String> {
        let mut expanded = toml::Value::try_from(&self.combat_curriculum)
            .map_err(|error| format!("Failed to encode base combat curriculum: {error}"))?;
        merge_toml_override(
            &mut expanded,
            &toml::Value::Table(combat_curriculum.clone()),
        );
        let mut config = self.clone();
        config.combat_curriculum = expanded
            .try_into()
            .map_err(|error| format!("Failed to parse combat-curriculum override: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Clone this training configuration and apply a strict partial
    /// specialist-distillation override. Keeping this as a dedicated sweep
    /// seam prevents a retention experiment from silently changing PPO,
    /// curriculum cadence, physics, rewards, or model capacity.
    pub fn with_specialist_distillation_override(
        &self,
        specialist_distillation: &toml::Table,
    ) -> Result<Self, String> {
        let mut expanded = toml::Value::try_from(&self.specialist_distillation)
            .map_err(|error| format!("Failed to encode specialist distillation: {error}"))?;
        merge_toml_override(
            &mut expanded,
            &toml::Value::Table(specialist_distillation.clone()),
        );
        let mut config = self.clone();
        config.specialist_distillation = expanded.try_into().map_err(|error| {
            format!("Failed to parse specialist-distillation override: {error}")
        })?;
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
        let feeding = &self.feeding_curriculum;
        let gate = &feeding.promotion;
        if feeding.enabled
            && feeding.rollout_stages_enabled
            && (feeding.on_food_until_sim_time_quanta_per_env == 0
                || feeding.on_food_until_sim_time_quanta_per_env
                    >= feeding.adjacent_food_until_sim_time_quanta_per_env)
        {
            return Err(
                "feeding curriculum simulation-time frontiers must be positive and strictly ordered"
                    .into(),
            );
        }
        let feeding_rates = [
            gate.min_on_food_episode_success_rate,
            gate.min_adjacent_episode_success_rate,
            gate.min_survival_rate,
        ];
        if feeding_rates
            .iter()
            .any(|rate| !rate.is_finite() || !(0.0..=1.0).contains(rate))
            || !gate.min_consumed_energy_per_initial_cell.is_finite()
            || gate.min_consumed_energy_per_initial_cell < 0.0
            || gate.evaluation_max_episode_len == 0
            || gate.evaluation_sim_time_limit_quanta == 0
        {
            return Err("feeding promotion thresholds and horizons are invalid".into());
        }
        let combat = &self.combat_curriculum;
        let distillation = &self.specialist_distillation;
        combat.micro_combat.validate_against(&self.env)?;
        if combat.micro_combat.enabled && !combat.enabled {
            return Err("micro-combat training requires the combat curriculum".into());
        }
        if combat.enabled
            && combat.adjacent_food_sim_time_quanta_per_cycle > 0
            && self.env.starting_cell_layout == StartingCellLayout::Block
            && self.env.cells_per_team >= 9
        {
            return Err(
                "adjacent-food combat retention cannot use a block with enclosed interior cells"
                    .into(),
            );
        }
        if distillation.enabled
            && (!feeding.enabled
                || !combat.enabled
                || !distillation.ecology_coeff.is_finite()
                || distillation.ecology_coeff < 0.0
                || !distillation.combat_coeff.is_finite()
                || distillation.combat_coeff < 0.0
                || (distillation.ecology_coeff == 0.0 && distillation.combat_coeff == 0.0))
        {
            return Err(
                "specialist distillation requires feeding and combat curricula plus a positive finite coefficient"
                    .into(),
            );
        }
        if distillation
            .combat_precursor_min_skirmish_damage
            .is_some_and(|minimum| {
                !distillation.enabled || minimum == 0 || distillation.combat_coeff <= 0.0
            })
        {
            return Err(
                "combat precursor distillation requires enabled specialist distillation, a positive combat coefficient, and a positive damage threshold"
                    .into(),
            );
        }
        if !combat.enabled && combat.contact_initial_policy_anchor_coeff.is_some() {
            return Err(
                "contact-specific anchoring requires the combat curriculum to be enabled".into(),
            );
        }
        if !combat.enabled
            && (combat.min_contact_kills_for_promotion > 0
                || combat.min_skirmish_kills_for_promotion > 0)
        {
            return Err(
                "combat kill thresholds require the combat curriculum to be enabled".into(),
            );
        }
        if combat.enabled {
            let competency_frontiers_valid = combat
                .competency_evaluation_frontiers_sim_time_quanta_per_cycle
                .iter()
                .copied()
                .all(|frontier| frontier > 0 && frontier < combat.cycle_sim_time_quanta_per_env)
                && combat
                    .competency_evaluation_frontiers_sim_time_quanta_per_cycle
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]);
            let scheduled = combat
                .on_food_sim_time_quanta_per_cycle
                .checked_add(combat.adjacent_food_sim_time_quanta_per_cycle)
                .and_then(|value| value.checked_add(combat.contact_sim_time_quanta_per_cycle))
                .and_then(|value| value.checked_add(combat.skirmish_sim_time_quanta_per_cycle));
            if !feeding.enabled
                || combat.cycle_sim_time_quanta_per_env == 0
                || combat.on_food_sim_time_quanta_per_cycle == 0
                || combat.adjacent_food_sim_time_quanta_per_cycle == 0
                || combat.contact_sim_time_quanta_per_cycle == 0
                || combat.skirmish_sim_time_quanta_per_cycle == 0
                || scheduled.is_none_or(|value| value >= combat.cycle_sim_time_quanta_per_env)
                || combat.retention_episode_sim_time_limit_quanta == 0
                || combat.contact_episode_sim_time_limit_quanta == 0
                || combat.skirmish_episode_sim_time_limit_quanta == 0
                || combat.contact_evaluation_sim_time_limit_quanta == 0
                || combat.skirmish_evaluation_sim_time_limit_quanta == 0
                || !competency_frontiers_valid
                || !combat.contact_action_kind_exploration_floor.is_finite()
                || !(0.0..1.0).contains(&combat.contact_action_kind_exploration_floor)
                || !combat.skirmish_action_kind_exploration_floor.is_finite()
                || !(0.0..1.0).contains(&combat.skirmish_action_kind_exploration_floor)
                || combat
                    .contact_initial_policy_anchor_coeff
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
                || combat.contact_cells_per_team == 0
                || combat.skirmish_cells_per_team == 0
                || combat.contact_initial_energies.is_empty()
                || combat.contact_opponents.is_empty()
                || self.env.num_teams != 2
                || self.env.world_size < 2
                || combat.contact_cells_per_team
                    > (self.env.world_size / 2).saturating_mul(self.env.world_size)
                || self.env.world_size < 2
                || combat.skirmish_cells_per_team > self.env.world_size
                || combat.contact_initial_energies.iter().any(|energy| {
                    u64::from(*energy) <= self.env.rules.minimum_survival_energy
                        || *energy > self.env.max_energy
                })
                || combat.contact_opponents.iter().any(|opponent| {
                    !matches!(
                        opponent,
                        OpponentProfile::Aggressive
                            | OpponentProfile::StochasticAggressive
                            | OpponentProfile::Defensive
                    )
                })
            {
                return Err(
                    "combat curriculum cycle, contact variants, or retention settings are invalid"
                        .into(),
                );
            }
        } else if !combat
            .competency_evaluation_frontiers_sim_time_quanta_per_cycle
            .is_empty()
        {
            return Err(
                "competency evaluation frontiers require the combat curriculum to be enabled"
                    .into(),
            );
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
            if feeding.enabled && initial.qualification.is_none() {
                return Err("feeding-gated initial_policy requires qualification evidence".into());
            }
            if let Some(qualification) = &initial.qualification {
                if qualification.path.trim().is_empty()
                    || qualification.artifact_hash.len() != 64
                    || !qualification
                        .artifact_hash
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
                {
                    return Err(
                        "initial_policy qualification requires a path and 64-character lowercase artifact hash"
                            .into(),
                    );
                }
            }
        }
        if self.fixed_evaluation_enabled() && self.eval_episodes == 0 {
            return Err("eval_episodes must be positive when evaluation is enabled".into());
        }
        if self.self_play.max_opponent_pool > 0 && self.eval_episodes == 0 {
            return Err("eval_episodes must be positive when self-play is enabled".into());
        }
        if self.fixed_evaluation_enabled()
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
            || !self.ppo.action_kind_exploration_floor.is_finite()
            || !(0.0..1.0).contains(&self.ppo.action_kind_exploration_floor)
            || self.ppo.value_loss_coeff < 0.0
            || self.ppo.max_grad_norm <= 0.0
            || self
                .ppo
                .target_kl
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
            || !self.ppo.initial_policy_anchor_coeff.is_finite()
            || self.ppo.initial_policy_anchor_coeff < 0.0
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
        let layout_side = (self.env.cells_per_team as f64).sqrt().ceil() as usize;
        match self.env.starting_cell_layout {
            StartingCellLayout::Line if self.env.cells_per_team > self.env.world_size => {
                return Err("line starting layout population exceeds world width".into());
            }
            StartingCellLayout::Checkerboard
                if layout_side.saturating_mul(2).saturating_sub(1) > self.env.world_size =>
            {
                return Err("checkerboard starting layout exceeds world dimensions".into());
            }
            StartingCellLayout::Ring
                if self.env.cells_per_team > 1
                    && self
                        .env
                        .cells_per_team
                        .div_ceil(8)
                        .saturating_mul(2)
                        .saturating_add(1)
                        > self.env.world_size =>
            {
                return Err("ring starting layout exceeds world dimensions".into());
            }
            StartingCellLayout::PairedContact
                if self.env.num_teams != 2
                    || self.env.world_size < 2
                    || self.env.cells_per_team
                        > (self.env.world_size / 2).saturating_mul(self.env.world_size) =>
            {
                return Err(
                    "paired-contact starting layout requires two teams and fits paired tiles"
                        .into(),
                );
            }
            StartingCellLayout::OpposedLines
                if self.env.num_teams != 2
                    || self.env.world_size < 2
                    || self.env.cells_per_team > self.env.world_size =>
            {
                return Err(
                    "opposed-lines starting layout requires two teams and fits two adjacent lines"
                        .into(),
                );
            }
            _ => {}
        }
        let resource_sources = self
            .env
            .num_scattered_energy
            .checked_add(self.env.num_plants)
            .ok_or_else(|| "initial resource count overflows usize".to_string())?;
        if resource_sources > tiles {
            return Err("initial resource sources exceed world area".into());
        }
        self.env
            .scattered_energy_layout
            .validate_for_scenario(self.env.world_size, self.env.world_size, self.env.num_teams)
            .map_err(|error| format!("invalid loose-energy layout: {error}"))?;
        self.env
            .plant_layout
            .validate_for_scenario(self.env.world_size, self.env.world_size, self.env.num_teams)
            .map_err(|error| format!("invalid plant layout: {error}"))?;
        if self.env.resource_placement != ResourcePlacement::Random && self.env.plant_max_energy < 2
        {
            return Err(
                "curriculum resource placement requires plant_max_energy of at least 2".into(),
            );
        }
        let rewards = [
            self.reward.survive_tick,
            self.reward.eat_energy,
            self.reward.damage_enemy,
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
            qualification: None,
        });
        assert!(config
            .validate()
            .unwrap_err()
            .contains("initial_policy requires"));
        config.initial_policy = Some(InitialPolicyConfig {
            directory: "artifact".into(),
            artifact_sha256: "a".repeat(64),
            qualification: None,
        });
        config.feeding_curriculum.enabled = true;
        assert_eq!(
            config.validate().unwrap_err(),
            "feeding-gated initial_policy requires qualification evidence"
        );
        config.feeding_curriculum.enabled = false;
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
    fn feeding_curriculum_stages_are_exact_and_preserve_the_mind_boundary() {
        let curriculum = FeedingCurriculumConfig {
            enabled: true,
            on_food_until_sim_time_quanta_per_env: 10,
            adjacent_food_until_sim_time_quanta_per_env: 20,
            ..FeedingCurriculumConfig::default()
        };
        assert_eq!(curriculum.stage(0), FeedingCurriculumStage::OnFood);
        assert_eq!(curriculum.stage(9), FeedingCurriculumStage::OnFood);
        assert_eq!(curriculum.stage(10), FeedingCurriculumStage::AdjacentFood);
        assert_eq!(curriculum.stage(19), FeedingCurriculumStage::AdjacentFood);
        assert_eq!(curriculum.stage(20), FeedingCurriculumStage::Competitive);
        let competitive_transfer = FeedingCurriculumConfig {
            rollout_stages_enabled: false,
            ..curriculum.clone()
        };
        assert_eq!(
            competitive_transfer.stage(0),
            FeedingCurriculumStage::Competitive
        );

        let base = EnvConfig {
            opponent: OpponentProfile::Aggressive,
            ..EnvConfig::default()
        };
        let on_food = curriculum.environment_for_stage(&base, FeedingCurriculumStage::OnFood);
        assert_eq!(
            on_food.resource_placement,
            ResourcePlacement::OnTrainingCells
        );
        assert_eq!(on_food.opponent, OpponentProfile::Wait);
        assert_eq!(on_food.rules, base.rules);
        let competitive =
            curriculum.environment_for_stage(&base, FeedingCurriculumStage::Competitive);
        assert_eq!(competitive.resource_placement, ResourcePlacement::Random);
        assert_eq!(competitive.opponent, OpponentProfile::Aggressive);
        assert_eq!(competitive.rules, base.rules);
    }

    #[test]
    fn combat_curriculum_cycles_retention_contact_and_competitive_stages() {
        let mut config = TrainingConfig::default();
        config.feeding_curriculum.enabled = true;
        config.feeding_curriculum.rollout_stages_enabled = false;
        config.combat_curriculum.enabled = true;

        assert_eq!(config.rollout_stage(0), FeedingCurriculumStage::OnFood);
        assert_eq!(config.rollout_stage(32_767), FeedingCurriculumStage::OnFood);
        assert_eq!(
            config.rollout_stage(32_768),
            FeedingCurriculumStage::AdjacentFood
        );
        assert_eq!(
            config.rollout_stage(65_536),
            FeedingCurriculumStage::Contact
        );
        assert_eq!(
            config.rollout_stage(131_072),
            FeedingCurriculumStage::Skirmish
        );
        assert_eq!(
            config.rollout_stage(196_608),
            FeedingCurriculumStage::Competitive
        );
        assert_eq!(
            config.rollout_stage(262_144),
            FeedingCurriculumStage::OnFood
        );

        let aggressive = config.rollout_environment(FeedingCurriculumStage::Contact, 65_536);
        assert_eq!(aggressive.cells_per_team, 1);
        assert_eq!(
            aggressive.starting_cell_layout,
            StartingCellLayout::PairedContact
        );
        assert_eq!(aggressive.initial_energy, 180);
        assert_eq!(aggressive.opponent, OpponentProfile::Aggressive);
        assert_eq!(aggressive.num_plants, 0);
        assert_eq!(aggressive.num_scattered_energy, 0);
        assert_eq!(aggressive.victory.sim_time_limit_quanta, 32_768);

        let defensive = config.rollout_environment(FeedingCurriculumStage::Contact, 98_304);
        assert_eq!(defensive.initial_energy, 60);
        assert_eq!(defensive.opponent, OpponentProfile::Defensive);
        assert_eq!(
            config.rollout_baseline_opponent(FeedingCurriculumStage::Contact, 98_304),
            OpponentProfile::Defensive
        );
        assert_eq!(
            config.rollout_action_kind_exploration_floor(FeedingCurriculumStage::Contact),
            0.50
        );
        assert_eq!(
            config.rollout_action_kind_exploration_floor(FeedingCurriculumStage::Competitive),
            config.ppo.action_kind_exploration_floor
        );

        let skirmish = config.rollout_environment(FeedingCurriculumStage::Skirmish, 131_072);
        assert_eq!(skirmish.cells_per_team, 4);
        assert_eq!(
            skirmish.starting_cell_layout,
            StartingCellLayout::OpposedLines
        );
        assert_eq!(skirmish.initial_energy, 180);
        assert_eq!(skirmish.opponent, OpponentProfile::Aggressive);
        assert_eq!(skirmish.num_plants, 0);
        assert_eq!(skirmish.num_scattered_energy, 0);
        assert_eq!(skirmish.victory.sim_time_limit_quanta, 65_536);
        assert_eq!(
            config.rollout_action_kind_exploration_floor(FeedingCurriculumStage::Skirmish),
            0.25
        );
        let late_skirmish = config.rollout_environment(FeedingCurriculumStage::Skirmish, 190_000);
        assert_eq!(late_skirmish.victory.sim_time_limit_quanta, 6_608);
        let late_competitive =
            config.rollout_environment(FeedingCurriculumStage::Competitive, 200_000);
        assert_eq!(late_competitive.victory.sim_time_limit_quanta, 62_144);
    }

    #[test]
    fn combat_curriculum_episode_horizons_do_not_cross_stage_boundaries() {
        for profile in [
            "competitive_transfer_large_skirmish.toml",
            "competitive_transfer_large_skirmish_frequent.toml",
            "competitive_transfer_large_specialist_distillation.toml",
        ] {
            let path = format!("{}/config/{profile}", env!("CARGO_MANIFEST_DIR"));
            let config = TrainingConfig::from_file(&path).unwrap();
            config.validate().unwrap();
            let curriculum = &config.combat_curriculum;
            let stage_lengths = [
                curriculum.on_food_sim_time_quanta_per_cycle,
                curriculum.adjacent_food_sim_time_quanta_per_cycle,
                curriculum.contact_sim_time_quanta_per_cycle,
                curriculum.skirmish_sim_time_quanta_per_cycle,
                curriculum.cycle_sim_time_quanta_per_env.saturating_sub(
                    curriculum
                        .on_food_sim_time_quanta_per_cycle
                        .saturating_add(curriculum.adjacent_food_sim_time_quanta_per_cycle)
                        .saturating_add(curriculum.contact_sim_time_quanta_per_cycle)
                        .saturating_add(curriculum.skirmish_sim_time_quanta_per_cycle),
                ),
            ];

            for cycle in 0..2 {
                let mut stage_start = cycle * curriculum.cycle_sim_time_quanta_per_env;
                for stage_length in stage_lengths {
                    for offset in [0, 1, stage_length / 2, stage_length - 1] {
                        let time = stage_start + offset;
                        let stage = config.rollout_stage(time);
                        let environment = config.rollout_environment(stage, time);
                        let end = time + environment.victory.sim_time_limit_quanta;
                        assert!(
                            end <= stage_start + stage_length,
                            "{profile} episode at {time} crossed its {stage} boundary"
                        );
                        assert_eq!(config.rollout_stage(end - 1), stage);
                    }
                    stage_start += stage_length;
                }
            }
        }
    }

    #[test]
    fn combat_rollout_and_evaluation_horizons_are_independent() {
        let feeding = FeedingCurriculumConfig {
            enabled: true,
            ..Default::default()
        };
        let combat = CombatCurriculumConfig {
            enabled: true,
            contact_episode_sim_time_limit_quanta: 1_024,
            skirmish_episode_sim_time_limit_quanta: 2_048,
            contact_evaluation_sim_time_limit_quanta: 4_096,
            skirmish_evaluation_sim_time_limit_quanta: 8_192,
            ..Default::default()
        };
        let base = EnvConfig::default();
        let contact_start = combat
            .on_food_sim_time_quanta_per_cycle
            .saturating_add(combat.adjacent_food_sim_time_quanta_per_cycle);
        let skirmish_start = contact_start + combat.contact_sim_time_quanta_per_cycle;

        let contact_rollout = combat.environment_for_stage(
            &feeding,
            &base,
            FeedingCurriculumStage::Contact,
            contact_start,
        );
        let contact_evaluation = combat.evaluation_environment_for_stage(
            &feeding,
            &base,
            FeedingCurriculumStage::Contact,
            contact_start,
        );
        assert_eq!(contact_rollout.victory.sim_time_limit_quanta, 1_024);
        assert_eq!(contact_evaluation.victory.sim_time_limit_quanta, 4_096);

        let skirmish_rollout = combat.environment_for_stage(
            &feeding,
            &base,
            FeedingCurriculumStage::Skirmish,
            skirmish_start,
        );
        let skirmish_evaluation = combat.evaluation_environment_for_stage(
            &feeding,
            &base,
            FeedingCurriculumStage::Skirmish,
            skirmish_start,
        );
        assert_eq!(skirmish_rollout.victory.sim_time_limit_quanta, 2_048);
        assert_eq!(skirmish_evaluation.victory.sim_time_limit_quanta, 8_192);
    }

    #[test]
    fn combat_competency_frontiers_are_periodic_strict_and_world_time_based() {
        let mut config = TrainingConfig {
            eval_interval: 0,
            ..TrainingConfig::default()
        };
        config.feeding_curriculum.enabled = true;
        config.combat_curriculum.enabled = true;
        config
            .combat_curriculum
            .competency_evaluation_frontiers_sim_time_quanta_per_cycle = vec![65_536, 98_304];
        config.validate().unwrap();
        assert!(config.fixed_evaluation_enabled());

        let mut missing_suite = config.clone();
        missing_suite.eval_episodes = 0;
        assert!(missing_suite.validate().is_err());
        let mut missing_opponents = config.clone();
        missing_opponents.evaluation_opponents.clear();
        assert!(missing_opponents.validate().is_err());

        let combat = &config.combat_curriculum;
        assert_eq!(
            combat.crossed_competency_evaluation_frontier(65_535, 65_536),
            Some(65_536)
        );
        assert_eq!(
            combat.crossed_competency_evaluation_frontier(65_536, 98_400),
            Some(98_304)
        );
        assert_eq!(
            combat.crossed_competency_evaluation_frontier(98_304, 327_680),
            Some(327_680)
        );
        assert_eq!(
            combat.crossed_competency_evaluation_frontier(327_680, 327_680),
            None
        );

        config
            .combat_curriculum
            .competency_evaluation_frontiers_sim_time_quanta_per_cycle = vec![98_304, 65_536];
        assert!(config.validate().is_err());
        config
            .combat_curriculum
            .competency_evaluation_frontiers_sim_time_quanta_per_cycle = vec![262_144];
        assert!(config.validate().is_err());
        config.combat_curriculum.enabled = false;
        assert!(config.validate().is_err());
    }

    #[test]
    fn combat_curriculum_rejects_nonlocal_or_incomplete_contact_schedules() {
        let mut config = TrainingConfig::default();
        config.combat_curriculum.contact_initial_policy_anchor_coeff = Some(0.5);
        assert_eq!(
            config.validate().unwrap_err(),
            "contact-specific anchoring requires the combat curriculum to be enabled"
        );

        config.combat_curriculum.contact_initial_policy_anchor_coeff = None;
        config.combat_curriculum.min_skirmish_kills_for_promotion = 1;
        assert_eq!(
            config.validate().unwrap_err(),
            "combat kill thresholds require the combat curriculum to be enabled"
        );
        config.combat_curriculum.min_skirmish_kills_for_promotion = 0;
        config.combat_curriculum.enabled = true;
        assert!(config.validate().is_err());

        config.feeding_curriculum.enabled = true;
        config.combat_curriculum.cycle_sim_time_quanta_per_env = 196_608;
        assert!(config.validate().is_err());

        config.combat_curriculum.cycle_sim_time_quanta_per_env = 262_144;
        config.combat_curriculum.contact_initial_energies.clear();
        assert!(config.validate().is_err());

        config.combat_curriculum.contact_initial_energies = vec![100];
        config.combat_curriculum.contact_opponents = vec![OpponentProfile::Random];
        assert!(config.validate().is_err());

        config.combat_curriculum.contact_opponents = vec![OpponentProfile::Defensive];
        config
            .combat_curriculum
            .contact_action_kind_exploration_floor = 1.0;
        assert!(config.validate().is_err());

        config
            .combat_curriculum
            .contact_action_kind_exploration_floor = 0.5;
        config
            .combat_curriculum
            .skirmish_action_kind_exploration_floor = 1.0;
        assert!(config.validate().is_err());

        config
            .combat_curriculum
            .skirmish_action_kind_exploration_floor = 0.25;
        for invalid in [-0.01, f32::INFINITY, f32::NAN] {
            config.combat_curriculum.contact_initial_policy_anchor_coeff = Some(invalid);
            assert!(
                config.validate().is_err(),
                "accepted contact anchor coefficient {invalid}"
            );
        }
    }

    #[test]
    fn adjacent_retention_rejects_blocks_with_enclosed_cells() {
        let mut config = TrainingConfig::from_toml_str(include_str!(
            "../config/micro_combat_ablation_256_base.toml"
        ))
        .unwrap();
        assert!(config
            .validate()
            .unwrap_err()
            .contains("enclosed interior cells"));
        config.env.starting_cell_layout = StartingCellLayout::Checkerboard;
        config.validate().unwrap();
    }

    #[test]
    fn specialist_distillation_is_explicit_and_requires_both_curricula() {
        let default_json = serde_json::to_value(TrainingConfig::default()).unwrap();
        assert!(default_json.get("specialist_distillation").is_none());

        let mut config = TrainingConfig::default();
        config.specialist_distillation.enabled = true;
        assert!(config.validate().is_err());
        config.feeding_curriculum.enabled = true;
        config.combat_curriculum.enabled = true;
        assert!(config.validate().is_ok());
        assert_eq!(
            config.rollout_specialist_distillation_coeff(FeedingCurriculumStage::OnFood),
            Some(0.1)
        );
        assert_eq!(
            config.rollout_specialist_distillation_coeff(FeedingCurriculumStage::Skirmish),
            Some(0.1)
        );
        assert_eq!(
            config.rollout_specialist_distillation_coeff(FeedingCurriculumStage::Competitive),
            None
        );

        config.specialist_distillation.ecology_coeff = f32::NAN;
        assert!(config.validate().is_err());

        let mut precursor = TrainingConfig::default();
        precursor.feeding_curriculum.enabled = true;
        precursor.combat_curriculum.enabled = true;
        precursor.specialist_distillation.enabled = true;
        precursor
            .specialist_distillation
            .combat_precursor_min_skirmish_damage = Some(1);
        assert!(precursor.validate().is_ok());
        precursor
            .specialist_distillation
            .combat_precursor_min_skirmish_damage = Some(0);
        assert!(precursor.validate().is_err());
        precursor
            .specialist_distillation
            .combat_precursor_min_skirmish_damage = Some(1);
        precursor.specialist_distillation.combat_coeff = 0.0;
        assert!(precursor.validate().is_err());
        precursor.specialist_distillation.combat_coeff = 0.1;
        precursor.specialist_distillation.enabled = false;
        assert!(precursor.validate().is_err());
    }

    #[test]
    fn disabled_default_combat_curriculum_is_omitted_from_serialized_configs() {
        let config = TrainingConfig::default();
        let encoded = serde_json::to_value(&config).unwrap();
        assert!(encoded.get("combat_curriculum").is_none());
        assert_eq!(
            serde_json::from_value::<TrainingConfig>(encoded).unwrap(),
            config
        );
    }

    #[test]
    fn feeding_curriculum_rejects_invalid_frontiers_and_gate_thresholds() {
        let mut config = TrainingConfig::default();
        config.feeding_curriculum.enabled = true;
        config
            .feeding_curriculum
            .on_food_until_sim_time_quanta_per_env = 10;
        config
            .feeding_curriculum
            .adjacent_food_until_sim_time_quanta_per_env = 10;
        assert_eq!(
            config.validate().unwrap_err(),
            "feeding curriculum simulation-time frontiers must be positive and strictly ordered"
        );
        config
            .feeding_curriculum
            .adjacent_food_until_sim_time_quanta_per_env = 20;
        config.feeding_curriculum.promotion.min_survival_rate = f64::NAN;
        assert_eq!(
            config.validate().unwrap_err(),
            "feeding promotion thresholds and horizons are invalid"
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
            (
                "starting_cell_layout".into(),
                toml::Value::String("ring".into()),
            ),
        ]);
        let changed = base.with_scenario_override(&scenario).unwrap();
        assert_eq!(changed.env.num_plants, 12);
        assert_eq!(changed.env.initial_energy, 80);
        assert_eq!(changed.env.starting_cell_layout, StartingCellLayout::Ring);
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
    fn combat_curriculum_overrides_are_strict_and_isolated() {
        let mut base = TrainingConfig::default();
        base.feeding_curriculum.enabled = true;
        base.combat_curriculum.enabled = true;
        let changed = base
            .with_combat_curriculum_override(&toml::Table::from_iter([
                (
                    "cycle_sim_time_quanta_per_env".into(),
                    toml::Value::Integer(131_072),
                ),
                (
                    "on_food_sim_time_quanta_per_cycle".into(),
                    toml::Value::Integer(16_384),
                ),
                (
                    "adjacent_food_sim_time_quanta_per_cycle".into(),
                    toml::Value::Integer(16_384),
                ),
                (
                    "contact_sim_time_quanta_per_cycle".into(),
                    toml::Value::Integer(16_384),
                ),
                (
                    "skirmish_sim_time_quanta_per_cycle".into(),
                    toml::Value::Integer(49_152),
                ),
                (
                    "retention_episode_sim_time_limit_quanta".into(),
                    toml::Value::Integer(16_384),
                ),
                (
                    "contact_episode_sim_time_limit_quanta".into(),
                    toml::Value::Integer(16_384),
                ),
                (
                    "skirmish_episode_sim_time_limit_quanta".into(),
                    toml::Value::Integer(49_152),
                ),
                (
                    "contact_evaluation_sim_time_limit_quanta".into(),
                    toml::Value::Integer(8_192),
                ),
                (
                    "skirmish_evaluation_sim_time_limit_quanta".into(),
                    toml::Value::Integer(24_576),
                ),
            ]))
            .unwrap();
        assert_eq!(
            changed.combat_curriculum.cycle_sim_time_quanta_per_env,
            131_072
        );
        assert_eq!(
            changed.combat_curriculum.skirmish_sim_time_quanta_per_cycle,
            49_152
        );
        assert_eq!(
            changed
                .combat_curriculum
                .contact_evaluation_sim_time_limit_quanta,
            8_192
        );
        assert_eq!(
            changed
                .combat_curriculum
                .skirmish_evaluation_sim_time_limit_quanta,
            24_576
        );
        assert_eq!(changed.env, base.env);
        assert_eq!(changed.reward, base.reward);
        assert_eq!(changed.ppo, base.ppo);
        assert_eq!(changed.model, base.model);
        assert_eq!(changed.evaluation_seed, base.evaluation_seed);

        let typo = toml::Table::from_iter([(
            "skirmish_sim_time_quanta_per_cyle".into(),
            toml::Value::Integer(49_152),
        )]);
        assert!(base
            .with_combat_curriculum_override(&typo)
            .unwrap_err()
            .contains("unknown field `skirmish_sim_time_quanta_per_cyle`"));
    }

    #[test]
    fn specialist_distillation_overrides_are_strict_and_isolated() {
        let mut base = TrainingConfig::default();
        base.feeding_curriculum.enabled = true;
        base.combat_curriculum.enabled = true;
        let changed = base
            .with_specialist_distillation_override(&toml::Table::from_iter([
                ("enabled".into(), toml::Value::Boolean(true)),
                ("ecology_coeff".into(), toml::Value::Float(0.2)),
            ]))
            .unwrap();
        assert!(changed.specialist_distillation.enabled);
        assert_eq!(changed.specialist_distillation.ecology_coeff, 0.2);
        assert_eq!(changed.specialist_distillation.combat_coeff, 0.1);
        assert_eq!(changed.env, base.env);
        assert_eq!(changed.reward, base.reward);
        assert_eq!(changed.ppo, base.ppo);
        assert_eq!(changed.model, base.model);
        assert_eq!(changed.combat_curriculum, base.combat_curriculum);

        let typo =
            toml::Table::from_iter([("ecology_coefficient".into(), toml::Value::Float(0.2))]);
        assert!(base
            .with_specialist_distillation_override(&typo)
            .unwrap_err()
            .contains("unknown field `ecology_coefficient`"));
    }

    #[test]
    fn resource_layouts_are_strict_validated_and_scenario_hash_bound() {
        let configured = TrainingConfig::from_toml_str(
            r#"
            [env]
            world_size = 32
            cells_per_team = 4
            num_teams = 2
            num_scattered_energy = 40
            num_plants = 12

            [env.plant_layout]
            mode = "islands"
            island_count = 3
            radius = 2
            minimum_separation = 6

            [env.scattered_energy_layout]
            mode = "corridors"
            corridor_count = 2
            half_width = 1
            "#,
        )
        .unwrap();
        assert!(matches!(
            configured.env.plant_layout,
            ResourceLayout::Islands {
                island_count: 3,
                radius: 2,
                minimum_separation: 6
            }
        ));
        assert!(matches!(
            configured.env.scattered_energy_layout,
            ResourceLayout::Corridors {
                corridor_count: 2,
                half_width: 1
            }
        ));
        assert_ne!(
            ScenarioProfile::from(&configured.env)
                .semantic_hash()
                .unwrap(),
            ScenarioProfile::from(&TrainingConfig::default().env)
                .semantic_hash()
                .unwrap()
        );

        let typo = TrainingConfig::from_toml_str(
            r#"
            [env.plant_layout]
            mode = "patches"
            patch_count = 2
            raduis = 3
            "#,
        )
        .unwrap_err();
        assert!(typo.contains("unknown field `raduis`"));

        let mut invalid = TrainingConfig::default();
        invalid.env.plant_layout = ResourceLayout::FavoredTerritory {
            team: invalid.env.num_teams,
            radius: 4,
        };
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("favored resource territory names no starting team"));
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

    #[test]
    fn combat_warm_start_qualification_profile_is_large_world_and_local_only() {
        let path = format!(
            "{}/config/combat_warm_start_256.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let config = TrainingConfig::from_file(&path).unwrap();
        config.validate().unwrap();

        assert_eq!(config.env.world_size, 256);
        assert_eq!(config.env.cells_per_team, 256);
        assert_eq!(
            config.env.starting_cell_layout,
            StartingCellLayout::Checkerboard
        );
        assert_eq!(config.model.hidden1, 256);
        assert_eq!(config.model.hidden2, 128);
        assert_eq!(config.model.recurrent_size, 128);
        assert!(config.feeding_curriculum.enabled);
        assert!(config.combat_curriculum.enabled);
        assert!(!config.combat_curriculum.micro_combat.enabled);
        assert_eq!(crate::model::policy_memory_bytes(128), Some(264));
        assert!(264 <= config.env.rules.max_private_memory_bytes);
    }

    #[test]
    fn competitive_transfer_profile_keeps_retention_without_rollout_staging() {
        let path = format!(
            "{}/config/competitive_transfer.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let config = TrainingConfig::from_file(&path).unwrap();
        config.validate().unwrap();
        assert!(config.feeding_curriculum.enabled);
        assert!(!config.feeding_curriculum.rollout_stages_enabled);
        assert_eq!(
            config.feeding_curriculum.stage(0),
            FeedingCurriculumStage::Competitive
        );
        assert_eq!(config.ppo.learning_rate, 1e-5);
        assert_eq!(config.ppo.action_kind_exploration_floor, 0.1);
        assert_eq!(config.reward.damage_enemy, 0.02);
        assert!(config.combat_curriculum.enabled);
        assert_eq!(config.total_simulation_quanta_per_env, Some(262_144));

        let large_path = format!(
            "{}/config/competitive_transfer_large.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let large = TrainingConfig::from_file(&large_path).unwrap();
        large.validate().unwrap();
        assert_eq!(large.env, config.env);
        assert_eq!(large.reward, config.reward);
        assert_eq!(large.feeding_curriculum, config.feeding_curriculum);
        assert_eq!(large.combat_curriculum, config.combat_curriculum);
        assert_eq!(large.model.hidden1, 256);
        assert_eq!(large.model.hidden2, 128);
        assert_eq!(large.model.recurrent_size, 128);
        assert_eq!(crate::model::policy_memory_bytes(128), Some(264));
        assert!(264 <= large.env.rules.max_private_memory_bytes);

        let long_path = format!(
            "{}/config/competitive_transfer_large_long.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let long = TrainingConfig::from_file(&long_path).unwrap();
        long.validate().unwrap();
        assert_eq!(long.env, large.env);
        assert_eq!(long.reward, large.reward);
        assert_eq!(long.feeding_curriculum, large.feeding_curriculum);
        assert_eq!(long.combat_curriculum, large.combat_curriculum);
        assert_eq!(long.ppo, large.ppo);
        assert_eq!(long.model, large.model);
        assert_eq!(long.total_simulation_quanta_per_env, Some(1_048_576));
        assert_eq!(long.eval_interval, 16);

        let skirmish_path = format!(
            "{}/config/competitive_transfer_large_skirmish.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let skirmish = TrainingConfig::from_file(&skirmish_path).unwrap();
        skirmish.validate().unwrap();
        assert_eq!(skirmish.env, large.env);
        assert_eq!(skirmish.reward, large.reward);
        assert_eq!(skirmish.feeding_curriculum, large.feeding_curriculum);
        assert_eq!(skirmish.combat_curriculum, large.combat_curriculum);
        assert_eq!(skirmish.ppo, large.ppo);
        assert_eq!(skirmish.model, large.model);
        assert_eq!(skirmish.total_simulation_quanta_per_env, Some(524_288));
        assert_eq!(skirmish.eval_interval, 16);
        assert_eq!(
            skirmish.combat_curriculum.min_skirmish_kills_for_promotion,
            1
        );

        let frequent_path = format!(
            "{}/config/competitive_transfer_large_skirmish_frequent.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let frequent = TrainingConfig::from_file(&frequent_path).unwrap();
        frequent.validate().unwrap();
        assert_eq!(frequent.env, skirmish.env);
        assert_eq!(frequent.reward, skirmish.reward);
        assert_eq!(frequent.feeding_curriculum, skirmish.feeding_curriculum);
        assert_eq!(frequent.ppo, skirmish.ppo);
        assert_eq!(frequent.model, skirmish.model);
        assert_eq!(frequent.total_simulation_quanta_per_env, Some(524_288));
        assert_eq!(
            frequent.combat_curriculum.cycle_sim_time_quanta_per_env,
            131_072
        );
        assert_eq!(
            frequent.combat_curriculum.min_skirmish_kills_for_promotion,
            1
        );

        let specialist_path = format!(
            "{}/config/competitive_transfer_large_specialist_distillation.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let specialist = TrainingConfig::from_file(&specialist_path).unwrap();
        specialist.validate().unwrap();
        assert_eq!(specialist.env, frequent.env);
        assert_eq!(specialist.reward, frequent.reward);
        assert_eq!(specialist.feeding_curriculum, frequent.feeding_curriculum);
        assert_eq!(specialist.combat_curriculum, frequent.combat_curriculum);
        assert_eq!(specialist.ppo, frequent.ppo);
        assert!(specialist.specialist_distillation.enabled);
        assert_eq!(specialist.specialist_distillation.ecology_coeff, 0.05);
        assert_eq!(specialist.specialist_distillation.combat_coeff, 0.05);

        let contact_anchor_path = format!(
            "{}/config/competitive_transfer_large_contact_anchor.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let contact_anchor = TrainingConfig::from_file(&contact_anchor_path).unwrap();
        contact_anchor.validate().unwrap();
        assert_eq!(contact_anchor.env, large.env);
        assert_eq!(contact_anchor.reward, large.reward);
        assert_eq!(contact_anchor.ppo, large.ppo);
        assert_eq!(
            contact_anchor
                .combat_curriculum
                .contact_initial_policy_anchor_coeff,
            Some(0.5)
        );
        assert_eq!(
            contact_anchor.rollout_initial_policy_anchor_coeff(FeedingCurriculumStage::Contact),
            0.5
        );
        assert_eq!(
            contact_anchor.rollout_initial_policy_anchor_coeff(FeedingCurriculumStage::Competitive),
            0.25
        );
    }

    #[test]
    fn action_kind_exploration_floor_must_be_a_finite_fraction_below_one() {
        for invalid in [-0.01, 1.0, f32::INFINITY, f32::NAN] {
            let mut config = TrainingConfig::default();
            config.ppo.action_kind_exploration_floor = invalid;
            assert!(
                config.validate().is_err(),
                "accepted exploration floor {invalid}"
            );
        }

        let mut config = TrainingConfig::default();
        config.ppo.action_kind_exploration_floor = 0.25;
        config.validate().unwrap();
    }

    #[test]
    fn large_world_profiles_are_valid_and_scale_world_time() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let expected = [
            ("large_world_256.toml", 256, 67_108_864, 16_777_216),
            ("large_world_512.toml", 512, 134_217_728, 33_554_432),
            ("large_world_1024.toml", 1_024, 268_435_456, 67_108_864),
        ];

        for (name, world_size, training_quanta, episode_quanta) in expected {
            let path = format!("{manifest}/config/{name}");
            let config = TrainingConfig::from_file(&path).unwrap();
            config.validate().unwrap();
            assert_eq!(config.env.world_size, world_size);
            assert_eq!(
                config.total_simulation_quanta_per_env,
                Some(training_quanta)
            );
            assert_eq!(config.env.victory.sim_time_limit_quanta, episode_quanta);
            assert!(config.total_timesteps > training_quanta / 1_024);
        }
    }
}
