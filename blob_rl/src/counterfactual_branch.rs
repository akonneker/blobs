//! Replay-verifiable counterfactual action branches from frozen-policy states.
//!
//! Schema 1 deliberately covers one narrow causal question: when the maintained
//! teacher chooses Guard and the frozen policy chooses Move for one surviving
//! training cell, what follows from each legal Guard/Move intervention? Search
//! state and scoring are trusted-host diagnostics; deployed policies still see
//! only their ordinary anonymous Mind input and cell-private memory.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::resolution::{OutcomeStatus, ReferenceCompiledTopology, RejectReason};
use blob_interface::types::CellId;
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

use crate::action::{decompose_policy_action, PolicyActionKind, PolicyChoice, NUM_ACTIONS};
use crate::config::TrainingConfig;
use crate::control_matrix::MaintainedMindProfile;
use crate::env::{
    BlobEnv, BlobEnvPlannerCheckpoint, EpisodeOutcome, OpponentStartingState, PolicyObservation,
    StepOutput,
};
use crate::evaluation::{greedy_policy_choices, greedy_policy_choices_with_kind_logits};
use crate::micro_combat_planner::mind_observation_sha256;
use crate::model::PolicyValueNet;
use crate::observation::{observation_expert_context, Observation, ObservationExpertContext};
use crate::sweep::sha256;
use crate::telemetry::TelemetryConfig;
use crate::viability::mind_abi_hash;

pub const COUNTERFACTUAL_BRANCH_SCHEMA_VERSION: u32 = 4;
const MAX_STATES: usize = 1_024;
const MAX_HORIZONS: usize = 16;
const MAX_TOTAL_BRANCHES: usize = 250_000;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BranchContinuation {
    FrozenPolicy,
    Teacher,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BranchCandidateScope {
    /// Enumerate every legal base Guard and Move action. Best for short,
    /// exploratory probes of the local action surface.
    AllLegal,
    /// Evaluate only the normalized teacher and frozen-policy choices. Best
    /// for deep confirmation after a short probe has located useful states.
    TeacherPolicy,
    /// Evaluate a caller-declared action-id set, filtered through each exact
    /// observation's legal mask. Teacher and policy choices remain included as
    /// audit baselines even when dominated by the declared set.
    Explicit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualBranchOptions {
    pub teacher: MaintainedMindProfile,
    pub seeds: Vec<u64>,
    pub max_states: usize,
    /// Prevent one trajectory from consuming the global state budget.
    pub max_states_per_seed: usize,
    /// Prevent a single simultaneous ready-cell frontier from being treated as
    /// many independent causal observations.
    pub max_states_per_frontier: usize,
    /// Bounded prefix of each source trajectory searched for qualifying states.
    pub max_source_frontiers_per_seed: u64,
    /// Requested elapsed simulation-clock horizons after the intervention.
    pub horizon_quanta: Vec<u64>,
    pub continuations: Vec<BranchContinuation>,
    pub candidate_scope: BranchCandidateScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explicit_candidate_actions: Vec<usize>,
    /// Restrict the causal probe to the context implicated by the current
    /// feeding failure rather than pooling unrelated Guard semantics.
    pub exploration_only: bool,
    /// Independent host guard. Scientific termination still uses world time.
    pub max_frontiers_per_branch: u64,
}

impl CounterfactualBranchOptions {
    pub fn validate(&self) -> Result<(), String> {
        let balanced_capacity = self.seeds.len().saturating_mul(self.max_states_per_seed);
        let explicit_actions_valid = match self.candidate_scope {
            BranchCandidateScope::Explicit => {
                !self.explicit_candidate_actions.is_empty()
                    && self.explicit_candidate_actions.len() <= NUM_ACTIONS
                    && self
                        .explicit_candidate_actions
                        .iter()
                        .copied()
                        .collect::<HashSet<_>>()
                        .len()
                        == self.explicit_candidate_actions.len()
                    && self.explicit_candidate_actions.iter().all(|action| {
                        decompose_policy_action(*action).is_some_and(|decomposition| {
                            matches!(
                                PolicyActionKind::from_index(decomposition.kind),
                                Some(PolicyActionKind::Guard | PolicyActionKind::Move)
                            )
                        })
                    })
            }
            BranchCandidateScope::AllLegal | BranchCandidateScope::TeacherPolicy => {
                self.explicit_candidate_actions.is_empty()
            }
        };
        if self.seeds.is_empty()
            || self.seeds.len() > 4_096
            || self.seeds.iter().copied().collect::<HashSet<_>>().len() != self.seeds.len()
            || self.max_states == 0
            || self.max_states > MAX_STATES
            || self.max_states_per_seed == 0
            || self.max_states_per_seed > self.max_states
            || self.max_states < balanced_capacity
            || self.max_states_per_frontier == 0
            || self.max_states_per_frontier > self.max_states_per_seed
            || self.max_source_frontiers_per_seed == 0
            || !explicit_actions_valid
            || self.horizon_quanta.is_empty()
            || self.horizon_quanta.len() > MAX_HORIZONS
            || self.horizon_quanta.contains(&0)
            || !self.horizon_quanta.windows(2).all(|pair| pair[0] < pair[1])
            || self.continuations.is_empty()
            || self
                .continuations
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != self.continuations.len()
            || self.max_frontiers_per_branch == 0
        {
            return Err("counterfactual branch options are invalid or unbounded".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BranchEpisodeOutcome {
    Win,
    Loss,
    Timeout,
    SafetyAbort,
}

impl From<EpisodeOutcome> for BranchEpisodeOutcome {
    fn from(value: EpisodeOutcome) -> Self {
        match value {
            EpisodeOutcome::Win => Self::Win,
            EpisodeOutcome::Loss => Self::Loss,
            EpisodeOutcome::Timeout => Self::Timeout,
            EpisodeOutcome::SafetyAbort => Self::SafetyAbort,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BranchActionResolution {
    Pending,
    Success,
    Frustrated,
    Contested,
    Interrupted,
    Rejected,
    ActorDied,
}

fn branch_resolution(status: Option<&OutcomeStatus>, pending: bool) -> BranchActionResolution {
    if pending {
        return BranchActionResolution::Pending;
    }
    match status {
        Some(OutcomeStatus::Success) => BranchActionResolution::Success,
        Some(OutcomeStatus::Frustrated) => BranchActionResolution::Frustrated,
        Some(OutcomeStatus::Contested) => BranchActionResolution::Contested,
        Some(OutcomeStatus::Interrupted) => BranchActionResolution::Interrupted,
        Some(OutcomeStatus::Rejected(
            RejectReason::InvalidSlot
            | RejectReason::ActionNotAllowedInSlot
            | RejectReason::TargetOutsideWorld
            | RejectReason::TargetsSelf
            | RejectReason::ZeroPayload
            | RejectReason::InsufficientGutEnergy
            | RejectReason::ChildAllocationTooSmall
            | RejectReason::PrivateMemoryTooLarge
            | RejectReason::InsufficientEnergy
            | RejectReason::InsufficientMaterial
            | RejectReason::TerrainLimit
            | RejectReason::ArithmeticOverflow,
        )) => BranchActionResolution::Rejected,
        None => BranchActionResolution::Pending,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BranchTotals {
    pub reward_microunits: i64,
    pub moves_succeeded: u64,
    pub guards_succeeded: u64,
    pub consumes_succeeded: u64,
    pub consumed_energy: u128,
    pub attacks_succeeded: u64,
    pub damage_dealt: u128,
    pub damage_received: u128,
    pub births: u64,
    pub deaths: u64,
    pub kills: u64,
}

impl BranchTotals {
    fn observe(&mut self, output: &StepOutput) -> Result<(), String> {
        let reward = output.rewards.values().try_fold(0.0_f64, |total, reward| {
            let reward = f64::from(*reward);
            reward
                .is_finite()
                .then_some(total + reward)
                .ok_or("counterfactual branch produced a non-finite reward")
        })?;
        let reward_microunits = (reward * 1_000_000.0).round();
        if reward_microunits < i64::MIN as f64 || reward_microunits > i64::MAX as f64 {
            return Err("counterfactual branch reward exceeded fixed-point range".into());
        }
        self.reward_microunits = self
            .reward_microunits
            .checked_add(reward_microunits as i64)
            .ok_or("counterfactual branch cumulative reward overflowed")?;
        let Some(telemetry) = output.telemetry.as_ref() else {
            return Err("counterfactual branch telemetry is disabled".into());
        };
        let side = &telemetry.training;
        self.moves_succeeded = self
            .moves_succeeded
            .saturating_add(side.actions.movement.succeeded);
        self.guards_succeeded = self
            .guards_succeeded
            .saturating_add(side.actions.guard.succeeded);
        self.consumes_succeeded = self
            .consumes_succeeded
            .saturating_add(side.actions.consume.succeeded);
        self.consumed_energy = self
            .consumed_energy
            .saturating_add(side.actions.consume.consumed_energy);
        self.attacks_succeeded = self
            .attacks_succeeded
            .saturating_add(side.actions.attack.succeeded);
        self.damage_dealt = self.damage_dealt.saturating_add(side.damage.applied_dealt);
        self.damage_received = self
            .damage_received
            .saturating_add(side.damage.applied_received);
        self.births = self.births.saturating_add(side.births);
        self.deaths = self.deaths.saturating_add(side.deaths);
        self.kills = self.kills.saturating_add(side.kills);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BranchHorizonResult {
    pub requested_elapsed_quanta: u64,
    /// A terminal or host-bounded branch may stop before the requested time.
    pub horizon_reached: bool,
    pub observed_elapsed_quanta: u64,
    pub controlled_frontiers: u64,
    pub target_alive: bool,
    pub target_position: Option<usize>,
    pub target_core_mass: u64,
    pub target_assimilated_energy: u64,
    pub target_gut_energy: u64,
    pub training_cells: usize,
    pub opponent_cells: usize,
    pub training_core_mass: u128,
    pub training_assimilated_energy: u128,
    pub training_gut_energy: u128,
    pub training_stored_energy: u128,
    pub training_total_mass_energy: u128,
    pub opponent_core_mass: u128,
    pub opponent_assimilated_energy: u128,
    pub opponent_gut_energy: u128,
    pub opponent_stored_energy: u128,
    pub opponent_total_mass_energy: u128,
    pub episode_outcome: Option<BranchEpisodeOutcome>,
    pub branch_host_bound_reached: bool,
    pub totals: BranchTotals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidateBranch {
    pub choice: PolicyChoice,
    pub continuation: BranchContinuation,
    pub immediate_resolution: BranchActionResolution,
    pub horizons: Vec<BranchHorizonResult>,
}

/// Exact anonymous policy input retained for downstream, target-free
/// correction synthesis. Floating-point observations are stored as raw bits so
/// the artifact remains exactly comparable and JSON round trips cannot alter
/// a feature. The private memory is the cell's own policy state, not a public
/// identity or cross-cell communication channel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualMindEvidence {
    pub observation_bits: Vec<u32>,
    pub action_mask: Vec<bool>,
    pub amount_choice_bits: Vec<u8>,
    pub sidecar_strength_bits: Vec<u32>,
    pub explicit_signal_strength_bits: Vec<u8>,
    pub private_memory: Vec<u8>,
}

impl CounterfactualMindEvidence {
    fn new(observation: &Observation, private_memory: &[u8]) -> Self {
        Self {
            observation_bits: observation
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            action_mask: observation.action_mask.to_vec(),
            amount_choice_bits: observation.amount_choice_bits.to_vec(),
            sidecar_strength_bits: observation.sidecar_strength_bits.to_vec(),
            explicit_signal_strength_bits: observation.explicit_signal_strength_bits.to_vec(),
            private_memory: private_memory.to_vec(),
        }
    }

    pub fn observation(&self) -> Result<Observation, String> {
        Ok(Observation {
            data: self
                .observation_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| "counterfactual evidence has the wrong observation length")?,
            action_mask: self
                .action_mask
                .clone()
                .try_into()
                .map_err(|_| "counterfactual evidence has the wrong action-mask length")?,
            amount_choice_bits: self
                .amount_choice_bits
                .clone()
                .try_into()
                .map_err(|_| "counterfactual evidence has the wrong amount-mask length")?,
            sidecar_strength_bits: self
                .sidecar_strength_bits
                .clone()
                .try_into()
                .map_err(|_| "counterfactual evidence has the wrong sidecar-mask length")?,
            explicit_signal_strength_bits: self
                .explicit_signal_strength_bits
                .clone()
                .try_into()
                .map_err(|_| "counterfactual evidence has the wrong explicit-signal length")?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualState {
    pub seed: u64,
    pub seed_frontier: u64,
    pub state_ordinal: usize,
    pub sim_time_quanta: u64,
    pub source_cell: u64,
    pub checkpoint_sha256: String,
    pub mind_observation_sha256: String,
    /// Absent only in historical schema-3 artifacts created before correction
    /// synthesis. Omitting `None` preserves their exact serialized identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mind_evidence: Option<CounterfactualMindEvidence>,
    pub expert_context: String,
    pub teacher_choice: PolicyChoice,
    pub policy_choice: PolicyChoice,
    /// Base physical choices only: no amount payload or signal sidecar. This
    /// isolates Guard/Move causality from communication policy.
    pub candidates: Vec<PolicyChoice>,
    pub branches: Vec<CandidateBranch>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairwiseHorizonSummary {
    pub continuation: BranchContinuation,
    pub requested_elapsed_quanta: u64,
    pub states: usize,
    pub teacher_target_survival_better: usize,
    pub policy_target_survival_better: usize,
    pub target_survival_ties: usize,
    pub teacher_training_population_better: usize,
    pub policy_training_population_better: usize,
    pub training_population_ties: usize,
    pub teacher_target_stored_energy_better: usize,
    pub policy_target_stored_energy_better: usize,
    pub target_stored_energy_ties: usize,
    pub teacher_team_stored_energy_better: usize,
    pub policy_team_stored_energy_better: usize,
    pub team_stored_energy_ties: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SeedSelectionSummary {
    pub seed: u64,
    pub selected_states: usize,
}

/// Seed-balanced preferences. Each seed contributes one vote after summing its
/// selected states, so a busy frontier or trajectory cannot dominate merely by
/// supplying more correlated cells.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairwiseSeedHorizonSummary {
    pub continuation: BranchContinuation,
    pub requested_elapsed_quanta: u64,
    pub seeds: usize,
    pub teacher_target_survival_better: usize,
    pub policy_target_survival_better: usize,
    pub target_survival_ties: usize,
    pub teacher_training_population_better: usize,
    pub policy_training_population_better: usize,
    pub training_population_ties: usize,
    pub teacher_target_stored_energy_better: usize,
    pub policy_target_stored_energy_better: usize,
    pub target_stored_energy_ties: usize,
    pub teacher_team_stored_energy_better: usize,
    pub policy_team_stored_energy_better: usize,
    pub team_stored_energy_ties: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualBranchReport {
    pub schema_version: u32,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub selected_states: usize,
    pub candidate_actions: usize,
    pub evaluated_branches: usize,
    pub seed_selection: Vec<SeedSelectionSummary>,
    pub states: Vec<CounterfactualState>,
    pub pairwise_teacher_policy: Vec<PairwiseHorizonSummary>,
    pub pairwise_seed_teacher_policy: Vec<PairwiseSeedHorizonSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualBranchArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub mind_abi_sha256: String,
    pub source_config_sha256: String,
    pub behavior_clone_metadata_sha256: String,
    pub behavior_clone_model_sha256: String,
    pub config: TrainingConfig,
    pub options: CounterfactualBranchOptions,
    pub report: CounterfactualBranchReport,
}

#[derive(Serialize)]
struct ArtifactIdentity<'a> {
    schema_version: u32,
    package_version: &'a str,
    code_revision: &'a Option<String>,
    mind_abi_sha256: &'a str,
    source_config_sha256: &'a str,
    behavior_clone_metadata_sha256: &'a str,
    behavior_clone_model_sha256: &'a str,
    config: &'a TrainingConfig,
    options: &'a CounterfactualBranchOptions,
    report: &'a CounterfactualBranchReport,
}

impl CounterfactualBranchArtifact {
    pub fn new(
        source_config_sha256: String,
        behavior_clone_metadata_sha256: String,
        behavior_clone_model_sha256: String,
        config: TrainingConfig,
        options: CounterfactualBranchOptions,
        report: CounterfactualBranchReport,
    ) -> Result<Self, String> {
        let mut artifact = Self {
            schema_version: COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            artifact_hash: String::new(),
            mind_abi_sha256: mind_abi_hash(),
            source_config_sha256,
            behavior_clone_metadata_sha256,
            behavior_clone_model_sha256,
            config,
            options,
            report,
        };
        artifact.artifact_hash = artifact.recompute_hash()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn recompute_hash(&self) -> Result<String, String> {
        let identity = ArtifactIdentity {
            schema_version: self.schema_version,
            package_version: &self.package_version,
            code_revision: &self.code_revision,
            mind_abi_sha256: &self.mind_abi_sha256,
            source_config_sha256: &self.source_config_sha256,
            behavior_clone_metadata_sha256: &self.behavior_clone_metadata_sha256,
            behavior_clone_model_sha256: &self.behavior_clone_model_sha256,
            config: &self.config,
            options: &self.options,
            report: &self.report,
        };
        serde_json::to_vec(&identity)
            .map(|bytes| sha256(&bytes))
            .map_err(|error| format!("failed to encode counterfactual artifact identity: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != COUNTERFACTUAL_BRANCH_SCHEMA_VERSION
            || self.package_version.is_empty()
            || self.mind_abi_sha256 != mind_abi_hash()
            || !valid_sha256(&self.source_config_sha256)
            || !valid_sha256(&self.behavior_clone_metadata_sha256)
            || !valid_sha256(&self.behavior_clone_model_sha256)
            || !valid_sha256(&self.artifact_hash)
        {
            return Err("counterfactual branch artifact identity is invalid".into());
        }
        self.config.validate()?;
        self.options.validate()?;
        self.report.validate_against(&self.config, &self.options)?;
        if self.artifact_hash != self.recompute_hash()? {
            return Err("counterfactual branch artifact hash mismatch".into());
        }
        Ok(())
    }
}

impl CounterfactualBranchReport {
    pub fn validate_against(
        &self,
        config: &TrainingConfig,
        options: &CounterfactualBranchOptions,
    ) -> Result<(), String> {
        options.validate()?;
        if self.schema_version != COUNTERFACTUAL_BRANCH_SCHEMA_VERSION
            || !valid_sha256(&self.compiled_ruleset_hash)
            || self.scenario_hash
                != crate::config::ScenarioProfile::from(&config.env).semantic_hash()?
            || self.selected_states != self.states.len()
            || self.selected_states == 0
            || self.selected_states > options.max_states
            || self.candidate_actions
                != self
                    .states
                    .iter()
                    .map(|state| state.candidates.len())
                    .sum::<usize>()
            || self.evaluated_branches
                != self
                    .states
                    .iter()
                    .map(|state| state.branches.len())
                    .sum::<usize>()
            || self.evaluated_branches > MAX_TOTAL_BRANCHES
        {
            return Err("counterfactual branch report envelope is invalid".into());
        }
        for (ordinal, state) in self.states.iter().enumerate() {
            validate_state(state, ordinal, options)?;
        }
        if self.seed_selection != summarize_seed_selection(&self.states, options)
            || self
                .seed_selection
                .iter()
                .any(|seed| seed.selected_states > options.max_states_per_seed)
        {
            return Err("counterfactual seed selection summary was modified".into());
        }
        let mut per_frontier = HashMap::new();
        for state in &self.states {
            let count = per_frontier
                .entry((state.seed, state.seed_frontier))
                .or_insert(0_usize);
            *count += 1;
            if *count > options.max_states_per_frontier {
                return Err("counterfactual frontier exceeds its sampling bound".into());
            }
        }
        if self.pairwise_teacher_policy != summarize_pairwise(&self.states, options)? {
            return Err("counterfactual pairwise summaries were modified".into());
        }
        if self.pairwise_seed_teacher_policy != summarize_pairwise_by_seed(&self.states, options)? {
            return Err("counterfactual seed-balanced summaries were modified".into());
        }
        Ok(())
    }
}

fn validate_state(
    state: &CounterfactualState,
    ordinal: usize,
    options: &CounterfactualBranchOptions,
) -> Result<(), String> {
    let teacher = decompose_policy_action(state.teacher_choice.action)
        .ok_or("counterfactual teacher action is outside the catalog")?;
    let policy = decompose_policy_action(state.policy_choice.action)
        .ok_or("counterfactual policy action is outside the catalog")?;
    let evidence = state
        .mind_evidence
        .as_ref()
        .map(CounterfactualMindEvidence::observation)
        .transpose()?;
    if state.state_ordinal != ordinal
        || !options.seeds.contains(&state.seed)
        || !valid_sha256(&state.checkpoint_sha256)
        || !valid_sha256(&state.mind_observation_sha256)
        || teacher.kind != PolicyActionKind::Guard.index()
        || policy.kind != PolicyActionKind::Move.index()
        || (options.exploration_only && state.expert_context != "exploration")
        || state.candidates.is_empty()
        || evidence.as_ref().is_some_and(|observation| {
            observation.data.iter().any(|value| !value.is_finite())
                || context_name(observation_expert_context(&observation.data))
                    != state.expert_context
                || !observation
                    .action_mask
                    .get(state.policy_choice.action)
                    .copied()
                    .unwrap_or(false)
        })
    {
        return Err("counterfactual selected-state identity is invalid".into());
    }
    let mut choices = HashSet::new();
    for choice in &state.candidates {
        let decomposition = decompose_policy_action(choice.action)
            .ok_or("counterfactual candidate action is outside the catalog")?;
        if !choices.insert(*choice)
            || !matches!(
                PolicyActionKind::from_index(decomposition.kind),
                Some(PolicyActionKind::Guard | PolicyActionKind::Move)
            )
            || choice.amount != 0
            || choice.signal != 0
            || choice.signal_strength != 0
        {
            return Err("counterfactual candidate set is invalid".into());
        }
    }
    let teacher_physical = base_choice(state.teacher_choice.action);
    let policy_physical = base_choice(state.policy_choice.action);
    if !choices.contains(&teacher_physical) || !choices.contains(&policy_physical) {
        return Err("counterfactual candidates omit the teacher or policy action".into());
    }
    if options.candidate_scope == BranchCandidateScope::TeacherPolicy {
        let expected = [teacher_physical, policy_physical]
            .into_iter()
            .collect::<HashSet<_>>();
        if choices != expected {
            return Err("teacher-policy scope contains an unrelated candidate".into());
        }
    }
    if options.candidate_scope == BranchCandidateScope::Explicit
        && choices.iter().any(|choice| {
            choice != &teacher_physical
                && choice != &policy_physical
                && !options.explicit_candidate_actions.contains(&choice.action)
        })
    {
        return Err("explicit scope contains an undeclared candidate".into());
    }
    let expected_branches = state
        .candidates
        .len()
        .checked_mul(options.continuations.len())
        .ok_or("counterfactual state branch count overflowed")?;
    if state.branches.len() != expected_branches {
        return Err("counterfactual state has incomplete branch coverage".into());
    }
    let candidate_set = state.candidates.iter().copied().collect::<HashSet<_>>();
    let mut branch_keys = HashSet::new();
    for branch in &state.branches {
        if !candidate_set.contains(&branch.choice)
            || !options.continuations.contains(&branch.continuation)
            || !branch_keys.insert((branch.choice, branch.continuation))
            || branch.horizons.len() != options.horizon_quanta.len()
        {
            return Err("counterfactual branch key or horizon coverage is invalid".into());
        }
        for (result, requested) in branch.horizons.iter().zip(&options.horizon_quanta) {
            if result.requested_elapsed_quanta != *requested
                || result.horizon_reached != (result.observed_elapsed_quanta >= *requested)
                || result.controlled_frontiers == 0
                || result.controlled_frontiers > options.max_frontiers_per_branch
                || result.target_alive != result.target_position.is_some()
                || result.training_stored_energy
                    != result
                        .training_assimilated_energy
                        .saturating_add(result.training_gut_energy)
                || result.training_total_mass_energy
                    != result
                        .training_core_mass
                        .saturating_add(result.training_stored_energy)
                || result.opponent_stored_energy
                    != result
                        .opponent_assimilated_energy
                        .saturating_add(result.opponent_gut_energy)
                || result.opponent_total_mass_energy
                    != result
                        .opponent_core_mass
                        .saturating_add(result.opponent_stored_energy)
                || result.branch_host_bound_reached
                    && result.controlled_frontiers != options.max_frontiers_per_branch
            {
                return Err("counterfactual horizon result is inconsistent".into());
            }
        }
    }
    Ok(())
}

pub fn evaluate_counterfactual_branches<B: Backend>(
    model: &PolicyValueNet<B>,
    config: &TrainingConfig,
    options: &CounterfactualBranchOptions,
    device: &B::Device,
) -> Result<CounterfactualBranchReport, String>
where
    f32: From<B::FloatElem>,
{
    config.validate()?;
    options.validate()?;
    let scenario_hash = crate::config::ScenarioProfile::from(&config.env).semantic_hash()?;
    let mut compiled_ruleset_hash = None;
    let mut states = Vec::new();

    'seeds: for seed in &options.seeds {
        if states.len() >= options.max_states {
            break;
        }
        let mut env = BlobEnv::new(config.env.clone(), config.reward.clone(), *seed);
        let topology = env
            .reference_simulation_for_diagnostics()?
            .compiled_topology();
        let opponent_start = OpponentStartingState {
            cells_per_team: config.env.cells_per_team,
            initial_energy: config.env.initial_energy,
        };
        let compiled = env.compiled_ruleset_hash();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &compiled)
        {
            return Err("counterfactual seeds compiled different rulesets".into());
        }
        compiled_ruleset_hash = Some(compiled);
        let mut seed_frontier = 0_u64;
        let mut selected_for_seed = 0_usize;
        loop {
            if seed_frontier >= options.max_source_frontiers_per_seed {
                break;
            }
            let prepared = env.prepare_training_reference_inputs()?;
            let policy_observations = prepared
                .iter()
                .map(|(cell_id, input)| PolicyObservation {
                    cell_id: *cell_id,
                    observation: Observation::from_reference(input),
                    private_memory: input.private_memory.clone(),
                })
                .collect::<Vec<_>>();
            let policy_choices =
                greedy_policy_choices_with_kind_logits(model, &policy_observations, device);
            if policy_choices.len() != prepared.len() {
                return Err("counterfactual policy lost a ready-cell observation".into());
            }

            let step_policy_choices = policy_choices
                .iter()
                .map(|(cell_id, choice, memory, _)| (*cell_id, *choice, memory.clone()))
                .collect::<Vec<_>>();
            let mut selected_this_frontier = 0_usize;
            for (cell_index, ((cell_id, input), (policy_cell, policy_choice, _, _))) in
                prepared.iter().zip(&policy_choices).enumerate()
            {
                if selected_for_seed >= options.max_states_per_seed
                    || selected_this_frontier >= options.max_states_per_frontier
                {
                    break;
                }
                if cell_id != policy_cell {
                    return Err("counterfactual policy changed ready-cell ordering".into());
                }
                let teacher_choice =
                    crate::action::encode_decision(&options.teacher.decide(input), input)
                        .ok_or("counterfactual teacher emitted an unencodable decision")?;
                let teacher_kind = decompose_policy_action(teacher_choice.action)
                    .ok_or("counterfactual teacher action has no kind")?
                    .kind;
                let policy_kind = decompose_policy_action(policy_choice.action)
                    .ok_or("counterfactual policy action has no kind")?
                    .kind;
                let context =
                    observation_expert_context(&policy_observations[cell_index].observation.data);
                if teacher_kind == PolicyActionKind::Guard.index()
                    && policy_kind == PolicyActionKind::Move.index()
                    && (!options.exploration_only
                        || context == ObservationExpertContext::Exploration)
                {
                    let (checkpoint, checkpoint_identity) = env.planner_checkpoint(None)?;
                    let required = [
                        base_choice(teacher_choice.action),
                        base_choice(policy_choice.action),
                    ];
                    let candidates = match options.candidate_scope {
                        BranchCandidateScope::AllLegal => {
                            guard_move_candidates(&policy_observations[cell_index].observation)
                        }
                        BranchCandidateScope::TeacherPolicy => {
                            let mut candidates = required.to_vec();
                            candidates.sort_by_key(|choice| choice.action);
                            candidates.dedup();
                            candidates
                        }
                        BranchCandidateScope::Explicit => {
                            let mut candidates = options
                                .explicit_candidate_actions
                                .iter()
                                .copied()
                                .filter(|action| {
                                    policy_observations[cell_index]
                                        .observation
                                        .action_mask
                                        .get(*action)
                                        .copied()
                                        .unwrap_or(false)
                                })
                                .map(base_choice)
                                .chain(required)
                                .collect::<Vec<_>>();
                            candidates.sort_by_key(|choice| choice.action);
                            candidates.dedup();
                            candidates
                        }
                    };
                    if required.iter().all(|choice| candidates.contains(choice)) {
                        let state_ordinal = states.len();
                        let sim_time_quanta = env.sim_time_quanta();
                        let branches = evaluate_state_branches(
                            model,
                            config,
                            options,
                            device,
                            &checkpoint,
                            &topology,
                            opponent_start,
                            sim_time_quanta,
                            *cell_id,
                            &step_policy_choices,
                            &candidates,
                        )?;
                        states.push(CounterfactualState {
                            seed: *seed,
                            seed_frontier,
                            state_ordinal,
                            sim_time_quanta,
                            source_cell: u64::try_from(cell_id.0)
                                .map_err(|_| "counterfactual cell identity does not fit u64")?,
                            checkpoint_sha256: hex_digest(&checkpoint_identity),
                            mind_observation_sha256: mind_observation_sha256(input)?,
                            mind_evidence: Some(CounterfactualMindEvidence::new(
                                &policy_observations[cell_index].observation,
                                &input.private_memory,
                            )),
                            expert_context: context_name(context).into(),
                            teacher_choice,
                            policy_choice: *policy_choice,
                            candidates,
                            branches,
                        });
                        selected_for_seed += 1;
                        selected_this_frontier += 1;
                        if states.len() >= options.max_states {
                            break 'seeds;
                        }
                    }
                }
            }

            if selected_for_seed >= options.max_states_per_seed {
                break;
            }

            let output = env.step_with_policy_memory(&step_policy_choices);
            seed_frontier = seed_frontier.saturating_add(1);
            if output.done {
                if output.outcome == Some(EpisodeOutcome::SafetyAbort) {
                    return Err(format!(
                        "counterfactual source seed {seed} reached its frontier safety bound"
                    ));
                }
                break;
            }
        }
    }
    if states.is_empty() {
        return Err(
            "counterfactual source trajectories contained no selected Guard-to-Move states".into(),
        );
    }
    let seed_selection = summarize_seed_selection(&states, options);
    let pairwise_teacher_policy = summarize_pairwise(&states, options)?;
    let pairwise_seed_teacher_policy = summarize_pairwise_by_seed(&states, options)?;
    let candidate_actions = states.iter().map(|state| state.candidates.len()).sum();
    let evaluated_branches = states.iter().map(|state| state.branches.len()).sum();
    if evaluated_branches > MAX_TOTAL_BRANCHES {
        return Err("counterfactual evaluation exceeded the total branch bound".into());
    }
    let report = CounterfactualBranchReport {
        schema_version: COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
        compiled_ruleset_hash: compiled_ruleset_hash
            .ok_or("counterfactual evaluation did not compile a ruleset")?,
        scenario_hash,
        selected_states: states.len(),
        candidate_actions,
        evaluated_branches,
        seed_selection,
        states,
        pairwise_teacher_policy,
        pairwise_seed_teacher_policy,
    };
    report.validate_against(config, options)?;
    Ok(report)
}

/// Replay only the source trajectory of a historical branch artifact and
/// attach exact anonymous Mind evidence. Every selected state must match its
/// original checkpoint, observation, policy choice, teacher choice, context,
/// and simulation time before the artifact receives a new identity.
pub fn enrich_counterfactual_mind_evidence<B: Backend>(
    model: &PolicyValueNet<B>,
    source: &CounterfactualBranchArtifact,
    device: &B::Device,
) -> Result<CounterfactualBranchArtifact, String>
where
    f32: From<B::FloatElem>,
{
    source.validate()?;
    if source
        .report
        .states
        .iter()
        .any(|state| state.mind_evidence.is_some())
    {
        return Err("counterfactual source already contains Mind evidence".into());
    }
    let mut enriched = source.clone();
    let mut remaining = source.report.states.len();
    for seed in &source.options.seeds {
        let wanted = source
            .report
            .states
            .iter()
            .filter(|state| state.seed == *seed)
            .map(|state| {
                (
                    (state.seed_frontier, state.source_cell),
                    state.state_ordinal,
                )
            })
            .collect::<HashMap<_, _>>();
        if wanted.is_empty() {
            continue;
        }
        let mut env = BlobEnv::new(
            source.config.env.clone(),
            source.config.reward.clone(),
            *seed,
        );
        let mut seed_frontier = 0_u64;
        loop {
            if seed_frontier >= source.options.max_source_frontiers_per_seed {
                break;
            }
            let prepared = env.prepare_training_reference_inputs()?;
            let policy_observations = prepared
                .iter()
                .map(|(cell_id, input)| PolicyObservation {
                    cell_id: *cell_id,
                    observation: Observation::from_reference(input),
                    private_memory: input.private_memory.clone(),
                })
                .collect::<Vec<_>>();
            let policy_choices =
                greedy_policy_choices_with_kind_logits(model, &policy_observations, device);
            if policy_choices.len() != prepared.len() {
                return Err("counterfactual evidence replay lost a ready-cell observation".into());
            }
            let step_policy_choices = policy_choices
                .iter()
                .map(|(cell_id, choice, memory, _)| (*cell_id, *choice, memory.clone()))
                .collect::<Vec<_>>();
            for (index, ((cell_id, input), (policy_cell, policy_choice, _, _))) in
                prepared.iter().zip(&policy_choices).enumerate()
            {
                if cell_id != policy_cell {
                    return Err("counterfactual evidence replay changed ready-cell ordering".into());
                }
                let source_cell = u64::try_from(cell_id.0)
                    .map_err(|_| "counterfactual cell identity does not fit u64")?;
                let Some(state_ordinal) = wanted.get(&(seed_frontier, source_cell)).copied() else {
                    continue;
                };
                let expected = &source.report.states[state_ordinal];
                let teacher_choice =
                    crate::action::encode_decision(&source.options.teacher.decide(input), input)
                        .ok_or("counterfactual evidence teacher action is unencodable")?;
                let context =
                    observation_expert_context(&policy_observations[index].observation.data);
                let (_, checkpoint_identity) = env.planner_checkpoint(None)?;
                if expected.sim_time_quanta != env.sim_time_quanta()
                    || expected.policy_choice != *policy_choice
                    || expected.teacher_choice != teacher_choice
                    || expected.checkpoint_sha256 != hex_digest(&checkpoint_identity)
                    || expected.mind_observation_sha256 != mind_observation_sha256(input)?
                    || expected.expert_context != context_name(context)
                {
                    return Err(
                        "counterfactual evidence replay did not match its source state".into(),
                    );
                }
                enriched.report.states[state_ordinal].mind_evidence =
                    Some(CounterfactualMindEvidence::new(
                        &policy_observations[index].observation,
                        &input.private_memory,
                    ));
                remaining -= 1;
            }
            if remaining == 0 {
                break;
            }
            let output = env.step_with_policy_memory(&step_policy_choices);
            seed_frontier = seed_frontier.saturating_add(1);
            if output.done {
                if output.outcome == Some(EpisodeOutcome::SafetyAbort) {
                    return Err("counterfactual evidence replay reached its safety bound".into());
                }
                break;
            }
        }
    }
    if remaining != 0 {
        return Err(format!(
            "counterfactual evidence replay missed {remaining} selected states"
        ));
    }
    enriched.artifact_hash.clear();
    enriched.artifact_hash = enriched.recompute_hash()?;
    enriched.validate()?;
    Ok(enriched)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_state_branches<B: Backend>(
    model: &PolicyValueNet<B>,
    config: &TrainingConfig,
    options: &CounterfactualBranchOptions,
    device: &B::Device,
    checkpoint: &BlobEnvPlannerCheckpoint,
    topology: &ReferenceCompiledTopology,
    opponent_start: OpponentStartingState,
    sim_time_quanta: u64,
    target_cell: CellId,
    initial_policy_choices: &[(CellId, PolicyChoice, Option<Vec<u8>>)],
    candidates: &[PolicyChoice],
) -> Result<Vec<CandidateBranch>, String>
where
    f32: From<B::FloatElem>,
{
    let mut branches = Vec::with_capacity(candidates.len() * options.continuations.len());
    for candidate in candidates {
        for continuation in &options.continuations {
            branches.push(evaluate_candidate_branch(
                model,
                config,
                options,
                device,
                checkpoint.clone(),
                topology.clone(),
                opponent_start,
                sim_time_quanta,
                target_cell,
                initial_policy_choices,
                *candidate,
                *continuation,
            )?);
        }
    }
    Ok(branches)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_candidate_branch<B: Backend>(
    model: &PolicyValueNet<B>,
    config: &TrainingConfig,
    options: &CounterfactualBranchOptions,
    device: &B::Device,
    checkpoint: BlobEnvPlannerCheckpoint,
    topology: ReferenceCompiledTopology,
    opponent_start: OpponentStartingState,
    started_at: u64,
    target_cell: CellId,
    initial_policy_choices: &[(CellId, PolicyChoice, Option<Vec<u8>>)],
    choice: PolicyChoice,
    continuation: BranchContinuation,
) -> Result<CandidateBranch, String>
where
    f32: From<B::FloatElem>,
{
    let (mut env, _) = BlobEnv::from_planner_checkpoint_with_topology_profiled(
        config.env.clone(),
        config.reward.clone(),
        opponent_start,
        topology,
        checkpoint,
    )?;
    env.enable_telemetry(branch_telemetry());
    let mut totals = BranchTotals::default();
    let mut frontiers = 1_u64;
    let mut branch_choices = initial_policy_choices.to_vec();
    let target = branch_choices
        .iter_mut()
        .find(|(cell_id, _, _)| *cell_id == target_cell)
        .ok_or("counterfactual target is absent from the initial policy frontier")?;
    target.1 = choice;
    let mut output = env.step_with_policy_memory(&branch_choices);
    totals.observe(&output)?;
    let immediate_resolution = env
        .cell_state_for_diagnostics(target_cell)?
        .map_or(BranchActionResolution::ActorDied, |cell| {
            branch_resolution(cell.last_outcome.as_ref(), cell.pending_action.is_some())
        });
    let mut horizons = Vec::with_capacity(options.horizon_quanta.len());
    let mut next_horizon = 0_usize;

    loop {
        let elapsed = env.sim_time_quanta().saturating_sub(started_at);
        while next_horizon < options.horizon_quanta.len()
            && elapsed >= options.horizon_quanta[next_horizon]
        {
            horizons.push(snapshot_horizon(
                &env,
                &output,
                target_cell,
                options.horizon_quanta[next_horizon],
                true,
                elapsed,
                frontiers,
                false,
                &totals,
            )?);
            next_horizon += 1;
        }
        if next_horizon == options.horizon_quanta.len() {
            break;
        }
        let host_bound = frontiers >= options.max_frontiers_per_branch;
        if output.done || host_bound {
            while next_horizon < options.horizon_quanta.len() {
                horizons.push(snapshot_horizon(
                    &env,
                    &output,
                    target_cell,
                    options.horizon_quanta[next_horizon],
                    false,
                    elapsed,
                    frontiers,
                    host_bound,
                    &totals,
                )?);
                next_horizon += 1;
            }
            break;
        }

        output = match continuation {
            BranchContinuation::FrozenPolicy => {
                let prepared = env.prepare_training_reference_inputs()?;
                let observations = prepared
                    .iter()
                    .map(|(cell_id, input)| PolicyObservation {
                        cell_id: *cell_id,
                        observation: Observation::from_reference(input),
                        private_memory: input.private_memory.clone(),
                    })
                    .collect::<Vec<_>>();
                let choices = greedy_policy_choices(model, &observations, device);
                env.step_with_policy_memory(&choices)
            }
            BranchContinuation::Teacher => {
                let prepared = env.prepare_training_reference_inputs()?;
                let decisions = prepared
                    .into_iter()
                    .map(|(cell_id, input)| (cell_id, options.teacher.decide(&input)))
                    .collect::<HashMap<_, _>>();
                env.step_with_training_decisions(decisions)
            }
        };
        frontiers = frontiers.saturating_add(1);
        totals.observe(&output)?;
    }

    Ok(CandidateBranch {
        choice,
        continuation,
        immediate_resolution,
        horizons,
    })
}

#[allow(clippy::too_many_arguments)]
fn snapshot_horizon(
    env: &BlobEnv,
    output: &StepOutput,
    target_cell: CellId,
    requested_elapsed_quanta: u64,
    horizon_reached: bool,
    observed_elapsed_quanta: u64,
    controlled_frontiers: u64,
    branch_host_bound_reached: bool,
    totals: &BranchTotals,
) -> Result<BranchHorizonResult, String> {
    let target = env.cell_state_for_diagnostics(target_cell)?;
    let energies = env.side_energy_for_diagnostics()?;
    Ok(BranchHorizonResult {
        requested_elapsed_quanta,
        horizon_reached,
        observed_elapsed_quanta,
        controlled_frontiers,
        target_alive: target.is_some(),
        target_position: target.map(|cell| cell.position.0),
        target_core_mass: target.map_or(0, |cell| cell.core_mass),
        target_assimilated_energy: target.map_or(0, |cell| cell.assimilated_energy),
        target_gut_energy: target.map_or(0, |cell| cell.gut_energy),
        training_cells: output.training_cells,
        opponent_cells: output.opponent_cells,
        training_core_mass: energies[0].core_mass,
        training_assimilated_energy: energies[0].assimilated_energy,
        training_gut_energy: energies[0].gut_energy,
        training_stored_energy: energies[0].stored_energy(),
        training_total_mass_energy: energies[0].total_mass_energy(),
        opponent_core_mass: energies[1].core_mass,
        opponent_assimilated_energy: energies[1].assimilated_energy,
        opponent_gut_energy: energies[1].gut_energy,
        opponent_stored_energy: energies[1].stored_energy(),
        opponent_total_mass_energy: energies[1].total_mass_energy(),
        episode_outcome: output.outcome.map(Into::into),
        branch_host_bound_reached,
        totals: totals.clone(),
    })
}

fn guard_move_candidates(observation: &Observation) -> Vec<PolicyChoice> {
    (0..NUM_ACTIONS)
        .filter(|action| observation.action_mask[*action])
        .filter(|action| {
            decompose_policy_action(*action).is_some_and(|decomposition| {
                matches!(
                    PolicyActionKind::from_index(decomposition.kind),
                    Some(PolicyActionKind::Guard | PolicyActionKind::Move)
                )
            })
        })
        .map(base_choice)
        .collect()
}

const fn base_choice(action: usize) -> PolicyChoice {
    PolicyChoice {
        action,
        amount: 0,
        signal: 0,
        signal_strength: 0,
    }
}

fn context_name(context: ObservationExpertContext) -> &'static str {
    match context {
        ObservationExpertContext::Foraging => "foraging",
        ObservationExpertContext::Interaction => "interaction",
        ObservationExpertContext::Exploration => "exploration",
    }
}

fn summarize_seed_selection(
    states: &[CounterfactualState],
    options: &CounterfactualBranchOptions,
) -> Vec<SeedSelectionSummary> {
    options
        .seeds
        .iter()
        .map(|seed| SeedSelectionSummary {
            seed: *seed,
            selected_states: states.iter().filter(|state| state.seed == *seed).count(),
        })
        .collect()
}

fn summarize_pairwise(
    states: &[CounterfactualState],
    options: &CounterfactualBranchOptions,
) -> Result<Vec<PairwiseHorizonSummary>, String> {
    let mut summaries =
        Vec::with_capacity(options.continuations.len() * options.horizon_quanta.len());
    for continuation in &options.continuations {
        for (horizon_index, requested_elapsed_quanta) in options.horizon_quanta.iter().enumerate() {
            let mut summary = PairwiseHorizonSummary {
                continuation: *continuation,
                requested_elapsed_quanta: *requested_elapsed_quanta,
                states: 0,
                teacher_target_survival_better: 0,
                policy_target_survival_better: 0,
                target_survival_ties: 0,
                teacher_training_population_better: 0,
                policy_training_population_better: 0,
                training_population_ties: 0,
                teacher_target_stored_energy_better: 0,
                policy_target_stored_energy_better: 0,
                target_stored_energy_ties: 0,
                teacher_team_stored_energy_better: 0,
                policy_team_stored_energy_better: 0,
                team_stored_energy_ties: 0,
            };
            for state in states {
                let teacher = find_horizon(
                    state,
                    base_choice(state.teacher_choice.action),
                    *continuation,
                    horizon_index,
                )?;
                let policy = find_horizon(
                    state,
                    base_choice(state.policy_choice.action),
                    *continuation,
                    horizon_index,
                )?;
                summary.states += 1;
                count_ordering(
                    teacher.target_alive.cmp(&policy.target_alive),
                    &mut summary.teacher_target_survival_better,
                    &mut summary.policy_target_survival_better,
                    &mut summary.target_survival_ties,
                );
                count_ordering(
                    teacher.training_cells.cmp(&policy.training_cells),
                    &mut summary.teacher_training_population_better,
                    &mut summary.policy_training_population_better,
                    &mut summary.training_population_ties,
                );
                let teacher_target_energy = teacher
                    .target_assimilated_energy
                    .saturating_add(teacher.target_gut_energy);
                let policy_target_energy = policy
                    .target_assimilated_energy
                    .saturating_add(policy.target_gut_energy);
                count_ordering(
                    teacher_target_energy.cmp(&policy_target_energy),
                    &mut summary.teacher_target_stored_energy_better,
                    &mut summary.policy_target_stored_energy_better,
                    &mut summary.target_stored_energy_ties,
                );
                count_ordering(
                    teacher
                        .training_stored_energy
                        .cmp(&policy.training_stored_energy),
                    &mut summary.teacher_team_stored_energy_better,
                    &mut summary.policy_team_stored_energy_better,
                    &mut summary.team_stored_energy_ties,
                );
            }
            summaries.push(summary);
        }
    }
    Ok(summaries)
}

#[derive(Default)]
struct SeedPairTotals {
    teacher_target_survival: u128,
    policy_target_survival: u128,
    teacher_training_population: u128,
    policy_training_population: u128,
    teacher_target_stored_energy: u128,
    policy_target_stored_energy: u128,
    teacher_team_stored_energy: u128,
    policy_team_stored_energy: u128,
}

fn add_total(total: &mut u128, value: u128) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or("counterfactual seed-balanced metric overflowed")?;
    Ok(())
}

fn summarize_pairwise_by_seed(
    states: &[CounterfactualState],
    options: &CounterfactualBranchOptions,
) -> Result<Vec<PairwiseSeedHorizonSummary>, String> {
    let mut summaries =
        Vec::with_capacity(options.continuations.len() * options.horizon_quanta.len());
    for continuation in &options.continuations {
        for (horizon_index, requested_elapsed_quanta) in options.horizon_quanta.iter().enumerate() {
            let mut summary = PairwiseSeedHorizonSummary {
                continuation: *continuation,
                requested_elapsed_quanta: *requested_elapsed_quanta,
                seeds: 0,
                teacher_target_survival_better: 0,
                policy_target_survival_better: 0,
                target_survival_ties: 0,
                teacher_training_population_better: 0,
                policy_training_population_better: 0,
                training_population_ties: 0,
                teacher_target_stored_energy_better: 0,
                policy_target_stored_energy_better: 0,
                target_stored_energy_ties: 0,
                teacher_team_stored_energy_better: 0,
                policy_team_stored_energy_better: 0,
                team_stored_energy_ties: 0,
            };
            for seed in &options.seeds {
                let seed_states = states
                    .iter()
                    .filter(|state| state.seed == *seed)
                    .collect::<Vec<_>>();
                if seed_states.is_empty() {
                    continue;
                }
                let mut totals = SeedPairTotals::default();
                for state in seed_states {
                    let teacher = find_horizon(
                        state,
                        base_choice(state.teacher_choice.action),
                        *continuation,
                        horizon_index,
                    )?;
                    let policy = find_horizon(
                        state,
                        base_choice(state.policy_choice.action),
                        *continuation,
                        horizon_index,
                    )?;
                    add_total(
                        &mut totals.teacher_target_survival,
                        u128::from(teacher.target_alive),
                    )?;
                    add_total(
                        &mut totals.policy_target_survival,
                        u128::from(policy.target_alive),
                    )?;
                    add_total(
                        &mut totals.teacher_training_population,
                        teacher.training_cells as u128,
                    )?;
                    add_total(
                        &mut totals.policy_training_population,
                        policy.training_cells as u128,
                    )?;
                    add_total(
                        &mut totals.teacher_target_stored_energy,
                        u128::from(teacher.target_assimilated_energy)
                            + u128::from(teacher.target_gut_energy),
                    )?;
                    add_total(
                        &mut totals.policy_target_stored_energy,
                        u128::from(policy.target_assimilated_energy)
                            + u128::from(policy.target_gut_energy),
                    )?;
                    add_total(
                        &mut totals.teacher_team_stored_energy,
                        teacher.training_stored_energy,
                    )?;
                    add_total(
                        &mut totals.policy_team_stored_energy,
                        policy.training_stored_energy,
                    )?;
                }
                summary.seeds += 1;
                count_ordering(
                    totals
                        .teacher_target_survival
                        .cmp(&totals.policy_target_survival),
                    &mut summary.teacher_target_survival_better,
                    &mut summary.policy_target_survival_better,
                    &mut summary.target_survival_ties,
                );
                count_ordering(
                    totals
                        .teacher_training_population
                        .cmp(&totals.policy_training_population),
                    &mut summary.teacher_training_population_better,
                    &mut summary.policy_training_population_better,
                    &mut summary.training_population_ties,
                );
                count_ordering(
                    totals
                        .teacher_target_stored_energy
                        .cmp(&totals.policy_target_stored_energy),
                    &mut summary.teacher_target_stored_energy_better,
                    &mut summary.policy_target_stored_energy_better,
                    &mut summary.target_stored_energy_ties,
                );
                count_ordering(
                    totals
                        .teacher_team_stored_energy
                        .cmp(&totals.policy_team_stored_energy),
                    &mut summary.teacher_team_stored_energy_better,
                    &mut summary.policy_team_stored_energy_better,
                    &mut summary.team_stored_energy_ties,
                );
            }
            summaries.push(summary);
        }
    }
    Ok(summaries)
}

fn find_horizon(
    state: &CounterfactualState,
    choice: PolicyChoice,
    continuation: BranchContinuation,
    horizon_index: usize,
) -> Result<&BranchHorizonResult, String> {
    state
        .branches
        .iter()
        .find(|branch| branch.choice == choice && branch.continuation == continuation)
        .and_then(|branch| branch.horizons.get(horizon_index))
        .ok_or_else(|| "counterfactual summary could not find a required branch horizon".into())
}

fn count_ordering(
    ordering: std::cmp::Ordering,
    teacher_better: &mut usize,
    policy_better: &mut usize,
    ties: &mut usize,
) {
    match ordering {
        std::cmp::Ordering::Greater => *teacher_better += 1,
        std::cmp::Ordering::Less => *policy_better += 1,
        std::cmp::Ordering::Equal => *ties += 1,
    }
}

fn branch_telemetry() -> TelemetryConfig {
    TelemetryConfig {
        enabled: true,
        state_sample_interval_steps: u64::MAX,
        max_state_samples_per_episode: 2,
        episode_log_stride: 0,
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn publish_counterfactual_branch_artifact(
    output: &Path,
    artifact: &CounterfactualBranchArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable counterfactual artifact {}",
            output.display()
        ));
    }
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("counterfactual output needs a UTF-8 file name")?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
    serde_json::to_writer_pretty(&mut file, artifact)
        .map_err(|error| format!("failed to encode counterfactual artifact: {error}"))?;
    file.write_all(b"\n")
        .map_err(|error| format!("failed to finish {}: {error}", temporary.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", temporary.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", temporary.display()))?
        .len();
    if size > MAX_ARTIFACT_BYTES {
        let _ = fs::remove_file(&temporary);
        return Err("counterfactual artifact exceeds its byte bound".into());
    }
    drop(file);
    fs::rename(&temporary, output)
        .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
    let parent_file = File::open(parent)
        .map_err(|error| format!("failed to open {} for sync: {error}", parent.display()))?;
    parent_file
        .sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", parent.display()))?;
    Ok(output.to_path_buf())
}

pub fn load_counterfactual_branch_artifact(
    path: &Path,
) -> Result<CounterfactualBranchArtifact, String> {
    let size = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if size > MAX_ARTIFACT_BYTES {
        return Err(format!(
            "counterfactual artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        ));
    }
    let artifact: CounterfactualBranchArtifact = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::engine::StartingCellLayout;
    use burn::backend::NdArray;

    use crate::config::OpponentProfile;
    use crate::model::PolicyValueNetConfig;

    type TestBackend = NdArray;

    fn tiny_config() -> TrainingConfig {
        let mut config = TrainingConfig::default();
        config.env.world_size = 8;
        config.env.cells_per_team = 1;
        config.env.starting_cell_layout = StartingCellLayout::PairedContact;
        config.env.opponent = OpponentProfile::Wait;
        config.env.num_scattered_energy = 0;
        config.env.num_plants = 0;
        config.env.initial_energy = 100;
        config.env.max_episode_len = 128;
        config.env.victory.sim_time_limit_quanta = 4_096;
        config
    }

    fn options() -> CounterfactualBranchOptions {
        CounterfactualBranchOptions {
            teacher: MaintainedMindProfile::CollisionAwareForager,
            seeds: vec![71],
            max_states: 1,
            max_states_per_seed: 1,
            max_states_per_frontier: 1,
            max_source_frontiers_per_seed: 32,
            horizon_quanta: vec![64, 256],
            continuations: vec![BranchContinuation::FrozenPolicy],
            candidate_scope: BranchCandidateScope::AllLegal,
            explicit_candidate_actions: Vec::new(),
            exploration_only: false,
            max_frontiers_per_branch: 32,
        }
    }

    #[test]
    fn counterfactual_options_reject_ambiguous_or_unbounded_work() {
        let mut invalid = options();
        invalid.seeds.push(71);
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.horizon_quanta = vec![256, 64];
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.continuations.push(BranchContinuation::FrozenPolicy);
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.max_frontiers_per_branch = 0;
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.max_states_per_seed = 2;
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.max_states_per_frontier = 2;
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.max_source_frontiers_per_seed = 0;
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.seeds.push(72);
        assert!(invalid.validate().is_err());
        let mut invalid = options();
        invalid.candidate_scope = BranchCandidateScope::Explicit;
        assert!(invalid.validate().is_err());
        let mut valid = options();
        valid.candidate_scope = BranchCandidateScope::Explicit;
        valid.explicit_candidate_actions = vec![1, 5];
        assert!(valid.validate().is_ok());
        let mut invalid = options();
        invalid.explicit_candidate_actions = vec![1];
        assert!(invalid.validate().is_err());
    }

    fn synthetic_state(
        seed: u64,
        seed_frontier: u64,
        ordinal: usize,
        teacher_energy: u128,
        policy_energy: u128,
    ) -> CounterfactualState {
        let teacher_choice = base_choice(2);
        let policy_choice = base_choice(6);
        let horizon = |training_stored_energy: u128| BranchHorizonResult {
            requested_elapsed_quanta: 64,
            horizon_reached: true,
            observed_elapsed_quanta: 64,
            controlled_frontiers: 1,
            target_alive: true,
            target_position: Some(0),
            target_core_mass: 1,
            target_assimilated_energy: u64::try_from(training_stored_energy).unwrap(),
            target_gut_energy: 0,
            training_cells: 1,
            opponent_cells: 1,
            training_core_mass: 0,
            training_assimilated_energy: training_stored_energy,
            training_gut_energy: 0,
            training_stored_energy,
            training_total_mass_energy: training_stored_energy,
            opponent_core_mass: 0,
            opponent_assimilated_energy: 0,
            opponent_gut_energy: 0,
            opponent_stored_energy: 0,
            opponent_total_mass_energy: 0,
            episode_outcome: None,
            branch_host_bound_reached: false,
            totals: BranchTotals::default(),
        };
        CounterfactualState {
            seed,
            seed_frontier,
            state_ordinal: ordinal,
            sim_time_quanta: seed_frontier,
            source_cell: ordinal as u64,
            checkpoint_sha256: "0".repeat(64),
            mind_observation_sha256: "1".repeat(64),
            mind_evidence: None,
            expert_context: "exploration".into(),
            teacher_choice,
            policy_choice,
            candidates: vec![teacher_choice, policy_choice],
            branches: vec![
                CandidateBranch {
                    choice: teacher_choice,
                    continuation: BranchContinuation::FrozenPolicy,
                    immediate_resolution: BranchActionResolution::Success,
                    horizons: vec![horizon(teacher_energy)],
                },
                CandidateBranch {
                    choice: policy_choice,
                    continuation: BranchContinuation::FrozenPolicy,
                    immediate_resolution: BranchActionResolution::Success,
                    horizons: vec![horizon(policy_energy)],
                },
            ],
        }
    }

    #[test]
    fn seed_balancing_prevents_a_multi_state_seed_from_dominating() {
        let mut options = options();
        options.seeds = vec![71, 72];
        options.max_states = 4;
        options.max_states_per_seed = 2;
        options.horizon_quanta = vec![64];
        let states = vec![
            synthetic_state(71, 1, 0, 20, 10),
            synthetic_state(71, 2, 1, 20, 10),
            synthetic_state(72, 1, 2, 10, 40),
        ];
        let state_summary = summarize_pairwise(&states, &options).unwrap();
        assert_eq!(state_summary[0].teacher_team_stored_energy_better, 2);
        assert_eq!(state_summary[0].policy_team_stored_energy_better, 1);

        let seed_summary = summarize_pairwise_by_seed(&states, &options).unwrap();
        assert_eq!(seed_summary[0].seeds, 2);
        assert_eq!(seed_summary[0].teacher_team_stored_energy_better, 1);
        assert_eq!(seed_summary[0].policy_team_stored_energy_better, 1);
    }

    #[test]
    fn identical_checkpoint_interventions_replay_identically() {
        let _guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let config = tiny_config();
        config.validate().unwrap();
        let device = Default::default();
        let model = PolicyValueNetConfig {
            hidden1: config.model.hidden1,
            hidden2: config.model.hidden2,
            recurrent_size: config.model.recurrent_size,
        }
        .init::<TestBackend>(&device);
        let mut env = BlobEnv::new(config.env.clone(), config.reward.clone(), 71);
        let topology = env
            .reference_simulation_for_diagnostics()
            .unwrap()
            .compiled_topology();
        let prepared = env.prepare_training_reference_inputs().unwrap();
        assert!(prepared.iter().all(|(_, input)| input
            .randomness
            .as_bytes()
            .iter()
            .any(|byte| *byte != 0)));
        let observations = prepared
            .iter()
            .map(|(cell_id, input)| PolicyObservation {
                cell_id: *cell_id,
                observation: Observation::from_reference(input),
                private_memory: input.private_memory.clone(),
            })
            .collect::<Vec<_>>();
        let initial_choices = greedy_policy_choices(&model, &observations, &device);
        let target = prepared[0].0;
        let candidate = guard_move_candidates(&observations[0].observation)
            .into_iter()
            .find(|choice| {
                decompose_policy_action(choice.action)
                    .is_some_and(|action| action.kind == PolicyActionKind::Guard.index())
            })
            .expect("the reference rules expose Guard");
        let (checkpoint, _) = env.planner_checkpoint(None).unwrap();
        let opponent_start = OpponentStartingState {
            cells_per_team: 1,
            initial_energy: config.env.initial_energy,
        };
        let left = evaluate_candidate_branch(
            &model,
            &config,
            &options(),
            &device,
            checkpoint.clone(),
            topology.clone(),
            opponent_start,
            env.sim_time_quanta(),
            target,
            &initial_choices,
            candidate,
            BranchContinuation::FrozenPolicy,
        )
        .unwrap();
        let right = evaluate_candidate_branch(
            &model,
            &config,
            &options(),
            &device,
            checkpoint,
            topology,
            opponent_start,
            env.sim_time_quanta(),
            target,
            &initial_choices,
            candidate,
            BranchContinuation::FrozenPolicy,
        )
        .unwrap();
        assert_eq!(left, right);
        assert_eq!(left.horizons.len(), 2);
        assert!(left.horizons.last().unwrap().controlled_frontiers < 32);
    }
}
