//! Exact simulator branching primitives for bounded single-cell planning.
//!
//! This module intentionally stops at the search boundary. It provides exact,
//! reproducible successor states from complete host continuation checkpoints;
//! beam search, dynamic programming, and information-set planning can share
//! this transition layer without reimplementing physics.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use blob_engine::resolution::{
    CellKey, ReferenceCompiledTopology, ReferenceRuleset, ReferenceStateRestoreProfile,
};
use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::ReferenceMindInput;
use blob_interface::reference_mind_converter::{
    reference_mind_input_to_capnp, ReferenceMindLimits,
};

use crate::action::{
    attach_policy_memory, decode_policy_choice, decompose_policy_action, policy_action_family,
    policy_masks, PolicyActionFamily, PolicyActionKind, PolicyChoice, NUM_AMOUNT_CHOICES,
};
use crate::config::{EnvConfig, RewardConfig};
use crate::env::{BlobEnv, BlobEnvPlannerCheckpoint, EpisodeOutcome, OpponentStartingState};
use crate::micro_combat::{scenario_environment, MicroCombatObjective, MicroCombatScenario};

pub const MICRO_COMBAT_BOUNDED_SEARCH_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone)]
pub struct SingleCellSearchState {
    pub(crate) checkpoint: BlobEnvPlannerCheckpoint,
    pub continuation_sha256: String,
    pub sim_time_quanta: u64,
    pub training_cells: usize,
    pub opponent_cells: usize,
    pub training_stored_energy: u128,
    pub opponent_stored_energy: u128,
    pub terminal_outcome: Option<EpisodeOutcome>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PlannerRestoreAggregate {
    pub count: u64,
    pub tile_materialization_ns: u128,
    pub cell_materialization_ns: u128,
    pub hash_seed_materialization_ns: u128,
    pub resolver_reconstruction_ns: u128,
    pub topology_validation_ns: u128,
    pub cell_validation_store_and_passive_index_ns: u128,
    pub tile_validation_and_passive_index_ns: u128,
    pub hash_initialization_ns: u128,
    pub scratch_initialization_ns: u128,
    pub metabolic_index_ns: u128,
    pub integrity_validation_ns: u128,
    pub environment_total_ns: u128,
    pub total_ns: u128,
    pub restored_tiles: u128,
    pub restored_cells: u128,
}

impl PlannerRestoreAggregate {
    fn record(&mut self, profile: ReferenceStateRestoreProfile, environment_total_ns: u128) {
        self.count = self.count.saturating_add(1);
        self.tile_materialization_ns = self
            .tile_materialization_ns
            .saturating_add(u128::from(profile.tile_materialization_ns));
        self.cell_materialization_ns = self
            .cell_materialization_ns
            .saturating_add(u128::from(profile.cell_materialization_ns));
        self.hash_seed_materialization_ns = self
            .hash_seed_materialization_ns
            .saturating_add(u128::from(profile.hash_seed_materialization_ns));
        self.resolver_reconstruction_ns = self
            .resolver_reconstruction_ns
            .saturating_add(u128::from(profile.resolver_reconstruction_ns));
        self.topology_validation_ns = self
            .topology_validation_ns
            .saturating_add(u128::from(profile.topology_validation_ns));
        self.cell_validation_store_and_passive_index_ns = self
            .cell_validation_store_and_passive_index_ns
            .saturating_add(u128::from(
                profile.cell_validation_store_and_passive_index_ns,
            ));
        self.tile_validation_and_passive_index_ns = self
            .tile_validation_and_passive_index_ns
            .saturating_add(u128::from(profile.tile_validation_and_passive_index_ns));
        self.hash_initialization_ns = self
            .hash_initialization_ns
            .saturating_add(u128::from(profile.hash_initialization_ns));
        self.scratch_initialization_ns = self
            .scratch_initialization_ns
            .saturating_add(u128::from(profile.scratch_initialization_ns));
        self.metabolic_index_ns = self
            .metabolic_index_ns
            .saturating_add(u128::from(profile.metabolic_index_ns));
        self.integrity_validation_ns = self
            .integrity_validation_ns
            .saturating_add(u128::from(profile.integrity_validation_ns));
        self.environment_total_ns = self
            .environment_total_ns
            .saturating_add(environment_total_ns);
        self.total_ns = self.total_ns.saturating_add(u128::from(profile.total_ns));
        self.restored_tiles = self
            .restored_tiles
            .saturating_add(profile.tile_count as u128);
        self.restored_cells = self
            .restored_cells
            .saturating_add(profile.cell_count as u128);
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.count = self.count.saturating_add(other.count);
        self.tile_materialization_ns = self
            .tile_materialization_ns
            .saturating_add(other.tile_materialization_ns);
        self.cell_materialization_ns = self
            .cell_materialization_ns
            .saturating_add(other.cell_materialization_ns);
        self.hash_seed_materialization_ns = self
            .hash_seed_materialization_ns
            .saturating_add(other.hash_seed_materialization_ns);
        self.resolver_reconstruction_ns = self
            .resolver_reconstruction_ns
            .saturating_add(other.resolver_reconstruction_ns);
        self.topology_validation_ns = self
            .topology_validation_ns
            .saturating_add(other.topology_validation_ns);
        self.cell_validation_store_and_passive_index_ns = self
            .cell_validation_store_and_passive_index_ns
            .saturating_add(other.cell_validation_store_and_passive_index_ns);
        self.tile_validation_and_passive_index_ns = self
            .tile_validation_and_passive_index_ns
            .saturating_add(other.tile_validation_and_passive_index_ns);
        self.hash_initialization_ns = self
            .hash_initialization_ns
            .saturating_add(other.hash_initialization_ns);
        self.scratch_initialization_ns = self
            .scratch_initialization_ns
            .saturating_add(other.scratch_initialization_ns);
        self.metabolic_index_ns = self
            .metabolic_index_ns
            .saturating_add(other.metabolic_index_ns);
        self.integrity_validation_ns = self
            .integrity_validation_ns
            .saturating_add(other.integrity_validation_ns);
        self.environment_total_ns = self
            .environment_total_ns
            .saturating_add(other.environment_total_ns);
        self.total_ns = self.total_ns.saturating_add(other.total_ns);
        self.restored_tiles = self.restored_tiles.saturating_add(other.restored_tiles);
        self.restored_cells = self.restored_cells.saturating_add(other.restored_cells);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BoundedSearchConfig {
    pub beam_width: usize,
    pub max_depth: usize,
    pub max_expansions: usize,
    /// Remove effort variants with exactly the same cost and completion time
    /// for the same action, target, and amount under the current state/rules.
    /// A merely more expensive effort is not considered dominated: spending
    /// energy also sheds inertial mass and deposits energy into the world.
    pub equivalent_effort_pruning: bool,
}

impl Default for BoundedSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 128,
            max_depth: 64,
            max_expansions: 100_000,
            equivalent_effort_pruning: true,
        }
    }
}

impl BoundedSearchConfig {
    pub fn validate(self) -> Result<Self, String> {
        if self.beam_width == 0 || self.max_depth == 0 || self.max_expansions == 0 {
            return Err("bounded-search limits must be positive".into());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlannedChoice {
    pub action: usize,
    pub amount: usize,
    pub family: String,
}

impl From<PolicyChoice> for PlannedChoice {
    fn from(choice: PolicyChoice) -> Self {
        Self {
            action: choice.action,
            amount: choice.amount,
            family: policy_action_family(choice.action)
                .map_or("unknown", PolicyActionFamily::name)
                .to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BoundedSearchReport {
    pub schema_version: u32,
    pub evidence_class: String,
    pub search_domain: String,
    pub scenario: String,
    pub seed: u64,
    pub objective: MicroCombatObjective,
    pub config: BoundedSearchConfig,
    pub expansions: usize,
    pub unique_states: usize,
    pub depth_reached: usize,
    pub root_choices_before_pruning: usize,
    pub root_choices_after_pruning: usize,
    pub root_family_choices_before_pruning: BTreeMap<String, usize>,
    pub root_family_choices_after_pruning: BTreeMap<String, usize>,
    /// Beam pruning or the expansion cap discarded reachable work.
    pub truncated: bool,
    /// Every reachable branch was explored only when true.
    pub exhaustive: bool,
    pub objective_success: bool,
    pub final_outcome: String,
    pub final_sim_time_quanta: u64,
    pub final_training_cells: usize,
    pub final_opponent_cells: usize,
    pub final_training_stored_energy: u128,
    pub final_opponent_stored_energy: u128,
    pub final_continuation_sha256: String,
    pub sequence: Vec<PlannedChoice>,
}

#[derive(Clone)]
struct SearchNode {
    state: SingleCellSearchState,
    sequence: Vec<PolicyChoice>,
}

impl SingleCellSearchState {
    pub fn done(&self) -> bool {
        self.terminal_outcome.is_some()
    }

    pub fn objective_success(&self, objective: MicroCombatObjective) -> bool {
        match objective {
            MicroCombatObjective::Survival => {
                self.done()
                    && self.training_cells > 0
                    && self.terminal_outcome != Some(EpisodeOutcome::SafetyAbort)
            }
            MicroCombatObjective::Elimination => self.terminal_outcome == Some(EpisodeOutcome::Win),
        }
    }
}

/// Exact generative model for a scenario with one non-reproducing controlled
/// cell. Split and explicit Signal are excluded from the search catalog:
/// allowing Split changes the problem into decentralized multi-cell planning,
/// while Signal is physically dominated for a solitary cell in these no-food
/// micro scenarios. All other masked physical actions and amounts are retained.
pub struct SingleCellSearchSimulator {
    environment: EnvConfig,
    reward: RewardConfig,
    opponent_start: OpponentStartingState,
    objective: MicroCombatObjective,
    root: SingleCellSearchState,
    topology: ReferenceCompiledTopology,
    restore_profile: Mutex<PlannerRestoreAggregate>,
}

/// Mind-visible frontier data derived from one restored continuation. The
/// full catalog is part of the observation abstraction; the search catalog
/// may additionally remove effort variants that are physically equivalent.
pub(crate) struct SingleCellFrontier {
    pub input: ReferenceMindInput,
    pub legal_choices: Vec<PolicyChoice>,
    pub search_choices: Vec<PolicyChoice>,
}

impl SingleCellSearchSimulator {
    pub fn new(
        base: &EnvConfig,
        reward: &RewardConfig,
        scenario: &MicroCombatScenario,
        seed: u64,
    ) -> Result<Self, String> {
        if scenario.training_cells != 1 {
            return Err("single-cell planner requires exactly one controlled founder".into());
        }
        let environment = scenario_environment(base, scenario);
        let opponent_start = OpponentStartingState {
            cells_per_team: scenario.opponent_cells,
            initial_energy: scenario.opponent_initial_energy,
        };
        let env = BlobEnv::new_with_opponent_starting_state(
            environment.clone(),
            reward.clone(),
            seed,
            opponent_start,
        );
        let topology = env
            .reference_simulation_for_diagnostics()?
            .compiled_topology();
        let root = state_from_env(&env, None, scenario.opponent_cells, None)?;
        Ok(Self {
            environment,
            reward: reward.clone(),
            opponent_start,
            objective: scenario.objective,
            root,
            topology,
            restore_profile: Mutex::new(PlannerRestoreAggregate::default()),
        })
    }

    pub fn root(&self) -> &SingleCellSearchState {
        &self.root
    }

    pub fn objective(&self) -> MicroCombatObjective {
        self.objective
    }

    pub(crate) fn restore_profile(&self) -> PlannerRestoreAggregate {
        *self
            .restore_profile
            .lock()
            .expect("planner restore profile mutex was poisoned")
    }

    pub fn search_bounded(
        &self,
        scenario_name: &str,
        seed: u64,
        config: BoundedSearchConfig,
    ) -> Result<BoundedSearchReport, String> {
        let config = config.validate()?;
        let root_frontier = self.frontier(&self.root, config.equivalent_effort_pruning)?;
        let root_choices_before = root_frontier.legal_choices;
        let root_choices_after = root_frontier.search_choices;
        let root = SearchNode {
            state: self.root.clone(),
            sequence: Vec::new(),
        };
        let mut best = root.clone();
        let mut frontier = vec![root];
        let mut seen = HashSet::from([self.root.continuation_sha256.clone()]);
        let mut expansions = 0usize;
        let mut depth_reached = 0usize;
        let mut truncated = false;

        'depths: for depth in 0..config.max_depth {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            let current = std::mem::take(&mut frontier);
            for node in current {
                for choice in
                    self.legal_choices_for_search(&node.state, config.equivalent_effort_pruning)?
                {
                    if expansions == config.max_expansions {
                        truncated = true;
                        break 'depths;
                    }
                    expansions += 1;
                    let state = self.transition_known_legal(&node.state, choice)?;
                    if !seen.insert(state.continuation_sha256.clone()) {
                        continue;
                    }
                    let mut sequence = node.sequence.clone();
                    sequence.push(choice);
                    let child = SearchNode { state, sequence };
                    depth_reached = depth + 1;
                    if compare_nodes(&child, &best, self.objective).is_gt() {
                        best = child.clone();
                    }
                    if !child.state.done() {
                        next.push(child);
                    }
                }
            }
            next.sort_unstable_by(|left, right| {
                compare_nodes(right, left, self.objective).then_with(|| {
                    left.state
                        .continuation_sha256
                        .cmp(&right.state.continuation_sha256)
                })
            });
            if next.len() > config.beam_width {
                next.truncate(config.beam_width);
                truncated = true;
            }
            frontier = next;
        }
        if !frontier.is_empty() {
            truncated = true;
        }

        let replayed = self.replay(&best.sequence)?;
        if replayed.continuation_sha256 != best.state.continuation_sha256 {
            return Err(
                "bounded-search sequence did not replay to its selected continuation".into(),
            );
        }
        Ok(BoundedSearchReport {
            schema_version: MICRO_COMBAT_BOUNDED_SEARCH_SCHEMA_VERSION,
            evidence_class: "privileged_fixed_seed_feasibility".into(),
            search_domain: "single_cell_no_split_no_signal".into(),
            scenario: scenario_name.to_string(),
            seed,
            objective: self.objective,
            config,
            expansions,
            unique_states: seen.len(),
            depth_reached,
            root_choices_before_pruning: root_choices_before.len(),
            root_choices_after_pruning: root_choices_after.len(),
            root_family_choices_before_pruning: family_counts(&root_choices_before),
            root_family_choices_after_pruning: family_counts(&root_choices_after),
            truncated,
            exhaustive: !truncated && frontier.is_empty(),
            objective_success: best.state.objective_success(self.objective),
            final_outcome: outcome_name(best.state.terminal_outcome).to_string(),
            final_sim_time_quanta: best.state.sim_time_quanta,
            final_training_cells: best.state.training_cells,
            final_opponent_cells: best.state.opponent_cells,
            final_training_stored_energy: best.state.training_stored_energy,
            final_opponent_stored_energy: best.state.opponent_stored_energy,
            final_continuation_sha256: best.state.continuation_sha256,
            sequence: best.sequence.into_iter().map(PlannedChoice::from).collect(),
        })
    }

    pub fn replay(&self, sequence: &[PolicyChoice]) -> Result<SingleCellSearchState, String> {
        let mut state = self.root.clone();
        for choice in sequence {
            state = self.transition(&state, *choice)?;
        }
        Ok(state)
    }

    pub fn legal_choices(
        &self,
        state: &SingleCellSearchState,
    ) -> Result<Vec<PolicyChoice>, String> {
        if state.done() {
            return Ok(Vec::new());
        }
        Ok(self.frontier(state, false)?.legal_choices)
    }

    pub(crate) fn frontier(
        &self,
        state: &SingleCellSearchState,
        equivalent_effort_pruning: bool,
    ) -> Result<SingleCellFrontier, String> {
        if state.done() {
            return Err("terminal planner state has no Mind observation".into());
        }
        let env = self.restore(state)?;
        let input = frontier_input(&env)?;
        let legal_choices = legal_choices_from_input(&input)?;
        let search_choices = if equivalent_effort_pruning {
            let simulation = env.reference_simulation_for_diagnostics()?;
            prune_equivalent_efforts(legal_choices.clone(), &input, simulation.rules())
        } else {
            legal_choices.clone()
        };
        Ok(SingleCellFrontier {
            input,
            legal_choices,
            search_choices,
        })
    }

    /// Return the exact anonymous Mind input at this frontier, with randomness
    /// zeroed because random samples are policy noise rather than observable
    /// physical state. Private memory remains part of the input.
    #[cfg(test)]
    pub(crate) fn observation_input(
        &self,
        state: &SingleCellSearchState,
    ) -> Result<ReferenceMindInput, String> {
        Ok(self.frontier(state, false)?.input)
    }

    pub(crate) fn legal_choices_for_search(
        &self,
        state: &SingleCellSearchState,
        equivalent_effort_pruning: bool,
    ) -> Result<Vec<PolicyChoice>, String> {
        if state.done() {
            return Ok(Vec::new());
        }
        Ok(self
            .frontier(state, equivalent_effort_pruning)?
            .search_choices)
    }

    pub fn transition(
        &self,
        state: &SingleCellSearchState,
        choice: PolicyChoice,
    ) -> Result<SingleCellSearchState, String> {
        if state.done() {
            return Err("planner choice is not legal at this decision frontier".into());
        }
        let mut env = self.restore(state)?;
        let input = frontier_input(&env)?;
        if !legal_choices_from_input(&input)?.contains(&choice) {
            return Err("planner choice is not legal at this decision frontier".into());
        }
        transition_restored_env(&mut env, state, choice)
    }

    fn transition_known_legal(
        &self,
        state: &SingleCellSearchState,
        choice: PolicyChoice,
    ) -> Result<SingleCellSearchState, String> {
        let mut env = self.restore(state)?;
        transition_restored_env(&mut env, state, choice)
    }

    /// Advance through the real anonymous Mind-input path. This consumes the
    /// controlled cell's private random block exactly as native/Wasm dispatch
    /// does, but the deterministic table key deliberately ignores its value.
    pub(crate) fn transition_observation_choice(
        &self,
        state: &SingleCellSearchState,
        expected_observation_sha256: &str,
        choice: PolicyChoice,
        next_private_memory: Vec<u8>,
    ) -> Result<SingleCellSearchState, String> {
        if state.done() {
            return Err("cannot advance a terminal observation-policy particle".into());
        }
        let mut env = self.restore(state)?;
        let mut prepared = env.prepare_training_reference_inputs()?;
        if prepared.len() != 1 {
            return Err(format!(
                "single-cell observation policy received {} ready controlled cells",
                prepared.len()
            ));
        }
        let (cell_id, input) = prepared.pop().unwrap();
        let actual_key = mind_observation_sha256(&input)?;
        if actual_key != expected_observation_sha256 {
            return Err(format!(
                "observation-policy key mismatch: expected {expected_observation_sha256}, got {actual_key}"
            ));
        }
        if !legal_choices_from_input(&input)?.contains(&choice) {
            return Err("observation-policy choice is not legal at this frontier".into());
        }
        if next_private_memory.len() > input.action_space.max_private_memory_bytes as usize {
            return Err("observation-policy memory exceeds the Mind ABI limit".into());
        }
        let mut decision = decode_policy_choice(choice, &input);
        attach_policy_memory(&mut decision, next_private_memory);
        let output = env.step_with_training_decisions(HashMap::from([(cell_id, decision)]));
        state_from_env(&env, output.outcome, output.opponent_cells, Some(state))
    }

    fn restore(&self, state: &SingleCellSearchState) -> Result<BlobEnv, String> {
        let restore_started = Instant::now();
        let (env, profile) = BlobEnv::from_planner_checkpoint_with_topology_profiled(
            self.environment.clone(),
            self.reward.clone(),
            self.opponent_start,
            self.topology.clone(),
            state.checkpoint.clone(),
        )?;
        self.restore_profile
            .lock()
            .expect("planner restore profile mutex was poisoned")
            .record(profile, restore_started.elapsed().as_nanos());
        Ok(env)
    }
}

fn legal_choices_from_input(input: &ReferenceMindInput) -> Result<Vec<PolicyChoice>, String> {
    let masks = policy_masks(input);
    let mut choices = Vec::new();
    for (action, allowed) in masks.actions.iter().copied().enumerate() {
        if !allowed
            || matches!(
                policy_action_family(action),
                Some(PolicyActionFamily::Split | PolicyActionFamily::Signal) | None
            )
        {
            continue;
        }
        for amount in 0..NUM_AMOUNT_CHOICES {
            if masks.amount_choice_bits[action] & (1 << amount) != 0 {
                choices.push(PolicyChoice {
                    action,
                    amount,
                    signal: 0,
                    signal_strength: 0,
                });
            }
        }
    }
    if choices.is_empty() {
        return Err("single-cell search frontier has no legal physical choice".into());
    }
    Ok(choices)
}

fn frontier_input(env: &BlobEnv) -> Result<ReferenceMindInput, String> {
    let ready = env.ready_training_cell_ids();
    if ready.len() != 1 {
        return Err(format!(
            "single-cell search frontier contains {} ready controlled cells",
            ready.len()
        ));
    }
    let cell_id = ready[0];
    let actor = CellKey(
        u64::try_from(cell_id.0).map_err(|_| "planner cell identity does not fit canonical key")?,
    );
    env.reference_simulation_for_diagnostics()?
        .observation_batch()
        .reference_mind_input(actor, PrivateRandom::ZERO)
        .map_err(|error| format!("failed to reconstruct planner Mind input: {error}"))
}

fn transition_restored_env(
    env: &mut BlobEnv,
    state: &SingleCellSearchState,
    choice: PolicyChoice,
) -> Result<SingleCellSearchState, String> {
    let ready = env.ready_training_cell_ids();
    let cell_id = ready
        .first()
        .ok_or("planner frontier lost its controlled cell")?
        .to_owned();
    let output = env.step_with_policy_memory(&[(cell_id, choice, None)]);
    state_from_env(env, output.outcome, output.opponent_cells, Some(state))
}

/// Hash exactly the Mind-visible state and private memory, deliberately
/// excluding the supplied random sample. A deterministic policy may ignore
/// randomness while remaining fully ABI legal; hidden canonical state, team
/// identity, cell identity, and global coordinates never enter this key.
pub fn mind_observation_sha256(input: &ReferenceMindInput) -> Result<String, String> {
    let mut canonical = input.clone();
    canonical.randomness = PrivateRandom::ZERO;
    let bytes = reference_mind_input_to_capnp(&canonical, ReferenceMindLimits::default())
        .map_err(|error| format!("failed to encode Mind observation: {error}"))?;
    let mut digest = Sha256::new();
    digest.update(b"blob.observation-policy.key.v1");
    digest.update(bytes);
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn family_counts(choices: &[PolicyChoice]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for choice in choices {
        let family = policy_action_family(choice.action)
            .map_or("unknown", PolicyActionFamily::name)
            .to_string();
        *counts.entry(family).or_default() += 1;
    }
    counts
}

fn prune_equivalent_efforts(
    choices: Vec<PolicyChoice>,
    input: &ReferenceMindInput,
    rules: &ReferenceRuleset,
) -> Vec<PolicyChoice> {
    let metrics = choices
        .iter()
        .map(|choice| effort_cost_and_duration(*choice, input, rules))
        .collect::<Vec<_>>();
    choices
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, choice)| {
            let decomposition = decompose_policy_action(choice.action)?;
            let kind = PolicyActionKind::from_index(decomposition.kind)?;
            if !kind.uses_effort() {
                return Some(choice);
            }
            let Some((cost, duration)) = metrics[index] else {
                return Some(choice);
            };
            let equivalent_predecessor = choices.iter().enumerate().any(|(other_index, other)| {
                if index == other_index || choice.amount != other.amount {
                    return false;
                }
                let Some(other_decomposition) = decompose_policy_action(other.action) else {
                    return false;
                };
                if decomposition.kind != other_decomposition.kind
                    || decomposition.target != other_decomposition.target
                {
                    return false;
                }
                let Some((other_cost, other_duration)) = metrics[other_index] else {
                    return false;
                };
                other_cost == cost && other_duration == duration && other.action < choice.action
            });
            (!equivalent_predecessor).then_some(choice)
        })
        .collect()
}

fn effort_cost_and_duration(
    choice: PolicyChoice,
    input: &ReferenceMindInput,
    rules: &ReferenceRuleset,
) -> Option<(u64, u64)> {
    let decomposition = decompose_policy_action(choice.action)?;
    let kind = PolicyActionKind::from_index(decomposition.kind)?;
    if !kind.uses_effort() {
        return None;
    }
    let profile = rules.effort_profiles.get(decomposition.effort)?;
    let mass = u128::from(input.self_state.core_mass)
        .checked_add(u128::from(input.self_state.assimilated_energy))?
        .checked_add(u128::from(input.self_state.gut_energy))?
        .checked_add(u128::from(input.self_state.carried_material_mass))?;
    let target_distance_q10 = || {
        input
            .slots
            .iter()
            .find(|slot| usize::from(slot.slot) == decomposition.target)
            .map(|slot| u64::from(slot.distance_cost_q10))
    };
    let raw_cost = match kind {
        PolicyActionKind::Move => {
            let denominator = rules.move_mass_units_per_effort.checked_mul(1024)?;
            rules.move_effort_base.checked_add(ceil_mul_u128(
                mass,
                target_distance_q10()?,
                denominator,
            )?)?
        }
        PolicyActionKind::Attack => rules.attack_effort_base,
        PolicyActionKind::Guard => rules.guard_effort_base,
        _ => return None,
    };
    let cost = ceil_mul(
        raw_cost,
        u64::from(profile.cost_numerator),
        u64::from(profile.cost_denominator),
    )?;
    let duration = if kind == PolicyActionKind::Guard {
        rules.time.decision_interval_floor
    } else {
        let duration_rule = match kind {
            PolicyActionKind::Move => rules.move_duration,
            PolicyActionKind::Attack => rules.attack_duration,
            _ => return None,
        };
        let mass_term = ceil_mul_u128(
            mass,
            duration_rule.mass_quanta_numerator,
            duration_rule.mass_units_denominator,
        )?;
        let mut duration = duration_rule.base_quanta.checked_add(mass_term)?;
        if kind == PolicyActionKind::Move {
            duration = ceil_mul(duration, target_distance_q10()?, 1024)?;
        }
        duration = ceil_mul(
            duration,
            u64::from(profile.duration_numerator),
            u64::from(profile.duration_denominator),
        )?;
        round_up(
            duration.max(rules.time.decision_interval_floor),
            rules.time.completion_bucket,
        )?
    };
    Some((cost, duration))
}

fn ceil_mul(value: u64, numerator: u64, denominator: u64) -> Option<u64> {
    let product = value.checked_mul(numerator)?;
    let adjusted = product.checked_add(denominator.checked_sub(1)?)?;
    Some(adjusted / denominator)
}

fn ceil_mul_u128(value: u128, numerator: u64, denominator: u64) -> Option<u64> {
    let product = value.checked_mul(u128::from(numerator))?;
    let adjusted = product.checked_add(u128::from(denominator.checked_sub(1)?))?;
    u64::try_from(adjusted / u128::from(denominator)).ok()
}

fn round_up(value: u64, bucket: u64) -> Option<u64> {
    ceil_mul(value, 1, bucket)?.checked_mul(bucket)
}

fn state_from_env(
    env: &BlobEnv,
    terminal_outcome: Option<EpisodeOutcome>,
    opponent_cells: usize,
    parent: Option<&SingleCellSearchState>,
) -> Result<SingleCellSearchState, String> {
    let (checkpoint, continuation_identity) =
        env.planner_checkpoint(parent.map(|parent| &parent.checkpoint))?;
    let (training_stored_energy, opponent_stored_energy) = stored_energy_by_side(env, &checkpoint)?;
    Ok(SingleCellSearchState {
        continuation_sha256: hex_digest(&continuation_identity),
        checkpoint,
        sim_time_quanta: env.sim_time_quanta(),
        training_cells: env.training_cells_alive(),
        opponent_cells,
        training_stored_energy,
        opponent_stored_energy,
        terminal_outcome,
    })
}

fn stored_energy_by_side(
    env: &BlobEnv,
    checkpoint: &BlobEnvPlannerCheckpoint,
) -> Result<(u128, u128), String> {
    let training = checkpoint
        .host_cells
        .iter()
        .filter_map(|cell| {
            (cell.team_id.0 == 0)
                .then(|| u64::try_from(cell.id.0).ok())
                .flatten()
        })
        .collect::<HashSet<_>>();
    let simulation = env.reference_simulation_for_diagnostics()?;
    let mut energies = (0_u128, 0_u128);
    for (key, cell) in simulation.cells() {
        let stored = u128::from(cell.assimilated_energy) + u128::from(cell.gut_energy);
        if training.contains(&key.0) {
            energies.0 = energies.0.saturating_add(stored);
        } else {
            energies.1 = energies.1.saturating_add(stored);
        }
    }
    Ok(energies)
}

fn compare_nodes(
    left: &SearchNode,
    right: &SearchNode,
    objective: MicroCombatObjective,
) -> Ordering {
    let success = |node: &SearchNode| node.state.objective_success(objective);
    match objective {
        MicroCombatObjective::Survival => success(left)
            .cmp(&success(right))
            .then_with(|| (left.state.training_cells > 0).cmp(&(right.state.training_cells > 0)))
            .then_with(|| left.state.sim_time_quanta.cmp(&right.state.sim_time_quanta))
            .then_with(|| {
                left.state
                    .training_stored_energy
                    .cmp(&right.state.training_stored_energy)
            })
            .then_with(|| right.sequence.len().cmp(&left.sequence.len())),
        MicroCombatObjective::Elimination => success(left)
            .cmp(&success(right))
            .then_with(|| (left.state.training_cells > 0).cmp(&(right.state.training_cells > 0)))
            .then_with(|| right.state.opponent_cells.cmp(&left.state.opponent_cells))
            .then_with(|| {
                right
                    .state
                    .opponent_stored_energy
                    .cmp(&left.state.opponent_stored_energy)
            })
            .then_with(|| {
                left.state
                    .training_stored_energy
                    .cmp(&right.state.training_stored_energy)
            })
            .then_with(|| right.state.sim_time_quanta.cmp(&left.state.sim_time_quanta)),
    }
}

fn outcome_name(outcome: Option<EpisodeOutcome>) -> &'static str {
    match outcome {
        None => "incomplete",
        Some(EpisodeOutcome::Win) => "win",
        Some(EpisodeOutcome::Loss) => "loss",
        Some(EpisodeOutcome::Timeout) => "timeout",
        Some(EpisodeOutcome::SafetyAbort) => "safety_abort",
    }
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn publish_bounded_search_report(
    output: &Path,
    report: &BoundedSearchReport,
) -> Result<PathBuf, String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode bounded-search report: {error}"))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create bounded-search directory: {error}"))?;
    }
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, bytes)
        .map_err(|error| format!("failed to write bounded-search report: {error}"))?;
    fs::rename(&temporary, output)
        .map_err(|error| format!("failed to publish bounded-search report: {error}"))?;
    Ok(output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{policy_action_family, policy_wait_action};
    use crate::config::OpponentProfile;
    use blob_engine::engine::StartingCellLayout;

    fn scenario() -> MicroCombatScenario {
        MicroCombatScenario {
            name: "planner-branch".into(),
            objective: MicroCombatObjective::Survival,
            world_size: 7,
            training_cells: 1,
            opponent_cells: 1,
            training_initial_energy: 100,
            opponent_initial_energy: 100,
            starting_layout: StartingCellLayout::PairedContact,
            opponent: OpponentProfile::Pursuer,
            sim_time_limit_quanta: 2_048,
        }
    }

    #[test]
    fn exact_frontier_branching_is_reproducible_and_non_mutating() {
        let planner = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            91,
        )
        .unwrap();
        let root = planner.root();
        let root_hash = root.continuation_sha256.clone();
        let choices = planner.legal_choices(root).unwrap();
        assert!(choices.iter().any(|choice| {
            policy_action_family(choice.action) == Some(PolicyActionFamily::Move)
        }));
        assert!(choices.iter().any(|choice| {
            policy_action_family(choice.action) == Some(PolicyActionFamily::Guard)
        }));
        assert!(choices.iter().any(|choice| {
            policy_action_family(choice.action) == Some(PolicyActionFamily::Attack)
        }));
        assert!(!choices.iter().any(|choice| {
            matches!(
                policy_action_family(choice.action),
                Some(PolicyActionFamily::Split | PolicyActionFamily::Signal)
            )
        }));

        let wait = choices
            .iter()
            .copied()
            .find(|choice| choice.action == policy_wait_action())
            .unwrap();
        let left = planner.transition(root, wait).unwrap();
        let right = planner.transition(root, wait).unwrap();
        assert_eq!(left.continuation_sha256, right.continuation_sha256);
        assert_eq!(left.sim_time_quanta, right.sim_time_quanta);
        assert_eq!(left.training_cells, right.training_cells);
        assert_eq!(left.opponent_cells, right.opponent_cells);
        assert_eq!(root.continuation_sha256, root_hash);
        assert_ne!(left.continuation_sha256, root_hash);
    }

    #[test]
    fn frontier_and_observation_transition_each_restore_once() {
        let planner = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            94,
        )
        .unwrap();
        let root = planner.root();
        let before = planner.restore_profile().count;
        let frontier = planner.frontier(root, true).unwrap();
        assert_eq!(planner.restore_profile().count - before, 1);
        assert!(!frontier.legal_choices.is_empty());
        assert!(frontier.search_choices.len() <= frontier.legal_choices.len());

        let exact = mind_observation_sha256(&frontier.input).unwrap();
        let before_transition = planner.restore_profile().count;
        planner
            .transition_observation_choice(root, &exact, frontier.search_choices[0], Vec::new())
            .unwrap();
        assert_eq!(planner.restore_profile().count - before_transition, 1);
    }

    #[test]
    fn bounded_search_finds_and_replays_a_survival_witness() {
        let scenario = scenario();
        let planner = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario,
            92,
        )
        .unwrap();
        let report = planner
            .search_bounded(
                &scenario.name,
                92,
                BoundedSearchConfig {
                    beam_width: 8,
                    max_depth: 4,
                    max_expansions: 2_000,
                    equivalent_effort_pruning: true,
                },
            )
            .unwrap();
        assert!(report.objective_success);
        assert_eq!(report.final_outcome, "timeout");
        assert_eq!(report.final_sim_time_quanta, 2_048);
        assert!(!report.sequence.is_empty());
        assert!(report.expansions > 0);
        assert!(report.root_choices_after_pruning <= report.root_choices_before_pruning);
    }

    #[test]
    fn effort_pruning_only_removes_exact_cost_duration_equivalents() {
        let planner = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            93,
        )
        .unwrap();
        let root = planner.root();
        let all = planner.legal_choices(root).unwrap();
        let pruned = planner.legal_choices_for_search(root, true).unwrap();
        assert!(pruned.len() <= all.len());

        let env = planner.restore(root).unwrap();
        let actor = CellKey(u64::try_from(env.ready_training_cell_ids()[0].0).unwrap());
        let simulation = env.reference_simulation_for_diagnostics().unwrap();
        let input = simulation
            .observation_batch()
            .reference_mind_input(actor, PrivateRandom::ZERO)
            .unwrap();
        for removed in all.iter().filter(|choice| !pruned.contains(choice)) {
            let removed_parts = decompose_policy_action(removed.action).unwrap();
            let removed_metrics = effort_cost_and_duration(*removed, &input, simulation.rules());
            let retained = pruned
                .iter()
                .find(|retained| {
                    let retained_parts = decompose_policy_action(retained.action).unwrap();
                    removed.amount == retained.amount
                        && removed_parts.kind == retained_parts.kind
                        && removed_parts.target == retained_parts.target
                        && retained.action < removed.action
                        && effort_cost_and_duration(**retained, &input, simulation.rules())
                            == removed_metrics
                })
                .expect("every removed tier must have an exactly equivalent predecessor");
            assert_eq!(
                planner
                    .transition(root, *removed)
                    .unwrap()
                    .continuation_sha256,
                planner
                    .transition(root, *retained)
                    .unwrap()
                    .continuation_sha256,
                "equal cost/duration variants must reach the same continuation",
            );
        }
    }
}
