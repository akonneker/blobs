//! Deterministic micro-characterizations of ecological and combat seams.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::resolution::{
    ActionKind, ActionRequest, EffortTier, LocalSlot, OutcomeStatus, ReferenceCheckpoint,
    ReferenceRuleset, ReferenceSimulation, SimTime, TargetingAction, TileIndex,
};
use blob_interface::types::TeamId;
use serde::{Deserialize, Serialize};

use crate::config::{EnvConfig, RewardConfig, ScenarioProfile};
use crate::env::BlobEnv;
use crate::viability::hash_json;

pub const ECOLOGICAL_CHARACTERIZATION_SCHEMA_VERSION: u32 = 4;
static CHARACTERIZATION_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EcologicalCharacterizationOptions {
    pub seeds: Vec<u64>,
    pub max_micro_actions: usize,
}

impl EcologicalCharacterizationOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.seeds.is_empty() || self.max_micro_actions == 0 {
            return Err(
                "ecological characterization requires seeds and a positive micro-action cap".into(),
            );
        }
        let mut unique = HashSet::with_capacity(self.seeds.len());
        if self.seeds.iter().any(|seed| !unique.insert(*seed)) {
            return Err("ecological characterization seeds must be unique".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CharacterizedEffort {
    Low,
    Standard,
    High,
}

impl CharacterizedEffort {
    const ALL: [Self; 3] = [Self::Low, Self::Standard, Self::High];

    const fn resolver(self) -> EffortTier {
        match self {
            Self::Low => EffortTier::Low,
            Self::Standard => EffortTier::Standard,
            Self::High => EffortTier::High,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Low => 0,
            Self::Standard => 1,
            Self::High => 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistanceDistribution {
    pub samples: usize,
    pub reachable_samples: usize,
    pub unreachable_samples: usize,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub maximum: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FoodDistanceCharacterization {
    pub source_tiles_across_seeds: usize,
    pub move_steps: DistanceDistribution,
    /// Shortest configured movement cost, with 1.0 equal to an orthogonal
    /// unit-cost step in the default neighborhood.
    pub weighted_distance_units: DistanceDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TravelEndurance {
    pub effort: CharacterizedEffort,
    pub completed_moves: u64,
    pub completed_distance_q10: u64,
    pub travel_time_quanta: u64,
    pub death_time_quanta: Option<u64>,
    pub final_assimilated_energy: u64,
    pub censored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AttackVolleyCharacterization {
    pub effort: CharacterizedEffort,
    pub full_strength_payload: Option<u64>,
    pub maximum_local_attackers: usize,
    pub single_attack_lands: bool,
    pub single_attacker_survives: bool,
    pub single_attack_raw_damage: u64,
    pub single_attack_applied_damage: u64,
    pub unguarded_attackers_required: Option<usize>,
    pub guarded_attackers_required: Option<usize>,
    pub unguarded_victim_energy_at_volley: u64,
    pub guarded_victim_energy_at_volley: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeedingCycleCharacterization {
    pub requested_energy: u64,
    pub consumed_energy: u64,
    pub consume_effort_spent: u64,
    pub consume_completion_quanta: u64,
    pub digestion_completion_quanta: Option<u64>,
    pub energy_before_consume: u64,
    pub energy_after_commit: u64,
    pub energy_after_digestion: Option<u64>,
    pub gut_after_consume: u64,
    pub gut_after_digestion: Option<u64>,
    pub survived_full_digestion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReproductionBreakEvenCharacterization {
    pub minimum_child_allocation: u64,
    pub minimum_commit_energy: u64,
    pub minimum_successful_parent_energy: Option<u64>,
    pub split_effort_spent: Option<u64>,
    pub split_completion_quanta: Option<u64>,
    pub parent_energy_after_split: Option<u64>,
    pub child_energy_after_split: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TerrainCycleCharacterization {
    pub initial_energy: u64,
    pub material_mass_per_elevation: u64,
    pub excavation_succeeded: bool,
    pub deposition_succeeded: bool,
    pub excavation_effort_spent: Option<u64>,
    pub deposition_effort_spent: Option<u64>,
    pub total_completion_quanta: u64,
    pub final_energy: Option<u64>,
    pub final_carried_material_mass: Option<u64>,
    pub elevation_restored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalImpulseTrial {
    pub strength_multiplier: u64,
    pub requested_energy: Option<u64>,
    pub accepted: bool,
    pub actor_energy_after_commit: Option<u64>,
    pub action_completion_quanta: Option<u64>,
    pub field_energy_after_commit: u64,
    pub field_energy_after_completion: u64,
    pub decayed_during_action: u64,
    /// Linear-decay time until at most floor(initial / 2) remains.
    pub time_to_half_energy_quanta: Option<u64>,
    pub time_to_extinction_quanta: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalTerrainDisruptionCharacterization {
    pub strength_multiplier: u64,
    pub emitted_energy: u64,
    pub signal_before_terrain_action: u64,
    pub decayed_during_terrain_action: u64,
    pub erased_by_terrain_action: u64,
    pub signal_after_terrain_action: u64,
    pub diffuse_energy_after_terrain_action: u64,
    pub terrain_action_succeeded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalMassSheddingTrial {
    /// Zero is the duration-matched Wait control; positive values are signal
    /// emission quanta.
    pub strength_multiplier: u64,
    pub requested_energy: Option<u64>,
    pub preparation_is_signal: bool,
    pub preparation_accepted: bool,
    pub energy_after_commit: Option<u64>,
    pub energy_after_preparation: Option<u64>,
    pub total_mass_after_preparation: Option<u64>,
    pub preparation_completion_quanta: Option<u64>,
    pub metabolism_during_preparation: Option<u64>,
    pub stationary_death_time_quanta: Option<u64>,
    pub stationary_remaining_lifetime_quanta: Option<u64>,
    pub first_move_effort_spent: Option<u64>,
    pub first_move_duration_quanta: Option<u64>,
    pub total_move_effort_spent: u64,
    pub completed_moves: u64,
    pub completed_distance_q10: u64,
    pub movement_elapsed_quanta: u64,
    pub energy_after_last_completed_move: Option<u64>,
    pub travel_death_time_quanta: Option<u64>,
    pub censored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalCharacterization {
    pub emission_quantum: u64,
    pub neighbor_observation_slots: usize,
    pub minimum_distinct_observable_neighbors: usize,
    pub maximum_distinct_observable_neighbors: usize,
    pub directed_observation_edges: u64,
    pub impulse_trials: Vec<SignalImpulseTrial>,
    pub mass_shedding_trials: Vec<SignalMassSheddingTrial>,
    pub terrain_disruption: Option<SignalTerrainDisruptionCharacterization>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SustainedSiegeTrial {
    pub attackers: usize,
    pub victim_killed: bool,
    pub elapsed_quanta: u64,
    pub completed_attacks: usize,
    pub applied_damage: u64,
    pub completed_consumes: usize,
    pub completed_guards: usize,
    pub surviving_attackers: usize,
    pub victim_final_energy: Option<u64>,
    pub victim_final_gut_energy: Option<u64>,
    pub plant_final_energy: u64,
    pub censored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SustainedSiegeCharacterization {
    pub attacker_effort: CharacterizedEffort,
    pub payload_fraction_numerator: u64,
    pub payload_fraction_denominator: u64,
    /// Scientific horizon shared by every attacker-count trial. The separate
    /// micro-action cap is only a host safety limit.
    pub time_limit_quanta: u64,
    pub maximum_local_attackers: usize,
    pub minimum_attackers_to_kill: Option<usize>,
    pub trials: Vec<SustainedSiegeTrial>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CollapseIndicators {
    pub no_bidirectional_movement: bool,
    pub no_plant_sources: bool,
    pub some_starting_cells_cannot_reach_a_plant: bool,
    pub median_plant_beyond_best_travel_endurance: bool,
    pub p90_plant_beyond_best_travel_endurance: bool,
    pub full_strength_high_attack_fails_to_land: bool,
    pub initial_cell_cannot_establish_standard_guard: bool,
    pub unguarded_initial_cell_survives_maximum_local_high_volley: bool,
    pub guarded_initial_cell_survives_maximum_local_high_volley: bool,
    pub maximum_bite_cannot_be_fully_digested: bool,
    pub minimum_viable_reproduction_is_impossible: bool,
    pub initial_cell_cannot_complete_terrain_cycle: bool,
    pub plant_defender_survives_bounded_maximum_siege: bool,
    pub signaling_disabled: bool,
    pub minimum_signal_is_unaffordable: bool,
    pub minimum_signal_expires_before_its_action_completes: bool,
    pub no_neighbor_signal_observation_edges: bool,
    pub terrain_disruption_probe_unavailable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EcologicalCharacterizationReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub characterization_hash: String,
    pub results_hash: String,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub scenario: ScenarioProfile,
    pub rules: ReferenceRuleset,
    pub options: EcologicalCharacterizationOptions,
    pub stationary_lifetime_quanta: Option<u64>,
    pub travel: Vec<TravelEndurance>,
    pub distance_to_plants: FoodDistanceCharacterization,
    pub distance_to_major_food: FoodDistanceCharacterization,
    pub attack_volleys: Vec<AttackVolleyCharacterization>,
    pub feeding_cycle: FeedingCycleCharacterization,
    pub reproduction_break_even: ReproductionBreakEvenCharacterization,
    pub terrain_cycle: TerrainCycleCharacterization,
    pub signals: SignalCharacterization,
    pub sustained_siege: SustainedSiegeCharacterization,
    pub collapse_indicators: CollapseIndicators,
}

#[derive(Serialize)]
struct CharacterizationIdentity<'a> {
    semantic_ruleset_hash: &'a str,
    compiled_ruleset_hash: &'a str,
    scenario_hash: &'a str,
    options: &'a EcologicalCharacterizationOptions,
}

#[derive(Serialize)]
struct CharacterizationResults<'a> {
    stationary_lifetime_quanta: Option<u64>,
    travel: &'a [TravelEndurance],
    distance_to_plants: &'a FoodDistanceCharacterization,
    distance_to_major_food: &'a FoodDistanceCharacterization,
    attack_volleys: &'a [AttackVolleyCharacterization],
    feeding_cycle: &'a FeedingCycleCharacterization,
    reproduction_break_even: &'a ReproductionBreakEvenCharacterization,
    terrain_cycle: &'a TerrainCycleCharacterization,
    signals: &'a SignalCharacterization,
    sustained_siege: &'a SustainedSiegeCharacterization,
    collapse_indicators: &'a CollapseIndicators,
}

pub fn validate_ecological_characterization(
    report: &EcologicalCharacterizationReport,
) -> Result<(), String> {
    if report.schema_version != ECOLOGICAL_CHARACTERIZATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported ecological characterization schema {}; expected {}",
            report.schema_version, ECOLOGICAL_CHARACTERIZATION_SCHEMA_VERSION
        ));
    }
    report.options.validate()?;
    report
        .rules
        .validate()
        .map_err(|error| format!("invalid characterized ruleset: {error}"))?;
    if report.scenario.semantic_hash()? != report.scenario_hash {
        return Err("ecological characterization scenario hash mismatch".into());
    }
    if report.rules.semantic_hash().to_string() != report.semantic_ruleset_hash {
        return Err("ecological characterization semantic ruleset hash mismatch".into());
    }
    let compiled = ReferenceSimulation::new(
        report.scenario.world_size,
        report.scenario.world_size,
        report.rules.clone(),
    )
    .map_err(|error| format!("failed to compile characterized ruleset: {error}"))?
    .compiled_ruleset_hash()
    .to_string();
    if compiled != report.compiled_ruleset_hash {
        return Err("ecological characterization compiled ruleset hash mismatch".into());
    }
    let expected_hash = hash_json(&CharacterizationIdentity {
        semantic_ruleset_hash: &report.semantic_ruleset_hash,
        compiled_ruleset_hash: &report.compiled_ruleset_hash,
        scenario_hash: &report.scenario_hash,
        options: &report.options,
    })?;
    if expected_hash != report.characterization_hash {
        return Err("ecological characterization identity hash mismatch".into());
    }
    let expected_results_hash = hash_json(&CharacterizationResults {
        stationary_lifetime_quanta: report.stationary_lifetime_quanta,
        travel: &report.travel,
        distance_to_plants: &report.distance_to_plants,
        distance_to_major_food: &report.distance_to_major_food,
        attack_volleys: &report.attack_volleys,
        feeding_cycle: &report.feeding_cycle,
        reproduction_break_even: &report.reproduction_break_even,
        terrain_cycle: &report.terrain_cycle,
        signals: &report.signals,
        sustained_siege: &report.sustained_siege,
        collapse_indicators: &report.collapse_indicators,
    })?;
    if expected_results_hash != report.results_hash {
        return Err("ecological characterization results hash mismatch".into());
    }
    let expected_samples = report
        .options
        .seeds
        .len()
        .checked_mul(report.scenario.cells_per_team)
        .ok_or("ecological characterization sample count overflowed")?;
    for distance in [
        &report.distance_to_plants.move_steps,
        &report.distance_to_plants.weighted_distance_units,
        &report.distance_to_major_food.move_steps,
        &report.distance_to_major_food.weighted_distance_units,
    ] {
        if distance.samples != expected_samples
            || distance.reachable_samples + distance.unreachable_samples != distance.samples
        {
            return Err("ecological characterization distance accounting mismatch".into());
        }
    }
    if report.attack_volleys.len() != CharacterizedEffort::ALL.len()
        || CharacterizedEffort::ALL.iter().any(|effort| {
            report
                .attack_volleys
                .iter()
                .filter(|volley| volley.effort == *effort)
                .count()
                != 1
        })
    {
        return Err("ecological characterization attack effort matrix is incomplete".into());
    }
    if report.feeding_cycle.consumed_energy > report.feeding_cycle.requested_energy
        || report.feeding_cycle.gut_after_digestion == Some(0)
            && !report.feeding_cycle.survived_full_digestion
    {
        return Err("ecological characterization feeding-cycle accounting is invalid".into());
    }
    if report
        .reproduction_break_even
        .minimum_successful_parent_energy
        .is_some_and(|energy| energy < report.reproduction_break_even.minimum_commit_energy)
    {
        return Err("ecological characterization reproduction break-even is invalid".into());
    }
    let expected_signal_multipliers = [1, 2, 4, 8, 16];
    let expected_shedding_multipliers = [0, 1, 2, 4, 8, 16];
    if report.signals.emission_quantum != report.rules.signal_emission_cost
        || report.signals.impulse_trials.len() != expected_signal_multipliers.len()
        || report
            .signals
            .impulse_trials
            .iter()
            .zip(expected_signal_multipliers)
            .any(|(trial, multiplier)| {
                trial.strength_multiplier != multiplier
                    || trial.requested_energy
                        != report
                            .rules
                            .signal_emission_cost
                            .checked_mul(multiplier)
                            .filter(|amount| *amount > 0)
                    || trial.field_energy_after_completion > trial.field_energy_after_commit
                    || trial.decayed_during_action
                        != trial
                            .field_energy_after_commit
                            .saturating_sub(trial.field_energy_after_completion)
            })
        || report.signals.mass_shedding_trials.len() != expected_shedding_multipliers.len()
        || report
            .signals
            .mass_shedding_trials
            .iter()
            .zip(expected_shedding_multipliers)
            .any(|(trial, multiplier)| {
                let expected_energy = if multiplier == 0 {
                    Some(0)
                } else {
                    report
                        .rules
                        .signal_emission_cost
                        .checked_mul(multiplier)
                        .filter(|amount| *amount > 0)
                };
                trial.strength_multiplier != multiplier
                    || trial.requested_energy != expected_energy
                    || trial.preparation_is_signal != (multiplier > 0)
                    || trial.preparation_accepted != trial.preparation_completion_quanta.is_some()
                    || trial.energy_after_preparation > trial.energy_after_commit
                    || trial.metabolism_during_preparation
                        != trial
                            .energy_after_commit
                            .zip(trial.energy_after_preparation)
                            .map(|(after_commit, after_preparation)| {
                                after_commit.saturating_sub(after_preparation)
                            })
                    || trial.stationary_remaining_lifetime_quanta
                        != trial
                            .stationary_death_time_quanta
                            .zip(trial.preparation_completion_quanta)
                            .map(|(death, preparation)| death.saturating_sub(preparation))
                    || trial.completed_moves == 0
                        && (trial.completed_distance_q10 != 0
                            || trial.energy_after_last_completed_move.is_some())
            })
        || report.signals.neighbor_observation_slots > report.rules.neighborhood.slots.len()
        || report.signals.minimum_distinct_observable_neighbors
            > report.signals.maximum_distinct_observable_neighbors
        || report
            .signals
            .terrain_disruption
            .as_ref()
            .is_some_and(|probe| {
                probe.signal_after_terrain_action > probe.signal_before_terrain_action
                    || probe.erased_by_terrain_action
                        > probe
                            .signal_before_terrain_action
                            .saturating_sub(probe.decayed_during_terrain_action)
            })
    {
        return Err("ecological characterization signal accounting is invalid".into());
    }
    if report.sustained_siege.trials.len() != report.sustained_siege.maximum_local_attackers
        || report.sustained_siege.time_limit_quanta != report.scenario.victory.sim_time_limit_quanta
        || report
            .sustained_siege
            .trials
            .iter()
            .enumerate()
            .any(|(index, trial)| trial.attackers != index + 1)
    {
        return Err("ecological characterization sustained-siege matrix is incomplete".into());
    }
    let expected_siege_minimum = report
        .sustained_siege
        .trials
        .iter()
        .find(|trial| trial.victim_killed)
        .map(|trial| trial.attackers);
    if report.sustained_siege.minimum_attackers_to_kill != expected_siege_minimum {
        return Err("ecological characterization sustained-siege minimum is inconsistent".into());
    }
    Ok(())
}

fn ceil_ratio(value: u64, numerator: u64, denominator: u64) -> Option<u64> {
    let scaled = u128::from(value).checked_mul(u128::from(numerator))?;
    let rounded =
        scaled.checked_add(u128::from(denominator).checked_sub(1)?)? / u128::from(denominator);
    u64::try_from(rounded).ok()
}

fn effort_cost(rules: &ReferenceRuleset, base: u64, effort: CharacterizedEffort) -> Option<u64> {
    let profile = rules.effort_profiles[effort.index()];
    ceil_ratio(
        base,
        u64::from(profile.cost_numerator),
        u64::from(profile.cost_denominator),
    )
}

fn signal_decay_time(energy: u64, remaining: u64, rules: &ReferenceRuleset) -> Option<u64> {
    if energy <= remaining || rules.signal_decay_rate_numerator == 0 {
        return None;
    }
    ceil_ratio(
        energy - remaining,
        rules.signal_decay_rate_denominator,
        rules.signal_decay_rate_numerator,
    )
}

fn preparation_request(signal_energy: Option<u64>, is_signal: bool) -> ActionRequest {
    if is_signal {
        ActionRequest::Signal {
            amounts: [signal_energy.unwrap_or(0), 0, 0, 0],
        }
    } else {
        ActionRequest::Wait
    }
}

fn stationary_death_after_preparation(
    env: &EnvConfig,
    signal_energy: Option<u64>,
    is_signal: bool,
) -> Result<Option<u64>, String> {
    if is_signal && signal_energy.is_none() {
        return Ok(None);
    }
    let mut simulation = ReferenceSimulation::new(1, 1, env.rules.clone())
        .map_err(|error| format!("failed to create signal-lifetime micro-world: {error}"))?;
    let actor = simulation
        .add_cell(
            TileIndex(0),
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add signal-lifetime micro-cell: {error}"))?;
    let receipt = simulation
        .commit_action(actor, preparation_request(signal_energy, is_signal))
        .map_err(|error| format!("failed to commit signal-lifetime preparation: {error}"))?;
    if !receipt.accepted {
        return Ok(None);
    }
    resolve_actor_action(&mut simulation, actor, "signal-lifetime preparation")?;
    if env.rules.metabolism_rate_numerator == 0 {
        return Ok(None);
    }
    while simulation.cell(actor).is_some() {
        simulation
            .resolve_next_batch()
            .map_err(|error| format!("failed to resolve signal-lifetime expiration: {error}"))?;
    }
    Ok(Some(simulation.now().0))
}

fn characterize_signal_mass_shedding(
    env: &EnvConfig,
    max_micro_actions: usize,
) -> Result<Vec<SignalMassSheddingTrial>, String> {
    const MULTIPLIERS: [u64; 6] = [0, 1, 2, 4, 8, 16];
    let mut trials = Vec::with_capacity(MULTIPLIERS.len());
    for strength_multiplier in MULTIPLIERS {
        let preparation_is_signal = strength_multiplier > 0;
        let requested_energy = if preparation_is_signal {
            env.rules
                .signal_emission_cost
                .checked_mul(strength_multiplier)
                .filter(|amount| *amount > 0)
        } else {
            Some(0)
        };
        let mut trial = SignalMassSheddingTrial {
            strength_multiplier,
            requested_energy,
            preparation_is_signal,
            preparation_accepted: false,
            energy_after_commit: None,
            energy_after_preparation: None,
            total_mass_after_preparation: None,
            preparation_completion_quanta: None,
            metabolism_during_preparation: None,
            stationary_death_time_quanta: None,
            stationary_remaining_lifetime_quanta: None,
            first_move_effort_spent: None,
            first_move_duration_quanta: None,
            total_move_effort_spent: 0,
            completed_moves: 0,
            completed_distance_q10: 0,
            movement_elapsed_quanta: 0,
            energy_after_last_completed_move: None,
            travel_death_time_quanta: None,
            censored: false,
        };
        if preparation_is_signal && requested_energy.is_none() {
            trials.push(trial);
            continue;
        }
        let mut simulation =
            ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone()).map_err(
                |error| format!("failed to create signal-shedding micro-world: {error}"),
            )?;
        let movement_pair = bidirectional_move_pair(&simulation);
        let origin = movement_pair.map_or(TileIndex(0), |pair| pair.0);
        let actor = simulation
            .add_cell(
                origin,
                u64::from(env.min_energy),
                u64::from(env.initial_energy),
                0,
            )
            .map_err(|error| format!("failed to add signal-shedding micro-cell: {error}"))?;
        let receipt = simulation
            .commit_action(
                actor,
                preparation_request(requested_energy, preparation_is_signal),
            )
            .map_err(|error| format!("failed to commit signal-shedding preparation: {error}"))?;
        trial.preparation_accepted = receipt.accepted;
        if !receipt.accepted {
            trials.push(trial);
            continue;
        }
        let energy_after_commit = simulation.cell(actor).map(|state| state.assimilated_energy);
        trial.energy_after_commit = energy_after_commit;
        trial.preparation_completion_quanta = Some(receipt.completes_at.0);
        resolve_actor_action(&mut simulation, actor, "signal-shedding preparation")?;
        let preparation_time = simulation.now().0;
        if let Some(state) = simulation.cell(actor) {
            trial.energy_after_preparation = Some(state.assimilated_energy);
            trial.total_mass_after_preparation = Some(
                u64::try_from(state.total_mass())
                    .map_err(|_| "signal-shedding cell mass exceeded u64")?,
            );
            trial.metabolism_during_preparation =
                energy_after_commit.map(|energy| energy.saturating_sub(state.assimilated_energy));
        }
        trial.stationary_death_time_quanta =
            stationary_death_after_preparation(env, requested_energy, preparation_is_signal)?;
        trial.stationary_remaining_lifetime_quanta = trial
            .stationary_death_time_quanta
            .map(|death| death.saturating_sub(preparation_time));

        let Some((first, outward, second, return_slot, outward_cost, return_cost)) = movement_pair
        else {
            trials.push(trial);
            continue;
        };
        let mut movement_frontier = preparation_time;
        for action_index in 0..max_micro_actions {
            let Some(state) = simulation.cell(actor) else {
                break;
            };
            let (slot, cost) = if state.position == first {
                (outward, outward_cost)
            } else if state.position == second {
                (return_slot, return_cost)
            } else {
                return Err("signal-shedding cell left its two-tile corridor".into());
            };
            let action_started = simulation.now().0;
            let old_position = state.position;
            let move_receipt = simulation
                .commit_action(
                    actor,
                    ActionRequest::Move {
                        target: slot,
                        effort: EffortTier::Standard,
                    },
                )
                .map_err(|error| format!("failed to commit signal-shedding move: {error}"))?;
            if !move_receipt.accepted {
                break;
            }
            if trial.first_move_duration_quanta.is_none() {
                trial.first_move_effort_spent = Some(move_receipt.effort_spent);
                trial.first_move_duration_quanta = Some(
                    move_receipt
                        .completes_at
                        .0
                        .checked_sub(action_started)
                        .ok_or("signal-shedding move completed before it started")?,
                );
            }
            trial.total_move_effort_spent = trial
                .total_move_effort_spent
                .checked_add(move_receipt.effort_spent)
                .ok_or("signal-shedding movement effort overflowed")?;
            resolve_actor_action(&mut simulation, actor, "signal-shedding move")?;
            movement_frontier = simulation.now().0;
            let Some(state) = simulation.cell(actor) else {
                break;
            };
            if state.position == old_position {
                break;
            }
            trial.completed_moves += 1;
            trial.completed_distance_q10 = trial
                .completed_distance_q10
                .checked_add(cost)
                .ok_or("signal-shedding travel distance overflowed")?;
            trial.energy_after_last_completed_move = Some(state.assimilated_energy);
            if action_index + 1 == max_micro_actions {
                trial.censored = true;
            }
        }
        trial.movement_elapsed_quanta = movement_frontier.saturating_sub(preparation_time);
        if !trial.censored && env.rules.metabolism_rate_numerator > 0 {
            while simulation.cell(actor).is_some() {
                simulation.resolve_next_batch().map_err(|error| {
                    format!("failed to resolve signal-shedding travel expiration: {error}")
                })?;
            }
        }
        trial.travel_death_time_quanta =
            (!trial.censored && simulation.cell(actor).is_none()).then_some(simulation.now().0);
        trials.push(trial);
    }
    Ok(trials)
}

fn characterize_signals(
    env: &EnvConfig,
    max_micro_actions: usize,
) -> Result<SignalCharacterization, String> {
    const MULTIPLIERS: [u64; 5] = [1, 2, 4, 8, 16];
    let compiled = ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
        .map_err(|error| format!("failed to create signal-observation micro-world: {error}"))?;
    let neighborhood = compiled.neighborhood();
    let neighbor_observation_slots = (0..neighborhood.slot_count())
        .filter(|slot| {
            neighborhood
                .spec()
                .observations
                .signal
                .contains(LocalSlot(*slot as u8))
        })
        .count();
    let mut minimum_distinct_observable_neighbors = usize::MAX;
    let mut maximum_distinct_observable_neighbors = 0_usize;
    let mut directed_observation_edges = 0_u64;
    for tile_index in 0..compiled.tiles().len() {
        let tile = TileIndex(tile_index);
        let mut targets = HashSet::with_capacity(neighbor_observation_slots);
        for slot in 0..neighborhood.slot_count() {
            let slot = LocalSlot(slot as u8);
            if neighborhood.spec().observations.signal.contains(slot) {
                if let Some(target) = neighborhood
                    .target(tile, slot)
                    .filter(|target| *target != tile)
                {
                    targets.insert(target);
                }
            }
        }
        minimum_distinct_observable_neighbors =
            minimum_distinct_observable_neighbors.min(targets.len());
        maximum_distinct_observable_neighbors =
            maximum_distinct_observable_neighbors.max(targets.len());
        directed_observation_edges = directed_observation_edges
            .saturating_add(u64::try_from(targets.len()).unwrap_or(u64::MAX));
    }
    if minimum_distinct_observable_neighbors == usize::MAX {
        minimum_distinct_observable_neighbors = 0;
    }

    let mut impulse_trials = Vec::with_capacity(MULTIPLIERS.len());
    for strength_multiplier in MULTIPLIERS {
        let requested_energy = env
            .rules
            .signal_emission_cost
            .checked_mul(strength_multiplier)
            .filter(|amount| *amount > 0);
        let time_to_half_energy_quanta =
            requested_energy.and_then(|energy| signal_decay_time(energy, energy / 2, &env.rules));
        let time_to_extinction_quanta =
            requested_energy.and_then(|energy| signal_decay_time(energy, 0, &env.rules));
        let mut trial = SignalImpulseTrial {
            strength_multiplier,
            requested_energy,
            accepted: false,
            actor_energy_after_commit: None,
            action_completion_quanta: None,
            field_energy_after_commit: 0,
            field_energy_after_completion: 0,
            decayed_during_action: 0,
            time_to_half_energy_quanta,
            time_to_extinction_quanta,
        };
        let Some(requested_energy) = requested_energy else {
            impulse_trials.push(trial);
            continue;
        };
        let mut simulation = ReferenceSimulation::new(1, 1, env.rules.clone())
            .map_err(|error| format!("failed to create signal impulse micro-world: {error}"))?;
        let actor = simulation
            .add_cell(
                TileIndex(0),
                u64::from(env.min_energy),
                u64::from(env.initial_energy),
                0,
            )
            .map_err(|error| format!("failed to add signal impulse micro-cell: {error}"))?;
        let Ok(receipt) = simulation.commit_action(
            actor,
            ActionRequest::Signal {
                amounts: [requested_energy, 0, 0, 0],
            },
        ) else {
            impulse_trials.push(trial);
            continue;
        };
        trial.accepted = receipt.accepted;
        trial.actor_energy_after_commit =
            simulation.cell(actor).map(|cell| cell.assimilated_energy);
        trial.action_completion_quanta = receipt.accepted.then_some(receipt.completes_at.0);
        trial.field_energy_after_commit = simulation.tiles()[0].signal_energy[0];
        if receipt.accepted {
            simulation
                .resolve_next_batch()
                .map_err(|error| format!("failed to resolve signal impulse: {error}"))?;
            trial.field_energy_after_completion = simulation.tiles()[0].signal_energy[0];
            trial.decayed_during_action =
                u64::try_from(simulation.last_resolution_metrics().signal_energy_decayed[0])
                    .unwrap_or(u64::MAX);
        }
        impulse_trials.push(trial);
    }

    let mut terrain_disruption = None;
    for strength_multiplier in MULTIPLIERS.into_iter().rev() {
        let Some(emitted_energy) = env
            .rules
            .signal_emission_cost
            .checked_mul(strength_multiplier)
            .filter(|amount| *amount > 0)
        else {
            continue;
        };
        let mut simulation = ReferenceSimulation::new(1, 1, env.rules.clone())
            .map_err(|error| format!("failed to create terrain-signal micro-world: {error}"))?;
        let actor = simulation
            .add_cell(
                TileIndex(0),
                u64::from(env.min_energy),
                u64::from(env.initial_energy),
                0,
            )
            .map_err(|error| format!("failed to add terrain-signal micro-cell: {error}"))?;
        let Ok(signal_receipt) = simulation.commit_action(
            actor,
            ActionRequest::Signal {
                amounts: [emitted_energy, 0, 0, 0],
            },
        ) else {
            continue;
        };
        if !signal_receipt.accepted {
            continue;
        }
        simulation
            .resolve_next_batch()
            .map_err(|error| format!("failed to resolve terrain-signal impulse: {error}"))?;
        let signal_before_terrain_action = simulation.tiles()[0].signal_energy[0];
        let Ok(terrain_receipt) = simulation.commit_action(actor, ActionRequest::Excavate) else {
            continue;
        };
        if !terrain_receipt.accepted {
            continue;
        }
        let report = simulation
            .resolve_next_batch()
            .map_err(|error| format!("failed to resolve terrain-signal excavation: {error}"))?;
        let terrain_action_succeeded = report.outcomes.iter().any(|outcome| {
            outcome.actor == actor
                && outcome.status == OutcomeStatus::Success
                && outcome.terrain_change.is_some()
        });
        let erased_by_terrain_action = report
            .delta
            .tiles
            .iter()
            .find(|delta| delta.tile == TileIndex(0))
            .map_or(0, |delta| delta.before.signal_energy[0]);
        terrain_disruption = Some(SignalTerrainDisruptionCharacterization {
            strength_multiplier,
            emitted_energy,
            signal_before_terrain_action,
            decayed_during_terrain_action: u64::try_from(
                simulation.last_resolution_metrics().signal_energy_decayed[0],
            )
            .unwrap_or(u64::MAX),
            erased_by_terrain_action,
            signal_after_terrain_action: simulation.tiles()[0].signal_energy[0],
            diffuse_energy_after_terrain_action: simulation.tiles()[0].diffuse_energy,
            terrain_action_succeeded,
        });
        break;
    }

    let mass_shedding_trials = characterize_signal_mass_shedding(env, max_micro_actions)?;
    Ok(SignalCharacterization {
        emission_quantum: env.rules.signal_emission_cost,
        neighbor_observation_slots,
        minimum_distinct_observable_neighbors,
        maximum_distinct_observable_neighbors,
        directed_observation_edges,
        impulse_trials,
        mass_shedding_trials,
        terrain_disruption,
    })
}

fn stationary_lifetime(env: &EnvConfig) -> Result<Option<u64>, String> {
    if env.rules.metabolism_rate_numerator == 0 {
        return Ok(None);
    }
    let mut simulation = ReferenceSimulation::new(1, 1, env.rules.clone())
        .map_err(|error| format!("failed to create stationary micro-world: {error}"))?;
    let cell = simulation
        .add_cell(
            TileIndex(0),
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add stationary micro-cell: {error}"))?;
    simulation
        .resolve_next_batch()
        .map_err(|error| format!("failed to resolve stationary lifetime: {error}"))?;
    if simulation.cell(cell).is_some() {
        return Err("stationary metabolic deadline did not expire the micro-cell".into());
    }
    Ok(Some(simulation.now().0))
}

fn resolve_actor_action(
    simulation: &mut ReferenceSimulation,
    actor: blob_engine::resolution::CellKey,
    context: &str,
) -> Result<(), String> {
    while simulation
        .cell(actor)
        .is_some_and(|state| state.pending_action.is_some())
    {
        simulation
            .resolve_next_batch()
            .map_err(|error| format!("failed to resolve {context}: {error}"))?;
    }
    Ok(())
}

fn characterize_feeding_cycle(env: &EnvConfig) -> Result<FeedingCycleCharacterization, String> {
    let mut simulation = ReferenceSimulation::new(1, 1, env.rules.clone())
        .map_err(|error| format!("failed to create feeding micro-world: {error}"))?;
    let requested = env.rules.bite_capacity.min(env.rules.gut_capacity);
    {
        let tile = simulation
            .tile_state_mut(TileIndex(0))
            .ok_or("feeding micro-world tile is absent")?;
        tile.plant_energy = requested;
        tile.plant_capacity = requested;
        tile.plant_growth_rate = 0;
    }
    let cell = simulation
        .add_cell(
            TileIndex(0),
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add feeding micro-cell: {error}"))?;
    let energy_before_consume = simulation
        .cell(cell)
        .ok_or("feeding micro-cell disappeared before commitment")?
        .assimilated_energy;
    let receipt = simulation
        .commit_action(cell, ActionRequest::Consume { amount: requested })
        .map_err(|error| format!("failed to commit feeding action: {error}"))?;
    let energy_after_commit = simulation
        .cell(cell)
        .map_or(0, |state| state.assimilated_energy);
    if !receipt.accepted {
        return Ok(FeedingCycleCharacterization {
            requested_energy: requested,
            consumed_energy: 0,
            consume_effort_spent: 0,
            consume_completion_quanta: receipt.completes_at.0,
            digestion_completion_quanta: None,
            energy_before_consume,
            energy_after_commit,
            energy_after_digestion: simulation.cell(cell).map(|state| state.assimilated_energy),
            gut_after_consume: 0,
            gut_after_digestion: simulation.cell(cell).map(|state| state.gut_energy),
            survived_full_digestion: false,
        });
    }
    resolve_actor_action(&mut simulation, cell, "feeding action")?;
    let gut_after_consume = simulation.cell(cell).map_or(0, |state| state.gut_energy);
    let consumed_energy = gut_after_consume;
    let digestion_duration = if gut_after_consume == 0 || env.rules.digestion_rate_numerator == 0 {
        None
    } else {
        ceil_ratio(
            gut_after_consume,
            env.rules.digestion_rate_denominator,
            env.rules.digestion_rate_numerator,
        )
    };
    let digestion_completion_quanta =
        digestion_duration.and_then(|duration| simulation.now().0.checked_add(duration));
    if let Some(completes_at) = digestion_completion_quanta {
        while simulation.now().0 < completes_at && simulation.cell(cell).is_some() {
            if simulation
                .next_event_time()
                .map_err(|error| format!("failed to inspect feeding events: {error}"))?
                .is_some_and(|event| event.0 <= completes_at)
            {
                simulation
                    .resolve_next_batch()
                    .map_err(|error| format!("failed to resolve feeding passive event: {error}"))?;
            } else {
                simulation
                    .advance_clock_to(SimTime(completes_at))
                    .map_err(|error| {
                        format!("failed to advance full feeding digestion: {error}")
                    })?;
            }
        }
    }
    let final_state = simulation.cell(cell);
    Ok(FeedingCycleCharacterization {
        requested_energy: requested,
        consumed_energy,
        consume_effort_spent: receipt.effort_spent,
        consume_completion_quanta: receipt.completes_at.0,
        digestion_completion_quanta,
        energy_before_consume,
        energy_after_commit,
        energy_after_digestion: final_state.map(|state| state.assimilated_energy),
        gut_after_consume,
        gut_after_digestion: final_state.map(|state| state.gut_energy),
        survived_full_digestion: final_state.is_some_and(|state| state.gut_energy == 0),
    })
}

fn split_source(simulation: &ReferenceSimulation) -> Option<(TileIndex, LocalSlot, TileIndex)> {
    let neighborhood = simulation.neighborhood();
    for origin_index in 0..neighborhood.tile_count() {
        let origin = TileIndex(origin_index);
        for slot_index in 0..neighborhood.slot_count() {
            let slot = LocalSlot(u8::try_from(slot_index).ok()?);
            if neighborhood.action_allows(TargetingAction::Split, slot) {
                if let Some(target) = neighborhood
                    .target(origin, slot)
                    .filter(|target| *target != origin)
                {
                    return Some((origin, slot, target));
                }
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct SplitTrial {
    succeeded: bool,
    effort_spent: u64,
    completion_quanta: u64,
    parent_energy: Option<u64>,
    child_energy: Option<u64>,
}

fn split_trial(
    env: &EnvConfig,
    parent_energy: u64,
    child_allocation: u64,
) -> Result<Option<SplitTrial>, String> {
    let mut simulation =
        ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
            .map_err(|error| format!("failed to create reproduction micro-world: {error}"))?;
    let Some((origin, slot, target)) = split_source(&simulation) else {
        return Ok(None);
    };
    let parent = simulation
        .add_cell(origin, u64::from(env.min_energy), parent_energy, 0)
        .map_err(|error| format!("failed to add reproduction parent: {error}"))?;
    let receipt = simulation
        .commit_action(
            parent,
            ActionRequest::Split {
                target: slot,
                child_allocation,
                marker: 0,
                private_memory: Vec::new(),
            },
        )
        .map_err(|error| format!("failed to commit reproduction action: {error}"))?;
    if !receipt.accepted {
        return Ok(Some(SplitTrial {
            succeeded: false,
            effort_spent: receipt.effort_spent,
            completion_quanta: receipt.completes_at.0,
            parent_energy: simulation
                .cell(parent)
                .map(|state| state.assimilated_energy),
            child_energy: None,
        }));
    }
    resolve_actor_action(&mut simulation, parent, "reproduction action")?;
    let child_energy = simulation.cells().iter().find_map(|(key, state)| {
        (*key != parent && state.position == target).then_some(state.assimilated_energy)
    });
    Ok(Some(SplitTrial {
        succeeded: simulation.cell(parent).is_some() && child_energy.is_some(),
        effort_spent: receipt.effort_spent,
        completion_quanta: receipt.completes_at.0,
        parent_energy: simulation
            .cell(parent)
            .map(|state| state.assimilated_energy),
        child_energy,
    }))
}

fn characterize_reproduction(
    env: &EnvConfig,
    max_iterations: usize,
) -> Result<ReproductionBreakEvenCharacterization, String> {
    let child_allocation = env
        .rules
        .child_core_mass
        .checked_add(env.rules.minimum_survival_energy)
        .ok_or("minimum child allocation overflowed")?;
    let split_effort = effort_cost(
        &env.rules,
        env.rules.split_effort_base,
        CharacterizedEffort::Standard,
    )
    .ok_or("split effort cost overflowed")?;
    let minimum_commit_energy = child_allocation
        .checked_add(split_effort)
        .and_then(|value| value.checked_add(env.rules.minimum_survival_energy))
        .ok_or("minimum split commitment energy overflowed")?;
    let probe = ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
        .map_err(|error| format!("failed to compile reproduction neighborhood: {error}"))?;
    if split_source(&probe).is_none() {
        return Ok(ReproductionBreakEvenCharacterization {
            minimum_child_allocation: child_allocation,
            minimum_commit_energy,
            minimum_successful_parent_energy: None,
            split_effort_spent: None,
            split_completion_quanta: None,
            parent_energy_after_split: None,
            child_energy_after_split: None,
        });
    }

    let mut candidate = minimum_commit_energy;
    let mut successful = None;
    for _ in 0..max_iterations.min(4_096) {
        let trial = split_trial(env, candidate, child_allocation)?
            .ok_or("reproduction target disappeared after neighborhood probe")?;
        if trial.succeeded {
            successful = Some((candidate, trial));
            break;
        }
        let duration = trial.completion_quanta;
        let metabolic_spend = u128::from(duration)
            .checked_mul(u128::from(env.rules.metabolism_rate_numerator))
            .map(|value| value / u128::from(env.rules.metabolism_rate_denominator))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or("reproduction windup metabolism overflowed")?;
        let required_reserve = env.rules.minimum_survival_energy.max(
            metabolic_spend
                .checked_add(1)
                .ok_or("reproduction survival reserve overflowed")?,
        );
        let required = child_allocation
            .checked_add(trial.effort_spent)
            .and_then(|value| value.checked_add(required_reserve))
            .ok_or("reproduction break-even energy overflowed")?;
        candidate = required.max(candidate.saturating_add(1));
    }
    let Some((energy, trial)) = successful else {
        return Ok(ReproductionBreakEvenCharacterization {
            minimum_child_allocation: child_allocation,
            minimum_commit_energy,
            minimum_successful_parent_energy: None,
            split_effort_spent: None,
            split_completion_quanta: None,
            parent_energy_after_split: None,
            child_energy_after_split: None,
        });
    };
    Ok(ReproductionBreakEvenCharacterization {
        minimum_child_allocation: child_allocation,
        minimum_commit_energy,
        minimum_successful_parent_energy: Some(energy),
        split_effort_spent: Some(trial.effort_spent),
        split_completion_quanta: Some(trial.completion_quanta),
        parent_energy_after_split: trial.parent_energy,
        child_energy_after_split: trial.child_energy,
    })
}

fn characterize_terrain_cycle(env: &EnvConfig) -> Result<TerrainCycleCharacterization, String> {
    let mut simulation = ReferenceSimulation::new(1, 1, env.rules.clone())
        .map_err(|error| format!("failed to create terrain-cycle micro-world: {error}"))?;
    let initial_elevation = simulation
        .tile_state(TileIndex(0))
        .ok_or("terrain-cycle tile is absent")?
        .elevation;
    let cell = simulation
        .add_cell(
            TileIndex(0),
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add terrain-cycle cell: {error}"))?;
    let excavate = simulation
        .commit_action(cell, ActionRequest::Excavate)
        .map_err(|error| format!("failed to commit excavation: {error}"))?;
    let excavation_succeeded = if excavate.accepted {
        resolve_actor_action(&mut simulation, cell, "excavation")?;
        simulation.cell(cell).is_some_and(|state| {
            state.carried_material_mass >= env.rules.terrain_mass_per_elevation
        })
    } else {
        false
    };
    let mut deposit_receipt = None;
    let deposition_succeeded = if excavation_succeeded && simulation.cell(cell).is_some() {
        let deposit = simulation
            .commit_action(cell, ActionRequest::DepositTerrain)
            .map_err(|error| format!("failed to commit terrain deposition: {error}"))?;
        let accepted = deposit.accepted;
        deposit_receipt = Some(deposit);
        if accepted {
            resolve_actor_action(&mut simulation, cell, "terrain deposition")?;
        }
        accepted
            && simulation
                .tile_state(TileIndex(0))
                .is_some_and(|tile| tile.elevation == initial_elevation)
    } else {
        false
    };
    let final_state = simulation.cell(cell);
    Ok(TerrainCycleCharacterization {
        initial_energy: u64::from(env.initial_energy),
        material_mass_per_elevation: env.rules.terrain_mass_per_elevation,
        excavation_succeeded,
        deposition_succeeded,
        excavation_effort_spent: excavate.accepted.then_some(excavate.effort_spent),
        deposition_effort_spent: deposit_receipt
            .as_ref()
            .filter(|receipt| receipt.accepted)
            .map(|receipt| receipt.effort_spent),
        total_completion_quanta: simulation.now().0,
        final_energy: final_state.map(|state| state.assimilated_energy),
        final_carried_material_mass: final_state.map(|state| state.carried_material_mass),
        elevation_restored: simulation
            .tile_state(TileIndex(0))
            .is_some_and(|tile| tile.elevation == initial_elevation),
    })
}

fn bidirectional_move_pair(
    simulation: &ReferenceSimulation,
) -> Option<(TileIndex, LocalSlot, TileIndex, LocalSlot, u64, u64)> {
    let neighborhood = simulation.neighborhood();
    let mut best = None;
    for origin_index in 0..neighborhood.tile_count() {
        let origin = TileIndex(origin_index);
        for slot_index in 0..neighborhood.slot_count() {
            let slot = LocalSlot(u8::try_from(slot_index).ok()?);
            if !neighborhood.action_allows(TargetingAction::Move, slot) {
                continue;
            }
            let Some(target) = neighborhood.target(origin, slot) else {
                continue;
            };
            if target == origin {
                continue;
            }
            for reverse_index in 0..neighborhood.slot_count() {
                let reverse = LocalSlot(u8::try_from(reverse_index).ok()?);
                if neighborhood.action_allows(TargetingAction::Move, reverse)
                    && neighborhood.target(target, reverse) == Some(origin)
                {
                    let forward_cost = u64::from(neighborhood.offset(slot)?.distance_cost_q10);
                    let reverse_cost = u64::from(neighborhood.offset(reverse)?.distance_cost_q10);
                    let candidate = (origin, slot, target, reverse, forward_cost, reverse_cost);
                    if best.as_ref().is_none_or(
                        |current: &(TileIndex, LocalSlot, TileIndex, LocalSlot, u64, u64)| {
                            forward_cost + reverse_cost < current.4 + current.5
                        },
                    ) {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best
}

fn travel_endurance(
    env: &EnvConfig,
    effort: CharacterizedEffort,
    max_micro_actions: usize,
) -> Result<Option<TravelEndurance>, String> {
    let mut simulation =
        ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
            .map_err(|error| format!("failed to create travel micro-world: {error}"))?;
    let Some((first, outward, second, return_slot, outward_cost, return_cost)) =
        bidirectional_move_pair(&simulation)
    else {
        return Ok(None);
    };
    let cell = simulation
        .add_cell(
            first,
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add travel micro-cell: {error}"))?;
    let mut moves = 0_u64;
    let mut distance_q10 = 0_u64;
    let mut travel_time = 0_u64;
    let mut censored = false;
    for action_index in 0..max_micro_actions {
        let Some(state) = simulation.cell(cell) else {
            break;
        };
        let (slot, cost) = if state.position == first {
            (outward, outward_cost)
        } else if state.position == second {
            (return_slot, return_cost)
        } else {
            return Err("travel micro-cell left its two-tile corridor".into());
        };
        let origin = state.position;
        let receipt = simulation
            .commit_action(
                cell,
                ActionRequest::Move {
                    target: slot,
                    effort: effort.resolver(),
                },
            )
            .map_err(|error| format!("failed to commit travel action: {error}"))?;
        if !receipt.accepted {
            break;
        }
        while simulation
            .cell(cell)
            .is_some_and(|state| state.pending_action.is_some())
        {
            simulation
                .resolve_next_batch()
                .map_err(|error| format!("failed to resolve travel action: {error}"))?;
        }
        let Some(state) = simulation.cell(cell) else {
            break;
        };
        if state.position == origin {
            break;
        }
        moves += 1;
        distance_q10 = distance_q10
            .checked_add(cost)
            .ok_or("travel distance overflowed")?;
        travel_time = simulation.now().0;
        if action_index + 1 == max_micro_actions {
            censored = true;
        }
    }
    if !censored && env.rules.metabolism_rate_numerator > 0 {
        while simulation.cell(cell).is_some() {
            simulation
                .resolve_next_batch()
                .map_err(|error| format!("failed to resolve post-travel expiration: {error}"))?;
        }
    }
    let final_energy = simulation
        .cell(cell)
        .map_or(0, |state| state.assimilated_energy);
    Ok(Some(TravelEndurance {
        effort,
        completed_moves: moves,
        completed_distance_q10: distance_q10,
        travel_time_quanta: travel_time,
        death_time_quanta: (!censored && simulation.cell(cell).is_none())
            .then_some(simulation.now().0),
        final_assimilated_energy: final_energy,
        censored,
    }))
}

fn shortest_distances(
    simulation: &ReferenceSimulation,
    sources: &[TileIndex],
    weighted: bool,
) -> Vec<Option<u64>> {
    let neighborhood = simulation.neighborhood();
    let mut incoming = vec![Vec::<(usize, u64)>::new(); neighborhood.tile_count()];
    for origin in 0..neighborhood.tile_count() {
        for slot_index in 0..neighborhood.slot_count() {
            let Ok(slot_u8) = u8::try_from(slot_index) else {
                continue;
            };
            let slot = LocalSlot(slot_u8);
            if !neighborhood.action_allows(TargetingAction::Move, slot) {
                continue;
            }
            let Some(target) = neighborhood.target(TileIndex(origin), slot) else {
                continue;
            };
            if target.0 == origin {
                continue;
            }
            let cost = if weighted {
                u64::from(
                    neighborhood
                        .offset(slot)
                        .expect("compiled slot has an offset")
                        .distance_cost_q10,
                )
            } else {
                1
            };
            incoming[target.0].push((origin, cost));
        }
    }
    let mut distances = vec![u64::MAX; neighborhood.tile_count()];
    let mut frontier = BinaryHeap::new();
    for source in sources {
        distances[source.0] = 0;
        frontier.push((Reverse(0_u64), source.0));
    }
    while let Some((Reverse(distance), tile)) = frontier.pop() {
        if distance != distances[tile] {
            continue;
        }
        for (predecessor, cost) in &incoming[tile] {
            let Some(candidate) = distance.checked_add(*cost) else {
                continue;
            };
            if candidate < distances[*predecessor] {
                distances[*predecessor] = candidate;
                frontier.push((Reverse(candidate), *predecessor));
            }
        }
    }
    distances
        .into_iter()
        .map(|distance| (distance != u64::MAX).then_some(distance))
        .collect()
}

fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    let index = (sorted.len().saturating_sub(1) * numerator).div_ceil(denominator);
    sorted[index]
}

fn distance_distribution(values: &[Option<u64>], scale: f64) -> DistanceDistribution {
    let mut reachable = values.iter().flatten().copied().collect::<Vec<_>>();
    reachable.sort_unstable();
    let mean = (!reachable.is_empty()).then(|| {
        reachable.iter().map(|value| *value as f64).sum::<f64>() / reachable.len() as f64 / scale
    });
    DistanceDistribution {
        samples: values.len(),
        reachable_samples: reachable.len(),
        unreachable_samples: values.len() - reachable.len(),
        mean,
        p50: (!reachable.is_empty()).then(|| percentile(&reachable, 1, 2) as f64 / scale),
        p90: (!reachable.is_empty()).then(|| percentile(&reachable, 9, 10) as f64 / scale),
        maximum: reachable.last().map(|value| *value as f64 / scale),
    }
}

fn characterize_food_distances(
    env: &EnvConfig,
    seeds: &[u64],
) -> Result<
    (
        FoodDistanceCharacterization,
        FoodDistanceCharacterization,
        String,
    ),
    String,
> {
    let mut plant_steps = Vec::new();
    let mut plant_weighted = Vec::new();
    let mut food_steps = Vec::new();
    let mut food_weighted = Vec::new();
    let mut total_plant_tiles = 0usize;
    let mut total_food_tiles = 0usize;
    let mut compiled_ruleset_hash = None;
    for seed in seeds {
        let env_instance = BlobEnv::new(env.clone(), RewardConfig::default(), *seed);
        let compiled = env_instance.compiled_ruleset_hash();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &compiled)
        {
            return Err("identical characterization rules compiled inconsistently".into());
        }
        compiled_ruleset_hash = Some(compiled);
        let checkpoint = env_instance.checkpoint()?;
        let canonical = ReferenceCheckpoint::from_bytes(&checkpoint.canonical_checkpoint)
            .map_err(|error| format!("failed to decode characterization checkpoint: {error}"))?;
        let host_teams = checkpoint
            .host_cells
            .iter()
            .map(|cell| (cell.id.0, cell.team_id))
            .collect::<HashMap<_, _>>();
        let starts = canonical
            .state()
            .cells
            .iter()
            .filter_map(|(key, cell)| {
                usize::try_from(key.0).ok().and_then(|id| {
                    (host_teams.get(&id) == Some(&TeamId(0))).then_some(cell.position)
                })
            })
            .collect::<Vec<_>>();
        let plants = canonical
            .state()
            .tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0).then_some(TileIndex(index))
            })
            .collect::<Vec<_>>();
        let major_food = canonical
            .state()
            .tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0 || tile.loose_energy > 0)
                    .then_some(TileIndex(index))
            })
            .collect::<Vec<_>>();
        total_plant_tiles += plants.len();
        total_food_tiles += major_food.len();
        let simulation = canonical
            .clone()
            .into_simulation()
            .map_err(|error| format!("failed to restore characterization state: {error}"))?;
        let plant_step_map = shortest_distances(&simulation, &plants, false);
        let plant_weighted_map = shortest_distances(&simulation, &plants, true);
        let food_step_map = shortest_distances(&simulation, &major_food, false);
        let food_weighted_map = shortest_distances(&simulation, &major_food, true);
        plant_steps.extend(starts.iter().map(|start| plant_step_map[start.0]));
        plant_weighted.extend(starts.iter().map(|start| plant_weighted_map[start.0]));
        food_steps.extend(starts.iter().map(|start| food_step_map[start.0]));
        food_weighted.extend(starts.iter().map(|start| food_weighted_map[start.0]));
    }
    let plant = FoodDistanceCharacterization {
        source_tiles_across_seeds: total_plant_tiles,
        move_steps: distance_distribution(&plant_steps, 1.0),
        weighted_distance_units: distance_distribution(&plant_weighted, 1024.0),
    };
    let major = FoodDistanceCharacterization {
        source_tiles_across_seeds: total_food_tiles,
        move_steps: distance_distribution(&food_steps, 1.0),
        weighted_distance_units: distance_distribution(&food_weighted, 1024.0),
    };
    Ok((
        plant,
        major,
        compiled_ruleset_hash.expect("validated nonempty seeds"),
    ))
}

#[derive(Debug, Clone, Copy)]
struct AttackTrial {
    victim_energy_at_volley: u64,
    killed: bool,
    landed: usize,
    raw_damage: u64,
    applied_damage: u64,
    surviving_attackers: usize,
}

fn attack_sources(
    simulation: &ReferenceSimulation,
    victim: TileIndex,
) -> Vec<(TileIndex, LocalSlot)> {
    let neighborhood = simulation.neighborhood();
    let mut sources = Vec::new();
    for source in 0..neighborhood.tile_count() {
        if source == victim.0 {
            continue;
        }
        for slot_index in 0..neighborhood.slot_count() {
            let Ok(slot_u8) = u8::try_from(slot_index) else {
                continue;
            };
            let slot = LocalSlot(slot_u8);
            if neighborhood.action_allows(TargetingAction::Attack, slot)
                && neighborhood.target(TileIndex(source), slot) == Some(victim)
            {
                sources.push((TileIndex(source), slot));
                break;
            }
        }
    }
    sources
}

fn siege_trial(
    env: &EnvConfig,
    attacker_count: usize,
    max_micro_actions: usize,
) -> Result<SustainedSiegeTrial, String> {
    let mut simulation =
        ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
            .map_err(|error| format!("failed to create sustained-siege micro-world: {error}"))?;
    let victim_tile = TileIndex((env.world_size / 2) * env.world_size + env.world_size / 2);
    {
        let plant = simulation
            .tile_state_mut(victim_tile)
            .ok_or("sustained-siege victim tile is absent")?;
        plant.plant_energy = u64::from(env.plant_max_energy);
        plant.plant_capacity = u64::from(env.plant_max_energy);
        plant.plant_growth_rate = u64::from(env.plant_rate);
    }
    let victim = simulation
        .add_cell(
            victim_tile,
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add sustained-siege victim: {error}"))?;
    let sources = attack_sources(&simulation, victim_tile);
    if attacker_count > sources.len() {
        return Err("sustained siege requested too many local attackers".into());
    }
    let mut attackers = Vec::with_capacity(attacker_count);
    for (tile, slot) in sources.into_iter().take(attacker_count) {
        let actor = simulation
            .add_cell(
                tile,
                u64::from(env.min_energy),
                u64::from(env.initial_energy),
                1,
            )
            .map_err(|error| format!("failed to add sustained-siege attacker: {error}"))?;
        attackers.push((actor, slot));
    }
    let attack_effort = CharacterizedEffort::High;
    let attack_cost = effort_cost(&env.rules, env.rules.attack_effort_base, attack_effort)
        .ok_or("sustained-siege attack cost overflowed")?;
    // A plant-turtling defender reacts to the already-adjacent attackers by
    // establishing guard first, then alternates a feeding window and a guard
    // window. This makes the exposed consume interval explicit.
    let mut prefer_consume = false;
    let mut commitments = 0usize;
    let mut completed_attacks = 0usize;
    let mut applied_damage = 0_u64;
    let mut completed_consumes = 0usize;
    let mut completed_guards = 0usize;

    while simulation.cell(victim).is_some()
        && simulation.now().0 < env.victory.sim_time_limit_quanta
        && commitments < max_micro_actions
    {
        let threat_exists = attackers.iter().any(|(actor, _)| {
            simulation.cell(*actor).is_some_and(|state| {
                state.pending_action.is_some()
                    || state
                        .assimilated_energy
                        .checked_sub(attack_cost)
                        .and_then(|energy| energy.checked_sub(env.rules.minimum_survival_energy))
                        .is_some_and(|maximum| maximum > 0)
            })
        });
        if !threat_exists {
            break;
        }

        if simulation.cell(victim).is_some_and(|state| {
            state.pending_action.is_none() && state.is_ready_at(simulation.now())
        }) && commitments < max_micro_actions
        {
            let state = simulation
                .cell(victim)
                .ok_or("sustained-siege victim disappeared before its decision")?;
            let tile = simulation
                .tile_state(victim_tile)
                .ok_or("sustained-siege victim tile disappeared")?;
            let available_food = tile.plant_energy.saturating_add(tile.loose_energy);
            let gut_room = env.rules.gut_capacity.saturating_sub(state.gut_energy);
            let amount = env.rules.bite_capacity.min(gut_room).min(available_food);
            let request = if prefer_consume && amount > 0 {
                ActionRequest::Consume { amount }
            } else {
                ActionRequest::Guard {
                    effort: EffortTier::Standard,
                }
            };
            let receipt = simulation
                .commit_action(victim, request)
                .map_err(|error| format!("failed to commit sustained-siege defense: {error}"))?;
            if receipt.accepted {
                prefer_consume = !prefer_consume;
            }
            commitments += 1;
        }

        for (attacker, slot) in &attackers {
            if commitments >= max_micro_actions {
                break;
            }
            let Some(state) = simulation.cell(*attacker) else {
                continue;
            };
            if state.pending_action.is_some() || !state.is_ready_at(simulation.now()) {
                continue;
            }
            let Some(maximum) = state
                .assimilated_energy
                .checked_sub(attack_cost)
                .and_then(|energy| energy.checked_sub(env.rules.minimum_survival_energy))
                .filter(|maximum| *maximum > 0)
            else {
                continue;
            };
            let payload = ceil_ratio(maximum, 1, 4)
                .ok_or("sustained-siege payload calculation overflowed")?;
            let receipt = simulation
                .commit_action(
                    *attacker,
                    ActionRequest::Attack {
                        target: *slot,
                        effort: attack_effort.resolver(),
                        payload,
                    },
                )
                .map_err(|error| format!("failed to commit sustained-siege attack: {error}"))?;
            if !receipt.accepted {
                return Err("derived sustained-siege attack was rejected".into());
            }
            commitments += 1;
        }

        let any_pending = simulation
            .cells()
            .values()
            .any(|state| state.pending_action.is_some());
        if !any_pending {
            break;
        }
        let report = simulation
            .resolve_next_batch()
            .map_err(|error| format!("failed to resolve sustained-siege frontier: {error}"))?;
        for outcome in report.outcomes {
            if !matches!(outcome.status, OutcomeStatus::Success) {
                continue;
            }
            if outcome.actor == victim {
                match outcome.action {
                    ActionKind::Consume => completed_consumes += 1,
                    ActionKind::Guard => completed_guards += 1,
                    _ => {}
                }
            } else if outcome.action == ActionKind::Attack {
                completed_attacks += 1;
                if let Some(damage) = outcome.attack_damage {
                    applied_damage = applied_damage
                        .checked_add(damage.applied)
                        .ok_or("sustained-siege applied damage overflowed")?;
                }
            }
        }
    }

    let final_victim = simulation.cell(victim);
    let threat_remains = attackers.iter().any(|(actor, _)| {
        simulation.cell(*actor).is_some_and(|state| {
            state.pending_action.is_some()
                || state
                    .assimilated_energy
                    .checked_sub(attack_cost)
                    .and_then(|energy| energy.checked_sub(env.rules.minimum_survival_energy))
                    .is_some_and(|maximum| maximum > 0)
        })
    });
    Ok(SustainedSiegeTrial {
        attackers: attacker_count,
        victim_killed: final_victim.is_none(),
        elapsed_quanta: simulation.now().0,
        completed_attacks,
        applied_damage,
        completed_consumes,
        completed_guards,
        surviving_attackers: attackers
            .iter()
            .filter(|(actor, _)| simulation.cell(*actor).is_some())
            .count(),
        victim_final_energy: final_victim.map(|state| state.assimilated_energy),
        victim_final_gut_energy: final_victim.map(|state| state.gut_energy),
        plant_final_energy: simulation
            .tile_state(victim_tile)
            .ok_or("sustained-siege plant tile disappeared")?
            .plant_energy,
        censored: final_victim.is_some()
            && threat_remains
            && (commitments >= max_micro_actions
                || simulation.now().0 >= env.victory.sim_time_limit_quanta),
    })
}

fn characterize_sustained_siege(
    env: &EnvConfig,
    max_micro_actions: usize,
) -> Result<SustainedSiegeCharacterization, String> {
    let probe = ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
        .map_err(|error| format!("failed to compile sustained-siege neighborhood: {error}"))?;
    let victim_tile = TileIndex((env.world_size / 2) * env.world_size + env.world_size / 2);
    let maximum_local_attackers = attack_sources(&probe, victim_tile).len();
    let mut trials = Vec::with_capacity(maximum_local_attackers);
    for attackers in 1..=maximum_local_attackers {
        trials.push(siege_trial(env, attackers, max_micro_actions)?);
    }
    let minimum_attackers_to_kill = trials
        .iter()
        .find(|trial| trial.victim_killed)
        .map(|trial| trial.attackers);
    Ok(SustainedSiegeCharacterization {
        attacker_effort: CharacterizedEffort::High,
        payload_fraction_numerator: 1,
        payload_fraction_denominator: 4,
        time_limit_quanta: env.victory.sim_time_limit_quanta,
        maximum_local_attackers,
        minimum_attackers_to_kill,
        trials,
    })
}

fn attack_trial(
    env: &EnvConfig,
    effort: CharacterizedEffort,
    attacker_count: usize,
    guarded: bool,
    full_payload: u64,
) -> Result<Option<AttackTrial>, String> {
    let mut simulation =
        ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
            .map_err(|error| format!("failed to create attack micro-world: {error}"))?;
    let victim_tile = TileIndex((env.world_size / 2) * env.world_size + env.world_size / 2);
    {
        let plant = simulation
            .tile_state_mut(victim_tile)
            .ok_or("attack victim tile is absent")?;
        plant.plant_energy = u64::from(env.plant_max_energy / 2);
        plant.plant_capacity = u64::from(env.plant_max_energy);
        plant.plant_growth_rate = u64::from(env.plant_rate);
    }
    let victim = simulation
        .add_cell(
            victim_tile,
            u64::from(env.min_energy),
            u64::from(env.initial_energy),
            0,
        )
        .map_err(|error| format!("failed to add attack victim: {error}"))?;
    if guarded {
        let receipt = simulation
            .commit_action(
                victim,
                ActionRequest::Guard {
                    effort: EffortTier::Standard,
                },
            )
            .map_err(|error| format!("failed to commit victim guard: {error}"))?;
        if !receipt.accepted {
            return Ok(None);
        }
        if simulation.cell(victim).is_none() {
            return Ok(None);
        }
        while simulation
            .cell(victim)
            .is_some_and(|state| state.pending_action.is_some())
        {
            simulation
                .resolve_next_batch()
                .map_err(|error| format!("failed to establish victim guard: {error}"))?;
        }
    }
    let victim_energy_at_volley = simulation
        .cell(victim)
        .ok_or("victim died before the attack volley")?
        .assimilated_energy;
    let sources = attack_sources(&simulation, victim_tile);
    if attacker_count > sources.len() {
        return Err("attack trial requested more attackers than local attack positions".into());
    }
    let mut attackers = Vec::with_capacity(attacker_count);
    for (tile, slot) in sources.into_iter().take(attacker_count) {
        let attacker = simulation
            .add_cell(
                tile,
                u64::from(env.min_energy),
                u64::from(env.initial_energy),
                0,
            )
            .map_err(|error| format!("failed to add attack micro-cell: {error}"))?;
        let receipt = simulation
            .commit_action(
                attacker,
                ActionRequest::Attack {
                    target: slot,
                    effort: effort.resolver(),
                    payload: full_payload,
                },
            )
            .map_err(|error| format!("failed to commit full-strength attack: {error}"))?;
        if !receipt.accepted {
            return Err("derived full-strength attack was rejected by the resolver".into());
        }
        attackers.push(attacker);
    }
    let mut landed = 0usize;
    let mut raw_damage = 0_u64;
    let mut applied_damage = 0_u64;
    while attackers.iter().any(|attacker| {
        simulation
            .cell(*attacker)
            .is_some_and(|state| state.pending_action.is_some())
    }) {
        let report = simulation
            .resolve_next_batch()
            .map_err(|error| format!("failed to resolve attack volley: {error}"))?;
        for outcome in report.outcomes {
            if outcome.action != ActionKind::Attack
                || !matches!(outcome.status, OutcomeStatus::Success)
            {
                continue;
            }
            if let Some(damage) = outcome.attack_damage {
                landed += 1;
                raw_damage = raw_damage
                    .checked_add(damage.raw)
                    .ok_or("attack raw damage overflowed")?;
                applied_damage = applied_damage
                    .checked_add(damage.applied)
                    .ok_or("attack applied damage overflowed")?;
            }
        }
    }
    Ok(Some(AttackTrial {
        victim_energy_at_volley,
        killed: simulation.cell(victim).is_none(),
        landed,
        raw_damage,
        applied_damage,
        surviving_attackers: attackers
            .iter()
            .filter(|attacker| simulation.cell(**attacker).is_some())
            .count(),
    }))
}

fn characterize_attack(
    env: &EnvConfig,
    effort: CharacterizedEffort,
) -> Result<AttackVolleyCharacterization, String> {
    let cost = effort_cost(&env.rules, env.rules.attack_effort_base, effort)
        .ok_or("attack effort cost overflowed")?;
    let full_payload = u64::from(env.initial_energy)
        .checked_sub(cost)
        .and_then(|energy| energy.checked_sub(env.rules.minimum_survival_energy));
    let probe = ReferenceSimulation::new(env.world_size, env.world_size, env.rules.clone())
        .map_err(|error| format!("failed to compile attack neighborhood: {error}"))?;
    let victim_tile = TileIndex((env.world_size / 2) * env.world_size + env.world_size / 2);
    let maximum_local_attackers = attack_sources(&probe, victim_tile).len();
    let Some(full_payload) = full_payload.filter(|payload| *payload > 0) else {
        return Ok(AttackVolleyCharacterization {
            effort,
            full_strength_payload: None,
            maximum_local_attackers,
            single_attack_lands: false,
            single_attacker_survives: false,
            single_attack_raw_damage: 0,
            single_attack_applied_damage: 0,
            unguarded_attackers_required: None,
            guarded_attackers_required: None,
            unguarded_victim_energy_at_volley: u64::from(env.initial_energy),
            guarded_victim_energy_at_volley: None,
        });
    };
    if maximum_local_attackers == 0 {
        return Ok(AttackVolleyCharacterization {
            effort,
            full_strength_payload: Some(full_payload),
            maximum_local_attackers,
            single_attack_lands: false,
            single_attacker_survives: false,
            single_attack_raw_damage: 0,
            single_attack_applied_damage: 0,
            unguarded_attackers_required: None,
            guarded_attackers_required: None,
            unguarded_victim_energy_at_volley: u64::from(env.initial_energy),
            guarded_victim_energy_at_volley: attack_trial(env, effort, 0, true, full_payload)?
                .map(|trial| trial.victim_energy_at_volley),
        });
    }
    let single = attack_trial(env, effort, 1, false, full_payload)?
        .expect("an unguarded initial victim is available");
    let guarded_single = attack_trial(env, effort, 1, true, full_payload)?;
    let mut unguarded_required = single.killed.then_some(1);
    let mut guarded_required = guarded_single
        .as_ref()
        .and_then(|trial| trial.killed.then_some(1));
    for attackers in 2..=maximum_local_attackers {
        if unguarded_required.is_none()
            && attack_trial(env, effort, attackers, false, full_payload)?
                .is_some_and(|trial| trial.killed)
        {
            unguarded_required = Some(attackers);
        }
        if guarded_single.is_some()
            && guarded_required.is_none()
            && attack_trial(env, effort, attackers, true, full_payload)?
                .is_some_and(|trial| trial.killed)
        {
            guarded_required = Some(attackers);
        }
        if unguarded_required.is_some() && guarded_required.is_some() {
            break;
        }
    }
    Ok(AttackVolleyCharacterization {
        effort,
        full_strength_payload: Some(full_payload),
        maximum_local_attackers,
        single_attack_lands: single.landed == 1,
        single_attacker_survives: single.surviving_attackers == 1,
        single_attack_raw_damage: single.raw_damage,
        single_attack_applied_damage: single.applied_damage,
        unguarded_attackers_required: unguarded_required,
        guarded_attackers_required: guarded_required,
        unguarded_victim_energy_at_volley: single.victim_energy_at_volley,
        guarded_victim_energy_at_volley: guarded_single
            .as_ref()
            .map(|trial| trial.victim_energy_at_volley),
    })
}

pub fn characterize_ecology(
    env: &EnvConfig,
    options: &EcologicalCharacterizationOptions,
) -> Result<EcologicalCharacterizationReport, String> {
    options.validate()?;
    let scenario = ScenarioProfile::from(env);
    let scenario_hash = scenario.semantic_hash()?;
    let semantic_ruleset_hash = env.rules.semantic_hash().to_string();
    let (distance_to_plants, distance_to_major_food, compiled_ruleset_hash) =
        characterize_food_distances(env, &options.seeds)?;
    let stationary_lifetime_quanta = stationary_lifetime(env)?;
    let mut travel = Vec::new();
    for effort in CharacterizedEffort::ALL {
        if let Some(result) = travel_endurance(env, effort, options.max_micro_actions)? {
            travel.push(result);
        }
    }
    let mut attack_volleys = Vec::new();
    for effort in CharacterizedEffort::ALL {
        attack_volleys.push(characterize_attack(env, effort)?);
    }
    let feeding_cycle = characterize_feeding_cycle(env)?;
    let reproduction_break_even = characterize_reproduction(env, options.max_micro_actions)?;
    let terrain_cycle = characterize_terrain_cycle(env)?;
    let signals = characterize_signals(env, options.max_micro_actions)?;
    let sustained_siege = characterize_sustained_siege(env, options.max_micro_actions)?;
    let best_distance = travel
        .iter()
        .filter(|travel| !travel.censored)
        .map(|travel| travel.completed_distance_q10 as f64 / 1024.0)
        .fold(0.0_f64, f64::max);
    let high = attack_volleys
        .iter()
        .find(|volley| volley.effort == CharacterizedEffort::High)
        .expect("all effort tiers were characterized");
    let collapse_indicators = CollapseIndicators {
        no_bidirectional_movement: travel.is_empty(),
        no_plant_sources: distance_to_plants.source_tiles_across_seeds == 0,
        some_starting_cells_cannot_reach_a_plant: distance_to_plants
            .weighted_distance_units
            .unreachable_samples
            > 0,
        median_plant_beyond_best_travel_endurance: distance_to_plants
            .weighted_distance_units
            .p50
            .is_some_and(|distance| distance > best_distance),
        p90_plant_beyond_best_travel_endurance: distance_to_plants
            .weighted_distance_units
            .p90
            .is_some_and(|distance| distance > best_distance),
        full_strength_high_attack_fails_to_land: !high.single_attack_lands,
        initial_cell_cannot_establish_standard_guard: high
            .guarded_victim_energy_at_volley
            .is_none(),
        unguarded_initial_cell_survives_maximum_local_high_volley: high
            .unguarded_attackers_required
            .is_none(),
        guarded_initial_cell_survives_maximum_local_high_volley: high
            .guarded_victim_energy_at_volley
            .is_some()
            && high.guarded_attackers_required.is_none(),
        maximum_bite_cannot_be_fully_digested: !feeding_cycle.survived_full_digestion,
        minimum_viable_reproduction_is_impossible: reproduction_break_even
            .minimum_successful_parent_energy
            .is_none(),
        initial_cell_cannot_complete_terrain_cycle: !terrain_cycle.deposition_succeeded,
        plant_defender_survives_bounded_maximum_siege: sustained_siege
            .trials
            .last()
            .is_some_and(|trial| !trial.victim_killed && !trial.censored),
        signaling_disabled: signals.emission_quantum == 0,
        minimum_signal_is_unaffordable: signals
            .impulse_trials
            .first()
            .is_none_or(|trial| !trial.accepted),
        minimum_signal_expires_before_its_action_completes: signals
            .impulse_trials
            .first()
            .is_some_and(|trial| trial.accepted && trial.field_energy_after_completion == 0),
        no_neighbor_signal_observation_edges: signals.directed_observation_edges == 0,
        terrain_disruption_probe_unavailable: signals.terrain_disruption.is_none(),
    };
    let characterization_hash = hash_json(&CharacterizationIdentity {
        semantic_ruleset_hash: &semantic_ruleset_hash,
        compiled_ruleset_hash: &compiled_ruleset_hash,
        scenario_hash: &scenario_hash,
        options,
    })?;
    let results_hash = hash_json(&CharacterizationResults {
        stationary_lifetime_quanta,
        travel: &travel,
        distance_to_plants: &distance_to_plants,
        distance_to_major_food: &distance_to_major_food,
        attack_volleys: &attack_volleys,
        feeding_cycle: &feeding_cycle,
        reproduction_break_even: &reproduction_break_even,
        terrain_cycle: &terrain_cycle,
        signals: &signals,
        sustained_siege: &sustained_siege,
        collapse_indicators: &collapse_indicators,
    })?;
    Ok(EcologicalCharacterizationReport {
        schema_version: ECOLOGICAL_CHARACTERIZATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        characterization_hash,
        results_hash,
        semantic_ruleset_hash,
        compiled_ruleset_hash,
        scenario_hash,
        scenario,
        rules: env.rules.clone(),
        options: options.clone(),
        stationary_lifetime_quanta,
        travel,
        distance_to_plants,
        distance_to_major_food,
        attack_volleys,
        feeding_cycle,
        reproduction_break_even,
        terrain_cycle,
        signals,
        sustained_siege,
        collapse_indicators,
    })
}

pub fn publish_ecological_characterization(
    output: &Path,
    report: &EcologicalCharacterizationReport,
) -> Result<PathBuf, String> {
    validate_ecological_characterization(report)?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable ecological characterization {}",
            output.display()
        ));
    }
    let nonce = CHARACTERIZATION_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "characterization output needs a UTF-8 file name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode ecological characterization: {error}"))?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        fs::hard_link(&temporary, output)
            .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
        fs::remove_file(&temporary)
            .map_err(|error| format!("failed to remove {}: {error}", temporary.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map(|()| output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VictoryConfig;

    fn small_env() -> EnvConfig {
        EnvConfig {
            world_size: 8,
            cells_per_team: 2,
            num_scattered_energy: 4,
            num_plants: 2,
            victory: VictoryConfig {
                sim_time_limit_quanta: 16_384,
                ..VictoryConfig::default()
            },
            ..EnvConfig::default()
        }
    }

    #[test]
    fn characterization_is_reproducible_and_exercises_authoritative_seams() {
        let options = EcologicalCharacterizationOptions {
            seeds: vec![10, 11, 12],
            max_micro_actions: 1_000,
        };
        let left = characterize_ecology(&small_env(), &options).unwrap();
        let right = characterize_ecology(&small_env(), &options).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.distance_to_plants.move_steps.samples, 6);
        assert_eq!(left.attack_volleys.len(), 3);
        assert_eq!(left.stationary_lifetime_quanta, Some(102_400));
        assert!(left.travel.iter().all(|travel| !travel.censored));
        assert_eq!(
            left.travel
                .iter()
                .map(|travel| (travel.effort, travel.completed_moves))
                .collect::<Vec<_>>(),
            [
                (CharacterizedEffort::Low, 43),
                (CharacterizedEffort::Standard, 31),
                (CharacterizedEffort::High, 19),
            ]
        );
        assert!(!left.attack_volleys[0].single_attack_lands);
        assert_eq!(left.attack_volleys[1].unguarded_attackers_required, Some(2));
        assert_eq!(left.attack_volleys[1].guarded_attackers_required, Some(2));
        assert_eq!(left.attack_volleys[2].unguarded_attackers_required, Some(2));
        assert_eq!(left.attack_volleys[2].guarded_attackers_required, Some(3));
        assert_eq!(left.feeding_cycle.consumed_energy, 16);
        assert!(left.feeding_cycle.survived_full_digestion);
        assert!(left
            .reproduction_break_even
            .minimum_successful_parent_energy
            .is_some());
        assert!(left.terrain_cycle.deposition_succeeded);
        assert_eq!(left.signals.emission_quantum, 1);
        assert_eq!(left.signals.neighbor_observation_slots, 8);
        assert_eq!(left.signals.minimum_distinct_observable_neighbors, 8);
        assert_eq!(left.signals.maximum_distinct_observable_neighbors, 8);
        assert_eq!(left.signals.directed_observation_edges, 8 * 8 * 8);
        assert_eq!(left.signals.impulse_trials.len(), 5);
        assert_eq!(left.signals.mass_shedding_trials.len(), 6);
        assert!(left
            .signals
            .impulse_trials
            .iter()
            .all(|trial| trial.accepted));
        assert_eq!(
            left.signals.impulse_trials[0].time_to_extinction_quanta,
            Some(1_024)
        );
        assert_eq!(
            left.signals.impulse_trials[0].field_energy_after_completion,
            0
        );
        let wait_control = &left.signals.mass_shedding_trials[0];
        let one_quantum = &left.signals.mass_shedding_trials[1];
        let sixteen_quanta = &left.signals.mass_shedding_trials[5];
        assert_eq!(wait_control.total_mass_after_preparation, Some(109));
        assert_eq!(wait_control.first_move_effort_spent, Some(3));
        assert_eq!(wait_control.first_move_duration_quanta, Some(1_216));
        assert_eq!(wait_control.completed_moves, 31);
        assert_eq!(one_quantum.total_mass_after_preparation, Some(108));
        assert_eq!(one_quantum.first_move_effort_spent, Some(3));
        assert_eq!(one_quantum.completed_moves, 31);
        assert_eq!(one_quantum.total_move_effort_spent, 64);
        assert_eq!(wait_control.total_move_effort_spent, 65);
        assert_eq!(
            one_quantum.travel_death_time_quanta,
            wait_control.travel_death_time_quanta
        );
        assert_eq!(sixteen_quanta.first_move_effort_spent, Some(2));
        assert_eq!(sixteen_quanta.first_move_duration_quanta, Some(1_152));
        assert_eq!(sixteen_quanta.completed_moves, 27);
        assert!(left
            .signals
            .terrain_disruption
            .as_ref()
            .is_some_and(|probe| probe.erased_by_terrain_action > 0));
        assert!(
            left.collapse_indicators
                .minimum_signal_expires_before_its_action_completes
        );
        assert_eq!(
            left.sustained_siege.trials.len(),
            left.sustained_siege.maximum_local_attackers
        );
        assert!(left
            .sustained_siege
            .trials
            .iter()
            .all(|trial| !trial.censored));
    }

    #[test]
    fn missing_food_and_immutable_publication_are_explicit() {
        let env = EnvConfig {
            world_size: 4,
            cells_per_team: 1,
            num_scattered_energy: 0,
            num_plants: 0,
            ..EnvConfig::default()
        };
        let report = characterize_ecology(
            &env,
            &EcologicalCharacterizationOptions {
                seeds: vec![5],
                max_micro_actions: 1_000,
            },
        )
        .unwrap();
        assert!(report.collapse_indicators.no_plant_sources);
        assert_eq!(report.distance_to_plants.move_steps.reachable_samples, 0);
        assert_eq!(report.distance_to_plants.move_steps.unreachable_samples, 1);

        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("characterization.json");
        publish_ecological_characterization(&output, &report).unwrap();
        assert!(publish_ecological_characterization(&output, &report).is_err());
        let loaded: EcologicalCharacterizationReport =
            serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(loaded, report);
        validate_ecological_characterization(&loaded).unwrap();

        let mut tampered = loaded.clone();
        tampered.scenario.initial_energy += 1;
        assert!(validate_ecological_characterization(&tampered).is_err());

        let mut tampered = report;
        tampered.stationary_lifetime_quanta = Some(1);
        assert!(validate_ecological_characterization(&tampered).is_err());

        let mut tampered = loaded;
        tampered.sustained_siege.trials[0].applied_damage += 1;
        assert!(validate_ecological_characterization(&tampered).is_err());
    }

    #[test]
    fn lifecycle_collapse_and_siege_clock_horizon_are_explicit() {
        let mut no_digestion = small_env();
        no_digestion.rules.digestion_rate_numerator = 0;
        let feeding = characterize_feeding_cycle(&no_digestion).unwrap();
        assert!(!feeding.survived_full_digestion);
        assert_eq!(feeding.gut_after_digestion, Some(16));

        let mut poor_builder = small_env();
        poor_builder.initial_energy = 2;
        let terrain = characterize_terrain_cycle(&poor_builder).unwrap();
        assert!(!terrain.excavation_succeeded);
        assert!(!terrain.deposition_succeeded);

        let mut no_split_target = small_env();
        no_split_target.world_size = 1;
        let reproduction = characterize_reproduction(&no_split_target, 100).unwrap();
        assert!(reproduction.minimum_successful_parent_energy.is_none());

        let mut short_siege = small_env();
        short_siege.victory.sim_time_limit_quanta = 1_024;
        let siege = characterize_sustained_siege(&short_siege, 1_000).unwrap();
        assert_eq!(siege.time_limit_quanta, 1_024);
        assert!(siege.trials[0].censored);
        assert_eq!(siege.trials[0].elapsed_quanta, 1_024);

        let mut no_signal = small_env();
        no_signal.rules.signal_emission_cost = 0;
        let signals = characterize_signals(&no_signal, 1_000).unwrap();
        assert!(signals.impulse_trials.iter().all(|trial| !trial.accepted));
        assert!(signals.mass_shedding_trials[0].preparation_accepted);
        assert!(signals
            .mass_shedding_trials
            .iter()
            .skip(1)
            .all(|trial| !trial.preparation_accepted));
        assert!(signals.terrain_disruption.is_none());

        let mut current_tile_only = small_env();
        current_tile_only.rules.neighborhood.observations.signal =
            blob_engine::resolution::SlotMask::empty();
        let signals = characterize_signals(&current_tile_only, 1_000).unwrap();
        assert_eq!(signals.neighbor_observation_slots, 0);
        assert_eq!(signals.directed_observation_edges, 0);
    }
}
