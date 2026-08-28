//! BlobEnv — wraps blob_engine::Engine as an RL environment.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use blob_engine::engine::{CellConfig, Engine, ReferenceRuntimeCheckpoint, TickEvents};
use blob_engine::resolution::{
    BoundaryRule, CellKey, IntegrityMode, ReferenceCheckpoint, ReferenceSimulation,
    ReplayBundleLimits, TargetingAction, TileIndex,
};
use blob_engine::world_gen;
use blob_interface::cell::Cell;
use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{
    ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate, ReferenceMind,
    ReferenceMindAction, ReferenceMindDecision, ReferenceMindInput,
};
use blob_interface::types::{CellId, Coordinate, TeamId};
use blob_interface::world::EnergySource;
use burn::prelude::Backend;

use crate::action::{
    action_is_commit_legal, action_mask, attach_policy_memory, decode_action, decode_policy_choice,
    PolicyChoice,
};
use crate::config::{
    DeadlineRewardMode, EnvConfig, OpponentProfile, ResourcePlacement, RewardConfig,
};
use crate::match_explorer::{MatchExplorerConfig, MatchExplorerRecorder, RecordedExplorerMatch};
use crate::model::{policy_memory_bytes, PolicyValueNet};
use crate::observation::Observation;
use crate::opponent::{BatchedSnapshotPolicy, SnapshotBatchPolicy, SnapshotPolicyMind};
use crate::telemetry::{EcologySample, StepTelemetry, TelemetryConfig, TelemetrySide};

pub(crate) type OpponentMindFactory = Arc<dyn Fn() -> Box<dyn ReferenceMind> + Send + Sync>;
type OpponentBatchPolicyFactory = Arc<dyn Fn() -> Box<dyn SnapshotBatchPolicy> + Send + Sync>;

/// Host-only pre-match override for asymmetric evaluation scenarios. The
/// resulting cells enter the same canonical state and receive the same Mind
/// inputs as cells created through the symmetric environment constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpponentStartingState {
    pub cells_per_team: usize,
    pub initial_energy: u32,
}

/// Exact update-boundary state for one RL environment. Canonical physics stays
/// in the engine checkpoint; host cells preserve team ownership and private
/// dispatch/randomness metadata that intentionally is not part of the Mind ABI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlobEnvCheckpoint {
    pub canonical_checkpoint: Vec<u8>,
    pub iteration: u64,
    pub host_cells: Vec<Cell>,
    pub episode_step: u64,
}

/// An RL environment wrapping the blob game engine.
///
/// The training team's cells get actions injected externally.
/// Opponent teams use a fixed mind implementation.
pub struct BlobEnv {
    engine: Engine,
    reward_config: RewardConfig,
    env_config: EnvConfig,
    training_team: TeamId,
    episode_step: u64,
    /// Previous canonical stored biological energy (assimilated + gut) and
    /// host-private team ownership for reward attribution.
    prev_cell_energies: HashMap<CellId, (u64, TeamId)>,
    prev_cell_count: HashMap<TeamId, usize>,
    /// Previous cell positions for proximity reward shaping
    prev_cell_positions: HashMap<CellId, Coordinate>,
    opponent_mind_factory: OpponentMindFactory,
    opponent_batch_policy_factory: Option<OpponentBatchPolicyFactory>,
    opponent_batch_policy: Option<Box<dyn SnapshotBatchPolicy>>,
    opponent_starting_state: Option<OpponentStartingState>,
    telemetry: Option<EnvTelemetryRuntime>,
    match_explorer: Option<MatchExplorerRecorder>,
}

/// Minimal trusted-host view of the initial ecology used by offline
/// characterization. It is never exposed through the Mind ABI and avoids
/// serializing a complete large-world checkpoint merely to inspect placement.
pub(crate) struct InitialEcologySnapshot {
    pub starts_by_team: Vec<Vec<TileIndex>>,
    pub plants: Vec<TileIndex>,
    pub major_food: Vec<TileIndex>,
}

struct EnvTelemetryRuntime {
    config: TelemetryConfig,
    sides: HashMap<CellKey, TelemetrySide>,
}

/// Opponent/fallback mind. RL actions bypass the mind ABI through engine-side
/// action overrides keyed by private `CellId` values.
pub struct ActionBufferMind {
    fallback: OpponentProfile,
}

impl ActionBufferMind {
    pub fn new_training() -> Self {
        ActionBufferMind {
            fallback: OpponentProfile::Wait,
        }
    }

    pub fn new_opponent(profile: OpponentProfile) -> Self {
        ActionBufferMind { fallback: profile }
    }
}

impl ReferenceMind for ActionBufferMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let action = match &self.fallback {
            OpponentProfile::Wait => ReferenceMindAction::Wait,
            OpponentProfile::Random => {
                let mask = action_mask(input);
                let valid_count = mask.iter().filter(|allowed| **allowed).count();
                if valid_count == 0 {
                    return ReferenceMindDecision {
                        action: ReferenceMindAction::Wait,
                        signal: None,
                        memory_update: ReferenceMemoryUpdate::Retain,
                    };
                }
                let selected_rank = input.randomness.sample_u64(0) as usize % valid_count;
                let selected = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(index, allowed)| allowed.then_some(index))
                    .nth(selected_rank)
                    .expect("Wait guarantees at least one allowed baseline action");
                decode_action(selected, input).action
            }
            OpponentProfile::Forager => {
                if input.action_space.consume_enabled
                    && input.action_space.max_consume_amount > 0
                    && input.current_tile.plant_energy + input.current_tile.loose_energy > 0
                {
                    ReferenceMindAction::Consume {
                        amount: input.action_space.max_consume_amount,
                    }
                } else {
                    let open_split_target = input.slots.iter().find(|slot| {
                        slot.reachable
                            && slot.neighbor.is_none()
                            && ReferenceActionSpace::allows_target(
                                input.action_space.split_targets,
                                slot.slot,
                            )
                    });
                    let minimum_child = input
                        .action_space
                        .child_core_mass
                        .saturating_add(input.action_space.minimum_survival_energy);
                    let split_threshold = minimum_child
                        .saturating_mul(3)
                        .max(input.action_space.minimum_survival_energy.saturating_add(1));
                    if input.self_state.assimilated_energy >= split_threshold {
                        if let Some(target) = open_split_target {
                            ReferenceMindAction::Split {
                                target_slot: target.slot,
                                child_allocation: minimum_child,
                                marker: 0,
                                private_memory: Vec::new(),
                            }
                        } else {
                            forager_move(input)
                        }
                    } else {
                        forager_move(input)
                    }
                }
            }
            OpponentProfile::Aggressive => aggressive_action(input, false),
            OpponentProfile::StochasticAggressive => aggressive_action(input, true),
            OpponentProfile::Defensive => {
                let threatened = input.slots.iter().any(|slot| slot.neighbor.is_some());
                if threatened {
                    ReferenceMindAction::Guard {
                        effort: ReferenceEffort::Standard,
                    }
                } else if input.action_space.consume_enabled
                    && input.action_space.max_consume_amount > 0
                    && input.current_tile.plant_energy + input.current_tile.loose_energy > 0
                {
                    ReferenceMindAction::Consume {
                        amount: input.action_space.max_consume_amount,
                    }
                } else {
                    forager_move(input)
                }
            }
        };
        let action = if action_is_commit_legal(input, &action, false) {
            action
        } else {
            ReferenceMindAction::Wait
        };
        ReferenceMindDecision {
            action,
            signal: None,
            memory_update: ReferenceMemoryUpdate::Retain,
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn aggressive_action(input: &ReferenceMindInput, stochastic: bool) -> ReferenceMindAction {
    if input.self_state.assimilated_energy < 40
        && input.action_space.consume_enabled
        && input.action_space.max_consume_amount > 0
        && input.current_tile.plant_energy + input.current_tile.loose_energy > 0
    {
        return ReferenceMindAction::Consume {
            amount: input.action_space.max_consume_amount,
        };
    }
    if let Some(target) = input.slots.iter().find(|slot| {
        slot.neighbor.is_some()
            && slot.reachable
            && ReferenceActionSpace::allows_target(input.action_space.attack_targets, slot.slot)
    }) {
        let sample = input.randomness.sample_u64(1);
        if stochastic && sample.is_multiple_of(4) {
            return ReferenceMindAction::Wait;
        }
        let divisor = if stochastic {
            match (sample / 4) % 3 {
                0 => 16,
                1 => 8,
                _ => 4,
            }
        } else {
            8
        };
        let payload = input
            .self_state
            .assimilated_energy
            .saturating_sub(input.action_space.minimum_survival_energy)
            / divisor;
        return if payload > 0 {
            ReferenceMindAction::Attack {
                target_slot: target.slot,
                effort: ReferenceEffort::Standard,
                payload,
            }
        } else {
            ReferenceMindAction::Wait
        };
    }
    let target = input.slots.iter().max_by_key(|slot| {
        let energy = slot.plant_energy.unwrap_or(0)
            + slot.loose_energy.unwrap_or(0)
            + if slot.plant_growth_rate.unwrap_or(0) > 0 {
                slot.diffuse_energy.unwrap_or(0)
            } else {
                0
            };
        let allowed = slot.reachable
            && slot.neighbor.is_none()
            && ReferenceActionSpace::allows_target(input.action_space.move_targets, slot.slot);
        (allowed, energy)
    });
    target.map_or(ReferenceMindAction::Wait, |target| {
        if target.reachable
            && target.neighbor.is_none()
            && ReferenceActionSpace::allows_target(input.action_space.move_targets, target.slot)
        {
            ReferenceMindAction::Move {
                target_slot: target.slot,
                effort: ReferenceEffort::Standard,
            }
        } else {
            ReferenceMindAction::Wait
        }
    })
}

fn forager_move(input: &ReferenceMindInput) -> ReferenceMindAction {
    let target = input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(input.action_space.move_targets, slot.slot)
        })
        .max_by_key(|slot| {
            slot.plant_energy.unwrap_or(0)
                + slot.loose_energy.unwrap_or(0)
                + if slot.plant_growth_rate.unwrap_or(0) > 0 {
                    slot.diffuse_energy.unwrap_or(0)
                } else {
                    0
                }
        });
    target.map_or(ReferenceMindAction::Wait, |target| {
        ReferenceMindAction::Move {
            target_slot: target.slot,
            effort: ReferenceEffort::Standard,
        }
    })
}

/// How an episode ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeOutcome {
    Win,         // all opponents eliminated
    Loss,        // all training cells died
    Timeout,     // canonical simulated-time deadline reached
    SafetyAbort, // non-scientific host decision-frontier ceiling reached
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeEndReason {
    Extermination,
    SimTimeDeadline,
    DecisionFrontierSafetyLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeadlineRewardOutcome {
    Win,
    Loss,
    Draw,
}

/// Output from a single environment step.
#[derive(Debug)]
pub struct StepOutput {
    /// Observations for each alive cell on the training team
    pub observations: Vec<(CellId, Observation)>,
    /// Same ready-cell frontier with each cell's isolated canonical private
    /// memory. This host routing metadata is never added to `Observation`.
    pub policy_observations: Vec<PolicyObservation>,
    /// Per-cell rewards
    pub rewards: HashMap<CellId, f32>,
    /// Whether the episode is done
    pub done: bool,
    /// If done, how the episode ended
    pub outcome: Option<EpisodeOutcome>,
    /// Canonical objective termination versus a non-scientific host guard.
    pub end_reason: Option<EpisodeEndReason>,
    /// Current episode step count
    pub episode_step: u64,
    /// Training cells alive at end of step
    pub training_cells: usize,
    /// Opponent cells alive at end of step
    pub opponent_cells: usize,
    /// Host-only authoritative telemetry. Disabled environments pay no state
    /// scan cost and return `None`.
    pub telemetry: Option<StepTelemetry>,
}

#[derive(Debug, Clone)]
pub struct PolicyObservation {
    pub cell_id: CellId,
    pub observation: Observation,
    pub private_memory: Vec<u8>,
}

fn configure_episode_resources(
    engine: &mut Engine,
    config: &EnvConfig,
    seed: u64,
) -> Result<(), String> {
    if config.resource_placement == ResourcePlacement::Random {
        let mut territories = (0..config.num_teams)
            .map(|team| (TeamId(team), Vec::new()))
            .collect::<Vec<_>>();
        for (cell_id, cell) in &engine.cells {
            let coordinate = *engine
                .inv_coordinate_map
                .get(cell_id)
                .ok_or("starting cell has no host coordinate")?;
            let (_, coordinates) = territories
                .get_mut(cell.team_id.0)
                .ok_or("starting cell names a team outside the environment")?;
            coordinates.push(coordinate);
        }
        for (_, coordinates) in &mut territories {
            coordinates.sort_unstable_by_key(|coordinate| (coordinate.y, coordinate.x));
        }
        engine.world.energy = world_gen::scatter_energy_with_layouts(
            config.world_size,
            config.world_size,
            config.num_scattered_energy,
            config.scattered_energy_amount,
            &config.scattered_energy_layout,
            config.num_plants,
            config.plant_rate,
            config.plant_max_energy,
            &config.plant_layout,
            &territories,
            seed,
        )?;
        return Ok(());
    }

    engine.world.energy.fill(None);
    let source = || EnergySource::Plant {
        rate: config.plant_rate,
        current_energy: config.plant_max_energy / 2,
        max_energy: config.plant_max_energy,
    };
    let include_all_teams = config.resource_placement == ResourcePlacement::OnAllCells;
    let mut training_cells = engine
        .cells
        .iter()
        .filter_map(|(id, cell)| (include_all_teams || cell.team_id == TeamId(0)).then_some(*id))
        .collect::<Vec<_>>();
    training_cells.sort_unstable_by_key(|cell| cell.0);
    let dimensions = engine.world.dimensions;

    for cell_id in training_cells {
        let origin = *engine
            .inv_coordinate_map
            .get(&cell_id)
            .ok_or("curriculum cell has no host coordinate")?;
        let target = match config.resource_placement {
            ResourcePlacement::Random => unreachable!(),
            ResourcePlacement::OnAllCells | ResourcePlacement::OnTrainingCells => origin,
            ResourcePlacement::AdjacentToTrainingCells => {
                adjacent_food_coordinate(engine, config, cell_id, origin, seed)?
            }
        };
        let index = target
            .y
            .checked_mul(dimensions.0)
            .and_then(|value| value.checked_add(target.x))
            .ok_or("curriculum resource coordinate overflowed")?;
        let tile = engine
            .world
            .energy
            .get_mut(index)
            .ok_or("curriculum resource coordinate is outside the world")?;
        *tile = Some(source());
    }
    Ok(())
}

fn adjacent_food_coordinate(
    engine: &Engine,
    config: &EnvConfig,
    cell_id: CellId,
    origin: Coordinate,
    seed: u64,
) -> Result<Coordinate, String> {
    let neighborhood = &config.rules.neighborhood;
    let movable = neighborhood.target_mask(TargetingAction::Move);
    let visible = neighborhood.observations.energy;
    let slot_count = neighborhood.slots.len();
    if slot_count == 0 {
        return Err("adjacent-food curriculum requires at least one local slot".into());
    }
    let rotation = usize::try_from(
        seed.wrapping_add((cell_id.0 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            % slot_count as u64,
    )
    .map_err(|_| "curriculum slot rotation does not fit usize")?;
    for offset_index in 0..slot_count {
        let slot_index = (rotation + offset_index) % slot_count;
        let slot = blob_engine::resolution::LocalSlot(slot_index as u8);
        if !movable.contains(slot) || !visible.contains(slot) {
            continue;
        }
        let offset = neighborhood.slots[slot_index];
        let Some(target) = offset_coordinate(
            origin,
            offset.dx,
            offset.dy,
            engine.world.dimensions,
            neighborhood.boundary_rule,
        ) else {
            continue;
        };
        if !engine.coordinate_map.contains_key(&target) {
            return Ok(target);
        }
    }
    Err(format!(
        "adjacent-food curriculum found no visible reachable vacancy for cell {}",
        cell_id.0
    ))
}

fn offset_coordinate(
    origin: Coordinate,
    dx: i8,
    dy: i8,
    dimensions: (usize, usize),
    boundary: BoundaryRule,
) -> Option<Coordinate> {
    let width = i64::try_from(dimensions.0).ok()?;
    let height = i64::try_from(dimensions.1).ok()?;
    let x = i64::try_from(origin.x).ok()?.checked_add(i64::from(dx))?;
    let y = i64::try_from(origin.y).ok()?.checked_add(i64::from(dy))?;
    let (x, y) = match boundary {
        BoundaryRule::Wrap => (x.rem_euclid(width), y.rem_euclid(height)),
        BoundaryRule::Bounded if x >= 0 && y >= 0 && x < width && y < height => (x, y),
        BoundaryRule::Bounded => return None,
    };
    Some(Coordinate {
        x: usize::try_from(x).ok()?,
        y: usize::try_from(y).ok()?,
    })
}

impl BlobEnv {
    /// Create a new BlobEnv.
    pub fn new(env_config: EnvConfig, reward_config: RewardConfig, seed: u64) -> Self {
        let profile = env_config.opponent;
        let opponent_mind_factory: OpponentMindFactory =
            Arc::new(move || Box::new(ActionBufferMind::new_opponent(profile)));
        Self::new_with_opponent_factory(
            env_config,
            reward_config,
            seed,
            opponent_mind_factory,
            None,
            None,
        )
    }

    /// Create an otherwise ordinary environment with an explicit opponent
    /// starting population and energy. Intended for strict host-side micro
    /// scenarios; no scenario identity is projected into policy input.
    pub fn new_with_opponent_starting_state(
        env_config: EnvConfig,
        reward_config: RewardConfig,
        seed: u64,
        opponent_starting_state: OpponentStartingState,
    ) -> Self {
        let profile = env_config.opponent;
        let opponent_mind_factory: OpponentMindFactory =
            Arc::new(move || Box::new(ActionBufferMind::new_opponent(profile)));
        Self::new_with_opponent_factory(
            env_config,
            reward_config,
            seed,
            opponent_mind_factory,
            None,
            Some(opponent_starting_state),
        )
    }

    /// Create an environment whose non-training teams use one immutable policy
    /// snapshot. Every pool worker owns a model clone, receives only canonical
    /// local input, and retains no invocation state between cells.
    pub fn new_with_snapshot<B: Backend>(
        env_config: EnvConfig,
        reward_config: RewardConfig,
        seed: u64,
        model: PolicyValueNet<B>,
        device: B::Device,
    ) -> Self
    where
        f32: From<B::FloatElem>,
    {
        assert!(
            policy_memory_bytes(model.recurrent_size())
                .is_some_and(|bytes| bytes <= env_config.rules.max_private_memory_bytes),
            "snapshot recurrent state exceeds this environment's private-memory limit"
        );
        // Burn modules are Send but not Sync. Synchronize only factory-time
        // cloning so workers never share mutable parameter internals.
        let model = Arc::new(Mutex::new(model));
        let mind_model = Arc::clone(&model);
        let batch_device = device.clone();
        let opponent_mind_factory: OpponentMindFactory = Arc::new(move || {
            let model = mind_model
                .lock()
                .expect("snapshot opponent model factory was poisoned")
                .clone();
            Box::new(SnapshotPolicyMind::new(model, device.clone()))
        });
        let opponent_batch_policy_factory: OpponentBatchPolicyFactory = Arc::new(move || {
            let model = model
                .lock()
                .expect("snapshot batch-policy factory was poisoned")
                .clone();
            Box::new(BatchedSnapshotPolicy::new(model, batch_device.clone()))
        });
        Self::new_with_opponent_factory(
            env_config,
            reward_config,
            seed,
            opponent_mind_factory,
            Some(opponent_batch_policy_factory),
            None,
        )
    }

    pub(crate) fn new_with_opponent_factory(
        env_config: EnvConfig,
        reward_config: RewardConfig,
        seed: u64,
        opponent_mind_factory: OpponentMindFactory,
        opponent_batch_policy_factory: Option<OpponentBatchPolicyFactory>,
        opponent_starting_state: Option<OpponentStartingState>,
    ) -> Self {
        let cell_config = CellConfig {
            starting_cells_per_team: env_config.cells_per_team,
            min_energy: env_config.min_energy,
            initial_energy: env_config.initial_energy,
            max_energy: env_config.max_energy,
            min_attack_power: env_config.min_attack_power,
            max_attack_power: env_config.max_attack_power,
            max_energy_for_attack_scaling: env_config.max_energy_for_attack_scaling,
        };

        let mut engine = Engine::new(
            env_config.world_size,
            env_config.world_size,
            env_config.max_episode_len,
            cell_config,
            Some(seed),
            env_config.rules.clone(),
        );
        engine
            .set_starting_cell_layout(env_config.starting_cell_layout)
            .expect("a new RL engine has no starting teams");
        engine
            .set_reference_integrity_mode(IntegrityMode::OnDemand)
            .expect("a new RL engine has no active replay recorder");

        // Training team
        engine
            .add_team_with_minds(TeamId(0), vec![ActionBufferMind::new_training()])
            .unwrap();

        // Opponent team(s)
        for i in 1..env_config.num_teams {
            let minds = vec![opponent_mind_factory()];
            if let Some(starting_state) = opponent_starting_state {
                engine
                    .add_team_with_boxed_minds_and_starting_state(
                        TeamId(i),
                        minds,
                        starting_state.cells_per_team,
                        starting_state.initial_energy,
                    )
                    .unwrap();
            } else {
                engine.add_team_with_boxed_minds(TeamId(i), minds).unwrap();
            }
        }
        configure_episode_resources(&mut engine, &env_config, seed)
            .expect("validated RL resource placement failed");
        engine.initialize_reference_state().unwrap();

        let opponent_batch_policy = opponent_batch_policy_factory
            .as_ref()
            .map(|factory| factory());
        BlobEnv {
            engine,
            reward_config,
            env_config,
            training_team: TeamId(0),
            episode_step: 0,
            prev_cell_energies: HashMap::new(),
            prev_cell_count: HashMap::new(),
            prev_cell_positions: HashMap::new(),
            opponent_mind_factory,
            opponent_batch_policy_factory,
            opponent_batch_policy,
            opponent_starting_state,
            telemetry: None,
            match_explorer: None,
        }
    }

    /// Enable expensive, verified, per-resolution-batch recording for one
    /// explicitly selected evaluation match. Ordinary rollout environments
    /// never call this and retain on-demand hashing with no replay copies.
    pub(crate) fn enable_match_explorer_recording(
        &mut self,
        config: MatchExplorerConfig,
    ) -> Result<(), String> {
        if self.match_explorer.is_some() {
            return Err("match explorer recording is already active".into());
        }
        let recorder = MatchExplorerRecorder::start(&self.engine, config)?;
        self.engine
            .set_reference_integrity_mode(IntegrityMode::Verified)?;
        self.engine.start_reference_replay_recording(128)?;
        self.match_explorer = Some(recorder);
        Ok(())
    }

    pub(crate) fn finish_match_explorer_recording(
        &mut self,
        outcome: impl Into<String>,
        end_reason: impl Into<String>,
    ) -> Result<RecordedExplorerMatch, String> {
        let replay = self
            .engine
            .export_reference_replay_bundle(ReplayBundleLimits::default())?;
        let recorder = self
            .match_explorer
            .take()
            .ok_or("match explorer recording is not active")?;
        if recorder.event_count() != replay.archive().event_count() {
            return Err("presentation trace and canonical replay event counts diverged".into());
        }
        Ok(recorder.finish(replay, outcome, end_reason))
    }

    /// Enable bounded host telemetry. This changes neither canonical state nor
    /// the observation presented to any Mind.
    pub fn enable_telemetry(&mut self, config: TelemetryConfig) {
        if !config.enabled {
            self.telemetry = None;
            return;
        }
        config.validate().expect("invalid telemetry configuration");
        let sides = self
            .engine
            .cells
            .iter()
            .filter_map(|(id, cell)| {
                u64::try_from(id.0).ok().map(|key| {
                    (
                        CellKey(key),
                        if cell.team_id == self.training_team {
                            TelemetrySide::Training
                        } else {
                            TelemetrySide::Opponents
                        },
                    )
                })
            })
            .collect();
        self.telemetry = Some(EnvTelemetryRuntime { config, sides });
    }

    pub fn telemetry_sample(&self) -> Option<EcologySample> {
        let runtime = self.telemetry.as_ref()?;
        let simulation = self
            .engine
            .reference_simulation()
            .expect("deadline reward requires an initialized reference simulation");
        Some(EcologySample::capture(
            simulation,
            &runtime.sides,
            self.episode_step,
            self.env_config.world_size,
            self.env_config.world_size,
        ))
    }

    /// Get observations for all cells on the training team.
    pub fn get_observations(&self) -> Vec<(CellId, Observation)> {
        self.get_policy_observations()
            .into_iter()
            .map(|input| (input.cell_id, input.observation))
            .collect()
    }

    /// Get the anonymous observation and isolated private-memory bytes for all
    /// ready training cells. Cell IDs are host-only routing handles.
    pub fn get_policy_observations(&self) -> Vec<PolicyObservation> {
        let Some(simulation) = self.engine.reference_simulation() else {
            return Vec::new();
        };
        let observations = simulation.observation_batch();
        let mut projected = Vec::new();
        let mut scratch_input = None;
        for cell_id in self.engine.ready_cell_ids().into_iter().filter(|cell_id| {
            self.engine
                .cells
                .get(cell_id)
                .is_some_and(|cell| cell.team_id == self.training_team)
        }) {
            let Ok(actor) = u64::try_from(cell_id.0).map(CellKey) else {
                continue;
            };
            let result = if let Some(input) = scratch_input.as_mut() {
                observations.reference_mind_input_into(actor, PrivateRandom::ZERO, input)
            } else {
                observations
                    .reference_mind_input(actor, PrivateRandom::ZERO)
                    .map(|input| scratch_input = Some(input))
            };
            if result.is_err() {
                continue;
            }
            let Some(input) = scratch_input.as_ref() else {
                continue;
            };
            projected.push(PolicyObservation {
                cell_id,
                observation: Observation::from_reference(input),
                private_memory: input.private_memory.clone(),
            });
        }
        projected
    }

    /// Current authoritative event time in resolver quanta.
    pub fn sim_time_quanta(&self) -> u64 {
        self.engine
            .reference_simulation()
            .map_or(0, |simulation| simulation.now().0)
    }

    /// Converts resolver quanta to the nominal time unit used by RL discounting.
    pub fn elapsed_time_units(&self, started_at: u64) -> f32 {
        let elapsed = self.sim_time_quanta().saturating_sub(started_at);
        let quanta_per_unit = self.engine.reference_simulation().map_or(1, |simulation| {
            simulation.rules().time.arithmetic_quanta_per_unit.max(1)
        });
        (elapsed as f64 / quanta_per_unit as f64) as f32
    }

    pub fn cell_is_alive(&self, cell_id: CellId) -> bool {
        self.engine.cells.contains_key(&cell_id)
    }

    pub fn training_cells_alive(&self) -> usize {
        self.engine
            .cells
            .values()
            .filter(|cell| cell.team_id == self.training_team)
            .count()
    }

    pub fn compiled_ruleset_hash(&self) -> String {
        self.engine
            .reference_simulation()
            .map(|simulation| simulation.compiled_ruleset_hash().to_string())
            .unwrap_or_else(|| "uninitialized".to_string())
    }

    pub(crate) fn initial_ecology_snapshot(&self) -> Result<InitialEcologySnapshot, String> {
        let simulation = self
            .engine
            .reference_simulation()
            .ok_or("reference state has not been initialized")?;
        let host_teams = self
            .engine
            .cells
            .iter()
            .map(|(id, cell)| (id.0, cell.team_id))
            .collect::<HashMap<_, _>>();
        let mut starts_by_team = vec![Vec::new(); self.env_config.num_teams];
        for (key, cell) in simulation.cells() {
            let id = usize::try_from(key.0)
                .map_err(|_| "characterization cell identity does not fit usize")?;
            let team = host_teams
                .get(&id)
                .ok_or("characterization cell has no host-private team")?
                .0;
            starts_by_team
                .get_mut(team)
                .ok_or("characterization cell names an unknown team")?
                .push(cell.position);
        }
        let plants = simulation
            .tiles()
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0).then_some(TileIndex(index))
            })
            .collect();
        let major_food = simulation
            .tiles()
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0 || tile.loose_energy > 0)
                    .then_some(TileIndex(index))
            })
            .collect();
        Ok(InitialEcologySnapshot {
            starts_by_team,
            plants,
            major_food,
        })
    }

    pub(crate) fn reference_simulation_for_diagnostics(
        &self,
    ) -> Result<&ReferenceSimulation, String> {
        self.engine
            .reference_simulation()
            .ok_or_else(|| "reference state has not been initialized".to_string())
    }

    pub fn checkpoint(&self) -> Result<BlobEnvCheckpoint, String> {
        let runtime = self.engine.export_reference_checkpoint()?;
        if runtime.replay.is_some() || runtime.replay_stream.is_some() {
            return Err("RL environment checkpoint unexpectedly contains replay state".into());
        }
        let mut host_cells = self.engine.cells.values().cloned().collect::<Vec<_>>();
        host_cells.sort_unstable_by_key(|cell| cell.id.0);
        Ok(BlobEnvCheckpoint {
            canonical_checkpoint: runtime.canonical.to_bytes(),
            iteration: runtime.iteration,
            host_cells,
            episode_step: self.episode_step,
        })
    }

    pub fn from_checkpoint(
        env_config: EnvConfig,
        reward_config: RewardConfig,
        checkpoint: BlobEnvCheckpoint,
    ) -> Result<Self, String> {
        Self::new(env_config, reward_config, 0).restore_checkpoint(checkpoint)
    }

    pub fn from_checkpoint_with_opponent_starting_state(
        env_config: EnvConfig,
        reward_config: RewardConfig,
        opponent_starting_state: OpponentStartingState,
        checkpoint: BlobEnvCheckpoint,
    ) -> Result<Self, String> {
        Self::new_with_opponent_starting_state(
            env_config,
            reward_config,
            0,
            opponent_starting_state,
        )
        .restore_checkpoint(checkpoint)
    }

    pub fn from_checkpoint_with_snapshot<B: Backend>(
        env_config: EnvConfig,
        reward_config: RewardConfig,
        model: PolicyValueNet<B>,
        device: B::Device,
        checkpoint: BlobEnvCheckpoint,
    ) -> Result<Self, String>
    where
        f32: From<B::FloatElem>,
    {
        Self::new_with_snapshot(env_config, reward_config, 0, model, device)
            .restore_checkpoint(checkpoint)
    }

    fn restore_checkpoint(mut self, checkpoint: BlobEnvCheckpoint) -> Result<Self, String> {
        let env = &mut self;
        env.engine.cells = checkpoint
            .host_cells
            .into_iter()
            .map(|cell| (cell.id, cell))
            .collect();
        let canonical = ReferenceCheckpoint::from_bytes(&checkpoint.canonical_checkpoint)
            .map_err(|error| format!("invalid RL environment checkpoint: {error}"))?;
        env.engine
            .restore_reference_checkpoint(ReferenceRuntimeCheckpoint {
                canonical,
                iteration: checkpoint.iteration,
                replay: None,
                replay_stream: None,
            })?;
        env.episode_step = checkpoint.episode_step;
        env.prev_cell_energies.clear();
        env.prev_cell_count.clear();
        env.prev_cell_positions.clear();
        Ok(self)
    }

    /// Step the environment with the given actions for the training team.
    pub fn step(&mut self, actions: &[(CellId, usize)]) -> StepOutput {
        let actions = actions
            .iter()
            .map(|(cell_id, action)| {
                (
                    *cell_id,
                    PolicyChoice {
                        action: *action,
                        amount: 0,
                        signal: 0,
                        signal_strength: 0,
                    },
                    None,
                )
            })
            .collect::<Vec<_>>();
        self.step_with_policy_memory(&actions)
    }

    pub(crate) fn step_with_policy_memory(
        &mut self,
        actions: &[(CellId, PolicyChoice, Option<Vec<u8>>)],
    ) -> StepOutput {
        // Route training actions outside the mind ABI. Cell IDs remain private
        // engine handles and are never arguments to `Mind::decide`.
        let training_actions: HashMap<CellId, ReferenceMindDecision> = {
            let observations = self
                .engine
                .reference_simulation()
                .expect("RL reference simulation was initialized")
                .observation_batch();
            let mut decoded = HashMap::with_capacity(actions.len());
            let mut scratch_input = None;
            for (cell_id, choice, memory) in actions {
                let Ok(actor) = u64::try_from(cell_id.0).map(CellKey) else {
                    continue;
                };
                let result = if let Some(input) = scratch_input.as_mut() {
                    observations.reference_mind_input_into(actor, PrivateRandom::ZERO, input)
                } else {
                    observations
                        .reference_mind_input(actor, PrivateRandom::ZERO)
                        .map(|input| scratch_input = Some(input))
                };
                if result.is_err() {
                    continue;
                }
                let Some(input) = scratch_input.as_ref() else {
                    continue;
                };
                let mut decision = decode_policy_choice(*choice, input);
                if let Some(memory) = memory {
                    attach_policy_memory(&mut decision, memory.clone());
                }
                decoded.insert(*cell_id, decision);
            }
            decoded
        };
        self.step_with_training_decisions(training_actions)
    }

    /// Step team zero through the same anonymous canonical Mind boundary used
    /// by native opponents. Host cell IDs are retained only long enough to
    /// route each independently prepared decision back to the resolver.
    pub fn step_with_baseline(&mut self, profile: OpponentProfile) -> StepOutput {
        let mut mind = ActionBufferMind::new_opponent(profile);
        self.step_with_reference_mind(&mut mind)
    }

    /// Step team zero through an exact maintained or experimental native Mind.
    /// The Mind receives one independently prepared anonymous input at a time;
    /// persistent behavior must travel only through its returned private-memory
    /// update, exactly as it does for a Wasm guest.
    pub(crate) fn step_with_reference_mind(&mut self, mind: &mut dyn ReferenceMind) -> StepOutput {
        let prepared = self
            .prepare_training_reference_inputs()
            .expect("failed to prepare baseline Mind inputs");
        let decisions = prepared
            .into_iter()
            .map(|(cell_id, input)| (cell_id, mind.decide(&input)))
            .collect();
        self.step_with_training_decisions(decisions)
    }

    /// Prepare independently owned anonymous inputs for every ready training
    /// cell. Calling this consumes exactly the private randomness that a normal
    /// native/Wasm Mind dispatch would receive.
    pub(crate) fn prepare_training_reference_inputs(
        &mut self,
    ) -> Result<Vec<(CellId, ReferenceMindInput)>, String> {
        let ready = self
            .engine
            .ready_cell_ids()
            .into_iter()
            .filter(|cell_id| {
                self.engine
                    .cells
                    .get(cell_id)
                    .is_some_and(|cell| cell.team_id == self.training_team)
            })
            .collect::<Vec<_>>();
        self.engine.prepare_reference_mind_inputs(&ready)
    }

    pub(crate) fn step_with_training_decisions(
        &mut self,
        training_actions: HashMap<CellId, ReferenceMindDecision>,
    ) -> StepOutput {
        // Snapshot state before step for reward attribution.
        self.snapshot_state();

        // Advance the event clock until the training team reaches another
        // decision frontier, the canonical event-time deadline is observed,
        // or a side is exterminated. Checking the deadline after every
        // resolver batch keeps it independent of which Mind occupies team zero.
        // A single RL step may therefore contain several resolver batches in
        // which only opponents finish and choose new actions.
        let mut tick_events = TickEvents::default();
        let mut step_telemetry = self.telemetry.as_ref().map(|_| StepTelemetry::default());
        let mut first_batch = true;
        loop {
            let mut overrides = self.batched_opponent_overrides();
            if first_batch {
                overrides.extend(
                    training_actions
                        .iter()
                        .map(|(cell_id, decision)| (*cell_id, decision.clone())),
                );
            }
            let batch = self
                .engine
                .tick_reference_with_overrides(&overrides, false)
                .unwrap();
            let signal_decayed = self.telemetry.as_ref().map(|_| {
                self.engine
                    .reference_simulation()
                    .expect("reference simulation initialized")
                    .last_resolution_metrics()
                    .signal_energy_decayed
            });
            first_batch = false;
            if let (Some(runtime), Some(telemetry)) =
                (self.telemetry.as_mut(), step_telemetry.as_mut())
            {
                telemetry.observe_commitments(&batch.reference_commitments, &runtime.sides);
                if let Some(report) = batch.reference_batch.as_ref() {
                    telemetry.observe_batch(
                        report,
                        &mut runtime.sides,
                        signal_decayed.expect("telemetry captured signal decay effects"),
                    );
                }
                telemetry.observe_kills(&batch.kills, &runtime.sides);
            }
            if let (Some(recorder), Some(report)) =
                (self.match_explorer.as_mut(), batch.reference_batch.as_ref())
            {
                recorder
                    .observe(report)
                    .expect("match explorer recording diverged from canonical resolution");
            }
            tick_events.kills.extend(batch.kills);
            tick_events.splits.extend(batch.splits);
            tick_events.reference_batch = batch.reference_batch;

            let training_alive = self
                .engine
                .cells
                .values()
                .any(|cell| cell.team_id == self.training_team);
            let opponents_alive = self
                .engine
                .cells
                .values()
                .any(|cell| cell.team_id != self.training_team);
            let deadline_reached = self
                .engine
                .reference_simulation()
                .expect("reference simulation initialized")
                .now()
                .0
                >= self.env_config.victory.sim_time_limit_quanta;
            let training_ready = self.engine.ready_cell_ids().into_iter().any(|cell_id| {
                self.engine
                    .cells
                    .get(&cell_id)
                    .is_some_and(|cell| cell.team_id == self.training_team)
            });
            if !training_alive || !opponents_alive || deadline_reached || training_ready {
                break;
            }
        }
        self.episode_step += 1;

        // Compute rewards using tick events
        let rewards = self.compute_rewards(&tick_events);

        // Check if done
        let training_cells: usize = self
            .engine
            .cells
            .values()
            .filter(|c| c.team_id == self.training_team)
            .count();
        let opponent_cells: usize = self
            .engine
            .cells
            .values()
            .filter(|c| c.team_id != self.training_team)
            .count();
        let exterminated = training_cells == 0 || opponent_cells == 0;
        let sim_time_quanta = self
            .engine
            .reference_simulation()
            .expect("reference simulation initialized")
            .now()
            .0;
        let deadline_reached = sim_time_quanta >= self.env_config.victory.sim_time_limit_quanta;
        let safety_limit_reached = self.episode_step >= self.env_config.max_episode_len;
        let done = exterminated || deadline_reached || safety_limit_reached;

        let end_reason = if !done {
            None
        } else if exterminated {
            Some(EpisodeEndReason::Extermination)
        } else if deadline_reached {
            Some(EpisodeEndReason::SimTimeDeadline)
        } else {
            Some(EpisodeEndReason::DecisionFrontierSafetyLimit)
        };

        let outcome = if done {
            if training_cells == 0 {
                Some(EpisodeOutcome::Loss)
            } else if opponent_cells == 0 {
                Some(EpisodeOutcome::Win)
            } else if deadline_reached {
                Some(EpisodeOutcome::Timeout)
            } else {
                Some(EpisodeOutcome::SafetyAbort)
            }
        } else {
            None
        };

        if let Some(telemetry) = step_telemetry.as_mut() {
            let should_sample = self.telemetry.as_ref().is_some_and(|runtime| {
                done || self
                    .episode_step
                    .is_multiple_of(runtime.config.state_sample_interval_steps)
            });
            if should_sample {
                telemetry.state_sample = self.telemetry_sample();
            }
        }

        // Get new observations
        let policy_observations = self.get_policy_observations();
        let observations = policy_observations
            .iter()
            .map(|input| (input.cell_id, input.observation.clone()))
            .collect();

        StepOutput {
            observations,
            policy_observations,
            rewards,
            done,
            outcome,
            end_reason,
            episode_step: self.episode_step,
            training_cells,
            opponent_cells,
            telemetry: step_telemetry,
        }
    }

    fn batched_opponent_overrides(&mut self) -> HashMap<CellId, ReferenceMindDecision> {
        if self.opponent_batch_policy.is_none() {
            return HashMap::new();
        }
        let ready = self
            .engine
            .ready_cell_ids()
            .into_iter()
            .filter(|cell_id| {
                self.engine
                    .cells
                    .get(cell_id)
                    .is_some_and(|cell| cell.team_id != self.training_team)
            })
            .collect::<Vec<_>>();
        if ready.is_empty() {
            return HashMap::new();
        }
        let prepared = self
            .engine
            .prepare_reference_mind_inputs(&ready)
            .expect("failed to prepare batched snapshot-opponent inputs");
        let (cell_ids, inputs): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        let decisions = self
            .opponent_batch_policy
            .as_mut()
            .expect("checked snapshot batch policy disappeared")
            .decide_batch(&inputs)
            .expect("batched snapshot-opponent inference failed");
        assert_eq!(
            cell_ids.len(),
            decisions.len(),
            "snapshot batch policy returned the wrong decision count"
        );
        cell_ids.into_iter().zip(decisions).collect()
    }

    pub fn snapshot_batch_forward_calls(&self) -> u64 {
        self.opponent_batch_policy
            .as_ref()
            .map_or(0, |policy| policy.forward_calls())
    }

    pub fn snapshot_batch_inferred_rows(&self) -> u64 {
        self.opponent_batch_policy
            .as_ref()
            .map_or(0, |policy| policy.inferred_rows())
    }

    /// Reset the environment for a new episode.
    pub fn reset(&mut self, seed: u64) -> Vec<(CellId, Observation)> {
        let cell_config = CellConfig {
            starting_cells_per_team: self.env_config.cells_per_team,
            min_energy: self.env_config.min_energy,
            initial_energy: self.env_config.initial_energy,
            max_energy: self.env_config.max_energy,
            min_attack_power: self.env_config.min_attack_power,
            max_attack_power: self.env_config.max_attack_power,
            max_energy_for_attack_scaling: self.env_config.max_energy_for_attack_scaling,
        };

        let mut engine = Engine::new(
            self.env_config.world_size,
            self.env_config.world_size,
            self.env_config.max_episode_len,
            cell_config,
            Some(seed),
            self.env_config.rules.clone(),
        );
        engine
            .set_starting_cell_layout(self.env_config.starting_cell_layout)
            .expect("a reset RL engine has no starting teams");
        engine
            .set_reference_integrity_mode(IntegrityMode::OnDemand)
            .expect("a new RL engine has no active replay recorder");

        engine
            .add_team_with_minds(TeamId(0), vec![ActionBufferMind::new_training()])
            .unwrap();
        for i in 1..self.env_config.num_teams {
            let minds = vec![(self.opponent_mind_factory)()];
            if let Some(starting_state) = self.opponent_starting_state {
                engine
                    .add_team_with_boxed_minds_and_starting_state(
                        TeamId(i),
                        minds,
                        starting_state.cells_per_team,
                        starting_state.initial_energy,
                    )
                    .unwrap();
            } else {
                engine.add_team_with_boxed_minds(TeamId(i), minds).unwrap();
            }
        }
        configure_episode_resources(&mut engine, &self.env_config, seed)
            .expect("validated RL resource placement failed");
        engine.initialize_reference_state().unwrap();

        self.engine = engine;
        self.episode_step = 0;
        self.prev_cell_energies.clear();
        self.prev_cell_count.clear();
        self.prev_cell_positions.clear();
        self.opponent_batch_policy = self
            .opponent_batch_policy_factory
            .as_ref()
            .map(|factory| factory());
        if let Some(config) = self
            .telemetry
            .as_ref()
            .map(|runtime| runtime.config.clone())
        {
            self.enable_telemetry(config);
        }

        self.get_observations()
    }

    fn snapshot_state(&mut self) {
        self.prev_cell_energies.clear();
        self.prev_cell_count.clear();
        self.prev_cell_positions.clear();
        let simulation = self
            .engine
            .reference_simulation()
            .expect("reference simulation initialized");
        for (key, state) in simulation.cells() {
            let id = CellId(
                usize::try_from(key.0)
                    .expect("canonical cell key does not fit host dispatch identity"),
            );
            let cell = self
                .engine
                .cells
                .get(&id)
                .expect("canonical cell is missing host-private team ownership");
            self.prev_cell_energies.insert(
                id,
                (
                    state
                        .assimilated_energy
                        .checked_add(state.gut_energy)
                        .expect("canonical stored energy overflowed"),
                    cell.team_id,
                ),
            );
            *self.prev_cell_count.entry(cell.team_id).or_insert(0) += 1;
            if let Some(&coord) = self.engine.inv_coordinate_map.get(&id) {
                self.prev_cell_positions.insert(id, coord);
            }
        }
    }

    fn compute_rewards(&self, events: &TickEvents) -> HashMap<CellId, f32> {
        let mut rewards = HashMap::new();
        let rc = &self.reward_config;

        // Count current cells per team
        let mut current_training_count = 0usize;
        let mut current_opponent_count = 0usize;
        for cell in self.engine.cells.values() {
            if cell.team_id == self.training_team {
                current_training_count += 1;
            } else {
                current_opponent_count += 1;
            }
        }
        let deadline_reward = if current_training_count > 0 && current_opponent_count > 0 {
            self.deadline_reward_outcome()
        } else {
            None
        };

        // A dead cell still owns the transition that led to its death. Emit
        // its terminal reward explicitly instead of distributing the penalty
        // only across unrelated survivors.
        for (cell_id, (_, team_id)) in &self.prev_cell_energies {
            if *team_id == self.training_team && !self.engine.cells.contains_key(cell_id) {
                rewards.insert(*cell_id, rc.cell_died);
            }
        }

        // Per-cell rewards for surviving training cells
        for (cell_id, cell) in &self.engine.cells {
            if cell.team_id != self.training_team {
                continue;
            }

            let mut reward = rc.survive_tick;

            // Reward net food stored by the cell as soon as it enters the gut.
            // Digestion merely transfers gut energy to assimilated energy and
            // therefore cannot misattribute an earlier Consume to a later
            // action interval.
            if let Some(&(previous_stored, _)) = self.prev_cell_energies.get(cell_id) {
                let key = CellKey(
                    u64::try_from(cell_id.0)
                        .expect("host dispatch identity does not fit canonical cell key"),
                );
                let current_stored = self
                    .engine
                    .reference_simulation()
                    .and_then(|simulation| simulation.cell(key))
                    .map(|state| {
                        state
                            .assimilated_energy
                            .checked_add(state.gut_energy)
                            .expect("canonical stored energy overflowed")
                    })
                    .unwrap_or(0);
                if current_stored > previous_stored {
                    reward += (current_stored - previous_stored) as f32 * rc.eat_energy;
                }
            }

            // Proximity reward shaping: reward moving toward food/enemies
            if let Some(&cur_coord) = self.engine.inv_coordinate_map.get(cell_id) {
                if let Some(&prev_coord) = self.prev_cell_positions.get(cell_id) {
                    let radius = rc.proximity_search_radius;

                    // Food proximity shaping
                    if rc.move_toward_food != 0.0 {
                        let prev_food_dist = self.nearest_food_distance(prev_coord, radius);
                        let cur_food_dist = self.nearest_food_distance(cur_coord, radius);
                        if let (Some(prev_d), Some(cur_d)) = (prev_food_dist, cur_food_dist) {
                            let delta = prev_d as f32 - cur_d as f32; // positive = moved closer
                            reward += delta * rc.move_toward_food;
                        } else if prev_food_dist.is_some() && cur_food_dist.is_none() {
                            // Moved away from food (out of radius)
                            reward -= rc.move_toward_food;
                        } else if prev_food_dist.is_none() && cur_food_dist.is_some() {
                            // Moved into range of food
                            reward += rc.move_toward_food;
                        }
                    }

                    // Enemy proximity shaping (only when energy is sufficient)
                    if rc.move_toward_enemy != 0.0 && cell.energy > cell.min_energy * 3 {
                        let prev_enemy_dist = self.nearest_enemy_distance(prev_coord, radius);
                        let cur_enemy_dist = self.nearest_enemy_distance(cur_coord, radius);
                        if let (Some(prev_d), Some(cur_d)) = (prev_enemy_dist, cur_enemy_dist) {
                            let delta = prev_d as f32 - cur_d as f32;
                            reward += delta * rc.move_toward_enemy;
                        }
                    }
                }
            }

            // Team-level terminal rewards
            if (current_opponent_count == 0 && current_training_count > 0)
                || deadline_reward == Some(DeadlineRewardOutcome::Win)
            {
                reward += rc.team_wins;
            } else if deadline_reward == Some(DeadlineRewardOutcome::Loss) {
                reward += rc.team_loses;
            }

            rewards.insert(*cell_id, reward);
        }

        if current_training_count == 0 {
            for (cell_id, (_, team_id)) in &self.prev_cell_energies {
                if *team_id == self.training_team {
                    *rewards.entry(*cell_id).or_default() += rc.team_loses;
                }
            }
        }

        // Attribute only damage the authoritative resolver actually applied to
        // an opposing cell. Requested payload, mitigated damage, overkill, and
        // friendly fire are deliberately excluded.
        if rc.damage_enemy != 0.0 {
            if let Some(report) = &events.reference_batch {
                for outcome in &report.outcomes {
                    let Some(damage) = &outcome.attack_damage else {
                        continue;
                    };
                    if let Some((attacker, applied)) =
                        self.enemy_damage_credit(outcome.actor, damage)
                    {
                        *rewards.entry(attacker).or_default() += applied as f32 * rc.damage_enemy;
                    }
                }
            }
        }

        // Per-cell kill attribution from TickEvents
        for (attacker_id, _victim_id, victim_team) in &events.kills {
            if *victim_team != self.training_team {
                // Our cell killed an enemy — reward the specific attacker
                let belongs_to_training_team = self
                    .engine
                    .cells
                    .get(attacker_id)
                    .is_some_and(|cell| cell.team_id == self.training_team)
                    || self
                        .prev_cell_energies
                        .get(attacker_id)
                        .is_some_and(|(_, team)| *team == self.training_team);
                if belongs_to_training_team {
                    *rewards.entry(*attacker_id).or_default() += rc.kill_enemy;
                }
            }
        }

        // Per-cell split attribution from TickEvents
        for (parent_id, _child_id) in &events.splits {
            // Reward the specific parent for a successful split
            let belongs_to_training_team = self
                .engine
                .cells
                .get(parent_id)
                .is_some_and(|cell| cell.team_id == self.training_team)
                || self
                    .prev_cell_energies
                    .get(parent_id)
                    .is_some_and(|(_, team)| *team == self.training_team);
            if belongs_to_training_team {
                *rewards.entry(*parent_id).or_default() += rc.split_success;
            }
        }

        rewards
    }

    fn enemy_damage_credit(
        &self,
        actor: CellKey,
        damage: &blob_engine::resolution::AttackDamage,
    ) -> Option<(CellId, u64)> {
        let actor_id = CellId(usize::try_from(actor.0).ok()?);
        let victim_id = CellId(usize::try_from(damage.victim.0).ok()?);
        let team = |cell_id: CellId| {
            self.engine
                .cells
                .get(&cell_id)
                .map(|cell| cell.team_id)
                .or_else(|| self.prev_cell_energies.get(&cell_id).map(|(_, team)| *team))
        };
        (damage.applied > 0
            && team(actor_id) == Some(self.training_team)
            && team(victim_id).is_some_and(|victim_team| victim_team != self.training_team))
        .then_some((actor_id, damage.applied))
    }

    fn deadline_reward_outcome(&self) -> Option<DeadlineRewardOutcome> {
        let deadline = &self.reward_config.deadline;
        if deadline.mode == DeadlineRewardMode::None
            || self.sim_time_quanta() < self.env_config.victory.sim_time_limit_quanta
        {
            return None;
        }

        let simulation = self.engine.reference_simulation()?;
        let mut training_score = 0u128;
        let mut opponent_score = 0u128;
        for (key, cell) in simulation.cells() {
            let host_id = CellId(
                usize::try_from(key.0)
                    .expect("canonical cell key does not fit the host dispatch identity"),
            );
            let host = self
                .engine
                .cells
                .get(&host_id)
                .expect("canonical cell is missing host-private team ownership");
            let score = deadline.score([
                cell.core_mass,
                cell.assimilated_energy,
                cell.gut_energy,
                cell.carried_material_mass,
                cell.pending_action
                    .as_ref()
                    .map_or(0, |action| action.payload_escrow),
            ]);
            if host.team_id == self.training_team {
                training_score = training_score
                    .checked_add(score)
                    .expect("validated deadline reward score overflowed");
            } else {
                opponent_score = opponent_score
                    .checked_add(score)
                    .expect("validated deadline reward score overflowed");
            }
        }
        let margin = deadline.minimum_margin_score();
        Some(
            if training_score > opponent_score && training_score - opponent_score >= margin {
                DeadlineRewardOutcome::Win
            } else if opponent_score > training_score && opponent_score - training_score >= margin {
                DeadlineRewardOutcome::Loss
            } else {
                DeadlineRewardOutcome::Draw
            },
        )
    }

    fn chebyshev_distance(
        a: Coordinate,
        b: Coordinate,
        dims: (usize, usize),
        boundary: BoundaryRule,
    ) -> usize {
        let dx = a.x.abs_diff(b.x);
        let dy = a.y.abs_diff(b.y);
        let (dx, dy) = match boundary {
            BoundaryRule::Bounded => (dx, dy),
            BoundaryRule::Wrap => (dx.min(dims.0 - dx), dy.min(dims.1 - dy)),
        };
        dx.max(dy)
    }

    /// Find distance to nearest food source within radius. Returns None if no food found.
    fn nearest_food_distance(&self, coord: Coordinate, radius: usize) -> Option<usize> {
        let dims = self.engine.world.dimensions;
        let mut best = None;
        let boundary = self.env_config.rules.neighborhood.boundary_rule;
        let radius_x = radius.min(dims.0 - 1);
        let radius_y = radius.min(dims.1 - 1);

        for dy_offset in 0..=(2 * radius_y) {
            for dx_offset in 0..=(2 * radius_x) {
                let (x, y) = match boundary {
                    BoundaryRule::Wrap => (
                        (coord.x + dx_offset + dims.0 - radius_x) % dims.0,
                        (coord.y + dy_offset + dims.1 - radius_y) % dims.1,
                    ),
                    BoundaryRule::Bounded => {
                        let x = coord.x as isize + dx_offset as isize - radius_x as isize;
                        let y = coord.y as isize + dy_offset as isize - radius_y as isize;
                        if x < 0 || y < 0 || x >= dims.0 as isize || y >= dims.1 as isize {
                            continue;
                        }
                        (x as usize, y as usize)
                    }
                };
                let idx = y * dims.0 + x;
                if idx < self.engine.world.energy.len() {
                    if let Some(ref _energy_source) = self.engine.world.energy[idx] {
                        let target = Coordinate { x, y };
                        let dist = Self::chebyshev_distance(coord, target, dims, boundary);
                        if dist > 0 {
                            // don't count self-tile as 0 distance
                            best = Some(best.map_or(dist, |b: usize| b.min(dist)));
                        } else {
                            // Food on our tile — distance 0 is the best
                            return Some(0);
                        }
                    }
                }
            }
        }
        best
    }

    /// Find distance to nearest enemy cell within radius. Returns None if no enemy found.
    fn nearest_enemy_distance(&self, coord: Coordinate, radius: usize) -> Option<usize> {
        let dims = self.engine.world.dimensions;
        let mut best = None;
        let boundary = self.env_config.rules.neighborhood.boundary_rule;

        for (enemy_id, enemy_cell) in &self.engine.cells {
            if enemy_cell.team_id == self.training_team {
                continue;
            }
            if let Some(&enemy_coord) = self.engine.inv_coordinate_map.get(enemy_id) {
                let dist = Self::chebyshev_distance(coord, enemy_coord, dims, boundary);
                if dist <= radius {
                    best = Some(best.map_or(dist, |b: usize| b.min(dist)));
                    if dist == 0 {
                        return Some(0);
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EnvConfig, RewardConfig};
    use crate::model::{encode_policy_memory, PolicyValueNet, PolicyValueNetConfig};
    use crate::observation::OBS_DIM;
    use blob_engine::world_gen::ResourceLayout;
    use burn::backend::NdArray;

    fn test_env() -> BlobEnv {
        BlobEnv::new(EnvConfig::default(), RewardConfig::default(), 42)
    }

    #[test]
    fn test_env_creation() {
        let env = test_env();
        assert_eq!(
            env.engine.reference_integrity_mode(),
            IntegrityMode::OnDemand
        );
        assert!(!env.engine.cells.is_empty());
        let obs = env.get_observations();
        assert!(!obs.is_empty());
    }

    #[test]
    fn on_cell_curriculum_places_food_under_every_training_cell() {
        let config = EnvConfig {
            world_size: 8,
            cells_per_team: 3,
            num_scattered_energy: 0,
            num_plants: 0,
            resource_placement: ResourcePlacement::OnTrainingCells,
            opponent: OpponentProfile::Wait,
            ..EnvConfig::default()
        };
        let mut env = BlobEnv::new(config, RewardConfig::default(), 17);
        let inputs = env.prepare_training_reference_inputs().unwrap();
        assert_eq!(inputs.len(), 3);
        assert!(inputs.iter().all(|(_, input)| {
            input.current_tile.plant_energy > 0 && input.current_tile.loose_energy == 0
        }));
    }

    #[test]
    fn symmetric_single_founder_scenario_places_each_cell_on_its_own_plant() {
        let config = EnvConfig {
            world_size: 16,
            cells_per_team: 1,
            starting_cell_layout: blob_engine::engine::StartingCellLayout::Block,
            num_scattered_energy: 0,
            num_plants: 0,
            resource_placement: ResourcePlacement::OnAllCells,
            opponent: OpponentProfile::Wait,
            ..EnvConfig::default()
        };
        let env = BlobEnv::new(config, RewardConfig::default(), 18);
        assert_eq!(env.engine.cells.len(), 2);
        assert_eq!(
            env.engine.starting_cell_layout(),
            blob_engine::engine::StartingCellLayout::Block
        );
        let width = env.engine.world.dimensions.0;
        assert!(env.engine.inv_coordinate_map.values().all(|coordinate| {
            matches!(
                env.engine.world.energy[coordinate.y * width + coordinate.x],
                Some(EnergySource::Plant { .. })
            )
        }));
    }

    #[test]
    fn independent_resource_geometries_are_exact_and_reset_deterministically() {
        let config = EnvConfig {
            world_size: 32,
            cells_per_team: 4,
            starting_cell_layout: blob_engine::engine::StartingCellLayout::Block,
            num_scattered_energy: 48,
            scattered_energy_layout: ResourceLayout::Corridors {
                corridor_count: 3,
                half_width: 1,
            },
            num_plants: 12,
            plant_layout: ResourceLayout::Islands {
                island_count: 3,
                radius: 2,
                minimum_separation: 7,
            },
            opponent: OpponentProfile::Wait,
            ..EnvConfig::default()
        };
        let mut env = BlobEnv::new(config, RewardConfig::default(), 181);
        let first = env.engine.world.energy.clone();
        assert_eq!(
            first
                .iter()
                .filter(|source| matches!(source, Some(EnergySource::Scattered(_))))
                .count(),
            48
        );
        assert_eq!(
            first
                .iter()
                .filter(|source| matches!(source, Some(EnergySource::Plant { .. })))
                .count(),
            12
        );
        env.reset(181);
        assert!(env.engine.world.energy == first);
        env.reset(182);
        assert!(env.engine.world.energy != first);
    }

    #[test]
    fn scale_qualification_profile_initializes_its_full_uniform_ecology() {
        let training = crate::config::TrainingConfig::from_toml_str(include_str!(
            "../config/large_world_1024.toml"
        ))
        .unwrap();
        let env = BlobEnv::new(training.env, training.reward, 1024001);
        assert_eq!(env.engine.world.energy.len(), 1024 * 1024);
        assert_eq!(env.engine.cells.len(), 8192 * 2);
        assert_eq!(
            env.engine
                .world
                .energy
                .iter()
                .filter(|source| matches!(source, Some(EnergySource::Scattered(25))))
                .count(),
            32768
        );
        assert_eq!(
            env.engine
                .world
                .energy
                .iter()
                .filter(|source| matches!(source, Some(EnergySource::Plant { .. })))
                .count(),
            4096
        );
    }

    #[test]
    fn adjacent_curriculum_places_food_only_on_visible_reachable_vacancies() {
        let config = EnvConfig {
            world_size: 8,
            cells_per_team: 3,
            num_scattered_energy: 0,
            num_plants: 0,
            resource_placement: ResourcePlacement::AdjacentToTrainingCells,
            opponent: OpponentProfile::Wait,
            ..EnvConfig::default()
        };
        let mut env = BlobEnv::new(config, RewardConfig::default(), 19);
        let inputs = env.prepare_training_reference_inputs().unwrap();
        assert_eq!(inputs.len(), 3);
        assert!(inputs.iter().all(|(_, input)| {
            input.current_tile.plant_energy == 0
                && input.current_tile.loose_energy == 0
                && input.slots.iter().any(|slot| {
                    ReferenceActionSpace::allows_target(input.action_space.move_targets, slot.slot)
                        && slot.reachable
                        && slot.neighbor.is_none()
                        && slot.plant_energy.unwrap_or(0) > 0
                })
        }));
    }

    #[test]
    fn opt_in_match_explorer_records_every_canonical_batch_and_host_sidecar() {
        use crate::match_explorer::MatchExplorerTeamSpec;
        use std::collections::BTreeMap;

        let mut env = test_env();
        let teams = BTreeMap::from([
            (
                0,
                MatchExplorerTeamSpec {
                    name: "candidate".into(),
                    mind: "test-policy".into(),
                    color: "#65e6a8".into(),
                },
            ),
            (
                1,
                MatchExplorerTeamSpec {
                    name: "baseline".into(),
                    mind: "wait".into(),
                    color: "#ffb55e".into(),
                },
            ),
        ]);
        env.enable_match_explorer_recording(MatchExplorerConfig {
            run_id: "test-match".into(),
            title: "test match".into(),
            ruleset: "test-rules".into(),
            teams,
        })
        .unwrap();
        assert_eq!(
            env.engine.reference_integrity_mode(),
            IntegrityMode::Verified
        );
        let actions = env
            .get_policy_observations()
            .into_iter()
            .map(|observation| (observation.cell_id, 0))
            .collect::<Vec<_>>();
        env.step(&actions);
        let recorded = env
            .finish_match_explorer_recording("test_complete", "test_boundary")
            .unwrap();
        assert!(!recorded.bundle.events.is_empty());
        assert_eq!(recorded.bundle.events.len(), recorded.replay_event_count());
        assert!(recorded
            .bundle
            .events
            .iter()
            .all(|event| event.state_hash.is_some()));
        assert!(recorded.bundle.events.iter().all(|event| {
            event.actions == event.resolved_actions.len()
                && event.resolved_actions.iter().all(|action| {
                    recorded
                        .bundle
                        .presentation
                        .cell_teams
                        .contains_key(&action.actor)
                })
        }));
        assert!(
            recorded.bundle.presentation.cell_teams.len() >= recorded.bundle.initial.cells.len()
        );
        let json = serde_json::to_string(&recorded.bundle).unwrap();
        assert!(!json.contains("\"team\":"));

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("match.json");
        let (_, replay_path) = recorded.publish(&output).unwrap();
        let published: crate::match_explorer::MatchExplorerBundle =
            serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(
            published.run.canonical_replay.as_ref().unwrap().events,
            published.events.len()
        );
        blob_engine::resolution::ReplayBundle::from_bytes(&std::fs::read(replay_path).unwrap())
            .unwrap();
    }

    #[test]
    fn recurrent_policy_memory_is_cell_private_canonical_and_checkpointed() {
        let config = EnvConfig {
            cells_per_team: 2,
            ..EnvConfig::default()
        };
        let mut env = BlobEnv::new(config.clone(), RewardConfig::default(), 73);
        let ready = env.get_policy_observations();
        assert_eq!(ready.len(), 2);
        let first_memory = encode_policy_memory(&[0.25, -0.5]);
        let second_memory = encode_policy_memory(&[0.75, 0.125]);
        let actions = vec![
            (
                ready[0].cell_id,
                PolicyChoice {
                    action: 0,
                    amount: 0,
                    signal: 0,
                    signal_strength: 0,
                },
                Some(first_memory.clone()),
            ),
            (
                ready[1].cell_id,
                PolicyChoice {
                    action: 0,
                    amount: 0,
                    signal: 0,
                    signal_strength: 0,
                },
                Some(second_memory.clone()),
            ),
        ];
        let result = env.step_with_policy_memory(&actions);
        let memories = result
            .policy_observations
            .iter()
            .map(|input| (input.cell_id, input.private_memory.clone()))
            .collect::<HashMap<_, _>>();
        assert_eq!(memories[&ready[0].cell_id], first_memory);
        assert_eq!(memories[&ready[1].cell_id], second_memory);

        let checkpoint = env.checkpoint().unwrap();
        let restored =
            BlobEnv::from_checkpoint(config, RewardConfig::default(), checkpoint).unwrap();
        let restored_memories = restored
            .get_policy_observations()
            .into_iter()
            .map(|input| (input.cell_id, input.private_memory))
            .collect::<HashMap<_, _>>();
        assert_eq!(restored_memories, memories);
    }

    #[test]
    fn compiled_rules_hash_tracks_physics_but_not_rewards() {
        let base_config = EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            num_scattered_energy: 2,
            num_plants: 1,
            ..EnvConfig::default()
        };
        let base = BlobEnv::new(base_config.clone(), RewardConfig::default(), 7);

        let mut changed_rules = base_config.clone();
        changed_rules.rules.bite_capacity += 1;
        let changed_rules = BlobEnv::new(changed_rules, RewardConfig::default(), 7);
        assert_ne!(
            base.compiled_ruleset_hash(),
            changed_rules.compiled_ruleset_hash()
        );

        let mut changed_reward = RewardConfig::default();
        changed_reward.survive_tick += 1.0;
        let changed_reward = BlobEnv::new(base_config, changed_reward, 7);
        assert_eq!(
            base.compiled_ruleset_hash(),
            changed_reward.compiled_ruleset_hash()
        );
    }

    #[test]
    fn reward_distance_uses_the_configured_boundary_rule() {
        let west = Coordinate { x: 0, y: 3 };
        let east = Coordinate { x: 7, y: 3 };
        assert_eq!(
            BlobEnv::chebyshev_distance(west, east, (8, 8), BoundaryRule::Wrap),
            1
        );
        assert_eq!(
            BlobEnv::chebyshev_distance(west, east, (8, 8), BoundaryRule::Bounded),
            7
        );
    }

    #[test]
    fn baseline_profiles_are_typed_deterministic_and_use_only_mind_input() {
        let env = BlobEnv::new(
            EnvConfig {
                world_size: 8,
                cells_per_team: 1,
                starting_cell_layout: blob_engine::engine::StartingCellLayout::PairedContact,
                num_scattered_energy: 0,
                num_plants: 0,
                ..EnvConfig::default()
            },
            RewardConfig::default(),
            42,
        );
        let opponent = env
            .engine
            .cells
            .iter()
            .find_map(|(cell_id, cell)| (cell.team_id != env.training_team).then_some(*cell_id))
            .unwrap();
        let mut input = env
            .engine
            .reference_mind_input_for(
                opponent,
                PrivateRandom::from_bytes([
                    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ]),
            )
            .unwrap();

        let wait = ActionBufferMind::new_opponent(OpponentProfile::Wait).decide(&input);
        assert_eq!(wait.action, ReferenceMindAction::Wait);

        let mut left = ActionBufferMind::new_opponent(OpponentProfile::Random);
        let mut right = ActionBufferMind::new_opponent(OpponentProfile::Random);
        assert_eq!(left.decide(&input), right.decide(&input));

        input.self_state.assimilated_energy = 20;
        input.current_tile.plant_energy = 5;
        let aggressive = ActionBufferMind::new_opponent(OpponentProfile::Aggressive).decide(&input);
        assert!(matches!(
            aggressive.action,
            ReferenceMindAction::Consume { .. }
        ));
        assert_eq!(aggressive.memory_update, ReferenceMemoryUpdate::Retain);

        input.self_state.assimilated_energy = 100;
        let stochastic_actions = (0_u8..32)
            .map(|seed| {
                let mut randomized = input.clone();
                let mut bytes = [0_u8; 32];
                bytes[8] = seed;
                randomized.randomness = PrivateRandom::from_bytes(bytes);
                let left = ActionBufferMind::new_opponent(OpponentProfile::StochasticAggressive)
                    .decide(&randomized);
                let right = ActionBufferMind::new_opponent(OpponentProfile::StochasticAggressive)
                    .decide(&randomized);
                assert_eq!(left, right);
                left.action
            })
            .collect::<Vec<_>>();
        assert!(stochastic_actions
            .iter()
            .any(|action| matches!(action, ReferenceMindAction::Wait)));
        assert!(stochastic_actions
            .iter()
            .any(|action| matches!(action, ReferenceMindAction::Attack { .. })));

        let defensive = ActionBufferMind::new_opponent(OpponentProfile::Defensive).decide(&input);
        assert!(matches!(
            defensive.action,
            ReferenceMindAction::Guard {
                effort: ReferenceEffort::Standard
            }
        ));
    }

    #[test]
    fn asymmetric_opponent_setup_survives_reset_without_mind_privilege() {
        let config = EnvConfig {
            world_size: 7,
            cells_per_team: 1,
            initial_energy: 80,
            starting_cell_layout: blob_engine::engine::StartingCellLayout::OpposedLines,
            num_scattered_energy: 0,
            num_plants: 0,
            ..EnvConfig::default()
        };
        let setup = OpponentStartingState {
            cells_per_team: 3,
            initial_energy: 150,
        };
        let mut env =
            BlobEnv::new_with_opponent_starting_state(config, RewardConfig::default(), 919, setup);
        for seed in [919, 920] {
            if seed != 919 {
                env.reset(seed);
            }
            let training = env
                .engine
                .cells
                .values()
                .filter(|cell| cell.team_id == TeamId(0))
                .collect::<Vec<_>>();
            let opponents = env
                .engine
                .cells
                .values()
                .filter(|cell| cell.team_id == TeamId(1))
                .collect::<Vec<_>>();
            assert_eq!(training.len(), 1);
            assert_eq!(opponents.len(), 3);
            assert!(training.iter().all(|cell| cell.energy == 80));
            assert!(opponents.iter().all(|cell| cell.energy == 150));
        }
    }

    #[test]
    fn opposed_lines_create_a_local_multi_cell_front_without_privileged_neighbors() {
        let visible_occupants = |layout, cells_per_team| {
            let env = BlobEnv::new(
                EnvConfig {
                    world_size: 8,
                    cells_per_team,
                    starting_cell_layout: layout,
                    num_scattered_energy: 0,
                    num_plants: 0,
                    ..EnvConfig::default()
                },
                RewardConfig::default(),
                43,
            );
            env.engine
                .cells
                .iter()
                .filter(|(_, cell)| cell.team_id == env.training_team)
                .map(|(cell_id, _)| {
                    env.engine
                        .reference_mind_input_for(*cell_id, PrivateRandom::ZERO)
                        .unwrap()
                        .slots
                        .iter()
                        .filter(|slot| slot.neighbor.is_some())
                        .count()
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(
            visible_occupants(blob_engine::engine::StartingCellLayout::PairedContact, 1),
            vec![1]
        );
        let mut skirmish =
            visible_occupants(blob_engine::engine::StartingCellLayout::OpposedLines, 4);
        skirmish.sort_unstable();
        assert_eq!(skirmish, vec![3, 3, 5, 5]);
        // ReferenceNeighbor deliberately exposes neither occupant identity nor
        // team; the additional context is purely local geometry and activity.
    }

    #[test]
    fn every_masked_action_is_accepted_by_the_canonical_commit_planner() {
        let env = test_env();
        let simulation = env.engine.reference_simulation().unwrap();
        let actor = simulation.cells().iter().next().unwrap().0;
        let input = simulation
            .reference_mind_input(*actor, PrivateRandom::ZERO)
            .unwrap();
        for (action_id, allowed) in action_mask(&input).into_iter().enumerate() {
            if !allowed {
                continue;
            }
            let decision = decode_action(action_id, &input);
            let mut isolated = simulation.clone();
            let receipt = isolated
                .commit_memory_update_with_signal(
                    *actor,
                    decision.action.into(),
                    decision.signal,
                    decision.memory_update,
                )
                .unwrap();
            assert!(
                receipt.accepted,
                "masked action {action_id} was rejected: {:?}",
                receipt.rejection
            );
        }
    }

    #[test]
    fn test_env_step() {
        let mut env = test_env();
        let obs = env.get_observations();
        let started_at = env.sim_time_quanta();
        // Give DoNothing actions to all training cells
        let actions: Vec<(CellId, usize)> = obs.iter().map(|(id, _)| (*id, 0)).collect();
        let result = env.step(&actions);
        assert!(result.done || !result.observations.is_empty());
        assert!(env.elapsed_time_units(started_at) > 0.0);
        assert!(actions
            .iter()
            .all(|(cell_id, _)| result.rewards.contains_key(cell_id)));
    }

    #[test]
    fn scientific_deadline_and_host_safety_limit_are_distinct() {
        let base = EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            num_scattered_energy: 2,
            num_plants: 2,
            ..EnvConfig::default()
        };
        let mut deadline_config = base.clone();
        deadline_config.max_episode_len = 100;
        // Passive-field events occur before either side's minimum action
        // duration. The scientific deadline must stop on that event frontier
        // rather than waiting for team zero to become ready again.
        deadline_config.rules.diffusion_interval_quanta = 64;
        deadline_config.victory.sim_time_limit_quanta = 64;
        let mut deadline = BlobEnv::new(deadline_config, RewardConfig::default(), 91);
        let actions = deadline
            .get_observations()
            .into_iter()
            .map(|(cell_id, _)| (cell_id, 0))
            .collect::<Vec<_>>();
        let result = deadline.step(&actions);
        assert!(result.done);
        assert_eq!(result.outcome, Some(EpisodeOutcome::Timeout));
        assert_eq!(result.end_reason, Some(EpisodeEndReason::SimTimeDeadline));
        assert_eq!(deadline.sim_time_quanta(), 64);

        let mut safety_config = base;
        safety_config.max_episode_len = 1;
        safety_config.victory.sim_time_limit_quanta = u64::MAX;
        let mut safety = BlobEnv::new(safety_config, RewardConfig::default(), 91);
        let actions = safety
            .get_observations()
            .into_iter()
            .map(|(cell_id, _)| (cell_id, 0))
            .collect::<Vec<_>>();
        let result = safety.step(&actions);
        assert!(result.done);
        assert_eq!(result.outcome, Some(EpisodeOutcome::SafetyAbort));
        assert_eq!(
            result.end_reason,
            Some(EpisodeEndReason::DecisionFrontierSafetyLimit)
        );
    }

    #[test]
    fn telemetry_counts_authoritative_actions_and_conserves_tracked_compartments() {
        let mut env = test_env();
        env.enable_telemetry(TelemetryConfig {
            enabled: true,
            state_sample_interval_steps: 1,
            max_state_samples_per_episode: 8,
            episode_log_stride: 1,
        });
        let initial = env.telemetry_sample().unwrap();
        let actions = env
            .get_observations()
            .into_iter()
            .map(|(cell_id, _)| (cell_id, 0))
            .collect::<Vec<_>>();
        let result = env.step(&actions);
        let telemetry = result.telemetry.unwrap();
        assert_eq!(
            telemetry.training.actions.wait.committed,
            actions.len() as u64
        );
        assert!(telemetry.training.actions.wait.accepted > 0);
        assert!(telemetry.training.actions.wait.completed > 0);
        let sample = telemetry.state_sample.unwrap();
        assert_eq!(sample.episode_step, 1);
        assert_eq!(sample.tracked_mass_energy, initial.tracked_mass_energy);
        assert_eq!(
            sample.tracked_mass_energy,
            sample
                .training_energy
                .total()
                .saturating_add(sample.opponent_energy.total())
                .saturating_add(sample.environment_energy.total())
        );
        assert_eq!(
            sample.training_cells + sample.opponent_cells,
            env.engine.reference_simulation().unwrap().cells().len()
        );
    }

    #[test]
    fn batched_snapshot_inference_matches_isolated_scalar_minds() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        NdArray::<f32>::seed(&device, 414);
        let model: PolicyValueNet<NdArray<f32>> = PolicyValueNetConfig::new().init(&device);
        let config = EnvConfig {
            world_size: 12,
            cells_per_team: 4,
            max_episode_len: 16,
            num_scattered_energy: 8,
            num_plants: 4,
            ..EnvConfig::default()
        };
        let reward = RewardConfig::default();
        let mut batched =
            BlobEnv::new_with_snapshot(config.clone(), reward.clone(), 808, model.clone(), device);

        let scalar_model = Arc::new(Mutex::new(model));
        let scalar_factory: OpponentMindFactory = Arc::new(move || {
            Box::new(SnapshotPolicyMind::new(
                scalar_model.lock().unwrap().clone(),
                device,
            ))
        });
        let mut scalar =
            BlobEnv::new_with_opponent_factory(config, reward, 808, scalar_factory, None, None);

        for _ in 0..4 {
            let actions = batched
                .get_observations()
                .into_iter()
                .map(|(cell_id, _)| (cell_id, 0))
                .collect::<Vec<_>>();
            let left = batched.step(&actions);
            let right = scalar.step(&actions);
            assert_eq!(left.rewards, right.rewards);
            assert_eq!(left.done, right.done);
            assert_eq!(left.outcome, right.outcome);
            assert_eq!(batched.engine.cells, scalar.engine.cells);
            assert_eq!(
                batched.engine.reference_simulation().unwrap().state_hash(),
                scalar.engine.reference_simulation().unwrap().state_hash()
            );
            if left.done {
                break;
            }
        }
        assert!(batched.snapshot_batch_forward_calls() > 0);
        assert!(
            batched.snapshot_batch_inferred_rows() > batched.snapshot_batch_forward_calls(),
            "a multi-cell opponent should amortize at least one forward pass"
        );
        assert_eq!(batched.engine.cells, scalar.engine.cells);
    }

    #[test]
    #[ignore = "manual release-mode snapshot-opponent batching benchmark"]
    fn benchmark_batched_snapshot_opponent_inference() {
        fn run(mut env: BlobEnv, steps: usize) -> (f64, u64, u64) {
            let started = std::time::Instant::now();
            for index in 0..steps {
                let actions = env
                    .get_observations()
                    .into_iter()
                    .map(|(cell_id, _)| (cell_id, 0))
                    .collect::<Vec<_>>();
                let result = env.step(&actions);
                if result.done {
                    env.reset(9000 + index as u64);
                }
            }
            (
                started.elapsed().as_secs_f64(),
                env.snapshot_batch_forward_calls(),
                env.snapshot_batch_inferred_rows(),
            )
        }

        let device = Default::default();
        NdArray::<f32>::seed(&device, 515);
        let model: PolicyValueNet<NdArray<f32>> = PolicyValueNetConfig::new().init(&device);
        let config = EnvConfig {
            world_size: 32,
            cells_per_team: 16,
            max_episode_len: 64,
            num_scattered_energy: 64,
            num_plants: 32,
            ..EnvConfig::default()
        };
        let reward = RewardConfig::default();
        let batched =
            BlobEnv::new_with_snapshot(config.clone(), reward.clone(), 909, model.clone(), device);
        let scalar_model = Arc::new(Mutex::new(model));
        let scalar_factory: OpponentMindFactory = Arc::new(move || {
            Box::new(SnapshotPolicyMind::new(
                scalar_model.lock().unwrap().clone(),
                device,
            ))
        });
        let scalar =
            BlobEnv::new_with_opponent_factory(config, reward, 909, scalar_factory, None, None);

        let steps = 100;
        let (batched_seconds, calls, rows) = run(batched, steps);
        let (scalar_seconds, _, _) = run(scalar, steps);
        eprintln!(
            "snapshot-opponent batching: steps={steps} rows={rows} forwards={calls} scalar={scalar_seconds:.3}s batched={batched_seconds:.3}s speedup={:.2}x",
            scalar_seconds / batched_seconds
        );
    }

    #[test]
    fn test_env_reset() {
        let mut env = test_env();
        env.step(&[]);
        env.step(&[]);
        let obs = env.reset(99);
        assert!(!obs.is_empty());
        assert_eq!(env.episode_step, 0);
    }

    #[test]
    fn trusted_ecology_snapshot_matches_canonical_checkpoint_state() {
        let env = test_env();
        let snapshot = env.initial_ecology_snapshot().unwrap();
        let checkpoint = env.checkpoint().unwrap();
        let canonical = ReferenceCheckpoint::from_bytes(&checkpoint.canonical_checkpoint).unwrap();
        let expected_plants = canonical
            .state()
            .tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0).then_some(TileIndex(index))
            })
            .collect::<Vec<_>>();
        let expected_major_food = canonical
            .state()
            .tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                (tile.plant_capacity > 0 || tile.plant_growth_rate > 0 || tile.loose_energy > 0)
                    .then_some(TileIndex(index))
            })
            .collect::<Vec<_>>();
        assert_eq!(snapshot.plants, expected_plants);
        assert_eq!(snapshot.major_food, expected_major_food);

        let host_teams = checkpoint
            .host_cells
            .iter()
            .map(|cell| (cell.id.0, cell.team_id.0))
            .collect::<HashMap<_, _>>();
        let mut expected_starts = vec![Vec::new(); env.env_config.num_teams];
        for (key, cell) in &canonical.state().cells {
            expected_starts[host_teams[&(key.0 as usize)]].push(cell.position);
        }
        assert_eq!(snapshot.starts_by_team, expected_starts);
    }

    #[test]
    fn rl_environment_checkpoint_restores_exact_continuation_state() {
        let mut env = test_env();
        let actions = env
            .get_observations()
            .into_iter()
            .map(|(cell_id, _)| (cell_id, 0))
            .collect::<Vec<_>>();
        let _ = env.step(&actions);
        let checkpoint = env.checkpoint().unwrap();
        let mut restored = BlobEnv::from_checkpoint(
            env.env_config.clone(),
            env.reward_config.clone(),
            checkpoint,
        )
        .unwrap();
        assert_eq!(restored.episode_step, env.episode_step);
        assert_eq!(restored.engine.cells, env.engine.cells);
        assert_eq!(
            restored.engine.reference_simulation().unwrap().state_hash(),
            env.engine.reference_simulation().unwrap().state_hash()
        );

        let actions = env
            .get_observations()
            .into_iter()
            .map(|(cell_id, _)| (cell_id, 0))
            .collect::<Vec<_>>();
        let left = env.step(&actions);
        let right = restored.step(&actions);
        assert_eq!(left.rewards, right.rewards);
        assert_eq!(left.done, right.done);
        assert_eq!(left.outcome, right.outcome);
        assert_eq!(left.episode_step, right.episode_step);
        assert_eq!(
            restored.engine.reference_simulation().unwrap().state_hash(),
            env.engine.reference_simulation().unwrap().state_hash()
        );
    }

    #[test]
    fn test_env_observations_have_correct_dim() {
        let env = test_env();
        let obs = env.get_observations();
        for (_, o) in &obs {
            assert_eq!(o.data.len(), OBS_DIM);
        }
    }

    #[test]
    fn consume_reward_is_attributed_when_food_enters_the_gut() {
        let config = EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            max_episode_len: 16,
            opponent: OpponentProfile::Wait,
            num_scattered_energy: 0,
            num_plants: 64,
            victory: crate::config::VictoryConfig {
                sim_time_limit_quanta: u64::MAX,
                ..crate::config::VictoryConfig::default()
            },
            ..EnvConfig::default()
        };
        let reward = RewardConfig {
            survive_tick: 0.0,
            eat_energy: 1.0,
            move_toward_food: 0.0,
            move_toward_enemy: 0.0,
            ..RewardConfig::default()
        };
        let mut env = BlobEnv::new(config, reward, 81);
        let observation = env.get_policy_observations().into_iter().next().unwrap();
        let actor = observation.cell_id;
        let key = CellKey(u64::try_from(actor.0).unwrap());
        let stored_before = env
            .engine
            .reference_simulation()
            .unwrap()
            .cell(key)
            .map(|cell| cell.assimilated_energy + cell.gut_energy)
            .unwrap();

        let result = env.step_with_policy_memory(&[(
            actor,
            PolicyChoice {
                action: 4,
                amount: 4,
                signal: 0,
                signal_strength: 0,
            },
            None,
        )]);
        let cell = env
            .engine
            .reference_simulation()
            .unwrap()
            .cell(key)
            .unwrap();
        let stored_after = cell.assimilated_energy + cell.gut_energy;

        assert!(cell.gut_energy > 0);
        assert!(stored_after > stored_before);
        assert_eq!(
            result.rewards[&actor],
            (stored_after - stored_before) as f32
        );
    }

    #[test]
    fn damage_credit_requires_applied_enemy_damage_from_a_training_cell() {
        let env = test_env();
        let training = env
            .engine
            .cells
            .iter()
            .find_map(|(id, cell)| (cell.team_id == env.training_team).then_some(*id))
            .unwrap();
        let opponent = env
            .engine
            .cells
            .iter()
            .find_map(|(id, cell)| (cell.team_id != env.training_team).then_some(*id))
            .unwrap();
        let key = |id: CellId| CellKey(u64::try_from(id.0).unwrap());
        let damage = |victim, applied| blob_engine::resolution::AttackDamage {
            victim: key(victim),
            target_was_guarded: false,
            raw: applied + 7,
            mitigated: 3,
            applied,
            overkill: 4,
        };

        assert_eq!(
            env.enemy_damage_credit(key(training), &damage(opponent, 11)),
            Some((training, 11))
        );
        assert_eq!(
            env.enemy_damage_credit(key(training), &damage(training, 11)),
            None,
            "friendly fire must not earn damage reward"
        );
        assert_eq!(
            env.enemy_damage_credit(key(opponent), &damage(training, 11)),
            None,
            "opponent damage must not enter the training reward"
        );
        assert_eq!(
            env.enemy_damage_credit(key(training), &damage(opponent, 0)),
            None,
            "mitigation and overkill alone must not earn damage reward"
        );
    }

    #[test]
    fn dead_cells_receive_their_own_death_and_team_loss_rewards() {
        let mut env = test_env();
        env.snapshot_state();
        let training_cells = env
            .engine
            .cells
            .iter()
            .filter_map(|(cell_id, cell)| (cell.team_id == env.training_team).then_some(*cell_id))
            .collect::<Vec<_>>();
        for cell_id in &training_cells {
            env.engine.cells.remove(cell_id);
        }
        let rewards = env.compute_rewards(&TickEvents::default());
        let expected = env.reward_config.cell_died + env.reward_config.team_loses;
        for cell_id in training_cells {
            assert_eq!(rewards.get(&cell_id), Some(&expected));
        }
    }

    #[test]
    fn weighted_deadline_reward_does_not_relabel_the_canonical_timeout() {
        let env_config = EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            max_episode_len: 8,
            opponent: OpponentProfile::Wait,
            num_scattered_energy: 0,
            num_plants: 0,
            victory: crate::config::VictoryConfig {
                sim_time_limit_quanta: 1024,
                ..crate::config::VictoryConfig::default()
            },
            ..EnvConfig::default()
        };
        let mut reward = RewardConfig {
            survive_tick: 0.0,
            eat_energy: 0.0,
            kill_enemy: 0.0,
            cell_died: 0.0,
            split_success: 0.0,
            team_wins: 7.0,
            team_loses: -7.0,
            move_toward_food: 0.0,
            move_toward_enemy: 0.0,
            ..RewardConfig::default()
        };
        reward.deadline = crate::config::DeadlineRewardConfig {
            mode: DeadlineRewardMode::WeightedCellEnergy,
            core_basis_points: 10_000,
            assimilated_basis_points: 10_000,
            gut_basis_points: 0,
            carried_material_basis_points: 0,
            payload_escrow_basis_points: 0,
            minimum_margin_mass_energy: 0,
        };
        let mut env = BlobEnv::new(env_config, reward, 144);
        let training_cell = env
            .engine
            .cells
            .iter()
            .find_map(|(id, cell)| (cell.team_id == env.training_team).then_some(*id))
            .unwrap();
        let decision = ReferenceMindDecision {
            action: ReferenceMindAction::Wait,
            signal: Some(blob_interface::reference_mind::ReferenceSignalEmission {
                channel: 0,
                amount: 1,
            }),
            memory_update: ReferenceMemoryUpdate::Retain,
        };
        let result = env.step_with_training_decisions(HashMap::from([(training_cell, decision)]));

        assert!(result.done);
        assert_eq!(result.end_reason, Some(EpisodeEndReason::SimTimeDeadline));
        assert_eq!(result.outcome, Some(EpisodeOutcome::Timeout));
        assert_eq!(result.rewards.get(&training_cell), Some(&-7.0));
        assert_eq!(
            env.deadline_reward_outcome(),
            Some(DeadlineRewardOutcome::Loss)
        );
    }

    #[test]
    #[ignore = "manual large-frontier RL observation throughput diagnostic"]
    fn benchmark_batched_rl_observation_projection() {
        let config = EnvConfig {
            world_size: 128,
            cells_per_team: 5_000,
            num_scattered_energy: 0,
            num_plants: 0,
            ..EnvConfig::default()
        };
        let env = BlobEnv::new(config, RewardConfig::default(), 42);
        let ready: std::collections::HashSet<CellId> =
            env.engine.ready_cell_ids().into_iter().collect();

        let old_started = std::time::Instant::now();
        let mut old = env
            .engine
            .cells
            .iter()
            .filter(|(_, cell)| cell.team_id == env.training_team)
            .filter(|(cell_id, _)| ready.contains(cell_id))
            .filter_map(|(cell_id, _)| {
                let input = env
                    .engine
                    .reference_mind_input_for(*cell_id, PrivateRandom::ZERO)
                    .ok()?;
                Some((*cell_id, Observation::from_reference(&input)))
            })
            .collect::<Vec<_>>();
        let old_elapsed = old_started.elapsed();

        let new_started = std::time::Instant::now();
        let mut new = env.get_observations();
        let new_elapsed = new_started.elapsed();
        old.sort_by_key(|(cell_id, _)| cell_id.0);
        new.sort_by_key(|(cell_id, _)| cell_id.0);
        assert_eq!(old.len(), new.len());
        for ((old_id, old), (new_id, new)) in old.iter().zip(&new) {
            assert_eq!(old_id, new_id);
            assert_eq!(old.data, new.data);
            assert_eq!(old.action_mask, new.action_mask);
        }
        println!(
            "{} RL observations: former {:.3} ms, batched {:.3} ms, {:.2}x speedup",
            new.len(),
            old_elapsed.as_secs_f64() * 1_000.0,
            new_elapsed.as_secs_f64() * 1_000.0,
            old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64(),
        );
    }
}
