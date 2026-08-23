//! BlobEnv — wraps blob_engine::Engine as an RL environment.

use std::collections::HashMap;

use blob_engine::engine::{CellConfig, Engine, TickEvents};
use blob_engine::resolution::{CellKey, IntegrityMode, ReferenceRuleset};
use blob_engine::world_gen;
use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{
    ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate, ReferenceMind,
    ReferenceMindAction, ReferenceMindDecision, ReferenceMindInput,
};
use blob_interface::types::{CellId, Coordinate, TeamId};

use crate::action::{action_mask, decode_action};
use crate::config::{EnvConfig, RewardConfig};
use crate::observation::Observation;

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
    /// Previous state snapshot for reward computation (energy, team_id)
    prev_cell_energies: HashMap<CellId, (u32, TeamId)>,
    prev_cell_count: HashMap<TeamId, usize>,
    /// Previous cell positions for proximity reward shaping
    prev_cell_positions: HashMap<CellId, Coordinate>,
}

/// Opponent/fallback mind. RL actions bypass the mind ABI through engine-side
/// action overrides keyed by private `CellId` values.
pub struct ActionBufferMind {
    fallback: FallbackBehavior,
}

#[derive(Clone)]
enum FallbackBehavior {
    DoNothing,
    Random,
    Aggressive,
}

impl ActionBufferMind {
    pub fn new_training() -> Self {
        ActionBufferMind {
            fallback: FallbackBehavior::DoNothing,
        }
    }

    pub fn new_opponent(behavior: &str) -> Self {
        let fallback = match behavior {
            "random" => FallbackBehavior::Random,
            "aggressive" => FallbackBehavior::Aggressive,
            _ => FallbackBehavior::DoNothing,
        };
        ActionBufferMind { fallback }
    }
}

impl ReferenceMind for ActionBufferMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let action = match &self.fallback {
            FallbackBehavior::DoNothing => ReferenceMindAction::Wait,
            FallbackBehavior::Random => {
                let mask = action_mask(input);
                let valid: Vec<_> = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(index, allowed)| allowed.then_some(index))
                    .collect();
                let selected = valid[input.randomness.sample_u64(0) as usize % valid.len()];
                decode_action(selected, input).action
            }
            FallbackBehavior::Aggressive => {
                if input.self_state.assimilated_energy < 40
                    && input.action_space.consume_enabled
                    && input.action_space.max_consume_amount > 0
                    && input.current_tile.plant_energy + input.current_tile.loose_energy > 0
                {
                    ReferenceMindAction::Consume {
                        amount: input.action_space.max_consume_amount,
                    }
                } else if let Some(target) = input.slots.iter().find(|slot| {
                    slot.neighbor.is_some()
                        && slot.reachable
                        && ReferenceActionSpace::allows_target(
                            input.action_space.attack_targets,
                            slot.slot,
                        )
                }) {
                    let payload = input
                        .self_state
                        .assimilated_energy
                        .saturating_sub(input.action_space.minimum_survival_energy)
                        / 8;
                    if payload > 0 {
                        ReferenceMindAction::Attack {
                            target_slot: target.slot,
                            effort: ReferenceEffort::Standard,
                            payload,
                        }
                    } else {
                        ReferenceMindAction::Wait
                    }
                } else if let Some(target) = input.slots.iter().max_by_key(|slot| {
                    let energy = slot.plant_energy.unwrap_or(0)
                        + slot.loose_energy.unwrap_or(0)
                        + if slot.plant_growth_rate.unwrap_or(0) > 0 {
                            slot.diffuse_energy.unwrap_or(0)
                        } else {
                            0
                        };
                    let allowed = slot.reachable
                        && slot.neighbor.is_none()
                        && ReferenceActionSpace::allows_target(
                            input.action_space.move_targets,
                            slot.slot,
                        );
                    (allowed, energy)
                }) {
                    if target.reachable
                        && target.neighbor.is_none()
                        && ReferenceActionSpace::allows_target(
                            input.action_space.move_targets,
                            target.slot,
                        )
                    {
                        ReferenceMindAction::Move {
                            target_slot: target.slot,
                            effort: ReferenceEffort::Standard,
                        }
                    } else {
                        ReferenceMindAction::Wait
                    }
                } else {
                    ReferenceMindAction::Wait
                }
            }
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

/// How an episode ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeOutcome {
    Win,     // all opponents eliminated
    Loss,    // all training cells died
    Timeout, // max_episode_len reached
}

/// Output from a single environment step.
#[derive(Debug)]
pub struct StepOutput {
    /// Observations for each alive cell on the training team
    pub observations: Vec<(CellId, Observation)>,
    /// Per-cell rewards
    pub rewards: HashMap<CellId, f32>,
    /// Whether the episode is done
    pub done: bool,
    /// If done, how the episode ended
    pub outcome: Option<EpisodeOutcome>,
    /// Current episode step count
    pub episode_step: u64,
    /// Training cells alive at end of step
    pub training_cells: usize,
    /// Opponent cells alive at end of step
    pub opponent_cells: usize,
}

impl BlobEnv {
    /// Create a new BlobEnv.
    pub fn new(env_config: EnvConfig, reward_config: RewardConfig, seed: u64) -> Self {
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
            ReferenceRuleset::default(),
        );
        engine
            .set_reference_integrity_mode(IntegrityMode::OnDemand)
            .expect("a new RL engine has no active replay recorder");

        // Add energy to world using config params
        engine.world.energy = world_gen::scatter_energy(
            env_config.world_size,
            env_config.world_size,
            env_config.num_scattered_energy,
            env_config.scattered_energy_amount,
            env_config.num_plants,
            env_config.plant_rate,
            env_config.plant_max_energy,
            seed,
        );

        // Training team
        engine
            .add_team_with_minds(TeamId(0), vec![ActionBufferMind::new_training()])
            .unwrap();

        // Opponent team(s)
        for i in 1..env_config.num_teams {
            let opponent_mind = ActionBufferMind::new_opponent("aggressive");
            engine
                .add_team_with_minds(TeamId(i), vec![opponent_mind])
                .unwrap();
        }
        engine.initialize_reference_state().unwrap();

        BlobEnv {
            engine,
            reward_config,
            env_config,
            training_team: TeamId(0),
            episode_step: 0,
            prev_cell_energies: HashMap::new(),
            prev_cell_count: HashMap::new(),
            prev_cell_positions: HashMap::new(),
        }
    }

    /// Get observations for all cells on the training team.
    pub fn get_observations(&self) -> Vec<(CellId, Observation)> {
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
                observations
                    .reference_mind_input_into(actor, PrivateRandom::ZERO, input)
                    .map(|()| ())
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
            projected.push((cell_id, Observation::from_reference(input)));
        }
        projected
    }

    /// Step the environment with the given actions for the training team.
    pub fn step(&mut self, actions: &[(CellId, usize)]) -> StepOutput {
        // Snapshot state before step
        self.snapshot_state();

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
            for (cell_id, action_id) in actions {
                let Ok(actor) = u64::try_from(cell_id.0).map(CellKey) else {
                    continue;
                };
                let result = if let Some(input) = scratch_input.as_mut() {
                    observations
                        .reference_mind_input_into(actor, PrivateRandom::ZERO, input)
                        .map(|()| ())
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
                decoded.insert(*cell_id, decode_action(*action_id, input));
            }
            decoded
        };

        // Advance the event clock until the training team reaches another
        // decision frontier (or the episode terminates). A single RL step may
        // therefore contain several resolver batches in which only opponents
        // finish and choose new actions.
        let mut tick_events = TickEvents::default();
        let mut first_batch = true;
        loop {
            let empty_actions = HashMap::new();
            let batch = self
                .engine
                .tick_reference_with_overrides(
                    if first_batch {
                        &training_actions
                    } else {
                        &empty_actions
                    },
                    false,
                )
                .unwrap();
            first_batch = false;
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
            let training_ready = self.engine.ready_cell_ids().into_iter().any(|cell_id| {
                self.engine
                    .cells
                    .get(&cell_id)
                    .is_some_and(|cell| cell.team_id == self.training_team)
            });
            if !training_alive || !opponents_alive || training_ready {
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
        let done = training_cells == 0
            || opponent_cells == 0
            || self.episode_step >= self.env_config.max_episode_len;

        let outcome = if done {
            if training_cells == 0 {
                Some(EpisodeOutcome::Loss)
            } else if opponent_cells == 0 {
                Some(EpisodeOutcome::Win)
            } else {
                Some(EpisodeOutcome::Timeout)
            }
        } else {
            None
        };

        // Get new observations
        let observations = self.get_observations();

        StepOutput {
            observations,
            rewards,
            done,
            outcome,
            episode_step: self.episode_step,
            training_cells,
            opponent_cells,
        }
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
            ReferenceRuleset::default(),
        );
        engine
            .set_reference_integrity_mode(IntegrityMode::OnDemand)
            .expect("a new RL engine has no active replay recorder");

        engine.world.energy = world_gen::scatter_energy(
            self.env_config.world_size,
            self.env_config.world_size,
            self.env_config.num_scattered_energy,
            self.env_config.scattered_energy_amount,
            self.env_config.num_plants,
            self.env_config.plant_rate,
            self.env_config.plant_max_energy,
            seed,
        );

        engine
            .add_team_with_minds(TeamId(0), vec![ActionBufferMind::new_training()])
            .unwrap();
        for i in 1..self.env_config.num_teams {
            engine
                .add_team_with_minds(
                    TeamId(i),
                    vec![ActionBufferMind::new_opponent("aggressive")],
                )
                .unwrap();
        }
        engine.initialize_reference_state().unwrap();

        self.engine = engine;
        self.episode_step = 0;
        self.prev_cell_energies.clear();
        self.prev_cell_count.clear();
        self.prev_cell_positions.clear();

        self.get_observations()
    }

    fn snapshot_state(&mut self) {
        self.prev_cell_energies.clear();
        self.prev_cell_count.clear();
        self.prev_cell_positions.clear();
        for (id, cell) in &self.engine.cells {
            self.prev_cell_energies
                .insert(*id, (cell.energy, cell.team_id));
            *self.prev_cell_count.entry(cell.team_id).or_insert(0) += 1;
            if let Some(&coord) = self.engine.inv_coordinate_map.get(id) {
                self.prev_cell_positions.insert(*id, coord);
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

        // Count own deaths (training cells that died this tick)
        let mut own_deaths = 0u32;
        for (_cell_id, (_, team_id)) in &self.prev_cell_energies {
            if *team_id == self.training_team && !self.engine.cells.contains_key(_cell_id) {
                own_deaths += 1;
            }
        }

        // Per-cell rewards for surviving training cells
        for (cell_id, cell) in &self.engine.cells {
            if cell.team_id != self.training_team {
                continue;
            }

            let mut reward = rc.survive_tick;

            // Energy gain reward
            if let Some(&(prev_energy, _)) = self.prev_cell_energies.get(cell_id) {
                if cell.energy > prev_energy {
                    reward += (cell.energy - prev_energy) as f32 * rc.eat_energy;
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

            // Death penalty distributed across survivors
            if own_deaths > 0 && current_training_count > 0 {
                reward += (own_deaths as f32 * rc.cell_died) / current_training_count as f32;
            }

            // Team-level terminal rewards
            if current_opponent_count == 0 && current_training_count > 0 {
                reward += rc.team_wins;
            }

            rewards.insert(*cell_id, reward);
        }

        // Per-cell kill attribution from TickEvents
        for (attacker_id, _victim_id, victim_team) in &events.kills {
            if *victim_team != self.training_team {
                // Our cell killed an enemy — reward the specific attacker
                if let Some(reward) = rewards.get_mut(attacker_id) {
                    *reward += rc.kill_enemy;
                }
            }
        }

        // Per-cell split attribution from TickEvents
        for (parent_id, _child_id) in &events.splits {
            // Reward the specific parent for a successful split
            if let Some(reward) = rewards.get_mut(parent_id) {
                *reward += rc.split_success;
            }
        }

        rewards
    }

    /// Toroidal Chebyshev distance (max of wrapped dx, dy).
    fn toroidal_chebyshev(a: Coordinate, b: Coordinate, dims: (usize, usize)) -> usize {
        let dx = {
            let d = a.x.abs_diff(b.x);
            d.min(dims.0 - d)
        };
        let dy = {
            let d = a.y.abs_diff(b.y);
            d.min(dims.1 - d)
        };
        dx.max(dy)
    }

    /// Find distance to nearest food source within radius. Returns None if no food found.
    fn nearest_food_distance(&self, coord: Coordinate, radius: usize) -> Option<usize> {
        let dims = self.engine.world.dimensions;
        let mut best = None;

        for dy_offset in 0..=(2 * radius) {
            for dx_offset in 0..=(2 * radius) {
                let x = (coord.x + dx_offset + dims.0 - radius) % dims.0;
                let y = (coord.y + dy_offset + dims.1 - radius) % dims.1;
                let idx = y * dims.0 + x;
                if idx < self.engine.world.energy.len() {
                    if let Some(ref _energy_source) = self.engine.world.energy[idx] {
                        let target = Coordinate { x, y };
                        let dist = Self::toroidal_chebyshev(coord, target, dims);
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

        for (enemy_id, enemy_cell) in &self.engine.cells {
            if enemy_cell.team_id == self.training_team {
                continue;
            }
            if let Some(&enemy_coord) = self.engine.inv_coordinate_map.get(enemy_id) {
                let dist = Self::toroidal_chebyshev(coord, enemy_coord, dims);
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
    use crate::observation::OBS_DIM;

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
    fn test_env_step() {
        let mut env = test_env();
        let obs = env.get_observations();
        // Give DoNothing actions to all training cells
        let actions: Vec<(CellId, usize)> = obs.iter().map(|(id, _)| (*id, 0)).collect();
        let result = env.step(&actions);
        assert!(result.done || !result.observations.is_empty());
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
    fn test_env_observations_have_correct_dim() {
        let env = test_env();
        let obs = env.get_observations();
        for (_, o) in &obs {
            assert_eq!(o.data.len(), OBS_DIM);
        }
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
