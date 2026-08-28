//! Bounded host-side ecology and action telemetry for RL training.
//!
//! These records observe authoritative resolver reports and canonical state.
//! They are never inputs to Minds and never participate in physics or replay
//! hashes.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::resolution::{
    ActionKind, ActionRequest, BatchReport, CellKey, LocalSlot, OutcomeStatus, ReferenceSimulation,
    ReplayCommitment,
};
use blob_interface::reference_mind::REFERENCE_SIGNAL_CHANNELS;
use serde::{Deserialize, Serialize};

pub const TELEMETRY_SCHEMA_VERSION: u32 = 8;
static TELEMETRY_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct TelemetryConfig {
    pub enabled: bool,
    /// RL environment steps between full canonical cell/tile scans. Action
    /// counters remain exact at every resolver batch.
    pub state_sample_interval_steps: u64,
    /// Hard bound on retained state samples in one episode. A terminal sample
    /// is always retained by replacing the last periodic sample if necessary.
    pub max_state_samples_per_episode: usize,
    /// Write one detailed episode JSON record for every N completed episodes
    /// in each environment. Zero disables the detailed JSONL stream while
    /// preserving the compact all-episode summary.
    pub episode_log_stride: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            state_sample_interval_steps: 32,
            max_state_samples_per_episode: 64,
            episode_log_stride: 16,
        }
    }
}

impl TelemetryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled
            && (self.state_sample_interval_steps == 0 || self.max_state_samples_per_episode < 2)
        {
            return Err(
                "enabled telemetry requires a positive sample interval and at least two samples per episode"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn should_log_episode(&self, episode_id: u64) -> bool {
        self.episode_log_stride > 0 && episode_id.is_multiple_of(self.episode_log_stride)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TelemetrySide {
    Training,
    Opponents,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ActionTelemetry {
    pub committed: u64,
    pub accepted: u64,
    pub rejected_at_commit: u64,
    pub completed: u64,
    pub succeeded: u64,
    pub rejected: u64,
    pub frustrated: u64,
    pub contested: u64,
    pub interrupted: u64,
    pub effort_spent: u128,
    pub payload: u128,
    /// Exact environmental energy extracted by completed consume actions.
    /// Unlike `payload`, this excludes requested-but-unavailable amounts.
    pub consumed_energy: u128,
}

impl ActionTelemetry {
    fn observe_commitment(&mut self, commitment: &ReplayCommitment) {
        self.committed = self.committed.saturating_add(1);
        if commitment.receipt.accepted {
            self.accepted = self.accepted.saturating_add(1);
        } else {
            self.rejected_at_commit = self.rejected_at_commit.saturating_add(1);
        }
    }

    fn observe_outcome(&mut self, outcome: &blob_engine::resolution::ActionOutcome) {
        self.completed = self.completed.saturating_add(1);
        match outcome.status {
            OutcomeStatus::Success => self.succeeded = self.succeeded.saturating_add(1),
            OutcomeStatus::Rejected(_) => self.rejected = self.rejected.saturating_add(1),
            OutcomeStatus::Frustrated => self.frustrated = self.frustrated.saturating_add(1),
            OutcomeStatus::Contested => self.contested = self.contested.saturating_add(1),
            OutcomeStatus::Interrupted => self.interrupted = self.interrupted.saturating_add(1),
        }
        self.effort_spent = self
            .effort_spent
            .saturating_add(u128::from(outcome.effort_spent));
        self.payload = self.payload.saturating_add(u128::from(outcome.payload));
        self.consumed_energy = self
            .consumed_energy
            .saturating_add(u128::from(outcome.consumed_energy));
    }

    fn merge(&mut self, other: &Self) {
        self.committed = self.committed.saturating_add(other.committed);
        self.accepted = self.accepted.saturating_add(other.accepted);
        self.rejected_at_commit = self
            .rejected_at_commit
            .saturating_add(other.rejected_at_commit);
        self.completed = self.completed.saturating_add(other.completed);
        self.succeeded = self.succeeded.saturating_add(other.succeeded);
        self.rejected = self.rejected.saturating_add(other.rejected);
        self.frustrated = self.frustrated.saturating_add(other.frustrated);
        self.contested = self.contested.saturating_add(other.contested);
        self.interrupted = self.interrupted.saturating_add(other.interrupted);
        self.effort_spent = self.effort_spent.saturating_add(other.effort_spent);
        self.payload = self.payload.saturating_add(other.payload);
        self.consumed_energy = self.consumed_energy.saturating_add(other.consumed_energy);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ActionFamilyTelemetry {
    pub wait: ActionTelemetry,
    pub movement: ActionTelemetry,
    pub attack: ActionTelemetry,
    pub guard: ActionTelemetry,
    pub consume: ActionTelemetry,
    pub split: ActionTelemetry,
    pub regurgitate: ActionTelemetry,
    pub signal: ActionTelemetry,
    pub excavate: ActionTelemetry,
    pub deposit_terrain: ActionTelemetry,
}

impl ActionFamilyTelemetry {
    fn action_mut(&mut self, action: ActionKind) -> &mut ActionTelemetry {
        match action {
            ActionKind::Wait => &mut self.wait,
            ActionKind::Move => &mut self.movement,
            ActionKind::Attack => &mut self.attack,
            ActionKind::Guard => &mut self.guard,
            ActionKind::Consume => &mut self.consume,
            ActionKind::Split => &mut self.split,
            ActionKind::Regurgitate => &mut self.regurgitate,
            ActionKind::Signal => &mut self.signal,
            ActionKind::Excavate => &mut self.excavate,
            ActionKind::DepositTerrain => &mut self.deposit_terrain,
        }
    }

    fn merge(&mut self, other: &Self) {
        self.wait.merge(&other.wait);
        self.movement.merge(&other.movement);
        self.attack.merge(&other.attack);
        self.guard.merge(&other.guard);
        self.consume.merge(&other.consume);
        self.split.merge(&other.split);
        self.regurgitate.merge(&other.regurgitate);
        self.signal.merge(&other.signal);
        self.excavate.merge(&other.excavate);
        self.deposit_terrain.merge(&other.deposit_terrain);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalEmissionTelemetry {
    /// Accepted decisions that deposited at least one signal channel.
    pub emitted_decisions: u64,
    pub explicit_signal_actions: u64,
    pub sidecar_emissions: u64,
    /// Number of accepted deposits touching each anonymous channel.
    pub channel_deposits: [u64; REFERENCE_SIGNAL_CHANNELS],
    /// Conserved energy deposited into each anonymous channel.
    pub channel_energy: [u128; REFERENCE_SIGNAL_CHANNELS],
    /// Exact nonzero four-bit channel pattern selected by each deposit.
    pub pattern_counts: [u64; 1 << REFERENCE_SIGNAL_CHANNELS],
}

impl SignalEmissionTelemetry {
    fn observe_commitment(&mut self, commitment: &ReplayCommitment) {
        if !commitment.receipt.accepted {
            return;
        }
        let (amounts, explicit) = match (&commitment.request, commitment.signal) {
            (ActionRequest::Signal { amounts }, None) => (*amounts, true),
            (_, Some(signal)) if usize::from(signal.channel) < REFERENCE_SIGNAL_CHANNELS => {
                let mut amounts = [0_u64; REFERENCE_SIGNAL_CHANNELS];
                amounts[usize::from(signal.channel)] = signal.amount;
                (amounts, false)
            }
            _ => return,
        };
        let pattern = amounts
            .iter()
            .enumerate()
            .fold(0_usize, |pattern, (channel, amount)| {
                pattern | (usize::from(*amount > 0) << channel)
            });
        if pattern == 0 {
            return;
        }
        self.emitted_decisions = self.emitted_decisions.saturating_add(1);
        if explicit {
            self.explicit_signal_actions = self.explicit_signal_actions.saturating_add(1);
        } else {
            self.sidecar_emissions = self.sidecar_emissions.saturating_add(1);
        }
        self.pattern_counts[pattern] = self.pattern_counts[pattern].saturating_add(1);
        for (channel, amount) in amounts.into_iter().enumerate() {
            if amount == 0 {
                continue;
            }
            self.channel_deposits[channel] = self.channel_deposits[channel].saturating_add(1);
            self.channel_energy[channel] =
                self.channel_energy[channel].saturating_add(u128::from(amount));
        }
    }

    pub fn total_energy(&self) -> u128 {
        self.channel_energy
            .into_iter()
            .fold(0_u128, u128::saturating_add)
    }

    fn merge(&mut self, other: &Self) {
        self.emitted_decisions = self
            .emitted_decisions
            .saturating_add(other.emitted_decisions);
        self.explicit_signal_actions = self
            .explicit_signal_actions
            .saturating_add(other.explicit_signal_actions);
        self.sidecar_emissions = self
            .sidecar_emissions
            .saturating_add(other.sidecar_emissions);
        for channel in 0..REFERENCE_SIGNAL_CHANNELS {
            self.channel_deposits[channel] =
                self.channel_deposits[channel].saturating_add(other.channel_deposits[channel]);
            self.channel_energy[channel] =
                self.channel_energy[channel].saturating_add(other.channel_energy[channel]);
        }
        for (pattern, count) in other.pattern_counts.iter().enumerate() {
            self.pattern_counts[pattern] = self.pattern_counts[pattern].saturating_add(*count);
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalFieldTelemetry {
    /// Signal energy converted to diffuse energy by passive decay.
    pub decayed_energy: [u128; REFERENCE_SIGNAL_CHANNELS],
    /// Signal energy converted to diffuse energy by successful terrain edits.
    pub terrain_erased_energy: [u128; REFERENCE_SIGNAL_CHANNELS],
    pub terrain_erasure_events: u64,
}

impl SignalFieldTelemetry {
    pub fn total_decayed_energy(&self) -> u128 {
        self.decayed_energy
            .into_iter()
            .fold(0_u128, u128::saturating_add)
    }

    pub fn total_terrain_erased_energy(&self) -> u128 {
        self.terrain_erased_energy
            .into_iter()
            .fold(0_u128, u128::saturating_add)
    }

    fn merge(&mut self, other: &Self) {
        for channel in 0..REFERENCE_SIGNAL_CHANNELS {
            self.decayed_energy[channel] =
                self.decayed_energy[channel].saturating_add(other.decayed_energy[channel]);
            self.terrain_erased_energy[channel] = self.terrain_erased_energy[channel]
                .saturating_add(other.terrain_erased_energy[channel]);
        }
        self.terrain_erasure_events = self
            .terrain_erasure_events
            .saturating_add(other.terrain_erasure_events);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DamageTelemetry {
    pub raw_dealt: u128,
    pub mitigated_by_target_guard: u128,
    pub applied_dealt: u128,
    pub overkill_dealt: u128,
    pub raw_received: u128,
    pub mitigated_by_own_guard: u128,
    pub applied_received: u128,
    pub overkill_received: u128,
}

impl DamageTelemetry {
    fn observe_dealt(&mut self, damage: &blob_engine::resolution::AttackDamage) {
        self.raw_dealt = self.raw_dealt.saturating_add(u128::from(damage.raw));
        self.mitigated_by_target_guard = self
            .mitigated_by_target_guard
            .saturating_add(u128::from(damage.mitigated));
        self.applied_dealt = self
            .applied_dealt
            .saturating_add(u128::from(damage.applied));
        self.overkill_dealt = self
            .overkill_dealt
            .saturating_add(u128::from(damage.overkill));
    }

    fn observe_received(&mut self, damage: &blob_engine::resolution::AttackDamage) {
        self.raw_received = self.raw_received.saturating_add(u128::from(damage.raw));
        self.mitigated_by_own_guard = self
            .mitigated_by_own_guard
            .saturating_add(u128::from(damage.mitigated));
        self.applied_received = self
            .applied_received
            .saturating_add(u128::from(damage.applied));
        self.overkill_received = self
            .overkill_received
            .saturating_add(u128::from(damage.overkill));
    }

    fn merge(&mut self, other: &Self) {
        self.raw_dealt = self.raw_dealt.saturating_add(other.raw_dealt);
        self.mitigated_by_target_guard = self
            .mitigated_by_target_guard
            .saturating_add(other.mitigated_by_target_guard);
        self.applied_dealt = self.applied_dealt.saturating_add(other.applied_dealt);
        self.overkill_dealt = self.overkill_dealt.saturating_add(other.overkill_dealt);
        self.raw_received = self.raw_received.saturating_add(other.raw_received);
        self.mitigated_by_own_guard = self
            .mitigated_by_own_guard
            .saturating_add(other.mitigated_by_own_guard);
        self.applied_received = self.applied_received.saturating_add(other.applied_received);
        self.overkill_received = self
            .overkill_received
            .saturating_add(other.overkill_received);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TerrainTelemetry {
    pub elevation_units_lifted: u64,
    pub elevation_units_dumped: u64,
    pub material_mass_lifted: u128,
    pub material_mass_dumped: u128,
    /// Anonymous field energy erased by this side's terrain actions.
    pub signal_energy_erased: [u128; REFERENCE_SIGNAL_CHANNELS],
}

impl TerrainTelemetry {
    fn observe(
        &mut self,
        change: &blob_engine::resolution::TerrainChange,
        signal_energy_erased: [u64; REFERENCE_SIGNAL_CHANNELS],
    ) {
        let delta = i32::from(change.elevation_after) - i32::from(change.elevation_before);
        if delta < 0 {
            self.elevation_units_lifted = self
                .elevation_units_lifted
                .saturating_add(u64::from(delta.unsigned_abs()));
            self.material_mass_lifted = self
                .material_mass_lifted
                .saturating_add(u128::from(change.material_mass));
        } else if delta > 0 {
            self.elevation_units_dumped = self
                .elevation_units_dumped
                .saturating_add(u64::from(delta.unsigned_abs()));
            self.material_mass_dumped = self
                .material_mass_dumped
                .saturating_add(u128::from(change.material_mass));
        }
        for (channel, amount) in signal_energy_erased.into_iter().enumerate() {
            self.signal_energy_erased[channel] =
                self.signal_energy_erased[channel].saturating_add(u128::from(amount));
        }
    }

    fn merge(&mut self, other: &Self) {
        self.elevation_units_lifted = self
            .elevation_units_lifted
            .saturating_add(other.elevation_units_lifted);
        self.elevation_units_dumped = self
            .elevation_units_dumped
            .saturating_add(other.elevation_units_dumped);
        self.material_mass_lifted = self
            .material_mass_lifted
            .saturating_add(other.material_mass_lifted);
        self.material_mass_dumped = self
            .material_mass_dumped
            .saturating_add(other.material_mass_dumped);
        for channel in 0..REFERENCE_SIGNAL_CHANNELS {
            self.signal_energy_erased[channel] = self.signal_energy_erased[channel]
                .saturating_add(other.signal_energy_erased[channel]);
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SideTelemetry {
    pub actions: ActionFamilyTelemetry,
    pub signals: SignalEmissionTelemetry,
    pub births: u64,
    pub deaths: u64,
    pub kills: u64,
    pub damage: DamageTelemetry,
    pub terrain: TerrainTelemetry,
}

impl SideTelemetry {
    fn merge(&mut self, other: &Self) {
        self.actions.merge(&other.actions);
        self.signals.merge(&other.signals);
        self.births = self.births.saturating_add(other.births);
        self.deaths = self.deaths.saturating_add(other.deaths);
        self.kills = self.kills.saturating_add(other.kills);
        self.damage.merge(&other.damage);
        self.terrain.merge(&other.terrain);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CellEnergyCompartments {
    pub core_mass: u128,
    pub assimilated_energy: u128,
    pub gut_energy: u128,
    pub carried_material_mass: u128,
    pub payload_escrow: u128,
}

impl CellEnergyCompartments {
    fn add_cell(&mut self, cell: &blob_engine::resolution::CellState) {
        self.core_mass = self.core_mass.saturating_add(u128::from(cell.core_mass));
        self.assimilated_energy = self
            .assimilated_energy
            .saturating_add(u128::from(cell.assimilated_energy));
        self.gut_energy = self.gut_energy.saturating_add(u128::from(cell.gut_energy));
        self.carried_material_mass = self
            .carried_material_mass
            .saturating_add(u128::from(cell.carried_material_mass));
        self.payload_escrow = self.payload_escrow.saturating_add(u128::from(
            cell.pending_action
                .as_ref()
                .map_or(0, |action| action.payload_escrow),
        ));
    }

    pub fn total(&self) -> u128 {
        self.core_mass
            .saturating_add(self.assimilated_energy)
            .saturating_add(self.gut_energy)
            .saturating_add(self.carried_material_mass)
            .saturating_add(self.payload_escrow)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentEnergyCompartments {
    /// Terrain's energy-equivalent mass, using the canonical signed-elevation
    /// offset and the active ruleset's conversion factor.
    pub terrain_mass: u128,
    pub plant_energy: u128,
    pub loose_energy: u128,
    pub diffuse_energy: u128,
    pub signal_energy: u128,
    pub signal_energy_by_channel: [u128; REFERENCE_SIGNAL_CHANNELS],
    pub signal_active_tiles_by_channel: [usize; REFERENCE_SIGNAL_CHANNELS],
    pub resource_active_tiles: usize,
}

impl EnvironmentEnergyCompartments {
    pub fn resource_total(&self) -> u128 {
        self.plant_energy
            .saturating_add(self.loose_energy)
            .saturating_add(self.diffuse_energy)
            .saturating_add(self.signal_energy)
    }

    pub fn total(&self) -> u128 {
        self.terrain_mass.saturating_add(self.resource_total())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EcologySample {
    pub episode_step: u64,
    pub sim_time_quanta: u64,
    pub training_cells: usize,
    pub opponent_cells: usize,
    pub training_cells_on_plants: usize,
    pub opponent_cells_on_plants: usize,
    pub training_cells_on_major_food: usize,
    pub opponent_cells_on_major_food: usize,
    pub training_energy: CellEnergyCompartments,
    pub opponent_energy: CellEnergyCompartments,
    pub environment_energy: EnvironmentEnergyCompartments,
    /// Cross-side Moore-neighborhood adjacency edges. Each training/opponent
    /// pair is counted once.
    pub encounter_edges: u64,
    /// Shannon entropy of occupied 8x8 spatial bins, in nats.
    pub training_spatial_entropy: f64,
    pub opponent_spatial_entropy: f64,
    /// Herfindahl concentration of non-terrain environmental energy by tile.
    pub resource_concentration: f64,
    /// Per-channel Herfindahl concentration across tiles.
    pub signal_concentration_by_channel: [f64; REFERENCE_SIGNAL_CHANNELS],
    /// Directed, distinct observable tile edges under the configured signal
    /// neighborhood semantics.
    pub signal_observation_edges: u64,
    /// Sum of absolute per-channel contrasts across those observable edges.
    pub signal_total_variation_by_channel: [u128; REFERENCE_SIGNAL_CHANNELS],
    /// Canonical total mass-energy, including terrain. The signed elevation
    /// baseline is offset by `i16::MIN`, matching the resolver invariant.
    pub tracked_mass_energy: u128,
}

fn spatial_entropy(counts: &[u64; 64], total: usize) -> f64 {
    if total <= 1 {
        return 0.0;
    }
    counts.iter().fold(0.0, |entropy, count| {
        if *count == 0 {
            entropy
        } else {
            let probability = *count as f64 / total as f64;
            entropy - probability * probability.ln()
        }
    })
}

impl EcologySample {
    pub fn capture(
        simulation: &ReferenceSimulation,
        sides: &HashMap<CellKey, TelemetrySide>,
        episode_step: u64,
        width: usize,
        height: usize,
    ) -> Self {
        let mut training_energy = CellEnergyCompartments::default();
        let mut opponent_energy = CellEnergyCompartments::default();
        let mut training_cells = 0usize;
        let mut opponent_cells = 0usize;
        let mut training_cells_on_plants = 0usize;
        let mut opponent_cells_on_plants = 0usize;
        let mut training_cells_on_major_food = 0usize;
        let mut opponent_cells_on_major_food = 0usize;
        let mut training_bins = [0_u64; 64];
        let mut opponent_bins = [0_u64; 64];
        let mut side_by_tile = vec![None; width.saturating_mul(height)];
        for (key, cell) in simulation.cells() {
            let Some(side) = sides.get(key).copied() else {
                continue;
            };
            if cell.position.0 < side_by_tile.len() {
                side_by_tile[cell.position.0] = Some(side);
            }
            let x = cell.position.0 % width;
            let y = cell.position.0 / width;
            let bin_x = x.saturating_mul(8) / width.max(1);
            let bin_y = y.saturating_mul(8) / height.max(1);
            let bin = (bin_y.min(7) * 8 + bin_x.min(7)).min(63);
            let tile = simulation
                .tile_state(cell.position)
                .expect("canonical cell occupies an existing tile");
            let on_plant = tile.plant_capacity > 0 || tile.plant_growth_rate > 0;
            let on_major_food = on_plant || tile.loose_energy > 0;
            match side {
                TelemetrySide::Training => {
                    training_cells += 1;
                    training_cells_on_plants += usize::from(on_plant);
                    training_cells_on_major_food += usize::from(on_major_food);
                    training_bins[bin] += 1;
                    training_energy.add_cell(cell);
                }
                TelemetrySide::Opponents => {
                    opponent_cells += 1;
                    opponent_cells_on_plants += usize::from(on_plant);
                    opponent_cells_on_major_food += usize::from(on_major_food);
                    opponent_bins[bin] += 1;
                    opponent_energy.add_cell(cell);
                }
            }
        }

        let mut environment_energy = EnvironmentEnergyCompartments::default();
        let mut energy_square_sum = 0.0;
        let mut signal_square_sum = [0.0; REFERENCE_SIGNAL_CHANNELS];
        for tile in simulation.tiles() {
            let terrain_units = i32::from(tile.elevation) - i32::from(i16::MIN);
            environment_energy.terrain_mass = environment_energy.terrain_mass.saturating_add(
                u128::from(terrain_units as u32)
                    .saturating_mul(u128::from(simulation.rules().terrain_mass_per_elevation)),
            );
            environment_energy.plant_energy = environment_energy
                .plant_energy
                .saturating_add(u128::from(tile.plant_energy));
            environment_energy.loose_energy = environment_energy
                .loose_energy
                .saturating_add(u128::from(tile.loose_energy));
            environment_energy.diffuse_energy = environment_energy
                .diffuse_energy
                .saturating_add(u128::from(tile.diffuse_energy));
            let signal = tile.signal_energy.into_iter().map(u128::from).sum::<u128>();
            environment_energy.signal_energy =
                environment_energy.signal_energy.saturating_add(signal);
            for (channel, amount) in tile.signal_energy.into_iter().enumerate() {
                environment_energy.signal_energy_by_channel[channel] = environment_energy
                    .signal_energy_by_channel[channel]
                    .saturating_add(u128::from(amount));
                if amount > 0 {
                    environment_energy.signal_active_tiles_by_channel[channel] += 1;
                    signal_square_sum[channel] += (amount as f64).powi(2);
                }
            }
            let resource = u128::from(tile.plant_energy)
                + u128::from(tile.loose_energy)
                + u128::from(tile.diffuse_energy)
                + signal;
            if resource > 0 {
                environment_energy.resource_active_tiles += 1;
                energy_square_sum += (resource as f64).powi(2);
            }
        }
        let resource_total = environment_energy.resource_total();
        let resource_concentration = if resource_total == 0 {
            0.0
        } else {
            energy_square_sum / (resource_total as f64).powi(2)
        };

        let signal_concentration_by_channel = std::array::from_fn(|channel| {
            let total = environment_energy.signal_energy_by_channel[channel];
            if total == 0 {
                0.0
            } else {
                signal_square_sum[channel] / (total as f64).powi(2)
            }
        });

        let mut encounter_edges = 0_u64;
        let neighborhood = simulation.neighborhood();
        for (key, cell) in simulation.cells() {
            if sides.get(key) != Some(&TelemetrySide::Training) {
                continue;
            }
            let mut seen_targets = Vec::with_capacity(neighborhood.slot_count());
            for slot_index in 0..neighborhood.slot_count() {
                let slot = LocalSlot(slot_index as u8);
                if !neighborhood.spec().observations.occupancy.contains(slot) {
                    continue;
                }
                let Some(target) = neighborhood.target(cell.position, slot) else {
                    continue;
                };
                if target == cell.position || seen_targets.contains(&target) {
                    continue;
                }
                seen_targets.push(target);
                if side_by_tile.get(target.0) == Some(&Some(TelemetrySide::Opponents)) {
                    encounter_edges = encounter_edges.saturating_add(1);
                }
            }
        }

        let mut signal_observation_edges = 0_u64;
        let mut signal_total_variation_by_channel = [0_u128; REFERENCE_SIGNAL_CHANNELS];
        for (source_index, source) in simulation.tiles().iter().enumerate() {
            let source_tile = blob_engine::resolution::TileIndex(source_index);
            let mut seen_targets = Vec::with_capacity(neighborhood.slot_count());
            for slot_index in 0..neighborhood.slot_count() {
                let slot = LocalSlot(slot_index as u8);
                if !neighborhood.spec().observations.signal.contains(slot) {
                    continue;
                }
                let Some(target) = neighborhood.target(source_tile, slot) else {
                    continue;
                };
                if target == source_tile || seen_targets.contains(&target) {
                    continue;
                }
                seen_targets.push(target);
                let target_signal = simulation.tiles()[target.0].signal_energy;
                signal_observation_edges = signal_observation_edges.saturating_add(1);
                for channel in 0..REFERENCE_SIGNAL_CHANNELS {
                    signal_total_variation_by_channel[channel] =
                        signal_total_variation_by_channel[channel].saturating_add(u128::from(
                            source.signal_energy[channel].abs_diff(target_signal[channel]),
                        ));
                }
            }
        }

        Self {
            episode_step,
            sim_time_quanta: simulation.now().0,
            training_cells,
            opponent_cells,
            training_cells_on_plants,
            opponent_cells_on_plants,
            training_cells_on_major_food,
            opponent_cells_on_major_food,
            training_spatial_entropy: spatial_entropy(&training_bins, training_cells),
            opponent_spatial_entropy: spatial_entropy(&opponent_bins, opponent_cells),
            encounter_edges,
            resource_concentration,
            signal_concentration_by_channel,
            signal_observation_edges,
            signal_total_variation_by_channel,
            tracked_mass_energy: simulation.total_energy_equivalent(),
            training_energy,
            opponent_energy,
            environment_energy,
        }
    }
}

pub fn signal_energy_by_channel(
    simulation: &ReferenceSimulation,
) -> [u128; REFERENCE_SIGNAL_CHANNELS] {
    let mut totals = [0_u128; REFERENCE_SIGNAL_CHANNELS];
    for tile in simulation.tiles() {
        for (channel, amount) in tile.signal_energy.into_iter().enumerate() {
            totals[channel] = totals[channel].saturating_add(u128::from(amount));
        }
    }
    totals
}

fn terrain_erased_signal(
    report: &BatchReport,
    tile: blob_engine::resolution::TileIndex,
) -> [u64; REFERENCE_SIGNAL_CHANNELS] {
    report
        .delta
        .tiles
        .iter()
        .find(|delta| delta.tile == tile)
        .map_or([0; REFERENCE_SIGNAL_CHANNELS], |delta| {
            delta.before.signal_energy
        })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StepTelemetry {
    pub training: SideTelemetry,
    pub opponents: SideTelemetry,
    pub signal_field: SignalFieldTelemetry,
    pub state_sample: Option<EcologySample>,
}

impl StepTelemetry {
    fn side_mut(&mut self, side: TelemetrySide) -> &mut SideTelemetry {
        match side {
            TelemetrySide::Training => &mut self.training,
            TelemetrySide::Opponents => &mut self.opponents,
        }
    }

    pub fn observe_commitments(
        &mut self,
        commitments: &[ReplayCommitment],
        sides: &HashMap<CellKey, TelemetrySide>,
    ) {
        for commitment in commitments {
            if let Some(side) = sides.get(&commitment.actor).copied() {
                let telemetry = self.side_mut(side);
                telemetry
                    .actions
                    .action_mut(commitment.request.kind())
                    .observe_commitment(commitment);
                telemetry.signals.observe_commitment(commitment);
            }
        }
    }

    pub fn observe_batch(
        &mut self,
        report: &BatchReport,
        sides: &mut HashMap<CellKey, TelemetrySide>,
        signal_decayed: [u128; REFERENCE_SIGNAL_CHANNELS],
    ) {
        let mut terrain_erased = [0_u128; REFERENCE_SIGNAL_CHANNELS];
        let mut terrain_erasure_events = 0_u64;
        for outcome in &report.outcomes {
            if let Some(side) = sides.get(&outcome.actor).copied() {
                let telemetry = self.side_mut(side);
                telemetry
                    .actions
                    .action_mut(outcome.action)
                    .observe_outcome(outcome);
                if let Some(damage) = &outcome.attack_damage {
                    telemetry.damage.observe_dealt(damage);
                }
                if let Some(change) = &outcome.terrain_change {
                    let erased = terrain_erased_signal(report, change.tile);
                    telemetry.terrain.observe(change, erased);
                    for (channel, amount) in erased.into_iter().enumerate() {
                        terrain_erased[channel] =
                            terrain_erased[channel].saturating_add(u128::from(amount));
                    }
                    terrain_erasure_events =
                        terrain_erasure_events.saturating_add(u64::from(erased != [0; 4]));
                }
            }
            if let Some(damage) = &outcome.attack_damage {
                if let Some(victim_side) = sides.get(&damage.victim).copied() {
                    self.side_mut(victim_side).damage.observe_received(damage);
                }
            }
        }
        self.signal_field.terrain_erasure_events = self
            .signal_field
            .terrain_erasure_events
            .saturating_add(terrain_erasure_events);
        for channel in 0..REFERENCE_SIGNAL_CHANNELS {
            self.signal_field.terrain_erased_energy[channel] =
                self.signal_field.terrain_erased_energy[channel]
                    .saturating_add(terrain_erased[channel]);
            self.signal_field.decayed_energy[channel] =
                self.signal_field.decayed_energy[channel].saturating_add(signal_decayed[channel]);
        }
        for death in &report.deaths {
            if let Some(side) = sides.get(death).copied() {
                let telemetry = self.side_mut(side);
                telemetry.deaths = telemetry.deaths.saturating_add(1);
            }
        }
        for (parent, child) in &report.births {
            if let Some(side) = sides.get(parent).copied() {
                let telemetry = self.side_mut(side);
                telemetry.births = telemetry.births.saturating_add(1);
                sides.insert(*child, side);
            }
        }
    }

    pub fn observe_kills(
        &mut self,
        kills: &[(
            blob_interface::types::CellId,
            blob_interface::types::CellId,
            blob_interface::types::TeamId,
        )],
        sides: &HashMap<CellKey, TelemetrySide>,
    ) {
        for (attacker, _, _) in kills {
            if let Ok(actor) = u64::try_from(attacker.0).map(CellKey) {
                if let Some(side) = sides.get(&actor).copied() {
                    let telemetry = self.side_mut(side);
                    telemetry.kills = telemetry.kills.saturating_add(1);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryEpisodeOutcome {
    Win,
    Loss,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EpisodeTelemetryRecord {
    pub schema_version: u32,
    pub environment_index: usize,
    pub episode_id: u64,
    pub environment_seed: u64,
    /// Host curriculum label for this episode. This is diagnostic metadata,
    /// never a Mind observation or canonical simulation input.
    pub curriculum_stage: String,
    pub completed: bool,
    pub outcome: Option<TelemetryEpisodeOutcome>,
    pub episode_steps: u64,
    pub final_sim_time_quanta: u64,
    pub training: SideTelemetry,
    pub opponents: SideTelemetry,
    pub signal_field: SignalFieldTelemetry,
    pub state_samples: Vec<EcologySample>,
}

impl EpisodeTelemetryRecord {
    fn new(
        environment_index: usize,
        episode_id: u64,
        environment_seed: u64,
        curriculum_stage: String,
        initial_sample: EcologySample,
    ) -> Self {
        Self {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            environment_index,
            episode_id,
            environment_seed,
            curriculum_stage,
            completed: false,
            outcome: None,
            episode_steps: 0,
            final_sim_time_quanta: initial_sample.sim_time_quanta,
            training: SideTelemetry::default(),
            opponents: SideTelemetry::default(),
            signal_field: SignalFieldTelemetry::default(),
            state_samples: vec![initial_sample],
        }
    }

    fn apply(&mut self, step: &StepTelemetry, max_samples: usize) {
        self.training.merge(&step.training);
        self.opponents.merge(&step.opponents);
        self.signal_field.merge(&step.signal_field);
        if let Some(sample) = &step.state_sample {
            self.episode_steps = sample.episode_step;
            self.final_sim_time_quanta = sample.sim_time_quanta;
            if self.state_samples.len() < max_samples {
                self.state_samples.push(sample.clone());
            } else if let Some(last) = self.state_samples.last_mut() {
                *last = sample.clone();
            }
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TelemetrySampleMeans {
    pub training_cells: f64,
    pub opponent_cells: f64,
    pub training_cells_on_plants: f64,
    pub opponent_cells_on_plants: f64,
    pub training_cells_on_major_food: f64,
    pub opponent_cells_on_major_food: f64,
    pub training_assimilated_energy: f64,
    pub opponent_assimilated_energy: f64,
    pub environment_plant_energy: f64,
    pub environment_loose_energy: f64,
    pub environment_diffuse_energy: f64,
    pub environment_signal_energy: f64,
    pub signal_active_channel_tiles: f64,
    pub signal_observation_total_variation: f64,
    pub encounter_edges: f64,
    pub training_spatial_entropy: f64,
    pub opponent_spatial_entropy: f64,
    pub resource_concentration: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TelemetrySampleAccumulator {
    pub samples: u64,
    pub sums: TelemetrySampleMeans,
}

impl TelemetrySampleAccumulator {
    fn observe(&mut self, sample: &EcologySample) {
        self.samples = self.samples.saturating_add(1);
        self.sums.training_cells += sample.training_cells as f64;
        self.sums.opponent_cells += sample.opponent_cells as f64;
        self.sums.training_cells_on_plants += sample.training_cells_on_plants as f64;
        self.sums.opponent_cells_on_plants += sample.opponent_cells_on_plants as f64;
        self.sums.training_cells_on_major_food += sample.training_cells_on_major_food as f64;
        self.sums.opponent_cells_on_major_food += sample.opponent_cells_on_major_food as f64;
        self.sums.training_assimilated_energy += sample.training_energy.assimilated_energy as f64;
        self.sums.opponent_assimilated_energy += sample.opponent_energy.assimilated_energy as f64;
        self.sums.environment_plant_energy += sample.environment_energy.plant_energy as f64;
        self.sums.environment_loose_energy += sample.environment_energy.loose_energy as f64;
        self.sums.environment_diffuse_energy += sample.environment_energy.diffuse_energy as f64;
        self.sums.environment_signal_energy += sample.environment_energy.signal_energy as f64;
        self.sums.signal_active_channel_tiles += sample
            .environment_energy
            .signal_active_tiles_by_channel
            .into_iter()
            .sum::<usize>() as f64;
        self.sums.signal_observation_total_variation += sample
            .signal_total_variation_by_channel
            .into_iter()
            .sum::<u128>() as f64;
        self.sums.encounter_edges += sample.encounter_edges as f64;
        self.sums.training_spatial_entropy += sample.training_spatial_entropy;
        self.sums.opponent_spatial_entropy += sample.opponent_spatial_entropy;
        self.sums.resource_concentration += sample.resource_concentration;
    }

    fn means(&self) -> TelemetrySampleMeans {
        if self.samples == 0 {
            return TelemetrySampleMeans::default();
        }
        let denominator = self.samples as f64;
        TelemetrySampleMeans {
            training_cells: self.sums.training_cells / denominator,
            opponent_cells: self.sums.opponent_cells / denominator,
            training_cells_on_plants: self.sums.training_cells_on_plants / denominator,
            opponent_cells_on_plants: self.sums.opponent_cells_on_plants / denominator,
            training_cells_on_major_food: self.sums.training_cells_on_major_food / denominator,
            opponent_cells_on_major_food: self.sums.opponent_cells_on_major_food / denominator,
            training_assimilated_energy: self.sums.training_assimilated_energy / denominator,
            opponent_assimilated_energy: self.sums.opponent_assimilated_energy / denominator,
            environment_plant_energy: self.sums.environment_plant_energy / denominator,
            environment_loose_energy: self.sums.environment_loose_energy / denominator,
            environment_diffuse_energy: self.sums.environment_diffuse_energy / denominator,
            environment_signal_energy: self.sums.environment_signal_energy / denominator,
            signal_active_channel_tiles: self.sums.signal_active_channel_tiles / denominator,
            signal_observation_total_variation: self.sums.signal_observation_total_variation
                / denominator,
            encounter_edges: self.sums.encounter_edges / denominator,
            training_spatial_entropy: self.sums.training_spatial_entropy / denominator,
            opponent_spatial_entropy: self.sums.opponent_spatial_entropy / denominator,
            resource_concentration: self.sums.resource_concentration / denominator,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingTelemetryState {
    pub schema_version: u32,
    pub completed_episodes: u64,
    pub wins: u64,
    pub losses: u64,
    pub timeouts: u64,
    pub completed_episode_steps: u128,
    pub training: SideTelemetry,
    pub opponents: SideTelemetry,
    pub signal_field: SignalFieldTelemetry,
    pub sampled_state: TelemetrySampleAccumulator,
    pub active_episodes: Vec<EpisodeTelemetryRecord>,
}

impl TrainingTelemetryState {
    pub fn new(initial: Vec<(u64, String, EcologySample)>) -> Self {
        let mut sampled_state = TelemetrySampleAccumulator::default();
        for (_, _, sample) in &initial {
            sampled_state.observe(sample);
        }
        let active_episodes = initial
            .into_iter()
            .enumerate()
            .map(|(index, (seed, stage, sample))| {
                EpisodeTelemetryRecord::new(index, 0, seed, stage, sample)
            })
            .collect();
        Self {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            completed_episodes: 0,
            wins: 0,
            losses: 0,
            timeouts: 0,
            completed_episode_steps: 0,
            training: SideTelemetry::default(),
            opponents: SideTelemetry::default(),
            signal_field: SignalFieldTelemetry::default(),
            sampled_state,
            active_episodes,
        }
    }

    pub fn validate(&self, environments: usize) -> Result<(), String> {
        if self.schema_version != TELEMETRY_SCHEMA_VERSION
            || self.active_episodes.len() != environments
        {
            return Err("telemetry resume state does not match the environment set".into());
        }
        Ok(())
    }

    pub fn apply_step(&mut self, environment: usize, step: &StepTelemetry, max_samples: usize) {
        self.training.merge(&step.training);
        self.opponents.merge(&step.opponents);
        self.signal_field.merge(&step.signal_field);
        if let Some(sample) = &step.state_sample {
            self.sampled_state.observe(sample);
        }
        self.active_episodes[environment].apply(step, max_samples);
    }

    pub fn finish_episode(
        &mut self,
        environment: usize,
        outcome: TelemetryEpisodeOutcome,
        episode_steps: u64,
        final_sample: EcologySample,
        max_samples: usize,
    ) -> EpisodeTelemetryRecord {
        let active = &mut self.active_episodes[environment];
        if active
            .state_samples
            .last()
            .is_none_or(|sample| sample.episode_step != final_sample.episode_step)
        {
            if active.state_samples.len() < max_samples {
                active.state_samples.push(final_sample.clone());
            } else if let Some(last) = active.state_samples.last_mut() {
                *last = final_sample.clone();
            }
            self.sampled_state.observe(&final_sample);
        }
        active.completed = true;
        active.outcome = Some(outcome);
        active.episode_steps = episode_steps;
        active.final_sim_time_quanta = final_sample.sim_time_quanta;
        self.completed_episodes = self.completed_episodes.saturating_add(1);
        self.completed_episode_steps = self
            .completed_episode_steps
            .saturating_add(u128::from(episode_steps));
        match outcome {
            TelemetryEpisodeOutcome::Win => self.wins = self.wins.saturating_add(1),
            TelemetryEpisodeOutcome::Loss => self.losses = self.losses.saturating_add(1),
            TelemetryEpisodeOutcome::Timeout => self.timeouts = self.timeouts.saturating_add(1),
        }
        active.clone()
    }

    pub fn start_episode(
        &mut self,
        environment: usize,
        episode_id: u64,
        seed: u64,
        curriculum_stage: String,
        initial_sample: EcologySample,
    ) {
        self.sampled_state.observe(&initial_sample);
        self.active_episodes[environment] = EpisodeTelemetryRecord::new(
            environment,
            episode_id,
            seed,
            curriculum_stage,
            initial_sample,
        );
    }

    pub fn sample_active_episode(
        &mut self,
        environment: usize,
        sample: EcologySample,
        max_samples: usize,
    ) {
        let active = &mut self.active_episodes[environment];
        if active.state_samples.last().is_some_and(|last| {
            last.episode_step == sample.episode_step
                && last.sim_time_quanta == sample.sim_time_quanta
        }) {
            return;
        }
        if active.state_samples.len() < max_samples {
            active.state_samples.push(sample.clone());
        } else if let Some(last) = active.state_samples.last_mut() {
            *last = sample.clone();
        }
        active.episode_steps = sample.episode_step;
        active.final_sim_time_quanta = sample.sim_time_quanta;
        self.sampled_state.observe(&sample);
    }

    pub fn summary(&self) -> TrainingTelemetrySummary {
        TrainingTelemetrySummary {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            completed_episodes: self.completed_episodes,
            censored_active_episodes: self.active_episodes.len(),
            wins: self.wins,
            losses: self.losses,
            timeouts: self.timeouts,
            average_completed_episode_steps: if self.completed_episodes == 0 {
                0.0
            } else {
                self.completed_episode_steps as f64 / self.completed_episodes as f64
            },
            training: self.training.clone(),
            opponents: self.opponents.clone(),
            signal_field: self.signal_field.clone(),
            state_samples: self.sampled_state.samples,
            sample_means: self.sampled_state.means(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingTelemetrySummary {
    pub schema_version: u32,
    pub completed_episodes: u64,
    pub censored_active_episodes: usize,
    pub wins: u64,
    pub losses: u64,
    pub timeouts: u64,
    pub average_completed_episode_steps: f64,
    pub training: SideTelemetry,
    pub opponents: SideTelemetry,
    pub signal_field: SignalFieldTelemetry,
    pub state_samples: u64,
    pub sample_means: TelemetrySampleMeans,
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let nonce = TELEMETRY_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} needs a UTF-8 file name", path.display()))?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode telemetry: {error}"))?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        fs::rename(&temporary, path)
            .map_err(|error| format!("failed to publish {}: {error}", path.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Publish a completed episode at a deterministic immutable path. Resuming
/// from an earlier checkpoint verifies an existing record instead of appending
/// duplicate JSONL rows.
pub fn publish_episode_record(
    artifact_root: &Path,
    record: &EpisodeTelemetryRecord,
) -> Result<PathBuf, String> {
    if !record.completed || record.outcome.is_none() {
        return Err("only completed telemetry episodes may be published".into());
    }
    if record.curriculum_stage.trim().is_empty() {
        return Err("telemetry episode curriculum stage must be nonempty".into());
    }
    let path = artifact_root
        .join("telemetry")
        .join("episodes")
        .join(format!("env-{:04}", record.environment_index))
        .join(format!("episode-{:012}.json", record.episode_id));
    if path.exists() {
        let existing: EpisodeTelemetryRecord = serde_json::from_slice(
            &fs::read(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?,
        )
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        if existing != *record {
            return Err(format!(
                "refusing to replace inconsistent telemetry episode {}",
                path.display()
            ));
        }
        return Ok(path);
    }
    write_json_atomic(&path, record)?;
    Ok(path)
}

pub fn publish_training_summary(
    artifact_root: &Path,
    summary: &TrainingTelemetrySummary,
) -> Result<PathBuf, String> {
    let path = artifact_root.join("telemetry").join("summary.json");
    write_json_atomic(&path, summary)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::resolution::{
        ActionRequest, BoundaryRule, DurationRule, EffortTier, NeighborhoodSpec, ReferenceRuleset,
        ReplayCommitment, SimTime,
    };
    use blob_interface::reference_mind::{ReferenceMemoryUpdate, ReferenceSignalEmission};

    fn rules() -> ReferenceRuleset {
        let duration = DurationRule::new(1024, 0, 1);
        ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
            wait_duration: duration,
            move_duration: duration,
            attack_duration: duration,
            consume_duration: duration,
            split_duration: duration,
            regurgitate_duration: duration,
            excavate_duration: duration,
            deposit_terrain_duration: duration,
            metabolism_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        }
    }

    #[test]
    fn exact_effect_telemetry_tracks_damage_and_terrain_mass() {
        let mut simulation = ReferenceSimulation::new(2, 1, rules()).unwrap();
        let attacker = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
            .unwrap();
        let victim = simulation
            .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 1)
            .unwrap();
        let mut sides = HashMap::from([
            (attacker, TelemetrySide::Training),
            (victim, TelemetrySide::Opponents),
        ]);
        let mut telemetry = StepTelemetry::default();

        simulation
            .commit_action(
                attacker,
                ActionRequest::Attack {
                    target: LocalSlot(4),
                    payload: 20,
                    effort: EffortTier::Standard,
                },
            )
            .unwrap();
        let attack = simulation.resolve_next_batch().unwrap();
        telemetry.observe_batch(
            &attack,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );
        assert_eq!(telemetry.training.damage.raw_dealt, 20);
        assert_eq!(telemetry.training.damage.applied_dealt, 20);
        assert_eq!(telemetry.training.damage.mitigated_by_target_guard, 0);
        assert_eq!(telemetry.training.damage.overkill_dealt, 0);
        assert_eq!(telemetry.opponents.damage.raw_received, 20);
        assert_eq!(telemetry.opponents.damage.applied_received, 20);

        simulation
            .commit_action(attacker, ActionRequest::Excavate)
            .unwrap();
        let excavation = simulation.resolve_next_batch().unwrap();
        telemetry.observe_batch(
            &excavation,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );
        assert_eq!(telemetry.training.terrain.elevation_units_lifted, 1);
        assert_eq!(
            telemetry.training.terrain.material_mass_lifted,
            u128::from(simulation.rules().terrain_mass_per_elevation)
        );

        simulation
            .commit_action(attacker, ActionRequest::DepositTerrain)
            .unwrap();
        let deposition = simulation.resolve_next_batch().unwrap();
        telemetry.observe_batch(
            &deposition,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );
        assert_eq!(telemetry.training.terrain.elevation_units_dumped, 1);
        assert_eq!(
            telemetry.training.terrain.material_mass_dumped,
            u128::from(simulation.rules().terrain_mass_per_elevation)
        );
    }

    #[test]
    fn consume_telemetry_counts_only_energy_actually_extracted() {
        let mut simulation = ReferenceSimulation::new(1, 1, rules()).unwrap();
        let tile = simulation.tile(0, 0).unwrap();
        let actor = simulation.add_cell(tile, 10, 100, 0).unwrap();
        {
            let plant = simulation.tile_state_mut(tile).unwrap();
            plant.plant_energy = 7;
            plant.plant_capacity = 7;
        }
        let mut sides = HashMap::from([(actor, TelemetrySide::Training)]);
        let mut telemetry = StepTelemetry::default();
        let occupancy = EcologySample::capture(&simulation, &sides, 0, 1, 1);
        assert_eq!(occupancy.training_cells_on_plants, 1);
        assert_eq!(occupancy.training_cells_on_major_food, 1);
        assert_eq!(occupancy.opponent_cells_on_plants, 0);
        simulation
            .commit_action(actor, ActionRequest::Consume { amount: 20 })
            .unwrap();
        let report = simulation.resolve_next_batch().unwrap();
        telemetry.observe_batch(
            &report,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );

        assert_eq!(telemetry.training.actions.consume.succeeded, 1);
        assert_eq!(telemetry.training.actions.consume.payload, 0);
        assert_eq!(telemetry.training.actions.consume.consumed_energy, 7);
    }

    #[test]
    fn signal_telemetry_closes_deposit_decay_and_terrain_erasure_accounting() {
        let mut simulation = ReferenceSimulation::new(2, 1, rules()).unwrap();
        let actor = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
            .unwrap();
        let mut sides = HashMap::from([(actor, TelemetrySide::Training)]);
        let mut telemetry = StepTelemetry::default();

        let request = ActionRequest::Signal {
            amounts: [3, 6, 0, 3],
        };
        let started_at = simulation.now();
        let receipt = simulation.commit_action(actor, request.clone()).unwrap();
        let explicit = ReplayCommitment {
            actor,
            request,
            signal: None,
            memory_update: ReferenceMemoryUpdate::Retain,
            started_at,
            receipt,
        };
        telemetry.observe_commitments(&[explicit], &sides);
        assert_eq!(telemetry.training.signals.channel_energy, [3, 6, 0, 3]);

        let sample = EcologySample::capture(&simulation, &sides, 0, 2, 1);
        assert_eq!(
            sample.environment_energy.signal_energy_by_channel,
            [3, 6, 0, 3]
        );
        assert_eq!(
            sample.environment_energy.signal_active_tiles_by_channel,
            [1, 1, 0, 1]
        );
        assert!(sample.signal_observation_edges > 0);
        assert!(sample.signal_total_variation_by_channel[0] > 0);
        assert_eq!(sample.signal_concentration_by_channel[0], 1.0);

        let report = simulation.resolve_next_batch().unwrap();
        let signal_after = signal_energy_by_channel(&simulation);
        telemetry.observe_batch(
            &report,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );
        assert_eq!(signal_after, [2, 5, 0, 2]);

        simulation
            .commit_action(actor, ActionRequest::Excavate)
            .unwrap();
        let report = simulation.resolve_next_batch().unwrap();
        let signal_after = signal_energy_by_channel(&simulation);
        telemetry.observe_batch(
            &report,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );
        assert_eq!(signal_after, [0; 4]);
        assert_eq!(
            telemetry.training.terrain.signal_energy_erased,
            [1, 4, 0, 1]
        );
        assert_eq!(telemetry.signal_field.terrain_erased_energy, [1, 4, 0, 1]);
        assert_eq!(telemetry.signal_field.terrain_erasure_events, 1);

        let request = ActionRequest::Wait;
        let sidecar = ReferenceSignalEmission {
            channel: 2,
            amount: 4,
        };
        let started_at = simulation.now();
        let receipt = simulation
            .commit_decision_with_signal(actor, request.clone(), Some(sidecar), Vec::new())
            .unwrap();
        let commitment = ReplayCommitment {
            actor,
            request,
            signal: Some(sidecar),
            memory_update: ReferenceMemoryUpdate::Retain,
            started_at,
            receipt,
        };
        telemetry.observe_commitments(&[commitment], &sides);
        let report = simulation.resolve_next_batch().unwrap();
        let signal_after = signal_energy_by_channel(&simulation);
        telemetry.observe_batch(
            &report,
            &mut sides,
            simulation.last_resolution_metrics().signal_energy_decayed,
        );

        assert_eq!(telemetry.training.signals.emitted_decisions, 2);
        assert_eq!(telemetry.training.signals.explicit_signal_actions, 1);
        assert_eq!(telemetry.training.signals.sidecar_emissions, 1);
        assert_eq!(telemetry.training.signals.channel_energy, [3, 6, 4, 3]);
        assert_eq!(telemetry.training.signals.pattern_counts[0b1011], 1);
        assert_eq!(telemetry.training.signals.pattern_counts[0b0100], 1);
        assert_eq!(telemetry.signal_field.decayed_energy, [2, 2, 1, 2]);
        assert_eq!(signal_after, [0, 0, 3, 0]);
        assert_eq!(simulation.now(), SimTime(3072));
    }
}
