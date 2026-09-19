//! Canonical anonymous Mind observations and reusable frontier projection.
//!
//! This child module can read authoritative resolver state, but exposes only
//! relative, ruleset-masked fields and the requesting cell's private values.

use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{
    CurrentTileObservation, LocalObservation as MindLocalObservation, ReferenceActionSpace,
    ReferenceActivity, ReferenceMindInput, ReferenceNeighbor, ReferenceOutcome,
    ReferenceOutcomeStatus, ReferenceProgress, ReferenceRejectReason, ReferenceSelfState,
    EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
};

use super::super::neighborhood::{LocalSlot, TargetingAction, TileIndex};
use super::{
    ActionRequest, ActivityCue, CellKey, CellState, CommitError, NeighborCue, OutcomeStatus,
    ProgressBucket, ReferenceSimulation, RejectReason, TileState, REFERENCE_SIGNAL_CHANNELS,
};

/// Read-only acceleration for projecting a decision frontier. It borrows the
/// canonical simulation, exposes no peer state, and is discarded after the
/// host has built each cell's isolated input.
pub struct ReferenceObservationBatch<'a> {
    simulation: &'a ReferenceSimulation,
}

trait ObservationCellLookup {
    fn observation_cell(&self, key: CellKey) -> Option<&CellState>;
}

impl ObservationCellLookup for ReferenceSimulation {
    fn observation_cell(&self, key: CellKey) -> Option<&CellState> {
        self.cells.get(&key)
    }
}

impl ObservationCellLookup for ReferenceObservationBatch<'_> {
    fn observation_cell(&self, key: CellKey) -> Option<&CellState> {
        self.simulation.cells.get(&key)
    }
}

impl ReferenceObservationBatch<'_> {
    pub fn reference_mind_input(
        &self,
        observer: CellKey,
        randomness: PrivateRandom,
    ) -> Result<ReferenceMindInput, CommitError> {
        let mut input = self.simulation.empty_reference_mind_input(randomness);
        self.reference_mind_input_into(observer, randomness, &mut input)?;
        Ok(input)
    }

    pub fn reference_mind_input_into(
        &self,
        observer: CellKey,
        randomness: PrivateRandom,
        input: &mut ReferenceMindInput,
    ) -> Result<(), CommitError> {
        self.simulation
            .reference_mind_input_into_with_lookup(self, observer, randomness, input)
    }
}

impl ReferenceSimulation {
    pub fn neighbor_cues(
        &self,
        observer: CellKey,
    ) -> Result<Vec<Option<NeighborCue>>, CommitError> {
        let observer_cell = self
            .cells
            .get(&observer)
            .ok_or(CommitError::UnknownCell(observer))?;
        let mut cues = Vec::with_capacity(self.neighborhood.slot_count());
        for slot_index in 0..self.neighborhood.slot_count() {
            let slot = LocalSlot(u8::try_from(slot_index).map_err(|_| {
                CommitError::InvariantViolation("compiled neighborhood exceeds u8 slot range")
            })?);
            let target = self.neighborhood.target(observer_cell.position, slot);
            cues.push(self.neighbor_cue_at_target(observer, slot, target)?);
        }
        Ok(cues)
    }

    fn neighbor_cue_at_target(
        &self,
        observer: CellKey,
        slot: LocalSlot,
        target: Option<TileIndex>,
    ) -> Result<Option<NeighborCue>, CommitError> {
        self.neighbor_cue_at_target_with_lookup(self, observer, slot, target)
    }

    fn neighbor_cue_at_target_with_lookup<L: ObservationCellLookup + ?Sized>(
        &self,
        lookup: &L,
        observer: CellKey,
        slot: LocalSlot,
        target: Option<TileIndex>,
    ) -> Result<Option<NeighborCue>, CommitError> {
        let observations = self.neighborhood.spec().observations;
        let observes_cell = observations.occupancy.contains(slot)
            || observations.marker.contains(slot)
            || observations.apparent_mass.contains(slot)
            || observations.activity.contains(slot);
        if !observes_cell {
            return Ok(None);
        }
        let Some(tile) = target else {
            return Ok(None);
        };
        let Some(neighbor_key) = self.tiles[tile.0].occupant else {
            return Ok(None);
        };
        if neighbor_key == observer {
            return Ok(None);
        }
        let neighbor =
            lookup
                .observation_cell(neighbor_key)
                .ok_or(CommitError::InvariantViolation(
                    "occupied tile references a missing cell",
                ))?;
        let mass_bucket = u8::try_from(
            (neighbor.total_mass() / u128::from(self.rules.apparent_mass_bucket_width))
                .min(u128::from(u8::MAX)),
        )
        .map_err(|_| CommitError::InvariantViolation("apparent mass bucket overflow"))?;
        let (activity, progress) = self.activity_cue(neighbor);
        Ok(Some(NeighborCue {
            marker: observations
                .marker
                .contains(slot)
                .then_some(neighbor.marker),
            apparent_mass_bucket: observations
                .apparent_mass
                .contains(slot)
                .then_some(mass_bucket),
            activity: observations.activity.contains(slot).then_some(activity),
            progress: observations
                .activity
                .contains(slot)
                .then_some(progress)
                .flatten(),
        }))
    }

    /// Builds the complete canonical Mind observation for one ready cell.
    /// All spatial values are relative slots filtered by ruleset observation
    /// masks; engine identity and absolute state never cross this boundary.
    pub fn reference_mind_input(
        &self,
        observer: CellKey,
        randomness: PrivateRandom,
    ) -> Result<ReferenceMindInput, CommitError> {
        let mut input = self.empty_reference_mind_input(randomness);
        self.reference_mind_input_into(observer, randomness, &mut input)?;
        Ok(input)
    }

    fn empty_reference_mind_input(&self, randomness: PrivateRandom) -> ReferenceMindInput {
        ReferenceMindInput {
            self_state: ReferenceSelfState {
                core_mass: 0,
                assimilated_energy: 0,
                gut_energy: 0,
                metabolism_remainder: 0,
                carried_material_mass: 0,
                marker: 0,
                guarded: false,
                last_outcome: None,
            },
            current_tile: CurrentTileObservation {
                elevation: 0,
                plant_energy: 0,
                plant_capacity: 0,
                plant_growth_rate: 0,
                loose_energy: 0,
                diffuse_energy: 0,
                signal_energy: [0; REFERENCE_SIGNAL_CHANNELS],
            },
            slots: Vec::with_capacity(self.neighborhood.slot_count()),
            action_space: ReferenceActionSpace {
                wait_enabled: false,
                guard_enabled: false,
                consume_enabled: false,
                excavate_enabled: false,
                deposit_terrain_enabled: false,
                signal_enabled: false,
                move_targets: 0,
                attack_targets: 0,
                split_targets: 0,
                regurgitate_targets: 0,
                effort_mask: 0,
                max_consume_amount: 0,
                gut_capacity: 0,
                max_private_memory_bytes: 0,
                minimum_survival_energy: 0,
                child_core_mass: 0,
                metabolism_rate_numerator: 0,
                metabolism_rate_denominator: 0,
                terrain_mass_per_elevation: 0,
                signal_emission_cost: 0,
                effort_cost_numerators: [0; 3],
                effort_cost_denominators: [1; 3],
                move_effort_base: 0,
                move_mass_units_per_effort: 1,
                attack_effort_base: 0,
                guard_effort_base: 0,
                consume_effort_base: 0,
                split_effort_base: 0,
                regurgitate_effort_base: 0,
                excavate_effort_base: 0,
                deposit_terrain_effort_base: 0,
            },
            private_memory: Vec::new(),
            randomness,
        }
    }

    /// Borrows the simulation's direct indexed lookup for a complete
    /// observation frontier. No per-frontier cell index is allocated.
    pub fn observation_batch(&self) -> ReferenceObservationBatch<'_> {
        ReferenceObservationBatch { simulation: self }
    }

    /// Rebuilds one isolated observation while retaining its owned buffers.
    /// Hosts can keep one scratch input per pristine Mind instance instead of
    /// allocating per cell; no state from a prior observer remains visible.
    pub fn reference_mind_input_into(
        &self,
        observer: CellKey,
        randomness: PrivateRandom,
        input: &mut ReferenceMindInput,
    ) -> Result<(), CommitError> {
        self.reference_mind_input_into_with_lookup(self, observer, randomness, input)
    }

    fn reference_mind_input_into_with_lookup<L: ObservationCellLookup + ?Sized>(
        &self,
        lookup: &L,
        observer: CellKey,
        randomness: PrivateRandom,
        input: &mut ReferenceMindInput,
    ) -> Result<(), CommitError> {
        let observer_cell = lookup
            .observation_cell(observer)
            .ok_or(CommitError::UnknownCell(observer))?;
        if !observer_cell.is_ready_at(self.now) {
            return Err(CommitError::CellNotReady {
                actor: observer,
                ready_at: observer_cell.ready_at,
            });
        }
        let spec = self.neighborhood.spec();
        let targets = self.neighborhood.targets(observer_cell.position).ok_or(
            CommitError::InvariantViolation(
                "observer position is outside the compiled neighborhood",
            ),
        )?;
        input.slots.clear();
        input.slots.reserve(self.neighborhood.slot_count());
        for (slot_index, (offset, target)) in
            spec.slots.iter().zip(targets.iter().copied()).enumerate()
        {
            let slot_u8 = u8::try_from(slot_index).map_err(|_| {
                CommitError::InvariantViolation("compiled neighborhood exceeds u8 slot range")
            })?;
            let slot = LocalSlot(slot_u8);
            let tile = target.map(|target| &self.tiles[target.0]);
            let cue = self.neighbor_cue_at_target_with_lookup(lookup, observer, slot, target)?;
            let observes_terrain = spec.observations.terrain.contains(slot);
            let observes_energy = spec.observations.energy.contains(slot);
            let observes_signal = spec.observations.signal.contains(slot);
            input.slots.push(MindLocalObservation {
                slot: slot_u8,
                dx: offset.dx,
                dy: offset.dy,
                distance_cost_q10: offset.distance_cost_q10,
                reachable: target.is_some(),
                elevation: observes_terrain
                    .then(|| tile.map(|tile| tile.elevation))
                    .flatten(),
                plant_energy: observed_energy(observes_energy, tile, |tile| tile.plant_energy),
                plant_capacity: observed_energy(observes_energy, tile, |tile| tile.plant_capacity),
                plant_growth_rate: observed_energy(observes_energy, tile, |tile| {
                    tile.plant_growth_rate
                }),
                loose_energy: observed_energy(observes_energy, tile, |tile| tile.loose_energy),
                diffuse_energy: observed_energy(observes_energy, tile, |tile| tile.diffuse_energy),
                signal_energy: observes_signal
                    .then(|| tile.map(|tile| tile.signal_energy))
                    .flatten(),
                neighbor: cue.map(reference_neighbor),
            });
        }
        let max_private_memory_bytes = u32::try_from(self.rules.max_private_memory_bytes)
            .map_err(|_| CommitError::InvariantViolation("private memory limit exceeds u32"))?;
        input.self_state = ReferenceSelfState {
            core_mass: observer_cell.core_mass,
            assimilated_energy: observer_cell.assimilated_energy,
            gut_energy: observer_cell.gut_energy,
            metabolism_remainder: observer_cell.metabolism_remainder,
            carried_material_mass: observer_cell.carried_material_mass,
            marker: observer_cell.marker,
            guarded: observer_cell.guarded,
            last_outcome: observer_cell.last_outcome.map(reference_outcome),
        };
        input.current_tile = {
            let tile = &self.tiles[observer_cell.position.0];
            CurrentTileObservation {
                elevation: tile.elevation,
                plant_energy: tile.plant_energy,
                plant_capacity: tile.plant_capacity,
                plant_growth_rate: tile.plant_growth_rate,
                loose_energy: tile.loose_energy,
                diffuse_energy: tile.diffuse_energy,
                signal_energy: tile.signal_energy,
            }
        };
        input.action_space = ReferenceActionSpace {
            wait_enabled: true,
            guard_enabled: true,
            consume_enabled: true,
            excavate_enabled: self.tiles[observer_cell.position.0].elevation > i16::MIN,
            deposit_terrain_enabled: self.tiles[observer_cell.position.0].elevation < i16::MAX
                && observer_cell.carried_material_mass >= self.rules.terrain_mass_per_elevation,
            signal_enabled: self.rules.signal_emission_cost > 0,
            move_targets: spec.target_mask(TargetingAction::Move).bits(),
            attack_targets: spec.target_mask(TargetingAction::Attack).bits(),
            split_targets: spec.target_mask(TargetingAction::Split).bits(),
            regurgitate_targets: spec.target_mask(TargetingAction::Regurgitate).bits(),
            effort_mask: EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT | EFFORT_BURST_BIT,
            max_consume_amount: self.rules.bite_capacity.min(
                self.rules
                    .gut_capacity
                    .saturating_sub(observer_cell.gut_energy),
            ),
            gut_capacity: self.rules.gut_capacity,
            max_private_memory_bytes,
            minimum_survival_energy: self.rules.minimum_survival_energy,
            child_core_mass: self.rules.child_core_mass,
            metabolism_rate_numerator: self.rules.metabolism_rate_numerator,
            metabolism_rate_denominator: self.rules.metabolism_rate_denominator,
            terrain_mass_per_elevation: self.rules.terrain_mass_per_elevation,
            signal_emission_cost: self.rules.signal_emission_cost,
            effort_cost_numerators: self
                .rules
                .effort_profiles
                .map(|profile| profile.cost_numerator),
            effort_cost_denominators: self
                .rules
                .effort_profiles
                .map(|profile| profile.cost_denominator),
            move_effort_base: self.rules.move_effort_base,
            move_mass_units_per_effort: self.rules.move_mass_units_per_effort,
            attack_effort_base: self.rules.attack_effort_base,
            guard_effort_base: self.rules.guard_effort_base,
            consume_effort_base: self.rules.consume_effort_base,
            split_effort_base: self.rules.split_effort_base,
            regurgitate_effort_base: self.rules.regurgitate_effort_base,
            excavate_effort_base: self.rules.excavate_effort_base,
            deposit_terrain_effort_base: self.rules.deposit_terrain_effort_base,
        };
        input.private_memory.clear();
        input
            .private_memory
            .extend_from_slice(&observer_cell.private_memory);
        input.randomness = randomness;
        Ok(())
    }

    fn activity_cue(&self, cell: &CellState) -> (ActivityCue, Option<ProgressBucket>) {
        let Some(pending) = &cell.pending_action else {
            return (
                if cell.guarded {
                    ActivityCue::Guarding
                } else {
                    ActivityCue::Ready
                },
                None,
            );
        };
        let activity = if pending.rejection.is_some() {
            ActivityCue::OtherBusy
        } else {
            match pending.request {
                ActionRequest::Wait => ActivityCue::OtherBusy,
                ActionRequest::Move { .. } => ActivityCue::Moving,
                ActionRequest::Attack { .. } => ActivityCue::AttackWindup,
                ActionRequest::Guard { .. } => ActivityCue::Guarding,
                ActionRequest::Consume { .. } => ActivityCue::Feeding,
                ActionRequest::Split { .. } => ActivityCue::Splitting,
                ActionRequest::Regurgitate { .. } => ActivityCue::OtherBusy,
                ActionRequest::Signal { .. } => ActivityCue::OtherBusy,
                ActionRequest::Excavate | ActionRequest::DepositTerrain => {
                    ActivityCue::ManipulatingTerrain
                }
            }
        };
        let duration = pending.completes_at.0.saturating_sub(pending.started_at.0);
        let elapsed = self
            .now
            .0
            .saturating_sub(pending.started_at.0)
            .min(duration);
        let progress = if duration == 0 {
            ProgressBucket::Late
        } else {
            match (u128::from(elapsed) * 3) / u128::from(duration) {
                0 => ProgressBucket::Early,
                1 => ProgressBucket::Middle,
                _ => ProgressBucket::Late,
            }
        };
        (activity, Some(progress))
    }
}

fn observed_energy(
    visible: bool,
    tile: Option<&TileState>,
    value: impl FnOnce(&TileState) -> u64,
) -> Option<u64> {
    visible.then(|| tile.map(value)).flatten()
}

fn reference_neighbor(value: NeighborCue) -> ReferenceNeighbor {
    ReferenceNeighbor {
        marker: value.marker,
        apparent_mass_bucket: value.apparent_mass_bucket,
        activity: value.activity.map(|activity| match activity {
            ActivityCue::Ready => ReferenceActivity::Ready,
            ActivityCue::Moving => ReferenceActivity::Moving,
            ActivityCue::AttackWindup => ReferenceActivity::AttackWindup,
            ActivityCue::Guarding => ReferenceActivity::Guarding,
            ActivityCue::Feeding => ReferenceActivity::Feeding,
            ActivityCue::Splitting => ReferenceActivity::Splitting,
            ActivityCue::ManipulatingTerrain => ReferenceActivity::ManipulatingTerrain,
            ActivityCue::OtherBusy => ReferenceActivity::OtherBusy,
        }),
        progress: value.progress.map(|progress| match progress {
            ProgressBucket::Early => ReferenceProgress::Early,
            ProgressBucket::Middle => ReferenceProgress::Middle,
            ProgressBucket::Late => ReferenceProgress::Late,
        }),
    }
}

fn reference_outcome(value: OutcomeStatus) -> ReferenceOutcome {
    match value {
        OutcomeStatus::Success => ReferenceOutcome {
            status: ReferenceOutcomeStatus::Success,
            rejected_reason: None,
        },
        OutcomeStatus::Frustrated => ReferenceOutcome {
            status: ReferenceOutcomeStatus::Frustrated,
            rejected_reason: None,
        },
        OutcomeStatus::Contested => ReferenceOutcome {
            status: ReferenceOutcomeStatus::Contested,
            rejected_reason: None,
        },
        OutcomeStatus::Interrupted => ReferenceOutcome {
            status: ReferenceOutcomeStatus::Interrupted,
            rejected_reason: None,
        },
        OutcomeStatus::Rejected(reason) => ReferenceOutcome {
            status: ReferenceOutcomeStatus::Rejected,
            rejected_reason: Some(match reason {
                RejectReason::InvalidSlot => ReferenceRejectReason::InvalidSlot,
                RejectReason::ActionNotAllowedInSlot => {
                    ReferenceRejectReason::ActionNotAllowedInSlot
                }
                RejectReason::TargetOutsideWorld => ReferenceRejectReason::TargetOutsideWorld,
                RejectReason::TargetsSelf => ReferenceRejectReason::TargetsSelf,
                RejectReason::ZeroPayload => ReferenceRejectReason::ZeroPayload,
                RejectReason::InsufficientGutEnergy => ReferenceRejectReason::InsufficientGutEnergy,
                RejectReason::ChildAllocationTooSmall => {
                    ReferenceRejectReason::ChildAllocationTooSmall
                }
                RejectReason::PrivateMemoryTooLarge => ReferenceRejectReason::PrivateMemoryTooLarge,
                RejectReason::InsufficientEnergy => ReferenceRejectReason::InsufficientEnergy,
                RejectReason::InsufficientMaterial => ReferenceRejectReason::InsufficientMaterial,
                RejectReason::TerrainLimit => ReferenceRejectReason::TerrainLimit,
                RejectReason::ArithmeticOverflow => ReferenceRejectReason::ArithmeticOverflow,
            }),
        },
    }
}
