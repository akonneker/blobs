//! A separation-safe, stateful colony policy.
//!
//! Every invocation can see only one cell's local ABI input. Coordination is
//! therefore deliberately lossy: private maps are inherited by split children,
//! role markers are locally observable but unauthenticated, and signal fields
//! are anonymous. There is no process-global memory or hidden team identity.

use blob_interface::reference_mind::{
    LocalObservation, ReferenceActionSpace, ReferenceEffort, ReferenceMemoryUpdate,
    ReferenceMindAction, ReferenceMindDecision, ReferenceMindInput, ReferenceOutcomeStatus,
    ReferenceRejectReason, ReferenceSignalEmission,
};
#[cfg(target_arch = "wasm32")]
use blob_interface::reference_mind_converter::{
    ReferenceMindLimits, capnp_to_reference_mind_input, reference_mind_decision_to_capnp,
};
use blob_mind_utils::{
    action_is_commit_legal, action_is_commit_legal_with_signal_amount, attack_payload,
    best_energy_slot, choose_slot, current_food, legal_action_or_wait, preferred_effort,
    safe_empty_slots, split_allocation,
};
#[cfg(target_arch = "wasm32")]
use extism_pdk::*;

#[cfg(not(target_arch = "wasm32"))]
pub mod telemetry;

const MAGIC: [u8; 4] = *b"CLN5";
const HEADER_BYTES: usize = 104;
const MAP_ENTRY_BYTES: usize = 12;
const MAX_MAP_ENTRIES: usize = 16;
const COVERAGE_SIDE: i16 = 8;
const COVERAGE_CENTER: i16 = 3;
const ESTIMATE_FAMILIES: usize = 8;
const NO_PENDING_SIGNAL: u8 = u8::MAX;
const NO_PENDING_PLANT: u8 = u8::MAX;
const ROLE_MARKER_BASE: u32 = 0x434f_4c00;
const PLANT_SIGNAL: u8 = 0;
const THREAT_SIGNAL: u8 = 1;
const BUILD_SIGNAL: u8 = 2;
const FRONTIER_SIGNAL: u8 = 3;
const PLANT_SIGNAL_QUANTA: u64 = 4;
const THREAT_SIGNAL_QUANTA: u64 = 8;
const BUILD_SIGNAL_QUANTA: u64 = 4;
const FRONTIER_SIGNAL_QUANTA: u64 = 2;
/// South cardinal ring position. The first layer remains traversable here and
/// later layers skip it so defenders can relieve the enclosure.
// Construction-ring indices start at north-west and proceed clockwise, so the
// south cardinal is 5 (unlike its Moore-neighborhood observation slot, 6).
const GATE_RING_INDEX: u8 = 5;

/// Experimental policy profiles used by deterministic native ablations. The
/// Wasm entrypoint always uses `FullMultichannel`; alternate profiles receive
/// the same isolated cell input and no shared state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalPolicy {
    Disabled,
    OneQuantumSidecar,
    SemanticSidecar,
    CadencedSemanticSidecar,
    FullMultichannel,
}

impl SignalPolicy {
    const fn allows_explicit_action(self) -> bool {
        matches!(self, Self::FullMultichannel)
    }

    const fn uses_semantic_strength(self) -> bool {
        !matches!(self, Self::Disabled | Self::OneQuantumSidecar)
    }

    const fn periodic_signal_due(self, decisions: u16) -> bool {
        !matches!(self, Self::CadencedSemanticSidecar) || decisions.is_multiple_of(8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Role {
    Feeder = 0,
    Explorer = 1,
    Builder = 2,
    Defender = 3,
    Attacker = 4,
}

impl Role {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Feeder),
            1 => Some(Self::Explorer),
            2 => Some(Self::Builder),
            3 => Some(Self::Defender),
            4 => Some(Self::Attacker),
            _ => None,
        }
    }

    fn marker(self) -> u32 {
        ROLE_MARKER_BASE + u32::from(self as u8)
    }

    fn from_marker(marker: u32) -> Option<Self> {
        marker
            .checked_sub(ROLE_MARKER_BASE)
            .and_then(|value| u8::try_from(value).ok())
            .and_then(Self::from_byte)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum PendingAction {
    None = 0,
    Move = 1,
    Attack = 2,
    Guard = 3,
    Consume = 4,
    Split = 5,
    Excavate = 6,
    Deposit = 7,
}

impl PendingAction {
    fn from_byte(value: u8) -> Self {
        match value {
            1 => Self::Move,
            2 => Self::Attack,
            3 => Self::Guard,
            4 => Self::Consume,
            5 => Self::Split,
            6 => Self::Excavate,
            7 => Self::Deposit,
            _ => Self::None,
        }
    }

    fn estimate_family(self) -> Option<EstimateFamily> {
        match self {
            Self::None => None,
            Self::Move => Some(EstimateFamily::Move),
            Self::Attack => Some(EstimateFamily::Attack),
            Self::Guard => Some(EstimateFamily::Guard),
            Self::Consume => Some(EstimateFamily::Consume),
            Self::Split => Some(EstimateFamily::Split),
            Self::Excavate => Some(EstimateFamily::Excavate),
            Self::Deposit => Some(EstimateFamily::Deposit),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum EstimateFamily {
    Move = 0,
    Attack = 1,
    Guard = 2,
    Consume = 3,
    Split = 4,
    Excavate = 5,
    Deposit = 6,
    Signal = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ConstructionPhase {
    None = 0,
    Source = 1,
    Wall = 2,
}

impl ConstructionPhase {
    fn from_byte(value: u8) -> Self {
        match value {
            1 => Self::Source,
            2 => Self::Wall,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlantRecord {
    x: i16,
    y: i16,
    capacity: u16,
    growth: u16,
    wall_cursor: u8,
    wall_layer: u8,
    pit_pressure: u8,
}

#[derive(Debug, Clone)]
struct ColonyMemory {
    role: Role,
    x: i16,
    y: i16,
    pending: PendingAction,
    pending_dx: i8,
    pending_dy: i8,
    route_cursor: u8,
    pending_plant_index: u8,
    pending_construction_phase: ConstructionPhase,
    child_cursor: u8,
    decisions: u16,
    move_successes: u16,
    move_failures: u16,
    min_growth: u16,
    max_growth: u16,
    max_capacity: u16,
    threat_score: u8,
    /// Rolling 8×8 visitation window centered near this cell's current
    /// lineage-local position. Bits are shifted after successful movement, so
    /// old distant history falls away instead of saturating a global filter.
    coverage: u64,
    outcome_successes: [u8; ESTIMATE_FAMILIES],
    outcome_setbacks: [u8; ESTIMATE_FAMILIES],
    outcome_contentions: [u8; ESTIMATE_FAMILIES],
    snapshot_assimilated_energy: u64,
    snapshot_gut_energy: u64,
    expected_signal_energy: u64,
    guarded_loss_ema: u16,
    unguarded_loss_ema: u16,
    digestion_progress_ema: u16,
    signal_retention_q8: u16,
    guarded_loss_samples: u8,
    unguarded_loss_samples: u8,
    digestion_samples: u8,
    signal_samples: u8,
    pending_signal: u8,
    snapshot_valid: bool,
    snapshot_guarded: bool,
    plants: Vec<PlantRecord>,
}

impl ColonyMemory {
    fn new(role: Role) -> Self {
        Self {
            role,
            x: 0,
            y: 0,
            pending: PendingAction::None,
            pending_dx: 0,
            pending_dy: 0,
            route_cursor: 0,
            pending_plant_index: NO_PENDING_PLANT,
            pending_construction_phase: ConstructionPhase::None,
            child_cursor: 0,
            decisions: 0,
            move_successes: 0,
            move_failures: 0,
            min_growth: 0,
            max_growth: 0,
            max_capacity: 0,
            threat_score: 0,
            coverage: 1_u64 << (COVERAGE_CENTER * COVERAGE_SIDE + COVERAGE_CENTER),
            outcome_successes: [0; ESTIMATE_FAMILIES],
            outcome_setbacks: [0; ESTIMATE_FAMILIES],
            outcome_contentions: [0; ESTIMATE_FAMILIES],
            snapshot_assimilated_energy: 0,
            snapshot_gut_energy: 0,
            expected_signal_energy: 0,
            guarded_loss_ema: 0,
            unguarded_loss_ema: 0,
            digestion_progress_ema: 0,
            signal_retention_q8: 0,
            guarded_loss_samples: 0,
            unguarded_loss_samples: 0,
            digestion_samples: 0,
            signal_samples: 0,
            pending_signal: NO_PENDING_SIGNAL,
            snapshot_valid: false,
            snapshot_guarded: false,
            plants: Vec::new(),
        }
    }

    fn coverage_bit(dx: i16, dy: i16) -> Option<u64> {
        let x = dx.checked_add(COVERAGE_CENTER)?;
        let y = dy.checked_add(COVERAGE_CENTER)?;
        if !(0..COVERAGE_SIDE).contains(&x) || !(0..COVERAGE_SIDE).contains(&y) {
            return None;
        }
        let index = y.checked_mul(COVERAGE_SIDE)?.checked_add(x)?;
        Some(1_u64 << u32::try_from(index).ok()?)
    }

    fn mark_visited(&mut self, dx: i16, dy: i16) {
        if let Some(bit) = Self::coverage_bit(dx, dy) {
            self.coverage |= bit;
        }
    }

    fn was_visited(&self, dx: i16, dy: i16) -> Option<bool> {
        Self::coverage_bit(dx, dy).map(|bit| self.coverage & bit != 0)
    }

    fn shift_coverage(&mut self, movement_dx: i8, movement_dy: i8) {
        let shift_x = movement_dx.unsigned_abs();
        let shift_y = movement_dy.unsigned_abs();
        if i16::from(shift_x) >= COVERAGE_SIDE || i16::from(shift_y) >= COVERAGE_SIDE {
            self.coverage = 0;
            self.mark_visited(0, 0);
            return;
        }

        let mut shifted = 0_u64;
        for source_y in 0..COVERAGE_SIDE {
            let destination_y = source_y - i16::from(movement_dy);
            if !(0..COVERAGE_SIDE).contains(&destination_y) {
                continue;
            }
            let row = ((self.coverage >> u32::try_from(source_y * COVERAGE_SIDE).unwrap())
                & u64::from(u8::MAX)) as u8;
            let shifted_row = if movement_dx >= 0 {
                row >> u32::from(shift_x)
            } else {
                row << u32::from(shift_x)
            };
            shifted |=
                u64::from(shifted_row) << u32::try_from(destination_y * COVERAGE_SIDE).unwrap();
        }
        self.coverage = shifted;
        self.mark_visited(0, 0);
    }

    fn record_family_outcome(&mut self, family: EstimateFamily, status: ReferenceOutcomeStatus) {
        let index = family as usize;
        if self.outcome_successes[index] == u8::MAX
            || self.outcome_setbacks[index] == u8::MAX
            || self.outcome_contentions[index] == u8::MAX
        {
            self.outcome_successes[index] /= 2;
            self.outcome_setbacks[index] /= 2;
            self.outcome_contentions[index] /= 2;
        }
        match status {
            ReferenceOutcomeStatus::Success => {
                self.outcome_successes[index] = self.outcome_successes[index].saturating_add(1);
            }
            ReferenceOutcomeStatus::Frustrated | ReferenceOutcomeStatus::Contested => {
                self.outcome_contentions[index] = self.outcome_contentions[index].saturating_add(1);
            }
            ReferenceOutcomeStatus::Interrupted | ReferenceOutcomeStatus::Rejected => {
                self.outcome_setbacks[index] = self.outcome_setbacks[index].saturating_add(1);
            }
        }
    }

    fn family_samples(&self, family: EstimateFamily) -> u16 {
        let index = family as usize;
        u16::from(self.outcome_successes[index])
            .saturating_add(u16::from(self.outcome_setbacks[index]))
            .saturating_add(u16::from(self.outcome_contentions[index]))
    }

    fn reliability_q8(&self, family: EstimateFamily) -> u16 {
        let index = family as usize;
        let successes = u16::from(self.outcome_successes[index]).saturating_add(1);
        let attempts = self.family_samples(family).saturating_add(2);
        successes.saturating_mul(256) / attempts.max(1)
    }

    fn contention_q8(&self, family: EstimateFamily) -> u16 {
        let index = family as usize;
        let contentions = u16::from(self.outcome_contentions[index]).saturating_add(1);
        let attempts = self.family_samples(family).saturating_add(4);
        contentions.saturating_mul(256) / attempts.max(1)
    }

    fn update_ema(target: &mut u16, samples: &mut u8, observation: u64) {
        let observation = observation.min(u64::from(u16::MAX)) as u16;
        *target = if *samples == 0 {
            observation
        } else {
            let weighted = u32::from(*target)
                .saturating_mul(3)
                .saturating_add(u32::from(observation));
            u16::try_from(weighted / 4).unwrap_or(u16::MAX)
        };
        *samples = samples.saturating_add(1);
    }

    fn observe_interval(&mut self, input: &ReferenceMindInput) {
        if !self.snapshot_valid {
            return;
        }
        let energy_loss = self
            .snapshot_assimilated_energy
            .saturating_sub(input.self_state.assimilated_energy);
        if self.snapshot_guarded {
            Self::update_ema(
                &mut self.guarded_loss_ema,
                &mut self.guarded_loss_samples,
                energy_loss,
            );
        } else {
            Self::update_ema(
                &mut self.unguarded_loss_ema,
                &mut self.unguarded_loss_samples,
                energy_loss,
            );
        }

        if self.snapshot_gut_energy > 0 {
            let digested = self
                .snapshot_gut_energy
                .saturating_sub(input.self_state.gut_energy);
            Self::update_ema(
                &mut self.digestion_progress_ema,
                &mut self.digestion_samples,
                digested,
            );
        }

        if self.pending_signal != NO_PENDING_SIGNAL && self.pending != PendingAction::Move {
            let channel = usize::from(self.pending_signal);
            if channel < input.current_tile.signal_energy.len() {
                let expected = self.expected_signal_energy.max(1);
                let retained_q8 = u128::from(input.current_tile.signal_energy[channel])
                    .saturating_mul(256)
                    / u128::from(expected);
                Self::update_ema(
                    &mut self.signal_retention_q8,
                    &mut self.signal_samples,
                    u64::try_from(retained_q8.min(1024)).unwrap_or(1024),
                );
            }
        }
    }

    fn prepare_snapshot(
        &mut self,
        input: &ReferenceMindInput,
        signal: Option<ReferenceSignalEmission>,
    ) {
        self.snapshot_assimilated_energy = input.self_state.assimilated_energy;
        self.snapshot_gut_energy = input.self_state.gut_energy;
        self.snapshot_guarded = input.self_state.guarded;
        self.pending_signal = signal.map_or(NO_PENDING_SIGNAL, |emission| emission.channel);
        self.expected_signal_energy = signal.map_or(0, |emission| {
            input.current_tile.signal_energy[usize::from(emission.channel)]
                .saturating_add(emission.amount)
        });
        self.snapshot_valid = true;
    }

    fn decode(bytes: &[u8], fallback_role: Role) -> Self {
        if bytes.len() < HEADER_BYTES || bytes.get(0..4) != Some(MAGIC.as_slice()) {
            return Self::new(fallback_role);
        }
        let mut memory = Self::new(Role::from_byte(bytes[4]).unwrap_or(fallback_role));
        memory.pending = PendingAction::from_byte(bytes[5]);
        memory.pending_dx = bytes[6] as i8;
        memory.pending_dy = bytes[7] as i8;
        memory.x = read_i16(bytes, 8);
        memory.y = read_i16(bytes, 10);
        memory.route_cursor = bytes[12];
        memory.pending_plant_index = bytes[13];
        memory.child_cursor = bytes[14];
        memory.threat_score = bytes[15];
        memory.decisions = read_u16(bytes, 16);
        memory.move_successes = read_u16(bytes, 18);
        memory.move_failures = read_u16(bytes, 20);
        memory.min_growth = read_u16(bytes, 22);
        memory.max_growth = read_u16(bytes, 24);
        memory.max_capacity = read_u16(bytes, 26);
        memory.pending_construction_phase = ConstructionPhase::from_byte(bytes[29]);
        memory.coverage = read_u64(bytes, 32);
        memory.outcome_successes.copy_from_slice(&bytes[40..48]);
        memory.outcome_setbacks.copy_from_slice(&bytes[48..56]);
        memory.outcome_contentions.copy_from_slice(&bytes[56..64]);
        memory.snapshot_assimilated_energy = read_u64(bytes, 64);
        memory.snapshot_gut_energy = read_u64(bytes, 72);
        memory.expected_signal_energy = read_u64(bytes, 80);
        memory.guarded_loss_ema = read_u16(bytes, 88);
        memory.unguarded_loss_ema = read_u16(bytes, 90);
        memory.digestion_progress_ema = read_u16(bytes, 92);
        memory.signal_retention_q8 = read_u16(bytes, 94);
        memory.guarded_loss_samples = bytes[96];
        memory.unguarded_loss_samples = bytes[97];
        memory.digestion_samples = bytes[98];
        memory.signal_samples = bytes[99];
        memory.pending_signal = bytes[100];
        memory.snapshot_valid = bytes[101] != 0;
        memory.snapshot_guarded = bytes[102] != 0;
        let count = usize::from(bytes[28]).min(MAX_MAP_ENTRIES);
        for index in 0..count {
            let offset = HEADER_BYTES + index * MAP_ENTRY_BYTES;
            if offset + MAP_ENTRY_BYTES > bytes.len() {
                break;
            }
            memory.plants.push(PlantRecord {
                x: read_i16(bytes, offset),
                y: read_i16(bytes, offset + 2),
                capacity: read_u16(bytes, offset + 4),
                growth: read_u16(bytes, offset + 6),
                wall_cursor: bytes[offset + 8],
                wall_layer: bytes[offset + 9],
                pit_pressure: bytes[offset + 10],
            });
        }
        memory
    }

    fn encode(&self, maximum: usize) -> Option<Vec<u8>> {
        if maximum < HEADER_BYTES {
            return None;
        }
        let entry_count = self
            .plants
            .len()
            .min(MAX_MAP_ENTRIES)
            .min((maximum - HEADER_BYTES) / MAP_ENTRY_BYTES);
        let mut bytes = vec![0; HEADER_BYTES + entry_count * MAP_ENTRY_BYTES];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = self.role as u8;
        bytes[5] = self.pending as u8;
        bytes[6] = self.pending_dx as u8;
        bytes[7] = self.pending_dy as u8;
        write_i16(&mut bytes, 8, self.x);
        write_i16(&mut bytes, 10, self.y);
        bytes[12] = self.route_cursor;
        bytes[13] = self.pending_plant_index;
        bytes[14] = self.child_cursor;
        bytes[15] = self.threat_score;
        write_u16(&mut bytes, 16, self.decisions);
        write_u16(&mut bytes, 18, self.move_successes);
        write_u16(&mut bytes, 20, self.move_failures);
        write_u16(&mut bytes, 22, self.min_growth);
        write_u16(&mut bytes, 24, self.max_growth);
        write_u16(&mut bytes, 26, self.max_capacity);
        bytes[28] = entry_count as u8;
        bytes[29] = self.pending_construction_phase as u8;
        write_u64(&mut bytes, 32, self.coverage);
        bytes[40..48].copy_from_slice(&self.outcome_successes);
        bytes[48..56].copy_from_slice(&self.outcome_setbacks);
        bytes[56..64].copy_from_slice(&self.outcome_contentions);
        write_u64(&mut bytes, 64, self.snapshot_assimilated_energy);
        write_u64(&mut bytes, 72, self.snapshot_gut_energy);
        write_u64(&mut bytes, 80, self.expected_signal_energy);
        write_u16(&mut bytes, 88, self.guarded_loss_ema);
        write_u16(&mut bytes, 90, self.unguarded_loss_ema);
        write_u16(&mut bytes, 92, self.digestion_progress_ema);
        write_u16(&mut bytes, 94, self.signal_retention_q8);
        bytes[96] = self.guarded_loss_samples;
        bytes[97] = self.unguarded_loss_samples;
        bytes[98] = self.digestion_samples;
        bytes[99] = self.signal_samples;
        bytes[100] = self.pending_signal;
        bytes[101] = u8::from(self.snapshot_valid);
        bytes[102] = u8::from(self.snapshot_guarded);
        for (index, plant) in self.plants.iter().take(entry_count).enumerate() {
            let offset = HEADER_BYTES + index * MAP_ENTRY_BYTES;
            write_i16(&mut bytes, offset, plant.x);
            write_i16(&mut bytes, offset + 2, plant.y);
            write_u16(&mut bytes, offset + 4, plant.capacity);
            write_u16(&mut bytes, offset + 6, plant.growth);
            bytes[offset + 8] = plant.wall_cursor;
            bytes[offset + 9] = plant.wall_layer;
            bytes[offset + 10] = plant.pit_pressure;
        }
        Some(bytes)
    }

    fn reconcile_previous_action(&mut self, input: &ReferenceMindInput) {
        self.observe_interval(input);
        let Some(outcome) = input.self_state.last_outcome else {
            self.pending = PendingAction::None;
            self.pending_signal = NO_PENDING_SIGNAL;
            self.pending_plant_index = NO_PENDING_PLANT;
            self.pending_construction_phase = ConstructionPhase::None;
            return;
        };
        if let Some(family) = self.pending.estimate_family() {
            self.record_family_outcome(family, outcome.status);
        }
        if self.pending_signal != NO_PENDING_SIGNAL {
            self.record_family_outcome(EstimateFamily::Signal, ReferenceOutcomeStatus::Success);
        }
        if outcome.status == ReferenceOutcomeStatus::Success {
            if self.pending == PendingAction::Move {
                self.move_successes = self.move_successes.saturating_add(1);
                self.shift_coverage(self.pending_dx, self.pending_dy);
                self.x = self.x.saturating_add(i16::from(self.pending_dx));
                self.y = self.y.saturating_add(i16::from(self.pending_dy));
            } else if self.pending == PendingAction::Deposit
                && self.pending_construction_phase == ConstructionPhase::Wall
            {
                self.advance_pending_wall();
            }
        } else {
            if self.pending == PendingAction::Move {
                self.move_failures = self.move_failures.saturating_add(1);
            }
            if self.pending == PendingAction::Deposit
                && outcome.rejected_reason == Some(ReferenceRejectReason::TerrainLimit)
                && self.pending_construction_phase == ConstructionPhase::Wall
            {
                self.advance_pending_wall();
            }
        }
        if self.pending_construction_phase == ConstructionPhase::Source
            && matches!(self.pending, PendingAction::Move | PendingAction::Excavate)
            && let Some(plant) = self.pending_plant_mut()
        {
            if matches!(
                outcome.status,
                ReferenceOutcomeStatus::Frustrated | ReferenceOutcomeStatus::Contested
            ) {
                plant.pit_pressure = plant.pit_pressure.saturating_add(1);
            } else if outcome.status == ReferenceOutcomeStatus::Success {
                plant.pit_pressure = plant.pit_pressure.saturating_sub(1);
            }
        }
        if self.pending == PendingAction::Attack
            && outcome.status != ReferenceOutcomeStatus::Success
        {
            self.threat_score = self.threat_score.saturating_add(1);
        } else {
            self.threat_score = self.threat_score.saturating_sub(1);
        }
        self.pending = PendingAction::None;
        self.pending_signal = NO_PENDING_SIGNAL;
        self.pending_plant_index = NO_PENDING_PLANT;
        self.pending_construction_phase = ConstructionPhase::None;
        self.pending_dx = 0;
        self.pending_dy = 0;
    }

    fn pending_plant_mut(&mut self) -> Option<&mut PlantRecord> {
        (self.pending_plant_index != NO_PENDING_PLANT)
            .then_some(usize::from(self.pending_plant_index))
            .and_then(|index| self.plants.get_mut(index))
    }

    fn advance_pending_wall(&mut self) {
        if let Some(plant) = self.pending_plant_mut() {
            advance_wall_progress(plant);
        }
    }

    fn observe_plant(&mut self, x: i16, y: i16, capacity: u64, growth: u64) {
        let capacity = capacity.min(u64::from(u16::MAX)) as u16;
        let growth = growth.min(u64::from(u16::MAX)) as u16;
        if growth > 0 {
            self.min_growth = if self.min_growth == 0 {
                growth
            } else {
                self.min_growth.min(growth)
            };
            self.max_growth = self.max_growth.max(growth);
        }
        self.max_capacity = self.max_capacity.max(capacity);
        if let Some(existing) = self
            .plants
            .iter_mut()
            .find(|plant| plant.x == x && plant.y == y)
        {
            existing.capacity = capacity;
            existing.growth = growth;
            return;
        }
        let record = PlantRecord {
            x,
            y,
            capacity,
            growth,
            wall_cursor: 0,
            wall_layer: 0,
            pit_pressure: 0,
        };
        if self.plants.len() < MAX_MAP_ENTRIES {
            self.plants.push(record);
        } else {
            let replacement = usize::from(self.route_cursor) % self.plants.len();
            self.plants[replacement] = record;
        }
    }

    fn update_observations(&mut self, input: &ReferenceMindInput) {
        if is_plant(
            input.current_tile.plant_capacity,
            input.current_tile.plant_growth_rate,
        ) {
            self.observe_plant(
                self.x,
                self.y,
                input.current_tile.plant_capacity,
                input.current_tile.plant_growth_rate,
            );
        }
        for slot in &input.slots {
            if is_plant(
                slot.plant_capacity.unwrap_or(0),
                slot.plant_growth_rate.unwrap_or(0),
            ) {
                self.observe_plant(
                    self.x.saturating_add(i16::from(slot.dx)),
                    self.y.saturating_add(i16::from(slot.dy)),
                    slot.plant_capacity.unwrap_or(0),
                    slot.plant_growth_rate.unwrap_or(0),
                );
            }
        }
    }

    fn planning_plant(&self) -> Option<PlantRecord> {
        self.plants.iter().copied().min_by_key(|plant| {
            let dx = i64::from(plant.x) - i64::from(self.x);
            let dy = i64::from(plant.y) - i64::from(self.y);
            let distance = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            // Prefer productive plants without sending a cell arbitrarily far
            // away. These rates are learned from local observations.
            distance.saturating_mul(i64::from(self.max_growth.max(1)))
                / i64::from(plant.growth.max(1))
        })
    }

    fn planning_build_plant_index(&self) -> Option<usize> {
        self.plants
            .iter()
            .enumerate()
            .filter(|(_, plant)| plant.wall_layer < 2)
            .min_by_key(|(_, plant)| {
                let dx = i64::from(plant.x) - i64::from(self.x);
                let dy = i64::from(plant.y) - i64::from(self.y);
                let distance = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
                distance.saturating_mul(i64::from(self.max_growth.max(1)))
                    / i64::from(plant.growth.max(1))
            })
            .map(|(index, _)| index)
    }

    fn is_known_plant_at(&self, x: i16, y: i16) -> bool {
        self.plants.iter().any(|plant| plant.x == x && plant.y == y)
    }

    fn record_construction_phase(&mut self, plant_index: usize, phase: ConstructionPhase) {
        self.pending_plant_index = u8::try_from(plant_index).unwrap_or(NO_PENDING_PLANT);
        self.pending_construction_phase = phase;
    }
}

fn advance_wall_progress(plant: &mut PlantRecord) {
    if usize::from(plant.wall_cursor) % 8 == 7 {
        plant.wall_layer = plant.wall_layer.saturating_add(1);
    }
    plant.wall_cursor = plant.wall_cursor.wrapping_add(1);
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    bytes
        .get(offset..offset + 2)
        .and_then(|value| value.try_into().ok())
        .map(u16::from_le_bytes)
        .unwrap_or(0)
}

fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    bytes
        .get(offset..offset + 2)
        .and_then(|value| value.try_into().ok())
        .map(i16::from_le_bytes)
        .unwrap_or(0)
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .unwrap_or(0)
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    if let Some(destination) = bytes.get_mut(offset..offset + 2) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
}

fn write_i16(bytes: &mut [u8], offset: usize, value: i16) {
    if let Some(destination) = bytes.get_mut(offset..offset + 2) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    if let Some(destination) = bytes.get_mut(offset..offset + 8) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
}

fn is_plant(capacity: u64, growth: u64) -> bool {
    capacity > 0 || growth > 0
}

fn initial_role(input: &ReferenceMindInput) -> Role {
    if let Some(role) = Role::from_marker(input.self_state.marker) {
        return role;
    }
    if is_plant(
        input.current_tile.plant_capacity,
        input.current_tile.plant_growth_rate,
    ) {
        return Role::Feeder;
    }
    match input.randomness.sample_u64(0) % 20 {
        0..=4 => Role::Feeder,
        5..=10 => Role::Explorer,
        11..=13 => Role::Builder,
        14..=16 => Role::Defender,
        _ => Role::Attacker,
    }
}

fn preferred_standard(input: &ReferenceMindInput) -> ReferenceEffort {
    preferred_effort(
        &input.action_space,
        &[
            ReferenceEffort::Standard,
            ReferenceEffort::Gentle,
            ReferenceEffort::Burst,
        ],
    )
}

fn preferred_gentle(input: &ReferenceMindInput) -> ReferenceEffort {
    preferred_effort(
        &input.action_space,
        &[
            ReferenceEffort::Gentle,
            ReferenceEffort::Standard,
            ReferenceEffort::Burst,
        ],
    )
}

fn travel_effort(input: &ReferenceMindInput, memory: &ColonyMemory) -> ReferenceEffort {
    let learned_contention = memory.family_samples(EstimateFamily::Move) >= 4
        && memory.contention_q8(EstimateFamily::Move) >= 96;
    if learned_contention
        || memory.move_failures > memory.move_successes.saturating_div(2).saturating_add(3)
    {
        preferred_standard(input)
    } else {
        preferred_gentle(input)
    }
}

fn move_toward(
    input: &ReferenceMindInput,
    target_x: i16,
    target_y: i16,
    memory: &ColonyMemory,
    effort: ReferenceEffort,
) -> Option<ReferenceMindAction> {
    let target_x = i32::from(target_x);
    let target_y = i32::from(target_y);
    let current_x = i32::from(memory.x);
    let current_y = i32::from(memory.y);
    let current_distance = (target_x - current_x).pow(2) + (target_y - current_y).pow(2);
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(input.action_space.move_targets, slot.slot)
                && slot.elevation.is_none_or(|elevation| {
                    (i32::from(elevation) - i32::from(input.current_tile.elevation)).abs() <= 1
                })
        })
        .min_by_key(|slot| {
            let x = current_x + i32::from(slot.dx);
            let y = current_y + i32::from(slot.dy);
            (target_x - x).pow(2) + (target_y - y).pow(2)
        })
        .filter(|slot| {
            let x = current_x + i32::from(slot.dx);
            let y = current_y + i32::from(slot.dy);
            (target_x - x).pow(2) + (target_y - y).pow(2) < current_distance
        })
        .map(|slot| ReferenceMindAction::Move {
            target_slot: slot.slot,
            effort,
        })
}

fn move_along_signal(
    input: &ReferenceMindInput,
    channel: u8,
    effort: ReferenceEffort,
) -> Option<ReferenceMindAction> {
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.reachable
                && slot.neighbor.is_none()
                && ReferenceActionSpace::allows_target(input.action_space.move_targets, slot.slot)
        })
        .filter_map(|slot| {
            slot.signal_energy
                .map(|signals| (slot.slot, signals[usize::from(channel)]))
        })
        .filter(|(_, energy)| *energy > input.current_tile.signal_energy[usize::from(channel)])
        .max_by_key(|(slot, energy)| (*energy, std::cmp::Reverse(*slot)))
        .map(|(target_slot, _)| ReferenceMindAction::Move {
            target_slot,
            effort,
        })
}

fn exploration_move(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> ReferenceMindAction {
    if let Some(target_slot) = best_energy_slot(input, input.action_space.move_targets) {
        return ReferenceMindAction::Move {
            target_slot,
            effort: preferred_gentle(input),
        };
    }
    let targets = safe_empty_slots(input, input.action_space.move_targets);
    if targets.is_empty() {
        return ReferenceMindAction::Wait;
    }
    let has_unvisited = input
        .slots
        .iter()
        .filter(|slot| targets.contains(&slot.slot))
        .any(|slot| memory.was_visited(i16::from(slot.dx), i16::from(slot.dy)) != Some(true));
    // A rotating preference is a cheap local approximation of a spiral sweep.
    let preferred = usize::from(memory.route_cursor) % input.slots.len().max(1);
    memory.route_cursor = memory.route_cursor.wrapping_add(1);
    let slot = input
        .slots
        .iter()
        .filter(|slot| targets.contains(&slot.slot))
        .filter(|slot| {
            !has_unvisited
                || memory.was_visited(i16::from(slot.dx), i16::from(slot.dy)) != Some(true)
        })
        .min_by_key(|slot| usize::from(slot.slot).abs_diff(preferred))
        .map(|slot| slot.slot)
        .or_else(|| choose_slot(&targets, input.randomness.sample_u64(2)))
        .expect("non-empty target set");
    ReferenceMindAction::Move {
        target_slot: slot,
        effort: preferred_gentle(input),
    }
}

fn consume_if_useful(
    input: &ReferenceMindInput,
    memory: &ColonyMemory,
) -> Option<ReferenceMindAction> {
    let mut amount = input.action_space.max_consume_amount;
    if memory.digestion_samples >= 4 && memory.digestion_progress_ema > 0 {
        amount = amount.min(u64::from(memory.digestion_progress_ema).saturating_mul(4));
    }
    (input.action_space.consume_enabled
        && amount > 0
        && current_food(input) > 0
        && input.self_state.gut_energy < input.action_space.gut_capacity)
        .then_some(ReferenceMindAction::Consume { amount })
}

fn learned_attack_divisor(memory: &ColonyMemory, baseline: u64) -> u64 {
    if memory.family_samples(EstimateFamily::Attack) < 4 {
        return baseline;
    }
    if memory.contention_q8(EstimateFamily::Attack) >= 96
        || memory.reliability_q8(EstimateFamily::Attack) < 128
    {
        baseline.saturating_mul(2)
    } else if memory.reliability_q8(EstimateFamily::Attack) >= 208 {
        baseline.saturating_mul(3).saturating_div(4).max(4)
    } else {
        baseline
    }
}

fn local_signal_level(input: &ReferenceMindInput, channel: u8) -> u64 {
    input
        .slots
        .iter()
        .filter_map(|slot| slot.signal_energy)
        .map(|signals| signals[usize::from(channel)])
        .chain(std::iter::once(
            input.current_tile.signal_energy[usize::from(channel)],
        ))
        .max()
        .unwrap_or(0)
}

fn signal_bucket(input: &ReferenceMindInput, channel: u8) -> u16 {
    let level = local_signal_level(input, channel);
    if level == 0 {
        0
    } else {
        u16::try_from(level.ilog2().saturating_add(1).min(8)).unwrap_or(8)
    }
}

fn local_role_counts(input: &ReferenceMindInput, own_role: Role) -> [u16; 5] {
    // Markers are deliberately only hints: an opponent can spoof these counts.
    // Capping their influence keeps one crowded or adversarial neighborhood
    // from permanently suppressing a caste.
    let mut counts = [0_u16; 5];
    counts[own_role as usize] = 1;
    for role in input
        .slots
        .iter()
        .filter_map(|slot| slot.neighbor.and_then(|neighbor| neighbor.marker))
        .filter_map(Role::from_marker)
    {
        counts[role as usize] = counts[role as usize].saturating_add(1).min(3);
    }
    counts
}

fn visible_wall_gaps(input: &ReferenceMindInput) -> u16 {
    if !is_plant(
        input.current_tile.plant_capacity,
        input.current_tile.plant_growth_rate,
    ) {
        return 0;
    }
    input
        .slots
        .iter()
        // The south cardinal is the deliberate relief gate and must not keep
        // requesting builders after the other seven positions are raised.
        .filter(|slot| !(slot.dx == 0 && slot.dy == 1))
        .filter(|slot| {
            slot.elevation.is_some_and(|elevation| {
                i32::from(elevation) - i32::from(input.current_tile.elevation) <= 1
            })
        })
        .count()
        .min(8) as u16
}

fn child_role(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> Role {
    // This is a local deficit heuristic, not a population census. It combines
    // private knowledge with the current neighborhood and anonymous fields.
    // Scores are intentionally small and capped so lossy or spoofed evidence
    // cannot dominate child allocation forever.
    let mut scores = [12_u16, 12, 4, 5, 2];
    let local_roles = local_role_counts(input, memory.role);

    let plant_here = is_plant(
        input.current_tile.plant_capacity,
        input.current_tile.plant_growth_rate,
    );
    let plant_visible = plant_here
        || input.slots.iter().any(|slot| {
            is_plant(
                slot.plant_capacity.unwrap_or(0),
                slot.plant_growth_rate.unwrap_or(0),
            )
        });
    let has_plant_knowledge = !memory.plants.is_empty();
    if has_plant_knowledge {
        scores[Role::Feeder as usize] = scores[Role::Feeder as usize].saturating_add(3);
        scores[Role::Builder as usize] = scores[Role::Builder as usize].saturating_add(6);
        scores[Role::Defender as usize] = scores[Role::Defender as usize].saturating_add(4);
    } else {
        scores[Role::Explorer as usize] = scores[Role::Explorer as usize].saturating_add(18);
    }
    if plant_visible {
        scores[Role::Feeder as usize] = scores[Role::Feeder as usize].saturating_add(8);
        scores[Role::Builder as usize] = scores[Role::Builder as usize]
            .saturating_add(6 + visible_wall_gaps(input).saturating_mul(2));
        scores[Role::Defender as usize] = scores[Role::Defender as usize].saturating_add(8);
    }
    if local_signal_level(input, PLANT_SIGNAL) > 0 {
        scores[Role::Feeder as usize] = scores[Role::Feeder as usize].saturating_add(4);
    }
    if local_signal_level(input, FRONTIER_SIGNAL) == 0 {
        scores[Role::Explorer as usize] = scores[Role::Explorer as usize].saturating_add(4);
    }
    if local_signal_level(input, BUILD_SIGNAL) > 0 {
        scores[Role::Defender as usize] = scores[Role::Defender as usize].saturating_add(4);
        scores[Role::Builder as usize] = scores[Role::Builder as usize].saturating_sub(4);
    }

    let hostile_neighbors = input
        .slots
        .iter()
        .filter(|slot| slot.neighbor.is_some() && !known_colony_marker(slot))
        .count()
        .min(4) as u16;
    let threat = signal_bucket(input, THREAT_SIGNAL);
    scores[Role::Defender as usize] = scores[Role::Defender as usize]
        .saturating_add(threat.saturating_mul(2))
        .saturating_add(hostile_neighbors.saturating_mul(4));
    scores[Role::Attacker as usize] = scores[Role::Attacker as usize]
        .saturating_add(threat.saturating_mul(4))
        .saturating_add(hostile_neighbors.saturating_mul(6));
    if memory.guarded_loss_samples >= 4 && memory.unguarded_loss_samples >= 4 {
        if u32::from(memory.guarded_loss_ema).saturating_mul(4)
            <= u32::from(memory.unguarded_loss_ema).saturating_mul(3)
        {
            scores[Role::Defender as usize] = scores[Role::Defender as usize].saturating_add(3);
        } else {
            scores[Role::Attacker as usize] = scores[Role::Attacker as usize].saturating_add(2);
        }
    }

    // Apply the observed local supply after contextual demand so the deficit
    // is meaningful even for strongly requested infrastructure roles.
    for (score, count) in scores.iter_mut().zip(local_roles) {
        *score = score.saturating_sub(count.saturating_mul(9));
    }

    if !input.action_space.consume_enabled {
        scores[Role::Feeder as usize] /= 2;
    }
    if input.action_space.terrain_mass_per_elevation == 0 {
        scores[Role::Builder as usize] = 0;
    } else if !input.action_space.excavate_enabled {
        // Availability is state-local (for example at the minimum elevation),
        // not proof that the ruleset lacks this action family entirely.
        scores[Role::Builder as usize] /= 2;
    }
    if !(input.action_space.guard_enabled || input.action_space.attack_targets != 0) {
        scores[Role::Defender as usize] = 0;
    }
    if input.action_space.attack_targets == 0 {
        scores[Role::Attacker as usize] = 0;
    }

    let tie_start = usize::from(memory.child_cursor) % scores.len();
    let role = (0..scores.len())
        .max_by_key(|index| {
            let tie_distance = (index + scores.len() - tie_start) % scores.len();
            (scores[*index], std::cmp::Reverse(tie_distance))
        })
        .and_then(|index| Role::from_byte(index as u8))
        .unwrap_or(Role::Explorer);
    memory.child_cursor = memory.child_cursor.wrapping_add(1);
    role
}

fn try_split(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> Option<ReferenceMindAction> {
    let minimum = input
        .action_space
        .minimum_survival_energy
        .saturating_add(input.action_space.child_core_mass);
    if input.self_state.assimilated_energy < minimum.saturating_mul(4) {
        return None;
    }
    let targets = safe_empty_slots(input, input.action_space.split_targets);
    let target_slot = choose_slot(&targets, input.randomness.sample_u64(3))?;
    let child_allocation = split_allocation(input)?;
    let target = input.slots.iter().find(|slot| slot.slot == target_slot)?;
    let role = child_role(input, memory);
    let mut child = memory.clone();
    child.role = role;
    child.x = child.x.saturating_add(i16::from(target.dx));
    child.y = child.y.saturating_add(i16::from(target.dy));
    child.shift_coverage(target.dx, target.dy);
    child.pending = PendingAction::None;
    child.pending_dx = 0;
    child.pending_dy = 0;
    child.pending_signal = NO_PENDING_SIGNAL;
    child.pending_plant_index = NO_PENDING_PLANT;
    child.pending_construction_phase = ConstructionPhase::None;
    child.snapshot_valid = false;
    child.child_cursor = child.child_cursor.wrapping_add(3);
    let private_memory = child.encode(input.action_space.max_private_memory_bytes as usize)?;
    Some(ReferenceMindAction::Split {
        target_slot,
        child_allocation,
        marker: role.marker(),
        private_memory,
    })
}

fn known_colony_marker(slot: &LocalObservation) -> bool {
    slot.neighbor
        .and_then(|neighbor| neighbor.marker)
        .and_then(Role::from_marker)
        .is_some()
}

fn adjacent_target(input: &ReferenceMindInput) -> Option<u8> {
    input
        .slots
        .iter()
        .filter(|slot| {
            slot.neighbor.is_some()
                && !known_colony_marker(slot)
                && slot.reachable
                && ReferenceActionSpace::allows_target(input.action_space.attack_targets, slot.slot)
        })
        .max_by_key(|slot| {
            let windup = slot
                .neighbor
                .and_then(|neighbor| neighbor.activity)
                .is_some_and(|activity| {
                    activity == blob_interface::reference_mind::ReferenceActivity::AttackWindup
                });
            (windup, std::cmp::Reverse(slot.slot))
        })
        .map(|slot| slot.slot)
}

fn plant_ring_offset(plant: PlantRecord) -> (i16, i16) {
    const RING: [(i16, i16); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    ];
    RING[usize::from(plant.wall_cursor) % RING.len()]
}

fn plant_ring_target(plant: PlantRecord) -> (i16, i16) {
    let offset = plant_ring_offset(plant);
    (
        plant.x.saturating_add(offset.0),
        plant.y.saturating_add(offset.1),
    )
}

fn excavation_source(plant: PlantRecord) -> (i16, i16) {
    let offset = plant_ring_offset(plant);
    let source_radius = i16::from(plant.wall_layer).saturating_add(2);
    let pressure = i16::from(plant.pit_pressure.min(3));
    if plant.wall_cursor % 8 == GATE_RING_INDEX {
        // Keep the exterior gate tile level and rotate eastward if another
        // local actor repeatedly contests the original relief pit.
        return (
            plant.x.saturating_add(1).saturating_add(pressure),
            plant.y.saturating_add(source_radius),
        );
    }
    let parity = if plant.wall_cursor.is_multiple_of(2) {
        1
    } else {
        -1
    };
    let flank_x = offset.1.saturating_neg().signum();
    let flank_y = offset.0.signum();
    (
        plant
            .x
            .saturating_add(offset.0.saturating_mul(source_radius))
            .saturating_add(flank_x.saturating_mul(pressure).saturating_mul(parity)),
        plant
            .y
            .saturating_add(offset.1.saturating_mul(source_radius))
            .saturating_add(flank_y.saturating_mul(pressure).saturating_mul(parity)),
    )
}

fn feeder_action(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> ReferenceMindAction {
    if let Some(action) = consume_if_useful(input, memory) {
        return action;
    }
    if let Some(action) = try_split(input, memory) {
        return action;
    }
    if let Some(plant) = memory.planning_plant()
        && let Some(action) = move_toward(
            input,
            plant.x,
            plant.y,
            memory,
            travel_effort(input, memory),
        )
    {
        return action;
    }
    if let Some(action) = move_along_signal(input, PLANT_SIGNAL, travel_effort(input, memory)) {
        return action;
    }
    exploration_move(input, memory)
}

fn explorer_action(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> ReferenceMindAction {
    if let Some(action) = consume_if_useful(input, memory) {
        return action;
    }
    if memory.plants.len() >= 2
        && let Some(action) = try_split(input, memory)
    {
        return action;
    }
    exploration_move(input, memory)
}

fn builder_action(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> ReferenceMindAction {
    if let Some(action) = consume_if_useful(input, memory)
        && input.self_state.assimilated_energy
            < input.action_space.minimum_survival_energy.saturating_mul(3)
    {
        return action;
    }
    let Some(plant_index) = memory.planning_build_plant_index() else {
        if let Some(action) = move_along_signal(input, PLANT_SIGNAL, travel_effort(input, memory)) {
            return action;
        }
        return exploration_move(input, memory);
    };
    // Productive tiles are never terrain sources or deposit targets. Newly
    // observed plants have already been added above, so this check protects
    // them on the same decision in which they first become visible.
    for _ in 0..8 {
        let plant = memory.plants[plant_index];
        let target = plant_ring_target(plant);
        let skip_gate = plant.wall_layer > 0 && plant.wall_cursor % 8 == GATE_RING_INDEX;
        if !skip_gate && !memory.is_known_plant_at(target.0, target.1) {
            break;
        }
        advance_wall_progress(&mut memory.plants[plant_index]);
    }
    if memory.plants[plant_index].wall_layer >= 2 {
        return exploration_move(input, memory);
    }
    let plant = memory.plants[plant_index];
    let target = plant_ring_target(plant);
    if (memory.x, memory.y) == target
        && input.self_state.carried_material_mass >= input.action_space.terrain_mass_per_elevation
    {
        if input.self_state.carried_material_mass >= input.action_space.terrain_mass_per_elevation
            && input.action_space.deposit_terrain_enabled
        {
            memory.record_construction_phase(plant_index, ConstructionPhase::Wall);
            return ReferenceMindAction::DepositTerrain;
        }
        advance_wall_progress(&mut memory.plants[plant_index]);
    }
    if input.self_state.carried_material_mass >= input.action_space.terrain_mass_per_elevation
        && let Some(action) =
            move_toward(input, target.0, target.1, memory, preferred_standard(input))
    {
        memory.record_construction_phase(plant_index, ConstructionPhase::Wall);
        return action;
    }
    if input.self_state.carried_material_mass < input.action_space.terrain_mass_per_elevation {
        let mut source = excavation_source(memory.plants[plant_index]);
        for _ in 0..4 {
            if !memory.is_known_plant_at(source.0, source.1) {
                break;
            }
            memory.plants[plant_index].pit_pressure =
                memory.plants[plant_index].pit_pressure.saturating_add(1);
            source = excavation_source(memory.plants[plant_index]);
        }
        if (memory.x, memory.y) == source
            && input.action_space.excavate_enabled
            && !is_plant(
                input.current_tile.plant_capacity,
                input.current_tile.plant_growth_rate,
            )
        {
            memory.record_construction_phase(plant_index, ConstructionPhase::Source);
            return ReferenceMindAction::Excavate;
        }
        if let Some(action) = move_toward(
            input,
            source.0,
            source.1,
            memory,
            travel_effort(input, memory),
        ) {
            memory.record_construction_phase(plant_index, ConstructionPhase::Source);
            return action;
        }
    }
    let action = move_toward(input, target.0, target.1, memory, preferred_standard(input))
        .unwrap_or_else(|| exploration_move(input, memory));
    memory.record_construction_phase(plant_index, ConstructionPhase::Wall);
    action
}

fn defender_action(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> ReferenceMindAction {
    if let Some(target_slot) = adjacent_target(input)
        && let Some(payload) = attack_payload(
            input,
            preferred_standard(input),
            learned_attack_divisor(memory, 12),
        )
    {
        return ReferenceMindAction::Attack {
            target_slot,
            effort: preferred_standard(input),
            payload,
        };
    }
    if let Some(plant) = memory.planning_plant() {
        if local_signal_level(input, THREAT_SIGNAL) > 0 {
            let gate = (plant.x, plant.y.saturating_add(1));
            if (memory.x, memory.y) == gate && input.action_space.guard_enabled {
                return ReferenceMindAction::Guard {
                    effort: preferred_standard(input),
                };
            }
            if let Some(action) =
                move_toward(input, gate.0, gate.1, memory, preferred_standard(input))
            {
                return action;
            }
        }
        let distance = (i32::from(plant.x) - i32::from(memory.x))
            .unsigned_abs()
            .max((i32::from(plant.y) - i32::from(memory.y)).unsigned_abs());
        if distance <= 1 && input.action_space.guard_enabled {
            return ReferenceMindAction::Guard {
                effort: preferred_standard(input),
            };
        }
        if let Some(action) =
            move_toward(input, plant.x, plant.y, memory, preferred_standard(input))
        {
            return action;
        }
    }
    if let Some(action) = move_along_signal(input, BUILD_SIGNAL, travel_effort(input, memory)) {
        return action;
    }
    if let Some(action) = move_along_signal(input, PLANT_SIGNAL, travel_effort(input, memory)) {
        return action;
    }
    exploration_move(input, memory)
}

fn attacker_action(input: &ReferenceMindInput, memory: &mut ColonyMemory) -> ReferenceMindAction {
    let effort = preferred_effort(
        &input.action_space,
        &[
            ReferenceEffort::Burst,
            ReferenceEffort::Standard,
            ReferenceEffort::Gentle,
        ],
    );
    if let Some(target_slot) = adjacent_target(input)
        && let Some(payload) = attack_payload(input, effort, learned_attack_divisor(memory, 8))
    {
        return ReferenceMindAction::Attack {
            target_slot,
            effort,
            payload,
        };
    }
    if let Some((slot, _)) = input
        .slots
        .iter()
        .filter_map(|slot| slot.signal_energy.map(|signals| (slot, signals)))
        .filter(|(_, signals)| signals[usize::from(THREAT_SIGNAL)] > 0)
        .max_by_key(|(_, signals)| signals[usize::from(THREAT_SIGNAL)])
        && slot.neighbor.is_none()
        && ReferenceActionSpace::allows_target(input.action_space.move_targets, slot.slot)
    {
        return ReferenceMindAction::Move {
            target_slot: slot.slot,
            effort,
        };
    }
    exploration_move(input, memory)
}

fn signal_quanta(channel: u8) -> u64 {
    match channel {
        PLANT_SIGNAL => PLANT_SIGNAL_QUANTA,
        THREAT_SIGNAL => THREAT_SIGNAL_QUANTA,
        BUILD_SIGNAL => BUILD_SIGNAL_QUANTA,
        FRONTIER_SIGNAL => FRONTIER_SIGNAL_QUANTA,
        _ => 0,
    }
}

fn scaled_signal_amount(
    input: &ReferenceMindInput,
    channel: u8,
    policy: SignalPolicy,
) -> Option<u64> {
    input
        .action_space
        .signal_enabled
        .then_some(input.action_space.signal_emission_cost)
        .filter(|quantum| *quantum > 0)?
        .checked_mul(if policy.uses_semantic_strength() {
            signal_quanta(channel)
        } else {
            1
        })
        .filter(|amount| *amount > 0)
}

fn desired_signals(
    input: &ReferenceMindInput,
    memory: &ColonyMemory,
    policy: SignalPolicy,
) -> [u64; 4] {
    let mut amounts = [0_u64; 4];
    if !input.action_space.signal_enabled || policy == SignalPolicy::Disabled {
        return amounts;
    }
    let plant_here = is_plant(
        input.current_tile.plant_capacity,
        input.current_tile.plant_growth_rate,
    );
    let hostile_neighbor = adjacent_target(input).is_some();
    if hostile_neighbor {
        amounts[usize::from(THREAT_SIGNAL)] =
            scaled_signal_amount(input, THREAT_SIGNAL, policy).unwrap_or(0);
    }
    let periodic_signal_due = policy.periodic_signal_due(memory.decisions);
    if periodic_signal_due && plant_here && matches!(memory.role, Role::Feeder | Role::Explorer) {
        amounts[usize::from(PLANT_SIGNAL)] =
            scaled_signal_amount(input, PLANT_SIGNAL, policy).unwrap_or(0);
    }
    if periodic_signal_due
        && memory.role == Role::Builder
        && memory.planning_build_plant_index().is_some()
    {
        amounts[usize::from(BUILD_SIGNAL)] =
            scaled_signal_amount(input, BUILD_SIGNAL, policy).unwrap_or(0);
    }
    if memory.role == Role::Explorer && memory.decisions.is_multiple_of(8) {
        amounts[usize::from(FRONTIER_SIGNAL)] =
            scaled_signal_amount(input, FRONTIER_SIGNAL, policy).unwrap_or(0);
    }
    for (channel, amount) in amounts.iter_mut().enumerate() {
        if *amount > 0 && !learned_signal_is_useful(input, memory, channel as u8, *amount) {
            *amount = 0;
        }
    }
    amounts
}

fn learned_signal_is_useful(
    input: &ReferenceMindInput,
    memory: &ColonyMemory,
    channel: u8,
    desired_amount: u64,
) -> bool {
    let local_level = input.current_tile.signal_energy[usize::from(channel)];
    if local_level >= desired_amount {
        return false;
    }
    memory.signal_samples < 3
        || memory.signal_retention_q8 < 192
        || local_level < desired_amount / 2
}

fn pending_for(action: &ReferenceMindAction) -> PendingAction {
    match action {
        ReferenceMindAction::Wait
        | ReferenceMindAction::Regurgitate { .. }
        | ReferenceMindAction::Signal { .. } => PendingAction::None,
        ReferenceMindAction::Move { .. } => PendingAction::Move,
        ReferenceMindAction::Attack { .. } => PendingAction::Attack,
        ReferenceMindAction::Guard { .. } => PendingAction::Guard,
        ReferenceMindAction::Consume { .. } => PendingAction::Consume,
        ReferenceMindAction::Split { .. } => PendingAction::Split,
        ReferenceMindAction::Excavate => PendingAction::Excavate,
        ReferenceMindAction::DepositTerrain => PendingAction::Deposit,
    }
}

pub fn decide(input: &ReferenceMindInput) -> ReferenceMindDecision {
    decide_with_signal_policy(input, SignalPolicy::FullMultichannel)
}

pub fn decide_with_signal_policy(
    input: &ReferenceMindInput,
    signal_policy: SignalPolicy,
) -> ReferenceMindDecision {
    let fallback_role = initial_role(input);
    let mut memory = ColonyMemory::decode(&input.private_memory, fallback_role);
    memory.reconcile_previous_action(input);
    memory.mark_visited(0, 0);
    memory.decisions = memory.decisions.saturating_add(1);
    memory.update_observations(input);

    let mut action = match memory.role {
        Role::Feeder => feeder_action(input, &mut memory),
        Role::Explorer => explorer_action(input, &mut memory),
        Role::Builder => builder_action(input, &mut memory),
        Role::Defender => defender_action(input, &mut memory),
        Role::Attacker => attacker_action(input, &mut memory),
    };
    let requested_signals = desired_signals(input, &memory, signal_policy);
    let requested_channels = requested_signals
        .iter()
        .filter(|amount| **amount > 0)
        .count();
    let explicit_signal = ReferenceMindAction::Signal {
        amounts: requested_signals,
    };
    let mut signal = None;
    if signal_policy.allows_explicit_action()
        && requested_channels >= 2
        && action_is_commit_legal(input, &explicit_signal, false)
    {
        // A multi-channel write consumes a complete decision and never carries
        // a sidecar. Co-occurring facts are rare and valuable enough to make
        // that synchronization cost explicit in the policy.
        action = explicit_signal;
    } else {
        // When a combined write is unnecessary or unaffordable, retain the
        // primary behavior and attach the most urgent affordable fact.
        for channel in [THREAT_SIGNAL, PLANT_SIGNAL, BUILD_SIGNAL, FRONTIER_SIGNAL] {
            let amount = requested_signals[usize::from(channel)];
            if amount > 0 && action_is_commit_legal_with_signal_amount(input, &action, amount) {
                signal = Some(ReferenceSignalEmission { channel, amount });
                break;
            }
        }
    }
    let signal_amount = signal.map_or(0, |emission| emission.amount);
    if !action_is_commit_legal_with_signal_amount(input, &action, signal_amount) {
        action = legal_action_or_wait(input, action, false);
        signal = None;
    }
    memory.pending = pending_for(&action);
    if let ReferenceMindAction::Move { target_slot, .. } = action
        && let Some(slot) = input.slots.iter().find(|slot| slot.slot == target_slot)
    {
        memory.pending_dx = slot.dx;
        memory.pending_dy = slot.dy;
    }
    memory.prepare_snapshot(input, signal);
    let memory_update = memory
        .encode(input.action_space.max_private_memory_bytes as usize)
        .map(ReferenceMemoryUpdate::Replace)
        .unwrap_or(ReferenceMemoryUpdate::Retain);
    ReferenceMindDecision {
        action,
        signal,
        memory_update,
    }
}

#[cfg(target_arch = "wasm32")]
#[plugin_fn]
pub fn reference_mind_function(bytes: Vec<u8>) -> FnResult<Vec<u8>> {
    let limits = ReferenceMindLimits::default();
    let input = capnp_to_reference_mind_input(&bytes, limits)
        .map_err(|error| Error::msg(error.to_string()))?;
    let decision = decide(&input);
    Ok(reference_mind_decision_to_capnp(&decision, limits)
        .map_err(|error| Error::msg(error.to_string()))?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use blob_engine::resolution::{
        BatchReport, BoundaryRule, DurationRule, NeighborhoodSpec, OutcomeStatus, ReferenceRuleset,
        ReferenceSimulation,
    };
    use blob_interface::randomness::PrivateRandom;
    use blob_interface::reference_mind::{
        CurrentTileObservation, EFFORT_BURST_BIT, EFFORT_GENTLE_BIT, EFFORT_STANDARD_BIT,
        LocalObservation, ReferenceActionSpace, ReferenceNeighbor, ReferenceOutcome,
        ReferenceSelfState,
    };

    fn input() -> ReferenceMindInput {
        const DIRECTIONS: [(i8, i8); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
        ];
        ReferenceMindInput {
            self_state: ReferenceSelfState {
                core_mass: 10,
                assimilated_energy: 300,
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
                signal_energy: [0; 4],
            },
            slots: DIRECTIONS
                .iter()
                .enumerate()
                .map(|(slot, (dx, dy))| LocalObservation {
                    slot: slot as u8,
                    dx: *dx,
                    dy: *dy,
                    distance_cost_q10: if *dx != 0 && *dy != 0 { 1448 } else { 1024 },
                    reachable: true,
                    elevation: Some(0),
                    plant_energy: Some(0),
                    plant_capacity: Some(0),
                    plant_growth_rate: Some(0),
                    loose_energy: Some(0),
                    diffuse_energy: Some(0),
                    signal_energy: Some([0; 4]),
                    neighbor: None,
                })
                .collect(),
            action_space: ReferenceActionSpace {
                wait_enabled: true,
                guard_enabled: true,
                consume_enabled: true,
                excavate_enabled: true,
                deposit_terrain_enabled: true,
                signal_enabled: true,
                move_targets: 0xff,
                attack_targets: 0xff,
                split_targets: 0xff,
                regurgitate_targets: 0xff,
                effort_mask: EFFORT_GENTLE_BIT | EFFORT_STANDARD_BIT | EFFORT_BURST_BIT,
                max_consume_amount: 12,
                gut_capacity: 64,
                max_private_memory_bytes: 2048,
                minimum_survival_energy: 10,
                child_core_mass: 10,
                metabolism_rate_numerator: 1,
                metabolism_rate_denominator: 1024,
                terrain_mass_per_elevation: 10,
                signal_emission_cost: 1,
                effort_cost_numerators: [1, 1, 2],
                effort_cost_denominators: [2, 1, 1],
                move_effort_base: 1,
                move_mass_units_per_effort: 100,
                attack_effort_base: 2,
                guard_effort_base: 1,
                consume_effort_base: 1,
                split_effort_base: 2,
                regurgitate_effort_base: 1,
                excavate_effort_base: 2,
                deposit_terrain_effort_base: 2,
            },
            private_memory: Vec::new(),
            randomness: PrivateRandom::ZERO,
        }
    }

    fn memory_from_decision(decision: &ReferenceMindDecision) -> ColonyMemory {
        let ReferenceMemoryUpdate::Replace(bytes) = &decision.memory_update else {
            panic!("colony decision did not persist its memory");
        };
        ColonyMemory::decode(bytes, Role::Feeder)
    }

    fn scenario_rules() -> ReferenceRuleset {
        let duration = DurationRule::new(1, 0, 1);
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
            digestion_rate_numerator: 0,
            signal_decay_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        }
    }

    fn run_scenario_decision(
        simulation: &mut ReferenceSimulation,
        actor: blob_engine::resolution::CellKey,
    ) -> (ReferenceMindDecision, BatchReport) {
        let input = simulation
            .reference_mind_input(actor, PrivateRandom::ZERO)
            .unwrap();
        let decision = decide(&input);
        let receipt = simulation
            .commit_memory_update_with_signal(
                actor,
                decision.action.clone().into(),
                decision.signal,
                decision.memory_update.clone(),
            )
            .unwrap();
        assert!(receipt.accepted, "rejected colony action: {decision:?}");
        let report = simulation.resolve_next_batch().unwrap();
        assert_eq!(report.outcomes.len(), 1);
        (decision, report)
    }

    #[test]
    fn memory_round_trip_preserves_role_position_and_map() {
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.x = -7;
        memory.y = 12;
        memory.pending_plant_index = 0;
        memory.pending_construction_phase = ConstructionPhase::Wall;
        memory.mark_visited(-2, 3);
        memory.outcome_successes[EstimateFamily::Attack as usize] = 7;
        memory.outcome_contentions[EstimateFamily::Move as usize] = 3;
        memory.snapshot_assimilated_energy = 900;
        memory.snapshot_gut_energy = 12;
        memory.expected_signal_energy = 4;
        memory.guarded_loss_ema = 6;
        memory.unguarded_loss_ema = 19;
        memory.digestion_progress_ema = 3;
        memory.signal_retention_q8 = 220;
        memory.guarded_loss_samples = 2;
        memory.unguarded_loss_samples = 4;
        memory.digestion_samples = 5;
        memory.signal_samples = 3;
        memory.pending_signal = PLANT_SIGNAL;
        memory.snapshot_valid = true;
        memory.snapshot_guarded = true;
        memory.observe_plant(3, -2, 900, 17);
        memory.plants[0].wall_cursor = 5;
        memory.plants[0].wall_layer = 2;
        memory.plants[0].pit_pressure = 3;
        let bytes = memory.encode(128).unwrap();
        let decoded = ColonyMemory::decode(&bytes, Role::Feeder);
        assert_eq!(decoded.role, Role::Builder);
        assert_eq!((decoded.x, decoded.y), (-7, 12));
        assert_eq!(decoded.pending_plant_index, 0);
        assert_eq!(decoded.pending_construction_phase, ConstructionPhase::Wall);
        assert_eq!(decoded.coverage, memory.coverage);
        assert_eq!(decoded.was_visited(-2, 3), Some(true));
        assert_eq!(decoded.outcome_successes, memory.outcome_successes);
        assert_eq!(decoded.outcome_contentions, memory.outcome_contentions);
        assert_eq!(decoded.snapshot_assimilated_energy, 900);
        assert_eq!(decoded.snapshot_gut_energy, 12);
        assert_eq!(decoded.expected_signal_energy, 4);
        assert_eq!(decoded.guarded_loss_ema, 6);
        assert_eq!(decoded.unguarded_loss_ema, 19);
        assert_eq!(decoded.digestion_progress_ema, 3);
        assert_eq!(decoded.signal_retention_q8, 220);
        assert_eq!(decoded.pending_signal, PLANT_SIGNAL);
        assert!(decoded.snapshot_valid);
        assert!(decoded.snapshot_guarded);
        assert_eq!(decoded.plants, memory.plants);
    }

    #[test]
    fn memory_encoding_obeys_small_abi_limit() {
        let mut memory = ColonyMemory::new(Role::Explorer);
        for coordinate in 0..MAX_MAP_ENTRIES as i16 {
            memory.observe_plant(coordinate, -coordinate, 100, 5);
        }
        let bytes = memory.encode(HEADER_BYTES + MAP_ENTRY_BYTES * 2).unwrap();
        assert_eq!(bytes.len(), HEADER_BYTES + MAP_ENTRY_BYTES * 2);
        assert_eq!(ColonyMemory::decode(&bytes, Role::Feeder).plants.len(), 2);
        assert!(memory.encode(HEADER_BYTES - 1).is_none());
    }

    #[test]
    fn known_role_markers_are_only_a_local_heuristic() {
        for role in [
            Role::Feeder,
            Role::Explorer,
            Role::Builder,
            Role::Defender,
            Role::Attacker,
        ] {
            assert_eq!(Role::from_marker(role.marker()), Some(role));
        }
        assert_eq!(Role::from_marker(0), None);
    }

    #[test]
    fn plant_founder_becomes_feeder_consumes_and_beacons() {
        let mut input = input();
        input.current_tile.plant_energy = 30;
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 7;
        let decision = decide(&input);
        assert!(matches!(
            decision.action,
            ReferenceMindAction::Consume { amount: 12 }
        ));
        assert_eq!(
            decision.signal,
            Some(ReferenceSignalEmission {
                channel: PLANT_SIGNAL,
                amount: PLANT_SIGNAL_QUANTA,
            })
        );
        let memory = memory_from_decision(&decision);
        assert_eq!(memory.role, Role::Feeder);
        assert_eq!(memory.plants.len(), 1);
        assert_eq!((memory.min_growth, memory.max_growth), (7, 7));
    }

    #[test]
    fn cooccurring_plant_and_threat_use_an_explicit_multichannel_action() {
        let mut input = input();
        input.current_tile.plant_energy = 30;
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 7;
        input.slots[0].neighbor = Some(ReferenceNeighbor {
            marker: None,
            apparent_mass_bucket: None,
            activity: None,
            progress: None,
        });

        let decision = decide(&input);

        assert_eq!(
            decision.action,
            ReferenceMindAction::Signal {
                amounts: [PLANT_SIGNAL_QUANTA, THREAT_SIGNAL_QUANTA, 0, 0],
            }
        );
        assert_eq!(decision.signal, None);
        assert!(action_is_commit_legal(&input, &decision.action, false));
    }

    #[test]
    fn signal_ablation_profiles_change_only_the_deposit_policy() {
        let mut input = input();
        input.current_tile.plant_energy = 30;
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 7;

        let disabled = decide_with_signal_policy(&input, SignalPolicy::Disabled);
        assert!(matches!(
            disabled.action,
            ReferenceMindAction::Consume { amount: 12 }
        ));
        assert_eq!(disabled.signal, None);

        let one_quantum = decide_with_signal_policy(&input, SignalPolicy::OneQuantumSidecar);
        assert!(matches!(
            one_quantum.action,
            ReferenceMindAction::Consume { amount: 12 }
        ));
        assert_eq!(
            one_quantum.signal,
            Some(ReferenceSignalEmission {
                channel: PLANT_SIGNAL,
                amount: 1,
            })
        );

        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.decisions = 1;
        input.private_memory = memory.encode(2048).unwrap();
        let suppressed = decide_with_signal_policy(&input, SignalPolicy::CadencedSemanticSidecar);
        assert!(matches!(
            suppressed.action,
            ReferenceMindAction::Consume { amount: 12 }
        ));
        assert_eq!(suppressed.signal, None);

        memory.decisions = 7;
        input.private_memory = memory.encode(2048).unwrap();
        let due = decide_with_signal_policy(&input, SignalPolicy::CadencedSemanticSidecar);
        assert_eq!(
            due.signal,
            Some(ReferenceSignalEmission {
                channel: PLANT_SIGNAL,
                amount: PLANT_SIGNAL_QUANTA,
            })
        );
    }

    #[test]
    fn semantic_sidecar_preserves_primary_action_when_facts_cooccur() {
        let mut input = input();
        input.current_tile.plant_energy = 30;
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 7;
        input.slots[0].neighbor = Some(ReferenceNeighbor {
            marker: None,
            apparent_mass_bucket: None,
            activity: None,
            progress: None,
        });

        let decision = decide_with_signal_policy(&input, SignalPolicy::SemanticSidecar);

        assert!(matches!(
            decision.action,
            ReferenceMindAction::Consume { amount: 12 }
        ));
        assert_eq!(
            decision.signal,
            Some(ReferenceSignalEmission {
                channel: THREAT_SIGNAL,
                amount: THREAT_SIGNAL_QUANTA,
            })
        );
    }

    #[test]
    fn unaffordable_multichannel_write_keeps_behavior_with_urgent_sidecar() {
        let mut input = input();
        input.self_state.assimilated_energy = 19;
        input.current_tile.plant_energy = 30;
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 7;
        input.slots[0].neighbor = Some(ReferenceNeighbor {
            marker: None,
            apparent_mass_bucket: None,
            activity: None,
            progress: None,
        });

        let decision = decide(&input);

        assert!(matches!(
            decision.action,
            ReferenceMindAction::Consume { amount: 12 }
        ));
        assert_eq!(
            decision.signal,
            Some(ReferenceSignalEmission {
                channel: THREAT_SIGNAL,
                amount: THREAT_SIGNAL_QUANTA,
            })
        );
        assert!(action_is_commit_legal_with_signal_amount(
            &input,
            &decision.action,
            THREAT_SIGNAL_QUANTA
        ));
    }

    #[test]
    fn successful_move_advances_only_the_private_dead_reckoned_map() {
        let mut input = input();
        let mut memory = ColonyMemory::new(Role::Explorer);
        memory.pending = PendingAction::Move;
        memory.pending_dx = 1;
        memory.pending_dy = -1;
        input.private_memory = memory.encode(2048).unwrap();
        input.self_state.last_outcome = Some(ReferenceOutcome {
            status: ReferenceOutcomeStatus::Success,
            rejected_reason: None,
        });
        let decision = decide(&input);
        let memory = memory_from_decision(&decision);
        assert_eq!((memory.x, memory.y), (1, -1));
        assert_eq!(memory.move_successes, 1);
        assert_eq!(memory.was_visited(0, 0), Some(true));
        assert_eq!(memory.was_visited(-1, 1), Some(true));
    }

    #[test]
    fn rolling_coverage_drops_distant_history_and_recenters() {
        let mut memory = ColonyMemory::new(Role::Explorer);
        memory.mark_visited(3, 0);
        memory.shift_coverage(1, 0);
        assert_eq!(memory.was_visited(-1, 0), Some(true));
        assert_eq!(memory.was_visited(2, 0), Some(true));
        assert_eq!(memory.was_visited(0, 0), Some(true));

        memory.shift_coverage(8, 0);
        assert_eq!(memory.coverage.count_ones(), 1);
        assert_eq!(memory.was_visited(0, 0), Some(true));
    }

    #[test]
    fn outcome_estimator_distinguishes_contention_from_rejection() {
        let mut input = input();
        let mut memory = ColonyMemory::new(Role::Explorer);
        memory.pending = PendingAction::Move;
        memory.snapshot_valid = true;
        memory.snapshot_assimilated_energy = input.self_state.assimilated_energy;
        input.self_state.last_outcome = Some(ReferenceOutcome {
            status: ReferenceOutcomeStatus::Contested,
            rejected_reason: None,
        });
        memory.reconcile_previous_action(&input);
        assert_eq!(memory.outcome_contentions[EstimateFamily::Move as usize], 1);
        assert_eq!(memory.outcome_setbacks[EstimateFamily::Move as usize], 0);

        memory.pending = PendingAction::Attack;
        input.self_state.last_outcome = Some(ReferenceOutcome {
            status: ReferenceOutcomeStatus::Rejected,
            rejected_reason: Some(ReferenceRejectReason::InsufficientEnergy),
        });
        memory.reconcile_previous_action(&input);
        assert_eq!(
            memory.outcome_contentions[EstimateFamily::Attack as usize],
            0
        );
        assert_eq!(memory.outcome_setbacks[EstimateFamily::Attack as usize], 1);
    }

    #[test]
    fn local_intervals_learn_gut_progress_guarded_loss_and_signal_retention() {
        let mut input = input();
        input.self_state.assimilated_energy = 90;
        input.self_state.gut_energy = 15;
        input.current_tile.signal_energy[usize::from(PLANT_SIGNAL)] = 9;
        input.self_state.last_outcome = Some(ReferenceOutcome {
            status: ReferenceOutcomeStatus::Success,
            rejected_reason: None,
        });
        let mut memory = ColonyMemory::new(Role::Defender);
        memory.pending = PendingAction::Guard;
        memory.pending_signal = PLANT_SIGNAL;
        memory.snapshot_valid = true;
        memory.snapshot_guarded = true;
        memory.snapshot_assimilated_energy = 100;
        memory.snapshot_gut_energy = 20;
        memory.expected_signal_energy = 11;

        memory.reconcile_previous_action(&input);
        assert_eq!(memory.guarded_loss_ema, 10);
        assert_eq!(memory.guarded_loss_samples, 1);
        assert_eq!(memory.digestion_progress_ema, 5);
        assert_eq!(memory.digestion_samples, 1);
        assert_eq!(memory.signal_retention_q8, 209);
        assert_eq!(memory.signal_samples, 1);
        assert_eq!(memory.outcome_successes[EstimateFamily::Guard as usize], 1);
        assert_eq!(memory.outcome_successes[EstimateFamily::Signal as usize], 1);
    }

    #[test]
    fn learned_digestion_progress_bounds_the_next_bite() {
        let mut input = input();
        input.current_tile.plant_energy = 100;
        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.digestion_samples = 4;
        memory.digestion_progress_ema = 2;
        assert_eq!(
            consume_if_useful(&input, &memory),
            Some(ReferenceMindAction::Consume { amount: 8 })
        );
    }

    #[test]
    fn learned_contention_changes_travel_effort_and_attack_commitment() {
        let input = input();
        let mut memory = ColonyMemory::new(Role::Attacker);
        memory.outcome_contentions[EstimateFamily::Move as usize] = 4;
        memory.outcome_contentions[EstimateFamily::Attack as usize] = 4;
        assert_eq!(travel_effort(&input, &memory), ReferenceEffort::Standard);
        assert_eq!(learned_attack_divisor(&memory, 8), 16);
    }

    #[test]
    fn retained_local_signal_suppresses_redundant_emission() {
        let mut input = input();
        input.current_tile.signal_energy[usize::from(PLANT_SIGNAL)] = 2;
        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.signal_samples = 3;
        memory.signal_retention_q8 = 256;
        assert!(!learned_signal_is_useful(
            &input,
            &memory,
            PLANT_SIGNAL,
            PLANT_SIGNAL_QUANTA
        ));
        input.current_tile.signal_energy[usize::from(PLANT_SIGNAL)] = 1;
        assert!(learned_signal_is_useful(
            &input,
            &memory,
            PLANT_SIGNAL,
            PLANT_SIGNAL_QUANTA
        ));
    }

    #[test]
    fn exploration_prefers_the_only_locally_unvisited_target() {
        let mut input = input();
        let mut memory = ColonyMemory::new(Role::Explorer);
        for slot in &input.slots {
            if slot.slot != 4 {
                memory.mark_visited(i16::from(slot.dx), i16::from(slot.dy));
            }
        }
        input.private_memory = memory.encode(2048).unwrap();
        assert!(matches!(
            decide(&input).action,
            ReferenceMindAction::Move { target_slot: 4, .. }
        ));
    }

    #[test]
    fn builder_deposits_carried_terrain_on_the_next_plant_ring_position() {
        let mut input = input();
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.x = -1;
        memory.y = -1;
        memory.observe_plant(0, 0, 100, 8);
        input.private_memory = memory.encode(2048).unwrap();
        input.self_state.carried_material_mass = 10;
        let decision = decide(&input);
        assert_eq!(decision.action, ReferenceMindAction::DepositTerrain);
        assert!(action_is_commit_legal_with_signal_amount(
            &input,
            &decision.action,
            decision.signal.map_or(0, |signal| signal.amount)
        ));
    }

    #[test]
    fn deposit_outcome_advances_only_the_selected_plant() {
        let mut input = input();
        input.self_state.last_outcome = Some(ReferenceOutcome {
            status: ReferenceOutcomeStatus::Success,
            rejected_reason: None,
        });
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.observe_plant(0, 0, 100, 8);
        memory.observe_plant(6, 0, 100, 8);
        memory.pending = PendingAction::Deposit;
        memory.pending_plant_index = 1;
        memory.pending_construction_phase = ConstructionPhase::Wall;

        memory.reconcile_previous_action(&input);

        assert_eq!(memory.plants[0].wall_cursor, 0);
        assert_eq!(memory.plants[1].wall_cursor, 1);
        assert_eq!(memory.plants[0].wall_layer, 0);
        assert_eq!(memory.plants[1].wall_layer, 0);
    }

    #[test]
    fn builder_never_deposits_on_a_newly_known_productive_tile() {
        let mut input = input();
        input.self_state.carried_material_mass = 10;
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 6;
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.x = -1;
        memory.y = -1;
        memory.observe_plant(0, 0, 100, 8);
        memory.observe_plant(-1, -1, 100, 6);
        memory.plants[1].wall_layer = 2;
        input.private_memory = memory.encode(2048).unwrap();

        let decision = decide(&input);
        let remembered = memory_from_decision(&decision);

        assert_ne!(decision.action, ReferenceMindAction::DepositTerrain);
        assert_eq!(remembered.plants[0].wall_cursor, 1);
        assert_eq!(remembered.plants[1].wall_layer, 2);
    }

    #[test]
    fn builder_rotates_away_from_a_productive_excavation_source() {
        let mut input = input();
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 6;
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.x = -2;
        memory.y = -2;
        memory.observe_plant(0, 0, 100, 8);
        memory.observe_plant(-2, -2, 100, 6);
        memory.plants[1].wall_layer = 2;

        let action = builder_action(&input, &mut memory);

        assert_ne!(action, ReferenceMindAction::Excavate);
        assert_eq!(memory.plants[0].pit_pressure, 1);
        assert_eq!(memory.pending_construction_phase, ConstructionPhase::Source);
    }

    #[test]
    fn contested_source_action_rotates_only_that_plants_pit() {
        let mut input = input();
        input.self_state.last_outcome = Some(ReferenceOutcome {
            status: ReferenceOutcomeStatus::Contested,
            rejected_reason: None,
        });
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.observe_plant(0, 0, 100, 8);
        memory.observe_plant(6, 0, 100, 8);
        memory.pending = PendingAction::Move;
        memory.pending_plant_index = 1;
        memory.pending_construction_phase = ConstructionPhase::Source;

        memory.reconcile_previous_action(&input);

        assert_eq!(memory.plants[0].pit_pressure, 0);
        assert_eq!(memory.plants[1].pit_pressure, 1);
    }

    #[test]
    fn threatened_defender_moves_to_and_guards_the_relief_gate() {
        let mut input = input();
        input.current_tile.signal_energy[usize::from(THREAT_SIGNAL)] = 1;
        let mut memory = ColonyMemory::new(Role::Defender);
        memory.x = 0;
        memory.y = 2;
        memory.observe_plant(0, 0, 100, 8);
        assert!(matches!(
            defender_action(&input, &mut memory),
            ReferenceMindAction::Move { target_slot: 1, .. }
        ));

        memory.y = 1;
        assert!(matches!(
            defender_action(&input, &mut memory),
            ReferenceMindAction::Guard { .. }
        ));
    }

    #[test]
    fn anonymous_threat_field_repositions_defender_from_plant_to_gate() {
        let mut simulation = ReferenceSimulation::new(5, 5, scenario_rules()).unwrap();
        let plant = simulation.tile(2, 2).unwrap();
        let gate = simulation.tile(2, 3).unwrap();
        let plant_state = simulation.tile_state_mut(plant).unwrap();
        plant_state.plant_capacity = 100;
        plant_state.signal_energy[usize::from(THREAT_SIGNAL)] = 10;
        let defender = simulation
            .add_cell(plant, 10, 1_000, Role::Defender.marker())
            .unwrap();
        let mut state = simulation.canonical_state();
        let defender_state = state
            .cells
            .iter_mut()
            .find(|(key, _)| *key == defender)
            .unwrap();
        let mut memory = ColonyMemory::new(Role::Defender);
        memory.observe_plant(0, 0, 100, 8);
        defender_state.1.private_memory = Arc::from(memory.encode(2048).unwrap());
        simulation =
            ReferenceSimulation::from_canonical_state(5, 5, simulation.rules().clone(), state)
                .unwrap();

        let (decision, report) = run_scenario_decision(&mut simulation, defender);

        assert!(matches!(
            decision.action,
            ReferenceMindAction::Move { target_slot: 6, .. }
        ));
        assert_eq!(report.outcomes[0].status, OutcomeStatus::Success);
        assert_eq!(simulation.cell(defender).unwrap().position, gate);
    }

    #[test]
    fn split_child_inherits_map_with_its_own_position_and_role() {
        let mut input = input();
        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.observe_plant(0, 0, 100, 8);
        input.private_memory = memory.encode(2048).unwrap();
        input.self_state.assimilated_energy = 1_000;
        let decision = decide(&input);
        let ReferenceMindAction::Split {
            target_slot,
            marker,
            private_memory,
            ..
        } = &decision.action
        else {
            panic!("feeder did not split: {:?}", decision.action);
        };
        let child = ColonyMemory::decode(private_memory, Role::Attacker);
        let slot = input
            .slots
            .iter()
            .find(|slot| slot.slot == *target_slot)
            .unwrap();
        assert_eq!((child.x, child.y), (i16::from(slot.dx), i16::from(slot.dy)));
        assert_eq!(Role::from_marker(*marker), Some(child.role));
        assert_eq!(child.plants, memory.plants);
        assert_eq!(child.was_visited(0, 0), Some(true));
        assert_eq!(
            child.was_visited(-i16::from(slot.dx), -i16::from(slot.dy)),
            Some(true)
        );
        assert!(!child.snapshot_valid);
        assert_eq!(child.pending_signal, NO_PENDING_SIGNAL);
    }

    #[test]
    fn unmapped_parent_locally_demands_an_explorer() {
        let input = input();
        let mut memory = ColonyMemory::new(Role::Feeder);
        assert_eq!(child_role(&input, &mut memory), Role::Explorer);
        assert_eq!(memory.child_cursor, 1);
    }

    #[test]
    fn flat_plant_ring_locally_demands_a_builder() {
        let mut input = input();
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 8;
        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.observe_plant(0, 0, 100, 8);
        assert_eq!(visible_wall_gaps(&input), 7);
        assert_eq!(child_role(&input, &mut memory), Role::Builder);
    }

    #[test]
    fn completed_local_wall_and_builder_signal_demand_a_defender() {
        let mut input = input();
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 8;
        input.current_tile.signal_energy[usize::from(BUILD_SIGNAL)] = 1;
        for slot in &mut input.slots {
            slot.elevation = Some(if slot.dx == 0 && slot.dy == 1 { 1 } else { 2 });
        }
        input.slots[0].neighbor = Some(ReferenceNeighbor {
            marker: Some(Role::Builder.marker()),
            apparent_mass_bucket: None,
            activity: None,
            progress: None,
        });
        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.observe_plant(0, 0, 100, 8);
        assert_eq!(visible_wall_gaps(&input), 0);
        assert_eq!(child_role(&input, &mut memory), Role::Defender);
    }

    #[test]
    fn builder_markers_can_redirect_but_not_authenticate_local_demand() {
        let mut input = input();
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 8;
        let mut unoccupied_memory = ColonyMemory::new(Role::Feeder);
        unoccupied_memory.observe_plant(0, 0, 100, 8);
        assert_eq!(child_role(&input, &mut unoccupied_memory), Role::Builder);

        for slot in input.slots.iter_mut().take(3) {
            slot.neighbor = Some(ReferenceNeighbor {
                marker: Some(Role::Builder.marker()),
                apparent_mass_bucket: None,
                activity: None,
                progress: None,
            });
        }
        let mut occupied_memory = ColonyMemory::new(Role::Feeder);
        occupied_memory.observe_plant(0, 0, 100, 8);
        assert_eq!(child_role(&input, &mut occupied_memory), Role::Defender);
    }

    #[test]
    fn strong_local_threat_demand_overrides_exploration_with_an_attacker() {
        let mut input = input();
        input.current_tile.signal_energy[usize::from(THREAT_SIGNAL)] = 256;
        input.slots[0].neighbor = Some(ReferenceNeighbor {
            marker: None,
            apparent_mass_bucket: None,
            activity: None,
            progress: None,
        });
        let mut memory = ColonyMemory::new(Role::Explorer);
        assert_eq!(child_role(&input, &mut memory), Role::Attacker);
    }

    #[test]
    fn unavailable_action_family_is_not_locally_requested() {
        let mut input = input();
        input.current_tile.plant_capacity = 100;
        input.current_tile.plant_growth_rate = 8;
        input.action_space.terrain_mass_per_elevation = 0;
        let mut memory = ColonyMemory::new(Role::Feeder);
        memory.observe_plant(0, 0, 100, 8);
        assert_ne!(child_role(&input, &mut memory), Role::Builder);
    }

    #[test]
    fn plant_feeder_split_materializes_the_locally_requested_builder() {
        let mut simulation = ReferenceSimulation::new(5, 5, scenario_rules()).unwrap();
        let plant = simulation.tile(2, 2).unwrap();
        simulation.tile_state_mut(plant).unwrap().plant_capacity = 100;
        let feeder = simulation
            .add_cell(plant, 10, 10_000, Role::Feeder.marker())
            .unwrap();

        let split_input = simulation
            .reference_mind_input(feeder, PrivateRandom::ZERO)
            .unwrap();
        let mut split_memory = ColonyMemory::new(Role::Feeder);
        split_memory.update_observations(&split_input);
        assert_eq!(visible_wall_gaps(&split_input), 7);
        assert_eq!(child_role(&split_input, &mut split_memory), Role::Builder);

        let (decision, report) = run_scenario_decision(&mut simulation, feeder);
        let ReferenceMindAction::Split { marker, .. } = decision.action else {
            panic!("plant feeder did not split: {:?}", decision.action);
        };
        assert_eq!(Role::from_marker(marker), Some(Role::Builder));
        assert_eq!(report.outcomes[0].status, OutcomeStatus::Success);

        let telemetry = telemetry::ColonyScenarioTelemetry::capture(&simulation, 5, 5);
        assert_eq!(telemetry.colony_cells, 2);
        assert_eq!(telemetry.cells_by_role[Role::Feeder as usize], 1);
        assert_eq!(telemetry.cells_by_role[Role::Builder as usize], 1);
        assert_eq!(telemetry.cells_with_valid_memory, 2);
    }

    #[test]
    fn every_role_returns_an_exactly_affordable_action() {
        for role in [
            Role::Feeder,
            Role::Explorer,
            Role::Builder,
            Role::Defender,
            Role::Attacker,
        ] {
            let mut input = input();
            input.private_memory = ColonyMemory::new(role).encode(2048).unwrap();
            let decision = decide(&input);
            assert!(
                action_is_commit_legal_with_signal_amount(
                    &input,
                    &decision.action,
                    decision.signal.map_or(0, |signal| signal.amount),
                ),
                "illegal {role:?} decision: {decision:?}"
            );
        }
    }

    #[test]
    fn builder_leaves_a_traversable_gate_in_the_second_wall_layer() {
        let mut simulation = ReferenceSimulation::new(9, 9, scenario_rules()).unwrap();
        let plant = simulation.tile(4, 4).unwrap();
        simulation.tile_state_mut(plant).unwrap().plant_capacity = 100;
        let builder_start = simulation.tile(3, 3).unwrap();
        let builder = simulation
            .add_cell(builder_start, 10, 10_000, Role::Builder.marker())
            .unwrap();

        let mut successful_deposits = 0;
        let mut first_ring = None;
        for _ in 0..256 {
            let (_, report) = run_scenario_decision(&mut simulation, builder);
            successful_deposits += report
                .outcomes
                .iter()
                .filter(|outcome| {
                    outcome
                        .terrain_change
                        .as_ref()
                        .is_some_and(|change| change.elevation_after > change.elevation_before)
                })
                .count();
            if successful_deposits == 8 && first_ring.is_none() {
                first_ring = Some(telemetry::ColonyScenarioTelemetry::capture(
                    &simulation,
                    9,
                    9,
                ));
            }
            if successful_deposits == 15 {
                break;
            }
        }
        assert_eq!(successful_deposits, 15);
        let first_ring = first_ring.unwrap();
        assert_eq!(first_ring.elevated_ring_tiles, 8);
        assert_eq!(first_ring.elevated_wall_coverage(), 1.0);
        assert_eq!(first_ring.impassable_wall_coverage(), 0.0);

        let telemetry = telemetry::ColonyScenarioTelemetry::capture(&simulation, 9, 9);
        assert_eq!(telemetry.world_plant_tiles, 1);
        assert_eq!(telemetry.confirmed_mapped_plant_tiles, 1);
        assert_eq!(telemetry.plant_discovery_coverage(), 1.0);
        assert_eq!(telemetry.elevated_ring_tiles, 8);
        assert_eq!(telemetry.elevated_wall_coverage(), 1.0);
        assert_eq!(telemetry.impassable_ring_tiles, 7);
        assert_eq!(telemetry.impassable_wall_coverage(), 0.875);
        assert_eq!(telemetry.traversable_ring_tiles(), 1);
        assert_eq!(
            simulation
                .tile_state(simulation.tile(4, 5).unwrap())
                .unwrap()
                .elevation,
            1
        );
        assert_eq!(
            simulation
                .tile_state(simulation.tile(4, 6).unwrap())
                .unwrap()
                .elevation,
            0
        );
        assert!(telemetry.signal_energy[usize::from(BUILD_SIGNAL)] > 0);
        assert_eq!(telemetry.cells_by_role[Role::Builder as usize], 1);
        assert_eq!(telemetry.cells_with_valid_memory, 1);
    }

    #[test]
    fn completed_first_plant_plan_does_not_redirect_second_plant_construction() {
        let mut simulation = ReferenceSimulation::new(11, 7, scenario_rules()).unwrap();
        let first_plant = simulation.tile(2, 3).unwrap();
        let second_plant = simulation.tile(8, 3).unwrap();
        simulation
            .tile_state_mut(first_plant)
            .unwrap()
            .plant_capacity = 100;
        simulation
            .tile_state_mut(second_plant)
            .unwrap()
            .plant_capacity = 100;
        let second_ring = simulation.tile(7, 2).unwrap();
        let builder = simulation
            .add_cell(second_ring, 10, 10_000, Role::Builder.marker())
            .unwrap();

        let mut state = simulation.canonical_state();
        let builder_state = state
            .cells
            .iter_mut()
            .find(|(key, _)| *key == builder)
            .unwrap();
        builder_state.1.carried_material_mass = 10;
        let mut memory = ColonyMemory::new(Role::Builder);
        memory.x = 7;
        memory.y = 2;
        memory.observe_plant(2, 3, 100, 8);
        memory.observe_plant(8, 3, 100, 8);
        memory.plants[0].wall_layer = 2;
        builder_state.1.private_memory = Arc::from(memory.encode(2048).unwrap());
        simulation =
            ReferenceSimulation::from_canonical_state(11, 7, simulation.rules().clone(), state)
                .unwrap();

        let (decision, report) = run_scenario_decision(&mut simulation, builder);
        assert_eq!(decision.action, ReferenceMindAction::DepositTerrain);
        assert_eq!(report.outcomes[0].status, OutcomeStatus::Success);
        assert_eq!(simulation.tile_state(second_ring).unwrap().elevation, 1);
        assert_eq!(
            simulation
                .tile_state(simulation.tile(1, 2).unwrap())
                .unwrap()
                .elevation,
            0
        );

        let next_input = simulation
            .reference_mind_input(builder, PrivateRandom::ZERO)
            .unwrap();
        let next = decide(&next_input);
        let reconciled = memory_from_decision(&next);
        assert_eq!(reconciled.plants[0].wall_layer, 2);
        assert_eq!(reconciled.plants[0].wall_cursor, 0);
        assert_eq!(reconciled.plants[1].wall_cursor, 1);
    }

    #[test]
    fn stationed_defender_guards_beside_a_known_colony_cell() {
        let mut simulation = ReferenceSimulation::new(5, 5, scenario_rules()).unwrap();
        let plant = simulation.tile(2, 2).unwrap();
        simulation.tile_state_mut(plant).unwrap().plant_capacity = 100;
        let defender = simulation
            .add_cell(plant, 10, 1_000, Role::Defender.marker())
            .unwrap();
        let builder_position = simulation.tile(2, 1).unwrap();
        simulation
            .add_cell(builder_position, 10, 1_000, Role::Builder.marker())
            .unwrap();

        let (decision, _) = run_scenario_decision(&mut simulation, defender);
        assert!(matches!(decision.action, ReferenceMindAction::Guard { .. }));
        assert_eq!(
            simulation.cell(defender).unwrap().last_outcome,
            Some(OutcomeStatus::Success)
        );
        assert!(simulation.cell(defender).unwrap().guarded);
        let telemetry = telemetry::ColonyScenarioTelemetry::capture(&simulation, 5, 5);
        assert_eq!(telemetry.defenders_on_station, 1);
        assert_eq!(telemetry.occupied_ring_tiles, 1);
    }

    #[test]
    fn defender_reaches_a_wall_station_through_the_gate() {
        let mut simulation = ReferenceSimulation::new(9, 9, scenario_rules()).unwrap();
        let plant = simulation.tile(4, 4).unwrap();
        simulation.tile_state_mut(plant).unwrap().plant_capacity = 100;
        let builder_start = simulation.tile(3, 3).unwrap();
        let builder = simulation
            .add_cell(builder_start, 10, 10_000, Role::Builder.marker())
            .unwrap();
        let mut deposits = 0;
        for _ in 0..256 {
            let (_, report) = run_scenario_decision(&mut simulation, builder);
            deposits += report
                .outcomes
                .iter()
                .filter(|outcome| {
                    outcome
                        .terrain_change
                        .as_ref()
                        .is_some_and(|change| change.elevation_after > change.elevation_before)
                })
                .count();
            if deposits == 15 {
                break;
            }
        }
        assert_eq!(deposits, 15);

        let outside_gate = simulation.tile(4, 6).unwrap();
        let defender = simulation
            .add_cell(outside_gate, 10, 1_000, Role::Defender.marker())
            .unwrap();
        let mut state = simulation.canonical_state();
        let defender_state = state
            .cells
            .iter_mut()
            .find(|(key, _)| *key == defender)
            .unwrap();
        let mut memory = ColonyMemory::new(Role::Defender);
        memory.observe_plant(0, -2, 100, 0);
        defender_state.1.private_memory = Arc::from(memory.encode(2048).unwrap());
        simulation =
            ReferenceSimulation::from_canonical_state(9, 9, simulation.rules().clone(), state)
                .unwrap();

        let (first, first_report) = run_scenario_decision(&mut simulation, defender);
        assert!(matches!(first.action, ReferenceMindAction::Move { .. }));
        assert_eq!(first_report.outcomes[0].status, OutcomeStatus::Success);
        assert_eq!(
            simulation.cell(defender).unwrap().position,
            simulation.tile(4, 5).unwrap()
        );

        let (second, second_report) = run_scenario_decision(&mut simulation, defender);
        assert!(matches!(second.action, ReferenceMindAction::Guard { .. }));
        assert_eq!(second_report.outcomes[0].status, OutcomeStatus::Success);
        assert_eq!(
            simulation.cell(defender).unwrap().position,
            simulation.tile(4, 5).unwrap()
        );
        assert!(simulation.cell(defender).unwrap().guarded);
        let telemetry = telemetry::ColonyScenarioTelemetry::capture(&simulation, 9, 9);
        assert_eq!(telemetry.defenders_on_station, 1);
        assert_eq!(telemetry.traversable_ring_tiles(), 1);
    }

    #[test]
    fn spoofed_role_marker_suppresses_attack_while_unmarked_neighbor_does_not() {
        let mut spoofed = input();
        spoofed.private_memory = ColonyMemory::new(Role::Attacker).encode(2048).unwrap();
        spoofed.slots[4].neighbor = Some(ReferenceNeighbor {
            marker: Some(Role::Defender.marker()),
            apparent_mass_bucket: None,
            activity: None,
            progress: None,
        });
        let spoofed_decision = decide(&spoofed);
        assert!(!matches!(
            spoofed_decision.action,
            ReferenceMindAction::Attack { .. }
        ));

        let mut unmarked = spoofed;
        unmarked.slots[4].neighbor.as_mut().unwrap().marker = None;
        let unmarked_decision = decide(&unmarked);
        assert!(matches!(
            unmarked_decision.action,
            ReferenceMindAction::Attack { target_slot: 4, .. }
        ));
    }

    #[test]
    fn colliding_signal_gradients_produce_role_specific_routes() {
        let mut input = input();
        input.self_state.assimilated_energy = 50;
        let mut plant_gradient = [0; 4];
        plant_gradient[usize::from(PLANT_SIGNAL)] = 20;
        input.slots[3].signal_energy = Some(plant_gradient);
        let mut threat_gradient = [0; 4];
        threat_gradient[usize::from(THREAT_SIGNAL)] = 100;
        input.slots[4].signal_energy = Some(threat_gradient);

        input.private_memory = ColonyMemory::new(Role::Feeder).encode(2048).unwrap();
        assert!(matches!(
            decide(&input).action,
            ReferenceMindAction::Move { target_slot: 3, .. }
        ));

        input.private_memory = ColonyMemory::new(Role::Attacker).encode(2048).unwrap();
        assert!(matches!(
            decide(&input).action,
            ReferenceMindAction::Move { target_slot: 4, .. }
        ));
    }
}
