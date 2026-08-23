//! Ergonomic Rust types for the canonical Mind ABI.

use crate::randomness::PrivateRandom;

pub const REFERENCE_MAX_LOCAL_SLOTS: usize = 32;
pub const REFERENCE_SIGNAL_CHANNELS: usize = 4;
pub const EFFORT_GENTLE_BIT: u8 = 1 << 0;
pub const EFFORT_STANDARD_BIT: u8 = 1 << 1;
pub const EFFORT_BURST_BIT: u8 = 1 << 2;
pub const LOCAL_ELEVATION_BIT: u16 = 1 << 0;
pub const LOCAL_PLANT_ENERGY_BIT: u16 = 1 << 1;
pub const LOCAL_PLANT_CAPACITY_BIT: u16 = 1 << 2;
pub const LOCAL_PLANT_GROWTH_RATE_BIT: u16 = 1 << 3;
pub const LOCAL_LOOSE_ENERGY_BIT: u16 = 1 << 4;
pub const LOCAL_DIFFUSE_ENERGY_BIT: u16 = 1 << 5;
pub const LOCAL_SIGNAL_ENERGY_BIT: u16 = 1 << 6;
pub const LOCAL_NEIGHBOR_BIT: u16 = 1 << 7;
pub const LOCAL_NEIGHBOR_MARKER_BIT: u16 = 1 << 8;
pub const LOCAL_NEIGHBOR_MASS_BIT: u16 = 1 << 9;
pub const LOCAL_NEIGHBOR_ACTIVITY_BIT: u16 = 1 << 10;
pub const LOCAL_NEIGHBOR_PROGRESS_BIT: u16 = 1 << 11;
pub const LOCAL_VISIBILITY_BITS: u16 = (1 << 12) - 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceMindInput {
    pub self_state: ReferenceSelfState,
    /// Environment at the cell's current position. This is deliberately not
    /// assigned a slot because it cannot be targeted by relative actions.
    pub current_tile: CurrentTileObservation,
    /// Canonically ordered, ruleset-defined relative slots.
    pub slots: Vec<LocalObservation>,
    pub action_space: ReferenceActionSpace,
    pub private_memory: Vec<u8>,
    pub randomness: PrivateRandom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentTileObservation {
    pub elevation: i16,
    pub plant_energy: u64,
    pub plant_capacity: u64,
    /// Plant energy produced per simulation time unit when diffuse energy is
    /// available on this tile.
    pub plant_growth_rate: u64,
    pub loose_energy: u64,
    pub diffuse_energy: u64,
    pub signal_energy: [u64; REFERENCE_SIGNAL_CHANNELS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceSelfState {
    pub core_mass: u64,
    pub assimilated_energy: u64,
    pub gut_energy: u64,
    pub metabolism_remainder: u64,
    pub carried_material_mass: u64,
    pub marker: u32,
    pub guarded: bool,
    pub last_outcome: Option<ReferenceOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalObservation {
    pub slot: u8,
    pub dx: i8,
    pub dy: i8,
    pub distance_cost_q10: u16,
    pub reachable: bool,
    pub elevation: Option<i16>,
    pub plant_energy: Option<u64>,
    pub plant_capacity: Option<u64>,
    pub plant_growth_rate: Option<u64>,
    pub loose_energy: Option<u64>,
    pub diffuse_energy: Option<u64>,
    pub signal_energy: Option<[u64; REFERENCE_SIGNAL_CHANNELS]>,
    /// `Some` means that an occupant is locally observable. Its identity and
    /// team are intentionally unavailable.
    pub neighbor: Option<ReferenceNeighbor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceNeighbor {
    pub marker: Option<u32>,
    pub apparent_mass_bucket: Option<u8>,
    pub activity: Option<ReferenceActivity>,
    pub progress: Option<ReferenceProgress>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceActivity {
    Ready,
    Moving,
    AttackWindup,
    Guarding,
    Feeding,
    Splitting,
    ManipulatingTerrain,
    OtherBusy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceProgress {
    Early,
    Middle,
    Late,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceOutcome {
    pub status: ReferenceOutcomeStatus,
    pub rejected_reason: Option<ReferenceRejectReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceOutcomeStatus {
    Success,
    Frustrated,
    Contested,
    Interrupted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceRejectReason {
    InvalidSlot,
    ActionNotAllowedInSlot,
    TargetOutsideWorld,
    TargetsSelf,
    ZeroPayload,
    InsufficientGutEnergy,
    ChildAllocationTooSmall,
    PrivateMemoryTooLarge,
    InsufficientEnergy,
    InsufficientMaterial,
    TerrainLimit,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceActionSpace {
    pub wait_enabled: bool,
    pub guard_enabled: bool,
    pub consume_enabled: bool,
    pub excavate_enabled: bool,
    pub deposit_terrain_enabled: bool,
    pub signal_enabled: bool,
    pub move_targets: u32,
    pub attack_targets: u32,
    pub split_targets: u32,
    pub regurgitate_targets: u32,
    pub effort_mask: u8,
    /// Maximum amount that can enter the gut from a single Consume decision
    /// at this snapshot, before accounting for resource contention.
    pub max_consume_amount: u64,
    pub gut_capacity: u64,
    pub max_private_memory_bytes: u32,
    pub minimum_survival_energy: u64,
    pub child_core_mass: u64,
    pub metabolism_rate_numerator: u64,
    pub metabolism_rate_denominator: u64,
    pub terrain_mass_per_elevation: u64,
    pub signal_emission_cost: u64,
}

impl ReferenceActionSpace {
    pub const fn supports_effort(&self, effort: ReferenceEffort) -> bool {
        self.effort_mask & effort.bit() != 0
    }

    pub const fn allows_target(mask: u32, slot: u8) -> bool {
        slot < REFERENCE_MAX_LOCAL_SLOTS as u8 && mask & (1_u32 << slot) != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceEffort {
    Gentle,
    Standard,
    Burst,
}

impl ReferenceEffort {
    pub const fn bit(self) -> u8 {
        match self {
            Self::Gentle => EFFORT_GENTLE_BIT,
            Self::Standard => EFFORT_STANDARD_BIT,
            Self::Burst => EFFORT_BURST_BIT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceMindAction {
    Wait,
    Move {
        target_slot: u8,
        effort: ReferenceEffort,
    },
    Attack {
        target_slot: u8,
        effort: ReferenceEffort,
        payload: u64,
    },
    Guard {
        effort: ReferenceEffort,
    },
    Consume {
        amount: u64,
    },
    Split {
        target_slot: u8,
        child_allocation: u64,
        marker: u32,
        private_memory: Vec<u8>,
    },
    Regurgitate {
        target_slot: u8,
        amount: u64,
    },
    Excavate,
    DepositTerrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceSignalEmission {
    pub channel: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceMindDecision {
    pub action: ReferenceMindAction,
    /// Optional local field emission committed alongside the primary action.
    pub signal: Option<ReferenceSignalEmission>,
    /// Explicit persistent-memory update for this cell. `Retain` avoids
    /// copying unchanged bytes without exposing shared canonical storage.
    pub memory_update: ReferenceMemoryUpdate,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReferenceMemoryUpdate {
    #[default]
    Retain,
    Replace(Vec<u8>),
}

impl ReferenceMemoryUpdate {
    pub fn replacement(&self) -> Option<&[u8]> {
        match self {
            Self::Retain => None,
            Self::Replace(bytes) => Some(bytes),
        }
    }
}

/// Native form of the reference ABI. Implementations still receive only one
/// cell's serialized-equivalent local input and must be reset before every
/// invocation, matching the pristine WASM execution model.
pub trait ReferenceMind: Send {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision;

    fn reset(&mut self) -> Result<(), String>;
}

pub trait ReferenceMindFactory: Send {
    type M: ReferenceMind;

    fn create(&self) -> Result<Self::M, String>;
}
