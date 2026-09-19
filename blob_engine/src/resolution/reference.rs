use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut, Range};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, AtomicUsize};

#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_PASSIVE_CELL_THRESHOLD: usize = 16_384;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_PASSIVE_TILE_THRESHOLD: usize = 65_536;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_CANONICAL_TILE_VALIDATION_THRESHOLD: usize = 512 * 512;
#[cfg(not(target_arch = "wasm32"))]
const CANONICAL_TILE_VALIDATION_PARTITION_LEN: usize = 16_384;

#[cfg(not(target_arch = "wasm32"))]
fn passive_frontier_is_dense(active: usize, total: usize) -> bool {
    total > 0 && active.saturating_mul(4) >= total.saturating_mul(3)
}

#[cfg(test)]
use blob_interface::randomness::PrivateRandom;
use blob_interface::reference_mind::{
    ReferenceEffort, ReferenceMemoryUpdate, ReferenceMindAction, ReferenceSignalEmission,
    REFERENCE_SIGNAL_CHANNELS,
};

mod observation;
mod passive;
pub use observation::ReferenceObservationBatch;
use passive::{
    advance_cell_digestion, advance_cell_metabolism_with_divisor,
    advance_plant_growth_with_divisor, advance_signal_decay, compile_diffusion_neighbors,
    metabolic_exhaustion_time, PassiveRateDivisor,
};
#[cfg(test)]
use passive::{advance_cell_metabolism, advance_plant_growth};
#[cfg(not(target_arch = "wasm32"))]
use passive::{
    compile_diffusion_sources, dense_diffusion_final_energy, dense_digestion_preflight,
    dense_frontier_from_page_bitmaps, plan_dense_diffusion, planned_signal_frontier_changes,
    DenseSignalDecayPage,
};

use super::delta::{CellDelta, SimulationDelta, TileDelta};
use super::hashing::{
    canonical_state_hash, compiled_ruleset_hash, semantic_ruleset_hash, CanonicalHash,
    IncrementalStateHash, IncrementalStateHashSeed,
};
use super::neighborhood::{
    CompiledNeighborhood, DiagonalCornerRule, LocalSlot, NeighborhoodError, NeighborhoodSpec,
    SlotMask, TargetingAction, TileIndex,
};

pub const SIGNAL_CHANNELS: usize = REFERENCE_SIGNAL_CHANNELS;
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_COMMIT_PREFLIGHT_THRESHOLD: usize = 2_048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SimTime(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellKey(pub u64);

pub use super::cell_storage::CellStore;
use super::paged_slots::PagedSlots;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionKind {
    Wait,
    Move,
    Attack,
    Guard,
    Consume,
    Split,
    Regurgitate,
    Signal,
    Excavate,
    DepositTerrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortTier {
    Low,
    Standard,
    High,
}

impl EffortTier {
    const fn index(self) -> usize {
        match self {
            Self::Low => 0,
            Self::Standard => 1,
            Self::High => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffortProfile {
    pub cost_numerator: u32,
    pub cost_denominator: u32,
    pub duration_numerator: u32,
    pub duration_denominator: u32,
}

impl EffortProfile {
    pub const fn new(
        cost_numerator: u32,
        cost_denominator: u32,
        duration_numerator: u32,
        duration_denominator: u32,
    ) -> Self {
        Self {
            cost_numerator,
            cost_denominator,
            duration_numerator,
            duration_denominator,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurationRule {
    pub base_quanta: u64,
    pub mass_quanta_numerator: u64,
    pub mass_units_denominator: u64,
}

impl DurationRule {
    pub const fn new(
        base_quanta: u64,
        mass_quanta_numerator: u64,
        mass_units_denominator: u64,
    ) -> Self {
        Self {
            base_quanta,
            mass_quanta_numerator,
            mass_units_denominator,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TimeConfig {
    pub arithmetic_quanta_per_unit: u64,
    pub completion_bucket: u64,
    pub decision_interval_floor: u64,
}

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            arithmetic_quanta_per_unit: 1024,
            completion_bucket: 64,
            decision_interval_floor: 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionRequest {
    Wait,
    Move {
        target: LocalSlot,
        effort: EffortTier,
    },
    Attack {
        target: LocalSlot,
        effort: EffortTier,
        payload: u64,
    },
    Guard {
        effort: EffortTier,
    },
    Consume {
        amount: u64,
    },
    Split {
        target: LocalSlot,
        child_allocation: u64,
        marker: u32,
        private_memory: Vec<u8>,
    },
    Regurgitate {
        target: LocalSlot,
        amount: u64,
    },
    Signal {
        amounts: [u64; REFERENCE_SIGNAL_CHANNELS],
    },
    Excavate,
    DepositTerrain,
}

impl ActionRequest {
    pub const fn kind(&self) -> ActionKind {
        match self {
            Self::Wait => ActionKind::Wait,
            Self::Move { .. } => ActionKind::Move,
            Self::Attack { .. } => ActionKind::Attack,
            Self::Guard { .. } => ActionKind::Guard,
            Self::Consume { .. } => ActionKind::Consume,
            Self::Split { .. } => ActionKind::Split,
            Self::Regurgitate { .. } => ActionKind::Regurgitate,
            Self::Signal { .. } => ActionKind::Signal,
            Self::Excavate => ActionKind::Excavate,
            Self::DepositTerrain => ActionKind::DepositTerrain,
        }
    }

    const fn targeting_action(&self) -> Option<TargetingAction> {
        match self {
            Self::Move { .. } => Some(TargetingAction::Move),
            Self::Attack { .. } => Some(TargetingAction::Attack),
            Self::Split { .. } => Some(TargetingAction::Split),
            Self::Regurgitate { .. } => Some(TargetingAction::Regurgitate),
            Self::Wait
            | Self::Guard { .. }
            | Self::Consume { .. }
            | Self::Signal { .. }
            | Self::Excavate
            | Self::DepositTerrain => None,
        }
    }

    const fn target_slot(&self) -> Option<LocalSlot> {
        match self {
            Self::Move { target, .. }
            | Self::Attack { target, .. }
            | Self::Split { target, .. }
            | Self::Regurgitate { target, .. } => Some(*target),
            Self::Wait
            | Self::Guard { .. }
            | Self::Consume { .. }
            | Self::Signal { .. }
            | Self::Excavate
            | Self::DepositTerrain => None,
        }
    }

    const fn effort_tier(&self) -> EffortTier {
        match self {
            Self::Move { effort, .. } | Self::Attack { effort, .. } | Self::Guard { effort } => {
                *effort
            }
            Self::Wait
            | Self::Consume { .. }
            | Self::Split { .. }
            | Self::Regurgitate { .. }
            | Self::Signal { .. }
            | Self::Excavate
            | Self::DepositTerrain => EffortTier::Standard,
        }
    }

    const fn payload(&self) -> u64 {
        match self {
            Self::Attack { payload, .. } => *payload,
            Self::Split {
                child_allocation, ..
            } => *child_allocation,
            Self::Regurgitate { amount, .. } => *amount,
            Self::Wait
            | Self::Move { .. }
            | Self::Guard { .. }
            | Self::Consume { .. }
            | Self::Signal { .. }
            | Self::Excavate
            | Self::DepositTerrain => 0,
        }
    }
}

impl From<ReferenceMindAction> for ActionRequest {
    fn from(value: ReferenceMindAction) -> Self {
        match value {
            ReferenceMindAction::Wait => Self::Wait,
            ReferenceMindAction::Move {
                target_slot,
                effort,
            } => Self::Move {
                target: LocalSlot(target_slot),
                effort: effort.into(),
            },
            ReferenceMindAction::Attack {
                target_slot,
                effort,
                payload,
            } => Self::Attack {
                target: LocalSlot(target_slot),
                effort: effort.into(),
                payload,
            },
            ReferenceMindAction::Guard { effort } => Self::Guard {
                effort: effort.into(),
            },
            ReferenceMindAction::Consume { amount } => Self::Consume { amount },
            ReferenceMindAction::Split {
                target_slot,
                child_allocation,
                marker,
                private_memory,
            } => Self::Split {
                target: LocalSlot(target_slot),
                child_allocation,
                marker,
                private_memory,
            },
            ReferenceMindAction::Regurgitate {
                target_slot,
                amount,
            } => Self::Regurgitate {
                target: LocalSlot(target_slot),
                amount,
            },
            ReferenceMindAction::Signal { amounts } => Self::Signal { amounts },
            ReferenceMindAction::Excavate => Self::Excavate,
            ReferenceMindAction::DepositTerrain => Self::DepositTerrain,
        }
    }
}

impl From<ReferenceEffort> for EffortTier {
    fn from(value: ReferenceEffort) -> Self {
        match value {
            ReferenceEffort::Gentle => Self::Low,
            ReferenceEffort::Standard => Self::Standard,
            ReferenceEffort::Burst => Self::High,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
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
pub enum OutcomeStatus {
    Rejected(RejectReason),
    Success,
    Frustrated,
    Contested,
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingAction {
    pub request: ActionRequest,
    pub origin: TileIndex,
    pub target: Option<TileIndex>,
    pub started_at: SimTime,
    pub completes_at: SimTime,
    pub effort_spent: u64,
    pub payload_escrow: u64,
    pub rejection: Option<RejectReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellColdState {
    /// Canonical private bytes use immutable shared ownership so snapshots,
    /// deltas, and rollback state do not duplicate unchanged Mind memory.
    /// Mind inputs still receive an owned copy at the isolation boundary.
    pub private_memory: Arc<[u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellState {
    pub position: TileIndex,
    pub core_mass: u64,
    pub assimilated_energy: u64,
    pub gut_energy: u64,
    /// Fractional digestion numerator carried between event-time advances.
    pub digestion_remainder: u64,
    /// Fractional metabolic-upkeep numerator carried between event advances.
    pub metabolism_remainder: u64,
    pub carried_material_mass: u64,
    pub marker: u32,
    pub guarded: bool,
    pub ready_at: SimTime,
    /// Pending actions are immutable after commitment. Sharing them keeps
    /// completion snapshots and journals from cloning the largest hot record.
    pub pending_action: Option<Arc<PendingAction>>,
    /// Small, frequently replaced observation state remains inline so every
    /// completion does not allocate merely to publish its outcome.
    pub last_outcome: Option<OutcomeStatus>,
    /// Private Mind memory is copy-on-write cold state. `Deref` preserves
    /// ergonomic field reads; writes automatically detach.
    pub cold: Arc<CellColdState>,
}

impl Deref for CellState {
    type Target = CellColdState;

    fn deref(&self) -> &Self::Target {
        &self.cold
    }
}

impl DerefMut for CellState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.cold)
    }
}

impl CellState {
    pub fn total_mass(&self) -> u128 {
        u128::from(self.core_mass)
            + u128::from(self.assimilated_energy)
            + u128::from(self.gut_energy)
            + u128::from(self.carried_material_mass)
            + u128::from(
                self.pending_action
                    .as_ref()
                    .map_or(0, |action| action.payload_escrow),
            )
    }

    pub fn is_ready_at(&self, now: SimTime) -> bool {
        self.pending_action.is_none() && self.ready_at <= now
    }
}

const OVERFLOWED_METABOLIC_SCHEDULE: usize = usize::MAX - 1;

const CHECKPOINT_TILE_CHUNK_LEN: usize = 8;
static NEXT_CHECKPOINT_MUTATION_TOKEN: AtomicU64 = AtomicU64::new(1);

fn fresh_checkpoint_mutation_token() -> u64 {
    let token = NEXT_CHECKPOINT_MUTATION_TOKEN.fetch_add(1, Ordering::Relaxed);
    assert_ne!(
        token,
        u64::MAX,
        "checkpoint mutation token space was exhausted"
    );
    token
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckpointMutationSummary {
    pub token: u64,
    pub compatible_parent: bool,
    pub tile_chunks: Vec<usize>,
    pub cells: Vec<CellKey>,
    pub earliest_structural_cell: Option<CellKey>,
}

#[derive(Debug, Clone)]
struct CheckpointMutationState {
    tracking: bool,
    current_token: u64,
    #[cfg(not(target_arch = "wasm32"))]
    base_checkpoint_token: u64,
    tile_chunks: BTreeSet<usize>,
    cells: BTreeSet<CellKey>,
    earliest_structural_cell: Option<CellKey>,
}

impl CheckpointMutationState {
    fn new() -> Self {
        let token = fresh_checkpoint_mutation_token();
        Self {
            tracking: false,
            current_token: token,
            #[cfg(not(target_arch = "wasm32"))]
            base_checkpoint_token: token,
            tile_chunks: BTreeSet::new(),
            cells: BTreeSet::new(),
            earliest_structural_cell: None,
        }
    }

    fn touch(&mut self) {
        self.current_token = fresh_checkpoint_mutation_token();
    }
}

/// Host-only mutation summary for trusted incremental planner checkpoints.
/// This is intentionally independent from state-hash dirtiness: consuming a
/// checkpoint summary cannot make a hash clean, and hashing cannot erase the
/// changes needed by a later checkpoint.
#[derive(Debug)]
struct CheckpointMutationTracker(Mutex<CheckpointMutationState>);

impl CheckpointMutationTracker {
    fn new() -> Self {
        Self(Mutex::new(CheckpointMutationState::new()))
    }

    fn mark_tile(&mut self, tile: TileIndex) {
        self.mark_tiles(std::iter::once(tile));
    }

    fn mark_tiles<I>(&mut self, tiles: I)
    where
        I: IntoIterator<Item = TileIndex>,
    {
        let state = self
            .0
            .get_mut()
            .expect("checkpoint mutation tracker mutex was poisoned");
        if !state.tracking {
            return;
        }
        let mut changed = false;
        for tile in tiles {
            state.tile_chunks.insert(tile.0 / CHECKPOINT_TILE_CHUNK_LEN);
            changed = true;
        }
        if changed {
            state.touch();
        }
    }

    fn mark_cell(&mut self, cell: CellKey) {
        self.mark_cells(std::iter::once(cell));
    }

    fn mark_cells<I>(&mut self, cells: I)
    where
        I: IntoIterator<Item = CellKey>,
    {
        let state = self
            .0
            .get_mut()
            .expect("checkpoint mutation tracker mutex was poisoned");
        if !state.tracking {
            return;
        }
        let mut changed = false;
        for cell in cells {
            state.cells.insert(cell);
            changed = true;
        }
        if changed {
            state.touch();
        }
    }

    fn mark_cell_structure(&mut self, cell: CellKey) {
        let state = self
            .0
            .get_mut()
            .expect("checkpoint mutation tracker mutex was poisoned");
        if !state.tracking {
            return;
        }
        state.touch();
        state.cells.insert(cell);
        state.earliest_structural_cell = Some(
            state
                .earliest_structural_cell
                .map_or(cell, |earliest| earliest.min(cell)),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn summary(&self, parent_token: Option<u64>) -> CheckpointMutationSummary {
        let mut state = self
            .0
            .lock()
            .expect("checkpoint mutation tracker mutex was poisoned");
        let compatible_parent = state.tracking && parent_token == Some(state.base_checkpoint_token);
        let summary = CheckpointMutationSummary {
            token: state.current_token,
            compatible_parent,
            tile_chunks: if compatible_parent {
                state.tile_chunks.iter().copied().collect()
            } else {
                Vec::new()
            },
            cells: if compatible_parent {
                state.cells.iter().copied().collect()
            } else {
                Vec::new()
            },
            earliest_structural_cell: compatible_parent
                .then_some(state.earliest_structural_cell)
                .flatten(),
        };
        state.tracking = true;
        state.base_checkpoint_token = state.current_token;
        state.tile_chunks.clear();
        state.cells.clear();
        state.earliest_structural_cell = None;
        summary
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn adopt_checkpoint_token(&mut self, token: u64) {
        *self
            .0
            .get_mut()
            .expect("checkpoint mutation tracker mutex was poisoned") = CheckpointMutationState {
            tracking: true,
            current_token: token,
            #[cfg(not(target_arch = "wasm32"))]
            base_checkpoint_token: token,
            tile_chunks: BTreeSet::new(),
            cells: BTreeSet::new(),
            earliest_structural_cell: None,
        };
    }
}

impl Clone for CheckpointMutationTracker {
    fn clone(&self) -> Self {
        let state = self
            .0
            .lock()
            .expect("checkpoint mutation tracker mutex was poisoned")
            .clone();
        Self(Mutex::new(state))
    }
}

/// Exact derived index of absolute metabolic-exhaustion deadlines.
///
/// The schedule is host acceleration only: canonical cells remain fully
/// materialized and hashes/checkpoints/replays do not include this structure.
/// Ordinary metabolic accrual preserves an absolute deadline, so entries only
/// need replacement when another process changes assimilated energy or the
/// fractional remainder. Sparse key pages track positions and release dead keys.
#[derive(Debug, Clone, Default)]
struct MetabolicExhaustionIndex {
    /// Indexed binary min-heap ordered by `(deadline, cell)`. The parallel
    /// paged position index permits exact replacement without lifetime holes;
    /// dense rebuilds use O(n) heapification.
    heap: Vec<(SimTime, CellKey)>,
    positions: PagedSlots<usize>,
    overflows: BTreeMap<CellKey, ResolutionError>,
}

impl MetabolicExhaustionIndex {
    fn clear(&mut self) {
        self.heap.clear();
        self.positions.clear();
        self.overflows.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn replace_events(&mut self, events: Vec<(SimTime, CellKey)>) {
        self.clear();
        self.heap = events;
        for (position, (_, key)) in self.heap.iter().enumerate() {
            self.positions.insert(key.0, position);
        }
        self.heapify();
    }

    fn heapify(&mut self) {
        self.compact_heap();
        for index in (0..self.heap.len() / 2).rev() {
            self.sift_down(index);
        }
    }

    fn compact_heap(&mut self) {
        if self.heap.capacity() > self.heap.len().saturating_mul(4).max(1024) {
            self.heap.shrink_to_fit();
        }
    }

    fn swap_heap(&mut self, left: usize, right: usize) {
        self.heap.swap(left, right);
        self.positions.insert(self.heap[left].1 .0, left);
        self.positions.insert(self.heap[right].1 .0, right);
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.heap[parent] <= self.heap[index] {
                break;
            }
            self.swap_heap(parent, index);
            index = parent;
        }
    }

    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = index.saturating_mul(2).saturating_add(1);
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let smallest = if right < self.heap.len() && self.heap[right] < self.heap[left] {
                right
            } else {
                left
            };
            if self.heap[index] <= self.heap[smallest] {
                break;
            }
            self.swap_heap(index, smallest);
            index = smallest;
        }
    }

    fn remove_current(&mut self, key: CellKey) {
        let Some(position) = self.positions.remove(key.0) else {
            return;
        };
        if position == OVERFLOWED_METABOLIC_SCHEDULE {
            self.overflows.remove(&key);
        } else {
            let removed = self.heap.swap_remove(position);
            debug_assert_eq!(removed.1, key);
            if position < self.heap.len() {
                self.positions.insert(self.heap[position].1 .0, position);
                if position > 0 && self.heap[position] < self.heap[(position - 1) / 2] {
                    self.sift_up(position);
                } else {
                    self.sift_down(position);
                }
            }
        }
        self.compact_heap();
    }

    fn insert_event(&mut self, key: CellKey, time: SimTime) {
        let position = self.heap.len();
        self.heap.push((time, key));
        self.positions.insert(key.0, position);
        self.sift_up(position);
    }

    fn sync(
        &mut self,
        key: CellKey,
        cell: Option<&CellState>,
        now: SimTime,
        numerator: u64,
        denominator: u64,
    ) {
        self.remove_current(key);

        match cell {
            Some(cell) if numerator > 0 && cell.assimilated_energy > 0 => {
                match metabolic_exhaustion_time(now, cell, numerator, denominator) {
                    Ok(time) => {
                        self.insert_event(key, time);
                    }
                    Err(error) => {
                        self.overflows.insert(key, error);
                        self.positions.insert(key.0, OVERFLOWED_METABOLIC_SCHEDULE);
                    }
                }
            }
            _ => {}
        }
    }

    fn rebuild(&mut self, cells: &CellStore, now: SimTime, numerator: u64, denominator: u64) {
        self.clear();
        for (key, cell) in cells {
            if numerator == 0 || cell.assimilated_energy == 0 {
                continue;
            }
            match metabolic_exhaustion_time(now, cell, numerator, denominator) {
                Ok(time) => {
                    self.positions.insert(key.0, self.heap.len());
                    self.heap.push((time, *key));
                }
                Err(error) => {
                    self.positions.insert(key.0, OVERFLOWED_METABOLIC_SCHEDULE);
                    self.overflows.insert(*key, error);
                }
            }
        }
        self.heapify();
    }

    fn next_event(&self) -> Result<Option<SimTime>, ResolutionError> {
        if let Some((_, error)) = self.overflows.first_key_value() {
            return Err(error.clone());
        }
        Ok(self.heap.first().map(|(time, _)| *time))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TileState {
    pub elevation: i16,
    pub occupant: Option<CellKey>,
    pub plant_energy: u64,
    /// Energy this plant can hold. A zero growth rate represents inert plant
    /// matter and does not require a nonzero capacity.
    pub plant_capacity: u64,
    /// Energy grown per simulation time unit. Growth conservatively converts
    /// diffuse energy on this tile into plant energy.
    pub plant_growth_rate: u64,
    /// Fractional growth numerator carried between event-time advances.
    pub plant_growth_remainder: u64,
    pub loose_energy: u64,
    pub diffuse_energy: u64,
    pub diffusion_remainder: u64,
    pub signal_energy: [u64; REFERENCE_SIGNAL_CHANNELS],
    pub signal_decay_remainder: [u64; REFERENCE_SIGNAL_CHANNELS],
}

pub(super) const REFERENCE_TILE_CHUNK_LEN: usize = 8;
pub(super) const REFERENCE_TILE_PAGE_CHUNKS: usize = 256;
#[cfg(not(target_arch = "wasm32"))]
const REFERENCE_TILE_PAGE_LEN: usize = REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS;
#[cfg(not(target_arch = "wasm32"))]
const REFERENCE_TILE_PAGE_BITMAP_WORDS: usize = REFERENCE_TILE_PAGE_LEN.div_ceil(64);
pub(super) type ReferenceTileChunk = Arc<[TileState]>;

/// One contiguous tile page plus sparse eight-tile COW replacements. Fresh
/// pages and pages prepared for dense mutation have no overlays, preserving
/// page-local traversal while sparse branches copy only a logical chunk.
#[derive(Debug, Clone)]
pub(super) struct ReferenceTilePageData {
    base: Arc<[TileState]>,
    overlays: Vec<Option<ReferenceTileChunk>>,
}

pub(super) enum ReferenceTilePageIter<'a> {
    Contiguous(std::slice::Iter<'a, TileState>),
    Overlaid {
        page: &'a ReferenceTilePageData,
        range: Range<usize>,
    },
}

impl<'a> Iterator for ReferenceTilePageIter<'a> {
    type Item = &'a TileState;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(iter) => iter.next(),
            Self::Overlaid { page, range } => range.next().map(|index| page.get(index)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl DoubleEndedIterator for ReferenceTilePageIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(iter) => iter.next_back(),
            Self::Overlaid { page, range } => range.next_back().map(|index| page.get(index)),
        }
    }
}

impl ExactSizeIterator for ReferenceTilePageIter<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Contiguous(iter) => iter.len(),
            Self::Overlaid { range, .. } => range.len(),
        }
    }
}

impl ReferenceTilePageData {
    fn from_vec(tiles: Vec<TileState>) -> Self {
        let chunk_count = tiles.len().div_ceil(REFERENCE_TILE_CHUNK_LEN);
        Self {
            base: Arc::from(tiles),
            overlays: vec![None; chunk_count],
        }
    }

    pub(super) fn len(&self) -> usize {
        self.base.len()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    pub(super) fn chunk_count(&self) -> usize {
        self.overlays.len()
    }

    pub(super) fn chunk(&self, chunk_index: usize) -> &[TileState] {
        if let Some(overlay) = self.overlays[chunk_index].as_deref() {
            return overlay;
        }
        let start = chunk_index * REFERENCE_TILE_CHUNK_LEN;
        &self.base[start..(start + REFERENCE_TILE_CHUNK_LEN).min(self.base.len())]
    }

    pub(super) fn chunks(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[TileState]> + ExactSizeIterator {
        (0..self.chunk_count()).map(|chunk_index| self.chunk(chunk_index))
    }

    pub(super) fn iter(&self) -> ReferenceTilePageIter<'_> {
        if self.overlays.iter().all(Option::is_none) {
            ReferenceTilePageIter::Contiguous(self.base.iter())
        } else {
            ReferenceTilePageIter::Overlaid {
                page: self,
                range: 0..self.len(),
            }
        }
    }

    pub(super) fn get(&self, page_offset: usize) -> &TileState {
        let chunk_index = page_offset / REFERENCE_TILE_CHUNK_LEN;
        &self.chunk(chunk_index)[page_offset % REFERENCE_TILE_CHUNK_LEN]
    }

    fn get_mut(&mut self, page_offset: usize) -> &mut TileState {
        let chunk_index = page_offset / REFERENCE_TILE_CHUNK_LEN;
        let chunk_offset = page_offset % REFERENCE_TILE_CHUNK_LEN;
        if self.overlays[chunk_index].is_none() {
            if Arc::strong_count(&self.base) == 1 {
                return &mut Arc::get_mut(&mut self.base)
                    .expect("single-owner tile page base was not mutable")[page_offset];
            }
            self.overlays[chunk_index] = Some(Arc::from(self.chunk(chunk_index).to_vec()));
        }
        &mut Arc::make_mut(
            self.overlays[chunk_index]
                .as_mut()
                .expect("tile overlay was not installed"),
        )[chunk_offset]
    }

    fn dense_tiles_mut(&mut self) -> &mut [TileState] {
        if self.overlays.iter().any(Option::is_some) {
            self.base = Arc::from(
                self.chunks()
                    .flat_map(|chunk| chunk.iter().cloned())
                    .collect::<Vec<_>>(),
            );
            self.overlays.fill(None);
        }
        Arc::make_mut(&mut self.base)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn valid_geometry(&self) -> bool {
        if self.is_empty()
            || self.len() > REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS
            || self.chunk_count() != self.len().div_ceil(REFERENCE_TILE_CHUNK_LEN)
        {
            return false;
        }
        self.overlays.iter().enumerate().all(|(index, overlay)| {
            overlay.as_ref().is_none_or(|chunk| {
                let start = index * REFERENCE_TILE_CHUNK_LEN;
                let expected_len = self
                    .base
                    .len()
                    .saturating_sub(start)
                    .min(REFERENCE_TILE_CHUNK_LEN);
                !chunk.is_empty()
                    && chunk.len() == expected_len
                    && chunk.len() <= REFERENCE_TILE_CHUNK_LEN
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn chunk_allocation(&self, chunk_index: usize) -> (*const TileState, usize) {
        let chunk = self.chunk(chunk_index);
        (chunk.as_ptr(), chunk.len())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn shares_chunk_with(&self, other: &Self, chunk_index: usize) -> bool {
        match (
            self.overlays[chunk_index].as_ref(),
            other.overlays[chunk_index].as_ref(),
        ) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, None) => Arc::ptr_eq(&self.base, &other.base),
            _ => false,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn overlay_allocation_bytes(&self) -> usize {
        self.overlays
            .iter()
            .flatten()
            .map(|chunk| chunk.len().saturating_mul(std::mem::size_of::<TileState>()))
            .sum()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn inline_heap_bytes(&self) -> usize {
        self.overlays
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<ReferenceTileChunk>>())
    }
}

impl PartialEq for ReferenceTilePageData {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for ReferenceTilePageData {}

pub(super) type ReferenceTilePage = Arc<ReferenceTilePageData>;

/// Canonical row-major tile storage with copy-on-write planner-sized chunks.
/// Public encodings remain flat; this host representation is derived and does
/// not participate in hashes, replay, or Mind observations.
#[derive(Debug, Clone)]
pub struct ReferenceTileStore {
    pages: Vec<ReferenceTilePage>,
    len: usize,
}

impl ReferenceTileStore {
    fn from_vec(tiles: Vec<TileState>) -> Self {
        let len = tiles.len();
        let mut values = tiles.into_iter();
        let mut pages =
            Vec::with_capacity(len.div_ceil(REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS));
        loop {
            let page = values
                .by_ref()
                .take(REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS)
                .collect::<Vec<_>>();
            if page.is_empty() {
                break;
            }
            pages.push(Arc::new(ReferenceTilePageData::from_vec(page)));
        }
        Self { pages, len }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn from_pages(pages: Vec<ReferenceTilePage>, len: usize) -> Option<Self> {
        let actual_len = pages.iter().map(|page| page.len()).sum::<usize>();
        if actual_len != len || (len > 0 && pages.is_empty()) {
            return None;
        }
        for (page_index, page) in pages.iter().enumerate() {
            let last_page = page_index + 1 == pages.len();
            if !page.valid_geometry()
                || (!last_page
                    && page.len() != REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS)
            {
                return None;
            }
        }
        Some(Self { pages, len })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn pages(&self) -> &[ReferenceTilePage] {
        &self.pages
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> Option<&TileState> {
        (index < self.len).then(|| &self[index])
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut TileState> {
        if index >= self.len {
            return None;
        }
        let page_tile_len = REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS;
        let page_offset = index % page_tile_len;
        let page = Arc::make_mut(&mut self.pages[index / page_tile_len]);
        Some(page.get_mut(page_offset))
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &TileState> {
        self.pages.iter().flat_map(|page| page.iter())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn par_iter(&self) -> impl rayon::iter::ParallelIterator<Item = &TileState> {
        self.pages.par_iter().flat_map_iter(|page| page.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut TileState> {
        self.pages
            .iter_mut()
            .flat_map(|page| Arc::make_mut(page).dense_tiles_mut().iter_mut())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn par_iter_mut(&mut self) -> impl rayon::iter::ParallelIterator<Item = &mut TileState> {
        self.pages
            .par_iter_mut()
            .flat_map_iter(|page| Arc::make_mut(page).dense_tiles_mut().iter_mut())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn par_for_each_mut_indexed(
        &mut self,
        operation: impl Fn(usize, &mut TileState) + Sync + Send,
    ) {
        let page_tile_len = REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS;
        self.pages
            .par_iter_mut()
            .enumerate()
            .for_each(|(page_index, page)| {
                for (tile_offset, tile) in
                    Arc::make_mut(page).dense_tiles_mut().iter_mut().enumerate()
                {
                    operation(page_index * page_tile_len + tile_offset, tile);
                }
            });
    }

    pub fn to_vec(&self) -> Vec<TileState> {
        self.iter().cloned().collect()
    }
}

impl PartialEq for ReferenceTileStore {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl Eq for ReferenceTileStore {}

impl std::ops::Index<usize> for ReferenceTileStore {
    type Output = TileState;

    fn index(&self, index: usize) -> &Self::Output {
        let page_tile_len = REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS;
        let page_offset = index % page_tile_len;
        self.pages[index / page_tile_len].get(page_offset)
    }
}

impl std::ops::IndexMut<usize> for ReferenceTileStore {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("tile index is out of bounds")
    }
}

impl<'a> IntoIterator for &'a ReferenceTileStore {
    type Item = &'a TileState;
    type IntoIter = Box<dyn DoubleEndedIterator<Item = &'a TileState> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityCue {
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
pub enum ProgressBucket {
    Early,
    Middle,
    Late,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborCue {
    pub marker: Option<u32>,
    pub apparent_mass_bucket: Option<u8>,
    pub activity: Option<ActivityCue>,
    pub progress: Option<ProgressBucket>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourceKey {
    Occupancy(TileIndex),
    CellState(CellKey),
    PlantEnergy(TileIndex),
    LooseEnergy(TileIndex),
    DiffuseEnergy(TileIndex),
    Terrain(TileIndex),
    Signal(TileIndex),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessMode {
    ReadSnapshot,
    Add,
    BoundedTake,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceClaim {
    pub actor: CellKey,
    pub key: ResourceKey,
    pub mode: AccessMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttackDamage {
    pub victim: CellKey,
    pub target_was_guarded: bool,
    /// This attack's payload/mass contribution before guard mitigation.
    pub raw: u64,
    /// This attack's deterministic share removed by guard mitigation.
    pub mitigated: u64,
    /// This attack's deterministic share actually removed from the victim.
    pub applied: u64,
    /// Post-guard damage that exceeded the victim's remaining energy.
    pub overkill: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerrainChange {
    pub tile: TileIndex,
    pub elevation_before: i16,
    pub elevation_after: i16,
    /// Conserved material mass moved between terrain and the actor.
    pub material_mass: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionOutcome {
    pub actor: CellKey,
    pub action: ActionKind,
    pub request: ActionRequest,
    pub origin: TileIndex,
    pub status: OutcomeStatus,
    pub started_at: SimTime,
    pub completed_at: SimTime,
    pub target: Option<TileIndex>,
    pub effort_spent: u64,
    pub payload: u64,
    /// Exact plant-plus-loose energy transferred into the actor's gut by a
    /// successful consume. Zero for every other action and for a consume that
    /// found no remaining food at its completion snapshot.
    pub consumed_energy: u64,
    /// Present only for a successful attack that reached a snapshot victim.
    pub attack_damage: Option<AttackDamage>,
    /// Present only for a successful terrain excavation or deposition.
    pub terrain_change: Option<TerrainChange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchReport {
    pub completed_at: SimTime,
    pub outcomes: Vec<ActionOutcome>,
    pub claims: Vec<ResourceClaim>,
    pub deaths: Vec<CellKey>,
    pub births: Vec<(CellKey, CellKey)>,
    pub compiled_ruleset_hash: CanonicalHash,
    pub integrity: BatchIntegrity,
    pub delta: SimulationDelta,
}

impl BatchReport {
    /// True when this report carries the authenticated state boundary needed
    /// for replay recording and external verification.
    pub const fn is_verified(&self) -> bool {
        matches!(self.integrity, BatchIntegrity::Verified { .. })
    }

    pub const fn pre_state_hash(&self) -> Option<CanonicalHash> {
        match self.integrity {
            BatchIntegrity::Verified { pre_state_hash, .. } => Some(pre_state_hash),
            BatchIntegrity::Unverified => None,
        }
    }

    pub const fn state_hash(&self) -> Option<CanonicalHash> {
        match self.integrity {
            BatchIntegrity::Verified { state_hash, .. } => Some(state_hash),
            BatchIntegrity::Unverified => None,
        }
    }
}

/// Authentication boundary carried by a completed batch report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchIntegrity {
    Verified {
        pre_state_hash: CanonicalHash,
        state_hash: CanonicalHash,
    },
    Unverified,
}

/// Host execution policy for state commitments. This is deliberately absent
/// from canonical state and the ruleset: both modes execute identical physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrityMode {
    /// Compute the pre- and post-state commitments for every resolved batch.
    #[default]
    Verified,
    /// Maintain incremental dirty metadata, but hash only when explicitly
    /// requested through [`ReferenceSimulation::state_hash`].
    OnDemand,
}

/// Non-authoritative diagnostics for the most recently resolved batch.
/// These values never enter state hashes, checkpoints, or replay frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResolutionMetrics {
    pub due_intents: usize,
    pub emitted_claims: usize,
    pub exclusive_components: usize,
    pub contended_components: usize,
    pub maximum_component_size: usize,
    pub parallel_validation: bool,
    /// True when occupancy contention used storage proportional to the number
    /// of claims rather than the number of world tiles.
    pub sparse_occupancy_claims: bool,
    /// Exact conserved signal energy moved into diffuse energy by the passive
    /// kernel. Host telemetry only; never canonical or replayed.
    pub signal_energy_decayed: [u128; REFERENCE_SIGNAL_CHANNELS],
    pub phase_timings: ResolutionPhaseTimings,
}

/// Wall-clock diagnostics for one batch. These values are deliberately
/// excluded from canonical state, hashes, checkpoints, and replay data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResolutionPhaseTimings {
    pub passive_updates_ns: u64,
    pub prestate_materialization_ns: u64,
    pub intent_validation_ns: u64,
    pub canonical_resolution_ns: u64,
    pub report_finalization_ns: u64,
    /// Elapsed wall time for the pre-resolution frontier hash and incremental
    /// post-resolution commitment.
    pub state_hashes_ns: u64,
    pub poststate_materialization_ns: u64,
    pub delta_generation_ns: u64,
    pub total_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitReceipt {
    pub actor: CellKey,
    pub action: ActionKind,
    pub accepted: bool,
    pub rejection: Option<RejectReason>,
    pub completes_at: SimTime,
    pub effort_spent: u64,
    pub payload_escrow: u64,
}

/// One actor-ordered pristine Mind output awaiting authoritative commitment.
/// The batch API borrows these values so the host can move them directly into
/// replay records after canonical state has been updated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionCommitment {
    pub actor: CellKey,
    pub request: ActionRequest,
    pub signal: Option<ReferenceSignalEmission>,
    pub memory_update: ReferenceMemoryUpdate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    UnknownCell(CellKey),
    CellNotReady { actor: CellKey, ready_at: SimTime },
    PrivateMemoryTooLarge { actual: usize, limit: usize },
    ActorsNotStrictlyOrdered { previous: CellKey, actor: CellKey },
    InvalidSignal(&'static str),
    InvariantViolation(&'static str),
}

impl Display for CommitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownCell(actor) => write!(f, "unknown cell {actor:?}"),
            Self::CellNotReady { actor, ready_at } => {
                write!(f, "cell {actor:?} is not ready until {ready_at:?}")
            }
            Self::PrivateMemoryTooLarge { actual, limit } => write!(
                f,
                "reference Mind private memory is {actual} bytes; limit is {limit}"
            ),
            Self::ActorsNotStrictlyOrdered { previous, actor } => write!(
                f,
                "decision actors are not strictly ordered: {actor:?} follows {previous:?}"
            ),
            Self::InvalidSignal(message) => write!(f, "invalid signal emission: {message}"),
            Self::InvariantViolation(message) => write!(f, "invariant violation: {message}"),
        }
    }
}

impl Error for CommitError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionError {
    Neighborhood(NeighborhoodError),
    InvalidRuleset(&'static str),
    InvalidTile(TileIndex),
    OccupiedTile(TileIndex),
    NoScheduledEvents,
    TimeMovedBackwards {
        now: SimTime,
        requested: SimTime,
    },
    CompletionBeforeRequestedTime {
        completion: SimTime,
        requested: SimTime,
    },
    ArithmeticOverflow(&'static str),
    InvalidState(&'static str),
    InvariantViolation(&'static str),
}

impl Display for ResolutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Neighborhood(error) => Display::fmt(error, f),
            Self::InvalidRuleset(message) => write!(f, "invalid ruleset: {message}"),
            Self::InvalidTile(tile) => write!(f, "invalid tile {tile:?}"),
            Self::OccupiedTile(tile) => write!(f, "tile {tile:?} is occupied"),
            Self::NoScheduledEvents => {
                write!(f, "there are no scheduled actions or passive events")
            }
            Self::TimeMovedBackwards { now, requested } => {
                write!(
                    f,
                    "cannot move time backwards from {now:?} to {requested:?}"
                )
            }
            Self::CompletionBeforeRequestedTime {
                completion,
                requested,
            } => write!(
                f,
                "completion at {completion:?} must resolve before advancing to {requested:?}"
            ),
            Self::ArithmeticOverflow(context) => {
                write!(f, "authoritative arithmetic overflow while {context}")
            }
            Self::InvalidState(message) => write!(f, "invalid canonical state: {message}"),
            Self::InvariantViolation(message) => write!(f, "invariant violation: {message}"),
        }
    }
}

impl Error for ResolutionError {}

impl From<NeighborhoodError> for ResolutionError {
    fn from(value: NeighborhoodError) -> Self {
        Self::Neighborhood(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceRuleset {
    pub neighborhood: NeighborhoodSpec,
    pub time: TimeConfig,
    pub effort_profiles: [EffortProfile; 3],
    pub wait_duration: DurationRule,
    pub move_duration: DurationRule,
    pub attack_duration: DurationRule,
    pub consume_duration: DurationRule,
    pub split_duration: DurationRule,
    pub regurgitate_duration: DurationRule,
    pub excavate_duration: DurationRule,
    pub deposit_terrain_duration: DurationRule,
    pub move_effort_base: u64,
    pub move_mass_units_per_effort: u64,
    pub attack_effort_base: u64,
    pub guard_effort_base: u64,
    pub consume_effort_base: u64,
    pub split_effort_base: u64,
    pub regurgitate_effort_base: u64,
    pub excavate_effort_base: u64,
    pub deposit_terrain_effort_base: u64,
    pub bite_capacity: u64,
    pub gut_capacity: u64,
    /// Energy digested per arithmetic quantum is numerator / denominator.
    pub digestion_rate_numerator: u64,
    pub digestion_rate_denominator: u64,
    /// Assimilated energy spent per arithmetic quantum is numerator /
    /// denominator. Spent upkeep becomes diffuse energy at the cell's current
    /// tile.
    pub metabolism_rate_numerator: u64,
    pub metabolism_rate_denominator: u64,
    /// Conserved energy quantum for signal deposits. Zero disables signaling;
    /// every nonzero channel amount must be an exact multiple of this value.
    pub signal_emission_cost: u64,
    /// Signal energy decayed per arithmetic quantum.
    pub signal_decay_rate_numerator: u64,
    pub signal_decay_rate_denominator: u64,
    /// Fixed interval and fraction for one synchronous diffuse transport step.
    pub diffusion_interval_quanta: u64,
    pub diffusion_rate_numerator: u64,
    pub diffusion_rate_denominator: u64,
    pub diffusion_targets: SlotMask,
    /// Mass-energy represented by one signed elevation unit.
    pub terrain_mass_per_elevation: u64,
    pub minimum_survival_energy: u64,
    pub child_core_mass: u64,
    pub maximum_elevation_delta: i16,
    pub guard_damage_numerator: u64,
    pub guard_damage_denominator: u64,
    pub attack_payload_damage_numerator: u64,
    pub attack_payload_damage_denominator: u64,
    pub attack_mass_damage_numerator: u64,
    pub attack_mass_damage_denominator: u64,
    pub apparent_mass_bucket_width: u64,
    pub max_private_memory_bytes: usize,
}

impl Default for ReferenceRuleset {
    fn default() -> Self {
        use super::neighborhood::BoundaryRule;

        Self {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
            time: TimeConfig::default(),
            effort_profiles: [
                EffortProfile::new(1, 2, 5, 4),
                EffortProfile::new(1, 1, 1, 1),
                EffortProfile::new(2, 1, 3, 4),
            ],
            wait_duration: DurationRule::new(1024, 0, 1),
            move_duration: DurationRule::new(768, 4, 1),
            attack_duration: DurationRule::new(640, 3, 1),
            consume_duration: DurationRule::new(768, 1, 1),
            split_duration: DurationRule::new(1024, 2, 1),
            regurgitate_duration: DurationRule::new(768, 1, 1),
            excavate_duration: DurationRule::new(1024, 2, 1),
            deposit_terrain_duration: DurationRule::new(1024, 2, 1),
            move_effort_base: 1,
            move_mass_units_per_effort: 100,
            attack_effort_base: 2,
            guard_effort_base: 1,
            consume_effort_base: 1,
            split_effort_base: 2,
            regurgitate_effort_base: 1,
            excavate_effort_base: 2,
            deposit_terrain_effort_base: 2,
            bite_capacity: 16,
            gut_capacity: 64,
            digestion_rate_numerator: 1,
            digestion_rate_denominator: 256,
            metabolism_rate_numerator: 1,
            metabolism_rate_denominator: 1024,
            signal_emission_cost: 1,
            signal_decay_rate_numerator: 1,
            signal_decay_rate_denominator: 1024,
            diffusion_interval_quanta: 1024,
            diffusion_rate_numerator: 1,
            diffusion_rate_denominator: 4,
            diffusion_targets: SlotMask::from_slots([
                LocalSlot(1),
                LocalSlot(3),
                LocalSlot(4),
                LocalSlot(6),
            ]),
            terrain_mass_per_elevation: 10,
            minimum_survival_energy: 1,
            child_core_mass: 10,
            maximum_elevation_delta: 1,
            guard_damage_numerator: 1,
            guard_damage_denominator: 2,
            attack_payload_damage_numerator: 1,
            attack_payload_damage_denominator: 1,
            attack_mass_damage_numerator: 0,
            attack_mass_damage_denominator: 1,
            apparent_mass_bucket_width: 32,
            max_private_memory_bytes: 2048,
        }
    }
}

impl ReferenceRuleset {
    pub fn semantic_hash(&self) -> CanonicalHash {
        semantic_ruleset_hash(self)
    }

    pub fn validate(&self) -> Result<(), ResolutionError> {
        if self.max_private_memory_bytes > u32::MAX as usize {
            return Err(ResolutionError::InvalidRuleset(
                "maximum private memory must fit the reference Mind ABI",
            ));
        }
        if self.time.arithmetic_quanta_per_unit == 0 {
            return Err(ResolutionError::InvalidRuleset(
                "arithmetic quanta per unit must be positive",
            ));
        }
        if self.time.completion_bucket == 0 {
            return Err(ResolutionError::InvalidRuleset(
                "completion bucket must be positive",
            ));
        }
        if self.time.decision_interval_floor == 0 {
            return Err(ResolutionError::InvalidRuleset(
                "decision interval floor must be positive",
            ));
        }
        if self.move_mass_units_per_effort == 0
            || self.guard_damage_denominator == 0
            || self.attack_payload_damage_denominator == 0
            || self.attack_mass_damage_denominator == 0
            || self.apparent_mass_bucket_width == 0
            || self.digestion_rate_denominator == 0
            || self.metabolism_rate_denominator == 0
            || self.signal_decay_rate_denominator == 0
            || self.diffusion_interval_quanta == 0
            || self.diffusion_rate_denominator == 0
            || self.terrain_mass_per_elevation == 0
        {
            return Err(ResolutionError::InvalidRuleset(
                "ruleset denominators must be positive",
            ));
        }
        if self.guard_damage_numerator > self.guard_damage_denominator {
            return Err(ResolutionError::InvalidRuleset(
                "guard cannot increase incoming damage in the reference kernel",
            ));
        }
        if self.diffusion_rate_numerator > self.diffusion_rate_denominator {
            return Err(ResolutionError::InvalidRuleset(
                "diffusion cannot transport more than the local diffuse reservoir",
            ));
        }
        if self.diffusion_targets.bits() & !SlotMask::first(self.neighborhood.slots.len()).bits()
            != 0
        {
            return Err(ResolutionError::InvalidRuleset(
                "diffusion mask names an absent local slot",
            ));
        }
        for profile in self.effort_profiles {
            if profile.cost_denominator == 0 || profile.duration_denominator == 0 {
                return Err(ResolutionError::InvalidRuleset(
                    "effort profile denominators must be positive",
                ));
            }
        }
        for duration in [
            self.wait_duration,
            self.move_duration,
            self.attack_duration,
            self.consume_duration,
            self.split_duration,
            self.regurgitate_duration,
            self.excavate_duration,
            self.deposit_terrain_duration,
        ] {
            if duration.mass_units_denominator == 0 {
                return Err(ResolutionError::InvalidRuleset(
                    "duration mass denominator must be positive",
                ));
            }
        }
        Ok(())
    }

    /// Validate both scalar rules and the neighborhood compiled for a specific
    /// board shape. Configuration loaders should use this before starting a
    /// run so malformed topology fails before environment construction.
    pub fn validate_for_world(&self, width: usize, height: usize) -> Result<(), ResolutionError> {
        self.validate()?;
        self.neighborhood.clone().compile(width, height)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationState {
    pub now: SimTime,
    pub tiles: Vec<TileState>,
    pub cells: Vec<(CellKey, CellState)>,
    pub next_cell_key: u64,
}

struct ReferenceRestoreState {
    now: SimTime,
    tiles: ReferenceTileStore,
    cells: Vec<(CellKey, CellState)>,
    next_cell_key: u64,
}

impl From<SimulationState> for ReferenceRestoreState {
    fn from(state: SimulationState) -> Self {
        Self {
            now: state.now,
            tiles: ReferenceTileStore::from_vec(state.tiles),
            cells: state.cells,
            next_cell_key: state.next_cell_key,
        }
    }
}

impl SimulationState {
    pub fn hash_with_compiled_ruleset(
        &self,
        compiled_ruleset_hash: CanonicalHash,
    ) -> CanonicalHash {
        canonical_state_hash(
            compiled_ruleset_hash,
            self.now,
            &self.tiles,
            &self.cells,
            self.next_cell_key,
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Default)]
struct DenseMetabolismDeposits(Vec<AtomicU64>);

#[cfg(not(target_arch = "wasm32"))]
impl DenseMetabolismDeposits {
    fn prepare(&mut self, len: usize) {
        if self.0.len() != len {
            self.0 = (0..len).map(|_| AtomicU64::new(0)).collect();
        } else {
            debug_assert!(self
                .0
                .iter()
                .all(|deposit| deposit.load(Ordering::Relaxed) == 0));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Clone for DenseMetabolismDeposits {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct ReferenceSimulation {
    rules: ReferenceRuleset,
    neighborhood: Arc<CompiledNeighborhood>,
    now: SimTime,
    tiles: ReferenceTileStore,
    /// Derived acceleration structure. Canonical state and hashes remain the
    /// dense tile array; this index only avoids scanning inert tiles on every
    /// passive-time advance.
    growing_plant_tiles: Vec<TileIndex>,
    growing_plant_index_dirty: bool,
    /// Exact canonical-order frontiers for sparse passive field work. These
    /// are derived from `tiles` and never enter canonical state.
    active_signal_tiles: BTreeSet<TileIndex>,
    active_diffuse_tiles: BTreeSet<TileIndex>,
    field_frontiers_dirty: bool,
    /// Exact derived indexes for cells whose canonical passive state can
    /// advance. They are rebuilt from cell energy on checkpoint restore and
    /// never enter canonical state, hashes, or replay data.
    active_digestion_cells: BTreeSet<CellKey>,
    active_metabolism_cells: BTreeSet<CellKey>,
    zero_energy_cells: BTreeSet<CellKey>,
    /// Exact derived minimum-deadline index for metabolic exhaustion. It is
    /// rebuilt from canonical cells on restore and rollback.
    metabolic_exhaustion: MetabolicExhaustionIndex,
    /// Derived, canonical-order destinations for synchronous diffuse flux.
    diffusion_neighbors: Arc<Vec<Vec<TileIndex>>>,
    /// Reverse adjacency for contention-free dense diffusion gathering.
    #[cfg(not(target_arch = "wasm32"))]
    diffusion_sources: Arc<Vec<Vec<TileIndex>>>,
    /// Reused dense accumulator with sparse clearing through `diffusion_touched`.
    diffusion_incoming: Vec<u64>,
    diffusion_touched: Vec<TileIndex>,
    /// Reused host-only dense metabolism deposits indexed by tile. Clones
    /// intentionally start empty so planner branches do not duplicate scratch.
    #[cfg(not(target_arch = "wasm32"))]
    dense_metabolism_deposits: DenseMetabolismDeposits,
    /// Host-only crossovers for dense passive kernels. These never enter
    /// canonical state, hashes, checkpoints, or replay.
    #[cfg(not(target_arch = "wasm32"))]
    passive_cell_parallel_threshold: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    passive_tile_parallel_threshold: Option<usize>,
    cells: CellStore,
    /// Derived exact completion index. It is rebuilt from canonical cells and
    /// never enters hashes or checkpoints.
    due_actions: BTreeMap<SimTime, Vec<CellKey>>,
    next_cell_key: u64,
    state_hash_cache: StateHashCache,
    checkpoint_mutations: CheckpointMutationTracker,
    last_resolution_metrics: ResolutionMetrics,
}

/// Immutable geometry-dependent resolver data. Hosts that repeatedly restore
/// checkpoints for the same board and ruleset may retain this opaque handle;
/// canonical state, hashes, and Mind-visible inputs remain simulation-local.
#[derive(Debug, Clone)]
pub struct ReferenceCompiledTopology {
    neighborhood: Arc<CompiledNeighborhood>,
    diffusion_neighbors: Arc<Vec<Vec<TileIndex>>>,
    #[cfg(not(target_arch = "wasm32"))]
    diffusion_sources: Arc<Vec<Vec<TileIndex>>>,
    compiled_ruleset_hash: CanonicalHash,
}

/// Nested host-only timings for rebuilding mutable resolver state around an
/// already compiled topology. These phases are diagnostic only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ReferenceReconstructionProfile {
    pub topology_validation_ns: u64,
    pub cell_validation_store_and_passive_index_ns: u64,
    pub tile_validation_and_passive_index_ns: u64,
    pub hash_initialization_ns: u64,
    pub scratch_initialization_ns: u64,
    pub metabolic_index_ns: u64,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct CanonicalTileIndexes {
    growing_plants: Vec<TileIndex>,
    active_signals: Vec<TileIndex>,
    active_diffuse: Vec<TileIndex>,
}

fn validate_canonical_tile(
    tile_index: usize,
    tile: &TileState,
    rules: &ReferenceRuleset,
    cells: &CellStore,
) -> Result<(), &'static str> {
    if tile.plant_growth_rate > 0
        && (tile.plant_capacity == 0 || tile.plant_energy > tile.plant_capacity)
    {
        return Err("growing plant energy exceeds its nonzero capacity");
    }
    if tile.plant_growth_remainder >= rules.time.arithmetic_quanta_per_unit {
        return Err("plant growth remainder exceeds its denominator");
    }
    if tile.plant_growth_rate == 0 && tile.plant_growth_remainder != 0 {
        return Err("inert plant matter has a growth remainder");
    }
    if tile.diffusion_remainder >= rules.diffusion_rate_denominator {
        return Err("diffusion remainder exceeds its denominator");
    }
    if tile.diffuse_energy == 0 && tile.diffusion_remainder != 0 {
        return Err("tile without diffuse energy has a diffusion remainder");
    }
    for channel in 0..REFERENCE_SIGNAL_CHANNELS {
        if tile.signal_decay_remainder[channel] >= rules.signal_decay_rate_denominator {
            return Err("signal decay remainder exceeds its denominator");
        }
        if tile.signal_energy[channel] == 0 && tile.signal_decay_remainder[channel] != 0 {
            return Err("empty signal channel has a decay remainder");
        }
    }
    if let Some(key) = tile.occupant {
        let Some(cell) = cells.get(&key) else {
            return Err("occupied tile references a missing cell");
        };
        if cell.position != TileIndex(tile_index) {
            return Err("occupied tile references a cell at another position");
        }
    }
    Ok(())
}

fn validate_canonical_tile_partition(
    tile_offset: usize,
    tile_count: usize,
    tiles: &ReferenceTileStore,
    rules: &ReferenceRuleset,
    cells: &CellStore,
) -> Result<CanonicalTileIndexes, &'static str> {
    let mut indexes = CanonicalTileIndexes::default();
    for tile_index in tile_offset..tile_offset + tile_count {
        let tile = &tiles[tile_index];
        validate_canonical_tile(tile_index, tile, rules, cells)?;
        let tile_index = TileIndex(tile_index);
        if tile.plant_growth_rate > 0 {
            indexes.growing_plants.push(tile_index);
        }
        if tile.signal_energy.iter().any(|energy| *energy > 0) {
            indexes.active_signals.push(tile_index);
        }
        if tile.diffuse_energy > 0 {
            indexes.active_diffuse.push(tile_index);
        }
    }
    Ok(indexes)
}

fn validate_canonical_tiles_serial(
    tiles: &ReferenceTileStore,
    rules: &ReferenceRuleset,
    cells: &CellStore,
) -> Result<CanonicalTileIndexes, ResolutionError> {
    validate_canonical_tile_partition(0, tiles.len(), tiles, rules, cells)
        .map_err(ResolutionError::InvalidState)
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_canonical_tiles_parallel(
    tiles: &ReferenceTileStore,
    rules: &ReferenceRuleset,
    cells: &CellStore,
) -> Result<CanonicalTileIndexes, ResolutionError> {
    let partition_count = tiles
        .len()
        .div_ceil(CANONICAL_TILE_VALIDATION_PARTITION_LEN);
    let partitions: Vec<Result<CanonicalTileIndexes, &'static str>> = (0..partition_count)
        .into_par_iter()
        .map(|partition_index| {
            let tile_offset = partition_index * CANONICAL_TILE_VALIDATION_PARTITION_LEN;
            validate_canonical_tile_partition(
                tile_offset,
                tiles
                    .len()
                    .saturating_sub(tile_offset)
                    .min(CANONICAL_TILE_VALIDATION_PARTITION_LEN),
                tiles,
                rules,
                cells,
            )
        })
        .collect();

    let mut indexes = CanonicalTileIndexes::default();
    for partition in partitions {
        let mut partition = partition.map_err(ResolutionError::InvalidState)?;
        indexes.growing_plants.append(&mut partition.growing_plants);
        indexes.active_signals.append(&mut partition.active_signals);
        indexes.active_diffuse.append(&mut partition.active_diffuse);
    }
    Ok(indexes)
}

fn validate_canonical_tiles(
    tiles: &ReferenceTileStore,
    rules: &ReferenceRuleset,
    cells: &CellStore,
) -> Result<CanonicalTileIndexes, ResolutionError> {
    #[cfg(not(target_arch = "wasm32"))]
    if tiles.len() >= PARALLEL_CANONICAL_TILE_VALIDATION_THRESHOLD
        && rayon::current_num_threads() > 1
    {
        return validate_canonical_tiles_parallel(tiles, rules, cells);
    }
    validate_canonical_tiles_serial(tiles, rules, cells)
}

impl ReferenceCompiledTopology {
    fn compile(
        width: usize,
        height: usize,
        rules: &ReferenceRuleset,
    ) -> Result<Self, ResolutionError> {
        let neighborhood = Arc::new(rules.neighborhood.clone().compile(width, height)?);
        let diffusion_neighbors = Arc::new(compile_diffusion_neighbors(
            &neighborhood,
            rules.diffusion_targets,
        ));
        #[cfg(not(target_arch = "wasm32"))]
        let diffusion_sources = Arc::new(compile_diffusion_sources(
            &diffusion_neighbors,
            neighborhood.tile_count(),
        ));
        Ok(Self {
            compiled_ruleset_hash: compiled_ruleset_hash(rules, &neighborhood),
            neighborhood,
            diffusion_neighbors,
            #[cfg(not(target_arch = "wasm32"))]
            diffusion_sources,
        })
    }

    fn validate_for_rules(&self, rules: &ReferenceRuleset) -> Result<(), ResolutionError> {
        rules.validate()?;
        if compiled_ruleset_hash(rules, &self.neighborhood) != self.compiled_ruleset_hash {
            return Err(ResolutionError::InvalidRuleset(
                "compiled topology does not match the supplied ruleset",
            ));
        }
        Ok(())
    }

    pub fn width(&self) -> usize {
        self.neighborhood.width()
    }

    pub fn height(&self) -> usize {
        self.neighborhood.height()
    }
}

/// First-write journal for the action-resolution portion of one event batch.
/// The immutable completion snapshot remains the semantic read source; this
/// journal retains only resources that are actually written and therefore
/// avoids materializing and comparing a second complete state.
#[derive(Debug)]
struct MutationJournal {
    before_time: SimTime,
    before_next_cell_key: u64,
    previous_metrics: ResolutionMetrics,
    tiles: BTreeMap<TileIndex, TileState>,
    ordered_cells: Vec<(CellKey, Option<CellState>)>,
    extra_cells: BTreeMap<CellKey, Option<CellState>>,
}

impl MutationJournal {
    fn new(
        before_time: SimTime,
        before_next_cell_key: u64,
        previous_metrics: ResolutionMetrics,
    ) -> Self {
        Self {
            before_time,
            before_next_cell_key,
            previous_metrics,
            tiles: BTreeMap::new(),
            ordered_cells: Vec::new(),
            extra_cells: BTreeMap::new(),
        }
    }

    fn record_tile(
        &mut self,
        tile: TileIndex,
        current_tiles: &ReferenceTileStore,
    ) -> Result<(), ResolutionError> {
        let before = current_tiles
            .get(tile.0)
            .ok_or(ResolutionError::InvalidTile(tile))?;
        self.tiles.entry(tile).or_insert_with(|| before.clone());
        Ok(())
    }

    fn record_cell(&mut self, cell: CellKey, current_cells: &CellStore) {
        if self
            .ordered_cells
            .binary_search_by_key(&cell, |(candidate, _)| *candidate)
            .is_ok()
        {
            return;
        }
        self.extra_cells
            .entry(cell)
            .or_insert_with(|| current_cells.get(&cell).cloned());
    }

    fn record_and_complete_cells_ordered<I>(
        &mut self,
        cells: I,
        current_cells: &mut CellStore,
        completion_time: SimTime,
    ) -> Result<(), ResolutionError>
    where
        I: IntoIterator<Item = CellKey>,
    {
        if !self.ordered_cells.is_empty() || !self.extra_cells.is_empty() {
            return Err(ResolutionError::InvariantViolation(
                "ordered journal capture was not the first cell write",
            ));
        }
        for cell in cells {
            let state = current_cells
                .get_mut(&cell)
                .ok_or(ResolutionError::InvariantViolation(
                    "ordered journal actor is absent from cells",
                ))?;
            self.ordered_cells.push((cell, Some(state.clone())));
            state.pending_action = None;
            state.ready_at = completion_time;
        }
        Ok(())
    }

    fn take_cells_ordered(&mut self) -> Vec<(CellKey, Option<CellState>)> {
        if self.extra_cells.is_empty() {
            return std::mem::take(&mut self.ordered_cells);
        }
        if self.ordered_cells.is_empty() {
            return std::mem::take(&mut self.extra_cells).into_iter().collect();
        }
        let ordered = std::mem::take(&mut self.ordered_cells);
        let extra = std::mem::take(&mut self.extra_cells);
        let mut left = ordered.into_iter().peekable();
        let mut right = extra.into_iter().peekable();
        let mut merged = Vec::with_capacity(left.len().saturating_add(right.len()));
        while left.peek().is_some() || right.peek().is_some() {
            let take_left = match (left.peek(), right.peek()) {
                (Some((left_key, _)), Some((right_key, _))) => left_key < right_key,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_left {
                merged.push(left.next().expect("peeked journal entry disappeared"));
            } else {
                merged.push(right.next().expect("peeked journal entry disappeared"));
            }
        }
        merged
    }

    fn finish(
        &mut self,
        after_time: SimTime,
        after_next_cell_key: u64,
        tiles: &ReferenceTileStore,
        cells: &CellStore,
    ) -> SimulationDelta {
        let changed_tiles = std::mem::take(&mut self.tiles)
            .into_iter()
            .filter_map(|(tile, before)| {
                let after = tiles[tile.0].clone();
                (before != after).then_some(TileDelta {
                    tile,
                    before,
                    after,
                })
            })
            .collect();
        let before_cells = self.take_cells_ordered();
        let mut current_cells = cells.iter().peekable();
        let changed_cells = before_cells
            .into_iter()
            .filter_map(|(cell, before)| {
                while current_cells
                    .peek()
                    .is_some_and(|(candidate, _)| **candidate < cell)
                {
                    current_cells.next();
                }
                let after = current_cells
                    .peek()
                    .filter(|(candidate, _)| **candidate == cell)
                    .map(|(_, state)| (*state).clone());
                (before != after).then_some(CellDelta {
                    cell,
                    before,
                    after,
                })
            })
            .collect();
        SimulationDelta {
            before_time: self.before_time,
            after_time,
            before_next_cell_key: self.before_next_cell_key,
            after_next_cell_key,
            tiles: changed_tiles,
            cells: changed_cells,
        }
    }

    fn rollback(&mut self, simulation: &mut ReferenceSimulation) {
        for (tile, before) in std::mem::take(&mut self.tiles) {
            simulation.tiles[tile.0] = before;
        }
        for (key, before) in self.take_cells_ordered() {
            match before {
                Some(cell) => {
                    simulation.cells.insert(key, cell);
                }
                None => {
                    simulation.cells.remove(&key);
                }
            }
        }
        simulation.now = self.before_time;
        simulation.next_cell_key = self.before_next_cell_key;
        simulation.due_actions.clear();
        for (key, cell) in &simulation.cells {
            if let Some(pending) = &cell.pending_action {
                simulation
                    .due_actions
                    .entry(pending.completes_at)
                    .or_default()
                    .push(*key);
            }
        }
        simulation.rebuild_cell_passive_indexes();
        simulation.last_resolution_metrics = self.previous_metrics;
        simulation.state_hash_cache = StateHashCache::new(IncrementalStateHash::new(
            simulation.compiled_ruleset_hash(),
            &simulation.tiles,
            &simulation.cells,
            simulation.next_cell_key,
        ));
        simulation.checkpoint_mutations = CheckpointMutationTracker::new();
    }
}

#[derive(Debug, Clone)]
struct PlannedAction {
    target: Option<TileIndex>,
    effort: u64,
    payload: u64,
    payload_source: PayloadSource,
    duration: u64,
}

struct PreparedDecision<'a> {
    decision: &'a DecisionCommitment,
    origin: TileIndex,
    planned: PlannedAction,
    rejection: Option<RejectReason>,
    completes_at: SimTime,
    signal_amounts: [u64; REFERENCE_SIGNAL_CHANNELS],
    signal_total: u64,
    private_memory_changed: bool,
}

#[derive(Debug)]
struct StateHashCache(Mutex<IncrementalStateHash>);

impl StateHashCache {
    fn new(cache: IncrementalStateHash) -> Self {
        Self(Mutex::new(cache))
    }

    fn get_mut(&mut self) -> &mut IncrementalStateHash {
        self.0
            .get_mut()
            .expect("state hash cache mutex was poisoned")
    }

    fn hash(
        &self,
        now: SimTime,
        tiles: &ReferenceTileStore,
        cells: &CellStore,
        next_cell_key: u64,
    ) -> CanonicalHash {
        self.0
            .lock()
            .expect("state hash cache mutex was poisoned")
            .hash(now, tiles, cells, next_cell_key)
    }

    fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.0
            .lock()
            .expect("state hash cache mutex was poisoned")
            .compiled_ruleset_hash()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn seed(&self) -> IncrementalStateHashSeed {
        self.0
            .lock()
            .expect("state hash cache mutex was poisoned")
            .seed()
            .expect("state hash cache must be clean after computing the checkpoint hash")
    }
}

impl Clone for StateHashCache {
    fn clone(&self) -> Self {
        let cache = self
            .0
            .lock()
            .expect("state hash cache mutex was poisoned")
            .clone();
        Self::new(cache)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PayloadSource {
    None,
    Assimilated,
    Gut,
    Material,
}

#[derive(Debug, Clone)]
struct WorkingIntent {
    actor: CellKey,
    pending: PendingAction,
    status: Option<OutcomeStatus>,
    target_occupant: Option<CellKey>,
    consumed_plant: u64,
    consumed_loose: u64,
    attack_mass: u128,
    attack_raw_damage: u64,
    target_was_guarded: Option<bool>,
    claims: [Option<ResourceClaim>; 2],
    occupancy_claim: Option<TileIndex>,
}

#[derive(Debug, Clone, Copy)]
struct DamageAllocation {
    mitigated: u64,
    applied: u64,
    overkill: u64,
}

#[derive(Debug)]
struct NontrivialDamageGroup {
    adjusted: u64,
    actual: u64,
    contributors: Vec<usize>,
}

impl WorkingIntent {
    fn push_claim(&mut self, claim: ResourceClaim) -> Result<(), ResolutionError> {
        let slot = self.claims.iter_mut().find(|slot| slot.is_none()).ok_or(
            ResolutionError::InvariantViolation("an intent exceeded the bounded claim footprint"),
        )?;
        *slot = Some(claim);
        Ok(())
    }
}

impl ReferenceSimulation {
    pub fn new(
        width: usize,
        height: usize,
        rules: ReferenceRuleset,
    ) -> Result<Self, ResolutionError> {
        rules.validate()?;
        let topology = ReferenceCompiledTopology::compile(width, height, &rules)?;
        Self::new_with_compiled_topology(rules, topology)
    }

    pub fn new_with_compiled_topology(
        rules: ReferenceRuleset,
        topology: ReferenceCompiledTopology,
    ) -> Result<Self, ResolutionError> {
        topology.validate_for_rules(&rules)?;
        let tile_count = topology.neighborhood.tile_count();
        let tiles = vec![TileState::default(); tile_count];
        let cells = CellStore::new();
        let state_hash_cache = StateHashCache::new(IncrementalStateHash::new(
            topology.compiled_ruleset_hash,
            &tiles,
            &cells,
            0,
        ));
        Ok(Self {
            rules,
            neighborhood: topology.neighborhood,
            now: SimTime(0),
            tiles: ReferenceTileStore::from_vec(tiles),
            growing_plant_tiles: Vec::new(),
            growing_plant_index_dirty: false,
            active_signal_tiles: BTreeSet::new(),
            active_diffuse_tiles: BTreeSet::new(),
            field_frontiers_dirty: false,
            active_digestion_cells: BTreeSet::new(),
            active_metabolism_cells: BTreeSet::new(),
            zero_energy_cells: BTreeSet::new(),
            metabolic_exhaustion: MetabolicExhaustionIndex::default(),
            diffusion_neighbors: topology.diffusion_neighbors,
            #[cfg(not(target_arch = "wasm32"))]
            diffusion_sources: topology.diffusion_sources,
            diffusion_incoming: vec![0; tile_count],
            diffusion_touched: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            dense_metabolism_deposits: DenseMetabolismDeposits::default(),
            #[cfg(not(target_arch = "wasm32"))]
            passive_cell_parallel_threshold: Some(PARALLEL_PASSIVE_CELL_THRESHOLD),
            #[cfg(not(target_arch = "wasm32"))]
            passive_tile_parallel_threshold: Some(PARALLEL_PASSIVE_TILE_THRESHOLD),
            cells,
            due_actions: BTreeMap::new(),
            next_cell_key: 0,
            state_hash_cache,
            checkpoint_mutations: CheckpointMutationTracker::new(),
            last_resolution_metrics: ResolutionMetrics::default(),
        })
    }

    /// Restores an authoritative simulation from a canonical checkpoint.
    /// The complete occupancy/cell bijection is validated before the state is
    /// accepted, so host projections cannot introduce a partial world.
    pub fn from_canonical_state(
        width: usize,
        height: usize,
        rules: ReferenceRuleset,
        state: SimulationState,
    ) -> Result<Self, ResolutionError> {
        rules.validate()?;
        let topology = ReferenceCompiledTopology::compile(width, height, &rules)?;
        Self::from_canonical_state_with_compiled_topology(rules, state, topology)
    }

    pub fn from_canonical_state_with_compiled_topology(
        rules: ReferenceRuleset,
        state: SimulationState,
        topology: ReferenceCompiledTopology,
    ) -> Result<Self, ResolutionError> {
        Self::from_canonical_state_with_compiled_topology_inner(
            rules,
            state.into(),
            topology,
            None,
            None,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn from_canonical_tile_store_with_compiled_topology_and_hash_seed_profiled(
        rules: ReferenceRuleset,
        now: SimTime,
        tiles: ReferenceTileStore,
        cells: Vec<(CellKey, CellState)>,
        next_cell_key: u64,
        topology: ReferenceCompiledTopology,
        hash_seed: Option<IncrementalStateHashSeed>,
    ) -> Result<(Self, ReferenceReconstructionProfile), ResolutionError> {
        let mut profile = ReferenceReconstructionProfile::default();
        let simulation = Self::from_canonical_state_with_compiled_topology_inner(
            rules,
            ReferenceRestoreState {
                now,
                tiles,
                cells,
                next_cell_key,
            },
            topology,
            hash_seed,
            Some(&mut profile),
        )?;
        Ok((simulation, profile))
    }

    fn from_canonical_state_with_compiled_topology_inner(
        rules: ReferenceRuleset,
        state: ReferenceRestoreState,
        topology: ReferenceCompiledTopology,
        hash_seed: Option<IncrementalStateHashSeed>,
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))] mut profile: Option<
            &mut ReferenceReconstructionProfile,
        >,
    ) -> Result<Self, ResolutionError> {
        let phase = profile.is_some().then(PhaseTimer::start);
        topology.validate_for_rules(&rules)?;
        if state.tiles.len() != topology.neighborhood.tile_count() {
            return Err(ResolutionError::InvalidState(
                "tile count does not match compiled dimensions",
            ));
        }
        if let (Some(profile), Some(phase)) = (profile.as_deref_mut(), phase) {
            profile.topology_validation_ns = phase.elapsed_ns();
        }

        let phase = profile.is_some().then(PhaseTimer::start);
        let mut cells = CellStore::new();
        let mut active_digestion_keys = Vec::new();
        let mut active_metabolism_keys = Vec::new();
        let mut zero_energy_keys = Vec::new();
        let mut due_actions: BTreeMap<SimTime, Vec<CellKey>> = BTreeMap::new();
        let mut previous_key = None;
        for (key, cell) in state.cells {
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(ResolutionError::InvalidState(
                    "cell keys are not strictly ordered",
                ));
            }
            previous_key = Some(key);
            if cell.position.0 >= state.tiles.len() {
                return Err(ResolutionError::InvalidState(
                    "cell position is outside the world",
                ));
            }
            if cell.private_memory.len() > rules.max_private_memory_bytes {
                return Err(ResolutionError::InvalidState(
                    "cell private memory exceeds the ruleset limit",
                ));
            }
            if cell.metabolism_remainder >= rules.metabolism_rate_denominator {
                return Err(ResolutionError::InvalidState(
                    "cell metabolism remainder exceeds its denominator",
                ));
            }
            if rules.metabolism_rate_numerator == 0 && cell.metabolism_remainder != 0 {
                return Err(ResolutionError::InvalidState(
                    "cell has a metabolism remainder while metabolism is disabled",
                ));
            }
            if cell.digestion_remainder >= rules.digestion_rate_denominator {
                return Err(ResolutionError::InvalidState(
                    "cell digestion remainder exceeds its denominator",
                ));
            }
            if rules.digestion_rate_numerator == 0 && cell.digestion_remainder != 0 {
                return Err(ResolutionError::InvalidState(
                    "cell has a digestion remainder while digestion is disabled",
                ));
            }
            if cell.gut_energy == 0 && cell.digestion_remainder != 0 {
                return Err(ResolutionError::InvalidState(
                    "cell without gut energy has a digestion remainder",
                ));
            }
            if cell.assimilated_energy == 0 && cell.metabolism_remainder != 0 {
                return Err(ResolutionError::InvalidState(
                    "cell without assimilated energy has a metabolism remainder",
                ));
            }
            if state.tiles[cell.position.0].occupant != Some(key) {
                return Err(ResolutionError::InvalidState(
                    "cell position and tile occupant disagree",
                ));
            }
            if let Some(pending) = &cell.pending_action {
                if pending.started_at > state.now
                    || pending.completes_at <= pending.started_at
                    || pending.completes_at < state.now
                    || pending.completes_at != cell.ready_at
                {
                    return Err(ResolutionError::InvalidState(
                        "pending action timestamps are inconsistent",
                    ));
                }
                if let ActionRequest::Split { private_memory, .. } = &pending.request {
                    if private_memory.len() > rules.max_private_memory_bytes {
                        return Err(ResolutionError::InvalidState(
                            "pending child private memory exceeds the ruleset limit",
                        ));
                    }
                }
            } else if cell.ready_at > state.now {
                return Err(ResolutionError::InvalidState(
                    "a cell without a pending action cannot become ready after checkpoint time",
                ));
            }
            if cell.gut_energy > 0 {
                active_digestion_keys.push(key);
            }
            if cell.assimilated_energy > 0 {
                active_metabolism_keys.push(key);
            } else {
                zero_energy_keys.push(key);
            }
            if let Some(pending) = &cell.pending_action {
                due_actions
                    .entry(pending.completes_at)
                    .or_default()
                    .push(key);
            }
            cells.insert(key, cell);
        }

        if previous_key.is_some_and(|key| state.next_cell_key <= key.0) {
            return Err(ResolutionError::InvalidState(
                "next cell key does not exceed existing keys",
            ));
        }
        let active_digestion_cells = active_digestion_keys.into_iter().collect();
        let active_metabolism_cells = active_metabolism_keys.into_iter().collect();
        let zero_energy_cells = zero_energy_keys.into_iter().collect();
        if let (Some(profile), Some(phase)) = (profile.as_deref_mut(), phase) {
            profile.cell_validation_store_and_passive_index_ns = phase.elapsed_ns();
        }

        let phase = profile.is_some().then(PhaseTimer::start);
        let tile_indexes = validate_canonical_tiles(&state.tiles, &rules, &cells)?;
        let growing_plant_tiles = tile_indexes.growing_plants;
        let active_signal_tiles = tile_indexes.active_signals.into_iter().collect();
        let active_diffuse_tiles = tile_indexes.active_diffuse.into_iter().collect();
        if let (Some(profile), Some(phase)) = (profile.as_deref_mut(), phase) {
            profile.tile_validation_and_passive_index_ns = phase.elapsed_ns();
        }

        let tile_count = state.tiles.len();

        let phase = profile.is_some().then(PhaseTimer::start);
        let incremental_hash = if let Some(seed) = hash_seed {
            IncrementalStateHash::from_seed(
                seed,
                topology.compiled_ruleset_hash,
                state.tiles.len(),
                &cells,
                state.next_cell_key,
            )
            .ok_or(ResolutionError::InvalidState(
                "incremental hash seed does not match canonical state dimensions or rules",
            ))?
        } else {
            IncrementalStateHash::new(
                topology.compiled_ruleset_hash,
                &state.tiles,
                &cells,
                state.next_cell_key,
            )
        };
        let state_hash_cache = StateHashCache::new(incremental_hash);
        if let (Some(profile), Some(phase)) = (profile.as_deref_mut(), phase) {
            profile.hash_initialization_ns = phase.elapsed_ns();
        }

        let phase = profile.is_some().then(PhaseTimer::start);
        let diffusion_incoming = vec![0; tile_count];
        let diffusion_touched = Vec::new();
        if let (Some(profile), Some(phase)) = (profile.as_deref_mut(), phase) {
            profile.scratch_initialization_ns = phase.elapsed_ns();
        }
        let mut simulation = Self {
            rules,
            neighborhood: topology.neighborhood,
            now: state.now,
            tiles: state.tiles,
            growing_plant_tiles,
            growing_plant_index_dirty: false,
            active_signal_tiles,
            active_diffuse_tiles,
            field_frontiers_dirty: false,
            active_digestion_cells,
            active_metabolism_cells,
            zero_energy_cells,
            metabolic_exhaustion: MetabolicExhaustionIndex::default(),
            diffusion_neighbors: topology.diffusion_neighbors,
            #[cfg(not(target_arch = "wasm32"))]
            diffusion_sources: topology.diffusion_sources,
            diffusion_incoming,
            diffusion_touched,
            #[cfg(not(target_arch = "wasm32"))]
            dense_metabolism_deposits: DenseMetabolismDeposits::default(),
            #[cfg(not(target_arch = "wasm32"))]
            passive_cell_parallel_threshold: Some(PARALLEL_PASSIVE_CELL_THRESHOLD),
            #[cfg(not(target_arch = "wasm32"))]
            passive_tile_parallel_threshold: Some(PARALLEL_PASSIVE_TILE_THRESHOLD),
            cells,
            due_actions,
            next_cell_key: state.next_cell_key,
            state_hash_cache,
            checkpoint_mutations: CheckpointMutationTracker::new(),
            last_resolution_metrics: ResolutionMetrics::default(),
        };
        let phase = profile.is_some().then(PhaseTimer::start);
        simulation.rebuild_metabolic_exhaustion_index();
        if let (Some(profile), Some(phase)) = (profile, phase) {
            profile.metabolic_index_ns = phase.elapsed_ns();
        }
        Ok(simulation)
    }

    pub fn compiled_topology(&self) -> ReferenceCompiledTopology {
        ReferenceCompiledTopology {
            neighborhood: Arc::clone(&self.neighborhood),
            diffusion_neighbors: Arc::clone(&self.diffusion_neighbors),
            #[cfg(not(target_arch = "wasm32"))]
            diffusion_sources: Arc::clone(&self.diffusion_sources),
            compiled_ruleset_hash: self.compiled_ruleset_hash(),
        }
    }

    pub fn rules(&self) -> &ReferenceRuleset {
        &self.rules
    }

    /// Selects host-only dense passive-kernel crossovers. `None` forces the
    /// serial semantic oracle for that resource family.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_passive_parallel_thresholds(
        &mut self,
        cell_threshold: Option<usize>,
        tile_threshold: Option<usize>,
    ) {
        self.passive_cell_parallel_threshold = cell_threshold.map(|value| value.max(2));
        self.passive_tile_parallel_threshold = tile_threshold.map(|value| value.max(2));
    }

    pub fn semantic_ruleset_hash(&self) -> CanonicalHash {
        self.rules.semantic_hash()
    }

    pub fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.state_hash_cache.compiled_ruleset_hash()
    }

    pub fn state_hash(&self) -> CanonicalHash {
        self.state_hash_cache
            .hash(self.now, &self.tiles, &self.cells, self.next_cell_key)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn incremental_hash_seed(&self) -> IncrementalStateHashSeed {
        self.state_hash_cache.seed()
    }

    pub fn neighborhood(&self) -> &CompiledNeighborhood {
        &self.neighborhood
    }

    pub const fn now(&self) -> SimTime {
        self.now
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) const fn next_cell_key(&self) -> u64 {
        self.next_cell_key
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn checkpoint_mutation_summary(
        &self,
        parent_token: Option<u64>,
    ) -> CheckpointMutationSummary {
        self.checkpoint_mutations.summary(parent_token)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn adopt_checkpoint_mutation_token(&mut self, token: u64) {
        self.checkpoint_mutations.adopt_checkpoint_token(token);
    }

    pub fn tile(&self, x: usize, y: usize) -> Option<TileIndex> {
        self.neighborhood.tile(x, y)
    }

    pub fn tile_state(&self, tile: TileIndex) -> Option<&TileState> {
        self.tiles.get(tile.0)
    }

    pub fn tile_state_mut(&mut self, tile: TileIndex) -> Option<&mut TileState> {
        self.state_hash_cache.get_mut().mark_tile(tile);
        self.checkpoint_mutations.mark_tile(tile);
        let state = self.tiles.get_mut(tile.0)?;
        // Callers can change a tile's growth rate through this mutable view.
        self.growing_plant_index_dirty = true;
        self.field_frontiers_dirty = true;
        Some(state)
    }

    pub fn cell(&self, key: CellKey) -> Option<&CellState> {
        self.cells.get(&key)
    }

    pub fn cells(&self) -> &CellStore {
        &self.cells
    }

    /// Canonical tile storage for trusted host-side diagnostics. This is never
    /// projected into a Mind input; callers receive an immutable view.
    pub fn tiles(&self) -> &ReferenceTileStore {
        &self.tiles
    }

    pub const fn last_resolution_metrics(&self) -> ResolutionMetrics {
        self.last_resolution_metrics
    }

    /// Derived host-side index for passive-update scheduling and sparse UI
    /// invalidation. It is not part of canonical state or Mind input.
    pub fn growing_plant_tiles(&self) -> &[TileIndex] {
        &self.growing_plant_tiles
    }

    fn rebuild_field_frontiers(&mut self) {
        if !self.field_frontiers_dirty {
            return;
        }
        self.active_signal_tiles.clear();
        self.active_diffuse_tiles.clear();
        for (index, tile) in self.tiles.iter().enumerate() {
            if tile.signal_energy.iter().any(|energy| *energy > 0) {
                self.active_signal_tiles.insert(TileIndex(index));
            }
            if tile.diffuse_energy > 0 {
                self.active_diffuse_tiles.insert(TileIndex(index));
            }
        }
        self.field_frontiers_dirty = false;
    }

    fn mark_signal_tile_active(&mut self, tile: TileIndex) {
        self.active_signal_tiles.insert(tile);
    }

    /// Terrain manipulation destroys the local signal pattern but conserves
    /// its mass-energy by moving every cleared channel into diffuse energy.
    fn clear_tile_signals_into_diffuse(&mut self, tile: TileIndex) -> Result<u64, ResolutionError> {
        let state = self
            .tiles
            .get_mut(tile.0)
            .ok_or(ResolutionError::InvalidTile(tile))?;
        let cleared = state
            .signal_energy
            .iter()
            .try_fold(0_u64, |total, amount| {
                total
                    .checked_add(*amount)
                    .ok_or(ResolutionError::ArithmeticOverflow(
                        "summing terrain-cleared signal energy",
                    ))
            })?;
        state.diffuse_energy = state.diffuse_energy.checked_add(cleared).ok_or(
            ResolutionError::ArithmeticOverflow("depositing terrain-cleared signal energy"),
        )?;
        state.signal_energy = [0; REFERENCE_SIGNAL_CHANNELS];
        state.signal_decay_remainder = [0; REFERENCE_SIGNAL_CHANNELS];
        self.active_signal_tiles.remove(&tile);
        if cleared > 0 {
            self.mark_diffuse_tile_active(tile);
        }
        Ok(cleared)
    }

    fn mark_diffuse_tile_active(&mut self, tile: TileIndex) {
        self.active_diffuse_tiles.insert(tile);
    }

    fn signal_frontier(&self) -> Vec<TileIndex> {
        if self.field_frontiers_dirty {
            self.tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    tile.signal_energy
                        .iter()
                        .any(|energy| *energy > 0)
                        .then_some(TileIndex(index))
                })
                .collect()
        } else {
            self.active_signal_tiles.iter().copied().collect()
        }
    }

    fn diffuse_frontier(&self) -> Vec<TileIndex> {
        if self.field_frontiers_dirty {
            self.tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| (tile.diffuse_energy > 0).then_some(TileIndex(index)))
                .collect()
        } else {
            self.active_diffuse_tiles.iter().copied().collect()
        }
    }

    fn sync_cell_passive_indexes(&mut self, key: CellKey) {
        let Some(cell) = self.cells.get(&key) else {
            self.active_digestion_cells.remove(&key);
            self.active_metabolism_cells.remove(&key);
            self.zero_energy_cells.remove(&key);
            self.metabolic_exhaustion.sync(
                key,
                None,
                self.now,
                self.rules.metabolism_rate_numerator,
                self.rules.metabolism_rate_denominator,
            );
            return;
        };
        if cell.gut_energy > 0 {
            self.active_digestion_cells.insert(key);
        } else {
            self.active_digestion_cells.remove(&key);
        }
        if cell.assimilated_energy > 0 {
            self.active_metabolism_cells.insert(key);
            self.zero_energy_cells.remove(&key);
        } else {
            self.active_metabolism_cells.remove(&key);
            self.zero_energy_cells.insert(key);
        }
        self.metabolic_exhaustion.sync(
            key,
            Some(cell),
            self.now,
            self.rules.metabolism_rate_numerator,
            self.rules.metabolism_rate_denominator,
        );
    }

    fn rebuild_metabolic_exhaustion_index(&mut self) {
        self.metabolic_exhaustion.rebuild(
            &self.cells,
            self.now,
            self.rules.metabolism_rate_numerator,
            self.rules.metabolism_rate_denominator,
        );
    }

    fn rebuild_cell_passive_indexes(&mut self) {
        self.active_digestion_cells.clear();
        self.active_metabolism_cells.clear();
        self.zero_energy_cells.clear();
        for (key, cell) in &self.cells {
            if cell.gut_energy > 0 {
                self.active_digestion_cells.insert(*key);
            }
            if cell.assimilated_energy > 0 {
                self.active_metabolism_cells.insert(*key);
            } else {
                self.zero_energy_cells.insert(*key);
            }
        }
        self.rebuild_metabolic_exhaustion_index();
    }

    /// Conservative sparse invalidation set for signal decay and one
    /// synchronous diffuse step at `requested`.
    pub fn field_changed_tiles_at(&self, requested: SimTime) -> Vec<TileIndex> {
        let diffusion_due = self.rules.diffusion_rate_numerator > 0
            && requested
                .0
                .is_multiple_of(self.rules.diffusion_interval_quanta);
        let mut changed = std::collections::BTreeSet::new();
        let signal_tiles = self.signal_frontier();
        changed.extend(signal_tiles.iter().copied());
        if diffusion_due {
            let mut sources = self.diffuse_frontier();
            sources.extend(signal_tiles);
            if self.rules.metabolism_rate_numerator > 0 {
                if self.active_metabolism_cells.len() == self.cells.len() {
                    sources.extend(self.cells.values().map(|cell| cell.position));
                } else {
                    sources.extend(
                        self.active_metabolism_cells
                            .iter()
                            .filter_map(|key| self.cells.get(key).map(|cell| cell.position)),
                    );
                    sources.extend(
                        self.active_digestion_cells
                            .iter()
                            .filter_map(|key| self.cells.get(key).map(|cell| cell.position)),
                    );
                }
            }
            sources.sort_unstable();
            sources.dedup();
            for source in sources {
                changed.insert(source);
                changed.extend(self.diffusion_neighbors[source.0].iter().copied());
            }
        } else if self.rules.diffusion_rate_numerator == 0 {
            changed.extend(self.diffuse_frontier());
        }
        changed.into_iter().collect()
    }

    /// Conservative resources whose host-visible projection may change while
    /// advancing passive physics to `requested`. The batch delta begins after
    /// this advance, so projection consumers must union these keys with the
    /// canonical action delta. False positives are allowed; false negatives
    /// are not.
    pub fn passive_projection_changes_at(
        &self,
        requested: SimTime,
    ) -> (Vec<TileIndex>, Vec<CellKey>) {
        let all_cells_change = (self.rules.metabolism_rate_numerator > 0
            && self.active_metabolism_cells.len() == self.cells.len())
            || (self.rules.digestion_rate_numerator > 0
                && self.active_digestion_cells.len() == self.cells.len());
        let cells = if all_cells_change {
            self.cells.keys().copied().collect()
        } else {
            let mut changed = BTreeSet::new();
            if self.rules.digestion_rate_numerator > 0 {
                changed.extend(self.active_digestion_cells.iter().copied());
            }
            if self.rules.metabolism_rate_numerator > 0 {
                changed.extend(self.active_metabolism_cells.iter().copied());
            }
            changed.into_iter().collect()
        };

        let mut tiles = std::collections::BTreeSet::new();
        if self.rules.metabolism_rate_numerator > 0 {
            if self.active_metabolism_cells.len() == self.cells.len() {
                tiles.extend(self.cells.values().map(|cell| cell.position));
            } else {
                tiles.extend(
                    self.active_metabolism_cells
                        .iter()
                        .filter_map(|key| self.cells.get(key).map(|cell| cell.position)),
                );
            }
            tiles.extend(
                self.active_digestion_cells
                    .iter()
                    .filter_map(|key| self.cells.get(key).map(|cell| cell.position)),
            );
        }
        if self.growing_plant_index_dirty {
            tiles.extend((0..self.tiles.len()).map(TileIndex));
        } else {
            tiles.extend(self.growing_plant_tiles.iter().copied());
        }
        tiles.extend(self.field_changed_tiles_at(requested));
        (tiles.into_iter().collect(), cells)
    }

    pub fn add_cell(
        &mut self,
        tile: TileIndex,
        core_mass: u64,
        assimilated_energy: u64,
        marker: u32,
    ) -> Result<CellKey, ResolutionError> {
        let tile_state = self
            .tiles
            .get_mut(tile.0)
            .ok_or(ResolutionError::InvalidTile(tile))?;
        if tile_state.occupant.is_some() {
            return Err(ResolutionError::OccupiedTile(tile));
        }
        let key = CellKey(self.next_cell_key);
        self.next_cell_key = self
            .next_cell_key
            .checked_add(1)
            .ok_or(ResolutionError::ArithmeticOverflow("allocating a cell key"))?;
        tile_state.occupant = Some(key);
        self.cells.insert(
            key,
            CellState {
                position: tile,
                core_mass,
                assimilated_energy,
                gut_energy: 0,
                digestion_remainder: 0,
                metabolism_remainder: 0,
                carried_material_mass: 0,
                marker,
                guarded: false,
                ready_at: self.now,
                pending_action: None,
                last_outcome: None,
                cold: Arc::new(CellColdState {
                    private_memory: Arc::from([]),
                }),
            },
        );
        self.sync_cell_passive_indexes(key);
        let cache = self.state_hash_cache.get_mut();
        cache.mark_tile(tile);
        cache.mark_cell(key);
        self.checkpoint_mutations.mark_tile(tile);
        self.checkpoint_mutations.mark_cell_structure(key);
        Ok(key)
    }

    pub fn canonical_state(&self) -> SimulationState {
        SimulationState {
            now: self.now,
            tiles: self.tiles.to_vec(),
            cells: self
                .cells
                .iter()
                .map(|(key, cell)| (*key, cell.clone()))
                .collect(),
            next_cell_key: self.next_cell_key,
        }
    }

    pub fn total_energy_equivalent(&self) -> u128 {
        let tile_energy = self.tiles.iter().fold(0_u128, |total, tile| {
            let terrain_units = i32::from(tile.elevation) - i32::from(i16::MIN);
            total
                + u128::from(tile.plant_energy)
                + u128::from(tile.loose_energy)
                + u128::from(tile.diffuse_energy)
                + tile.signal_energy.into_iter().map(u128::from).sum::<u128>()
                + u128::from(terrain_units as u32)
                    * u128::from(self.rules.terrain_mass_per_elevation)
        });
        self.cells.values().fold(tile_energy, |total, cell| {
            total
                + u128::from(cell.core_mass)
                + u128::from(cell.assimilated_energy)
                + u128::from(cell.gut_energy)
                + u128::from(cell.carried_material_mass)
                + u128::from(
                    cell.pending_action
                        .as_ref()
                        .map_or(0, |action| action.payload_escrow),
                )
        })
    }

    pub fn commit_action(
        &mut self,
        actor: CellKey,
        request: ActionRequest,
    ) -> Result<CommitReceipt, CommitError> {
        if matches!(request, ActionRequest::Signal { .. }) {
            return self.commit_memory_update_with_signal(
                actor,
                request,
                None,
                ReferenceMemoryUpdate::Retain,
            );
        }
        let cell = self
            .cells
            .get(&actor)
            .ok_or(CommitError::UnknownCell(actor))?;
        if !cell.is_ready_at(self.now) {
            return Err(CommitError::CellNotReady {
                actor,
                ready_at: cell.ready_at,
            });
        }
        let origin = cell.position;
        let plan = self.plan_action(cell, &request);
        let (planned, rejection) = match plan {
            Ok(plan) => (plan, None),
            Err(reason) => (
                PlannedAction {
                    target: None,
                    effort: 0,
                    payload: 0,
                    payload_source: PayloadSource::None,
                    duration: self.rules.time.decision_interval_floor,
                },
                Some(reason),
            ),
        };
        let completes_at = self
            .now
            .0
            .checked_add(planned.duration)
            .map(SimTime)
            .ok_or(CommitError::InvariantViolation(
                "completion timestamp overflow",
            ))?;
        let passive_energy_changed = rejection.is_none()
            && (planned.effort > 0
                || matches!(
                    planned.payload_source,
                    PayloadSource::Assimilated | PayloadSource::Gut
                ));

        let cache = self.state_hash_cache.get_mut();
        cache.mark_cell(actor);
        if rejection.is_none() && planned.effort > 0 {
            cache.mark_tile(origin);
        }
        self.checkpoint_mutations.mark_cell(actor);
        if rejection.is_none() && planned.effort > 0 {
            self.checkpoint_mutations.mark_tile(origin);
        }

        if rejection.is_none() && planned.effort > 0 {
            let diffuse = self.tiles[origin.0]
                .diffuse_energy
                .checked_add(planned.effort)
                .ok_or(CommitError::InvariantViolation("diffuse energy overflow"))?;
            self.tiles[origin.0].diffuse_energy = diffuse;
            self.mark_diffuse_tile_active(origin);
        }

        let cell = self
            .cells
            .get_mut(&actor)
            .ok_or(CommitError::InvariantViolation(
                "cell disappeared during action planning",
            ))?;
        if rejection.is_none() {
            cell.assimilated_energy = cell.assimilated_energy.checked_sub(planned.effort).ok_or(
                CommitError::InvariantViolation("accepted action effort exceeds cell energy"),
            )?;
            match planned.payload_source {
                PayloadSource::None => {}
                PayloadSource::Assimilated => {
                    cell.assimilated_energy = cell
                        .assimilated_energy
                        .checked_sub(planned.payload)
                        .ok_or(CommitError::InvariantViolation(
                            "accepted action payload exceeds cell energy",
                        ))?;
                }
                PayloadSource::Gut => {
                    cell.gut_energy = cell.gut_energy.checked_sub(planned.payload).ok_or(
                        CommitError::InvariantViolation(
                            "accepted action payload exceeds gut energy",
                        ),
                    )?;
                    if cell.gut_energy == 0 {
                        cell.digestion_remainder = 0;
                    }
                }
                PayloadSource::Material => {
                    cell.carried_material_mass = cell
                        .carried_material_mass
                        .checked_sub(planned.payload)
                        .ok_or(CommitError::InvariantViolation(
                            "accepted terrain payload exceeds carried material",
                        ))?;
                }
            }
            cell.guarded = matches!(request, ActionRequest::Guard { .. });
        } else {
            // Rejected decisions become a minimum-duration Wait.
            cell.guarded = false;
        }
        cell.ready_at = completes_at;
        cell.pending_action = Some(Arc::new(PendingAction {
            request: request.clone(),
            origin,
            target: planned.target,
            started_at: self.now,
            completes_at,
            effort_spent: planned.effort,
            payload_escrow: planned.payload,
            rejection,
        }));
        if passive_energy_changed {
            self.sync_cell_passive_indexes(actor);
        }
        self.due_actions
            .entry(completes_at)
            .or_default()
            .push(actor);

        Ok(CommitReceipt {
            actor,
            action: request.kind(),
            accepted: rejection.is_none(),
            rejection,
            completes_at,
            effort_spent: planned.effort,
            payload_escrow: planned.payload,
        })
    }

    /// Installs the complete output of one pristine Mind invocation. The
    /// action and next private state form one authoritative commitment: if the
    /// action is semantically rejected it becomes a minimum Wait, but the
    /// bounded private state still advances exactly once.
    pub fn commit_decision(
        &mut self,
        actor: CellKey,
        request: ActionRequest,
        next_private_memory: Vec<u8>,
    ) -> Result<CommitReceipt, CommitError> {
        self.commit_decision_with_signal(actor, request, None, next_private_memory)
    }

    pub fn commit_decision_with_signal(
        &mut self,
        actor: CellKey,
        request: ActionRequest,
        signal: Option<ReferenceSignalEmission>,
        next_private_memory: Vec<u8>,
    ) -> Result<CommitReceipt, CommitError> {
        self.commit_memory_update_with_signal(
            actor,
            request,
            signal,
            ReferenceMemoryUpdate::Replace(next_private_memory),
        )
    }

    /// Installs one action together with an explicit private-memory update.
    /// `Retain` leaves the actor's canonical allocation untouched.
    pub fn commit_memory_update_with_signal(
        &mut self,
        actor: CellKey,
        request: ActionRequest,
        signal: Option<ReferenceSignalEmission>,
        memory_update: ReferenceMemoryUpdate,
    ) -> Result<CommitReceipt, CommitError> {
        let decision = DecisionCommitment {
            actor,
            request,
            signal,
            memory_update,
        };
        self.commit_decisions_ordered(std::slice::from_ref(&decision))?
            .pop()
            .ok_or(CommitError::InvariantViolation(
                "single-decision batch produced no receipt",
            ))
    }

    /// Atomically preflights and commits a strictly actor-ordered decision
    /// frontier. Fatal input errors leave canonical and derived state
    /// untouched; semantic action rejection still installs the usual minimum
    /// Wait and advances bounded private memory.
    pub fn commit_decisions_ordered(
        &mut self,
        decisions: &[DecisionCommitment],
    ) -> Result<Vec<CommitReceipt>, CommitError> {
        for pair in decisions.windows(2) {
            if pair[0].actor >= pair[1].actor {
                return Err(CommitError::ActorsNotStrictlyOrdered {
                    previous: pair[0].actor,
                    actor: pair[1].actor,
                });
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let prepared_results = if decisions.len() >= PARALLEL_COMMIT_PREFLIGHT_THRESHOLD {
            decisions
                .par_iter()
                .map(|decision| self.prepare_decision_commitment(decision))
                .collect::<Vec<_>>()
        } else {
            decisions
                .iter()
                .map(|decision| self.prepare_decision_commitment(decision))
                .collect::<Vec<_>>()
        };
        #[cfg(target_arch = "wasm32")]
        let prepared_results = decisions
            .iter()
            .map(|decision| self.prepare_decision_commitment(decision))
            .collect::<Vec<_>>();
        let prepared = prepared_results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.apply_prepared_decisions(&prepared))
    }

    fn prepare_decision_commitment<'a>(
        &self,
        decision: &'a DecisionCommitment,
    ) -> Result<PreparedDecision<'a>, CommitError> {
        if let ReferenceMemoryUpdate::Replace(bytes) = &decision.memory_update {
            if bytes.len() > self.rules.max_private_memory_bytes {
                return Err(CommitError::PrivateMemoryTooLarge {
                    actual: bytes.len(),
                    limit: self.rules.max_private_memory_bytes,
                });
            }
        }
        let cell = self
            .cells
            .get(&decision.actor)
            .ok_or(CommitError::UnknownCell(decision.actor))?;
        if !cell.is_ready_at(self.now) {
            return Err(CommitError::CellNotReady {
                actor: decision.actor,
                ready_at: cell.ready_at,
            });
        }
        let (signal_amounts, signal_total) =
            self.validate_signal(decision.actor, &decision.request, decision.signal)?;
        let post_signal_cell = (signal_total > 0).then(|| {
            let mut cell = cell.clone();
            cell.assimilated_energy -= signal_total;
            cell
        });
        let planning_cell = post_signal_cell.as_ref().unwrap_or(cell);
        let (planned, rejection) = match self.plan_action(planning_cell, &decision.request) {
            Ok(plan) => (plan, None),
            Err(reason) => (
                PlannedAction {
                    target: None,
                    effort: 0,
                    payload: 0,
                    payload_source: PayloadSource::None,
                    duration: self.rules.time.decision_interval_floor,
                },
                Some(reason),
            ),
        };
        let completes_at = self
            .now
            .0
            .checked_add(planned.duration)
            .map(SimTime)
            .ok_or(CommitError::InvariantViolation(
                "completion timestamp overflow",
            ))?;
        if rejection.is_none() && planned.effort > 0 {
            self.tiles[cell.position.0]
                .diffuse_energy
                .checked_add(planned.effort)
                .ok_or(CommitError::InvariantViolation("diffuse energy overflow"))?;
        }
        Ok(PreparedDecision {
            decision,
            origin: cell.position,
            planned,
            rejection,
            completes_at,
            signal_amounts,
            signal_total,
            private_memory_changed: decision
                .memory_update
                .replacement()
                .is_some_and(|bytes| cell.private_memory.as_ref() != bytes),
        })
    }

    fn apply_prepared_decisions(
        &mut self,
        prepared: &[PreparedDecision<'_>],
    ) -> Vec<CommitReceipt> {
        let dense_cell_frontier =
            prepared.len() >= 512 && prepared.len().saturating_mul(4) >= self.cells.len();
        {
            let cache = self.state_hash_cache.get_mut();
            if dense_cell_frontier {
                cache.mark_all_cells();
            }
            for decision in prepared {
                if !dense_cell_frontier {
                    cache.mark_cell(decision.decision.actor);
                }
                if decision.private_memory_changed {
                    cache.mark_cell_memory(decision.decision.actor);
                }
                if decision.signal_total > 0
                    || (decision.rejection.is_none() && decision.planned.effort > 0)
                {
                    cache.mark_tile(decision.origin);
                }
            }
        }
        self.checkpoint_mutations
            .mark_cells(prepared.iter().map(|decision| decision.decision.actor));
        self.checkpoint_mutations.mark_tiles(
            prepared
                .iter()
                .filter(|decision| {
                    decision.signal_total > 0
                        || (decision.rejection.is_none() && decision.planned.effort > 0)
                })
                .map(|decision| decision.origin),
        );

        let mut scheduled: BTreeMap<SimTime, Vec<CellKey>> = BTreeMap::new();
        let mut receipts = Vec::with_capacity(prepared.len());
        for prepared in prepared {
            let actor = prepared.decision.actor;
            let signal_emitted = if prepared.signal_total > 0 {
                let cell = self
                    .cells
                    .get_mut(&actor)
                    .expect("preflighted signal actor must exist");
                cell.assimilated_energy -= prepared.signal_total;
                for (channel, amount) in prepared.signal_amounts.into_iter().enumerate() {
                    self.tiles[prepared.origin.0].signal_energy[channel] += amount;
                }
                self.mark_signal_tile_active(prepared.origin);
                true
            } else {
                false
            };

            let accepted = prepared.rejection.is_none();
            if accepted && prepared.planned.effort > 0 {
                self.tiles[prepared.origin.0].diffuse_energy += prepared.planned.effort;
                self.mark_diffuse_tile_active(prepared.origin);
            }
            let passive_energy_changed = signal_emitted
                || (accepted
                    && (prepared.planned.effort > 0
                        || matches!(
                            prepared.planned.payload_source,
                            PayloadSource::Assimilated | PayloadSource::Gut
                        )));
            let cell = self
                .cells
                .get_mut(&actor)
                .expect("preflighted decision actor must exist");
            if accepted {
                cell.assimilated_energy -= prepared.planned.effort;
                match prepared.planned.payload_source {
                    PayloadSource::None => {}
                    PayloadSource::Assimilated => {
                        cell.assimilated_energy -= prepared.planned.payload;
                    }
                    PayloadSource::Gut => {
                        cell.gut_energy -= prepared.planned.payload;
                        if cell.gut_energy == 0 {
                            cell.digestion_remainder = 0;
                        }
                    }
                    PayloadSource::Material => {
                        cell.carried_material_mass -= prepared.planned.payload;
                    }
                }
                cell.guarded = matches!(&prepared.decision.request, ActionRequest::Guard { .. });
            } else {
                cell.guarded = false;
            }
            cell.ready_at = prepared.completes_at;
            cell.pending_action = Some(Arc::new(PendingAction {
                request: prepared.decision.request.clone(),
                origin: prepared.origin,
                target: prepared.planned.target,
                started_at: self.now,
                completes_at: prepared.completes_at,
                effort_spent: prepared.planned.effort,
                payload_escrow: prepared.planned.payload,
                rejection: prepared.rejection,
            }));
            if prepared.private_memory_changed {
                let ReferenceMemoryUpdate::Replace(bytes) = &prepared.decision.memory_update else {
                    unreachable!("retained private memory cannot be marked changed");
                };
                cell.private_memory = bytes.clone().into();
            }
            if passive_energy_changed {
                self.sync_cell_passive_indexes(actor);
            }
            scheduled
                .entry(prepared.completes_at)
                .or_default()
                .push(actor);
            receipts.push(CommitReceipt {
                actor,
                action: prepared.decision.request.kind(),
                accepted,
                rejection: prepared.rejection,
                completes_at: prepared.completes_at,
                effort_spent: prepared.planned.effort,
                payload_escrow: prepared.planned.payload,
            });
        }
        for (time, actors) in scheduled {
            self.due_actions.entry(time).or_default().extend(actors);
        }
        receipts
    }

    fn validate_signal(
        &self,
        actor: CellKey,
        request: &ActionRequest,
        sidecar: Option<ReferenceSignalEmission>,
    ) -> Result<([u64; REFERENCE_SIGNAL_CHANNELS], u64), CommitError> {
        let cell = self
            .cells
            .get(&actor)
            .ok_or(CommitError::UnknownCell(actor))?;
        if !cell.is_ready_at(self.now) {
            return Err(CommitError::CellNotReady {
                actor,
                ready_at: cell.ready_at,
            });
        }
        let mut amounts = [0_u64; REFERENCE_SIGNAL_CHANNELS];
        match (request, sidecar) {
            (ActionRequest::Signal { .. }, Some(_)) => {
                return Err(CommitError::InvalidSignal(
                    "explicit signal action cannot include a sidecar emission",
                ));
            }
            (ActionRequest::Signal { amounts: requested }, None) => amounts = *requested,
            (_, Some(signal)) => {
                if usize::from(signal.channel) >= REFERENCE_SIGNAL_CHANNELS {
                    return Err(CommitError::InvalidSignal(
                        "channel is outside the canonical range",
                    ));
                }
                amounts[usize::from(signal.channel)] = signal.amount;
            }
            (_, None) => return Ok((amounts, 0)),
        }
        let quantum = self.rules.signal_emission_cost;
        if quantum == 0 {
            return Err(CommitError::InvalidSignal("signaling is disabled"));
        }
        if amounts.iter().all(|amount| *amount == 0) {
            return Err(CommitError::InvalidSignal(
                "signal deposit must contain positive energy",
            ));
        }
        if amounts
            .iter()
            .any(|amount| *amount > 0 && *amount % quantum != 0)
        {
            return Err(CommitError::InvalidSignal(
                "signal amount is not a multiple of the emission quantum",
            ));
        }
        let total = amounts.iter().try_fold(0_u64, |total, amount| {
            total
                .checked_add(*amount)
                .ok_or(CommitError::InvalidSignal("signal total overflows"))
        })?;
        let required = total
            .checked_add(self.rules.minimum_survival_energy)
            .ok_or(CommitError::InvalidSignal("signal reserve overflows"))?;
        if cell.assimilated_energy < required {
            return Err(CommitError::InvalidSignal("insufficient energy"));
        }
        for (channel, amount) in amounts.into_iter().enumerate() {
            self.tiles[cell.position.0].signal_energy[channel]
                .checked_add(amount)
                .ok_or(CommitError::InvalidSignal(
                    "target signal field would overflow",
                ))?;
        }
        Ok((amounts, total))
    }

    fn plan_action(
        &self,
        cell: &CellState,
        request: &ActionRequest,
    ) -> Result<PlannedAction, RejectReason> {
        let target = if let (Some(targeting_action), Some(slot)) =
            (request.targeting_action(), request.target_slot())
        {
            if self.neighborhood.offset(slot).is_none() {
                return Err(RejectReason::InvalidSlot);
            }
            if !self.neighborhood.action_allows(targeting_action, slot) {
                return Err(RejectReason::ActionNotAllowedInSlot);
            }
            let target = self
                .neighborhood
                .target(cell.position, slot)
                .ok_or(RejectReason::TargetOutsideWorld)?;
            if target == cell.position {
                return Err(RejectReason::TargetsSelf);
            }
            Some(target)
        } else {
            None
        };

        match request {
            ActionRequest::Attack { payload: 0, .. }
            | ActionRequest::Consume { amount: 0 }
            | ActionRequest::Regurgitate { amount: 0, .. } => {
                return Err(RejectReason::ZeroPayload);
            }
            ActionRequest::Split {
                child_allocation,
                private_memory,
                ..
            } => {
                let minimum_child = self
                    .rules
                    .child_core_mass
                    .checked_add(self.rules.minimum_survival_energy)
                    .ok_or(RejectReason::ArithmeticOverflow)?;
                if *child_allocation < minimum_child {
                    return Err(RejectReason::ChildAllocationTooSmall);
                }
                if private_memory.len() > self.rules.max_private_memory_bytes {
                    return Err(RejectReason::PrivateMemoryTooLarge);
                }
            }
            ActionRequest::Wait
            | ActionRequest::Move { .. }
            | ActionRequest::Attack { .. }
            | ActionRequest::Guard { .. }
            | ActionRequest::Consume { .. }
            | ActionRequest::Signal { .. }
            | ActionRequest::Regurgitate { .. } => {}
            ActionRequest::Excavate => {
                if self.tiles[cell.position.0].elevation == i16::MIN {
                    return Err(RejectReason::TerrainLimit);
                }
            }
            ActionRequest::DepositTerrain => {
                if self.tiles[cell.position.0].elevation == i16::MAX {
                    return Err(RejectReason::TerrainLimit);
                }
            }
        }

        let mass = cell.total_mass();
        let effort = self.action_effort(request, mass)?;
        let payload = if matches!(request, ActionRequest::DepositTerrain) {
            self.rules.terrain_mass_per_elevation
        } else {
            request.payload()
        };
        let payload_source = match request {
            ActionRequest::Attack { .. } | ActionRequest::Split { .. } => {
                PayloadSource::Assimilated
            }
            ActionRequest::Regurgitate { .. } => PayloadSource::Gut,
            ActionRequest::DepositTerrain => PayloadSource::Material,
            ActionRequest::Wait
            | ActionRequest::Move { .. }
            | ActionRequest::Guard { .. }
            | ActionRequest::Consume { .. } => PayloadSource::None,
            ActionRequest::Signal { .. } => PayloadSource::None,
            ActionRequest::Excavate => PayloadSource::None,
        };
        if payload_source == PayloadSource::Gut && cell.gut_energy < payload {
            return Err(RejectReason::InsufficientGutEnergy);
        }
        if payload_source == PayloadSource::Material && cell.carried_material_mass < payload {
            return Err(RejectReason::InsufficientMaterial);
        }
        let assimilated_payload = if payload_source == PayloadSource::Assimilated {
            payload
        } else {
            0
        };
        let required = effort
            .checked_add(assimilated_payload)
            .ok_or(RejectReason::ArithmeticOverflow)?
            .checked_add(self.rules.minimum_survival_energy)
            .ok_or(RejectReason::ArithmeticOverflow)?;
        if cell.assimilated_energy < required {
            return Err(RejectReason::InsufficientEnergy);
        }

        let distance_cost_q10 = request
            .target_slot()
            .and_then(|slot| self.neighborhood.offset(slot))
            .map_or(1024, |offset| u64::from(offset.distance_cost_q10));
        let duration = self.action_duration(request, mass, distance_cost_q10)?;
        Ok(PlannedAction {
            target,
            effort,
            payload,
            payload_source,
            duration,
        })
    }

    fn action_effort(&self, request: &ActionRequest, mass: u128) -> Result<u64, RejectReason> {
        let raw = match request {
            ActionRequest::Wait => 0,
            ActionRequest::Signal { .. } => 0,
            ActionRequest::Move { target, .. } => {
                let distance = self
                    .neighborhood
                    .offset(*target)
                    .ok_or(RejectReason::InvalidSlot)?
                    .distance_cost_q10;
                let denominator = self
                    .rules
                    .move_mass_units_per_effort
                    .checked_mul(1024)
                    .ok_or(RejectReason::ArithmeticOverflow)?;
                let inertial =
                    multiply_ratio_ceil_u128_to_u64(mass, u64::from(distance), denominator)?;
                self.rules
                    .move_effort_base
                    .checked_add(inertial)
                    .ok_or(RejectReason::ArithmeticOverflow)?
            }
            ActionRequest::Attack { .. } => self.rules.attack_effort_base,
            ActionRequest::Guard { .. } => self.rules.guard_effort_base,
            ActionRequest::Consume { .. } => self.rules.consume_effort_base,
            ActionRequest::Split { .. } => self.rules.split_effort_base,
            ActionRequest::Regurgitate { .. } => self.rules.regurgitate_effort_base,
            ActionRequest::Excavate => self.rules.excavate_effort_base,
            ActionRequest::DepositTerrain => self.rules.deposit_terrain_effort_base,
        };
        let profile = self.rules.effort_profiles[request.effort_tier().index()];
        multiply_ratio_ceil(
            raw,
            u64::from(profile.cost_numerator),
            u64::from(profile.cost_denominator),
        )
    }

    fn action_duration(
        &self,
        request: &ActionRequest,
        mass: u128,
        distance_cost_q10: u64,
    ) -> Result<u64, RejectReason> {
        if matches!(request, ActionRequest::Guard { .. }) {
            return Ok(self.rules.time.decision_interval_floor);
        }
        let duration_rule = match request {
            ActionRequest::Wait => self.rules.wait_duration,
            ActionRequest::Signal { .. } => self.rules.wait_duration,
            ActionRequest::Move { .. } => self.rules.move_duration,
            ActionRequest::Attack { .. } => self.rules.attack_duration,
            ActionRequest::Guard { .. } => unreachable!(),
            ActionRequest::Consume { .. } => self.rules.consume_duration,
            ActionRequest::Split { .. } => self.rules.split_duration,
            ActionRequest::Regurgitate { .. } => self.rules.regurgitate_duration,
            ActionRequest::Excavate => self.rules.excavate_duration,
            ActionRequest::DepositTerrain => self.rules.deposit_terrain_duration,
        };
        let mass_term = multiply_ratio_ceil_u128_to_u64(
            mass,
            duration_rule.mass_quanta_numerator,
            duration_rule.mass_units_denominator,
        )?;
        let mut duration = duration_rule
            .base_quanta
            .checked_add(mass_term)
            .ok_or(RejectReason::ArithmeticOverflow)?;
        if matches!(request, ActionRequest::Move { .. }) {
            duration = multiply_ratio_ceil(duration, distance_cost_q10, 1024)?;
        }
        let profile = self.rules.effort_profiles[request.effort_tier().index()];
        duration = multiply_ratio_ceil(
            duration,
            u64::from(profile.duration_numerator),
            u64::from(profile.duration_denominator),
        )?;
        duration = duration.max(self.rules.time.decision_interval_floor);
        round_up(duration, self.rules.time.completion_bucket)
    }

    pub fn next_completion_time(&self) -> Option<SimTime> {
        self.due_actions.first_key_value().map(|(time, _)| *time)
    }

    pub fn next_metabolic_event_time(&self) -> Result<Option<SimTime>, ResolutionError> {
        self.metabolic_exhaustion.next_event()
    }

    pub fn next_event_time(&self) -> Result<Option<SimTime>, ResolutionError> {
        Ok([
            self.next_completion_time(),
            self.next_metabolic_event_time()?,
            self.next_diffusion_event_time()?,
        ]
        .into_iter()
        .flatten()
        .min())
    }

    pub fn next_diffusion_event_time(&self) -> Result<Option<SimTime>, ResolutionError> {
        if !self
            .diffusion_neighbors
            .iter()
            .any(|neighbors| !neighbors.is_empty())
        {
            return Ok(None);
        }
        let is_transportable = |tile: TileIndex| {
            !self.diffusion_neighbors[tile.0].is_empty()
                && u64::try_from(self.diffusion_neighbors[tile.0].len())
                    .is_ok_and(|neighbor_count| self.tiles[tile.0].diffuse_energy >= neighbor_count)
        };
        let has_transportable_field = if self.field_frontiers_dirty {
            (0..self.tiles.len()).map(TileIndex).any(is_transportable)
        } else {
            self.active_diffuse_tiles
                .iter()
                .copied()
                .any(is_transportable)
        };
        let metabolism_can_emit =
            self.rules.metabolism_rate_numerator > 0 && !self.active_metabolism_cells.is_empty();
        let signal_can_emit = self.rules.signal_decay_rate_numerator > 0
            && if self.field_frontiers_dirty {
                self.tiles
                    .iter()
                    .any(|tile| tile.signal_energy.iter().any(|energy| *energy > 0))
            } else {
                !self.active_signal_tiles.is_empty()
            };
        if self.rules.diffusion_rate_numerator == 0
            || (!has_transportable_field && !metabolism_can_emit && !signal_can_emit)
        {
            return Ok(None);
        }
        let interval = self.rules.diffusion_interval_quanta;
        let elapsed_to_tick = interval - (self.now.0 % interval);
        self.now
            .0
            .checked_add(elapsed_to_tick)
            .map(SimTime)
            .map(Some)
            .ok_or(ResolutionError::ArithmeticOverflow(
                "scheduling diffuse transport",
            ))
    }

    pub fn advance_clock_to(&mut self, requested: SimTime) -> Result<(), ResolutionError> {
        if requested < self.now {
            return Err(ResolutionError::TimeMovedBackwards {
                now: self.now,
                requested,
            });
        }
        if let Some(event) = self.next_event_time()? {
            if event <= requested {
                return Err(ResolutionError::CompletionBeforeRequestedTime {
                    completion: event,
                    requested,
                });
            }
        }
        self.apply_passive_until(requested).map(|_| ())
    }

    fn apply_passive_until(
        &mut self,
        requested: SimTime,
    ) -> Result<[u128; REFERENCE_SIGNAL_CHANNELS], ResolutionError> {
        let elapsed =
            requested
                .0
                .checked_sub(self.now.0)
                .ok_or(ResolutionError::TimeMovedBackwards {
                    now: self.now,
                    requested,
                })?;
        if elapsed == 0 {
            self.now = requested;
            return Ok([0; REFERENCE_SIGNAL_CHANNELS]);
        }

        self.rebuild_field_frontiers();
        self.mark_passive_hash_pages(requested);

        self.apply_plant_growth(elapsed)?;

        self.apply_digestion(elapsed)?;
        self.apply_metabolism(elapsed)?;
        let signal_energy_decayed = self.apply_signal_decay(elapsed)?;
        if requested
            .0
            .is_multiple_of(self.rules.diffusion_interval_quanta)
        {
            self.apply_diffusion_step()?;
        }
        self.now = requested;
        Ok(signal_energy_decayed)
    }

    fn mark_passive_hash_pages(&mut self, requested: SimTime) {
        let mark_all_cells = self.rules.metabolism_rate_numerator > 0
            && self.active_metabolism_cells.len() == self.cells.len();
        let mut changed_cells = BTreeSet::new();
        if !mark_all_cells {
            if self.rules.digestion_rate_numerator > 0 {
                changed_cells.extend(self.active_digestion_cells.iter().copied());
            }
            if self.rules.metabolism_rate_numerator > 0 {
                changed_cells.extend(self.active_metabolism_cells.iter().copied());
            }
        }
        let mut changed_tiles = std::collections::BTreeSet::new();
        if self.rules.metabolism_rate_numerator > 0 {
            if self.active_metabolism_cells.len() == self.cells.len() {
                changed_tiles.extend(self.cells.values().map(|cell| cell.position));
            } else {
                changed_tiles.extend(
                    self.active_metabolism_cells
                        .iter()
                        .filter_map(|key| self.cells.get(key).map(|cell| cell.position)),
                );
                changed_tiles.extend(
                    self.active_digestion_cells
                        .iter()
                        .filter_map(|key| self.cells.get(key).map(|cell| cell.position)),
                );
            }
        }
        if self.growing_plant_index_dirty {
            changed_tiles.extend((0..self.tiles.len()).map(TileIndex));
        } else {
            changed_tiles.extend(self.growing_plant_tiles.iter().copied());
        }
        changed_tiles.extend(self.field_changed_tiles_at(requested));

        if mark_all_cells {
            self.checkpoint_mutations
                .mark_cells(self.cells.keys().copied());
        } else {
            self.checkpoint_mutations
                .mark_cells(changed_cells.iter().copied());
        }
        self.checkpoint_mutations
            .mark_tiles(changed_tiles.iter().copied());

        let cache = self.state_hash_cache.get_mut();
        if mark_all_cells {
            cache.mark_all_cells();
        } else {
            for cell in &changed_cells {
                cache.mark_cell(*cell);
            }
        }
        for tile in &changed_tiles {
            cache.mark_tile(*tile);
        }
    }

    fn apply_digestion(&mut self, elapsed: u64) -> Result<(), ResolutionError> {
        let numerator = self.rules.digestion_rate_numerator;
        if numerator == 0 {
            return Ok(());
        }
        let denominator = self.rules.digestion_rate_denominator;
        let dense = self.active_digestion_cells.len() == self.cells.len();
        let sparse_active = if dense {
            Vec::new()
        } else {
            self.active_digestion_cells
                .iter()
                .copied()
                .collect::<Vec<_>>()
        };
        let now = self.now;
        let metabolism_numerator = self.rules.metabolism_rate_numerator;
        let metabolism_denominator = self.rules.metabolism_rate_denominator;
        #[cfg(not(target_arch = "wasm32"))]
        let mut parallel_applied = false;
        #[cfg(target_arch = "wasm32")]
        let parallel_applied = false;
        #[cfg(not(target_arch = "wasm32"))]
        let mut exhaustion_fused = false;
        #[cfg(target_arch = "wasm32")]
        let exhaustion_fused = false;
        #[cfg(not(target_arch = "wasm32"))]
        let parallel_candidate = dense
            && self.active_metabolism_cells.len() == self.cells.len()
            && self
                .passive_cell_parallel_threshold
                .is_some_and(|threshold| self.cells.len() >= threshold);
        #[cfg(not(target_arch = "wasm32"))]
        if parallel_candidate {
            let (transfer_fits, deadlines_fit) = self
                .cells
                .par_iter()
                .map(|(_, cell)| {
                    dense_digestion_preflight(
                        cell,
                        elapsed,
                        numerator,
                        denominator,
                        now,
                        metabolism_numerator,
                        metabolism_denominator,
                    )
                })
                .reduce(
                    || (true, true),
                    |left, right| (left.0 && right.0, left.1 && right.1),
                );
            if transfer_fits && deadlines_fit {
                let events = self
                    .cells
                    .par_iter_mut()
                    .map(|(key, cell)| {
                        advance_cell_digestion(cell, elapsed, numerator, denominator)?;
                        Ok((
                            metabolic_exhaustion_time(
                                now,
                                cell,
                                metabolism_numerator,
                                metabolism_denominator,
                            )?,
                            key,
                        ))
                    })
                    .collect::<Result<Vec<_>, ResolutionError>>()?;
                self.metabolic_exhaustion.replace_events(events);
                parallel_applied = true;
                exhaustion_fused = true;
            } else if transfer_fits {
                self.cells.par_iter_mut().try_for_each(|(_, cell)| {
                    advance_cell_digestion(cell, elapsed, numerator, denominator)
                })?;
                parallel_applied = true;
            }
        }
        if !parallel_applied && dense {
            for (key, cell) in &mut self.cells {
                advance_cell_digestion(cell, elapsed, numerator, denominator)?;
                if cell.assimilated_energy > 0 {
                    self.active_metabolism_cells.insert(key);
                    self.zero_energy_cells.remove(&key);
                }
            }
        } else if !parallel_applied {
            for key in sparse_active.iter().copied() {
                let cell = self
                    .cells
                    .get_mut(&key)
                    .ok_or(ResolutionError::InvariantViolation(
                        "active digestion index references a missing cell",
                    ))?;
                advance_cell_digestion(cell, elapsed, numerator, denominator)?;
                if cell.assimilated_energy > 0 {
                    self.active_metabolism_cells.insert(key);
                    self.zero_energy_cells.remove(&key);
                }
            }
        }
        let cells = &self.cells;
        self.active_digestion_cells
            .retain(|key| cells.get(key).is_some_and(|cell| cell.gut_energy > 0));

        // Digestion changes assimilated energy before metabolism runs, so it
        // is one of the few passive operations that changes absolute
        // exhaustion deadlines. Refresh exactly the cells that participated.
        if dense {
            if !exhaustion_fused {
                self.metabolic_exhaustion.rebuild(
                    &self.cells,
                    now,
                    metabolism_numerator,
                    metabolism_denominator,
                );
            }
        } else {
            let (cells, metabolic_exhaustion) = (&self.cells, &mut self.metabolic_exhaustion);
            for key in sparse_active {
                metabolic_exhaustion.sync(
                    key,
                    cells.get(&key),
                    now,
                    metabolism_numerator,
                    metabolism_denominator,
                );
            }
        }
        Ok(())
    }

    fn apply_metabolism(&mut self, elapsed: u64) -> Result<(), ResolutionError> {
        let numerator = self.rules.metabolism_rate_numerator;
        if numerator == 0 {
            return Ok(());
        }
        let denominator = self.rules.metabolism_rate_denominator;
        let divisor = PassiveRateDivisor::new(denominator);
        let mut activated = Vec::new();
        let mut exhausted = Vec::new();
        let dense = self.active_metabolism_cells.len() == self.cells.len();
        #[cfg(not(target_arch = "wasm32"))]
        let dense_gather_candidate = dense
            && !self.field_frontiers_dirty
            && rayon::current_num_threads() >= 4
            && self
                .passive_cell_parallel_threshold
                .is_some_and(|threshold| self.cells.len() >= threshold)
            && self
                .passive_tile_parallel_threshold
                .is_some_and(|threshold| self.tiles.len() >= threshold)
            && passive_frontier_is_dense(self.cells.len(), self.tiles.len());
        #[cfg(not(target_arch = "wasm32"))]
        let dense_gather = dense_gather_candidate && {
            let check = |cell: &CellState| {
                self.tiles[cell.position.0].diffuse_energy <= u64::MAX - cell.assimilated_energy
            };
            self.cells.par_iter().all(|(_, cell)| check(cell))
        };
        #[cfg(target_arch = "wasm32")]
        let dense_gather = false;
        let dense_gather_applied = dense_gather;
        let (cells, tiles, zero_energy_cells) = (
            &mut self.cells,
            &mut self.tiles,
            &mut self.zero_energy_cells,
        );
        #[cfg(not(target_arch = "wasm32"))]
        if dense_gather {
            self.dense_metabolism_deposits.prepare(tiles.len());
            let deposits = &self.dense_metabolism_deposits.0;
            let activates_diffuse = AtomicBool::new(false);
            let exhausts_cell = AtomicBool::new(false);
            {
                let tiles = &*tiles;
                cells.par_iter_mut().for_each(|(_, cell)| {
                    let spent =
                        advance_cell_metabolism_with_divisor(cell, elapsed, numerator, divisor)
                            .expect("validated dense metabolism arithmetic must fit");
                    if cell.assimilated_energy == 0 {
                        exhausts_cell.store(true, Ordering::Relaxed);
                    }
                    if spent > 0 {
                        if tiles[cell.position.0].diffuse_energy == 0 {
                            activates_diffuse.store(true, Ordering::Relaxed);
                        }
                        debug_assert_eq!(
                            deposits[cell.position.0].load(Ordering::Relaxed),
                            0,
                            "two cells emitted onto the same tile"
                        );
                        deposits[cell.position.0].store(spent, Ordering::Relaxed);
                    }
                });
            }
            tiles.par_for_each_mut_indexed(|index, tile| {
                let deposited = deposits[index].load(Ordering::Relaxed);
                deposits[index].store(0, Ordering::Relaxed);
                if deposited > 0 {
                    tile.diffuse_energy = tile
                        .diffuse_energy
                        .checked_add(deposited)
                        .expect("dense metabolism deposit exceeded its preflight bound");
                }
            });
            if exhausts_cell.load(Ordering::Relaxed) {
                for (key, cell) in cells.iter() {
                    if cell.assimilated_energy == 0 {
                        zero_energy_cells.insert(*key);
                        exhausted.push(*key);
                    }
                }
            }
            if activates_diffuse.load(Ordering::Relaxed) {
                self.active_diffuse_tiles = tiles
                    .iter()
                    .enumerate()
                    .filter_map(|(index, tile)| {
                        (tile.diffuse_energy > 0).then_some(TileIndex(index))
                    })
                    .collect();
            }
        }
        if !dense_gather_applied {
            if dense {
                for (key, cell) in cells.iter_mut() {
                    let spent =
                        advance_cell_metabolism_with_divisor(cell, elapsed, numerator, divisor)?;
                    if cell.assimilated_energy == 0 {
                        zero_energy_cells.insert(key);
                        exhausted.push(key);
                    }
                    if spent > 0 {
                        let tile = &mut tiles[cell.position.0];
                        let activates_diffuse = tile.diffuse_energy == 0;
                        tile.diffuse_energy = tile.diffuse_energy.checked_add(spent).ok_or(
                            ResolutionError::ArithmeticOverflow("depositing metabolic energy"),
                        )?;
                        if activates_diffuse {
                            activated.push(cell.position);
                        }
                    }
                }
            } else {
                let active = self
                    .active_metabolism_cells
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                for key in active {
                    let cell = cells
                        .get_mut(&key)
                        .ok_or(ResolutionError::InvariantViolation(
                            "active metabolism index references a missing cell",
                        ))?;
                    let spent =
                        advance_cell_metabolism_with_divisor(cell, elapsed, numerator, divisor)?;
                    if cell.assimilated_energy == 0 {
                        zero_energy_cells.insert(key);
                        exhausted.push(key);
                    }
                    if spent > 0 {
                        let tile = &mut tiles[cell.position.0];
                        let activates_diffuse = tile.diffuse_energy == 0;
                        tile.diffuse_energy = tile.diffuse_energy.checked_add(spent).ok_or(
                            ResolutionError::ArithmeticOverflow("depositing metabolic energy"),
                        )?;
                        if activates_diffuse {
                            activated.push(cell.position);
                        }
                    }
                }
            }
        }
        self.active_diffuse_tiles.extend(activated);
        if !exhausted.is_empty() {
            for key in exhausted {
                self.active_metabolism_cells.remove(&key);
                self.metabolic_exhaustion.sync(
                    key,
                    self.cells.get(&key),
                    self.now,
                    numerator,
                    denominator,
                );
            }
        }
        Ok(())
    }

    fn apply_signal_decay(
        &mut self,
        elapsed: u64,
    ) -> Result<[u128; REFERENCE_SIGNAL_CHANNELS], ResolutionError> {
        let numerator = self.rules.signal_decay_rate_numerator;
        if numerator == 0 {
            #[cfg(not(target_arch = "wasm32"))]
            if self
                .passive_tile_parallel_threshold
                .is_some_and(|threshold| self.tiles.len() >= threshold)
                && passive_frontier_is_dense(self.active_signal_tiles.len(), self.tiles.len())
            {
                self.tiles
                    .par_iter_mut()
                    .for_each(|tile| tile.signal_decay_remainder = [0; REFERENCE_SIGNAL_CHANNELS]);
                return Ok([0; REFERENCE_SIGNAL_CHANNELS]);
            }
            for tile_index in self.active_signal_tiles.clone() {
                self.tiles[tile_index.0].signal_decay_remainder = [0; REFERENCE_SIGNAL_CHANNELS];
            }
            return Ok([0; REFERENCE_SIGNAL_CHANNELS]);
        }
        let denominator = self.rules.signal_decay_rate_denominator;
        let mut activated_diffuse = Vec::new();
        #[cfg(not(target_arch = "wasm32"))]
        let parallel_candidate = self
            .passive_tile_parallel_threshold
            .is_some_and(|threshold| self.tiles.len() >= threshold)
            && passive_frontier_is_dense(self.active_signal_tiles.len(), self.tiles.len());
        #[cfg(not(target_arch = "wasm32"))]
        if parallel_candidate {
            let preflight = self
                .tiles
                .par_iter()
                .map(|tile| {
                    let (signal_expires, diffuse_activates) =
                        planned_signal_frontier_changes(tile, elapsed, numerator, denominator);
                    (
                        u128::from(tile.diffuse_energy)
                            + tile.signal_energy.into_iter().map(u128::from).sum::<u128>()
                            <= u128::from(u64::MAX),
                        signal_expires,
                        diffuse_activates,
                    )
                })
                .reduce(
                    || (true, false, false),
                    |left, right| (left.0 && right.0, left.1 || right.1, left.2 || right.2),
                );
            if preflight.0 {
                return self.apply_signal_decay_parallel_dense(
                    elapsed,
                    numerator,
                    denominator,
                    self.field_frontiers_dirty || preflight.1,
                    self.field_frontiers_dirty || preflight.2,
                );
            }
        }
        let frontier = self.active_signal_tiles.clone();
        let mut decayed = [0_u128; REFERENCE_SIGNAL_CHANNELS];
        for tile_index in frontier {
            let tile_decayed = advance_signal_decay(
                &mut self.tiles[tile_index.0],
                elapsed,
                numerator,
                denominator,
            )?;
            if tile_decayed.into_iter().any(|amount| amount > 0) {
                activated_diffuse.push(tile_index);
            }
            for channel in 0..REFERENCE_SIGNAL_CHANNELS {
                decayed[channel] =
                    decayed[channel].saturating_add(u128::from(tile_decayed[channel]));
            }
        }
        self.active_signal_tiles.retain(|tile| {
            self.tiles[tile.0]
                .signal_energy
                .iter()
                .any(|energy| *energy > 0)
        });
        self.active_diffuse_tiles.extend(activated_diffuse);
        Ok(decayed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn apply_signal_decay_parallel_dense(
        &mut self,
        elapsed: u64,
        numerator: u64,
        denominator: u64,
        rebuild_signal_frontier: bool,
        rebuild_diffuse_frontier: bool,
    ) -> Result<[u128; REFERENCE_SIGNAL_CHANNELS], ResolutionError> {
        let pages = self
            .tiles
            .pages
            .par_iter_mut()
            .map(|page| {
                let mut result = DenseSignalDecayPage {
                    decayed: [0; REFERENCE_SIGNAL_CHANNELS],
                    active_signal: [0; REFERENCE_TILE_PAGE_BITMAP_WORDS],
                    active_diffuse: [0; REFERENCE_TILE_PAGE_BITMAP_WORDS],
                };
                for (offset, tile) in Arc::make_mut(page).dense_tiles_mut().iter_mut().enumerate() {
                    let tile_decayed = advance_signal_decay(tile, elapsed, numerator, denominator)?;
                    for (total, decayed) in result.decayed.iter_mut().zip(tile_decayed) {
                        *total = total.saturating_add(u128::from(decayed));
                    }
                    if rebuild_signal_frontier
                        && tile.signal_energy.iter().any(|energy| *energy > 0)
                    {
                        result.active_signal[offset / 64] |= 1_u64 << (offset % 64);
                    }
                    if rebuild_diffuse_frontier && tile.diffuse_energy > 0 {
                        result.active_diffuse[offset / 64] |= 1_u64 << (offset % 64);
                    }
                }
                Ok::<_, ResolutionError>(result)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut decayed = [0_u128; REFERENCE_SIGNAL_CHANNELS];
        for page in &pages {
            for (total, page_total) in decayed.iter_mut().zip(page.decayed) {
                *total = total.saturating_add(page_total);
            }
        }
        if rebuild_signal_frontier {
            self.active_signal_tiles =
                dense_frontier_from_page_bitmaps(pages.iter().map(|page| &page.active_signal));
        }
        if rebuild_diffuse_frontier {
            self.active_diffuse_tiles =
                dense_frontier_from_page_bitmaps(pages.iter().map(|page| &page.active_diffuse));
        }
        Ok(decayed)
    }

    fn apply_diffusion_step(&mut self) -> Result<(), ResolutionError> {
        if self.rules.diffusion_rate_numerator == 0 {
            for tile in &self.active_diffuse_tiles {
                self.tiles[tile.0].diffusion_remainder = 0;
            }
            return Ok(());
        }
        let numerator = self.rules.diffusion_rate_numerator;
        let denominator = self.rules.diffusion_rate_denominator;
        #[cfg(not(target_arch = "wasm32"))]
        if self
            .passive_tile_parallel_threshold
            .is_some_and(|threshold| self.tiles.len() >= threshold)
            && passive_frontier_is_dense(self.active_diffuse_tiles.len(), self.tiles.len())
        {
            return self.apply_diffusion_step_parallel_dense(numerator, denominator);
        }
        for tile in self.diffusion_touched.drain(..) {
            self.diffusion_incoming[tile.0] = 0;
        }
        let sources = self
            .active_diffuse_tiles
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for source in &sources {
            let index = source.0;
            let energy = self.tiles[index].diffuse_energy;
            let neighbors = &self.diffusion_neighbors[index];
            let neighbor_count = u64::try_from(neighbors.len()).map_err(|_| {
                ResolutionError::ArithmeticOverflow("converting diffusion neighbor count")
            })?;
            if neighbors.is_empty() || energy < neighbor_count {
                self.tiles[index].diffusion_remainder = 0;
                continue;
            }
            let generated = u128::from(energy)
                .checked_mul(u128::from(numerator))
                .and_then(|value| {
                    value.checked_add(u128::from(self.tiles[index].diffusion_remainder))
                })
                .ok_or(ResolutionError::ArithmeticOverflow(
                    "accumulating diffuse transport",
                ))?;
            let desired =
                u64::try_from((generated / u128::from(denominator)).min(u128::from(energy)))
                    .map_err(|_| {
                        ResolutionError::ArithmeticOverflow("converting diffuse transport")
                    })?;
            self.tiles[index].diffusion_remainder =
                u64::try_from(generated % u128::from(denominator)).map_err(|_| {
                    ResolutionError::ArithmeticOverflow("storing diffusion remainder")
                })?;
            let share = desired / neighbor_count;
            let transported =
                share
                    .checked_mul(neighbor_count)
                    .ok_or(ResolutionError::ArithmeticOverflow(
                        "summing diffuse transport",
                    ))?;
            self.tiles[index].diffuse_energy -= transported;
            if share == 0 {
                continue;
            }
            for &neighbor in neighbors {
                if self.diffusion_incoming[neighbor.0] == 0 {
                    self.diffusion_touched.push(neighbor);
                }
                self.diffusion_incoming[neighbor.0] = self.diffusion_incoming[neighbor.0]
                    .checked_add(share)
                    .ok_or(ResolutionError::ArithmeticOverflow(
                        "aggregating diffuse transport",
                    ))?;
            }
        }
        for &tile_index in &self.diffusion_touched {
            let incoming = self.diffusion_incoming[tile_index.0];
            let tile = &mut self.tiles[tile_index.0];
            tile.diffuse_energy = tile.diffuse_energy.checked_add(incoming).ok_or(
                ResolutionError::ArithmeticOverflow("committing diffuse transport"),
            )?;
            if tile.diffuse_energy == 0 {
                tile.diffusion_remainder = 0;
            }
            self.diffusion_incoming[tile_index.0] = 0;
        }
        self.active_diffuse_tiles
            .extend(self.diffusion_touched.iter().copied());
        self.active_diffuse_tiles
            .retain(|tile| self.tiles[tile.0].diffuse_energy > 0);
        self.diffusion_touched.clear();
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn apply_diffusion_step_parallel_dense(
        &mut self,
        numerator: u64,
        denominator: u64,
    ) -> Result<(), ResolutionError> {
        for tile in self.diffusion_touched.drain(..) {
            self.diffusion_incoming[tile.0] = 0;
        }

        let shares = self
            .diffusion_neighbors
            .par_iter()
            .enumerate()
            .map(|(index, neighbors)| {
                plan_dense_diffusion(&self.tiles[index], neighbors, numerator, denominator).share
            })
            .collect::<Vec<_>>();

        let first_error = AtomicUsize::new(usize::MAX);
        let frontier_changes = AtomicBool::new(self.field_frontiers_dirty);
        self.diffusion_incoming.par_iter_mut().enumerate().for_each(
            |(destination, final_energy)| match dense_diffusion_final_energy(
                destination,
                &self.tiles,
                &self.diffusion_neighbors,
                &self.diffusion_sources,
                &shares,
            ) {
                Ok(energy) => {
                    *final_energy = energy;
                    if (self.tiles[destination].diffuse_energy > 0) != (energy > 0) {
                        frontier_changes.store(true, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    *final_energy = 0;
                    first_error.fetch_min(destination, Ordering::Relaxed);
                }
            },
        );
        let error_index = first_error.load(Ordering::Relaxed);
        if error_index != usize::MAX {
            let error = dense_diffusion_final_energy(
                error_index,
                &self.tiles,
                &self.diffusion_neighbors,
                &self.diffusion_sources,
                &shares,
            )
            .expect_err("recorded dense diffusion overflow must reproduce");
            self.diffusion_incoming
                .par_iter_mut()
                .for_each(|energy| *energy = 0);
            return Err(error);
        }

        self.tiles.par_for_each_mut_indexed(|index, tile| {
            let plan = plan_dense_diffusion(
                tile,
                &self.diffusion_neighbors[index],
                numerator,
                denominator,
            );
            debug_assert_eq!(plan.share, shares[index]);
            let final_energy = self.diffusion_incoming[index];
            tile.diffuse_energy = final_energy;
            tile.diffusion_remainder = if final_energy == 0 { 0 } else { plan.remainder };
        });
        if frontier_changes.load(Ordering::Relaxed) {
            self.active_diffuse_tiles = self
                .diffusion_incoming
                .iter()
                .enumerate()
                .filter_map(|(index, energy)| (*energy > 0).then_some(TileIndex(index)))
                .collect();
        }
        self.diffusion_incoming
            .par_iter_mut()
            .for_each(|energy| *energy = 0);
        self.diffusion_touched.clear();
        Ok(())
    }

    fn apply_plant_growth(&mut self, elapsed: u64) -> Result<(), ResolutionError> {
        if self.growing_plant_index_dirty {
            self.growing_plant_tiles.clear();
            for (index, tile) in self.tiles.iter_mut().enumerate() {
                if tile.plant_growth_rate > 0 {
                    self.growing_plant_tiles.push(TileIndex(index));
                } else {
                    tile.plant_growth_remainder = 0;
                }
            }
            self.growing_plant_index_dirty = false;
        }
        self.rebuild_field_frontiers();
        let divisor = PassiveRateDivisor::new(self.rules.time.arithmetic_quanta_per_unit);
        #[cfg(not(target_arch = "wasm32"))]
        let mut parallel_applied = false;
        #[cfg(target_arch = "wasm32")]
        let parallel_applied = false;
        #[cfg(not(target_arch = "wasm32"))]
        if self
            .passive_tile_parallel_threshold
            .is_some_and(|threshold| self.tiles.len() >= threshold)
            && passive_frontier_is_dense(self.growing_plant_tiles.len(), self.tiles.len())
        {
            let diffuse_frontier_changes = AtomicBool::new(false);
            self.tiles.par_iter_mut().try_for_each(|tile| {
                let had_diffuse_energy = tile.diffuse_energy > 0;
                advance_plant_growth_with_divisor(tile, elapsed, divisor)?;
                if had_diffuse_energy && tile.diffuse_energy == 0 {
                    diffuse_frontier_changes.store(true, Ordering::Relaxed);
                }
                Ok::<(), ResolutionError>(())
            })?;
            if diffuse_frontier_changes.load(Ordering::Relaxed) {
                self.active_diffuse_tiles
                    .retain(|tile| self.tiles[tile.0].diffuse_energy > 0);
            }
            parallel_applied = true;
        }
        if !parallel_applied {
            let mut exhausted_diffuse_tiles = Vec::new();
            for tile_index in &self.growing_plant_tiles {
                let tile = &mut self.tiles[tile_index.0];
                let had_diffuse_energy = tile.diffuse_energy > 0;
                advance_plant_growth_with_divisor(tile, elapsed, divisor)?;
                if had_diffuse_energy && tile.diffuse_energy == 0 {
                    exhausted_diffuse_tiles.push(*tile_index);
                }
            }
            for tile in exhausted_diffuse_tiles {
                self.active_diffuse_tiles.remove(&tile);
            }
        }
        Ok(())
    }

    pub fn resolve_next_batch(&mut self) -> Result<BatchReport, ResolutionError> {
        self.resolve_next_batch_with_integrity(IntegrityMode::Verified)
    }

    pub fn resolve_next_batch_with_integrity(
        &mut self,
        integrity_mode: IntegrityMode,
    ) -> Result<BatchReport, ResolutionError> {
        self.resolve_next_batch_impl(None, integrity_mode)
    }

    /// Uses ordered parallel intent validation above `parallel_threshold`.
    /// Canonical reduction and commit remain identical to the serial oracle.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_next_batch_parallel(
        &mut self,
        parallel_threshold: usize,
    ) -> Result<BatchReport, ResolutionError> {
        self.resolve_next_batch_parallel_with_integrity(parallel_threshold, IntegrityMode::Verified)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_next_batch_parallel_with_integrity(
        &mut self,
        parallel_threshold: usize,
        integrity_mode: IntegrityMode,
    ) -> Result<BatchReport, ResolutionError> {
        self.resolve_next_batch_impl(Some(parallel_threshold.max(2)), integrity_mode)
    }

    fn resolve_next_batch_impl(
        &mut self,
        parallel_threshold: Option<usize>,
        integrity_mode: IntegrityMode,
    ) -> Result<BatchReport, ResolutionError> {
        let total_started = PhaseTimer::start();
        let completion_time = self
            .next_event_time()?
            .ok_or(ResolutionError::NoScheduledEvents)?;
        let passive_started = PhaseTimer::start();
        let signal_energy_decayed = self.apply_passive_until(completion_time)?;
        let passive_updates_ns = passive_started.elapsed_ns();
        let prestate_hash_started = PhaseTimer::start();
        let pre_state_hash = (integrity_mode == IntegrityMode::Verified).then(|| self.state_hash());
        let prestate_hash_ns = prestate_hash_started.elapsed_ns();
        let prestate_started = PhaseTimer::start();
        #[cfg(debug_assertions)]
        let oracle_before = self.canonical_state();
        let mut due_actors = self
            .due_actions
            .get(&completion_time)
            .cloned()
            .unwrap_or_default();
        due_actors.sort_unstable();
        let snapshot_tiles = self.tiles.clone();
        let prestate_materialization_ns = prestate_started.elapsed_ns();

        // Immutable resolution reads are bounded to all tiles plus due actors
        // and their target occupants. The journal owns first-write before-values
        // independently, so release builds never clone the full cell population.
        let mut journal =
            MutationJournal::new(self.now, self.next_cell_key, self.last_resolution_metrics);
        let result = (|| -> Result<BatchReport, ResolutionError> {
            let validation_started = PhaseTimer::start();
            #[cfg(not(target_arch = "wasm32"))]
            let parallel_validation = parallel_threshold
                .is_some_and(|threshold| due_actors.len() >= threshold)
                && rayon::current_num_threads() > 1;
            #[cfg(target_arch = "wasm32")]
            let parallel_validation = {
                let _ = parallel_threshold;
                false
            };

            #[cfg(not(target_arch = "wasm32"))]
            let mut intents: Vec<WorkingIntent> = if parallel_validation {
                due_actors
                    .par_iter()
                    .map(|actor| self.working_intent(*actor, completion_time, &snapshot_tiles))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                due_actors
                    .into_iter()
                    .map(|actor| self.working_intent(actor, completion_time, &snapshot_tiles))
                    .collect::<Result<Vec<_>, _>>()?
            };
            #[cfg(target_arch = "wasm32")]
            let mut intents: Vec<WorkingIntent> = due_actors
                .into_iter()
                .map(|actor| self.working_intent(actor, completion_time, &snapshot_tiles))
                .collect::<Result<Vec<_>, _>>()?;

            if !intents.is_empty() {
                self.due_actions.remove(&completion_time);
            }

            journal.record_and_complete_cells_ordered(
                intents.iter().map(|intent| intent.actor),
                &mut self.cells,
                completion_time,
            )?;

            let mut claims = Vec::with_capacity(intents.len().saturating_mul(2));
            let mut occupancy_targets = Vec::new();
            for intent in &intents {
                claims.extend(intent.claims.iter().flatten().copied());
                if let Some(target) = intent.occupancy_claim {
                    occupancy_targets.push(target);
                }
            }
            let occupancy_claimants =
                OccupancyClaimCounts::new(occupancy_targets, snapshot_tiles.len())?;
            let sparse_occupancy_claims = occupancy_claimants.is_sparse();
            #[cfg(not(target_arch = "wasm32"))]
            if parallel_validation {
                intents.par_iter_mut().for_each(|intent| {
                    if let Some(target) = intent.occupancy_claim {
                        intent.status = Some(if occupancy_claimants.count(target) == 1 {
                            OutcomeStatus::Success
                        } else {
                            OutcomeStatus::Contested
                        });
                    }
                });
            } else {
                assign_occupancy_outcomes(&mut intents, &occupancy_claimants);
            }
            #[cfg(target_arch = "wasm32")]
            assign_occupancy_outcomes(&mut intents, &occupancy_claimants);
            let intent_validation_ns = validation_started.elapsed_ns();
            let resolution_started = PhaseTimer::start();
            let mut damage_allocation_by_intent: Option<Vec<Option<DamageAllocation>>> = None;
            let mut terrain_change_by_intent: Option<Vec<Option<TerrainChange>>> = None;

            for intent in &intents {
                let status = intent.status.ok_or(ResolutionError::InvariantViolation(
                    "intent was not assigned an outcome",
                ))?;
                if status != OutcomeStatus::Success {
                    match intent.pending.request {
                        ActionRequest::Split { .. } => {
                            self.add_cell_energy(intent.actor, intent.pending.payload_escrow)?;
                        }
                        ActionRequest::Regurgitate { .. } => {
                            self.add_gut_energy(intent.actor, intent.pending.payload_escrow)?;
                        }
                        ActionRequest::DepositTerrain => {
                            self.add_carried_material(intent.actor, intent.pending.payload_escrow)?;
                        }
                        ActionRequest::Wait
                        | ActionRequest::Move { .. }
                        | ActionRequest::Attack { .. }
                        | ActionRequest::Guard { .. }
                        | ActionRequest::Consume { .. }
                        | ActionRequest::Signal { .. }
                        | ActionRequest::Excavate => {}
                    }
                }
            }

            for (intent_index, intent) in intents.iter().enumerate() {
                if intent.status != Some(OutcomeStatus::Success) {
                    continue;
                }
                match intent.pending.request {
                    ActionRequest::Excavate => {
                        journal.record_tile(intent.pending.origin, &self.tiles)?;
                        let tile = &mut self.tiles[intent.pending.origin.0];
                        let elevation_before = tile.elevation;
                        tile.elevation = tile.elevation.checked_sub(1).ok_or(
                            ResolutionError::InvariantViolation(
                                "successful excavation hit terrain minimum",
                            ),
                        )?;
                        let changes = terrain_change_by_intent
                            .get_or_insert_with(|| vec![None; intents.len()]);
                        changes[intent_index] = Some(TerrainChange {
                            tile: intent.pending.origin,
                            elevation_before,
                            elevation_after: tile.elevation,
                            material_mass: self.rules.terrain_mass_per_elevation,
                        });
                        self.add_carried_material(
                            intent.actor,
                            self.rules.terrain_mass_per_elevation,
                        )?;
                        self.clear_tile_signals_into_diffuse(intent.pending.origin)?;
                    }
                    ActionRequest::DepositTerrain => {
                        journal.record_tile(intent.pending.origin, &self.tiles)?;
                        let tile = &mut self.tiles[intent.pending.origin.0];
                        let elevation_before = tile.elevation;
                        tile.elevation = tile.elevation.checked_add(1).ok_or(
                            ResolutionError::InvariantViolation(
                                "successful deposition hit terrain maximum",
                            ),
                        )?;
                        let changes = terrain_change_by_intent
                            .get_or_insert_with(|| vec![None; intents.len()]);
                        changes[intent_index] = Some(TerrainChange {
                            tile: intent.pending.origin,
                            elevation_before,
                            elevation_after: tile.elevation,
                            material_mass: self.rules.terrain_mass_per_elevation,
                        });
                        self.clear_tile_signals_into_diffuse(intent.pending.origin)?;
                    }
                    _ => {}
                }
            }

            // Bounded takes use only the completion snapshot. Apply them before any
            // same-batch additions so regurgitated or scattered energy cannot be
            // consumed until a later completion.
            for intent in &intents {
                if intent.status != Some(OutcomeStatus::Success)
                    || !matches!(intent.pending.request, ActionRequest::Consume { .. })
                {
                    continue;
                }
                let origin = intent.pending.origin;
                journal.record_tile(origin, &self.tiles)?;
                let tile = self
                    .tiles
                    .get_mut(origin.0)
                    .ok_or(ResolutionError::InvalidTile(origin))?;
                tile.plant_energy = tile.plant_energy.checked_sub(intent.consumed_plant).ok_or(
                    ResolutionError::InvariantViolation("consume exceeded snapshot plant energy"),
                )?;
                tile.loose_energy = tile.loose_energy.checked_sub(intent.consumed_loose).ok_or(
                    ResolutionError::InvariantViolation("consume exceeded snapshot loose energy"),
                )?;
                let intake = intent
                    .consumed_plant
                    .checked_add(intent.consumed_loose)
                    .ok_or(ResolutionError::ArithmeticOverflow(
                        "summing consumed energy",
                    ))?;
                self.add_gut_energy(intent.actor, intake)?;
            }

            for intent in &intents {
                if intent.status != Some(OutcomeStatus::Success)
                    || !matches!(intent.pending.request, ActionRequest::Regurgitate { .. })
                {
                    continue;
                }
                let target = intent
                    .pending
                    .target
                    .ok_or(ResolutionError::InvariantViolation(
                        "successful regurgitation has no target",
                    ))?;
                journal.record_tile(target, &self.tiles)?;
                self.add_loose_energy(target, intent.pending.payload_escrow)?;
            }

            for intent in &intents {
                if intent.status != Some(OutcomeStatus::Success) {
                    continue;
                }
                if matches!(intent.pending.request, ActionRequest::Move { .. }) {
                    let target =
                        intent
                            .pending
                            .target
                            .ok_or(ResolutionError::InvariantViolation(
                                "successful move has no target",
                            ))?;
                    let origin = intent.pending.origin;
                    journal.record_tile(origin, &self.tiles)?;
                    journal.record_tile(target, &self.tiles)?;
                    self.tiles[origin.0].occupant = None;
                    self.tiles[target.0].occupant = Some(intent.actor);
                    self.cells
                        .get_mut(&intent.actor)
                        .ok_or(ResolutionError::InvariantViolation(
                            "successful mover disappeared",
                        ))?
                        .position = target;
                }
            }

            let mut births = Vec::new();
            for intent in &intents {
                if intent.status != Some(OutcomeStatus::Success) {
                    continue;
                }
                let ActionRequest::Split {
                    marker,
                    private_memory,
                    ..
                } = &intent.pending.request
                else {
                    continue;
                };
                let target = intent
                    .pending
                    .target
                    .ok_or(ResolutionError::InvariantViolation(
                        "successful split has no target",
                    ))?;
                let allocation = intent.pending.payload_escrow;
                let child_energy = allocation.checked_sub(self.rules.child_core_mass).ok_or(
                    ResolutionError::InvariantViolation("child allocation is below core mass"),
                )?;
                let child = CellKey(self.next_cell_key);
                self.next_cell_key = self.next_cell_key.checked_add(1).ok_or(
                    ResolutionError::ArithmeticOverflow("allocating a child cell key"),
                )?;
                journal.record_tile(target, &self.tiles)?;
                journal.record_cell(child, &self.cells);
                self.tiles[target.0].occupant = Some(child);
                self.cells.insert(
                    child,
                    CellState {
                        position: target,
                        core_mass: self.rules.child_core_mass,
                        assimilated_energy: child_energy,
                        gut_energy: 0,
                        digestion_remainder: 0,
                        metabolism_remainder: 0,
                        carried_material_mass: 0,
                        marker: *marker,
                        guarded: false,
                        ready_at: completion_time,
                        pending_action: None,
                        last_outcome: None,
                        cold: Arc::new(CellColdState {
                            private_memory: Arc::from(private_memory.clone()),
                        }),
                    },
                );
                self.sync_cell_passive_indexes(child);
                births.push((intent.actor, child));
            }

            let mut raw_damage: BTreeMap<CellKey, (u64, bool)> = BTreeMap::new();
            for intent in &mut intents {
                let ActionRequest::Attack { .. } = intent.pending.request else {
                    continue;
                };
                if matches!(intent.status, Some(OutcomeStatus::Rejected(_))) {
                    continue;
                }
                let target = intent
                    .pending
                    .target
                    .ok_or(ResolutionError::InvariantViolation(
                        "accepted attack has no target",
                    ))?;
                journal.record_tile(target, &self.tiles)?;
                self.add_loose_energy(target, intent.pending.payload_escrow)?;
                if intent.status != Some(OutcomeStatus::Success) {
                    continue;
                }
                let victim = intent
                    .target_occupant
                    .ok_or(ResolutionError::InvariantViolation(
                        "successful attack has no victim",
                    ))?;
                journal.record_cell(victim, &self.cells);
                let payload_damage = multiply_ratio_floor(
                    intent.pending.payload_escrow,
                    self.rules.attack_payload_damage_numerator,
                    self.rules.attack_payload_damage_denominator,
                )?;
                let mass_damage = multiply_ratio_floor_u128_to_u64(
                    intent.attack_mass,
                    self.rules.attack_mass_damage_numerator,
                    self.rules.attack_mass_damage_denominator,
                )?;
                let damage = payload_damage
                    .checked_add(mass_damage)
                    .ok_or(ResolutionError::ArithmeticOverflow("summing attack damage"))?;
                let target_was_guarded =
                    intent
                        .target_was_guarded
                        .ok_or(ResolutionError::InvariantViolation(
                            "successful attack is missing snapshot guard state",
                        ))?;
                intent.attack_raw_damage = damage;
                let entry = raw_damage.entry(victim).or_insert((0, target_was_guarded));
                if entry.1 != target_was_guarded {
                    return Err(ResolutionError::InvariantViolation(
                        "attack intents disagree about snapshot guard state",
                    ));
                }
                entry.0 =
                    entry
                        .0
                        .checked_add(damage)
                        .ok_or(ResolutionError::ArithmeticOverflow(
                            "aggregating attack damage",
                        ))?;
            }

            let mut nontrivial_damage = BTreeMap::<CellKey, NontrivialDamageGroup>::new();
            for (victim, (raw, target_was_guarded)) in raw_damage {
                let adjusted = if target_was_guarded {
                    multiply_ratio_floor(
                        raw,
                        self.rules.guard_damage_numerator,
                        self.rules.guard_damage_denominator,
                    )?
                } else {
                    raw
                };
                let victim_state =
                    self.cells
                        .get(&victim)
                        .ok_or(ResolutionError::InvariantViolation(
                            "attack victim disappeared before damage",
                        ))?;
                let actual = adjusted.min(victim_state.assimilated_energy);
                if adjusted != raw || actual != adjusted {
                    nontrivial_damage.insert(
                        victim,
                        NontrivialDamageGroup {
                            adjusted,
                            actual,
                            contributors: Vec::new(),
                        },
                    );
                }

                let victim_state =
                    self.cells
                        .get_mut(&victim)
                        .ok_or(ResolutionError::InvariantViolation(
                            "attack victim disappeared before damage",
                        ))?;
                victim_state.assimilated_energy -= actual;
                let scatter_tile = victim_state.position;
                self.sync_cell_passive_indexes(victim);
                journal.record_tile(scatter_tile, &self.tiles)?;
                self.add_loose_energy(scatter_tile, actual)?;
            }

            if !nontrivial_damage.is_empty() {
                for (intent_index, intent) in intents.iter().enumerate() {
                    if intent.status != Some(OutcomeStatus::Success)
                        || !matches!(intent.pending.request, ActionRequest::Attack { .. })
                    {
                        continue;
                    }
                    let victim =
                        intent
                            .target_occupant
                            .ok_or(ResolutionError::InvariantViolation(
                                "successful attack lost its victim before attribution",
                            ))?;
                    if let Some(group) = nontrivial_damage.get_mut(&victim) {
                        group.contributors.push(intent_index);
                    }
                }

                let mut allocations = vec![None; intents.len()];
                for group in nontrivial_damage.values() {
                    if group.contributors.is_empty() {
                        return Err(ResolutionError::InvariantViolation(
                            "nontrivial damage group has no contributors",
                        ));
                    }
                    let raw_weights = group
                        .contributors
                        .iter()
                        .map(|index| (intents[*index].actor, intents[*index].attack_raw_damage))
                        .collect::<Vec<_>>();
                    let post_guard = proportional_largest_remainder(group.adjusted, &raw_weights)?;
                    let post_guard_weights = raw_weights
                        .iter()
                        .zip(&post_guard)
                        .map(|((actor, _), damage)| (*actor, *damage))
                        .collect::<Vec<_>>();
                    let applied =
                        proportional_largest_remainder(group.actual, &post_guard_weights)?;
                    for ((intent_index, post_guard), applied) in
                        group.contributors.iter().zip(post_guard).zip(applied)
                    {
                        let raw = intents[*intent_index].attack_raw_damage;
                        allocations[*intent_index] = Some(DamageAllocation {
                            mitigated: raw.checked_sub(post_guard).ok_or(
                                ResolutionError::InvariantViolation(
                                    "guard attribution exceeds raw attack damage",
                                ),
                            )?,
                            applied,
                            overkill: post_guard.checked_sub(applied).ok_or(
                                ResolutionError::InvariantViolation(
                                    "applied attribution exceeds post-guard attack damage",
                                ),
                            )?,
                        });
                    }
                }
                damage_allocation_by_intent = Some(allocations);
            }

            let mut deaths = Vec::new();
            let dead_keys = self.zero_energy_cells.iter().copied().collect::<Vec<_>>();
            let mut interrupted = Vec::new();
            for key in dead_keys {
                journal.record_cell(key, &self.cells);
                let cell = self
                    .cells
                    .remove(&key)
                    .ok_or(ResolutionError::InvariantViolation(
                        "dead cell disappeared during cleanup",
                    ))?;
                self.sync_cell_passive_indexes(key);
                journal.record_tile(cell.position, &self.tiles)?;
                if self.tiles[cell.position.0].occupant == Some(key) {
                    self.tiles[cell.position.0].occupant = None;
                }
                let pending_payload = cell
                    .pending_action
                    .as_ref()
                    .map_or(0, |action| action.payload_escrow);
                let spill = cell
                    .core_mass
                    .checked_add(cell.assimilated_energy)
                    .and_then(|value| value.checked_add(cell.gut_energy))
                    .and_then(|value| value.checked_add(cell.carried_material_mass))
                    .and_then(|value| value.checked_add(pending_payload))
                    .ok_or(ResolutionError::ArithmeticOverflow(
                        "summing a dead cell's spill",
                    ))?;
                self.add_loose_energy(cell.position, spill)?;
                if let Some(pending) = cell.pending_action {
                    self.unregister_due_action(pending.completes_at, key);
                    interrupted.push(ActionOutcome {
                        actor: key,
                        action: pending.request.kind(),
                        request: pending.request.clone(),
                        origin: pending.origin,
                        status: OutcomeStatus::Interrupted,
                        started_at: pending.started_at,
                        completed_at: completion_time,
                        target: pending.target,
                        effort_spent: pending.effort_spent,
                        payload: pending.payload_escrow,
                        consumed_energy: 0,
                        attack_damage: None,
                        terrain_change: None,
                    });
                }
                deaths.push(key);
            }

            let canonical_resolution_ns = resolution_started.elapsed_ns();
            let finalization_started = PhaseTimer::start();
            let mut outcomes = Vec::with_capacity(intents.len() + interrupted.len());
            for (intent_index, intent) in intents.iter().enumerate() {
                let status = intent.status.ok_or(ResolutionError::InvariantViolation(
                    "intent outcome disappeared before report construction",
                ))?;
                let attack_damage = if status == OutcomeStatus::Success
                    && matches!(intent.pending.request, ActionRequest::Attack { .. })
                {
                    let victim =
                        intent
                            .target_occupant
                            .ok_or(ResolutionError::InvariantViolation(
                                "successful attack is missing its attributed victim",
                            ))?;
                    let target_was_guarded =
                        intent
                            .target_was_guarded
                            .ok_or(ResolutionError::InvariantViolation(
                                "successful attack is missing attributed guard state",
                            ))?;
                    let allocation = damage_allocation_by_intent
                        .as_ref()
                        .and_then(|allocations| allocations[intent_index])
                        .unwrap_or(DamageAllocation {
                            mitigated: 0,
                            applied: intent.attack_raw_damage,
                            overkill: 0,
                        });
                    Some(AttackDamage {
                        victim,
                        target_was_guarded,
                        raw: intent.attack_raw_damage,
                        mitigated: allocation.mitigated,
                        applied: allocation.applied,
                        overkill: allocation.overkill,
                    })
                } else {
                    None
                };
                outcomes.push(ActionOutcome {
                    actor: intent.actor,
                    action: intent.pending.request.kind(),
                    request: intent.pending.request.clone(),
                    origin: intent.pending.origin,
                    status,
                    started_at: intent.pending.started_at,
                    completed_at: completion_time,
                    target: intent.pending.target,
                    effort_spent: intent.pending.effort_spent,
                    payload: intent.pending.payload_escrow,
                    consumed_energy: if status == OutcomeStatus::Success
                        && matches!(intent.pending.request, ActionRequest::Consume { .. })
                    {
                        intent
                            .consumed_plant
                            .checked_add(intent.consumed_loose)
                            .ok_or(ResolutionError::ArithmeticOverflow(
                                "reporting consumed energy",
                            ))?
                    } else {
                        0
                    },
                    attack_damage,
                    terrain_change: terrain_change_by_intent
                        .as_ref()
                        .and_then(|effects| effects[intent_index].clone()),
                });
            }
            outcomes.extend(interrupted);
            #[cfg(not(target_arch = "wasm32"))]
            if parallel_validation {
                outcomes.par_sort_unstable_by_key(|outcome| {
                    (outcome.actor, outcome.started_at, outcome.action)
                });
                claims.par_sort_unstable_by_key(|claim| (claim.key, claim.mode, claim.actor));
            } else {
                outcomes.sort_by_key(|outcome| (outcome.actor, outcome.started_at, outcome.action));
                claims.sort_by_key(|claim| (claim.key, claim.mode, claim.actor));
            }
            #[cfg(target_arch = "wasm32")]
            {
                outcomes.sort_by_key(|outcome| (outcome.actor, outcome.started_at, outcome.action));
                claims.sort_by_key(|claim| (claim.key, claim.mode, claim.actor));
            }
            let mut exclusive_components = 0;
            let mut contended_components = 0;
            let mut maximum_component_size = 0;
            let mut cursor = 0;
            while cursor < claims.len() {
                if claims[cursor].mode != AccessMode::Exclusive {
                    cursor += 1;
                    continue;
                }
                let key = claims[cursor].key;
                let start = cursor;
                while cursor < claims.len()
                    && claims[cursor].mode == AccessMode::Exclusive
                    && claims[cursor].key == key
                {
                    cursor += 1;
                }
                let size = cursor - start;
                exclusive_components += 1;
                contended_components += usize::from(size > 1);
                maximum_component_size = maximum_component_size.max(size);
            }
            let structural_metrics = ResolutionMetrics {
                due_intents: intents.len(),
                emitted_claims: claims.len(),
                exclusive_components,
                contended_components,
                maximum_component_size,
                parallel_validation,
                sparse_occupancy_claims,
                signal_energy_decayed,
                phase_timings: ResolutionPhaseTimings::default(),
            };
            deaths.sort();
            births.sort();

            for outcome in &outcomes {
                if let Some(cell) = self.cells.get_mut(&outcome.actor) {
                    cell.last_outcome = Some(outcome.status);
                }
            }
            let report_finalization_ns = finalization_started.elapsed_ns();

            let delta_started = PhaseTimer::start();
            let delta = journal.finish(self.now, self.next_cell_key, &self.tiles, &self.cells);
            #[cfg(debug_assertions)]
            {
                let oracle = SimulationDelta::between(&oracle_before, &self.canonical_state());
                assert_eq!(
                    delta, oracle,
                    "mutation journal diverged from full-state oracle"
                );
            }
            let delta_generation_ns = delta_started.elapsed_ns();
            {
                let cache = self.state_hash_cache.get_mut();
                for tile in &delta.tiles {
                    cache.mark_tile(tile.tile);
                }
                for cell in &delta.cells {
                    cache.mark_cell(cell.cell);
                }
            }
            for tile in &delta.tiles {
                self.checkpoint_mutations.mark_tile(tile.tile);
            }
            for cell in &delta.cells {
                if cell.before.is_some() == cell.after.is_some() {
                    self.checkpoint_mutations.mark_cell(cell.cell);
                } else {
                    self.checkpoint_mutations.mark_cell_structure(cell.cell);
                }
            }
            let poststate_hash_started = PhaseTimer::start();
            let compiled_ruleset_hash = self.compiled_ruleset_hash();
            let state_hash = (integrity_mode == IntegrityMode::Verified).then(|| self.state_hash());
            let integrity = match (pre_state_hash, state_hash) {
                (Some(pre_state_hash), Some(state_hash)) => BatchIntegrity::Verified {
                    pre_state_hash,
                    state_hash,
                },
                (None, None) => BatchIntegrity::Unverified,
                _ => unreachable!("integrity mode produced a partial state boundary"),
            };
            let state_hashes_ns =
                prestate_hash_ns.saturating_add(poststate_hash_started.elapsed_ns());
            self.last_resolution_metrics = ResolutionMetrics {
                signal_energy_decayed,
                phase_timings: ResolutionPhaseTimings {
                    passive_updates_ns,
                    prestate_materialization_ns,
                    intent_validation_ns,
                    canonical_resolution_ns,
                    report_finalization_ns,
                    state_hashes_ns,
                    poststate_materialization_ns: 0,
                    delta_generation_ns,
                    total_ns: total_started.elapsed_ns(),
                },
                ..structural_metrics
            };
            Ok(BatchReport {
                completed_at: completion_time,
                outcomes,
                claims,
                deaths,
                births,
                compiled_ruleset_hash,
                integrity,
                delta,
            })
        })();
        if result.is_err() {
            journal.rollback(self);
        }
        result
    }

    fn working_intent(
        &self,
        actor: CellKey,
        completion_time: SimTime,
        snapshot_tiles: &ReferenceTileStore,
    ) -> Result<WorkingIntent, ResolutionError> {
        let snapshot_actor = self
            .cells
            .get(&actor)
            .ok_or(ResolutionError::InvariantViolation(
                "due action actor is absent from snapshot",
            ))?;
        let pending =
            snapshot_actor
                .pending_action
                .as_ref()
                .ok_or(ResolutionError::InvariantViolation(
                    "due action actor has no pending action",
                ))?;
        if pending.completes_at != completion_time {
            return Err(ResolutionError::InvariantViolation(
                "completion index points at a different timestamp",
            ));
        }
        let mut intent = WorkingIntent {
            actor,
            pending: pending.as_ref().clone(),
            status: pending.rejection.map(OutcomeStatus::Rejected),
            target_occupant: None,
            consumed_plant: 0,
            consumed_loose: 0,
            attack_mass: 0,
            attack_raw_damage: 0,
            target_was_guarded: None,
            claims: [None; 2],
            occupancy_claim: None,
        };
        if intent.status.is_some() {
            return Ok(intent);
        }
        if snapshot_actor.position != intent.pending.origin {
            intent.status = Some(OutcomeStatus::Frustrated);
            return Ok(intent);
        }
        match &intent.pending.request {
            ActionRequest::Wait | ActionRequest::Guard { .. } | ActionRequest::Signal { .. } => {
                intent.status = Some(OutcomeStatus::Success);
            }
            ActionRequest::Move { target: slot, .. } => {
                let Some(target) = intent.pending.target else {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                };
                if !self.elevation_reachable(snapshot_tiles, intent.pending.origin, target)
                    || self.diagonal_move_blocked(snapshot_tiles, intent.pending.origin, *slot)
                    || snapshot_tiles[target.0].occupant.is_some()
                {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                }
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::Occupancy(target),
                    mode: AccessMode::Exclusive,
                })?;
                intent.occupancy_claim = Some(target);
            }
            ActionRequest::Split { .. } => {
                let Some(target) = intent.pending.target else {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                };
                if !self.elevation_reachable(snapshot_tiles, intent.pending.origin, target)
                    || snapshot_tiles[target.0].occupant.is_some()
                {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                }
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::Occupancy(target),
                    mode: AccessMode::Exclusive,
                })?;
                intent.occupancy_claim = Some(target);
            }
            ActionRequest::Attack { .. } => {
                let Some(target) = intent.pending.target else {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                };
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::LooseEnergy(target),
                    mode: AccessMode::Add,
                })?;
                if !self.elevation_reachable(snapshot_tiles, intent.pending.origin, target) {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                }
                intent.target_occupant = snapshot_tiles[target.0].occupant;
                if let Some(target_cell) = intent.target_occupant {
                    let target_state =
                        self.cells
                            .get(&target_cell)
                            .ok_or(ResolutionError::InvariantViolation(
                                "snapshot tile occupant is absent from cells",
                            ))?;
                    intent.attack_mass = snapshot_actor.total_mass();
                    intent.target_was_guarded = Some(target_state.guarded);
                    intent.push_claim(ResourceClaim {
                        actor: intent.actor,
                        key: ResourceKey::CellState(target_cell),
                        mode: AccessMode::Add,
                    })?;
                    intent.status = Some(OutcomeStatus::Success);
                } else {
                    intent.status = Some(OutcomeStatus::Frustrated);
                }
            }
            ActionRequest::Consume { amount } => {
                let amount = *amount;
                let origin = intent.pending.origin;
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::PlantEnergy(origin),
                    mode: AccessMode::BoundedTake,
                })?;
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::LooseEnergy(origin),
                    mode: AccessMode::BoundedTake,
                })?;
                let remaining_capacity = self
                    .rules
                    .gut_capacity
                    .saturating_sub(snapshot_actor.gut_energy);
                let requested = amount.min(self.rules.bite_capacity).min(remaining_capacity);
                let plant = snapshot_tiles[origin.0].plant_energy.min(requested);
                let loose = snapshot_tiles[origin.0].loose_energy.min(requested - plant);
                intent.consumed_plant = plant;
                intent.consumed_loose = loose;
                intent.status = Some(if plant + loose == 0 {
                    OutcomeStatus::Frustrated
                } else {
                    OutcomeStatus::Success
                });
            }
            ActionRequest::Regurgitate { .. } => {
                let Some(target) = intent.pending.target else {
                    intent.status = Some(OutcomeStatus::Frustrated);
                    return Ok(intent);
                };
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::LooseEnergy(target),
                    mode: AccessMode::Add,
                })?;
                intent.status = Some(
                    if self.elevation_reachable(snapshot_tiles, intent.pending.origin, target) {
                        OutcomeStatus::Success
                    } else {
                        OutcomeStatus::Frustrated
                    },
                );
            }
            ActionRequest::Excavate => {
                let origin = intent.pending.origin;
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::Terrain(origin),
                    mode: AccessMode::Exclusive,
                })?;
                intent.status = Some(if snapshot_tiles[origin.0].elevation > i16::MIN {
                    OutcomeStatus::Success
                } else {
                    OutcomeStatus::Frustrated
                });
            }
            ActionRequest::DepositTerrain => {
                let origin = intent.pending.origin;
                intent.push_claim(ResourceClaim {
                    actor: intent.actor,
                    key: ResourceKey::Terrain(origin),
                    mode: AccessMode::Exclusive,
                })?;
                intent.status = Some(if snapshot_tiles[origin.0].elevation < i16::MAX {
                    OutcomeStatus::Success
                } else {
                    OutcomeStatus::Frustrated
                });
            }
        }
        Ok(intent)
    }

    fn unregister_due_action(&mut self, time: SimTime, actor: CellKey) {
        let remove_bucket = self.due_actions.get_mut(&time).is_some_and(|actors| {
            actors.retain(|candidate| *candidate != actor);
            actors.is_empty()
        });
        if remove_bucket {
            self.due_actions.remove(&time);
        }
    }

    fn elevation_reachable(
        &self,
        snapshot_tiles: &ReferenceTileStore,
        origin: TileIndex,
        target: TileIndex,
    ) -> bool {
        let difference = i32::from(snapshot_tiles[origin.0].elevation)
            - i32::from(snapshot_tiles[target.0].elevation);
        difference.abs() <= i32::from(self.rules.maximum_elevation_delta)
    }

    fn diagonal_move_blocked(
        &self,
        snapshot_tiles: &ReferenceTileStore,
        origin: TileIndex,
        slot: LocalSlot,
    ) -> bool {
        let Some(offset) = self.neighborhood.offset(slot) else {
            return true;
        };
        if !offset.is_diagonal()
            || self.neighborhood.spec().diagonal_corner_rule == DiagonalCornerRule::Allow
        {
            return false;
        }
        let horizontal = self
            .neighborhood
            .target_at_offset(origin, offset.dx, 0)
            .is_none_or(|tile| snapshot_tiles[tile.0].occupant.is_some());
        let vertical = self
            .neighborhood
            .target_at_offset(origin, 0, offset.dy)
            .is_none_or(|tile| snapshot_tiles[tile.0].occupant.is_some());
        match self.neighborhood.spec().diagonal_corner_rule {
            DiagonalCornerRule::Allow => false,
            DiagonalCornerRule::BlockIfEitherOrthogonalOccupied => horizontal || vertical,
            DiagonalCornerRule::BlockIfBothOrthogonalsOccupied => horizontal && vertical,
        }
    }

    fn add_cell_energy(&mut self, actor: CellKey, amount: u64) -> Result<(), ResolutionError> {
        let cell = self
            .cells
            .get_mut(&actor)
            .ok_or(ResolutionError::InvariantViolation(
                "payload refund actor disappeared",
            ))?;
        cell.assimilated_energy = cell.assimilated_energy.checked_add(amount).ok_or(
            ResolutionError::ArithmeticOverflow("refunding action payload"),
        )?;
        self.sync_cell_passive_indexes(actor);
        Ok(())
    }

    fn add_gut_energy(&mut self, actor: CellKey, amount: u64) -> Result<(), ResolutionError> {
        let cell = self
            .cells
            .get_mut(&actor)
            .ok_or(ResolutionError::InvariantViolation(
                "gut transfer actor disappeared",
            ))?;
        cell.gut_energy = cell
            .gut_energy
            .checked_add(amount)
            .ok_or(ResolutionError::ArithmeticOverflow("adding gut energy"))?;
        self.sync_cell_passive_indexes(actor);
        Ok(())
    }

    fn add_carried_material(&mut self, actor: CellKey, amount: u64) -> Result<(), ResolutionError> {
        let cell = self
            .cells
            .get_mut(&actor)
            .ok_or(ResolutionError::InvariantViolation(
                "material transfer actor disappeared",
            ))?;
        cell.carried_material_mass = cell.carried_material_mass.checked_add(amount).ok_or(
            ResolutionError::ArithmeticOverflow("adding carried material"),
        )?;
        Ok(())
    }

    fn add_loose_energy(&mut self, tile: TileIndex, amount: u64) -> Result<(), ResolutionError> {
        let tile_state = self
            .tiles
            .get_mut(tile.0)
            .ok_or(ResolutionError::InvalidTile(tile))?;
        tile_state.loose_energy = tile_state
            .loose_energy
            .checked_add(amount)
            .ok_or(ResolutionError::ArithmeticOverflow("adding loose energy"))?;
        Ok(())
    }
}

fn ceil_ratio(numerator: u64, denominator: u64) -> Result<u64, RejectReason> {
    if denominator == 0 {
        return Err(RejectReason::ArithmeticOverflow);
    }
    let adjusted = numerator
        .checked_add(denominator - 1)
        .ok_or(RejectReason::ArithmeticOverflow)?;
    Ok(adjusted / denominator)
}

fn multiply_ratio_ceil(value: u64, numerator: u64, denominator: u64) -> Result<u64, RejectReason> {
    let product = value
        .checked_mul(numerator)
        .ok_or(RejectReason::ArithmeticOverflow)?;
    ceil_ratio(product, denominator)
}

fn multiply_ratio_ceil_u128_to_u64(
    value: u128,
    numerator: u64,
    denominator: u64,
) -> Result<u64, RejectReason> {
    if denominator == 0 {
        return Err(RejectReason::ArithmeticOverflow);
    }
    let product = value
        .checked_mul(u128::from(numerator))
        .ok_or(RejectReason::ArithmeticOverflow)?;
    let adjusted = product
        .checked_add(u128::from(denominator - 1))
        .ok_or(RejectReason::ArithmeticOverflow)?;
    u64::try_from(adjusted / u128::from(denominator)).map_err(|_| RejectReason::ArithmeticOverflow)
}

/// Allocates an already-authoritative aggregate across actor contributions
/// using Hamilton's largest-remainder method. Actor key breaks equal-remainder
/// ties, so attribution is independent of commit iteration and worker count.
fn proportional_largest_remainder(
    total: u64,
    weighted_actors: &[(CellKey, u64)],
) -> Result<Vec<u64>, ResolutionError> {
    let total_weight = weighted_actors
        .iter()
        .try_fold(0_u128, |sum, (_, weight)| {
            sum.checked_add(u128::from(*weight))
                .ok_or(ResolutionError::ArithmeticOverflow(
                    "summing damage-attribution weights",
                ))
        })?;
    if total_weight == 0 {
        if total == 0 {
            return Ok(vec![0; weighted_actors.len()]);
        }
        return Err(ResolutionError::InvariantViolation(
            "positive damage cannot be attributed across zero weight",
        ));
    }
    if u128::from(total) > total_weight {
        return Err(ResolutionError::InvariantViolation(
            "attributed damage exceeds its contribution weights",
        ));
    }

    let mut allocations = Vec::with_capacity(weighted_actors.len());
    let mut remainders = Vec::with_capacity(weighted_actors.len());
    let mut allocated = 0_u64;
    for (_, weight) in weighted_actors {
        let product = u128::from(total) * u128::from(*weight);
        let base = u64::try_from(product / total_weight).map_err(|_| {
            ResolutionError::ArithmeticOverflow("converting proportional damage allocation")
        })?;
        allocated = allocated
            .checked_add(base)
            .ok_or(ResolutionError::ArithmeticOverflow(
                "summing proportional damage allocations",
            ))?;
        allocations.push(base);
        remainders.push(product % total_weight);
    }

    let leftover = usize::try_from(total.checked_sub(allocated).ok_or(
        ResolutionError::InvariantViolation("base damage allocations exceed their total"),
    )?)
    .map_err(|_| ResolutionError::ArithmeticOverflow("converting leftover damage count"))?;
    if leftover > weighted_actors.len() {
        return Err(ResolutionError::InvariantViolation(
            "largest-remainder damage allocation exceeded its contributor count",
        ));
    }
    let mut ranked = (0..weighted_actors.len()).collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        remainders[*right]
            .cmp(&remainders[*left])
            .then_with(|| weighted_actors[*left].0.cmp(&weighted_actors[*right].0))
    });
    for index in ranked.into_iter().take(leftover) {
        allocations[index] =
            allocations[index]
                .checked_add(1)
                .ok_or(ResolutionError::ArithmeticOverflow(
                    "assigning leftover attributed damage",
                ))?;
    }
    Ok(allocations)
}

fn multiply_ratio_floor(
    value: u64,
    numerator: u64,
    denominator: u64,
) -> Result<u64, ResolutionError> {
    if denominator == 0 {
        return Err(ResolutionError::InvalidRuleset(
            "runtime ratio denominator is zero",
        ));
    }
    value
        .checked_mul(numerator)
        .map(|product| product / denominator)
        .ok_or(ResolutionError::ArithmeticOverflow(
            "multiplying a resolution ratio",
        ))
}

fn multiply_ratio_floor_u128_to_u64(
    value: u128,
    numerator: u64,
    denominator: u64,
) -> Result<u64, ResolutionError> {
    if denominator == 0 {
        return Err(ResolutionError::InvalidRuleset(
            "runtime ratio denominator is zero",
        ));
    }
    let product =
        value
            .checked_mul(u128::from(numerator))
            .ok_or(ResolutionError::ArithmeticOverflow(
                "multiplying a wide resolution ratio",
            ))?;
    u64::try_from(product / u128::from(denominator))
        .map_err(|_| ResolutionError::ArithmeticOverflow("converting a wide resolution ratio"))
}

enum OccupancyClaimCounts {
    Dense(Vec<usize>),
    Sparse(Vec<(TileIndex, usize)>),
}

impl OccupancyClaimCounts {
    fn new(mut targets: Vec<TileIndex>, tile_count: usize) -> Result<Self, ResolutionError> {
        // A dense counter has excellent lookup behavior when much of the board
        // participates, but zeroing it for a tiny local batch on a huge board
        // makes resolution scale with world area. Keep the dense fast path only
        // when claims are numerous enough to amortize it.
        if targets.len().saturating_mul(4) >= tile_count {
            let mut counts = vec![0_usize; tile_count];
            for target in targets {
                counts[target.0] =
                    counts[target.0]
                        .checked_add(1)
                        .ok_or(ResolutionError::ArithmeticOverflow(
                            "counting occupancy claimants",
                        ))?;
            }
            return Ok(Self::Dense(counts));
        }

        targets.sort_unstable();
        let mut counts: Vec<(TileIndex, usize)> = Vec::new();
        for target in targets {
            if let Some((previous, count)) = counts.last_mut() {
                if *previous == target {
                    *count = count
                        .checked_add(1)
                        .ok_or(ResolutionError::ArithmeticOverflow(
                            "counting occupancy claimants",
                        ))?;
                    continue;
                }
            }
            counts.push((target, 1));
        }
        Ok(Self::Sparse(counts))
    }

    fn count(&self, target: TileIndex) -> usize {
        match self {
            Self::Dense(counts) => counts[target.0],
            Self::Sparse(counts) => counts
                .binary_search_by_key(&target, |(candidate, _)| *candidate)
                .ok()
                .map_or(0, |index| counts[index].1),
        }
    }

    const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }
}

fn assign_occupancy_outcomes(
    intents: &mut [WorkingIntent],
    claimant_counts: &OccupancyClaimCounts,
) {
    for intent in intents {
        if let Some(target) = intent.occupancy_claim {
            intent.status = Some(if claimant_counts.count(target) == 1 {
                OutcomeStatus::Success
            } else {
                OutcomeStatus::Contested
            });
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct PhaseTimer(std::time::Instant);

#[cfg(not(target_arch = "wasm32"))]
impl PhaseTimer {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }

    fn elapsed_ns(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

#[cfg(target_arch = "wasm32")]
struct PhaseTimer;

#[cfg(target_arch = "wasm32")]
impl PhaseTimer {
    const fn start() -> Self {
        Self
    }

    const fn elapsed_ns(&self) -> u64 {
        0
    }
}

fn round_up(value: u64, bucket: u64) -> Result<u64, RejectReason> {
    let buckets = ceil_ratio(value, bucket)?;
    buckets
        .checked_mul(bucket)
        .ok_or(RejectReason::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_store_is_row_major_and_detaches_only_the_written_chunk() {
        let values = (0..(REFERENCE_TILE_CHUNK_LEN * REFERENCE_TILE_PAGE_CHUNKS + 3))
            .map(|index| TileState {
                loose_energy: u64::try_from(index).unwrap(),
                ..TileState::default()
            })
            .collect::<Vec<_>>();
        let original = ReferenceTileStore::from_vec(values.clone());
        let mut branch = original.clone();
        assert_eq!(branch.to_vec(), values);
        assert!(Arc::ptr_eq(&original.pages[0], &branch.pages[0]));
        assert!(Arc::ptr_eq(&original.pages[1], &branch.pages[1]));

        let changed = REFERENCE_TILE_CHUNK_LEN + 1;
        let changed_chunk = changed / REFERENCE_TILE_CHUNK_LEN;
        let original_changed_allocation = original.pages[0].chunk_allocation(changed_chunk).0;
        let original_first_allocation = original.pages[0].chunk_allocation(0).0;
        branch[changed].loose_energy = 99_999;
        assert!(!Arc::ptr_eq(&original.pages[0], &branch.pages[0]));
        assert!(Arc::ptr_eq(&original.pages[1], &branch.pages[1]));
        assert_ne!(
            original_changed_allocation,
            branch.pages[0].chunk_allocation(changed_chunk).0
        );
        assert_eq!(
            original_first_allocation,
            branch.pages[0].chunk_allocation(0).0
        );
        assert!(branch.pages[0].overlays[changed_chunk].is_some());
        assert_eq!(
            original[changed].loose_energy,
            u64::try_from(changed).unwrap()
        );
        assert_eq!(branch[changed].loose_energy, 99_999);

        let adopted = ReferenceTileStore::from_pages(branch.pages.clone(), branch.len()).unwrap();
        assert_eq!(adopted, branch);
        let short_page = Arc::new(ReferenceTilePageData::from_vec(vec![
            TileState::default();
            7
        ]));
        assert!(ReferenceTileStore::from_pages(vec![Arc::clone(&short_page)], 7).is_some());
        let malformed = vec![
            short_page,
            Arc::new(ReferenceTilePageData::from_vec(vec![TileState::default()])),
        ];
        assert!(ReferenceTileStore::from_pages(malformed, 8).is_none());

        branch.iter_mut().for_each(|tile| {
            std::hint::black_box(tile);
        });
        assert!(branch.pages[0].overlays.iter().all(Option::is_none));
        assert_eq!(branch[changed].loose_energy, 99_999);
        assert_eq!(
            branch.pages[0].chunk_allocation(1).0,
            branch.pages[0].base[REFERENCE_TILE_CHUNK_LEN..].as_ptr()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_canonical_tile_validation_matches_serial_order_and_indexes() {
        let rules = ReferenceRuleset::default();
        let cells = CellStore::new();
        let mut tiles = vec![TileState::default(); 513 * 517];
        for (index, tile) in tiles.iter_mut().enumerate() {
            if index % 97 == 0 {
                tile.plant_capacity = 1_000;
                tile.plant_energy = 100;
                tile.plant_growth_rate = 3;
            }
            if index % 101 == 0 {
                tile.signal_energy[index % REFERENCE_SIGNAL_CHANNELS] =
                    u64::try_from(index + 1).unwrap();
            }
            if index % 103 == 0 {
                tile.diffuse_energy = u64::try_from(index + 1).unwrap();
            }
        }
        let mut tiles = ReferenceTileStore::from_vec(tiles);
        assert_eq!(
            validate_canonical_tiles_parallel(&tiles, &rules, &cells),
            validate_canonical_tiles_serial(&tiles, &rules, &cells),
        );

        tiles[CANONICAL_TILE_VALIDATION_PARTITION_LEN + 17].diffusion_remainder = 1;
        tiles[CANONICAL_TILE_VALIDATION_PARTITION_LEN * 2 + 3].plant_growth_remainder =
            rules.time.arithmetic_quanta_per_unit;
        assert_eq!(
            validate_canonical_tiles_parallel(&tiles, &rules, &cells),
            validate_canonical_tiles_serial(&tiles, &rules, &cells),
        );
        assert_eq!(
            validate_canonical_tiles_parallel(&tiles, &rules, &cells),
            Err(ResolutionError::InvalidState(
                "tile without diffuse energy has a diffusion remainder"
            )),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "manual serial/parallel canonical tile-validation crossover diagnostic"]
    fn benchmark_parallel_canonical_tile_validation_crossover() {
        use std::time::Instant;

        let rules = ReferenceRuleset::default();
        let cells = CellStore::new();
        for (density, stride) in [("sparse", 97usize), ("dense", 1)] {
            for (board, iterations) in [
                (128usize, 256usize),
                (256, 128),
                (384, 64),
                (512, 32),
                (768, 16),
                (1_024, 8),
            ] {
                let mut tiles = vec![TileState::default(); board * board];
                for (index, tile) in tiles.iter_mut().enumerate() {
                    if index % stride == 0 {
                        tile.plant_capacity = 1_000;
                        tile.plant_energy = 100;
                        tile.plant_growth_rate = 3;
                        tile.signal_energy[index % REFERENCE_SIGNAL_CHANNELS] = 100;
                        tile.diffuse_energy = 100;
                    }
                }
                let tiles = ReferenceTileStore::from_vec(tiles);
                assert_eq!(
                    validate_canonical_tiles_parallel(&tiles, &rules, &cells),
                    validate_canonical_tiles_serial(&tiles, &rules, &cells),
                );

                let serial_started = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(
                        validate_canonical_tiles_serial(&tiles, &rules, &cells).unwrap(),
                    );
                }
                let serial_ns = serial_started.elapsed().as_nanos() / iterations as u128;

                let parallel_started = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(
                        validate_canonical_tiles_parallel(&tiles, &rules, &cells).unwrap(),
                    );
                }
                let parallel_ns = parallel_started.elapsed().as_nanos() / iterations as u128;
                println!(
                    "canonical tile validation density={density} {board}x{board} iterations={iterations} threads={} serial_ns={serial_ns} parallel_ns={parallel_ns}",
                    rayon::current_num_threads(),
                );
            }
        }
    }

    #[test]
    fn compiled_topology_is_shared_only_with_matching_rules_and_dimensions() {
        let rules = ReferenceRuleset::default();
        let simulation = ReferenceSimulation::new(8, 8, rules.clone()).unwrap();
        let topology = simulation.compiled_topology();
        let restored = ReferenceSimulation::from_canonical_state_with_compiled_topology(
            rules.clone(),
            simulation.canonical_state(),
            topology.clone(),
        )
        .unwrap();
        assert!(Arc::ptr_eq(
            &simulation.neighborhood,
            &restored.neighborhood
        ));
        assert!(Arc::ptr_eq(
            &simulation.diffusion_neighbors,
            &restored.diffusion_neighbors
        ));

        let mut mismatched_rules = rules;
        mismatched_rules.move_effort_base += 1;
        assert!(ReferenceSimulation::new_with_compiled_topology(
            mismatched_rules,
            topology.clone()
        )
        .is_err());
        assert!(
            ReferenceSimulation::from_canonical_state_with_compiled_topology(
                ReferenceRuleset::default(),
                SimulationState {
                    now: SimTime(0),
                    tiles: vec![TileState::default(); 63],
                    cells: Vec::new(),
                    next_cell_key: 0,
                },
                topology,
            )
            .is_err()
        );
    }
    use crate::resolution::BoundaryRule;

    #[test]
    fn exhaustion_index_releases_sparse_historical_keys() {
        let mut simulation = ReferenceSimulation::new(1, 1, ReferenceRuleset::default()).unwrap();
        let key = simulation.add_cell(TileIndex(0), 10, 100, 0).unwrap();
        let mut cell = simulation.cell(key).unwrap().clone();
        let mut index = MetabolicExhaustionIndex::default();
        let mut oracle = BTreeMap::new();
        for birth in 0..5000_u64 {
            let key = CellKey((1 << 60) + birth * 512);
            cell.assimilated_energy = 1 + birth % 17;
            index.sync(key, Some(&cell), SimTime(0), 1, 1);
            oracle.insert(
                key,
                metabolic_exhaustion_time(SimTime(0), &cell, 1, 1).unwrap(),
            );
            if birth >= 8 {
                let dead = CellKey((1 << 60) + (birth - 8) * 512);
                index.sync(dead, None, SimTime(0), 1, 1);
                oracle.remove(&dead);
            }
            assert_eq!(index.next_event().unwrap(), oracle.values().copied().min());
            assert_eq!(index.heap.len(), oracle.len());
            assert!(index.positions.page_count() <= 8);
        }
        for key in oracle.keys() {
            index.sync(*key, None, SimTime(0), 1, 1);
        }
        assert!(index.heap.is_empty());
        assert_eq!(index.positions.page_count(), 0);
    }

    #[test]
    fn indexed_cell_store_preserves_canonical_order_across_holes() {
        let mut seed = ReferenceSimulation::new(1, 1, ReferenceRuleset::default()).unwrap();
        let template_key = seed.add_cell(TileIndex(0), 10, 100, 0).unwrap();
        let template = seed.cell(template_key).unwrap().clone();
        let mut cells = CellStore::new();

        cells.insert(CellKey(9), template.clone());
        cells.insert(CellKey(2), template.clone());
        cells.insert(CellKey(5), template.clone());
        assert_eq!(
            cells.keys().copied().collect::<Vec<_>>(),
            vec![CellKey(2), CellKey(5), CellKey(9)]
        );
        assert_eq!(
            cells.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec![CellKey(2), CellKey(5), CellKey(9)]
        );
        assert!(cells.get(&CellKey(5)).is_some());

        let removed = cells.remove(&CellKey(5)).unwrap();
        assert_eq!(removed, template);
        assert!(cells.get(&CellKey(5)).is_none());
        assert_eq!(cells.len(), 2);

        let previous = cells.insert(CellKey(9), removed.clone()).unwrap();
        assert_eq!(previous, removed);
        assert_eq!(cells.len(), 2);
        assert_eq!(
            cells.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec![CellKey(2), CellKey(9)]
        );

        let logically_equal = cells
            .iter()
            .map(|(key, cell)| (*key, cell.clone()))
            .collect();
        assert_eq!(cells, logically_equal);
    }

    #[test]
    fn cold_cell_state_is_copy_on_write_and_pending_actions_are_immutable() {
        let mut simulation = ReferenceSimulation::new(1, 1, ReferenceRuleset::default()).unwrap();
        let actor = simulation.add_cell(TileIndex(0), 10, 100, 0).unwrap();
        simulation
            .commit_action(actor, ActionRequest::Wait)
            .unwrap();
        let mut snapshot = simulation.cell(actor).unwrap().clone();
        let live = simulation.cell(actor).unwrap();
        assert!(Arc::ptr_eq(&snapshot.cold, &live.cold));
        assert!(Arc::ptr_eq(
            snapshot.pending_action.as_ref().unwrap(),
            live.pending_action.as_ref().unwrap(),
        ));

        snapshot.last_outcome = Some(OutcomeStatus::Interrupted);
        snapshot.private_memory = Arc::from([1, 2, 3]);
        let live = simulation.cell(actor).unwrap();
        assert!(!Arc::ptr_eq(&snapshot.cold, &live.cold));
        assert_eq!(live.last_outcome, None);
        assert!(live.private_memory.is_empty());
        assert!(Arc::ptr_eq(
            snapshot.pending_action.as_ref().unwrap(),
            live.pending_action.as_ref().unwrap(),
        ));
    }

    #[test]
    #[ignore = "manual indexed-cell lookup throughput diagnostic"]
    fn benchmark_indexed_cell_lookup() {
        const CELLS: u64 = 100_000;
        const LOOKUPS: u64 = 2_000_000;
        let mut seed = ReferenceSimulation::new(1, 1, ReferenceRuleset::default()).unwrap();
        let template_key = seed.add_cell(TileIndex(0), 10, 100, 0).unwrap();
        let template = seed.cell(template_key).unwrap().clone();
        let mut tree = BTreeMap::new();
        let mut indexed = CellStore::new();
        for key in 0..CELLS {
            tree.insert(CellKey(key), template.clone());
            indexed.insert(CellKey(key), template.clone());
        }

        let tree_started = std::time::Instant::now();
        let mut tree_sum = 0_u64;
        for iteration in 0..LOOKUPS {
            let key = CellKey(iteration.wrapping_mul(104_729) % CELLS);
            tree_sum = tree_sum
                .wrapping_add(std::hint::black_box(tree.get(&key).unwrap()).assimilated_energy);
        }
        let tree_elapsed = tree_started.elapsed();

        let indexed_started = std::time::Instant::now();
        let mut indexed_sum = 0_u64;
        for iteration in 0..LOOKUPS {
            let key = CellKey(iteration.wrapping_mul(104_729) % CELLS);
            indexed_sum = indexed_sum
                .wrapping_add(std::hint::black_box(indexed.get(&key).unwrap()).assimilated_energy);
        }
        let indexed_elapsed = indexed_started.elapsed();
        assert_eq!(tree_sum, indexed_sum);
        println!(
            "{LOOKUPS} lookups across {CELLS} cells: tree {:.3} ms, indexed {:.3} ms, {:.2}x speedup",
            tree_elapsed.as_secs_f64() * 1_000.0,
            indexed_elapsed.as_secs_f64() * 1_000.0,
            tree_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "manual cell-layout size diagnostic"]
    fn benchmark_cell_layout_size() {
        println!(
            "CellState={} bytes, CellColdState={} bytes, PendingAction={} bytes",
            std::mem::size_of::<CellState>(),
            std::mem::size_of::<CellColdState>(),
            std::mem::size_of::<PendingAction>(),
        );
    }

    fn dense_diffusion_oracle(
        tiles: &mut [TileState],
        neighbors: &[Vec<TileIndex>],
        numerator: u64,
        denominator: u64,
    ) {
        let snapshot = tiles
            .iter()
            .map(|tile| (tile.diffuse_energy, tile.diffusion_remainder))
            .collect::<Vec<_>>();
        let mut incoming = vec![0_u64; tiles.len()];
        for (index, &(energy, remainder)) in snapshot.iter().enumerate() {
            let neighbor_count = u64::try_from(neighbors[index].len()).unwrap();
            if neighbor_count == 0 || energy < neighbor_count {
                tiles[index].diffusion_remainder = 0;
                continue;
            }
            let generated = u128::from(energy) * u128::from(numerator) + u128::from(remainder);
            let desired =
                u64::try_from((generated / u128::from(denominator)).min(u128::from(energy)))
                    .unwrap();
            tiles[index].diffusion_remainder =
                u64::try_from(generated % u128::from(denominator)).unwrap();
            let share = desired / neighbor_count;
            let transported = share * neighbor_count;
            tiles[index].diffuse_energy -= transported;
            for neighbor in &neighbors[index] {
                incoming[neighbor.0] += share;
            }
        }
        for (tile, incoming) in tiles.iter_mut().zip(incoming) {
            tile.diffuse_energy += incoming;
            if tile.diffuse_energy == 0 {
                tile.diffusion_remainder = 0;
            }
        }
    }

    fn dense_digestion_oracle(
        cells: &mut CellStore,
        elapsed: u64,
        numerator: u64,
        denominator: u64,
    ) {
        for cell in cells.values_mut() {
            if cell.gut_energy == 0 || numerator == 0 {
                cell.digestion_remainder = 0;
                continue;
            }
            let generated =
                u128::from(elapsed) * u128::from(numerator) + u128::from(cell.digestion_remainder);
            let digested = u64::try_from(
                (generated / u128::from(denominator)).min(u128::from(cell.gut_energy)),
            )
            .unwrap();
            cell.gut_energy -= digested;
            cell.assimilated_energy += digested;
            cell.digestion_remainder = if cell.gut_energy == 0 {
                0
            } else {
                u64::try_from(generated % u128::from(denominator)).unwrap()
            };
        }
    }

    fn dense_metabolism_oracle(
        cells: &mut CellStore,
        tiles: &mut [TileState],
        elapsed: u64,
        numerator: u64,
        denominator: u64,
    ) {
        for cell in cells.values_mut() {
            if numerator == 0 {
                cell.metabolism_remainder = 0;
                continue;
            }
            let generated =
                u128::from(elapsed) * u128::from(numerator) + u128::from(cell.metabolism_remainder);
            let spent = u64::try_from(
                (generated / u128::from(denominator)).min(u128::from(cell.assimilated_energy)),
            )
            .unwrap();
            cell.assimilated_energy -= spent;
            cell.metabolism_remainder = if cell.assimilated_energy == 0 {
                0
            } else {
                u64::try_from(generated % u128::from(denominator)).unwrap()
            };
            tiles[cell.position.0].diffuse_energy += spent;
        }
    }

    fn assert_cell_passive_indexes(simulation: &ReferenceSimulation) {
        let digestion = simulation
            .cells
            .iter()
            .filter_map(|(key, cell)| (cell.gut_energy > 0).then_some(*key))
            .collect::<BTreeSet<_>>();
        let metabolism = simulation
            .cells
            .iter()
            .filter_map(|(key, cell)| (cell.assimilated_energy > 0).then_some(*key))
            .collect::<BTreeSet<_>>();
        let zero = simulation
            .cells
            .iter()
            .filter_map(|(key, cell)| (cell.assimilated_energy == 0).then_some(*key))
            .collect::<BTreeSet<_>>();
        assert_eq!(simulation.active_digestion_cells, digestion);
        assert_eq!(simulation.active_metabolism_cells, metabolism);
        assert_eq!(simulation.zero_energy_cells, zero);

        let mut exhaustion_events = BTreeSet::new();
        let mut exhaustion_overflows = BTreeMap::new();
        for (key, cell) in &simulation.cells {
            if simulation.rules.metabolism_rate_numerator == 0 || cell.assimilated_energy == 0 {
                continue;
            }
            match metabolic_exhaustion_time(
                simulation.now,
                cell,
                simulation.rules.metabolism_rate_numerator,
                simulation.rules.metabolism_rate_denominator,
            ) {
                Ok(time) => {
                    exhaustion_events.insert((time, *key));
                }
                Err(error) => {
                    exhaustion_overflows.insert(*key, error);
                }
            }
        }
        assert_eq!(
            simulation
                .metabolic_exhaustion
                .heap
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            exhaustion_events
        );
        assert_eq!(
            simulation.metabolic_exhaustion.overflows,
            exhaustion_overflows
        );
    }

    fn uniform_rules() -> ReferenceRuleset {
        let uniform = DurationRule::new(1024, 0, 1);
        ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
            wait_duration: uniform,
            move_duration: uniform,
            attack_duration: uniform,
            split_duration: uniform,
            metabolism_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        }
    }

    fn former_single_decision_oracle(
        simulation: &mut ReferenceSimulation,
        decision: &DecisionCommitment,
    ) -> Result<CommitReceipt, CommitError> {
        if let ReferenceMemoryUpdate::Replace(bytes) = &decision.memory_update {
            if bytes.len() > simulation.rules.max_private_memory_bytes {
                return Err(CommitError::PrivateMemoryTooLarge {
                    actual: bytes.len(),
                    limit: simulation.rules.max_private_memory_bytes,
                });
            }
        }
        let private_memory_changed = match &decision.memory_update {
            ReferenceMemoryUpdate::Retain => false,
            ReferenceMemoryUpdate::Replace(bytes) => simulation
                .cells
                .get(&decision.actor)
                .is_some_and(|cell| cell.private_memory.as_ref() != bytes.as_slice()),
        };
        if let Some(signal) = decision.signal {
            let (amounts, total) =
                simulation.validate_signal(decision.actor, &decision.request, Some(signal))?;
            let position = simulation.cells[&decision.actor].position;
            let cache = simulation.state_hash_cache.get_mut();
            cache.mark_cell(decision.actor);
            cache.mark_tile(position);
            let cell = simulation.cells.get_mut(&decision.actor).unwrap();
            cell.assimilated_energy -= total;
            for (channel, amount) in amounts.into_iter().enumerate() {
                simulation.tiles[position.0].signal_energy[channel] += amount;
            }
            simulation.mark_signal_tile_active(position);
            simulation.sync_cell_passive_indexes(decision.actor);
        }
        let receipt = simulation.commit_action(decision.actor, decision.request.clone())?;
        if private_memory_changed {
            let ReferenceMemoryUpdate::Replace(bytes) = &decision.memory_update else {
                unreachable!("retained private memory cannot be changed")
            };
            simulation
                .cells
                .get_mut(&decision.actor)
                .unwrap()
                .private_memory = bytes.clone().into();
        }
        if private_memory_changed {
            simulation
                .state_hash_cache
                .get_mut()
                .mark_cell_memory(decision.actor);
        }
        Ok(receipt)
    }

    #[test]
    fn ordered_bulk_commit_matches_former_single_decision_oracle() {
        let mut bulk = ReferenceSimulation::new(5, 1, uniform_rules()).unwrap();
        let actors = (0..4)
            .map(|x| {
                bulk.add_cell(bulk.tile(x, 0).unwrap(), 10, 100, x as u32)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut oracle = ReferenceSimulation::from_canonical_state(
            5,
            1,
            uniform_rules(),
            bulk.canonical_state(),
        )
        .unwrap();
        let decisions = vec![
            DecisionCommitment {
                actor: actors[0],
                request: ActionRequest::Wait,
                signal: None,
                memory_update: ReferenceMemoryUpdate::Retain,
            },
            DecisionCommitment {
                actor: actors[1],
                request: ActionRequest::Guard {
                    effort: EffortTier::Standard,
                },
                signal: Some(ReferenceSignalEmission {
                    channel: 1,
                    amount: 1,
                }),
                memory_update: ReferenceMemoryUpdate::Replace(vec![1, 2, 3]),
            },
            DecisionCommitment {
                actor: actors[2],
                request: ActionRequest::Attack {
                    target: LocalSlot(4),
                    effort: EffortTier::Low,
                    payload: 7,
                },
                signal: None,
                memory_update: ReferenceMemoryUpdate::Replace(vec![9]),
            },
            DecisionCommitment {
                actor: actors[3],
                request: ActionRequest::Move {
                    target: LocalSlot(31),
                    effort: EffortTier::Standard,
                },
                signal: None,
                memory_update: ReferenceMemoryUpdate::Replace(vec![4, 5]),
            },
        ];

        let bulk_receipts = bulk.commit_decisions_ordered(&decisions).unwrap();
        let oracle_receipts = decisions
            .iter()
            .map(|decision| former_single_decision_oracle(&mut oracle, decision))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(bulk_receipts, oracle_receipts);
        assert_eq!(bulk.canonical_state(), oracle.canonical_state());
        assert_eq!(bulk.due_actions, oracle.due_actions);
        assert_eq!(bulk.active_signal_tiles, oracle.active_signal_tiles);
        assert_eq!(bulk.active_diffuse_tiles, oracle.active_diffuse_tiles);
        assert_eq!(bulk.state_hash(), oracle.state_hash());
        assert_eq!(
            bulk.resolve_next_batch().unwrap(),
            oracle.resolve_next_batch().unwrap()
        );
    }

    #[test]
    fn ordered_bulk_commit_preflight_is_atomic() {
        let mut simulation = ReferenceSimulation::new(2, 1, uniform_rules()).unwrap();
        let first = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
            .unwrap();
        let second = simulation
            .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 0)
            .unwrap();
        let before = simulation.canonical_state();
        let before_hash = simulation.state_hash();
        let decisions =
            vec![
                DecisionCommitment {
                    actor: first,
                    request: ActionRequest::Wait,
                    signal: None,
                    memory_update: ReferenceMemoryUpdate::Replace(vec![1]),
                },
                DecisionCommitment {
                    actor: second,
                    request: ActionRequest::Wait,
                    signal: None,
                    memory_update: ReferenceMemoryUpdate::Replace(vec![
                        0;
                        simulation.rules.max_private_memory_bytes
                            + 1
                    ]),
                },
            ];

        assert!(matches!(
            simulation.commit_decisions_ordered(&decisions),
            Err(CommitError::PrivateMemoryTooLarge { .. })
        ));
        assert_eq!(simulation.canonical_state(), before);
        assert_eq!(simulation.state_hash(), before_hash);
        assert!(simulation.due_actions.is_empty());

        let unordered = [decisions[1].clone(), decisions[0].clone()];
        assert!(matches!(
            simulation.commit_decisions_ordered(&unordered),
            Err(CommitError::ActorsNotStrictlyOrdered { .. })
        ));
        assert_eq!(simulation.canonical_state(), before);
    }

    #[test]
    fn unchanged_private_memory_keeps_its_canonical_allocation() {
        let mut seeded = ReferenceSimulation::new(1, 1, uniform_rules()).unwrap();
        let actor = seeded
            .add_cell(seeded.tile(0, 0).unwrap(), 10, 100, 0)
            .unwrap();
        let mut state = seeded.canonical_state();
        state.cells[0].1.private_memory = vec![7; 2_048].into();
        let mut simulation =
            ReferenceSimulation::from_canonical_state(1, 1, uniform_rules(), state).unwrap();
        let before = Arc::clone(&simulation.cells[&actor].private_memory);

        simulation
            .commit_decisions_ordered(&[DecisionCommitment {
                actor,
                request: ActionRequest::Wait,
                signal: None,
                memory_update: ReferenceMemoryUpdate::Retain,
            }])
            .unwrap();

        assert!(Arc::ptr_eq(
            &before,
            &simulation.cells[&actor].private_memory
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn mixed_bulk_fixture(count: usize) -> (ReferenceSimulation, Vec<DecisionCommitment>) {
        let mut rules = uniform_rules();
        rules.max_private_memory_bytes = 16;
        let mut simulation = ReferenceSimulation::new(64, 33, rules.clone()).unwrap();
        for tile in 0..count {
            simulation
                .add_cell(TileIndex(tile), 10, 100, tile as u32)
                .unwrap();
        }
        let mut state = simulation.canonical_state();
        for (_, cell) in &mut state.cells {
            cell.gut_energy = 32;
            cell.carried_material_mass = 20;
        }
        let simulation = ReferenceSimulation::from_canonical_state(64, 33, rules, state).unwrap();
        let decisions = simulation
            .cells
            .keys()
            .copied()
            .enumerate()
            .map(|(index, actor)| {
                let target = LocalSlot(if index % 20 == 1 { 31 } else { 4 });
                let effort = [EffortTier::Low, EffortTier::Standard, EffortTier::High][index % 3];
                let request = match index % 10 {
                    0 => ActionRequest::Wait,
                    1 => ActionRequest::Move { target, effort },
                    2 => ActionRequest::Attack {
                        target,
                        effort,
                        payload: 7,
                    },
                    3 => ActionRequest::Guard { effort },
                    4 => ActionRequest::Consume { amount: 8 },
                    5 => ActionRequest::Split {
                        target,
                        child_allocation: 20,
                        marker: index as u32,
                        private_memory: vec![3; 16],
                    },
                    6 => ActionRequest::Regurgitate { target, amount: 8 },
                    7 => ActionRequest::Signal {
                        amounts: [0, 0, 0, 1],
                    },
                    8 => ActionRequest::Excavate,
                    _ => ActionRequest::DepositTerrain,
                };
                DecisionCommitment {
                    actor,
                    request,
                    signal: (index % 3 == 0 && index % 10 != 7).then_some(
                        ReferenceSignalEmission {
                            channel: (index % 4) as u8,
                            amount: 1,
                        },
                    ),
                    memory_update: if index % 3 == 0 {
                        ReferenceMemoryUpdate::Retain
                    } else {
                        ReferenceMemoryUpdate::Replace(vec![index as u8; 1 + index % 16])
                    },
                }
            })
            .collect();
        (simulation, decisions)
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn dense_parallel_bulk_preflight_matches_former_oracle() {
        for count in [
            PARALLEL_COMMIT_PREFLIGHT_THRESHOLD - 1,
            PARALLEL_COMMIT_PREFLIGHT_THRESHOLD,
            PARALLEL_COMMIT_PREFLIGHT_THRESHOLD + 1,
        ] {
            let (baseline, decisions) = mixed_bulk_fixture(count);
            let mut oracle = baseline.clone();
            let expected = decisions
                .iter()
                .map(|decision| former_single_decision_oracle(&mut oracle, decision))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert!(expected.iter().any(|receipt| receipt.accepted));
            assert!(expected.iter().any(|receipt| !receipt.accepted));
            for workers in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let mut bulk = baseline.clone();
                assert_eq!(
                    pool.install(|| bulk.commit_decisions_ordered(&decisions))
                        .unwrap(),
                    expected
                );
                assert_eq!(bulk.canonical_state(), oracle.canonical_state());
                assert_eq!(bulk.due_actions, oracle.due_actions);
                assert_eq!(bulk.active_signal_tiles, oracle.active_signal_tiles);
                assert_eq!(bulk.active_diffuse_tiles, oracle.active_diffuse_tiles);
                assert_eq!(bulk.state_hash(), oracle.state_hash());
                let mut oracle_next = oracle.clone();
                assert_eq!(
                    bulk.resolve_next_batch_parallel(workers).unwrap(),
                    oracle_next.resolve_next_batch().unwrap()
                );
                assert_eq!(bulk.canonical_state(), oracle_next.canonical_state());
                assert_eq!(
                    bulk.total_energy_equivalent(),
                    baseline.total_energy_equivalent()
                );
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn dense_parallel_preflight_selects_first_error_without_mutation() {
        let (baseline, decisions) = mixed_bulk_fixture(PARALLEL_COMMIT_PREFLIGHT_THRESHOLD + 1);
        let before = baseline.canonical_state();
        let before_hash = baseline.state_hash();
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            for first in [0, decisions.len() / 2, decisions.len() - 2] {
                let mut invalid = decisions.clone();
                // A second distinct failure must not race ahead of the earlier
                // actor's oversized-memory error during parallel preflight.
                invalid[first].memory_update = ReferenceMemoryUpdate::Replace(vec![1; 17]);
                invalid[first + 1].signal = Some(ReferenceSignalEmission {
                    channel: 4,
                    amount: 1,
                });
                let mut simulation = baseline.clone();
                assert_eq!(
                    pool.install(|| simulation.commit_decisions_ordered(&invalid)),
                    Err(CommitError::PrivateMemoryTooLarge {
                        actual: 17,
                        limit: 16
                    })
                );
                assert_eq!(simulation.canonical_state(), before);
                assert_eq!(simulation.state_hash(), before_hash);
                assert_eq!(simulation.due_actions, baseline.due_actions);
                assert_eq!(simulation.active_signal_tiles, baseline.active_signal_tiles);
                assert_eq!(
                    simulation.active_diffuse_tiles,
                    baseline.active_diffuse_tiles
                );
                // Continue after rejection to catch derived-state residue too.
                let mut clean = baseline.clone();
                assert_eq!(
                    pool.install(|| simulation.commit_decisions_ordered(&decisions))
                        .unwrap(),
                    clean.commit_decisions_ordered(&decisions).unwrap()
                );
                assert_eq!(
                    simulation.resolve_next_batch_parallel(workers).unwrap(),
                    clean.resolve_next_batch().unwrap()
                );
                assert_eq!(simulation.canonical_state(), clean.canonical_state());
            }
        }
    }

    #[test]
    fn rejected_action_becomes_minimum_wait() {
        let mut simulation = ReferenceSimulation::new(3, 3, uniform_rules()).unwrap();
        let actor = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 0)
            .unwrap();
        let receipt = simulation
            .commit_action(
                actor,
                ActionRequest::Move {
                    target: LocalSlot(3),
                    effort: EffortTier::Standard,
                },
            )
            .unwrap();

        assert!(!receipt.accepted);
        assert_eq!(receipt.rejection, Some(RejectReason::TargetOutsideWorld));
        let report = simulation.resolve_next_batch().unwrap();
        assert_eq!(
            report.outcomes[0].status,
            OutcomeStatus::Rejected(RejectReason::TargetOutsideWorld)
        );
    }

    #[test]
    fn guard_is_immediate_and_telegraphed_attack_is_local() {
        let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
        let attacker = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
            .unwrap();
        let defender = simulation
            .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 2)
            .unwrap();
        simulation
            .commit_action(
                attacker,
                ActionRequest::Attack {
                    target: LocalSlot(4),
                    effort: EffortTier::Standard,
                    payload: 20,
                },
            )
            .unwrap();
        simulation
            .commit_action(
                defender,
                ActionRequest::Guard {
                    effort: EffortTier::Standard,
                },
            )
            .unwrap();

        assert!(simulation.cell(defender).unwrap().guarded);
        let defender_cues = simulation.neighbor_cues(defender).unwrap();
        assert_eq!(
            defender_cues[3].unwrap().activity,
            Some(ActivityCue::AttackWindup)
        );
        let before = simulation.cell(defender).unwrap().assimilated_energy;
        simulation.resolve_next_batch().unwrap();
        let after = simulation.cell(defender).unwrap().assimilated_energy;
        assert_eq!(before - after, 10);
    }

    #[test]
    fn effort_and_payload_conserve_energy_equivalent() {
        let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
        let attacker = simulation
            .add_cell(simulation.tile(0, 0).unwrap(), 10, 100, 1)
            .unwrap();
        let defender = simulation
            .add_cell(simulation.tile(1, 0).unwrap(), 10, 100, 2)
            .unwrap();
        let before = simulation.total_energy_equivalent();
        simulation
            .commit_action(
                attacker,
                ActionRequest::Attack {
                    target: LocalSlot(4),
                    effort: EffortTier::Standard,
                    payload: 20,
                },
            )
            .unwrap();
        simulation.resolve_next_batch().unwrap();

        assert_eq!(simulation.total_energy_equivalent(), before);
        assert!(simulation.cell(defender).is_some());
    }

    #[test]
    fn private_memory_is_copy_on_write_inside_state_and_owned_at_mind_boundary() {
        let rules = uniform_rules();
        let mut seeded = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
        let actor = seeded.add_cell(TileIndex(0), 10, 100, 0).unwrap();
        let mut state = seeded.canonical_state();
        state.cells[0].1.private_memory = vec![1, 2, 3, 4].into();
        let mut simulation = ReferenceSimulation::from_canonical_state(1, 1, rules, state).unwrap();

        let snapshot = simulation.canonical_state();
        assert!(Arc::ptr_eq(
            &simulation.cell(actor).unwrap().private_memory,
            &snapshot.cells[0].1.private_memory,
        ));

        let mut input = simulation
            .reference_mind_input(actor, PrivateRandom::ZERO)
            .unwrap();
        input.private_memory[0] = 99;
        assert_eq!(
            simulation.cell(actor).unwrap().private_memory.as_ref(),
            [1, 2, 3, 4]
        );

        simulation
            .commit_decision(actor, ActionRequest::Wait, vec![5, 6, 7, 8])
            .unwrap();
        assert_eq!(
            simulation.cell(actor).unwrap().private_memory.as_ref(),
            [5, 6, 7, 8]
        );
        assert_eq!(snapshot.cells[0].1.private_memory.as_ref(), [1, 2, 3, 4]);
        assert!(!Arc::ptr_eq(
            &simulation.cell(actor).unwrap().private_memory,
            &snapshot.cells[0].1.private_memory,
        ));
    }

    #[test]
    fn canonical_restore_enforces_private_memory_limit() {
        let rules = uniform_rules();
        let mut simulation = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
        simulation.add_cell(TileIndex(0), 10, 100, 0).unwrap();
        let mut state = simulation.canonical_state();
        state.cells[0].1.private_memory =
            vec![0; rules.max_private_memory_bytes.saturating_add(1)].into();

        assert!(matches!(
            ReferenceSimulation::from_canonical_state(1, 1, rules, state),
            Err(ResolutionError::InvalidState(
                "cell private memory exceeds the ruleset limit"
            ))
        ));
    }

    #[test]
    fn failed_resolution_rolls_back_every_journaled_write() {
        let mut simulation = ReferenceSimulation::new(3, 1, uniform_rules()).unwrap();
        let attacker_tile = simulation.tile(0, 0).unwrap();
        let target_tile = simulation.tile(1, 0).unwrap();
        let attacker = simulation.add_cell(attacker_tile, 10, 100, 1).unwrap();
        simulation.add_cell(target_tile, 10, 100, 2).unwrap();
        simulation.tile_state_mut(target_tile).unwrap().loose_energy = u64::MAX;
        simulation
            .commit_action(
                attacker,
                ActionRequest::Attack {
                    target: LocalSlot(4),
                    effort: EffortTier::Standard,
                    payload: 1,
                },
            )
            .unwrap();
        let mut expected = simulation.canonical_state();
        // Passive time advancement is a separate canonical phase and remains
        // committed; resolution itself must roll back to that completion
        // snapshot so the same due action can be retried.
        expected.now = SimTime(1024);
        let expected_hash = expected.hash_with_compiled_ruleset(simulation.compiled_ruleset_hash());

        let error = simulation.resolve_next_batch().unwrap_err();

        assert!(matches!(error, ResolutionError::ArithmeticOverflow(_)));
        assert_eq!(simulation.canonical_state(), expected);
        assert_eq!(simulation.state_hash(), expected_hash);
        assert_eq!(simulation.next_completion_time(), Some(SimTime(1024)));
        assert_cell_passive_indexes(&simulation);
    }

    #[test]
    fn sparse_diffusion_matches_dense_oracle_and_rebuilds_from_canonical_state() {
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
            metabolism_rate_numerator: 0,
            signal_decay_rate_numerator: 0,
            diffusion_rate_numerator: 3,
            diffusion_rate_denominator: 11,
            ..ReferenceRuleset::default()
        };
        let mut seeded = ReferenceSimulation::new(31, 29, rules.clone()).unwrap();
        for index in [0, 17, 113, 447, 898] {
            let tile = seeded.tile_state_mut(TileIndex(index)).unwrap();
            tile.diffuse_energy = 97 + u64::try_from(index).unwrap();
            tile.diffusion_remainder = u64::try_from(index).unwrap() % 11;
        }

        let state = seeded.canonical_state();
        let mut expected_tiles = state.tiles.clone();
        let mut sparse = ReferenceSimulation::from_canonical_state(31, 29, rules, state).unwrap();
        let neighbors = sparse.diffusion_neighbors.clone();

        for _ in 0..12 {
            sparse.apply_diffusion_step().unwrap();
            dense_diffusion_oracle(&mut expected_tiles, &neighbors, 3, 11);
            assert_eq!(sparse.tiles.to_vec(), expected_tiles);

            let expected_frontier = expected_tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| (tile.diffuse_energy > 0).then_some(TileIndex(index)))
                .collect::<Vec<_>>();
            assert_eq!(
                sparse
                    .active_diffuse_tiles
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                expected_frontier
            );
            assert!(sparse.diffusion_touched.is_empty());
            assert!(sparse
                .diffusion_incoming
                .iter()
                .all(|incoming| *incoming == 0));
        }
    }

    #[test]
    fn sparse_cell_passive_updates_match_dense_oracle_and_checkpoint_indexes() {
        let rules = ReferenceRuleset {
            digestion_rate_numerator: 3,
            digestion_rate_denominator: 11,
            metabolism_rate_numerator: 2,
            metabolism_rate_denominator: 13,
            signal_decay_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut seeded = ReferenceSimulation::new(32, 8, rules.clone()).unwrap();
        for index in 0..192 {
            seeded
                .add_cell(
                    TileIndex(index),
                    10,
                    20 + u64::try_from(index % 9).unwrap(),
                    0,
                )
                .unwrap();
        }
        let mut state = seeded.canonical_state();
        for (index, (_, cell)) in state.cells.iter_mut().enumerate() {
            if index % 29 == 0 {
                cell.gut_energy = 50 + u64::try_from(index).unwrap();
                cell.digestion_remainder = u64::try_from(index).unwrap() % 11;
            }
            if index == 58 {
                cell.assimilated_energy = 0;
                cell.metabolism_remainder = 0;
            }
        }
        let mut sparse = ReferenceSimulation::from_canonical_state(32, 8, rules, state).unwrap();
        let mut dense_cells = sparse.cells.clone();
        let mut dense_tiles = sparse.tiles.to_vec();
        assert_cell_passive_indexes(&sparse);

        for elapsed in [1, 7, 13, 29, 41] {
            sparse.apply_digestion(elapsed).unwrap();
            sparse.apply_metabolism(elapsed).unwrap();
            sparse.now = SimTime(sparse.now.0 + elapsed);
            dense_digestion_oracle(&mut dense_cells, elapsed, 3, 11);
            dense_metabolism_oracle(&mut dense_cells, &mut dense_tiles, elapsed, 2, 13);
            assert_eq!(sparse.cells, dense_cells);
            assert_eq!(sparse.tiles.to_vec(), dense_tiles);
            assert_cell_passive_indexes(&sparse);
        }
    }

    #[test]
    fn metabolic_exhaustion_index_tracks_mutation_accrual_restore_and_overflow() {
        let rules = ReferenceRuleset {
            metabolism_rate_numerator: 3,
            metabolism_rate_denominator: 10,
            diffusion_rate_numerator: 0,
            ..uniform_rules()
        };
        let mut simulation = ReferenceSimulation::new(2, 1, rules.clone()).unwrap();
        let first = simulation.add_cell(TileIndex(0), 1, 30, 0).unwrap();
        let second = simulation.add_cell(TileIndex(1), 1, 6, 0).unwrap();
        assert_eq!(
            simulation.next_metabolic_event_time().unwrap(),
            Some(SimTime(20))
        );

        simulation.cells.get_mut(&first).unwrap().assimilated_energy = 1;
        simulation.sync_cell_passive_indexes(first);
        assert_eq!(
            simulation.next_metabolic_event_time().unwrap(),
            Some(SimTime(4))
        );

        simulation.apply_metabolism(2).unwrap();
        simulation.now = SimTime(2);
        assert_eq!(
            simulation.next_metabolic_event_time().unwrap(),
            Some(SimTime(4))
        );
        assert_cell_passive_indexes(&simulation);

        simulation.cells.get_mut(&first).unwrap().assimilated_energy = 0;
        simulation
            .cells
            .get_mut(&first)
            .unwrap()
            .metabolism_remainder = 0;
        simulation.sync_cell_passive_indexes(first);
        assert_eq!(
            simulation.next_metabolic_event_time().unwrap(),
            Some(SimTime(20))
        );
        simulation.cells.get_mut(&first).unwrap().ready_at = simulation.now;
        simulation.cells.get_mut(&second).unwrap().ready_at = simulation.now;

        let restored =
            ReferenceSimulation::from_canonical_state(2, 1, rules, simulation.canonical_state())
                .unwrap();
        assert_eq!(restored.canonical_state(), simulation.canonical_state());
        assert_eq!(
            restored.next_metabolic_event_time().unwrap(),
            simulation.next_metabolic_event_time().unwrap()
        );
        assert_cell_passive_indexes(&restored);

        let overflow_rules = ReferenceRuleset {
            metabolism_rate_numerator: 1,
            metabolism_rate_denominator: u64::MAX,
            diffusion_rate_numerator: 0,
            ..uniform_rules()
        };
        let mut overflow = ReferenceSimulation::new(1, 1, overflow_rules.clone()).unwrap();
        overflow.add_cell(TileIndex(0), 1, u64::MAX, 0).unwrap();
        assert!(matches!(
            overflow.next_metabolic_event_time(),
            Err(ResolutionError::ArithmeticOverflow(
                "converting metabolic exhaustion time"
            ))
        ));
        let restored_overflow = ReferenceSimulation::from_canonical_state(
            1,
            1,
            overflow_rules,
            overflow.canonical_state(),
        )
        .unwrap();
        assert_eq!(
            restored_overflow.next_metabolic_event_time(),
            overflow.next_metabolic_event_time()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn plant_growth_divisor_matches_exact_integer_arithmetic() {
        for denominator in [1, 2, 3, 4, 1_024, 1_025, u64::from(u32::MAX), u64::MAX] {
            let divisor = PassiveRateDivisor::new(denominator);
            for numerator in [
                0,
                1,
                u128::from(denominator.saturating_sub(1)),
                u128::from(denominator),
                u128::from(denominator).saturating_add(1),
                u128::from(u64::MAX) * u128::from(u64::MAX),
                u128::MAX,
            ] {
                assert_eq!(
                    divisor.quotient(numerator),
                    numerator / u128::from(denominator)
                );
                assert_eq!(
                    divisor.remainder(numerator),
                    numerator % u128::from(denominator)
                );
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_plant_growth_removes_only_exhausted_diffuse_tiles() {
        const TILES: usize = 4_096;
        let rules = ReferenceRuleset {
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut state = ReferenceSimulation::new(TILES, 1, rules.clone())
            .unwrap()
            .canonical_state();
        for (index, tile) in state.tiles.iter_mut().enumerate() {
            tile.diffuse_energy = if index.is_multiple_of(5) { 1 } else { 10 };
            tile.plant_capacity = 100;
            tile.plant_growth_rate = 1_024;
        }
        let mut parallel =
            ReferenceSimulation::from_canonical_state(TILES, 1, rules, state).unwrap();
        parallel.set_passive_parallel_thresholds(Some(2), Some(2));
        let worker_seed = parallel.clone();
        let mut oracle_tiles = parallel.tiles.to_vec();
        for tile in &mut oracle_tiles {
            advance_plant_growth(tile, 1, 1_024).unwrap();
        }
        let expected_frontier = oracle_tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| (tile.diffuse_energy > 0).then_some(TileIndex(index)))
            .collect::<BTreeSet<_>>();

        parallel.apply_plant_growth(1).unwrap();

        assert_eq!(parallel.tiles.to_vec(), oracle_tiles);
        assert_eq!(parallel.active_diffuse_tiles, expected_frontier);
        assert_eq!(
            parallel.active_diffuse_tiles.len(),
            TILES - TILES.div_ceil(5)
        );
        for workers in [1, 2, 4] {
            let mut candidate = worker_seed.clone();
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| candidate.apply_plant_growth(1).unwrap());
            assert_eq!(candidate.tiles, parallel.tiles);
            assert_eq!(candidate.active_diffuse_tiles, expected_frontier);
        }
    }

    #[test]
    fn sparse_plant_growth_updates_a_dirty_diffuse_frontier_exactly() {
        const TILES: usize = 64;
        let rules = ReferenceRuleset {
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(TILES, 1, rules).unwrap();
        for index in 0..TILES {
            let tile = simulation.tile_state_mut(TileIndex(index)).unwrap();
            tile.diffuse_energy = 10;
        }
        for index in [3, 17, 41] {
            let tile = simulation.tile_state_mut(TileIndex(index)).unwrap();
            tile.diffuse_energy = if index == 17 { 2 } else { 1 };
            tile.plant_capacity = 100;
            tile.plant_growth_rate = 1_024;
        }

        simulation.apply_plant_growth(1).unwrap();

        assert_eq!(simulation.tiles[3].diffuse_energy, 0);
        assert_eq!(simulation.tiles[17].diffuse_energy, 1);
        assert_eq!(simulation.tiles[41].diffuse_energy, 0);
        assert!(!simulation.active_diffuse_tiles.contains(&TileIndex(3)));
        assert!(simulation.active_diffuse_tiles.contains(&TileIndex(17)));
        assert!(!simulation.active_diffuse_tiles.contains(&TileIndex(41)));
        assert_eq!(simulation.active_diffuse_tiles.len(), TILES - 2);
        assert!(!simulation.field_frontiers_dirty);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_parallel_passive_kernels_match_serial_oracles() {
        // Full occupancy exercises the gathered dense-metabolism deposit and
        // exact row-major diffuse-frontier rebuild.
        const BOARD: usize = 64;
        const CELLS: usize = 4_096;
        const ELAPSED: u64 = 17;
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
            digestion_rate_numerator: 3,
            digestion_rate_denominator: 11,
            metabolism_rate_numerator: 2,
            metabolism_rate_denominator: 13,
            signal_decay_rate_numerator: 5,
            signal_decay_rate_denominator: 17,
            diffusion_rate_numerator: 3,
            diffusion_rate_denominator: 11,
            ..ReferenceRuleset::default()
        };
        let mut seeded = ReferenceSimulation::new(BOARD, BOARD, rules.clone()).unwrap();
        for index in 0..CELLS {
            seeded.add_cell(TileIndex(index), 10, 100, 0).unwrap();
        }
        let mut state = seeded.canonical_state();
        for (_, cell) in state.cells.iter_mut() {
            cell.gut_energy = 50;
        }
        for (index, tile) in state.tiles.iter_mut().enumerate() {
            tile.diffuse_energy = 1_000 + u64::try_from(index % 13).unwrap();
            tile.plant_capacity = 2_000;
            tile.plant_energy = 100;
            tile.plant_growth_rate = 3;
            tile.signal_energy = [31, 17, 0, 9];
        }
        let mut parallel =
            ReferenceSimulation::from_canonical_state(BOARD, BOARD, rules, state).unwrap();
        parallel.set_passive_parallel_thresholds(Some(2), Some(2));
        let worker_seed = parallel.clone();
        let mut serial_cells = parallel.cells.clone();
        let mut serial_tiles = parallel.tiles.to_vec();
        let diffusion_neighbors = parallel.diffusion_neighbors.clone();

        parallel.apply_plant_growth(ELAPSED).unwrap();
        for tile in &mut serial_tiles {
            advance_plant_growth(tile, ELAPSED, 1024).unwrap();
        }
        parallel.apply_digestion(ELAPSED).unwrap();
        dense_digestion_oracle(&mut serial_cells, ELAPSED, 3, 11);
        parallel.apply_metabolism(ELAPSED).unwrap();
        dense_metabolism_oracle(&mut serial_cells, &mut serial_tiles, ELAPSED, 2, 13);
        let parallel_signal_decay = parallel.apply_signal_decay(ELAPSED).unwrap();
        let mut serial_signal_decay = [0_u128; REFERENCE_SIGNAL_CHANNELS];
        for tile in &mut serial_tiles {
            let decayed = advance_signal_decay(tile, ELAPSED, 5, 17).unwrap();
            for channel in 0..REFERENCE_SIGNAL_CHANNELS {
                serial_signal_decay[channel] =
                    serial_signal_decay[channel].saturating_add(u128::from(decayed[channel]));
            }
        }
        assert_eq!(parallel_signal_decay, serial_signal_decay);
        parallel.apply_diffusion_step().unwrap();
        dense_diffusion_oracle(&mut serial_tiles, &diffusion_neighbors, 3, 11);
        parallel.now = SimTime(ELAPSED);

        assert_eq!(parallel.cells, serial_cells);
        assert_eq!(parallel.tiles.to_vec(), serial_tiles);
        assert_cell_passive_indexes(&parallel);
        assert_eq!(
            parallel.active_diffuse_tiles,
            parallel
                .tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    (tile.diffuse_energy > 0).then_some(TileIndex(index))
                })
                .collect()
        );
        for workers in [1, 2, 4] {
            let mut candidate = worker_seed.clone();
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    candidate.apply_plant_growth(ELAPSED).unwrap();
                    candidate.apply_digestion(ELAPSED).unwrap();
                    candidate.apply_metabolism(ELAPSED).unwrap();
                    candidate.apply_signal_decay(ELAPSED).unwrap();
                    candidate.apply_diffusion_step().unwrap();
                });
            candidate.now = SimTime(ELAPSED);
            assert_eq!(candidate.cells, parallel.cells);
            assert_eq!(candidate.tiles, parallel.tiles);
            assert_eq!(
                candidate.active_diffuse_tiles,
                parallel.active_diffuse_tiles
            );
            assert_eq!(candidate.active_signal_tiles, parallel.active_signal_tiles);
            assert_eq!(
                candidate.metabolic_exhaustion.next_event().unwrap(),
                parallel.metabolic_exhaustion.next_event().unwrap()
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_metabolism_preserves_or_rebuilds_exact_frontiers_and_exhaustion() {
        const TILES: usize = 2_051;
        let rules = ReferenceRuleset {
            metabolism_rate_numerator: 1,
            metabolism_rate_denominator: 1,
            signal_decay_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut seeded = ReferenceSimulation::new(TILES, 1, rules.clone()).unwrap();
        for index in 0..TILES {
            seeded
                .add_cell(TileIndex(index), 1, if index % 7 == 0 { 1 } else { 10 }, 0)
                .unwrap();
        }
        let mut state = seeded.canonical_state();
        for (index, tile) in state.tiles.iter_mut().enumerate() {
            tile.diffuse_energy = if index % 5 == 0 { 0 } else { 100 };
        }
        let mut parallel =
            ReferenceSimulation::from_canonical_state(TILES, 1, rules.clone(), state).unwrap();
        parallel.set_passive_parallel_thresholds(Some(2), Some(2));
        let worker_seed = parallel.clone();
        let mut serial = parallel.clone();
        serial.set_passive_parallel_thresholds(None, None);

        parallel.apply_metabolism(1).unwrap();
        serial.apply_metabolism(1).unwrap();
        parallel.now = SimTime(1);
        serial.now = SimTime(1);
        assert_eq!(parallel.cells, serial.cells);
        assert_eq!(parallel.tiles, serial.tiles);
        assert_eq!(parallel.active_diffuse_tiles, serial.active_diffuse_tiles);
        assert_eq!(
            parallel.active_metabolism_cells,
            serial.active_metabolism_cells
        );
        assert_eq!(parallel.zero_energy_cells, serial.zero_energy_cells);
        assert_eq!(
            parallel.metabolic_exhaustion.next_event().unwrap(),
            serial.metabolic_exhaustion.next_event().unwrap()
        );
        assert_eq!(
            parallel.active_diffuse_tiles,
            parallel
                .tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    (tile.diffuse_energy > 0).then_some(TileIndex(index))
                })
                .collect()
        );
        assert!(!parallel.zero_energy_cells.is_empty());
        assert_cell_passive_indexes(&parallel);
        assert!(parallel
            .dense_metabolism_deposits
            .0
            .iter()
            .all(|deposit| deposit.load(Ordering::Relaxed) == 0));
        assert!(parallel.clone().dense_metabolism_deposits.0.is_empty());
        for workers in [1, 2, 4] {
            let mut candidate = worker_seed.clone();
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| candidate.apply_metabolism(1).unwrap());
            candidate.now = SimTime(1);
            assert_eq!(candidate.cells, parallel.cells);
            assert_eq!(candidate.tiles, parallel.tiles);
            assert_eq!(
                candidate.active_diffuse_tiles,
                parallel.active_diffuse_tiles
            );
            assert_eq!(
                candidate.active_metabolism_cells,
                parallel.active_metabolism_cells
            );
            assert_eq!(candidate.zero_energy_cells, parallel.zero_energy_cells);
            assert_eq!(
                candidate.metabolic_exhaustion.next_event().unwrap(),
                parallel.metabolic_exhaustion.next_event().unwrap()
            );
            assert!(candidate
                .dense_metabolism_deposits
                .0
                .iter()
                .all(|deposit| deposit.load(Ordering::Relaxed) == 0));
        }

        // When every destination is already active, the exact frontier is
        // unchanged even though every cell still emits metabolic energy.
        let mut seeded = ReferenceSimulation::new(TILES, 1, rules.clone()).unwrap();
        for index in 0..TILES {
            seeded.add_cell(TileIndex(index), 1, 10, 0).unwrap();
            seeded.tiles[index].diffuse_energy = 100;
        }
        seeded.rebuild_field_frontiers();
        seeded.set_passive_parallel_thresholds(Some(2), Some(2));
        let frontier = seeded.active_diffuse_tiles.clone();
        seeded.apply_metabolism(1).unwrap();
        assert_eq!(seeded.active_diffuse_tiles, frontier);

        // A removed historical key exercises the canonical active-key
        // preflight while retaining the same gathered deposit semantics.
        let mut state = seeded.canonical_state();
        state.cells.remove(1_024);
        state.tiles[1_024].occupant = None;
        let mut holes = ReferenceSimulation::from_canonical_state(TILES, 1, rules, state).unwrap();
        holes.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut holes_serial = holes.clone();
        holes_serial.set_passive_parallel_thresholds(None, None);
        holes.apply_metabolism(1).unwrap();
        holes_serial.apply_metabolism(1).unwrap();
        holes.now = SimTime(1);
        holes_serial.now = SimTime(1);
        assert_eq!(holes.cells, holes_serial.cells);
        assert_eq!(holes.tiles, holes_serial.tiles);
        assert_eq!(
            holes.active_diffuse_tiles,
            holes_serial.active_diffuse_tiles
        );
        assert_cell_passive_indexes(&holes);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_signal_page_bitmaps_cover_partial_pages_and_sparse_fallback() {
        let rules = ReferenceRuleset {
            signal_decay_rate_numerator: 5,
            signal_decay_rate_denominator: 17,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut partial = ReferenceSimulation::new(2_051, 1, rules.clone()).unwrap();
        for (index, tile) in partial.tiles.iter_mut().enumerate() {
            tile.diffuse_energy = if index % 7 == 0 { 0 } else { 100 };
            tile.signal_energy = if index % 11 == 0 {
                [3, 0, 0, 0]
            } else {
                [31, 17, 0, 9]
            };
            tile.signal_decay_remainder = [
                u64::try_from(index % 17).unwrap(),
                u64::try_from((index + 3) % 17).unwrap(),
                0,
                u64::try_from((index + 7) % 17).unwrap(),
            ];
        }
        partial.rebuild_field_frontiers();
        partial.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut partial_serial = partial.clone();
        partial_serial.set_passive_parallel_thresholds(None, None);

        let parallel_decayed = partial.apply_signal_decay(17).unwrap();
        let serial_decayed = partial_serial.apply_signal_decay(17).unwrap();
        assert_eq!(parallel_decayed, serial_decayed);
        assert_eq!(partial.tiles, partial_serial.tiles);
        assert_eq!(
            partial.active_signal_tiles,
            partial_serial.active_signal_tiles
        );
        assert_eq!(
            partial.active_diffuse_tiles,
            partial_serial.active_diffuse_tiles
        );

        let mut sparse = ReferenceSimulation::new(2_051, 1, rules).unwrap();
        for index in [0, 2_047, 2_048, 2_050] {
            let tile = &mut sparse.tiles[index];
            tile.signal_energy = [31, 17, 0, 9];
            tile.signal_decay_remainder = [1, 3, 0, 7];
        }
        sparse.rebuild_field_frontiers();
        sparse.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut sparse_serial = sparse.clone();
        sparse_serial.set_passive_parallel_thresholds(None, None);

        let parallel_decayed = sparse.apply_signal_decay(17).unwrap();
        let serial_decayed = sparse_serial.apply_signal_decay(17).unwrap();
        assert_eq!(parallel_decayed, serial_decayed);
        assert_eq!(sparse.tiles, sparse_serial.tiles);
        assert_eq!(
            sparse.active_signal_tiles,
            sparse_serial.active_signal_tiles
        );
        assert_eq!(
            sparse.active_diffuse_tiles,
            sparse_serial.active_diffuse_tiles
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn planned_signal_frontier_changes_match_exact_decay_transitions() {
        for diffuse_energy in [0, 1, 100] {
            for signal_energy in [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1], [31, 17, 0, 9]] {
                for signal_decay_remainder in [[0, 0, 0, 0], [16, 3, 0, 7]] {
                    for elapsed in [1, 17, 31] {
                        let tile = TileState {
                            diffuse_energy,
                            signal_energy,
                            signal_decay_remainder,
                            ..TileState::default()
                        };
                        let signal_before = tile.signal_energy.iter().any(|energy| *energy > 0);
                        let diffuse_before = tile.diffuse_energy > 0;
                        let planned = planned_signal_frontier_changes(&tile, elapsed, 5, 17);
                        let mut advanced = tile;
                        advance_signal_decay(&mut advanced, elapsed, 5, 17).unwrap();
                        let signal_after = advanced.signal_energy.iter().any(|energy| *energy > 0);
                        let diffuse_after = advanced.diffuse_energy > 0;
                        assert_eq!(planned.0, signal_before && !signal_after);
                        assert_eq!(planned.1, !diffuse_before && diffuse_after);
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_diffusion_tracks_directed_frontier_changes_and_clears_scratch() {
        const BOARD: usize = 64;
        let mut neighborhood = NeighborhoodSpec::moore_8(BoundaryRule::Wrap);
        let east_only = SlotMask::from_slots([LocalSlot(4)]);
        neighborhood.set_target_mask(TargetingAction::Move, east_only);
        let rules = ReferenceRuleset {
            neighborhood,
            diffusion_targets: east_only,
            diffusion_rate_numerator: 1,
            diffusion_rate_denominator: 1,
            ..ReferenceRuleset::default()
        };
        let mut parallel = ReferenceSimulation::new(BOARD, BOARD, rules).unwrap();
        for (index, tile) in parallel.tiles.iter_mut().enumerate() {
            let x = index % BOARD;
            tile.diffuse_energy = if x.is_multiple_of(4) { 0 } else { 8 };
        }
        parallel.field_frontiers_dirty = true;
        parallel.rebuild_field_frontiers();
        parallel.set_passive_parallel_thresholds(Some(2), Some(2));
        let initial_frontier = parallel.active_diffuse_tiles.clone();
        let mut serial = parallel.clone();
        serial.set_passive_parallel_thresholds(None, None);

        parallel.apply_diffusion_step().unwrap();
        serial.apply_diffusion_step().unwrap();
        assert_eq!(parallel.tiles, serial.tiles);
        assert_eq!(parallel.active_diffuse_tiles, serial.active_diffuse_tiles);
        assert_ne!(parallel.active_diffuse_tiles, initial_frontier);
        assert_eq!(
            parallel.active_diffuse_tiles,
            parallel
                .tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    (tile.diffuse_energy > 0).then_some(TileIndex(index))
                })
                .collect()
        );
        assert!(parallel.diffusion_touched.is_empty());
        assert!(parallel
            .diffusion_incoming
            .iter()
            .all(|incoming| *incoming == 0));
        assert!(parallel
            .dense_metabolism_deposits
            .0
            .iter()
            .all(|deposit| deposit.load(Ordering::Relaxed) == 0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_diffusion_reports_lowest_overflow_without_canonical_mutation() {
        let rules = ReferenceRuleset {
            diffusion_rate_numerator: 1,
            diffusion_rate_denominator: 1,
            ..ReferenceRuleset::default()
        };
        let mut aggregation = ReferenceSimulation::new(3, 1, rules.clone()).unwrap();
        for tile in aggregation.tiles.iter_mut() {
            tile.diffuse_energy = u64::MAX;
        }
        aggregation.field_frontiers_dirty = true;
        aggregation.rebuild_field_frontiers();
        aggregation.diffusion_neighbors =
            Arc::new(vec![vec![], vec![TileIndex(0)], vec![TileIndex(0)]]);
        aggregation.diffusion_sources =
            Arc::new(vec![vec![TileIndex(1), TileIndex(2)], vec![], vec![]]);
        aggregation.set_passive_parallel_thresholds(Some(2), Some(2));
        let aggregation_tiles = aggregation.tiles.clone();
        let aggregation_frontier = aggregation.active_diffuse_tiles.clone();

        assert!(matches!(
            aggregation.apply_diffusion_step(),
            Err(ResolutionError::ArithmeticOverflow(
                "aggregating diffuse transport"
            ))
        ));
        assert_eq!(aggregation.tiles, aggregation_tiles);
        assert_eq!(aggregation.active_diffuse_tiles, aggregation_frontier);
        assert!(aggregation.diffusion_touched.is_empty());
        assert!(aggregation
            .diffusion_incoming
            .iter()
            .all(|incoming| *incoming == 0));

        let mut commit = ReferenceSimulation::new(2, 1, rules).unwrap();
        for tile in commit.tiles.iter_mut() {
            tile.diffuse_energy = u64::MAX;
        }
        commit.field_frontiers_dirty = true;
        commit.rebuild_field_frontiers();
        commit.diffusion_neighbors = Arc::new(vec![vec![], vec![TileIndex(0)]]);
        commit.diffusion_sources = Arc::new(vec![vec![TileIndex(1)], vec![]]);
        commit.set_passive_parallel_thresholds(Some(2), Some(2));
        let commit_tiles = commit.tiles.clone();

        assert!(matches!(
            commit.apply_diffusion_step(),
            Err(ResolutionError::ArithmeticOverflow(
                "committing diffuse transport"
            ))
        ));
        assert_eq!(commit.tiles, commit_tiles);
        assert!(commit
            .diffusion_incoming
            .iter()
            .all(|incoming| *incoming == 0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_parallel_overflow_guards_preserve_serial_failure_state() {
        let rules = ReferenceRuleset {
            digestion_rate_numerator: 1,
            digestion_rate_denominator: 1,
            metabolism_rate_numerator: 0,
            signal_decay_rate_numerator: 1,
            signal_decay_rate_denominator: 1,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut seeded = ReferenceSimulation::new(2, 1, rules.clone()).unwrap();
        seeded.add_cell(TileIndex(0), 1, u64::MAX, 0).unwrap();
        seeded.add_cell(TileIndex(1), 1, 1, 0).unwrap();
        let mut state = seeded.canonical_state();
        for (_, cell) in state.cells.iter_mut() {
            cell.gut_energy = 1;
        }
        for tile in &mut state.tiles {
            tile.diffuse_energy = u64::MAX;
            tile.signal_energy[0] = 1;
        }
        let mut parallel = ReferenceSimulation::from_canonical_state(2, 1, rules, state).unwrap();
        parallel.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut serial = parallel.clone();
        serial.set_passive_parallel_thresholds(None, None);

        let parallel_error = parallel.apply_digestion(1).unwrap_err();
        let serial_error = serial.apply_digestion(1).unwrap_err();
        assert_eq!(parallel_error.to_string(), serial_error.to_string());
        assert_eq!(parallel.cells, serial.cells);

        let parallel_error = parallel.apply_signal_decay(1).unwrap_err();
        let serial_error = serial.apply_signal_decay(1).unwrap_err();
        assert_eq!(parallel_error.to_string(), serial_error.to_string());
        assert_eq!(parallel.tiles, serial.tiles);

        let metabolism_rules = ReferenceRuleset {
            digestion_rate_numerator: 0,
            metabolism_rate_numerator: 1,
            metabolism_rate_denominator: 1,
            signal_decay_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut parallel = ReferenceSimulation::new(2, 1, metabolism_rules).unwrap();
        parallel.add_cell(TileIndex(0), 1, 1, 0).unwrap();
        parallel.add_cell(TileIndex(1), 1, 1, 0).unwrap();
        parallel.tiles[0].diffuse_energy = u64::MAX;
        parallel.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut serial = parallel.clone();
        serial.set_passive_parallel_thresholds(None, None);

        let parallel_error = parallel.apply_metabolism(1).unwrap_err();
        let serial_error = serial.apply_metabolism(1).unwrap_err();
        assert_eq!(parallel_error.to_string(), serial_error.to_string());
        assert_eq!(parallel.cells, serial.cells);
        assert_eq!(parallel.tiles, serial.tiles);
        assert!(parallel.diffusion_touched.is_empty());
        assert!(parallel
            .diffusion_incoming
            .iter()
            .all(|incoming| *incoming == 0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dense_digestion_fusion_preserves_deadline_overflow_and_hole_fallback() {
        let rules = ReferenceRuleset {
            digestion_rate_numerator: 1,
            digestion_rate_denominator: 1,
            metabolism_rate_numerator: 1,
            metabolism_rate_denominator: 1,
            signal_decay_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };

        let mut overflow = ReferenceSimulation::new(2, 1, rules.clone()).unwrap();
        overflow.add_cell(TileIndex(0), 1, 1, 0).unwrap();
        overflow.add_cell(TileIndex(1), 1, 1, 0).unwrap();
        for cell in overflow.cells.values_mut() {
            cell.gut_energy = 1;
        }
        overflow.rebuild_cell_passive_indexes();
        overflow.now = SimTime(u64::MAX - 1);
        overflow.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut overflow_serial = overflow.clone();
        overflow_serial.set_passive_parallel_thresholds(None, None);

        overflow.apply_digestion(1).unwrap();
        overflow_serial.apply_digestion(1).unwrap();
        assert_eq!(overflow.cells, overflow_serial.cells);
        assert_eq!(
            overflow.next_metabolic_event_time(),
            overflow_serial.next_metabolic_event_time()
        );
        assert!(matches!(
            overflow.next_metabolic_event_time(),
            Err(ResolutionError::ArithmeticOverflow(
                "scheduling metabolic exhaustion time"
            ))
        ));

        let mut seeded = ReferenceSimulation::new(3, 1, rules).unwrap();
        for index in 0..3 {
            seeded.add_cell(TileIndex(index), 1, 10, 0).unwrap();
        }
        let mut state = seeded.canonical_state();
        state.cells.remove(1);
        state.tiles[1].occupant = None;
        for (_, cell) in &mut state.cells {
            cell.gut_energy = 3;
        }
        let mut holes =
            ReferenceSimulation::from_canonical_state(3, 1, seeded.rules.clone(), state).unwrap();
        assert_eq!(holes.cells.slots.page_count(), 1);
        assert_eq!(holes.cells.len(), 2);
        holes.set_passive_parallel_thresholds(Some(2), Some(2));
        let mut holes_serial = holes.clone();
        holes_serial.set_passive_parallel_thresholds(None, None);

        holes.apply_digestion(2).unwrap();
        holes_serial.apply_digestion(2).unwrap();
        assert_eq!(holes.cells, holes_serial.cells);
        assert_eq!(
            holes.next_metabolic_event_time(),
            holes_serial.next_metabolic_event_time()
        );
        assert_cell_passive_indexes(&holes);
    }

    #[test]
    fn checkpoint_rejects_invalid_or_disabled_passive_progress() {
        let rules = uniform_rules();
        let mut simulation = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
        simulation.add_cell(TileIndex(0), 10, 10, 0).unwrap();
        let mut state = simulation.canonical_state();
        state.cells[0].1.digestion_remainder = rules.digestion_rate_denominator;

        assert!(matches!(
            ReferenceSimulation::from_canonical_state(1, 1, rules, state),
            Err(ResolutionError::InvalidState(
                "cell digestion remainder exceeds its denominator"
            ))
        ));

        let rules = uniform_rules();
        let mut simulation = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
        simulation.add_cell(TileIndex(0), 10, 10, 0).unwrap();
        let mut state = simulation.canonical_state();
        state.cells[0].1.metabolism_remainder = 1;
        assert!(matches!(
            ReferenceSimulation::from_canonical_state(1, 1, rules, state),
            Err(ResolutionError::InvalidState(
                "cell has a metabolism remainder while metabolism is disabled"
            ))
        ));

        let rules = ReferenceRuleset {
            digestion_rate_numerator: 0,
            ..uniform_rules()
        };
        let mut simulation = ReferenceSimulation::new(1, 1, rules.clone()).unwrap();
        simulation.add_cell(TileIndex(0), 10, 10, 0).unwrap();
        let mut state = simulation.canonical_state();
        state.cells[0].1.gut_energy = 1;
        state.cells[0].1.digestion_remainder = 1;
        assert!(matches!(
            ReferenceSimulation::from_canonical_state(1, 1, rules, state),
            Err(ResolutionError::InvalidState(
                "cell has a digestion remainder while digestion is disabled"
            ))
        ));
    }

    #[test]
    #[ignore = "manual sparse passive-field throughput diagnostic"]
    fn benchmark_sparse_diffusion_against_dense_oracle() {
        use std::time::Instant;

        const STEPS: usize = 16;
        println!("board      sparse ms    dense ms    speedup    final frontier");
        for board in [128_usize, 256, 512] {
            let rules = ReferenceRuleset {
                neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
                metabolism_rate_numerator: 0,
                signal_decay_rate_numerator: 0,
                diffusion_rate_numerator: 3,
                diffusion_rate_denominator: 11,
                ..ReferenceRuleset::default()
            };
            let mut sparse = ReferenceSimulation::new(board, board, rules).unwrap();
            for (x, y, energy) in [
                (board / 4, board / 4, 10_000_000),
                (board / 2, board / 2, 20_000_000),
                (board * 3 / 4, board * 3 / 4, 30_000_000),
            ] {
                let tile = sparse.tile(x, y).unwrap();
                sparse.tile_state_mut(tile).unwrap().diffuse_energy = energy;
            }
            sparse.rebuild_field_frontiers();
            let mut dense_tiles = sparse.tiles.to_vec();
            let neighbors = sparse.diffusion_neighbors.clone();

            let sparse_started = Instant::now();
            for _ in 0..STEPS {
                sparse.apply_diffusion_step().unwrap();
            }
            let sparse_elapsed = sparse_started.elapsed();

            let dense_started = Instant::now();
            for _ in 0..STEPS {
                dense_diffusion_oracle(&mut dense_tiles, &neighbors, 3, 11);
            }
            let dense_elapsed = dense_started.elapsed();
            assert_eq!(sparse.tiles.to_vec(), dense_tiles);
            println!(
                "{board:>4}x{board:<4} {:>10.3}  {:>10.3}  {:>8.1}x  {:>14}",
                sparse_elapsed.as_secs_f64() * 1_000.0,
                dense_elapsed.as_secs_f64() * 1_000.0,
                dense_elapsed.as_secs_f64() / sparse_elapsed.as_secs_f64(),
                sparse.active_diffuse_tiles.len(),
            );
        }
    }

    #[test]
    #[ignore = "manual sparse passive-cell throughput diagnostic"]
    fn benchmark_sparse_digestion_against_dense_oracle() {
        use std::time::Instant;

        const STEPS: usize = 32;
        const DIGESTERS: usize = 32;
        println!("population   sparse ms    dense ms    speedup    digesters");
        for population in [1_000_usize, 10_000, 50_000] {
            let rules = ReferenceRuleset {
                digestion_rate_numerator: 3,
                digestion_rate_denominator: 11,
                metabolism_rate_numerator: 0,
                signal_decay_rate_numerator: 0,
                diffusion_rate_numerator: 0,
                ..ReferenceRuleset::default()
            };
            let mut seeded = ReferenceSimulation::new(256, 256, rules.clone()).unwrap();
            for index in 0..population {
                seeded.add_cell(TileIndex(index), 10, 100, 0).unwrap();
            }
            let mut state = seeded.canonical_state();
            for index in 0..DIGESTERS {
                let cell_index = index * population / DIGESTERS;
                state.cells[cell_index].1.gut_energy = 1_000_000;
            }
            let mut sparse =
                ReferenceSimulation::from_canonical_state(256, 256, rules, state).unwrap();
            let mut dense_cells = sparse.cells.clone();

            let sparse_started = Instant::now();
            for _ in 0..STEPS {
                sparse.apply_digestion(17).unwrap();
            }
            let sparse_elapsed = sparse_started.elapsed();

            let dense_started = Instant::now();
            for _ in 0..STEPS {
                dense_digestion_oracle(&mut dense_cells, 17, 3, 11);
            }
            let dense_elapsed = dense_started.elapsed();
            assert_eq!(sparse.cells, dense_cells);
            assert_cell_passive_indexes(&sparse);
            println!(
                "{population:>10} {:>10.3}  {:>10.3}  {:>8.1}x  {:>11}",
                sparse_elapsed.as_secs_f64() * 1_000.0,
                dense_elapsed.as_secs_f64() * 1_000.0,
                dense_elapsed.as_secs_f64() / sparse_elapsed.as_secs_f64(),
                sparse.active_digestion_cells.len(),
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "manual dense passive-kernel throughput diagnostic"]
    fn benchmark_dense_parallel_passive_kernels() {
        use std::time::Instant;

        const BOARD: usize = 512;
        const CELLS: usize = 200_000;
        const STEPS: usize = 8;
        const ELAPSED: u64 = 17;
        let rules = ReferenceRuleset {
            neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Wrap),
            digestion_rate_numerator: 3,
            digestion_rate_denominator: 11,
            metabolism_rate_numerator: 2,
            metabolism_rate_denominator: 13,
            signal_decay_rate_numerator: 5,
            signal_decay_rate_denominator: 17,
            diffusion_rate_numerator: 3,
            diffusion_rate_denominator: 11,
            ..ReferenceRuleset::default()
        };
        let mut seeded = ReferenceSimulation::new(BOARD, BOARD, rules.clone()).unwrap();
        for index in 0..CELLS {
            seeded.add_cell(TileIndex(index), 10, 1_000_000, 0).unwrap();
        }
        let mut state = seeded.canonical_state();
        for (_, cell) in state.cells.iter_mut() {
            cell.gut_energy = 1_000_000;
        }
        for tile in &mut state.tiles {
            tile.diffuse_energy = 1_000_000;
            tile.plant_capacity = 2_000_000;
            tile.plant_energy = 100;
            tile.plant_growth_rate = 3;
            tile.signal_energy = [1_000_000, 500_000, 0, 250_000];
        }
        let mut parallel =
            ReferenceSimulation::from_canonical_state(BOARD, BOARD, rules, state).unwrap();
        let mut serial = parallel.clone();
        serial.set_passive_parallel_thresholds(None, None);

        // Keep this a steady-state kernel comparison. The COW tile store makes
        // simulation cloning intentionally cheap, so detach both branches
        // before timing instead of charging the parallel branch alone for its
        // first write after the clone.
        parallel.tiles.iter_mut().for_each(|tile| {
            std::hint::black_box(tile);
        });
        serial.tiles.iter_mut().for_each(|tile| {
            std::hint::black_box(tile);
        });

        let parallel_started = Instant::now();
        let mut parallel_parts = [std::time::Duration::ZERO; 5];
        for _ in 0..STEPS {
            let part = Instant::now();
            parallel.apply_plant_growth(ELAPSED).unwrap();
            parallel_parts[0] += part.elapsed();
            let part = Instant::now();
            parallel.apply_digestion(ELAPSED).unwrap();
            parallel_parts[1] += part.elapsed();
            let part = Instant::now();
            parallel.apply_metabolism(ELAPSED).unwrap();
            parallel_parts[2] += part.elapsed();
            let part = Instant::now();
            parallel.apply_signal_decay(ELAPSED).unwrap();
            parallel_parts[3] += part.elapsed();
            let part = Instant::now();
            parallel.apply_diffusion_step().unwrap();
            parallel_parts[4] += part.elapsed();
        }
        let parallel_elapsed = parallel_started.elapsed();

        let serial_started = Instant::now();
        let mut serial_parts = [std::time::Duration::ZERO; 5];
        for _ in 0..STEPS {
            let part = Instant::now();
            serial.apply_plant_growth(ELAPSED).unwrap();
            serial_parts[0] += part.elapsed();
            let part = Instant::now();
            serial.apply_digestion(ELAPSED).unwrap();
            serial_parts[1] += part.elapsed();
            let part = Instant::now();
            serial.apply_metabolism(ELAPSED).unwrap();
            serial_parts[2] += part.elapsed();
            let part = Instant::now();
            serial.apply_signal_decay(ELAPSED).unwrap();
            serial_parts[3] += part.elapsed();
            let part = Instant::now();
            serial.apply_diffusion_step().unwrap();
            serial_parts[4] += part.elapsed();
        }
        let serial_elapsed = serial_started.elapsed();

        assert_eq!(parallel.cells, serial.cells);
        assert_eq!(parallel.tiles, serial.tiles);
        assert_eq!(parallel.active_diffuse_tiles, serial.active_diffuse_tiles);
        assert_eq!(parallel.active_signal_tiles, serial.active_signal_tiles);
        println!(
            "{CELLS} cells, {} tiles, {STEPS} combined steps: parallel {:.3} ms, serial {:.3} ms, {:.2}x speedup",
            BOARD * BOARD,
            parallel_elapsed.as_secs_f64() * 1_000.0,
            serial_elapsed.as_secs_f64() * 1_000.0,
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64(),
        );
        for (name, (parallel, serial)) in [
            "plant",
            "digestion",
            "metabolism(serial)",
            "signal",
            "diffusion",
        ]
        .into_iter()
        .zip(parallel_parts.into_iter().zip(serial_parts))
        {
            println!(
                "  {name:<10} parallel {:>8.3} ms serial {:>8.3} ms speedup {:>5.2}x",
                parallel.as_secs_f64() * 1_000.0,
                serial.as_secs_f64() * 1_000.0,
                serial.as_secs_f64() / parallel.as_secs_f64(),
            );
        }
    }

    #[test]
    #[ignore = "manual dense tile-store locality diagnostic"]
    fn benchmark_dense_tile_store_against_flat_vector() {
        use std::time::Instant;

        const TILES: usize = 512 * 512;
        const PASSES: usize = 64;
        let values = (0..TILES)
            .map(|index| TileState {
                loose_energy: u64::try_from(index % 251).unwrap(),
                signal_energy: [17, 31, 0, 9],
                ..TileState::default()
            })
            .collect::<Vec<_>>();
        let mut store = ReferenceTileStore::from_vec(values.clone());
        let mut flat = values;

        let store_started = Instant::now();
        for _ in 0..PASSES {
            for tile in store.iter_mut() {
                tile.loose_energy = tile.loose_energy.wrapping_add(1);
                std::hint::black_box(&mut tile.signal_energy[1]);
            }
        }
        let store_elapsed = store_started.elapsed();

        let flat_started = Instant::now();
        for _ in 0..PASSES {
            for tile in &mut flat {
                tile.loose_energy = tile.loose_energy.wrapping_add(1);
                std::hint::black_box(&mut tile.signal_energy[1]);
            }
        }
        let flat_elapsed = flat_started.elapsed();

        assert_eq!(store.to_vec(), flat);
        println!(
            "{TILES} tiles, {PASSES} dense passes: page-overlay {:.3} ms, flat {:.3} ms, {:.2}x flat cost",
            store_elapsed.as_secs_f64() * 1_000.0,
            flat_elapsed.as_secs_f64() * 1_000.0,
            store_elapsed.as_secs_f64() / flat_elapsed.as_secs_f64(),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "manual dense passive-cell subphase diagnostic"]
    fn benchmark_dense_cell_passive_subphases() {
        use std::time::Instant;

        const CELLS: usize = 200_000;
        const ELAPSED: u64 = 17;
        let rules = ReferenceRuleset {
            digestion_rate_numerator: 3,
            digestion_rate_denominator: 11,
            metabolism_rate_numerator: 2,
            metabolism_rate_denominator: 13,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(512, 512, rules).unwrap();
        for index in 0..CELLS {
            let key = simulation
                .add_cell(TileIndex(index), 10, 1_000_000, 0)
                .unwrap();
            simulation.cells.get_mut(&key).unwrap().gut_energy = 1_000_000;
        }
        simulation.rebuild_cell_passive_indexes();

        let scan_started = Instant::now();
        let transfer_fits = simulation.cells.values().all(|cell| {
            u128::from(cell.assimilated_energy) + u128::from(cell.gut_energy)
                <= u128::from(u64::MAX)
        });
        let scan_elapsed = scan_started.elapsed();
        assert!(transfer_fits);

        let mut parallel_digestion = simulation.cells.clone();
        let digestion_started = Instant::now();
        parallel_digestion
            .par_iter_mut()
            .try_for_each(|(_, cell)| advance_cell_digestion(cell, ELAPSED, 3, 11))
            .unwrap();
        let digestion_elapsed = digestion_started.elapsed();

        let mut digestion_frontier = simulation.active_digestion_cells.clone();
        let retain_started = Instant::now();
        digestion_frontier.retain(|key| {
            parallel_digestion
                .get(key)
                .is_some_and(|cell| cell.gut_energy > 0)
        });
        let retain_elapsed = retain_started.elapsed();

        let mut exhaustion = MetabolicExhaustionIndex::default();
        let exhaustion_started = Instant::now();
        exhaustion.rebuild(&parallel_digestion, SimTime(0), 2, 13);
        let exhaustion_elapsed = exhaustion_started.elapsed();

        let schedule_calculation_started = Instant::now();
        let schedules = parallel_digestion
            .par_iter()
            .map(|(key, cell)| (key, metabolic_exhaustion_time(SimTime(0), cell, 2, 13)))
            .collect::<Vec<_>>();
        let schedule_calculation_elapsed = schedule_calculation_started.elapsed();
        let mut assembled_exhaustion = MetabolicExhaustionIndex::default();
        let schedule_assembly_started = Instant::now();
        for (key, schedule) in &schedules {
            match schedule {
                Ok(time) => {
                    assembled_exhaustion
                        .positions
                        .insert(key.0, assembled_exhaustion.heap.len());
                    assembled_exhaustion.heap.push((*time, *key));
                }
                Err(error) => {
                    assembled_exhaustion
                        .positions
                        .insert(key.0, OVERFLOWED_METABOLIC_SCHEDULE);
                    assembled_exhaustion.overflows.insert(*key, error.clone());
                }
            }
        }
        for index in (0..assembled_exhaustion.heap.len() / 2).rev() {
            assembled_exhaustion.sift_down(index);
        }
        let schedule_assembly_elapsed = schedule_assembly_started.elapsed();
        assert_eq!(assembled_exhaustion.heap.first(), exhaustion.heap.first());

        let mut fused_cells = simulation.cells.clone();
        let fused_digestion_started = Instant::now();
        let fused_schedules = fused_cells
            .par_iter_mut()
            .map(|(key, cell)| {
                advance_cell_digestion(cell, ELAPSED, 3, 11).unwrap();
                (key, metabolic_exhaustion_time(SimTime(0), cell, 2, 13))
            })
            .collect::<Vec<_>>();
        let fused_digestion_elapsed = fused_digestion_started.elapsed();
        assert_eq!(fused_cells, parallel_digestion);
        assert_eq!(fused_schedules.len(), schedules.len());

        let mut serial_metabolism = parallel_digestion.clone();
        let serial_metabolism_started = Instant::now();
        for (_, cell) in &mut serial_metabolism {
            std::hint::black_box(advance_cell_metabolism(cell, ELAPSED, 2, 13).unwrap());
        }
        let serial_metabolism_elapsed = serial_metabolism_started.elapsed();

        let mut parallel_metabolism = parallel_digestion.clone();
        let parallel_metabolism_started = Instant::now();
        parallel_metabolism
            .par_iter_mut()
            .try_for_each(|(_, cell)| {
                std::hint::black_box(advance_cell_metabolism(cell, ELAPSED, 2, 13).unwrap());
                Ok::<_, ResolutionError>(())
            })
            .unwrap();
        let parallel_metabolism_elapsed = parallel_metabolism_started.elapsed();

        let mut deposit_tiles = simulation.tiles.clone();
        deposit_tiles.iter_mut().for_each(|tile| {
            std::hint::black_box(tile);
        });
        let deposit_started = Instant::now();
        for cell in parallel_digestion.values() {
            deposit_tiles[cell.position.0].diffuse_energy += 2;
        }
        let deposit_elapsed = deposit_started.elapsed();

        let activated = parallel_digestion
            .values()
            .map(|cell| cell.position)
            .collect::<Vec<_>>();
        let mut diffuse_frontier = BTreeSet::new();
        let frontier_extend_started = Instant::now();
        diffuse_frontier.extend(activated.iter().copied());
        let frontier_extend_elapsed = frontier_extend_started.elapsed();

        let frontier_scan_started = Instant::now();
        let scanned_frontier = deposit_tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| (tile.diffuse_energy > 0).then_some(TileIndex(index)))
            .collect::<BTreeSet<_>>();
        let frontier_scan_elapsed = frontier_scan_started.elapsed();
        assert_eq!(diffuse_frontier, scanned_frontier);

        let mut gathered_tiles = simulation.tiles.clone();
        let mut gathered_deposits = vec![0_u64; gathered_tiles.len()];
        let gather_started = Instant::now();
        for cell in parallel_digestion.values() {
            gathered_deposits[cell.position.0] = 2;
        }
        gathered_tiles.par_for_each_mut_indexed(|index, tile| {
            tile.diffuse_energy += gathered_deposits[index];
        });
        let gather_elapsed = gather_started.elapsed();
        assert_eq!(gathered_tiles, deposit_tiles);

        let mut metabolism_frontier = simulation.active_metabolism_cells.clone();
        let metabolism_retain_started = Instant::now();
        metabolism_frontier.retain(|key| {
            parallel_metabolism
                .get(key)
                .is_some_and(|cell| cell.assimilated_energy > 0)
        });
        let metabolism_retain_elapsed = metabolism_retain_started.elapsed();

        println!(
            "{CELLS} dense cells: overflow_scan_ms={:.3} digestion_arithmetic_parallel_ms={:.3} digestion_frontier_retain_ms={:.3} exhaustion_rebuild_ms={:.3} schedule_calculation_parallel_ms={:.3} schedule_assembly_ms={:.3} fused_digestion_schedule_ms={:.3} metabolism_arithmetic_serial_ms={:.3} metabolism_arithmetic_parallel_ms={:.3} metabolic_tile_deposit_ms={:.3} metabolism_frontier_extend_ms={:.3} metabolism_frontier_scan_ms={:.3} gathered_dense_deposit_ms={:.3} metabolism_frontier_retain_ms={:.3}",
            scan_elapsed.as_secs_f64() * 1_000.0,
            digestion_elapsed.as_secs_f64() * 1_000.0,
            retain_elapsed.as_secs_f64() * 1_000.0,
            exhaustion_elapsed.as_secs_f64() * 1_000.0,
            schedule_calculation_elapsed.as_secs_f64() * 1_000.0,
            schedule_assembly_elapsed.as_secs_f64() * 1_000.0,
            fused_digestion_elapsed.as_secs_f64() * 1_000.0,
            serial_metabolism_elapsed.as_secs_f64() * 1_000.0,
            parallel_metabolism_elapsed.as_secs_f64() * 1_000.0,
            deposit_elapsed.as_secs_f64() * 1_000.0,
            frontier_extend_elapsed.as_secs_f64() * 1_000.0,
            frontier_scan_elapsed.as_secs_f64() * 1_000.0,
            gather_elapsed.as_secs_f64() * 1_000.0,
            metabolism_retain_elapsed.as_secs_f64() * 1_000.0,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "manual dense signal-decay subphase diagnostic"]
    fn benchmark_dense_signal_decay_subphases() {
        use std::time::Instant;

        const TILES: usize = 512 * 512;
        const ELAPSED: u64 = 17;
        let values = (0..TILES)
            .map(|index| TileState {
                diffuse_energy: if index % 5 == 0 { 0 } else { 1_000_000 },
                signal_energy: [1_000_000, 500_000, 0, 250_000],
                signal_decay_remainder: [
                    u64::try_from(index % 17).unwrap(),
                    u64::try_from((index + 3) % 17).unwrap(),
                    0,
                    u64::try_from((index + 7) % 17).unwrap(),
                ],
                ..TileState::default()
            })
            .collect::<Vec<_>>();
        let mut tiles = ReferenceTileStore::from_vec(values);
        let mut fused_tiles = tiles.clone();
        fused_tiles.iter_mut().for_each(|tile| {
            std::hint::black_box(tile);
        });
        let mut wrapper_tiles = fused_tiles.clone();
        wrapper_tiles.iter_mut().for_each(|tile| {
            std::hint::black_box(tile);
        });

        let serial_preflight_started = Instant::now();
        let serial_fits = tiles.iter().all(|tile| {
            u128::from(tile.diffuse_energy)
                + tile.signal_energy.into_iter().map(u128::from).sum::<u128>()
                <= u128::from(u64::MAX)
        });
        let serial_preflight_elapsed = serial_preflight_started.elapsed();

        let parallel_preflight_started = Instant::now();
        let parallel_fits = tiles.par_iter().all(|tile| {
            u128::from(tile.diffuse_energy)
                + tile.signal_energy.into_iter().map(u128::from).sum::<u128>()
                <= u128::from(u64::MAX)
        });
        let parallel_preflight_elapsed = parallel_preflight_started.elapsed();
        assert!(serial_fits && parallel_fits);

        let arithmetic_started = Instant::now();
        let decayed = tiles
            .par_iter_mut()
            .try_fold(
                || [0_u128; REFERENCE_SIGNAL_CHANNELS],
                |mut totals, tile| {
                    let tile_decayed = advance_signal_decay(tile, ELAPSED, 5, 17)?;
                    for channel in 0..REFERENCE_SIGNAL_CHANNELS {
                        totals[channel] += u128::from(tile_decayed[channel]);
                    }
                    Ok::<_, ResolutionError>(totals)
                },
            )
            .try_reduce(
                || [0_u128; REFERENCE_SIGNAL_CHANNELS],
                |mut left, right| {
                    for channel in 0..REFERENCE_SIGNAL_CHANNELS {
                        left[channel] += right[channel];
                    }
                    Ok(left)
                },
            )
            .unwrap();
        let arithmetic_elapsed = arithmetic_started.elapsed();
        std::hint::black_box(decayed);

        let signal_scan_started = Instant::now();
        let signal_frontier = tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                tile.signal_energy
                    .iter()
                    .any(|energy| *energy > 0)
                    .then_some(TileIndex(index))
            })
            .collect::<BTreeSet<_>>();
        let signal_scan_elapsed = signal_scan_started.elapsed();

        let diffuse_scan_started = Instant::now();
        let diffuse_frontier = tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| (tile.diffuse_energy > 0).then_some(TileIndex(index)))
            .collect::<BTreeSet<_>>();
        let diffuse_scan_elapsed = diffuse_scan_started.elapsed();

        let frontier_clone_started = Instant::now();
        let cloned_frontier = std::hint::black_box(signal_frontier.clone());
        let frontier_clone_elapsed = frontier_clone_started.elapsed();
        assert_eq!(cloned_frontier, signal_frontier);

        let combined_scan_started = Instant::now();
        let mut combined_signal = BTreeSet::new();
        let mut combined_diffuse = BTreeSet::new();
        for (index, tile) in tiles.iter().enumerate() {
            if tile.signal_energy.iter().any(|energy| *energy > 0) {
                combined_signal.insert(TileIndex(index));
            }
            if tile.diffuse_energy > 0 {
                combined_diffuse.insert(TileIndex(index));
            }
        }
        let combined_scan_elapsed = combined_scan_started.elapsed();
        assert_eq!(combined_signal, signal_frontier);
        assert_eq!(combined_diffuse, diffuse_frontier);

        let combined_vectors_started = Instant::now();
        let mut signal_indexes = Vec::with_capacity(signal_frontier.len());
        let mut diffuse_indexes = Vec::with_capacity(diffuse_frontier.len());
        for (index, tile) in tiles.iter().enumerate() {
            if tile.signal_energy.iter().any(|energy| *energy > 0) {
                signal_indexes.push(TileIndex(index));
            }
            if tile.diffuse_energy > 0 {
                diffuse_indexes.push(TileIndex(index));
            }
        }
        let vector_signal = signal_indexes.into_iter().collect::<BTreeSet<_>>();
        let vector_diffuse = diffuse_indexes.into_iter().collect::<BTreeSet<_>>();
        let combined_vectors_elapsed = combined_vectors_started.elapsed();
        assert_eq!(vector_signal, signal_frontier);
        assert_eq!(vector_diffuse, diffuse_frontier);

        let rules = ReferenceRuleset {
            signal_decay_rate_numerator: 5,
            signal_decay_rate_denominator: 17,
            ..ReferenceRuleset::default()
        };
        let mut fused = ReferenceSimulation::new(512, 512, rules).unwrap();
        fused.tiles = fused_tiles;
        let fused_started = Instant::now();
        let fused_decayed = fused
            .apply_signal_decay_parallel_dense(ELAPSED, 5, 17, true, true)
            .unwrap();
        let fused_elapsed = fused_started.elapsed();
        assert_eq!(fused_decayed, decayed);
        assert_eq!(fused.tiles, tiles);
        assert_eq!(fused.active_signal_tiles, signal_frontier);
        assert_eq!(fused.active_diffuse_tiles, diffuse_frontier);

        let mut wrapper = ReferenceSimulation::new(512, 512, fused.rules.clone()).unwrap();
        wrapper.tiles = wrapper_tiles;
        wrapper.active_signal_tiles = signal_frontier.clone();
        let wrapper_started = Instant::now();
        let wrapper_decayed = wrapper.apply_signal_decay(ELAPSED).unwrap();
        let wrapper_elapsed = wrapper_started.elapsed();
        assert_eq!(wrapper_decayed, decayed);
        assert_eq!(wrapper.tiles, tiles);
        assert_eq!(wrapper.active_signal_tiles, signal_frontier);
        assert_eq!(wrapper.active_diffuse_tiles, diffuse_frontier);

        println!(
            "{TILES} dense signal tiles: serial_preflight_ms={:.3} parallel_preflight_ms={:.3} arithmetic_ms={:.3} signal_frontier_ms={:.3} diffuse_frontier_ms={:.3} frontier_clone_ms={:.3} combined_frontiers_ms={:.3} combined_vectors_ms={:.3} fused_page_bitmaps_ms={:.3} full_wrapper_ms={:.3}",
            serial_preflight_elapsed.as_secs_f64() * 1_000.0,
            parallel_preflight_elapsed.as_secs_f64() * 1_000.0,
            arithmetic_elapsed.as_secs_f64() * 1_000.0,
            signal_scan_elapsed.as_secs_f64() * 1_000.0,
            diffuse_scan_elapsed.as_secs_f64() * 1_000.0,
            frontier_clone_elapsed.as_secs_f64() * 1_000.0,
            combined_scan_elapsed.as_secs_f64() * 1_000.0,
            combined_vectors_elapsed.as_secs_f64() * 1_000.0,
            fused_elapsed.as_secs_f64() * 1_000.0,
            wrapper_elapsed.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "manual metabolic event-index throughput diagnostic"]
    fn benchmark_metabolic_exhaustion_index_against_population_scan() {
        use std::hint::black_box;
        use std::time::Instant;

        const CELLS: usize = 200_000;
        const QUERIES: usize = 128;
        let rules = ReferenceRuleset {
            metabolism_rate_numerator: 3,
            metabolism_rate_denominator: 1024,
            diffusion_rate_numerator: 0,
            ..uniform_rules()
        };
        let mut simulation = ReferenceSimulation::new(512, 512, rules).unwrap();
        for index in 0..CELLS {
            simulation
                .add_cell(
                    TileIndex(index),
                    1,
                    10_000 + u64::try_from(index % 997).unwrap(),
                    0,
                )
                .unwrap();
        }

        let indexed_started = Instant::now();
        for _ in 0..QUERIES {
            black_box(simulation.next_metabolic_event_time().unwrap());
        }
        let indexed = indexed_started.elapsed();

        let scanned_started = Instant::now();
        for _ in 0..QUERIES {
            let mut next = None;
            for cell in simulation.cells.values() {
                let event = metabolic_exhaustion_time(
                    simulation.now,
                    cell,
                    simulation.rules.metabolism_rate_numerator,
                    simulation.rules.metabolism_rate_denominator,
                )
                .unwrap();
                next = Some(next.map_or(event, |current: SimTime| current.min(event)));
            }
            black_box(next);
        }
        let scanned = scanned_started.elapsed();
        println!(
            "{CELLS} cells, {QUERIES} queries: indexed {:.3} ms, scan {:.3} ms, {:.1}x speedup",
            indexed.as_secs_f64() * 1_000.0,
            scanned.as_secs_f64() * 1_000.0,
            scanned.as_secs_f64() / indexed.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "manual private-memory snapshot throughput diagnostic"]
    fn benchmark_private_memory_snapshot_churn() {
        use std::time::{Duration, Instant};

        const CELLS: usize = 5_000;
        const MEMORY_BYTES: usize = 2_048;
        const BATCHES: usize = 20;
        let mut seeded = ReferenceSimulation::new(128, 128, uniform_rules()).unwrap();
        for index in 0..CELLS {
            seeded.add_cell(TileIndex(index), 10, 100, 0).unwrap();
        }
        let mut state = seeded.canonical_state();
        for (index, (_, cell)) in state.cells.iter_mut().enumerate() {
            cell.private_memory = vec![u8::try_from(index % 251).unwrap(); MEMORY_BYTES].into();
        }
        let mut simulation =
            ReferenceSimulation::from_canonical_state(128, 128, uniform_rules(), state).unwrap();
        let actors = simulation.cells.keys().copied().collect::<Vec<_>>();
        let mut elapsed = Duration::ZERO;
        for _ in 0..BATCHES {
            for actor in &actors {
                simulation
                    .commit_action(*actor, ActionRequest::Wait)
                    .unwrap();
            }
            let started = Instant::now();
            simulation.resolve_next_batch().unwrap();
            elapsed += started.elapsed();
        }
        println!(
            "{CELLS} cells x {MEMORY_BYTES} private bytes, {BATCHES} batches: {:.3} ms/batch, {:.0} actions/s",
            elapsed.as_secs_f64() * 1_000.0 / BATCHES as f64,
            (CELLS * BATCHES) as f64 / elapsed.as_secs_f64(),
        );
    }

    #[test]
    #[ignore = "manual fragmented completion-snapshot throughput diagnostic"]
    fn benchmark_fragmented_completion_snapshot() {
        use std::time::{Duration, Instant};

        const CELLS: usize = 50_000;
        const ACTIVE: usize = 8;
        const MEMORY_BYTES: usize = 2_048;
        const BATCHES: usize = 20;
        let rules = uniform_rules();
        let mut seeded = ReferenceSimulation::new(256, 256, rules.clone()).unwrap();
        for index in 0..CELLS {
            seeded.add_cell(TileIndex(index), 10, 100, 0).unwrap();
        }
        let mut state = seeded.canonical_state();
        for (index, (_, cell)) in state.cells.iter_mut().enumerate() {
            cell.private_memory = vec![u8::try_from(index % 251).unwrap(); MEMORY_BYTES].into();
        }
        let mut simulation =
            ReferenceSimulation::from_canonical_state(256, 256, rules, state).unwrap();
        let actors = simulation
            .cells
            .keys()
            .copied()
            .take(ACTIVE)
            .collect::<Vec<_>>();
        let mut elapsed = Duration::ZERO;
        let mut prestate_ns = 0_u64;
        let mut hash_ns = 0_u64;
        let mut delta_ns = 0_u64;
        let mut passive_ns = 0_u64;
        let mut total_ns = 0_u64;
        for _ in 0..BATCHES {
            for actor in &actors {
                simulation
                    .commit_action(*actor, ActionRequest::Wait)
                    .unwrap();
            }
            let started = Instant::now();
            simulation.resolve_next_batch().unwrap();
            elapsed += started.elapsed();
            prestate_ns += simulation
                .last_resolution_metrics()
                .phase_timings
                .prestate_materialization_ns;
            hash_ns += simulation
                .last_resolution_metrics()
                .phase_timings
                .state_hashes_ns;
            delta_ns += simulation
                .last_resolution_metrics()
                .phase_timings
                .delta_generation_ns;
            passive_ns += simulation
                .last_resolution_metrics()
                .phase_timings
                .passive_updates_ns;
            total_ns += simulation.last_resolution_metrics().phase_timings.total_ns;
        }
        println!(
            "{CELLS} resident cells, {ACTIVE} due, {MEMORY_BYTES} private bytes, {BATCHES} batches: {:.3} ms/batch, {:.3} ms resolver total, {:.3} ms passive, {:.3} ms prestate, {:.3} ms hashes, {:.3} ms delta",
            elapsed.as_secs_f64() * 1_000.0 / BATCHES as f64,
            total_ns as f64 / 1_000_000.0 / BATCHES as f64,
            passive_ns as f64 / 1_000_000.0 / BATCHES as f64,
            prestate_ns as f64 / 1_000_000.0 / BATCHES as f64,
            hash_ns as f64 / 1_000_000.0 / BATCHES as f64,
            delta_ns as f64 / 1_000_000.0 / BATCHES as f64,
        );
    }
}
