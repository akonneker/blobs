//! Canonical, bounded checkpoints for the complete reference simulation.

use std::error::Error;
use std::fmt::{Display, Formatter};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::hashing::CanonicalHash;
#[cfg(not(target_arch = "wasm32"))]
use super::hashing::{CellMerkleCache, IncrementalStateHashSeed};
use super::neighborhood::{
    BoundaryRule, DiagonalCornerRule, LocalOffset, NeighborhoodSpec, ObservationMasks, SlotMask,
    TargetingAction, MAX_LOCAL_SLOTS,
};
#[cfg(not(target_arch = "wasm32"))]
use super::reference::{
    CellState, CellStore, ReferenceCompiledTopology, ReferenceTilePage, ReferenceTileStore,
    TileState, REFERENCE_TILE_CHUNK_LEN, REFERENCE_TILE_PAGE_CHUNKS,
};
use super::reference::{
    DurationRule, EffortProfile, ReferenceRuleset, ReferenceSimulation, ResolutionError, SimTime,
    SimulationState, TimeConfig,
};
use super::replay::{
    decode_cell_state, decode_tile_state, decode_vec, encode_cell_state, encode_tile_state, Reader,
    ReplayError, Writer,
};
use super::CellKey;

pub const CHECKPOINT_FORMAT_VERSION: u16 = 6;
const CHECKPOINT_MAGIC: &[u8; 8] = b"BLBCHK06";
const CHECKPOINT_DOMAIN: &[u8] = b"blob.simulation.checkpoint";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLimits {
    pub max_checkpoint_bytes: usize,
    pub max_tiles: usize,
    pub max_cells: usize,
    /// Bounds lifetime key space, including dead cells and empty hash pages.
    /// This is independent of the live-cell limit: old survivors keep their keys.
    pub max_cell_slots: usize,
    pub max_private_memory_bytes: usize,
}

impl Default for CheckpointLimits {
    fn default() -> Self {
        Self {
            max_checkpoint_bytes: 64 * 1024 * 1024,
            max_tiles: 4_000_000,
            max_cells: 1_000_000,
            max_cell_slots: 4_000_000,
            max_private_memory_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    TooLarge {
        actual: usize,
        limit: usize,
    },
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u16),
    IntegrityHashMismatch {
        expected: CanonicalHash,
        actual: CanonicalHash,
    },
    SemanticRulesetHashMismatch,
    CompiledRulesetHashMismatch,
    StateHashMismatch,
    InvalidField(&'static str),
    TrailingBytes(usize),
    Replay(ReplayError),
    Resolution(ResolutionError),
}

impl Display for CheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { actual, limit } => {
                write!(formatter, "checkpoint is {actual} bytes; limit is {limit}")
            }
            Self::Truncated => write!(formatter, "checkpoint is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid checkpoint magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported checkpoint format version {version}")
            }
            Self::IntegrityHashMismatch { expected, actual } => write!(
                formatter,
                "checkpoint integrity hash mismatch: expected {expected}, got {actual}"
            ),
            Self::SemanticRulesetHashMismatch => {
                write!(formatter, "checkpoint semantic ruleset hash mismatch")
            }
            Self::CompiledRulesetHashMismatch => {
                write!(formatter, "checkpoint compiled ruleset hash mismatch")
            }
            Self::StateHashMismatch => write!(formatter, "checkpoint state hash mismatch"),
            Self::InvalidField(field) => write!(formatter, "invalid checkpoint field: {field}"),
            Self::TrailingBytes(count) => {
                write!(formatter, "checkpoint has {count} trailing body bytes")
            }
            Self::Replay(error) => Display::fmt(error, formatter),
            Self::Resolution(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for CheckpointError {}

impl From<ReplayError> for CheckpointError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<ResolutionError> for CheckpointError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

/// A validated snapshot of the complete reference state and its exact rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceCheckpoint {
    width: usize,
    height: usize,
    rules: ReferenceRuleset,
    state: SimulationState,
    semantic_ruleset_hash: CanonicalHash,
    compiled_ruleset_hash: CanonicalHash,
    state_hash: CanonicalHash,
    checkpoint_hash: CanonicalHash,
}

#[cfg(not(target_arch = "wasm32"))]
const PLANNER_TILE_CHUNK_LEN: usize = REFERENCE_TILE_CHUNK_LEN;
#[cfg(not(target_arch = "wasm32"))]
const PLANNER_TILE_PAGE_CHUNKS: usize = REFERENCE_TILE_PAGE_CHUNKS;
#[cfg(not(target_arch = "wasm32"))]
const PLANNER_CELL_CHUNK_LEN: usize = 8;
#[cfg(not(target_arch = "wasm32"))]
const PLANNER_HASH_CHUNK_LEN: usize = 64;
#[cfg(not(target_arch = "wasm32"))]
const PLANNER_HASH_SEED_MIN_TILE_PAGES: usize = 2;
#[cfg(not(target_arch = "wasm32"))]
const PLANNER_PARALLEL_TILE_MATERIALIZATION_THRESHOLD: usize = 512 * 512;

#[cfg(not(target_arch = "wasm32"))]
type PlannerTilePage = ReferenceTilePage;
#[cfg(not(target_arch = "wasm32"))]
type PlannerCellChunk = Arc<[(CellKey, CellState)]>;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannerDerivedChunks<T> {
    chunks: Vec<Arc<[T]>>,
    len: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Clone + PartialEq> PlannerDerivedChunks<T> {
    fn from_slice(values: &[T], parent: Option<&Self>) -> Self {
        let chunks = values
            .chunks(PLANNER_HASH_CHUNK_LEN)
            .enumerate()
            .map(|(index, values)| {
                parent
                    .and_then(|parent| parent.chunks.get(index))
                    .filter(|chunk| chunk.as_ref() == values)
                    .map_or_else(|| Arc::from(values.to_vec()), Arc::clone)
            })
            .collect();
        Self {
            chunks,
            len: values.len(),
        }
    }

    fn materialize(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(self.len);
        for chunk in &self.chunks {
            values.extend_from_slice(chunk);
        }
        debug_assert_eq!(values.len(), self.len);
        values
    }

    fn inline_heap_bytes(&self) -> usize {
        self.chunks
            .capacity()
            .saturating_mul(std::mem::size_of::<Arc<[T]>>())
    }

    fn allocation_bytes(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.len().saturating_mul(std::mem::size_of::<T>()))
            .sum()
    }

    fn allocations(&self) -> impl Iterator<Item = (*const u8, usize)> + '_ {
        self.chunks.iter().map(|chunk| {
            (
                chunk.as_ptr().cast::<u8>(),
                chunk.len().saturating_mul(std::mem::size_of::<T>()),
            )
        })
    }

    fn shared_chunk_count(&self, parent: &Self) -> usize {
        self.chunks
            .iter()
            .zip(&parent.chunks)
            .filter(|(left, right)| Arc::ptr_eq(left, right))
            .count()
    }

    fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannerHashSeed {
    compiled_ruleset_hash: CanonicalHash,
    tile_page_count: usize,
    tile_base: usize,
    tile_nodes: PlannerDerivedChunks<CanonicalHash>,
    cell_tree: CellMerkleCache,
}

#[cfg(not(target_arch = "wasm32"))]
impl PlannerHashSeed {
    fn from_seed(seed: IncrementalStateHashSeed, parent: Option<&Self>) -> Self {
        let compatible_parent = parent.filter(|parent| {
            parent.compiled_ruleset_hash == seed.compiled_ruleset_hash
                && parent.tile_page_count == seed.tile_page_count
                && parent.tile_base == seed.tile_base
        });
        Self {
            compiled_ruleset_hash: seed.compiled_ruleset_hash,
            tile_page_count: seed.tile_page_count,
            tile_base: seed.tile_base,
            tile_nodes: PlannerDerivedChunks::from_slice(
                &seed.tile_nodes[seed.tile_base..seed.tile_base + seed.tile_page_count],
                compatible_parent.map(|parent| &parent.tile_nodes),
            ),
            cell_tree: seed.cell_tree,
        }
    }

    fn materialize(&self) -> IncrementalStateHashSeed {
        IncrementalStateHashSeed {
            compiled_ruleset_hash: self.compiled_ruleset_hash,
            tile_page_count: self.tile_page_count,
            tile_base: self.tile_base,
            tile_nodes: self.tile_nodes.materialize(),
            cell_tree: self.cell_tree.clone(),
        }
    }

    fn inline_heap_bytes(&self) -> usize {
        self.tile_nodes.inline_heap_bytes()
    }

    fn allocation_bytes(&self) -> usize {
        self.tile_nodes.allocation_bytes()
            + self
                .cell_tree
                .allocations()
                .iter()
                .map(|(_, bytes)| bytes)
                .sum::<usize>()
    }

    fn allocations(&self) -> Vec<(*const u8, usize)> {
        self.tile_nodes
            .allocations()
            .chain(self.cell_tree.allocations())
            .collect()
    }

    fn chunk_count(&self) -> usize {
        self.tile_nodes.chunk_count() + self.cell_tree.allocations().len()
    }

    fn shared_chunk_count(&self, parent: &Self) -> usize {
        let parent_allocations = parent
            .cell_tree
            .allocations()
            .into_iter()
            .map(|(pointer, _)| pointer)
            .collect::<std::collections::BTreeSet<_>>();
        self.tile_nodes.shared_chunk_count(&parent.tile_nodes)
            + self
                .cell_tree
                .allocations()
                .iter()
                .filter(|(pointer, _)| parent_allocations.contains(pointer))
                .count()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum PlannerCellChunks {
    Empty,
    Single(PlannerCellChunk),
    Multiple(Vec<PlannerCellChunk>),
}

#[cfg(not(target_arch = "wasm32"))]
impl PlannerCellChunks {
    fn from_cell_store(cells: &CellStore, parent: Option<&Self>) -> Self {
        if cells.is_empty() {
            return Self::Empty;
        }
        let mut scratch = Vec::with_capacity(PLANNER_CELL_CHUNK_LEN);
        let mut chunks = Vec::with_capacity(cells.len().div_ceil(PLANNER_CELL_CHUNK_LEN));
        for (key, cell) in cells.iter() {
            scratch.push((*key, cell));
            if scratch.len() == PLANNER_CELL_CHUNK_LEN {
                chunks.push(Self::reuse_or_copy_borrowed(&scratch, parent, chunks.len()));
                scratch.clear();
            }
        }
        if !scratch.is_empty() {
            chunks.push(Self::reuse_or_copy_borrowed(&scratch, parent, chunks.len()));
        }
        match chunks.len() {
            1 => Self::Single(chunks.pop().expect("one cell chunk disappeared")),
            _ => Self::Multiple(chunks),
        }
    }

    fn reuse_or_copy_borrowed(
        chunk: &[(CellKey, &CellState)],
        parent: Option<&Self>,
        index: usize,
    ) -> PlannerCellChunk {
        parent
            .and_then(|parent| parent.as_slice().get(index))
            .filter(|parent| {
                parent.len() == chunk.len()
                    && parent
                        .iter()
                        .zip(chunk)
                        .all(|((parent_key, parent_cell), (key, cell))| {
                            parent_key == key && parent_cell == *cell
                        })
            })
            .map_or_else(
                || {
                    Arc::from(
                        chunk
                            .iter()
                            .map(|(key, cell)| (*key, (*cell).clone()))
                            .collect::<Vec<_>>(),
                    )
                },
                Arc::clone,
            )
    }

    fn as_slice(&self) -> &[PlannerCellChunk] {
        match self {
            Self::Empty => &[],
            Self::Single(chunk) => std::slice::from_ref(chunk),
            Self::Multiple(chunks) => chunks,
        }
    }

    fn from_chunks(mut chunks: Vec<PlannerCellChunk>) -> Self {
        match chunks.len() {
            0 => Self::Empty,
            1 => Self::Single(chunks.pop().expect("one cell chunk disappeared")),
            _ => Self::Multiple(chunks),
        }
    }

    fn incrementally_from_cell_store(
        cells: &CellStore,
        parent: &Self,
        changed_cells: &[CellKey],
        earliest_structural_cell: Option<CellKey>,
    ) -> Option<(Self, usize)> {
        let parent_chunks = parent.as_slice();
        if let Some(earliest) = earliest_structural_cell {
            let insertion = parent_chunks
                .partition_point(|chunk| chunk.last().is_some_and(|(key, _)| *key < earliest));
            let start_chunk = insertion.min(parent_chunks.len().saturating_sub(1));
            let start_key = parent_chunks
                .get(start_chunk)
                .and_then(|chunk| chunk.first())
                .map_or(earliest, |(key, _)| *key);
            let mut chunks = parent_chunks[..start_chunk].to_vec();
            let mut scratch = Vec::with_capacity(PLANNER_CELL_CHUNK_LEN);
            let mut visited = 0usize;
            for (key, cell) in cells.range(start_key..CellKey(u64::MAX)) {
                scratch.push((*key, cell));
                if scratch.len() == PLANNER_CELL_CHUNK_LEN {
                    chunks.push(Self::reuse_or_copy_borrowed(
                        &scratch,
                        Some(parent),
                        chunks.len(),
                    ));
                    scratch.clear();
                    visited = visited.saturating_add(1);
                }
            }
            if !scratch.is_empty() {
                chunks.push(Self::reuse_or_copy_borrowed(
                    &scratch,
                    Some(parent),
                    chunks.len(),
                ));
                visited = visited.saturating_add(1);
            }
            return Some((Self::from_chunks(chunks), visited));
        }

        if parent_chunks.iter().map(|chunk| chunk.len()).sum::<usize>() != cells.len() {
            return None;
        }
        let mut changed_chunks = std::collections::BTreeSet::new();
        for key in changed_cells {
            let index = parent_chunks.partition_point(|chunk| {
                chunk.last().is_some_and(|(candidate, _)| candidate < key)
            });
            let chunk = parent_chunks.get(index)?;
            if chunk
                .binary_search_by_key(key, |(candidate, _)| *candidate)
                .is_err()
            {
                return None;
            }
            changed_chunks.insert(index);
        }
        let mut chunks = parent_chunks.to_vec();
        for index in &changed_chunks {
            let parent_chunk = &parent_chunks[*index];
            let mut scratch = Vec::with_capacity(parent_chunk.len());
            for (key, _) in parent_chunk.iter() {
                scratch.push((*key, cells.get(key)?));
            }
            chunks[*index] = Self::reuse_or_copy_borrowed(&scratch, Some(parent), *index);
        }
        Some((Self::from_chunks(chunks), changed_chunks.len()))
    }

    fn estimated_inline_heap_bytes_lower_bound(&self) -> usize {
        match self {
            Self::Multiple(chunks) => chunks
                .capacity()
                .saturating_mul(std::mem::size_of::<PlannerCellChunk>()),
            Self::Empty | Self::Single(_) => 0,
        }
    }
}

/// Trusted in-memory checkpoint that omits immutable compiled rules and shares
/// unchanged fixed-size tile and cell chunks with its parent. The owning engine
/// supplies and validates rules during restore; canonical replay checkpoints
/// remain self-contained `ReferenceCheckpoint`s.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct ReferenceStateCheckpoint {
    width: usize,
    height: usize,
    now: SimTime,
    tile_pages: Vec<PlannerTilePage>,
    cell_chunks: PlannerCellChunks,
    hash_seed: Option<Arc<PlannerHashSeed>>,
    next_cell_key: u64,
    semantic_ruleset_hash: CanonicalHash,
    compiled_ruleset_hash: CanonicalHash,
    state_hash: CanonicalHash,
    mutation_token: u64,
    #[cfg(test)]
    incremental_construction: bool,
    #[cfg(test)]
    visited_tile_chunks: usize,
    #[cfg(test)]
    visited_cell_chunks: usize,
}

/// Host-dependent phase timings for one trusted planner-state restoration.
/// These diagnostics never enter canonical state, replay, or verification.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferenceStateRestoreProfile {
    pub tile_materialization_ns: u64,
    pub cell_materialization_ns: u64,
    pub hash_seed_materialization_ns: u64,
    pub resolver_reconstruction_ns: u64,
    pub topology_validation_ns: u64,
    pub cell_validation_store_and_passive_index_ns: u64,
    pub tile_validation_and_passive_index_ns: u64,
    pub hash_initialization_ns: u64,
    pub scratch_initialization_ns: u64,
    pub metabolic_index_ns: u64,
    pub integrity_validation_ns: u64,
    pub total_ns: u64,
    pub tile_count: usize,
    pub cell_count: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl PartialEq for ReferenceStateCheckpoint {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.now == other.now
            && self.tile_pages == other.tile_pages
            && self.cell_chunks == other.cell_chunks
            && self.next_cell_key == other.next_cell_key
            && self.semantic_ruleset_hash == other.semantic_ruleset_hash
            && self.compiled_ruleset_hash == other.compiled_ruleset_hash
            && self.state_hash == other.state_hash
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Eq for ReferenceStateCheckpoint {}

#[cfg(not(target_arch = "wasm32"))]
impl ReferenceStateCheckpoint {
    pub(crate) fn from_simulation(simulation: &ReferenceSimulation, parent: Option<&Self>) -> Self {
        let width = simulation.neighborhood().width();
        let height = simulation.neighborhood().height();
        let semantic_ruleset_hash = simulation.semantic_ruleset_hash();
        let compiled_ruleset_hash = simulation.compiled_ruleset_hash();
        let state_hash = simulation.state_hash();
        let mutation_summary =
            simulation.checkpoint_mutation_summary(parent.map(|parent| parent.mutation_token));
        let compatible_parent = mutation_summary.compatible_parent
            && parent.is_some_and(|parent| {
                parent.width == width
                    && parent.height == height
                    && parent.semantic_ruleset_hash == semantic_ruleset_hash
                    && parent.compiled_ruleset_hash == compiled_ruleset_hash
            });
        let (tile_pages, visited_tile_chunks, tile_incremental) = if compatible_parent {
            Self::incremental_tile_pages(
                simulation.tiles(),
                parent.expect("compatible checkpoint parent disappeared"),
                &mutation_summary.tile_chunks,
            )
            .map(|(pages, visited)| (pages, visited, true))
            .unwrap_or_else(|| {
                let (pages, visited) = Self::full_tile_pages(simulation.tiles(), parent);
                (pages, visited, false)
            })
        } else {
            let (pages, visited) = Self::full_tile_pages(simulation.tiles(), parent);
            (pages, visited, false)
        };
        let (cell_chunks, visited_cell_chunks, cell_incremental) = if compatible_parent {
            PlannerCellChunks::incrementally_from_cell_store(
                simulation.cells(),
                &parent
                    .expect("compatible checkpoint parent disappeared")
                    .cell_chunks,
                &mutation_summary.cells,
                mutation_summary.earliest_structural_cell,
            )
            .map(|(chunks, visited)| (chunks, visited, true))
            .unwrap_or_else(|| {
                let chunks = PlannerCellChunks::from_cell_store(
                    simulation.cells(),
                    parent.map(|parent| &parent.cell_chunks),
                );
                let visited = chunks.as_slice().len();
                (chunks, visited, false)
            })
        } else {
            let chunks = PlannerCellChunks::from_cell_store(
                simulation.cells(),
                parent.map(|parent| &parent.cell_chunks),
            );
            let visited = chunks.as_slice().len();
            (chunks, visited, false)
        };
        let live_hash_seed = simulation.incremental_hash_seed();
        let hash_seed = (live_hash_seed.tile_page_count >= PLANNER_HASH_SEED_MIN_TILE_PAGES
            || live_hash_seed.cell_tree.page_count() >= 2)
            .then(|| {
                Arc::new(PlannerHashSeed::from_seed(
                    live_hash_seed,
                    compatible_parent
                        .then(|| parent.and_then(|parent| parent.hash_seed.as_deref()))
                        .flatten(),
                ))
            });
        #[cfg(not(test))]
        let _ = (
            visited_tile_chunks,
            tile_incremental,
            visited_cell_chunks,
            cell_incremental,
        );
        Self {
            width,
            height,
            now: simulation.now(),
            tile_pages,
            cell_chunks,
            hash_seed,
            next_cell_key: simulation.next_cell_key(),
            semantic_ruleset_hash,
            compiled_ruleset_hash,
            state_hash,
            mutation_token: mutation_summary.token,
            #[cfg(test)]
            incremental_construction: tile_incremental && cell_incremental,
            #[cfg(test)]
            visited_tile_chunks,
            #[cfg(test)]
            visited_cell_chunks,
        }
    }

    fn full_tile_pages(
        tiles: &ReferenceTileStore,
        parent: Option<&Self>,
    ) -> (Vec<PlannerTilePage>, usize) {
        let pages = tiles
            .pages()
            .iter()
            .enumerate()
            .map(|(page_index, live_page)| {
                let parent_page = parent.and_then(|parent| parent.tile_pages.get(page_index));
                if let Some(parent_page) = parent_page.filter(|page| {
                    Arc::ptr_eq(page, live_page) || page.as_ref() == live_page.as_ref()
                }) {
                    return Arc::clone(parent_page);
                }
                Arc::clone(live_page)
            })
            .collect::<Vec<_>>();
        (pages, tiles.len().div_ceil(PLANNER_TILE_CHUNK_LEN))
    }

    fn incremental_tile_pages(
        tiles: &ReferenceTileStore,
        parent: &Self,
        changed_chunks: &[usize],
    ) -> Option<(Vec<PlannerTilePage>, usize)> {
        let expected_pages = tiles
            .len()
            .div_ceil(PLANNER_TILE_CHUNK_LEN * PLANNER_TILE_PAGE_CHUNKS);
        if parent.tile_pages.len() != expected_pages {
            return None;
        }
        let total_chunks = tiles.len().div_ceil(PLANNER_TILE_CHUNK_LEN);
        if changed_chunks
            .last()
            .is_some_and(|index| *index >= total_chunks)
        {
            return None;
        }
        let mut pages = parent.tile_pages.clone();
        let mut cursor = 0usize;
        while cursor < changed_chunks.len() {
            let page_index = changed_chunks[cursor] / PLANNER_TILE_PAGE_CHUNKS;
            let parent_page = parent.tile_pages.get(page_index)?;
            let page_tile_start = page_index
                .saturating_mul(PLANNER_TILE_PAGE_CHUNKS)
                .saturating_mul(PLANNER_TILE_CHUNK_LEN);
            let page_tile_len = tiles
                .len()
                .saturating_sub(page_tile_start)
                .min(PLANNER_TILE_CHUNK_LEN * PLANNER_TILE_PAGE_CHUNKS);
            if parent_page.len() != page_tile_len {
                return None;
            }
            let live_page = tiles.pages().get(page_index)?;
            while cursor < changed_chunks.len()
                && changed_chunks[cursor] / PLANNER_TILE_PAGE_CHUNKS == page_index
            {
                cursor += 1;
            }
            if !Arc::ptr_eq(parent_page, live_page) && parent_page.as_ref() != live_page.as_ref() {
                pages[page_index] = Arc::clone(live_page);
            }
        }
        Some((pages, changed_chunks.len()))
    }

    #[cfg(test)]
    pub(crate) fn into_simulation_profiled(
        self,
        rules: ReferenceRuleset,
        expected_semantic_ruleset_hash: CanonicalHash,
        expected_compiled_ruleset_hash: CanonicalHash,
    ) -> Result<(ReferenceSimulation, ReferenceStateRestoreProfile), CheckpointError> {
        self.into_simulation_profiled_inner(
            rules,
            expected_semantic_ruleset_hash,
            expected_compiled_ruleset_hash,
            None,
        )
    }

    pub(crate) fn into_simulation_profiled_with_topology(
        self,
        rules: ReferenceRuleset,
        expected_semantic_ruleset_hash: CanonicalHash,
        expected_compiled_ruleset_hash: CanonicalHash,
        topology: ReferenceCompiledTopology,
    ) -> Result<(ReferenceSimulation, ReferenceStateRestoreProfile), CheckpointError> {
        self.into_simulation_profiled_inner(
            rules,
            expected_semantic_ruleset_hash,
            expected_compiled_ruleset_hash,
            Some(topology),
        )
    }

    fn into_simulation_profiled_inner(
        self,
        rules: ReferenceRuleset,
        expected_semantic_ruleset_hash: CanonicalHash,
        expected_compiled_ruleset_hash: CanonicalHash,
        topology: Option<ReferenceCompiledTopology>,
    ) -> Result<(ReferenceSimulation, ReferenceStateRestoreProfile), CheckpointError> {
        let total_started = std::time::Instant::now();
        if self.semantic_ruleset_hash != expected_semantic_ruleset_hash {
            return Err(CheckpointError::SemanticRulesetHashMismatch);
        }
        if self.compiled_ruleset_hash != expected_compiled_ruleset_hash {
            return Err(CheckpointError::CompiledRulesetHashMismatch);
        }

        let cell_started = std::time::Instant::now();
        let cells = self.materialize_cells();
        let cell_materialization_ns = elapsed_ns(cell_started);
        let tile_count = self.width.saturating_mul(self.height);
        let cell_count = cells.len();

        let (hash_seed, hash_seed_materialization_ns) = if topology.is_some() {
            let started = std::time::Instant::now();
            let seed = self.hash_seed.as_deref().map(PlannerHashSeed::materialize);
            let elapsed = seed.as_ref().map_or(0, |_| elapsed_ns(started));
            (seed, elapsed)
        } else {
            (None, 0)
        };

        let reconstruction_started = std::time::Instant::now();
        let (mut simulation, reconstruction_profile, tile_materialization_ns) = if let Some(
            topology,
        ) = topology
        {
            if topology.width() != self.width || topology.height() != self.height {
                return Err(CheckpointError::CompiledRulesetHashMismatch);
            }
            let tile_started = std::time::Instant::now();
            let tiles = ReferenceTileStore::from_pages(self.tile_pages, tile_count)
                .ok_or(CheckpointError::InvalidField("planner tile page geometry"))?;
            let tile_materialization_ns = elapsed_ns(tile_started);
            let (simulation, profile) =
                ReferenceSimulation::from_canonical_tile_store_with_compiled_topology_and_hash_seed_profiled(
                    rules,
                    self.now,
                    tiles,
                    cells,
                    self.next_cell_key,
                    topology,
                    hash_seed,
                )?;
            (simulation, profile, tile_materialization_ns)
        } else {
            let tile_started = std::time::Instant::now();
            let tiles = self.materialize_tiles();
            let tile_materialization_ns = elapsed_ns(tile_started);
            let state = SimulationState {
                now: self.now,
                tiles,
                cells,
                next_cell_key: self.next_cell_key,
            };
            let simulation =
                ReferenceSimulation::from_canonical_state(self.width, self.height, rules, state)?;
            (simulation, Default::default(), tile_materialization_ns)
        };
        let resolver_reconstruction_ns = elapsed_ns(reconstruction_started);
        let validation_started = std::time::Instant::now();
        if simulation.state_hash() != self.state_hash {
            return Err(CheckpointError::StateHashMismatch);
        }
        let integrity_validation_ns = elapsed_ns(validation_started);
        simulation.adopt_checkpoint_mutation_token(self.mutation_token);
        Ok((
            simulation,
            ReferenceStateRestoreProfile {
                tile_materialization_ns,
                cell_materialization_ns,
                hash_seed_materialization_ns,
                resolver_reconstruction_ns,
                topology_validation_ns: reconstruction_profile.topology_validation_ns,
                cell_validation_store_and_passive_index_ns: reconstruction_profile
                    .cell_validation_store_and_passive_index_ns,
                tile_validation_and_passive_index_ns: reconstruction_profile
                    .tile_validation_and_passive_index_ns,
                hash_initialization_ns: reconstruction_profile.hash_initialization_ns,
                scratch_initialization_ns: reconstruction_profile.scratch_initialization_ns,
                metabolic_index_ns: reconstruction_profile.metabolic_index_ns,
                integrity_validation_ns,
                total_ns: elapsed_ns(total_started),
                tile_count,
                cell_count,
            },
        ))
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn semantic_ruleset_hash(&self) -> CanonicalHash {
        self.semantic_ruleset_hash
    }

    pub const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub const fn state_hash(&self) -> CanonicalHash {
        self.state_hash
    }

    pub fn estimated_retained_bytes_lower_bound(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.estimated_inline_heap_bytes_lower_bound())
            .saturating_add(
                self.tile_pages
                    .iter()
                    .map(|page| {
                        std::mem::size_of_val(page.as_ref())
                            .saturating_add(page.inline_heap_bytes())
                            .saturating_add(
                                page.len().saturating_mul(std::mem::size_of::<TileState>()),
                            )
                            .saturating_add(page.overlay_allocation_bytes())
                    })
                    .sum::<usize>(),
            )
            .saturating_add(
                self.cell_chunks
                    .as_slice()
                    .iter()
                    .map(|chunk| {
                        chunk
                            .len()
                            .saturating_mul(std::mem::size_of::<(CellKey, CellState)>())
                    })
                    .sum::<usize>(),
            )
            .saturating_add(
                self.hash_seed
                    .as_deref()
                    .map_or(0, PlannerHashSeed::allocation_bytes),
            )
    }

    pub fn estimated_inline_heap_bytes_lower_bound(&self) -> usize {
        self.tile_pages
            .capacity()
            .saturating_mul(std::mem::size_of::<PlannerTilePage>())
            .saturating_add(self.cell_chunks.estimated_inline_heap_bytes_lower_bound())
            .saturating_add(self.hash_seed.as_deref().map_or(0, |seed| {
                std::mem::size_of::<PlannerHashSeed>().saturating_add(seed.inline_heap_bytes())
            }))
    }

    pub fn tile_page_allocations(&self) -> impl Iterator<Item = (*const (), usize)> + '_ {
        self.tile_pages
            .iter()
            .map(|page| (Arc::as_ptr(page).cast::<()>(), page.chunk_count()))
    }

    pub fn tile_page_count(&self) -> usize {
        self.tile_pages.len()
    }

    pub fn tile_chunk_allocations(&self) -> impl Iterator<Item = (*const TileState, usize)> + '_ {
        self.tile_pages
            .iter()
            .flat_map(|page| (0..page.chunk_count()).map(|index| page.chunk_allocation(index)))
    }

    pub fn tile_chunk_allocations_in_page(
        &self,
        page_index: usize,
    ) -> impl Iterator<Item = (*const TileState, usize)> + '_ {
        self.tile_pages
            .get(page_index)
            .into_iter()
            .flat_map(|page| (0..page.chunk_count()).map(|index| page.chunk_allocation(index)))
    }

    pub fn tile_chunk_count(&self) -> usize {
        self.tile_pages.iter().map(|page| page.chunk_count()).sum()
    }

    pub fn shared_tile_page_count(&self, parent: &Self) -> usize {
        self.tile_pages
            .iter()
            .zip(&parent.tile_pages)
            .filter(|(left, right)| Arc::ptr_eq(left, right))
            .count()
    }

    pub fn shared_tile_chunk_count(&self, parent: &Self) -> usize {
        self.tile_pages
            .iter()
            .zip(&parent.tile_pages)
            .map(|(left, right)| {
                (0..left.chunk_count().min(right.chunk_count()))
                    .filter(|index| left.shares_chunk_with(right, *index))
                    .count()
            })
            .sum()
    }

    pub fn cell_chunk_allocations(
        &self,
    ) -> impl Iterator<Item = (*const (CellKey, CellState), usize)> + '_ {
        self.cell_chunks
            .as_slice()
            .iter()
            .map(|chunk| (chunk.as_ptr(), chunk.len()))
    }

    pub fn cell_chunk_count(&self) -> usize {
        self.cell_chunks.as_slice().len()
    }

    pub fn shared_cell_chunk_count(&self, parent: &Self) -> usize {
        self.cell_chunks
            .as_slice()
            .iter()
            .zip(parent.cell_chunks.as_slice())
            .filter(|(left, right)| Arc::ptr_eq(left, right))
            .count()
    }

    pub fn hash_seed_chunk_allocations(&self) -> Vec<(*const u8, usize)> {
        self.hash_seed
            .as_deref()
            .map_or_else(Vec::new, PlannerHashSeed::allocations)
    }

    pub fn hash_seed_chunk_count(&self) -> usize {
        self.hash_seed
            .as_deref()
            .map_or(0, PlannerHashSeed::chunk_count)
    }

    pub fn shared_hash_seed_chunk_count(&self, parent: &Self) -> usize {
        self.hash_seed
            .as_deref()
            .zip(parent.hash_seed.as_deref())
            .map_or(0, |(seed, parent)| seed.shared_chunk_count(parent))
    }

    #[cfg(test)]
    pub(crate) const fn used_incremental_construction(&self) -> bool {
        self.incremental_construction
    }

    #[cfg(test)]
    pub(crate) const fn visited_tile_chunks(&self) -> usize {
        self.visited_tile_chunks
    }

    #[cfg(test)]
    pub(crate) const fn visited_cell_chunks(&self) -> usize {
        self.visited_cell_chunks
    }

    pub(crate) const fn next_cell_key(&self) -> u64 {
        self.next_cell_key
    }

    fn materialize_tiles(&self) -> Vec<TileState> {
        let tile_count = self.width.saturating_mul(self.height);
        if tile_count >= PLANNER_PARALLEL_TILE_MATERIALIZATION_THRESHOLD
            && rayon::current_num_threads() > 1
        {
            return self.materialize_tiles_parallel();
        }
        self.materialize_tiles_serial()
    }

    fn materialize_tiles_serial(&self) -> Vec<TileState> {
        let mut tiles = Vec::with_capacity(self.width.saturating_mul(self.height));
        for page in &self.tile_pages {
            tiles.extend(page.iter().cloned());
        }
        tiles
    }

    fn materialize_tiles_parallel(&self) -> Vec<TileState> {
        let tile_count = self.width.saturating_mul(self.height);
        let page_tile_len = PLANNER_TILE_CHUNK_LEN * PLANNER_TILE_PAGE_CHUNKS;
        (0..tile_count)
            .into_par_iter()
            .map(|tile_index| {
                let page_offset = tile_index % page_tile_len;
                self.tile_pages[tile_index / page_tile_len]
                    .get(page_offset)
                    .clone()
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn materialize_tiles_for_test(&self, parallel: bool) -> Vec<TileState> {
        if parallel {
            self.materialize_tiles_parallel()
        } else {
            self.materialize_tiles_serial()
        }
    }

    fn materialize_cells(&self) -> Vec<(CellKey, CellState)> {
        let mut cells = Vec::with_capacity(
            self.cell_chunks
                .as_slice()
                .iter()
                .map(|chunk| chunk.len())
                .sum(),
        );
        for chunk in self.cell_chunks.as_slice() {
            cells.extend(chunk.iter().cloned());
        }
        cells
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

impl ReferenceCheckpoint {
    pub fn from_simulation(simulation: &ReferenceSimulation) -> Self {
        let width = simulation.neighborhood().width();
        let height = simulation.neighborhood().height();
        let rules = simulation.rules().clone();
        let state = simulation.canonical_state();
        let semantic_ruleset_hash = simulation.semantic_ruleset_hash();
        let compiled_ruleset_hash = simulation.compiled_ruleset_hash();
        let state_hash = simulation.state_hash();
        let mut checkpoint = Self {
            width,
            height,
            rules,
            state,
            semantic_ruleset_hash,
            compiled_ruleset_hash,
            state_hash,
            checkpoint_hash: CanonicalHash::from_bytes([0; 32]),
        };
        checkpoint.checkpoint_hash = hash_checkpoint_body(&checkpoint.encode_body());
        checkpoint
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CheckpointError> {
        Self::from_bytes_with_limits(bytes, CheckpointLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: CheckpointLimits,
    ) -> Result<Self, CheckpointError> {
        if bytes.len() > limits.max_checkpoint_bytes {
            return Err(CheckpointError::TooLarge {
                actual: bytes.len(),
                limit: limits.max_checkpoint_bytes,
            });
        }
        if bytes.len() < 32 {
            return Err(CheckpointError::Truncated);
        }
        let body_length = bytes.len() - 32;
        let (body, encoded_hash) = bytes.split_at(body_length);
        let expected = CanonicalHash::from_bytes(
            encoded_hash
                .try_into()
                .map_err(|_| CheckpointError::Truncated)?,
        );
        let actual = hash_checkpoint_body(body);
        if actual != expected {
            return Err(CheckpointError::IntegrityHashMismatch { expected, actual });
        }

        let mut reader = Reader::new(body);
        if reader.take(CHECKPOINT_MAGIC.len())? != CHECKPOINT_MAGIC {
            return Err(CheckpointError::InvalidMagic);
        }
        let version = reader.u16()?;
        if version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::UnsupportedVersion(version));
        }
        let width = read_usize(&mut reader, "width")?;
        let height = read_usize(&mut reader, "height")?;
        let area = width
            .checked_mul(height)
            .filter(|area| *area > 0 && *area <= limits.max_tiles)
            .ok_or(CheckpointError::InvalidField(
                "world dimensions exceed tile limit",
            ))?;
        let semantic_ruleset_hash = reader.hash()?;
        let compiled_ruleset_hash = reader.hash()?;
        let state_hash = reader.hash()?;
        let rules = decode_ruleset(&mut reader)?;
        let state = decode_state(&mut reader, limits)?;
        if state.tiles.len() != area {
            return Err(CheckpointError::InvalidField(
                "tile count does not match dimensions",
            ));
        }
        if reader.remaining() != 0 {
            return Err(CheckpointError::TrailingBytes(reader.remaining()));
        }

        let simulation =
            ReferenceSimulation::from_canonical_state(width, height, rules.clone(), state.clone())?;
        if simulation.semantic_ruleset_hash() != semantic_ruleset_hash {
            return Err(CheckpointError::SemanticRulesetHashMismatch);
        }
        if simulation.compiled_ruleset_hash() != compiled_ruleset_hash {
            return Err(CheckpointError::CompiledRulesetHashMismatch);
        }
        if simulation.state_hash() != state_hash {
            return Err(CheckpointError::StateHashMismatch);
        }

        Ok(Self {
            width,
            height,
            rules,
            state,
            semantic_ruleset_hash,
            compiled_ruleset_hash,
            state_hash,
            checkpoint_hash: expected,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.encode_body();
        bytes.extend_from_slice(self.checkpoint_hash.as_bytes());
        bytes
    }

    pub fn into_simulation(self) -> Result<ReferenceSimulation, ResolutionError> {
        ReferenceSimulation::from_canonical_state(self.width, self.height, self.rules, self.state)
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn rules(&self) -> &ReferenceRuleset {
        &self.rules
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub const fn semantic_ruleset_hash(&self) -> CanonicalHash {
        self.semantic_ruleset_hash
    }

    pub const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub const fn state_hash(&self) -> CanonicalHash {
        self.state_hash
    }

    pub const fn checkpoint_hash(&self) -> CanonicalHash {
        self.checkpoint_hash
    }

    fn encode_body(&self) -> Vec<u8> {
        let mut state_writer = Writer::new();
        encode_state(&mut state_writer, &self.state);
        encode_checkpoint_body(
            self.width,
            self.height,
            self.semantic_ruleset_hash,
            self.compiled_ruleset_hash,
            self.state_hash,
            &self.rules,
            &state_writer.finish(),
        )
    }
}

fn encode_checkpoint_body(
    width: usize,
    height: usize,
    semantic_ruleset_hash: CanonicalHash,
    compiled_ruleset_hash: CanonicalHash,
    state_hash: CanonicalHash,
    rules: &ReferenceRuleset,
    state_bytes: &[u8],
) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.raw(CHECKPOINT_MAGIC);
    writer.u16(CHECKPOINT_FORMAT_VERSION);
    writer.u64(width as u64);
    writer.u64(height as u64);
    writer.hash(semantic_ruleset_hash);
    writer.hash(compiled_ruleset_hash);
    writer.hash(state_hash);
    encode_ruleset(&mut writer, rules);
    writer.raw(state_bytes);
    writer.finish()
}

fn hash_checkpoint_body(body: &[u8]) -> CanonicalHash {
    let mut hasher = Sha256::new();
    hasher.update((CHECKPOINT_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(CHECKPOINT_DOMAIN);
    hasher.update(CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    hasher.update((body.len() as u64).to_le_bytes());
    hasher.update(body);
    CanonicalHash::from_bytes(hasher.finalize().into())
}

fn encode_ruleset(writer: &mut Writer, rules: &ReferenceRuleset) {
    encode_neighborhood(writer, &rules.neighborhood);
    writer.u64(rules.time.arithmetic_quanta_per_unit);
    writer.u64(rules.time.completion_bucket);
    writer.u64(rules.time.decision_interval_floor);
    writer.u64(rules.effort_profiles.len() as u64);
    for profile in rules.effort_profiles {
        writer.u32(profile.cost_numerator);
        writer.u32(profile.cost_denominator);
        writer.u32(profile.duration_numerator);
        writer.u32(profile.duration_denominator);
    }
    for duration in [
        rules.wait_duration,
        rules.move_duration,
        rules.attack_duration,
        rules.consume_duration,
        rules.split_duration,
        rules.regurgitate_duration,
        rules.excavate_duration,
        rules.deposit_terrain_duration,
    ] {
        writer.u64(duration.base_quanta);
        writer.u64(duration.mass_quanta_numerator);
        writer.u64(duration.mass_units_denominator);
    }
    for value in [
        rules.move_effort_base,
        rules.move_mass_units_per_effort,
        rules.attack_effort_base,
        rules.guard_effort_base,
        rules.consume_effort_base,
        rules.split_effort_base,
        rules.regurgitate_effort_base,
        rules.excavate_effort_base,
        rules.deposit_terrain_effort_base,
        rules.bite_capacity,
        rules.gut_capacity,
        rules.digestion_rate_numerator,
        rules.digestion_rate_denominator,
        rules.metabolism_rate_numerator,
        rules.metabolism_rate_denominator,
        rules.signal_emission_cost,
        rules.signal_decay_rate_numerator,
        rules.signal_decay_rate_denominator,
        rules.diffusion_interval_quanta,
        rules.diffusion_rate_numerator,
        rules.diffusion_rate_denominator,
        rules.terrain_mass_per_elevation,
    ] {
        writer.u64(value);
    }
    writer.u32(rules.diffusion_targets.bits());
    writer.u64(rules.minimum_survival_energy);
    writer.u64(rules.child_core_mass);
    writer.i16(rules.maximum_elevation_delta);
    for value in [
        rules.guard_damage_numerator,
        rules.guard_damage_denominator,
        rules.attack_payload_damage_numerator,
        rules.attack_payload_damage_denominator,
        rules.attack_mass_damage_numerator,
        rules.attack_mass_damage_denominator,
        rules.apparent_mass_bucket_width,
    ] {
        writer.u64(value);
    }
    writer.u64(rules.max_private_memory_bytes as u64);
}

fn decode_ruleset(reader: &mut Reader<'_>) -> Result<ReferenceRuleset, CheckpointError> {
    let neighborhood = decode_neighborhood(reader)?;
    let time = TimeConfig {
        arithmetic_quanta_per_unit: reader.u64()?,
        completion_bucket: reader.u64()?,
        decision_interval_floor: reader.u64()?,
    };
    if reader.u64()? != 3 {
        return Err(CheckpointError::InvalidField("effort profile count"));
    }
    let mut effort_profiles = [EffortProfile::new(0, 1, 0, 1); 3];
    for profile in &mut effort_profiles {
        *profile = EffortProfile::new(reader.u32()?, reader.u32()?, reader.u32()?, reader.u32()?);
    }
    let mut duration = || -> Result<DurationRule, CheckpointError> {
        Ok(DurationRule::new(
            reader.u64()?,
            reader.u64()?,
            reader.u64()?,
        ))
    };
    let wait_duration = duration()?;
    let move_duration = duration()?;
    let attack_duration = duration()?;
    let consume_duration = duration()?;
    let split_duration = duration()?;
    let regurgitate_duration = duration()?;
    let excavate_duration = duration()?;
    let deposit_terrain_duration = duration()?;

    Ok(ReferenceRuleset {
        neighborhood,
        time,
        effort_profiles,
        wait_duration,
        move_duration,
        attack_duration,
        consume_duration,
        split_duration,
        regurgitate_duration,
        excavate_duration,
        deposit_terrain_duration,
        move_effort_base: reader.u64()?,
        move_mass_units_per_effort: reader.u64()?,
        attack_effort_base: reader.u64()?,
        guard_effort_base: reader.u64()?,
        consume_effort_base: reader.u64()?,
        split_effort_base: reader.u64()?,
        regurgitate_effort_base: reader.u64()?,
        excavate_effort_base: reader.u64()?,
        deposit_terrain_effort_base: reader.u64()?,
        bite_capacity: reader.u64()?,
        gut_capacity: reader.u64()?,
        digestion_rate_numerator: reader.u64()?,
        digestion_rate_denominator: reader.u64()?,
        metabolism_rate_numerator: reader.u64()?,
        metabolism_rate_denominator: reader.u64()?,
        signal_emission_cost: reader.u64()?,
        signal_decay_rate_numerator: reader.u64()?,
        signal_decay_rate_denominator: reader.u64()?,
        diffusion_interval_quanta: reader.u64()?,
        diffusion_rate_numerator: reader.u64()?,
        diffusion_rate_denominator: reader.u64()?,
        terrain_mass_per_elevation: reader.u64()?,
        diffusion_targets: slot_mask_from_bits(reader.u32()?),
        minimum_survival_energy: reader.u64()?,
        child_core_mass: reader.u64()?,
        maximum_elevation_delta: reader.i16()?,
        guard_damage_numerator: reader.u64()?,
        guard_damage_denominator: reader.u64()?,
        attack_payload_damage_numerator: reader.u64()?,
        attack_payload_damage_denominator: reader.u64()?,
        attack_mass_damage_numerator: reader.u64()?,
        attack_mass_damage_denominator: reader.u64()?,
        apparent_mass_bucket_width: reader.u64()?,
        max_private_memory_bytes: read_usize(reader, "maximum private memory")?,
    })
}

fn encode_neighborhood(writer: &mut Writer, neighborhood: &NeighborhoodSpec) {
    writer.u64(neighborhood.slots.len() as u64);
    for offset in &neighborhood.slots {
        writer.i8(offset.dx);
        writer.i8(offset.dy);
        writer.u16(offset.distance_cost_q10);
    }
    for mask in [
        neighborhood.observations.occupancy,
        neighborhood.observations.marker,
        neighborhood.observations.apparent_mass,
        neighborhood.observations.activity,
        neighborhood.observations.terrain,
        neighborhood.observations.energy,
        neighborhood.observations.signal,
        neighborhood.target_mask(TargetingAction::Move),
        neighborhood.target_mask(TargetingAction::Attack),
        neighborhood.target_mask(TargetingAction::Split),
        neighborhood.target_mask(TargetingAction::Regurgitate),
    ] {
        writer.u32(mask.bits());
    }
    writer.u8(match neighborhood.diagonal_corner_rule {
        DiagonalCornerRule::Allow => 0,
        DiagonalCornerRule::BlockIfEitherOrthogonalOccupied => 1,
        DiagonalCornerRule::BlockIfBothOrthogonalsOccupied => 2,
    });
    writer.u8(match neighborhood.boundary_rule {
        BoundaryRule::Bounded => 0,
        BoundaryRule::Wrap => 1,
    });
    writer.u8(neighborhood.max_radius);
}

fn decode_neighborhood(reader: &mut Reader<'_>) -> Result<NeighborhoodSpec, CheckpointError> {
    let slot_count = reader.u64()?;
    if slot_count > MAX_LOCAL_SLOTS as u64 {
        return Err(CheckpointError::InvalidField("neighborhood slot count"));
    }
    let mut slots = Vec::with_capacity(slot_count as usize);
    for _ in 0..slot_count {
        slots.push(LocalOffset::new(reader.i8()?, reader.i8()?, reader.u16()?));
    }
    let masks: Vec<SlotMask> = (0..11)
        .map(|_| reader.u32().map(slot_mask_from_bits))
        .collect::<Result<_, _>>()?;
    let diagonal_corner_rule = match reader.u8()? {
        0 => DiagonalCornerRule::Allow,
        1 => DiagonalCornerRule::BlockIfEitherOrthogonalOccupied,
        2 => DiagonalCornerRule::BlockIfBothOrthogonalsOccupied,
        _ => return Err(CheckpointError::InvalidField("diagonal corner rule")),
    };
    let boundary_rule = match reader.u8()? {
        0 => BoundaryRule::Bounded,
        1 => BoundaryRule::Wrap,
        _ => return Err(CheckpointError::InvalidField("boundary rule")),
    };
    Ok(NeighborhoodSpec::new(
        slots,
        ObservationMasks {
            occupancy: masks[0],
            marker: masks[1],
            apparent_mass: masks[2],
            activity: masks[3],
            terrain: masks[4],
            energy: masks[5],
            signal: masks[6],
        },
        [masks[7], masks[8], masks[9], masks[10]],
        diagonal_corner_rule,
        boundary_rule,
        reader.u8()?,
    ))
}

fn slot_mask_from_bits(bits: u32) -> SlotMask {
    SlotMask::from_slots(
        (0..MAX_LOCAL_SLOTS)
            .filter(|slot| bits & (1_u32 << slot) != 0)
            .map(|slot| super::LocalSlot(slot as u8)),
    )
}

fn encode_state(writer: &mut Writer, state: &SimulationState) {
    writer.u64(state.now.0);
    writer.u64(state.next_cell_key);
    writer.vec(&state.tiles, encode_tile_state);
    writer.vec(&state.cells, |writer, (key, cell)| {
        writer.u64(key.0);
        encode_cell_state(writer, cell);
    });
}

fn decode_state(
    reader: &mut Reader<'_>,
    limits: CheckpointLimits,
) -> Result<SimulationState, CheckpointError> {
    let now = SimTime(reader.u64()?);
    let next_cell_key = reader.u64()?;
    if u128::from(next_cell_key) > limits.max_cell_slots as u128 {
        return Err(CheckpointError::InvalidField(
            "cell key space exceeds slot limit",
        ));
    }
    let tiles = decode_vec(
        reader,
        "checkpoint tiles",
        limits.max_tiles,
        decode_tile_state,
    )?;
    let cells = decode_vec(reader, "checkpoint cells", limits.max_cells, |reader| {
        let key = reader.u64()?;
        if key >= next_cell_key {
            return Err(ReplayError::InvalidBatch(
                "cell key exceeds allocated key space",
            ));
        }
        Ok((
            CellKey(key),
            decode_cell_state(reader, limits.max_private_memory_bytes)?,
        ))
    })?;
    Ok(SimulationState {
        now,
        tiles,
        cells,
        next_cell_key,
    })
}

fn read_usize(reader: &mut Reader<'_>, field: &'static str) -> Result<usize, CheckpointError> {
    usize::try_from(reader.u64()?).map_err(|_| CheckpointError::InvalidField(field))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod trusted_hash_tests {
    use super::*;
    use crate::resolution::{ActionRequest, TileIndex};

    fn historical_simulation(pages: u64) -> ReferenceSimulation {
        let rules = ReferenceRuleset {
            metabolism_rate_numerator: 0,
            diffusion_rate_numerator: 0,
            ..ReferenceRuleset::default()
        };
        let mut simulation = ReferenceSimulation::new(4, 4, rules.clone()).unwrap();
        simulation.add_cell(TileIndex(0), 10, 100, 1).unwrap();
        simulation.add_cell(TileIndex(1), 10, 100, 2).unwrap();
        let mut state = simulation.canonical_state();
        state.next_cell_key = pages * super::super::hashing::STATE_CELL_PAGE_SIZE;
        state.cells[1].0 = CellKey(state.next_cell_key - 1);
        state.tiles[1].occupant = Some(state.cells[1].0);
        ReferenceSimulation::from_canonical_state(4, 4, rules, state).unwrap()
    }

    fn restore(
        checkpoint: &ReferenceStateCheckpoint,
        simulation: &ReferenceSimulation,
    ) -> Result<ReferenceSimulation, CheckpointError> {
        checkpoint
            .clone()
            .into_simulation_profiled_with_topology(
                simulation.rules().clone(),
                simulation.semantic_ruleset_hash(),
                simulation.compiled_ruleset_hash(),
                simulation.compiled_topology(),
            )
            .map(|(simulation, _)| simulation)
    }

    #[test]
    fn historical_small_world_checkpoints_share_hashes_and_isolate_branches() {
        let mut simulation = historical_simulation(4097);
        let root = ReferenceStateCheckpoint::from_simulation(&simulation, None);
        assert!(
            root.hash_seed.is_some(),
            "cell history needs a seed even on small boards"
        );
        let restored = restore(&root, &simulation).unwrap();
        assert_eq!(restored.canonical_state(), simulation.canonical_state());
        let roundtrip = ReferenceStateCheckpoint::from_simulation(&restored, Some(&root));
        assert_eq!(
            roundtrip.shared_hash_seed_chunk_count(&root),
            root.hash_seed_chunk_count()
        );
        assert_eq!(
            root.hash_seed_chunk_allocations().len(),
            root.hash_seed_chunk_count()
        );
        assert!(root.hash_seed_chunk_count() < 64);
        assert!(root.hash_seed.as_ref().unwrap().allocation_bytes() < 8192);

        simulation
            .commit_decision(CellKey(0), ActionRequest::Wait, vec![1, 2, 3])
            .unwrap();
        let child = ReferenceStateCheckpoint::from_simulation(&simulation, Some(&root));
        assert!(child.shared_hash_seed_chunk_count(&root) > 0);
        let mut child_restored = restore(&child, &simulation).unwrap();
        assert_eq!(
            child_restored.state_hash(),
            child_restored
                .canonical_state()
                .hash_with_compiled_ruleset(child_restored.compiled_ruleset_hash())
        );
        assert_eq!(
            child_restored.resolve_next_batch().unwrap(),
            simulation.resolve_next_batch().unwrap()
        );
        let root_restored = restore(&root, &simulation).unwrap();
        assert_eq!(root_restored.state_hash(), root.state_hash());
        assert_ne!(root_restored.state_hash(), child_restored.state_hash());

        // A stale trusted tree must not hide modified canonical cell content.
        let mut corrupted = root.clone();
        let mut cells = corrupted.materialize_cells();
        cells[0].1.assimilated_energy += 1;
        corrupted.cell_chunks = PlannerCellChunks::Single(Arc::from(cells));
        assert!(matches!(
            restore(&corrupted, &simulation),
            Err(CheckpointError::StateHashMismatch)
        ));
    }

    #[test]
    #[ignore = "manual release comparison of seeded versus cold historical checkpoint restore"]
    fn benchmark_historical_checkpoint_restore() {
        let simulation = historical_simulation(65_536);
        let checkpoint = ReferenceStateCheckpoint::from_simulation(&simulation, None);
        let mut cold = checkpoint.clone();
        cold.hash_seed = None;
        let mut times = Vec::new();
        for (label, checkpoint) in [("shared", &checkpoint), ("cold", &cold)] {
            let started = std::time::Instant::now();
            for _ in 0..64 {
                let restored = restore(checkpoint, &simulation).unwrap();
                assert_eq!(restored.state_hash(), simulation.state_hash());
                std::hint::black_box(restored);
            }
            times.push((label, started.elapsed()));
        }
        println!("64 restores, 65,536 historical cell pages, 2 live cells, 4x4 board: shared {:.3} ms, cold {:.3} ms; retained hash allocations {} bytes", times[0].1.as_secs_f64()*1000.0, times[1].1.as_secs_f64()*1000.0, checkpoint.hash_seed.as_ref().unwrap().allocation_bytes());
    }
}
