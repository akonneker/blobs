//! Versioned, platform-independent hashes for replay and verification.
//!
//! Authoritative values are encoded explicitly in little-endian order. Enum
//! tags are assigned here rather than inherited from Rust discriminants, and
//! collection lengths use `u64`, so native and WASM builds share one byte
//! contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

#[cfg(target_arch = "wasm32")]
use sha2::{Digest, Sha256};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::neighborhood::{
    BoundaryRule, CompiledNeighborhood, DiagonalCornerRule, NeighborhoodSpec, TargetingAction,
    TileIndex,
};
use super::reference::{
    ActionRequest, CellKey, CellState, CellStore, DurationRule, EffortProfile, EffortTier,
    OutcomeStatus, PendingAction, ReferenceRuleset, ReferenceTileStore, RejectReason, SimTime,
    TileState, TimeConfig,
};

pub(crate) trait TileSource: Sync {
    fn tile_len(&self) -> usize;
    fn tile_at(&self, index: usize) -> Option<&TileState>;
}

impl TileSource for [TileState] {
    fn tile_len(&self) -> usize {
        self.len()
    }

    fn tile_at(&self, index: usize) -> Option<&TileState> {
        self.get(index)
    }
}

impl TileSource for Vec<TileState> {
    fn tile_len(&self) -> usize {
        self.len()
    }

    fn tile_at(&self, index: usize) -> Option<&TileState> {
        self.get(index)
    }
}

impl TileSource for ReferenceTileStore {
    fn tile_len(&self) -> usize {
        self.len()
    }

    fn tile_at(&self, index: usize) -> Option<&TileState> {
        self.get(index)
    }
}

pub const CANONICAL_HASH_FORMAT_VERSION: u16 = 7;
pub const REFERENCE_SEMANTIC_KERNEL_VERSION: u16 = 5;
pub const CANONICAL_HASH_ALGORITHM: &str = "sha256";

const SEMANTIC_RULESET_DOMAIN: &[u8] = b"blob.ruleset.semantic";
const COMPILED_RULESET_DOMAIN: &[u8] = b"blob.ruleset.compiled";
const STATE_DOMAIN: &[u8] = b"blob.simulation.state";
const STATE_TILE_PAGE_DOMAIN: &[u8] = b"blob.simulation.state.tile-page";
const STATE_CELL_PAGE_DOMAIN: &[u8] = b"blob.simulation.state.cell-page";
const STATE_CELL_LEAF_DOMAIN: &[u8] = b"blob.simulation.state.cell-leaf";
const STATE_CELL_MEMORY_DOMAIN: &[u8] = b"blob.simulation.state.cell-memory";
const STATE_MERKLE_EMPTY_DOMAIN: &[u8] = b"blob.simulation.state.merkle-empty";
const STATE_MERKLE_NODE_DOMAIN: &[u8] = b"blob.simulation.state.merkle-node";

pub(crate) const STATE_TILE_PAGE_SIZE: usize = 256;
pub(crate) const STATE_CELL_PAGE_SIZE: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalHash([u8; 32]);

impl CanonicalHash {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        use std::fmt::Write;

        let mut hex = String::with_capacity(64);
        for byte in self.0 {
            write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
        }
        hex
    }
}

impl AsRef<[u8]> for CanonicalHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Display for CanonicalHash {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct CanonicalEncoder(ring::digest::Context);

#[cfg(target_arch = "wasm32")]
struct CanonicalEncoder(Sha256);

impl CanonicalEncoder {
    fn new(domain: &[u8]) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let mut encoder = Self(ring::digest::Context::new(&ring::digest::SHA256));
        #[cfg(target_arch = "wasm32")]
        let mut encoder = Self(Sha256::new());
        encoder.bytes(domain);
        encoder.u16(CANONICAL_HASH_FORMAT_VERSION);
        encoder
    }

    fn finish(self) -> CanonicalHash {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let digest = self.0.finish();
            let mut bytes = [0; 32];
            bytes.copy_from_slice(digest.as_ref());
            CanonicalHash(bytes)
        }
        #[cfg(target_arch = "wasm32")]
        {
            CanonicalHash(self.0.finalize().into())
        }
    }

    fn raw(&mut self, bytes: &[u8]) {
        #[cfg(not(target_arch = "wasm32"))]
        self.0.update(bytes);
        #[cfg(target_arch = "wasm32")]
        self.0.update(bytes);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.raw(bytes);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u8(&mut self, value: u8) {
        self.raw(&[value]);
    }

    fn i8(&mut self, value: i8) {
        self.raw(&value.to_le_bytes());
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn i16(&mut self, value: i16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn hash(&mut self, value: CanonicalHash) {
        self.raw(value.as_bytes());
    }
}

pub(crate) fn semantic_ruleset_hash(rules: &ReferenceRuleset) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(SEMANTIC_RULESET_DOMAIN);
    encoder.u16(REFERENCE_SEMANTIC_KERNEL_VERSION);
    encode_ruleset(&mut encoder, rules);
    encoder.finish()
}

pub(crate) fn compiled_ruleset_hash(
    rules: &ReferenceRuleset,
    neighborhood: &CompiledNeighborhood,
) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(COMPILED_RULESET_DOMAIN);
    encoder.u16(REFERENCE_SEMANTIC_KERNEL_VERSION);
    encoder.hash(semantic_ruleset_hash(rules));
    encoder.usize(neighborhood.width());
    encoder.usize(neighborhood.height());
    encoder.finish()
}

pub(crate) fn canonical_state_hash(
    compiled_ruleset_hash: CanonicalHash,
    now: SimTime,
    tiles: &[TileState],
    cells: &[(CellKey, CellState)],
    next_cell_key: u64,
) -> CanonicalHash {
    let tile_pages = tile_page_hashes(tiles);
    let (_, cell_leaves) = cell_commitments_slice(cells);
    let cell_pages = cell_page_hashes_from_leaves(&cell_leaves, next_cell_key);
    state_root_hash(
        compiled_ruleset_hash,
        now,
        next_cell_key,
        tiles.len(),
        MerkleCache::new(0, tile_pages).root(),
        cells.len(),
        cell_pages.len(),
        MerkleCache::new(1, cell_pages).root(),
    )
}

/// Derived page hashes and Merkle nodes for the live authoritative state.
/// This cache is never serialized or included in canonical equality.
#[derive(Debug, Clone)]
pub(crate) struct IncrementalStateHash {
    compiled_ruleset_hash: CanonicalHash,
    tile_tree: MerkleCache,
    cell_tree: MerkleCache,
    /// Canonical per-cell commitments keep unchanged private-memory bytes out
    /// of dirty page rehashes. The map is derived and never serialized.
    cell_leaves: Vec<Option<CanonicalHash>>,
    cell_memory_hashes: Vec<Option<CanonicalHash>>,
    dirty_tile_pages: BTreeSet<usize>,
    dirty_cells: BTreeSet<CellKey>,
    dirty_cell_memories: BTreeSet<CellKey>,
    all_cells_dirty: bool,
}

/// Trusted in-memory seed for the complete incremental hash cache. Planner
/// checkpoints chunk and share these derived arrays; public checkpoints never
/// serialize them and server verification still reconstructs from canonical
/// state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IncrementalStateHashSeed {
    pub compiled_ruleset_hash: CanonicalHash,
    pub tile_page_count: usize,
    pub tile_base: usize,
    pub tile_nodes: Vec<CanonicalHash>,
    pub cell_page_count: usize,
    pub cell_base: usize,
    pub cell_nodes: Vec<CanonicalHash>,
    pub cell_leaves: Vec<Option<CanonicalHash>>,
    pub cell_memory_hashes: Vec<Option<CanonicalHash>>,
}

impl IncrementalStateHash {
    pub(crate) fn new<T: TileSource + ?Sized>(
        compiled_ruleset_hash: CanonicalHash,
        tiles: &T,
        cells: &CellStore,
        next_cell_key: u64,
    ) -> Self {
        let (cell_memory_hashes, cell_leaves) = cell_commitment_slots(cells, next_cell_key);
        Self {
            compiled_ruleset_hash,
            tile_tree: MerkleCache::new(0, tile_page_hashes(tiles)),
            cell_tree: MerkleCache::new(
                1,
                cell_page_hashes_from_slots(&cell_leaves, next_cell_key),
            ),
            cell_leaves,
            cell_memory_hashes,
            dirty_tile_pages: BTreeSet::new(),
            dirty_cells: BTreeSet::new(),
            dirty_cell_memories: BTreeSet::new(),
            all_cells_dirty: false,
        }
    }

    pub(crate) fn from_seed(
        seed: IncrementalStateHashSeed,
        compiled_ruleset_hash: CanonicalHash,
        tile_count: usize,
        cells: &CellStore,
        next_cell_key: u64,
    ) -> Option<Self> {
        if seed.compiled_ruleset_hash != compiled_ruleset_hash
            || seed.tile_page_count != tile_count.div_ceil(STATE_TILE_PAGE_SIZE)
            || seed.cell_page_count != cell_page_count(next_cell_key)
        {
            return None;
        }
        let (cell_memory_hashes, cell_leaves) =
            if seed.cell_leaves.is_empty() && seed.cell_memory_hashes.is_empty() {
                cell_commitment_slots(cells, next_cell_key)
            } else if seed.cell_leaves.len() == cell_slot_count(next_cell_key)
                && seed.cell_memory_hashes.len() == cell_slot_count(next_cell_key)
            {
                (seed.cell_memory_hashes, seed.cell_leaves)
            } else {
                return None;
            };
        let tile_tree = if seed.tile_nodes.len() == seed.tile_page_count {
            let tree = MerkleCache::new(0, seed.tile_nodes);
            (tree.base == seed.tile_base).then_some(tree)?
        } else {
            MerkleCache::from_seed(0, seed.tile_page_count, seed.tile_base, seed.tile_nodes)?
        };
        let cell_tree = if seed.cell_nodes.is_empty() {
            MerkleCache::new(1, cell_page_hashes_from_slots(&cell_leaves, next_cell_key))
        } else {
            MerkleCache::from_seed(1, seed.cell_page_count, seed.cell_base, seed.cell_nodes)?
        };
        Some(Self {
            compiled_ruleset_hash: seed.compiled_ruleset_hash,
            tile_tree,
            cell_tree,
            cell_leaves,
            cell_memory_hashes,
            dirty_tile_pages: BTreeSet::new(),
            dirty_cells: BTreeSet::new(),
            dirty_cell_memories: BTreeSet::new(),
            all_cells_dirty: false,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn seed(&self) -> Option<IncrementalStateHashSeed> {
        if self.all_cells_dirty
            || !self.dirty_tile_pages.is_empty()
            || !self.dirty_cells.is_empty()
            || !self.dirty_cell_memories.is_empty()
        {
            return None;
        }
        Some(IncrementalStateHashSeed {
            compiled_ruleset_hash: self.compiled_ruleset_hash,
            tile_page_count: self.tile_tree.page_count,
            tile_base: self.tile_tree.base,
            tile_nodes: self.tile_tree.nodes.clone(),
            cell_page_count: self.cell_tree.page_count,
            cell_base: self.cell_tree.base,
            cell_nodes: self.cell_tree.nodes.clone(),
            cell_leaves: self.cell_leaves.clone(),
            cell_memory_hashes: self.cell_memory_hashes.clone(),
        })
    }

    pub(crate) fn mark_tile(&mut self, tile: TileIndex) {
        self.dirty_tile_pages.insert(tile.0 / STATE_TILE_PAGE_SIZE);
    }

    pub(crate) const fn compiled_ruleset_hash(&self) -> CanonicalHash {
        self.compiled_ruleset_hash
    }

    pub(crate) fn mark_cell(&mut self, cell: CellKey) {
        if self.all_cells_dirty {
            return;
        }
        if usize::try_from(cell.0 / STATE_CELL_PAGE_SIZE).is_ok() {
            self.dirty_cells.insert(cell);
        } else {
            self.mark_all_cells();
        }
    }

    pub(crate) fn mark_cell_memory(&mut self, cell: CellKey) {
        self.mark_cell(cell);
        self.dirty_cell_memories.insert(cell);
    }

    pub(crate) fn mark_all_cells(&mut self) {
        self.all_cells_dirty = true;
        self.dirty_cells.clear();
    }

    fn refresh_all_cell_leaves(
        &mut self,
        cells: &CellStore,
        next_cell_key: u64,
    ) -> Vec<CanonicalHash> {
        let dirty_memories = std::mem::take(&mut self.dirty_cell_memories);
        for key in dirty_memories {
            match cells.get(&key) {
                Some(cell) => {
                    set_cell_slot(
                        &mut self.cell_memory_hashes,
                        key,
                        Some(cell_memory_hash(cell)),
                    );
                }
                None => {
                    set_cell_slot(&mut self.cell_memory_hashes, key, None);
                }
            }
        }
        // `mark_all_cells` deliberately subsumes later per-cell dirty marks.
        // A birth can therefore occur while the full refresh is pending and
        // have no pre-existing private-memory slot. The full leaf pass already
        // visits every live cell, so fill only those missing commitments first.
        for (key, cell) in cells {
            if cell_slot(&self.cell_memory_hashes, *key).is_none() {
                set_cell_slot(
                    &mut self.cell_memory_hashes,
                    *key,
                    Some(cell_memory_hash(cell)),
                );
            }
        }
        refresh_cell_pages(
            cells,
            &self.cell_memory_hashes,
            &mut self.cell_leaves,
            next_cell_key,
        )
    }

    pub(crate) fn hash<T: TileSource + ?Sized>(
        &mut self,
        now: SimTime,
        tiles: &T,
        cells: &CellStore,
        next_cell_key: u64,
    ) -> CanonicalHash {
        let required_cell_pages = cell_page_count(next_cell_key);
        if required_cell_pages != self.cell_tree.page_count() {
            (self.cell_memory_hashes, self.cell_leaves) =
                cell_commitment_slots(cells, next_cell_key);
            self.cell_tree = MerkleCache::new(
                1,
                cell_page_hashes_from_slots(&self.cell_leaves, next_cell_key),
            );
            self.dirty_cells.clear();
            self.dirty_cell_memories.clear();
            self.all_cells_dirty = false;
        }
        let dirty_tile_pages: Vec<_> = std::mem::take(&mut self.dirty_tile_pages)
            .into_iter()
            .filter(|page| *page < self.tile_tree.page_count())
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        let tile_updates: Vec<_> = if dirty_tile_pages.len() >= 8 {
            dirty_tile_pages
                .into_par_iter()
                .map(|page| (page, tile_page_hash(page, tiles)))
                .collect()
        } else {
            dirty_tile_pages
                .into_iter()
                .map(|page| (page, tile_page_hash(page, tiles)))
                .collect()
        };
        #[cfg(target_arch = "wasm32")]
        let tile_updates: Vec<_> = dirty_tile_pages
            .into_iter()
            .map(|page| (page, tile_page_hash(page, tiles)))
            .collect();
        self.tile_tree.set_many(tile_updates);

        if self.all_cells_dirty {
            let pages = self.refresh_all_cell_leaves(cells, next_cell_key);
            self.cell_tree = MerkleCache::new(1, pages);
            self.dirty_cells.clear();
            self.dirty_cell_memories.clear();
            self.all_cells_dirty = false;
        }
        let dirty_cells = std::mem::take(&mut self.dirty_cells);
        let dirty_memories = std::mem::take(&mut self.dirty_cell_memories);
        for key in dirty_memories {
            match cells.get(&key) {
                Some(cell) => {
                    set_cell_slot(
                        &mut self.cell_memory_hashes,
                        key,
                        Some(cell_memory_hash(cell)),
                    );
                }
                None => {
                    set_cell_slot(&mut self.cell_memory_hashes, key, None);
                }
            }
        }
        for key in &dirty_cells {
            match cells.get(key) {
                Some(cell) if cell_slot(&self.cell_memory_hashes, *key).is_none() => {
                    set_cell_slot(
                        &mut self.cell_memory_hashes,
                        *key,
                        Some(cell_memory_hash(cell)),
                    );
                }
                None => {
                    set_cell_slot(&mut self.cell_memory_hashes, *key, None);
                    set_cell_slot(&mut self.cell_leaves, *key, None);
                }
                Some(_) => {}
            }
        }
        let dense_cell_refresh =
            dirty_cells.len() >= 512 && dirty_cells.len().saturating_mul(4) >= cells.len();
        if dense_cell_refresh {
            let pages = refresh_cell_pages(
                cells,
                &self.cell_memory_hashes,
                &mut self.cell_leaves,
                next_cell_key,
            );
            self.cell_tree = MerkleCache::new(1, pages);
        }
        let mut dirty_cell_pages = BTreeSet::new();
        if !dense_cell_refresh {
            for key in dirty_cells {
                if let Some(cell) = cells.get(&key) {
                    let memory_hash = cell_slot(&self.cell_memory_hashes, key)
                        .expect("live cell is missing its cached private-memory commitment");
                    set_cell_slot(
                        &mut self.cell_leaves,
                        key,
                        Some(cell_leaf_hash(key, cell, memory_hash)),
                    );
                }
                if let Ok(page) = usize::try_from(key.0 / STATE_CELL_PAGE_SIZE) {
                    if page < self.cell_tree.page_count() {
                        dirty_cell_pages.insert(page);
                    }
                }
            }
        }
        let dirty_cell_pages = dirty_cell_pages.into_iter().collect::<Vec<_>>();
        #[cfg(not(target_arch = "wasm32"))]
        let cell_updates: Vec<_> = if dirty_cell_pages.len() >= 2 {
            dirty_cell_pages
                .into_par_iter()
                .map(|page| (page, cell_page_hash_from_slots(page, &self.cell_leaves)))
                .collect()
        } else {
            dirty_cell_pages
                .into_iter()
                .map(|page| (page, cell_page_hash_from_slots(page, &self.cell_leaves)))
                .collect()
        };
        #[cfg(target_arch = "wasm32")]
        let cell_updates: Vec<_> = dirty_cell_pages
            .into_iter()
            .map(|page| (page, cell_page_hash_from_slots(page, &self.cell_leaves)))
            .collect();
        self.cell_tree.set_many(cell_updates);
        state_root_hash(
            self.compiled_ruleset_hash,
            now,
            next_cell_key,
            tiles.tile_len(),
            self.tile_tree.root(),
            cells.len(),
            self.cell_tree.page_count(),
            self.cell_tree.root(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn state_root_hash(
    compiled_ruleset_hash: CanonicalHash,
    now: SimTime,
    next_cell_key: u64,
    tile_count: usize,
    tile_root: CanonicalHash,
    cell_count: usize,
    cell_page_count: usize,
    cell_root: CanonicalHash,
) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(STATE_DOMAIN);
    encoder.hash(compiled_ruleset_hash);
    encoder.u64(now.0);
    encoder.u64(next_cell_key);
    encoder.usize(tile_count);
    encoder.usize(STATE_TILE_PAGE_SIZE);
    encoder.usize(tile_count.div_ceil(STATE_TILE_PAGE_SIZE));
    encoder.hash(tile_root);
    encoder.usize(cell_count);
    encoder.u64(STATE_CELL_PAGE_SIZE);
    encoder.usize(cell_page_count);
    encoder.hash(cell_root);
    encoder.finish()
}

fn tile_page_hashes<T: TileSource + ?Sized>(tiles: &T) -> Vec<CanonicalHash> {
    (0..tiles.tile_len().div_ceil(STATE_TILE_PAGE_SIZE))
        .map(|page| tile_page_hash(page, tiles))
        .collect()
}

fn tile_page_hash<T: TileSource + ?Sized>(page: usize, tiles: &T) -> CanonicalHash {
    let start = page
        .saturating_mul(STATE_TILE_PAGE_SIZE)
        .min(tiles.tile_len());
    let end = start
        .saturating_add(STATE_TILE_PAGE_SIZE)
        .min(tiles.tile_len());
    let mut encoder = CanonicalEncoder::new(STATE_TILE_PAGE_DOMAIN);
    encoder.usize(page);
    encoder.usize(end - start);
    for index in start..end {
        encoder.usize(index);
        encode_tile(
            &mut encoder,
            tiles
                .tile_at(index)
                .expect("canonical tile source changed length while hashing"),
        );
    }
    encoder.finish()
}

fn cell_page_count(next_cell_key: u64) -> usize {
    usize::try_from(next_cell_key.div_ceil(STATE_CELL_PAGE_SIZE)).unwrap_or(usize::MAX)
}

fn cell_page_hashes_from_leaves(
    leaves: &BTreeMap<CellKey, CanonicalHash>,
    next_cell_key: u64,
) -> Vec<CanonicalHash> {
    (0..cell_page_count(next_cell_key))
        .map(|page| cell_page_hash_from_leaves(page, leaves))
        .collect()
}

fn cell_page_hash_from_leaves(
    page: usize,
    leaves: &BTreeMap<CellKey, CanonicalHash>,
) -> CanonicalHash {
    let start = u64::try_from(page)
        .unwrap_or(u64::MAX)
        .saturating_mul(STATE_CELL_PAGE_SIZE);
    let end = start.saturating_add(STATE_CELL_PAGE_SIZE);
    let range = leaves.range(CellKey(start)..CellKey(end));
    let count = range.clone().count();
    let mut encoder = CanonicalEncoder::new(STATE_CELL_PAGE_DOMAIN);
    encoder.usize(page);
    encoder.usize(count);
    for leaf in range.map(|(_, leaf)| leaf) {
        encoder.hash(*leaf);
    }
    encoder.finish()
}

fn cell_slot_count(next_cell_key: u64) -> usize {
    cell_page_count(next_cell_key).saturating_mul(STATE_CELL_PAGE_SIZE as usize)
}

fn cell_slot(slots: &[Option<CanonicalHash>], key: CellKey) -> Option<CanonicalHash> {
    usize::try_from(key.0)
        .ok()
        .and_then(|index| slots.get(index))
        .copied()
        .flatten()
}

fn set_cell_slot(
    slots: &mut Vec<Option<CanonicalHash>>,
    key: CellKey,
    value: Option<CanonicalHash>,
) {
    let index = usize::try_from(key.0).expect("cell key does not fit the host address space");
    if index >= slots.len() {
        slots.resize(index.saturating_add(1), None);
    }
    slots[index] = value;
}

fn cell_page_hashes_from_slots(
    leaves: &[Option<CanonicalHash>],
    next_cell_key: u64,
) -> Vec<CanonicalHash> {
    (0..cell_page_count(next_cell_key))
        .map(|page| cell_page_hash_from_slots(page, leaves))
        .collect()
}

fn cell_page_hash_from_slots(page: usize, leaves: &[Option<CanonicalHash>]) -> CanonicalHash {
    let start = page.saturating_mul(STATE_CELL_PAGE_SIZE as usize);
    let end = start
        .saturating_add(STATE_CELL_PAGE_SIZE as usize)
        .min(leaves.len());
    let page_leaves = &leaves[start.min(leaves.len())..end];
    let mut encoder = CanonicalEncoder::new(STATE_CELL_PAGE_DOMAIN);
    encoder.usize(page);
    encoder.usize(page_leaves.iter().filter(|leaf| leaf.is_some()).count());
    for leaf in page_leaves.iter().flatten() {
        encoder.hash(*leaf);
    }
    encoder.finish()
}

fn cell_commitment_slots(
    cells: &CellStore,
    next_cell_key: u64,
) -> (Vec<Option<CanonicalHash>>, Vec<Option<CanonicalHash>>) {
    #[cfg(not(target_arch = "wasm32"))]
    let commitments = if cells.len() >= 512 {
        cells
            .par_iter()
            .map(|(key, cell)| {
                let memory = cell_memory_hash(cell);
                (key, memory, cell_leaf_hash(key, cell, memory))
            })
            .collect::<Vec<_>>()
    } else {
        cells
            .iter()
            .map(|(key, cell)| {
                let memory = cell_memory_hash(cell);
                (*key, memory, cell_leaf_hash(*key, cell, memory))
            })
            .collect::<Vec<_>>()
    };
    #[cfg(target_arch = "wasm32")]
    let commitments = cells
        .iter()
        .map(|(key, cell)| {
            let memory = cell_memory_hash(cell);
            (*key, memory, cell_leaf_hash(*key, cell, memory))
        })
        .collect::<Vec<_>>();
    let slot_count = cell_slot_count(next_cell_key);
    let mut memories = vec![None; slot_count];
    let mut leaves = vec![None; slot_count];
    for (key, memory, leaf) in commitments {
        let index = usize::try_from(key.0).expect("cell key does not fit the host address space");
        memories[index] = Some(memory);
        leaves[index] = Some(leaf);
    }
    (memories, leaves)
}

fn refresh_cell_pages(
    cells: &CellStore,
    memories: &[Option<CanonicalHash>],
    leaves: &mut Vec<Option<CanonicalHash>>,
    next_cell_key: u64,
) -> Vec<CanonicalHash> {
    let page_count = cell_page_count(next_cell_key);
    let refresh_page = |page: usize| {
        let start = u64::try_from(page)
            .unwrap_or(u64::MAX)
            .saturating_mul(STATE_CELL_PAGE_SIZE);
        let end = start.saturating_add(STATE_CELL_PAGE_SIZE);
        let range = cells.range(CellKey(start)..CellKey(end));
        let count = range.clone().count();
        let mut encoder = CanonicalEncoder::new(STATE_CELL_PAGE_DOMAIN);
        encoder.usize(page);
        encoder.usize(count);
        let mut page_leaves = Vec::with_capacity(count);
        for (key, cell) in range {
            let memory_hash = cell_slot(memories, *key)
                .expect("live cell is missing its cached private-memory commitment");
            let leaf = cell_leaf_hash(*key, cell, memory_hash);
            encoder.hash(leaf);
            page_leaves.push((*key, leaf));
        }
        (encoder.finish(), page_leaves)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let refreshed = if page_count >= 2 {
        (0..page_count)
            .into_par_iter()
            .map(refresh_page)
            .collect::<Vec<_>>()
    } else {
        (0..page_count).map(refresh_page).collect::<Vec<_>>()
    };
    #[cfg(target_arch = "wasm32")]
    let refreshed = (0..page_count).map(refresh_page).collect::<Vec<_>>();

    leaves.clear();
    leaves.resize(cell_slot_count(next_cell_key), None);
    let mut pages = Vec::with_capacity(page_count);
    for (page_hash, page_leaves) in refreshed {
        pages.push(page_hash);
        for (key, leaf) in page_leaves {
            let index =
                usize::try_from(key.0).expect("cell key does not fit the host address space");
            leaves[index] = Some(leaf);
        }
    }
    pages
}

fn cell_commitments_slice(
    cells: &[(CellKey, CellState)],
) -> (
    BTreeMap<CellKey, CanonicalHash>,
    BTreeMap<CellKey, CanonicalHash>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let commitments = if cells.len() >= 512 {
        cells
            .par_iter()
            .map(|(key, cell)| {
                let memory = cell_memory_hash(cell);
                (*key, memory, cell_leaf_hash(*key, cell, memory))
            })
            .collect::<Vec<_>>()
    } else {
        cells
            .iter()
            .map(|(key, cell)| {
                let memory = cell_memory_hash(cell);
                (*key, memory, cell_leaf_hash(*key, cell, memory))
            })
            .collect::<Vec<_>>()
    };
    #[cfg(target_arch = "wasm32")]
    let commitments = cells
        .iter()
        .map(|(key, cell)| {
            let memory = cell_memory_hash(cell);
            (*key, memory, cell_leaf_hash(*key, cell, memory))
        })
        .collect::<Vec<_>>();
    let memories = commitments
        .iter()
        .map(|(key, memory, _)| (*key, *memory))
        .collect();
    let leaves = commitments
        .into_iter()
        .map(|(key, _, leaf)| (key, leaf))
        .collect();
    (memories, leaves)
}

fn cell_memory_hash(cell: &CellState) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(STATE_CELL_MEMORY_DOMAIN);
    encoder.bytes(&cell.private_memory);
    encoder.finish()
}

fn cell_leaf_hash(key: CellKey, cell: &CellState, memory_hash: CanonicalHash) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(STATE_CELL_LEAF_DOMAIN);
    encoder.u64(key.0);
    encode_cell(&mut encoder, cell, memory_hash);
    encoder.finish()
}

#[derive(Debug, Clone)]
struct MerkleCache {
    kind: u8,
    page_count: usize,
    base: usize,
    nodes: Vec<CanonicalHash>,
}

impl MerkleCache {
    fn new(kind: u8, pages: Vec<CanonicalHash>) -> Self {
        let page_count = pages.len();
        let base = page_count.max(1).next_power_of_two();
        let empty = merkle_empty_hash(kind);
        let mut nodes = vec![empty; base.saturating_mul(2)];
        for (index, hash) in pages.into_iter().enumerate() {
            nodes[base + index] = hash;
        }
        for index in (1..base).rev() {
            nodes[index] = merkle_node_hash(kind, nodes[index * 2], nodes[index * 2 + 1]);
        }
        Self {
            kind,
            page_count,
            base,
            nodes,
        }
    }

    fn from_seed(
        kind: u8,
        page_count: usize,
        base: usize,
        nodes: Vec<CanonicalHash>,
    ) -> Option<Self> {
        if base != page_count.max(1).next_power_of_two()
            || nodes.len() != base.checked_mul(2)?
            || nodes.len() < 2
        {
            return None;
        }
        Some(Self {
            kind,
            page_count,
            base,
            nodes,
        })
    }

    const fn page_count(&self) -> usize {
        self.page_count
    }

    fn root(&self) -> CanonicalHash {
        self.nodes[1]
    }

    fn set(&mut self, page: usize, hash: CanonicalHash) {
        debug_assert!(page < self.page_count);
        let mut index = self.base + page;
        self.nodes[index] = hash;
        while index > 1 {
            index /= 2;
            self.nodes[index] =
                merkle_node_hash(self.kind, self.nodes[index * 2], self.nodes[index * 2 + 1]);
        }
    }

    fn set_many(&mut self, updates: Vec<(usize, CanonicalHash)>) {
        if updates.is_empty() {
            return;
        }
        if updates.len().saturating_mul(8) < self.page_count {
            for (page, hash) in updates {
                self.set(page, hash);
            }
            return;
        }
        for (page, hash) in updates {
            debug_assert!(page < self.page_count);
            self.nodes[self.base + page] = hash;
        }
        for index in (1..self.base).rev() {
            self.nodes[index] =
                merkle_node_hash(self.kind, self.nodes[index * 2], self.nodes[index * 2 + 1]);
        }
    }
}

fn merkle_empty_hash(kind: u8) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(STATE_MERKLE_EMPTY_DOMAIN);
    encoder.u8(kind);
    encoder.finish()
}

fn merkle_node_hash(kind: u8, left: CanonicalHash, right: CanonicalHash) -> CanonicalHash {
    let mut encoder = CanonicalEncoder::new(STATE_MERKLE_NODE_DOMAIN);
    encoder.u8(kind);
    encoder.hash(left);
    encoder.hash(right);
    encoder.finish()
}

fn encode_ruleset(encoder: &mut CanonicalEncoder, rules: &ReferenceRuleset) {
    let ReferenceRuleset {
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
        move_effort_base,
        move_mass_units_per_effort,
        attack_effort_base,
        guard_effort_base,
        consume_effort_base,
        split_effort_base,
        regurgitate_effort_base,
        excavate_effort_base,
        deposit_terrain_effort_base,
        bite_capacity,
        gut_capacity,
        digestion_rate_numerator,
        digestion_rate_denominator,
        metabolism_rate_numerator,
        metabolism_rate_denominator,
        signal_emission_cost,
        signal_decay_rate_numerator,
        signal_decay_rate_denominator,
        diffusion_interval_quanta,
        diffusion_rate_numerator,
        diffusion_rate_denominator,
        diffusion_targets,
        terrain_mass_per_elevation,
        minimum_survival_energy,
        child_core_mass,
        maximum_elevation_delta,
        guard_damage_numerator,
        guard_damage_denominator,
        attack_payload_damage_numerator,
        attack_payload_damage_denominator,
        attack_mass_damage_numerator,
        attack_mass_damage_denominator,
        apparent_mass_bucket_width,
        max_private_memory_bytes,
    } = rules;

    encode_neighborhood(encoder, neighborhood);
    encode_time(encoder, time);
    encoder.usize(effort_profiles.len());
    for profile in effort_profiles {
        encode_effort_profile(encoder, profile);
    }
    for duration in [
        wait_duration,
        move_duration,
        attack_duration,
        consume_duration,
        split_duration,
        regurgitate_duration,
        excavate_duration,
        deposit_terrain_duration,
    ] {
        encode_duration(encoder, duration);
    }
    for value in [
        move_effort_base,
        move_mass_units_per_effort,
        attack_effort_base,
        guard_effort_base,
        consume_effort_base,
        split_effort_base,
        regurgitate_effort_base,
        excavate_effort_base,
        deposit_terrain_effort_base,
        bite_capacity,
        gut_capacity,
        digestion_rate_numerator,
        digestion_rate_denominator,
        metabolism_rate_numerator,
        metabolism_rate_denominator,
        signal_emission_cost,
        signal_decay_rate_numerator,
        signal_decay_rate_denominator,
        diffusion_interval_quanta,
        diffusion_rate_numerator,
        diffusion_rate_denominator,
        terrain_mass_per_elevation,
        minimum_survival_energy,
        child_core_mass,
    ] {
        encoder.u64(*value);
    }
    encoder.u32(diffusion_targets.bits());
    encoder.i16(*maximum_elevation_delta);
    for value in [
        guard_damage_numerator,
        guard_damage_denominator,
        attack_payload_damage_numerator,
        attack_payload_damage_denominator,
        attack_mass_damage_numerator,
        attack_mass_damage_denominator,
        apparent_mass_bucket_width,
    ] {
        encoder.u64(*value);
    }
    encoder.usize(*max_private_memory_bytes);
}

fn encode_neighborhood(encoder: &mut CanonicalEncoder, neighborhood: &NeighborhoodSpec) {
    encoder.usize(neighborhood.slots.len());
    for offset in &neighborhood.slots {
        encoder.i8(offset.dx);
        encoder.i8(offset.dy);
        encoder.u16(offset.distance_cost_q10);
    }
    let observations = neighborhood.observations;
    for mask in [
        observations.occupancy,
        observations.marker,
        observations.apparent_mass,
        observations.activity,
        observations.terrain,
        observations.energy,
        observations.signal,
    ] {
        encoder.u32(mask.bits());
    }
    for action in [
        TargetingAction::Move,
        TargetingAction::Attack,
        TargetingAction::Split,
        TargetingAction::Regurgitate,
    ] {
        encoder.u32(neighborhood.target_mask(action).bits());
    }
    encoder.u8(match neighborhood.diagonal_corner_rule {
        DiagonalCornerRule::Allow => 0,
        DiagonalCornerRule::BlockIfEitherOrthogonalOccupied => 1,
        DiagonalCornerRule::BlockIfBothOrthogonalsOccupied => 2,
    });
    encoder.u8(match neighborhood.boundary_rule {
        BoundaryRule::Bounded => 0,
        BoundaryRule::Wrap => 1,
    });
    encoder.u8(neighborhood.max_radius);
}

fn encode_time(encoder: &mut CanonicalEncoder, time: &TimeConfig) {
    let TimeConfig {
        arithmetic_quanta_per_unit,
        completion_bucket,
        decision_interval_floor,
    } = time;
    encoder.u64(*arithmetic_quanta_per_unit);
    encoder.u64(*completion_bucket);
    encoder.u64(*decision_interval_floor);
}

fn encode_effort_profile(encoder: &mut CanonicalEncoder, profile: &EffortProfile) {
    let EffortProfile {
        cost_numerator,
        cost_denominator,
        duration_numerator,
        duration_denominator,
    } = profile;
    encoder.u32(*cost_numerator);
    encoder.u32(*cost_denominator);
    encoder.u32(*duration_numerator);
    encoder.u32(*duration_denominator);
}

fn encode_duration(encoder: &mut CanonicalEncoder, duration: &DurationRule) {
    let DurationRule {
        base_quanta,
        mass_quanta_numerator,
        mass_units_denominator,
    } = duration;
    encoder.u64(*base_quanta);
    encoder.u64(*mass_quanta_numerator);
    encoder.u64(*mass_units_denominator);
}

fn encode_tile(encoder: &mut CanonicalEncoder, tile: &TileState) {
    let TileState {
        elevation,
        occupant,
        plant_energy,
        plant_capacity,
        plant_growth_rate,
        plant_growth_remainder,
        loose_energy,
        diffuse_energy,
        diffusion_remainder,
        signal_energy,
        signal_decay_remainder,
    } = tile;
    encoder.i16(*elevation);
    encode_cell_key_option(encoder, *occupant);
    encoder.u64(*plant_energy);
    encoder.u64(*plant_capacity);
    encoder.u64(*plant_growth_rate);
    encoder.u64(*plant_growth_remainder);
    encoder.u64(*loose_energy);
    encoder.u64(*diffuse_energy);
    encoder.u64(*diffusion_remainder);
    for value in signal_energy {
        encoder.u64(*value);
    }
    for value in signal_decay_remainder {
        encoder.u64(*value);
    }
}

fn encode_cell(encoder: &mut CanonicalEncoder, cell: &CellState, memory_hash: CanonicalHash) {
    encode_tile_index(encoder, cell.position);
    encoder.u64(cell.core_mass);
    encoder.u64(cell.assimilated_energy);
    encoder.u64(cell.gut_energy);
    encoder.u64(cell.digestion_remainder);
    encoder.u64(cell.metabolism_remainder);
    encoder.u64(cell.carried_material_mass);
    encoder.u32(cell.marker);
    encoder.bool(cell.guarded);
    encoder.hash(memory_hash);
    encoder.u64(cell.ready_at.0);
    match &cell.pending_action {
        Some(pending) => {
            encoder.u8(1);
            encode_pending_action(encoder, pending);
        }
        None => encoder.u8(0),
    }
    encode_outcome_option(encoder, cell.last_outcome);
}

fn encode_pending_action(encoder: &mut CanonicalEncoder, pending: &PendingAction) {
    let PendingAction {
        request,
        origin,
        target,
        started_at,
        completes_at,
        effort_spent,
        payload_escrow,
        rejection,
    } = pending;
    encode_action_request(encoder, request);
    encode_tile_index(encoder, *origin);
    encode_tile_index_option(encoder, *target);
    encoder.u64(started_at.0);
    encoder.u64(completes_at.0);
    encoder.u64(*effort_spent);
    encoder.u64(*payload_escrow);
    encode_reject_option(encoder, *rejection);
}

fn encode_action_request(encoder: &mut CanonicalEncoder, request: &ActionRequest) {
    match request {
        ActionRequest::Wait => encoder.u8(0),
        ActionRequest::Move { target, effort } => {
            encoder.u8(1);
            encoder.u8(target.0);
            encode_effort_tier(encoder, *effort);
        }
        ActionRequest::Attack {
            target,
            effort,
            payload,
        } => {
            encoder.u8(2);
            encoder.u8(target.0);
            encode_effort_tier(encoder, *effort);
            encoder.u64(*payload);
        }
        ActionRequest::Guard { effort } => {
            encoder.u8(3);
            encode_effort_tier(encoder, *effort);
        }
        ActionRequest::Consume { amount } => {
            encoder.u8(4);
            encoder.u64(*amount);
        }
        ActionRequest::Split {
            target,
            child_allocation,
            marker,
            private_memory,
        } => {
            encoder.u8(5);
            encoder.u8(target.0);
            encoder.u64(*child_allocation);
            encoder.u32(*marker);
            encoder.bytes(private_memory);
        }
        ActionRequest::Regurgitate { target, amount } => {
            encoder.u8(6);
            encoder.u8(target.0);
            encoder.u64(*amount);
        }
        ActionRequest::Signal { amounts } => {
            encoder.u8(9);
            for amount in amounts {
                encoder.u64(*amount);
            }
        }
        ActionRequest::Excavate => encoder.u8(7),
        ActionRequest::DepositTerrain => encoder.u8(8),
    }
}

fn encode_effort_tier(encoder: &mut CanonicalEncoder, effort: EffortTier) {
    encoder.u8(match effort {
        EffortTier::Low => 0,
        EffortTier::Standard => 1,
        EffortTier::High => 2,
    });
}

fn encode_outcome_option(encoder: &mut CanonicalEncoder, outcome: Option<OutcomeStatus>) {
    match outcome {
        Some(outcome) => {
            encoder.u8(1);
            match outcome {
                OutcomeStatus::Rejected(reason) => {
                    encoder.u8(0);
                    encode_reject_reason(encoder, reason);
                }
                OutcomeStatus::Success => encoder.u8(1),
                OutcomeStatus::Frustrated => encoder.u8(2),
                OutcomeStatus::Contested => encoder.u8(3),
                OutcomeStatus::Interrupted => encoder.u8(4),
            }
        }
        None => encoder.u8(0),
    }
}

fn encode_reject_option(encoder: &mut CanonicalEncoder, rejection: Option<RejectReason>) {
    match rejection {
        Some(reason) => {
            encoder.u8(1);
            encode_reject_reason(encoder, reason);
        }
        None => encoder.u8(0),
    }
}

fn encode_reject_reason(encoder: &mut CanonicalEncoder, reason: RejectReason) {
    encoder.u8(match reason {
        RejectReason::InvalidSlot => 0,
        RejectReason::ActionNotAllowedInSlot => 1,
        RejectReason::TargetOutsideWorld => 2,
        RejectReason::TargetsSelf => 3,
        RejectReason::ZeroPayload => 4,
        RejectReason::InsufficientGutEnergy => 5,
        RejectReason::ChildAllocationTooSmall => 6,
        RejectReason::PrivateMemoryTooLarge => 7,
        RejectReason::InsufficientEnergy => 8,
        RejectReason::InsufficientMaterial => 9,
        RejectReason::TerrainLimit => 10,
        RejectReason::ArithmeticOverflow => 11,
    });
}

fn encode_tile_index(encoder: &mut CanonicalEncoder, tile: TileIndex) {
    encoder.usize(tile.0);
}

fn encode_tile_index_option(encoder: &mut CanonicalEncoder, tile: Option<TileIndex>) {
    match tile {
        Some(tile) => {
            encoder.u8(1);
            encode_tile_index(encoder, tile);
        }
        None => encoder.u8(0),
    }
}

fn encode_cell_key_option(encoder: &mut CanonicalEncoder, cell: Option<CellKey>) {
    match cell {
        Some(cell) => {
            encoder.u8(1);
            encoder.u64(cell.0);
        }
        None => encoder.u8(0),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn display_is_fixed_width_lowercase_hex() {
        let hash = CanonicalHash::from_bytes([0xab; 32]);
        assert_eq!(hash.to_string(), "ab".repeat(32));
        assert_eq!(hash.to_hex(), hash.to_string());
    }

    #[test]
    fn slot_cache_matches_full_hash_across_sparse_topology_and_memory_changes() {
        let compiled = CanonicalHash::from_bytes([0x42; 32]);
        let mut cells = [0_u64, 255, 256, 511]
            .into_iter()
            .map(|key| {
                (
                    CellKey(key),
                    CellState {
                        position: TileIndex(usize::try_from(key % 8).unwrap()),
                        core_mass: 10,
                        assimilated_energy: 100 + key,
                        gut_energy: key % 7,
                        digestion_remainder: 0,
                        metabolism_remainder: 0,
                        carried_material_mass: 0,
                        marker: u32::try_from(key).unwrap(),
                        guarded: false,
                        ready_at: SimTime(0),
                        pending_action: None,
                        last_outcome: None,
                        cold: Arc::new(super::super::reference::CellColdState {
                            private_memory: Arc::from(vec![u8::try_from(key % 251).unwrap(); 17]),
                        }),
                    },
                )
            })
            .collect::<CellStore>();
        let tiles = vec![TileState::default(); 8];
        let mut next_cell_key = 700;
        let mut cache = IncrementalStateHash::new(compiled, &tiles, &cells, next_cell_key);
        let full_hash = |cells: &CellStore, next_cell_key| {
            canonical_state_hash(
                compiled,
                SimTime(0),
                &tiles,
                &cells
                    .iter()
                    .map(|(key, cell)| (*key, cell.clone()))
                    .collect::<Vec<_>>(),
                next_cell_key,
            )
        };
        assert_eq!(
            cache.hash(SimTime(0), &tiles, &cells, next_cell_key),
            full_hash(&cells, next_cell_key)
        );

        cells.get_mut(&CellKey(255)).unwrap().private_memory = Arc::from([9, 8, 7]);
        cache.mark_cell_memory(CellKey(255));
        cells.remove(&CellKey(256));
        cache.mark_cell(CellKey(256));
        let mut added = cells[&CellKey(255)].clone();
        added.marker = 300;
        cells.insert(CellKey(300), added);
        cache.mark_cell(CellKey(300));
        assert_eq!(
            cache.hash(SimTime(0), &tiles, &cells, next_cell_key),
            full_hash(&cells, next_cell_key)
        );

        for cell in cells.values_mut() {
            cell.guarded = true;
        }
        cache.mark_all_cells();
        let mut born_during_full_refresh = cells[&CellKey(255)].clone();
        born_during_full_refresh.marker = 301;
        born_during_full_refresh.private_memory = Arc::from([3, 0, 1]);
        cells.insert(CellKey(301), born_during_full_refresh);
        // This is intentionally subsumed by the pending full refresh.
        cache.mark_cell(CellKey(301));
        assert_eq!(
            cache.hash(SimTime(0), &tiles, &cells, next_cell_key),
            full_hash(&cells, next_cell_key)
        );

        let mut added = cells[&CellKey(511)].clone();
        added.marker = 700;
        cells.insert(CellKey(700), added);
        next_cell_key = 701;
        cache.mark_cell(CellKey(700));
        assert_eq!(
            cache.hash(SimTime(0), &tiles, &cells, next_cell_key),
            full_hash(&cells, next_cell_key)
        );
    }
}
