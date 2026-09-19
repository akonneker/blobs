//! The format-7 cell Merkle tree, retaining branches only around live pages.
//! Empty historical subtrees keep their exact digest. The rightmost page path
//! stays expanded so ordinary births never reconstruct a large empty subtree.
use super::{
    cell_page_hash_from_leaves, merkle_empty_hash, merkle_node_hash, CanonicalHash, CellKey,
    STATE_CELL_PAGE_SIZE,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(test)]
thread_local! {
    static BUILT_PAGES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CellMerkleCache {
    pub page_count: usize,
    pub base: usize,
    node: Node,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    hash: CanonicalHash,
    has_cells: bool,
    children: Option<Arc<[Node; 2]>>,
}
fn page_has_cells(page: usize, leaves: &BTreeMap<CellKey, CanonicalHash>) -> bool {
    let start = (page as u64).saturating_mul(STATE_CELL_PAGE_SIZE);
    leaves
        .range(CellKey(start)..CellKey(start.saturating_add(STATE_CELL_PAGE_SIZE)))
        .next()
        .is_some()
}
fn keep_children(start: usize, width: usize, count: usize, occupied: bool) -> bool {
    occupied || (count > 0 && (start..start + width).contains(&(count - 1)))
}
impl Node {
    fn build(
        start: usize,
        width: usize,
        count: usize,
        leaves: &BTreeMap<CellKey, CanonicalHash>,
    ) -> Self {
        if start >= count {
            let mut hash = merkle_empty_hash(1);
            for _ in 0..width.trailing_zeros() {
                hash = merkle_node_hash(1, hash, hash);
            }
            return Self {
                hash,
                has_cells: false,
                children: None,
            };
        }
        if width == 1 {
            #[cfg(test)]
            BUILT_PAGES.with(|pages| pages.set(pages.get() + 1));
            return Self {
                hash: cell_page_hash_from_leaves(start, leaves),
                has_cells: page_has_cells(start, leaves),
                children: None,
            };
        }
        let half = width / 2;
        let left = Self::build(start, half, count, leaves);
        let right = Self::build(start + half, half, count, leaves);
        Self::branch(start, width, count, [left, right])
    }
    fn branch(start: usize, width: usize, count: usize, children: [Self; 2]) -> Self {
        let hash = merkle_node_hash(1, children[0].hash, children[1].hash);
        let has_cells = children[0].has_cells || children[1].has_cells;
        let children = keep_children(start, width, count, has_cells).then(|| Arc::new(children));
        Self {
            hash,
            has_cells,
            children,
        }
    }
    /// Append logical pages without re-encoding the historical prefix.
    fn extend(
        &mut self,
        start: usize,
        width: usize,
        old_count: usize,
        count: usize,
        leaves: &BTreeMap<CellKey, CanonicalHash>,
    ) {
        if start >= count || start + width <= old_count {
            return;
        }
        if start >= old_count {
            *self = Self::build(start, width, count, leaves);
            return;
        }
        let half = width / 2;
        let children = self
            .children
            .as_mut()
            .expect("the old final-page path remains expanded for append");
        let children = Arc::make_mut(children);
        children[0].extend(start, half, old_count, count, leaves);
        children[1].extend(start + half, half, old_count, count, leaves);
        self.refresh_branch(start, width, count);
    }

    /// Remove the obsolete empty frontier path after appending. Ordinary live
    /// branches stay intact; only one root-to-leaf path needs inspection.
    fn prune_frontier(&mut self, start: usize, width: usize, old_last: usize, count: usize) {
        if let Some(children) = self.children.as_mut() {
            let children = Arc::make_mut(children);
            let half = width / 2;
            let side = usize::from(old_last >= start + half);
            children[side].prune_frontier(start + side * half, half, old_last, count);
            if !keep_children(start, width, count, self.has_cells) {
                self.children = None;
            }
        }
    }

    fn refresh_branch(&mut self, start: usize, width: usize, count: usize) {
        let children = self.children.as_ref().expect("branch has children");
        self.hash = merkle_node_hash(1, children[0].hash, children[1].hash);
        self.has_cells = children[0].has_cells || children[1].has_cells;
        if !keep_children(start, width, count, self.has_cells) {
            self.children = None;
        }
    }

    /// Sorted updates share traversal and ancestor hashing. Collapsed regions
    /// only need reconstruction when an old empty page becomes live again.
    fn set_many(
        &mut self,
        start: usize,
        width: usize,
        count: usize,
        updates: &[(usize, CanonicalHash)],
        leaves: &BTreeMap<CellKey, CanonicalHash>,
    ) {
        if updates.is_empty() {
            return;
        }
        if width == 1 {
            self.hash = updates.last().expect("nonempty page updates").1;
            self.has_cells = page_has_cells(start, leaves);
            return;
        }
        if self.children.is_none()
            && !self.has_cells
            && updates
                .iter()
                .all(|(page, _)| !page_has_cells(*page, leaves))
        {
            return;
        }
        let half = width / 2;
        let children = self.children.get_or_insert_with(|| {
            Arc::new([
                Self::build(start, half, count, leaves),
                Self::build(start + half, half, count, leaves),
            ])
        });
        let children = Arc::make_mut(children);
        let split = updates.partition_point(|(page, _)| *page < start + half);
        children[0].set_many(start, half, count, &updates[..split], leaves);
        children[1].set_many(start + half, half, count, &updates[split..], leaves);
        self.refresh_branch(start, width, count);
    }

    fn live_pages(&self, start: usize, width: usize, pages: &mut BTreeMap<usize, CanonicalHash>) {
        if !self.has_cells {
            return;
        }
        if width == 1 {
            pages.insert(start, self.hash);
            return;
        }
        let children = self
            .children
            .as_ref()
            .expect("live branches remain expanded");
        children[0].live_pages(start, width / 2, pages);
        children[1].live_pages(start + width / 2, width / 2, pages);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn allocations(&self, allocations: &mut Vec<(*const u8, usize)>) {
        if let Some(children) = &self.children {
            allocations.push((
                Arc::as_ptr(children).cast::<u8>(),
                std::mem::size_of::<[Node; 2]>(),
            ));
            for child in children.iter() {
                child.allocations(allocations);
            }
        }
    }
    #[cfg(test)]
    fn retained_nodes(&self) -> usize {
        1 + self.children.as_ref().map_or(0, |children| {
            children.iter().map(Self::retained_nodes).sum()
        })
    }
}
impl CellMerkleCache {
    pub fn new(page_count: usize, leaves: &BTreeMap<CellKey, CanonicalHash>) -> Self {
        let base = page_count.max(1).next_power_of_two();
        Self {
            page_count,
            base,
            node: Node::build(0, base, page_count, leaves),
        }
    }
    pub fn page_count(&self) -> usize {
        self.page_count
    }
    pub fn root(&self) -> CanonicalHash {
        self.node.hash
    }

    /// Validate live commitments against canonical cells without rebuilding
    /// collapsed historical subtrees. Unchanged pages retain shared branches.
    pub fn refresh_live_pages(&mut self, leaves: &BTreeMap<CellKey, CanonicalHash>) {
        let mut old_pages = BTreeMap::new();
        self.node.live_pages(0, self.base, &mut old_pages);
        let mut pages = old_pages
            .keys()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        pages.extend(
            leaves
                .keys()
                .map(|key| (key.0 / STATE_CELL_PAGE_SIZE) as usize),
        );
        let updates = pages
            .into_iter()
            .filter_map(|page| {
                let hash = cell_page_hash_from_leaves(page, leaves);
                (old_pages.get(&page) != Some(&hash)).then_some((page, hash))
            })
            .collect();
        self.update(self.page_count, updates, leaves);
    }

    /// Reachable shared branch allocations, excluding inline root/Arc headers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn allocations(&self) -> Vec<(*const u8, usize)> {
        let mut allocations = Vec::new();
        self.node.allocations(&mut allocations);
        allocations
    }
    /// Apply current commitments, extending only newly allocated pages. A
    /// decreasing key space is a restore operation and takes the cold path.
    pub fn update(
        &mut self,
        page_count: usize,
        mut updates: Vec<(usize, CanonicalHash)>,
        leaves: &BTreeMap<CellKey, CanonicalHash>,
    ) {
        if page_count < self.page_count {
            *self = Self::new(page_count, leaves);
            return;
        }
        let old_count = self.page_count;
        assert!(updates.iter().all(|(page, _)| *page < page_count));
        // Update the old frontier before extending/pruning it. A deferred batch
        // can refill that formerly empty page and append another page together.
        // Newly appended pages will be built from final commitments below.
        updates.retain(|(page, _)| *page < old_count);
        updates.sort_by_key(|(page, _)| *page);
        self.node
            .set_many(0, self.base, old_count, &updates, leaves);
        if page_count > old_count {
            self.node
                .extend(0, self.base, old_count, page_count, leaves);
            let required_base = page_count.max(1).next_power_of_two();
            while self.base < required_base {
                let right = Node::build(self.base, self.base, page_count, leaves);
                let left = std::mem::replace(
                    &mut self.node,
                    Node {
                        hash: merkle_empty_hash(1),
                        has_cells: false,
                        children: None,
                    },
                );
                self.base *= 2;
                self.node = Node::branch(0, self.base, page_count, [left, right]);
            }
            if old_count > 0 {
                self.node
                    .prune_frontier(0, self.base, old_count - 1, page_count);
            }
            self.page_count = page_count;
        }
    }

    #[cfg(test)]
    pub fn set_many(
        &mut self,
        updates: Vec<(usize, CanonicalHash)>,
        leaves: &BTreeMap<CellKey, CanonicalHash>,
    ) {
        self.update(self.page_count, updates, leaves);
    }
    #[cfg(test)]
    pub fn retained_nodes(&self) -> usize {
        self.node.retained_nodes()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{cell_page_hashes_from_leaves, MerkleCache};
    use super::*;
    #[test]
    fn append_batches_match_dense_roots_and_release_obsolete_frontier_paths() {
        let mut leaves = BTreeMap::new();
        let mut tree = CellMerkleCache::new(0, &leaves);
        for count in (1..=257).chain([511, 512, 513, 1025, 4097]) {
            let page = count - 1;
            let key = CellKey(page as u64 * STATE_CELL_PAGE_SIZE + 7);
            leaves.insert(key, CanonicalHash::from_bytes([count as u8; 32]));
            let mut changed = std::collections::BTreeSet::from([page]);
            while leaves.len() > 5 {
                let (key, _) = leaves.pop_first().unwrap();
                changed.insert((key.0 / STATE_CELL_PAGE_SIZE) as usize);
            }
            let oldest = *leaves.first_key_value().unwrap().0;
            leaves.insert(
                oldest,
                CanonicalHash::from_bytes([(count as u8).wrapping_add(1); 32]),
            );
            changed.insert((oldest.0 / STATE_CELL_PAGE_SIZE) as usize);
            tree.update(
                count,
                changed
                    .into_iter()
                    .map(|page| (page, cell_page_hash_from_leaves(page, &leaves)))
                    .collect(),
                &leaves,
            );
            assert_eq!(
                tree.root(),
                MerkleCache::new(
                    1,
                    cell_page_hashes_from_leaves(&leaves, count as u64 * STATE_CELL_PAGE_SIZE)
                )
                .root()
            );
            // Periodically empty all cells. Obsolete empty frontier paths
            // must not accumulate over repeated page transitions.
            if count % 7 != 0 && count != 4097 {
                continue;
            }
            let changed = leaves
                .keys()
                .map(|key| (key.0 / STATE_CELL_PAGE_SIZE) as usize)
                .collect::<Vec<_>>();
            leaves.clear();
            tree.update(
                count,
                changed
                    .into_iter()
                    .map(|page| (page, cell_page_hash_from_leaves(page, &leaves)))
                    .collect(),
                &leaves,
            );
            assert_eq!(
                tree.root(),
                MerkleCache::new(
                    1,
                    cell_page_hashes_from_leaves(&leaves, count as u64 * STATE_CELL_PAGE_SIZE)
                )
                .root()
            );
            assert!(tree.retained_nodes() <= 1 + 2 * tree.base.trailing_zeros() as usize);
        }
        tree.update(3, vec![], &leaves);
        assert_eq!(tree.root(), CellMerkleCache::new(3, &leaves).root());
    }

    #[test]
    fn extending_or_updating_live_pages_does_not_rebuild_historical_pages() {
        for old_count in [65_535, 65_536] {
            let mut leaves = BTreeMap::new();
            let mut tree = CellMerkleCache::new(old_count, &leaves);
            let page = old_count - 1;
            leaves.insert(
                CellKey(page as u64 * STATE_CELL_PAGE_SIZE + 1),
                CanonicalHash::from_bytes([8; 32]),
            );
            BUILT_PAGES.with(|pages| pages.set(0));
            tree.update(
                65_537,
                vec![(page, cell_page_hash_from_leaves(page, &leaves))],
                &leaves,
            );
            assert_eq!(BUILT_PAGES.with(std::cell::Cell::get), 65_537 - old_count);
            assert_eq!(tree.root(), CellMerkleCache::new(65_537, &leaves).root());
        }
        let count = 65_536;
        let key = CellKey((count as u64 - 1) * STATE_CELL_PAGE_SIZE);
        let mut leaves = BTreeMap::from([(key, CanonicalHash::from_bytes([1; 32]))]);
        let mut tree = CellMerkleCache::new(count, &leaves);
        BUILT_PAGES.with(|pages| pages.set(0));
        leaves.insert(key, CanonicalHash::from_bytes([2; 32]));
        tree.update(
            count,
            vec![(count - 1, cell_page_hash_from_leaves(count - 1, &leaves))],
            &leaves,
        );
        assert_eq!(BUILT_PAGES.with(std::cell::Cell::get), 0);
        tree.update(count + 1, vec![], &leaves);
        assert_eq!(BUILT_PAGES.with(std::cell::Cell::get), 1);
        // Redundant dead-cell marks in collapsed historical regions are no-ops.
        tree.update(
            count + 1,
            vec![(0, cell_page_hash_from_leaves(0, &leaves))],
            &leaves,
        );
        assert_eq!(BUILT_PAGES.with(std::cell::Cell::get), 1);
        assert_eq!(
            tree.root(),
            MerkleCache::new(
                1,
                cell_page_hashes_from_leaves(&leaves, (count as u64 + 1) * STATE_CELL_PAGE_SIZE)
            )
            .root()
        );
    }

    #[test]
    #[ignore = "manual release comparison of full rebuild versus live-page refresh"]
    fn benchmark_historical_cell_tree_updates() {
        use std::time::Instant;
        const COUNT: usize = 65_536;
        const ROUNDS: usize = 64;
        let leaves = (0..8)
            .map(|i| {
                (
                    CellKey((COUNT as u64 - 8 + i) * STATE_CELL_PAGE_SIZE),
                    CanonicalHash::from_bytes([i as u8; 32]),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut incremental = CellMerkleCache::new(COUNT, &leaves);
        let started = Instant::now();
        for _ in 0..ROUNDS {
            let updates = (COUNT - 8..COUNT)
                .map(|page| (page, cell_page_hash_from_leaves(page, &leaves)))
                .collect();
            incremental.update(COUNT, updates, &leaves);
            std::hint::black_box(incremental.root());
        }
        let update_time = started.elapsed();
        let started = Instant::now();
        for _ in 0..ROUNDS {
            let rebuilt = CellMerkleCache::new(COUNT, &leaves);
            assert_eq!(incremental.root(), rebuilt.root());
            std::hint::black_box(rebuilt);
        }
        println!("{ROUNDS} refreshes, {COUNT} historical pages, 8 live pages: update {:.3} ms, rebuild {:.3} ms", update_time.as_secs_f64()*1000.0, started.elapsed().as_secs_f64()*1000.0);
    }

    #[test]
    fn compact_tree_matches_dense_format_seven_through_pruning_and_reactivation() {
        for count in [0, 1, 2, 3, 17, 257, 4097] {
            let mut leaves = BTreeMap::new();
            let mut tree = CellMerkleCache::new(count, &leaves);
            let dense = |leaves: &BTreeMap<_, _>| {
                MerkleCache::new(
                    1,
                    cell_page_hashes_from_leaves(leaves, count as u64 * STATE_CELL_PAGE_SIZE),
                )
                .root()
            };
            assert_eq!(tree.root(), dense(&leaves));
            assert!(tree.retained_nodes() <= 1 + 2 * tree.base.trailing_zeros() as usize);
            if count == 0 {
                continue;
            }
            for step in 0..40_usize {
                let page = step.wrapping_mul(104729) % count;
                let key = CellKey(page as u64 * STATE_CELL_PAGE_SIZE + 5);
                leaves.insert(key, CanonicalHash::from_bytes([step as u8; 32]));
                tree.set_many(
                    vec![(page, cell_page_hash_from_leaves(page, &leaves))],
                    &leaves,
                );
                assert_eq!(tree.root(), dense(&leaves));
                // Delete everything, collapsing old branches, then reactivate
                // arbitrary old pages (not just the monotonic birth frontier).
                let updates = leaves
                    .keys()
                    .map(|key| (key.0 / STATE_CELL_PAGE_SIZE) as usize)
                    .collect::<std::collections::BTreeSet<_>>();
                leaves.clear();
                tree.set_many(
                    updates
                        .into_iter()
                        .map(|page| (page, cell_page_hash_from_leaves(page, &leaves)))
                        .collect(),
                    &leaves,
                );
                assert_eq!(tree.root(), dense(&leaves));
                assert!(tree.retained_nodes() <= 1 + 2 * tree.base.trailing_zeros() as usize);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn shared_restore_refreshes_live_pages_without_rebuilding_history() {
        let count = 65_536;
        let first = CellKey(0);
        let last = CellKey((count as u64 - 1) * STATE_CELL_PAGE_SIZE);
        let mut leaves = BTreeMap::from([
            (first, CanonicalHash::from_bytes([1; 32])),
            (last, CanonicalHash::from_bytes([2; 32])),
        ]);
        let original = CellMerkleCache::new(count, &leaves);
        let original_root = original.root();
        let original_allocations = original.allocations();
        BUILT_PAGES.with(|pages| pages.set(0));
        let mut restored = original.clone();
        restored.refresh_live_pages(&leaves);
        assert_eq!(BUILT_PAGES.with(std::cell::Cell::get), 0);
        assert_eq!(restored.allocations(), original_allocations);

        leaves.insert(first, CanonicalHash::from_bytes([3; 32]));
        restored.refresh_live_pages(&leaves);
        assert_eq!(BUILT_PAGES.with(std::cell::Cell::get), 0);
        assert_eq!(original.root(), original_root);
        let allocations = restored.allocations();
        assert!(allocations
            .iter()
            .any(|allocation| original_allocations.contains(allocation)));
        assert!(allocations
            .iter()
            .any(|allocation| !original_allocations.contains(allocation)));
        assert_eq!(restored.root(), CellMerkleCache::new(count, &leaves).root());

        // Stale live-page hashes must be cleared when canonical cells disappear.
        leaves.clear();
        restored.refresh_live_pages(&leaves);
        assert_eq!(restored.root(), CellMerkleCache::new(count, &leaves).root());
        assert!(restored.retained_nodes() <= 1 + 2 * restored.base.trailing_zeros() as usize);
        assert_eq!(original.root(), original_root);
    }
}
