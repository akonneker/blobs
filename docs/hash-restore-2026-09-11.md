# Trusted cell hash restoration — 2026-09-11

Planner checkpoints now retain compact cell Merkle branches, avoiding a walk
over every historical cell page when restoring a trusted checkpoint with its
compiled topology. Previously, the seed stored only cell-tree dimensions and
reconstructed every indexed empty-page digest from scratch.

Branches use `Arc` with copy-on-write along mutated paths. Snapshot capture and
unchanged restoration share the existing branches; later actions, deaths and
births preserve earlier checkpoints and sibling branches. Retained storage is
still bounded by live-page paths plus the current birth frontier. Allocation
accounting exposes the shared cell branches alongside tile-hash chunks, allowing
planner budgets to count unique retained allocations correctly.

Restoration recomputes live-cell commitments and compares page hashes across the
union of old and current live pages. Equal pages keep their branch allocations;
changed/deleted pages update or collapse as needed. A stale seed cannot hide
changed live-cell content from the checkpoint's final state-hash check. Only
historical empty-subtree digests are reused as trusted derived state.

Small boards now retain hash seeds when cell history spans multiple pages;
the earlier seed decision considered only tile count. Tile-chunk sharing no
longer depends on unrelated changes to cell-tree dimensions.

Canonical hash format 7, public checkpoint format 6, replay encodings and
simulation rules are unchanged. Public/unseeded restoration still reconstructs
from canonical state, and arbitrary old empty-page reactivation can still rebuild
a historical subtree. Live-cell commitment work, tile materialization and state
validation remain part of a seeded restore.

## Validation

- Structural counter: cloning and refreshing a seeded tree with 65,536 historical
  pages builds zero historical page leaves. Unchanged branches retain identical
  allocation identities; changing one live page copies its path and shares others.
- Deleting all cells refreshes stale live pages and prunes obsolete branches while
  preserving the original snapshot's root.
- A small-world planner checkpoint retains a bounded seed, shares all hash
  allocations after an unchanged round trip, resumes a pending action exactly,
  matches the independent full-hash oracle, and rejects modified canonical cells
  when the old state hash and seed are retained.
- Final gates: 598 workspace tests passed (33 ignored), workspace all-target/
  all-feature Clippy, engine/web WASM library Clippy, formatting, whitespace and
  59 schema constants/mirrors. Both shared fuzz-library tests pass, including
  128 resolver sequences and 12,480 generated Wasm admission cases. Evidence logs
  and final source hashes are in `training-output/hash-restore-2026-09-11`.

## Focused measurement

Release benchmark: 64 complete trusted restores of a 4x4 board with 2 live cells
and 65,536 historical cell pages took **0.565 ms with shared seeds versus
740.311 ms with the seed omitted**. Retained hash allocations were 3,008 bytes,
excluding inline root/vector/Arc header costs as in the existing lower-bound
accounting. The benchmark includes state reconstruction and validation, but the
world deliberately emphasizes long-dead history; this is not general simulation
or planner-search throughput. The ignored benchmark is
`resolution::checkpoint::trusted_hash_tests::benchmark_historical_checkpoint_restore`.
