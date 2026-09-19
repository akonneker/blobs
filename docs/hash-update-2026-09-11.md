# Cell hash updates after long churn — 2026-09-11

This completes the ordinary-update follow-up to [lifetime storage](lifetime-storage.md).
Previously, crossing a 256-key page boundary or marking all cells dirty rebuilt the
cell Merkle tree, visiting every historical page even when nearly all cells were
dead. The compact representation bounded retained memory, but not that update cost.

## Changes

- Apply dirty cell-page commitments through a shared sorted traversal, hashing
  each touched ancestor once per batch instead of once per dirty page.
- Extend the tree by hashing new logical pages and combining their subtrees with
  the cached historical prefix. Tree growth never re-encodes that prefix merely
  because the next-cell key crossed a page or power-of-two boundary.
- Keep the current final-page path expanded. After extension, prune the obsolete
  empty frontier path so historical paths do not accumulate. Apply changes to the
  old frontier before extending: one deferred batch can refill its empty final page
  and append another page together.
- Full-cell invalidation refreshes live commitments and updates the union of
  previously/currently live pages. This includes pages emptied by deaths. Dense
  dirty-cell batches use the same shared tree update instead of a cold rebuild.
- Redundant marks for dead cells in an already collapsed empty subtree are no-ops.

Canonical hash format 7, checkpoint schema 6, replay bytes, and simulation rules
are unchanged. Tile hashing is unchanged. All tree state remains derived.

## Bounds and remaining work

For `H` historical pages and `A` newly appended pages, extension hashes `A` new
page leaves plus their tree structure and boundary paths; it does not visit the
old `H` page leaves. Ordinary updates visit dirty live-page paths, sharing common
ancestors. All-cell invalidation still hashes all live cells and occupied pages.
Retained memory remains bounded by live page paths and the current frontier.

The [trusted-restore follow-up](hash-restore-2026-09-11.md) preserves compact
cell-tree branches in planner seeds. Cold/unseeded construction and restoration
still visit all historical pages: format 7 binds
the index of each empty page. Arbitrarily reactivating an old collapsed empty
subtree also requires rebuilding its digest structure. Monotonic engine births
use the retained frontier; arbitrary old-page reactivation is a separate path.
Shrinking logical key space takes the cold path. The independent full-hash oracle
still deliberately uses a dense page array/tree. Checkpoint lifetime-key limits
remain a resource guard for these operations.

## Verification and measurement

Regression coverage compares the compact tree with the independent dense Merkle
oracle through partial/full page boundaries, power-of-two growth, skipped ranges,
simultaneous old-page updates, deaths, full emptying, pruning, reactivation, and
shrinking. The state-cache oracle comparison also exercises 1,024 new cells after
more than a million historical keys, 600 dirty cells with private-memory changes,
512 deaths, and all-cell invalidation.

A deterministic work counter checks a 65,536-page tree: updating a live page
builds zero historical page leaves, appending one page builds exactly one, and
redundant old dead-page updates build none. Refilling an empty prior frontier
while crossing into a larger tree also builds only the newly appended pages.

The ignored release benchmark compares 64 refreshes with 65,536 historical pages
and eight live pages. One focused run measured **0.188 ms for updates versus
760.631 ms for cold rebuilds**, with identical roots. A final-code run during
workspace validation measured 0.208 ms versus 1,180.220 ms; absolute timing varies
with host load, while the deterministic page-work checks establish the scaling. This isolates page/tree
work; it excludes generation of cell commitments, tile hashing, Mind execution,
and other simulation phases, and is not an end-to-end throughput claim.

Final validation passed: 595 workspace tests (32 ignored), workspace all-target/
all-feature Clippy, engine/web WASM library Clippy, Rust formatting, whitespace
checks, and all 59 schema constants/mirrors. The manual benchmark ran separately
and its root-equivalence assertions passed.
