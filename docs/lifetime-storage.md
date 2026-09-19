# Lifetime cell-storage cleanup — 2026-09-05

This follows the [code review](code-review-2026-09-05.md). Long simulations no longer
retain a cell slot, scheduler position, and two hash commitments for every key ever
allocated. The change preserves canonical state ordering, hash format 7, checkpoint
schema 6, replay bytes, and simulation rules.

## Organization and storage bounds

- `blob_engine/src/resolution/cell_storage.rs` owns `CellStore` and its iterators.
  It retains the public store API and canonical active-key ordering. `reference.rs`
  continues to expose the store to existing callers.
- `blob_engine/src/resolution/paged_slots.rs` supplies shared sparse storage for
  cells and metabolic scheduler positions. Each page contains 256 optional slots;
  each directory indexes 256 pages. A small ordered map locates directories, then
  lookup indexes the page and slot directly. Removing the final value releases
  its page; removing the final page releases its directory. Mutable traversal
  supports mixed forward/reverse iteration without unsafe code or iterator
  allocations. Native parallel work splits at page granularity.
- The metabolic exhaustion heap retains exact indexed replacement and canonical
  `(deadline, key)` ordering. Its position pages release dead keys. Capacity is
  trimmed when it exceeds four times the live heap length, with a 1,024-entry
  floor to avoid repeatedly reallocating small heaps.
- `hashing.rs` retains leaf and private-memory commitments in ordered maps
  containing live keys only. Full refresh also removes private-memory commitments
  for deaths that were subsumed by an earlier all-cells invalidation.
- `hashing/cell_tree.rs` stores the cell Merkle tree. A subtree with no live cells
  collapses to its exact hash. The path to the last logical page stays expanded
  to support births within that page without rebuilding an empty historical
  subtree. Planner hash seeds carry shared tile hashes and cell-tree shape;
  cell commitments are reconstructed from live state, as planner restore already
  required.

For `L` live cells, `P` occupied slot pages, `D` occupied directories, and `H`
logical historical hash pages, retained slot storage is proportional to
`256P + 256D`; `P <= L` and `D <= P`. Hash commitments take `O(L)` entries and
retained cell-tree nodes take at most `O((P + 1) log(H + 1))` space, capped by the
full tree size. This does not impose a fixed memory cap: live populations and
widely separated live keys still cost memory. Dense hash maps also have more
per-entry overhead than the former flat arrays.

## Remaining time cost

The [2026-09-11 update follow-up](hash-update-2026-09-11.md) eliminates historical
page rebuilding during ordinary appends and dense/all-cell refreshes. It reuses
the historical prefix and updates live/emptied pages through shared tree paths.
The [trusted-restore follow-up](hash-restore-2026-09-11.md) also shares compact
cell-tree branches through planner checkpoints and rechecks only live-page
commitments on an unchanged seeded restore. Copy-on-write isolates later branches.

Format 7 commits the index of each empty historical hash page, so empty pages at
different indices have different hashes. A cold cache still visits all `H` logical
pages, and arbitrarily reactivating a collapsed old subtree reconstructs its hashes.
Those paths remain separate from the optimized monotonic-birth frontier.

The independent full-hash oracle intentionally retains its dense page/tree
implementation for comparison, so it also allocates temporary memory proportional
to historical pages. `CheckpointLimits.max_cell_slots` remains a resource guard
(default 4,000,000 historical keys), even though decoding no longer builds the old
lifetime-sized cell arrays. Raising it requires budgeting reconstruction time as
well as live-cell memory.

## Validation

Regression tests cover 20,000 sparse births with eight survivors; keys up to
`u64::MAX`; replacement and mixed-direction mutable iteration; directory/page
release; and 5,000 scheduler births with sparse old keys, checked against an
ordered deadline oracle. Compact Merkle roots are compared with the independent
dense tree across empty, partial, non-power-of-two, pruned, and reactivated trees.

A state-hash regression jumps to 1,048,576 historical keys, verifies full-hash
agreement through private-memory changes, births, and deaths, then removes all
cells. It retains zero cell commitments and 25 cell-tree nodes afterward. These
are deterministic structure-count checks, not allocator or RSS measurements.

Final gates: `cargo test --workspace --all-features --locked` passed 588 tests
with 31 ignored. Workspace Clippy (all targets/features) and engine/web WASM
library Clippy passed with warnings denied. Formatting, whitespace checks, and
all 59 schema constants/mirrors passed. Two ignored manual benchmarks were run
separately above the normal suite; their correctness assertions passed. Full
Rust/AssemblyScript/TinyGo conformance remains unrun because TinyGo is unavailable.

## Performance check

Three alternating runs on the same host, using release builds of baseline commit
`6dd66718d03507108d38844b623563a6f82fbfe0` and this working tree, produced these
medians. These are focused manual benchmarks, not end-to-end match throughput.

| Workload | Baseline dense storage | Sparse storage |
| --- | ---: | ---: |
| 2,000,000 indexed lookups, 100,000 live cells | 4.615 ms | 9.671 ms |
| 8 parallel passive steps, 200,000 cells, 512×512 board | 52.755 ms | 57.757 ms |

The lookup microbenchmark costs about 2.1× as much as flat indexing (about 5 ms
extra across two million lookups), while still beating a per-cell ordered-map
lookup by about 24×. The combined passive workload is about 9.5% slower; digestion
and rebuilding scheduler positions account for the main increase. Every passive
run checks exact cell and tile equality with its serial oracle. The workload
excludes per-step hashing and Mind execution. Dense/full hashing now uses ordered
live commitment maps and needs separate end-to-end performance characterization.

An initial one-level page map took 81 ms for the lookup workload. The retained
implementation uses grouped directories to avoid that regression. These tradeoffs
buy bounded storage after long churn; this change does not claim a throughput
improvement over the former dense arrays.

The [September 11 end-to-end follow-up](capacity-2026-09-11.md) now characterizes
verified and on-demand host capacity through 1024×1024 boards. It found and reduced
a dense-hash regression by combining live-page/leaf hashing and reusing stable
maps. The final verified cases remain 4.6–6.9% below the original dense baseline.
Separate eight-live-cell canonical fixtures after four million historical keys
show peak RSS of 319.55 versus 4.22 MiB and 100 active-metabolism batches of
360.27 versus 5.71 ms. Creating the churn and guest execution are outside this
fixture; cold restore still visits historical empty hashes.
