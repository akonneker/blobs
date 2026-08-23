# Performance baseline

Measured 2026-08-21 on an Apple M3 Max (12 performance and 4 efficiency
cores), using release builds. These numbers are local capacity measurements,
not portable guarantees.

## Native authoritative engine

Run with:

```sh
cargo run --release -p blob_engine --bin capacity_benchmark
```

The benchmark maintains a stable population whose cells all choose `Wait` and
disables passive processes that would change the workload. It includes local
Mind input projection and decision, commitment, resolution, state hashes,
deltas, and host projection synchronization. Fully synchronized waits are a
best-case workload because board-wide work is amortized across one large batch.

| Board | Cells | Actions/s | Milliseconds/batch |
| --- | ---: | ---: | ---: |
| 32x32 | 100 | 57,075 | 1.75 |
| 32x32 | 500 | 75,706 | 6.60 |
| 64x64 | 100 | 28,598 | 3.50 |
| 64x64 | 1,000 | 67,019 | 14.92 |
| 64x64 | 3,000 | 68,311 | 43.92 |
| 128x128 | 1,000 | 45,201 | 22.12 |
| 128x128 | 5,000 | 64,403 | 77.64 |
| 128x128 | 10,000 | 61,298 | 163.14 |
| 256x256 | 1,000 | 20,078 | 49.81 |
| 256x256 | 10,000 | 50,171 | 199.32 |
| 256x256 | 30,000 | 41,263 | 727.05 |

The resolver-only 5,000-way collision benchmark completed at approximately
468,000 actions/s (10.68 ms per batch). The large gap from full-engine
throughput shows that resolution is no longer the dominant end-to-end cost.

### First scalability slice

Phase telemetry showed that the two canonical full-state SHA passes consumed
about 90.5% of a sparse 256x256/1,000-cell batch. Native resolution now hashes
the immutable pre- and post-state snapshots concurrently without changing the
canonical hash format. Sparse occupancy conflicts also use claim-proportional
scratch storage rather than allocating a counter for every board tile.

Selected before/after release results:

| Board | Cells | Before actions/s | After actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 28,598 | 51,086 | 79% |
| 128x128 | 1,000 | 45,201 | 78,674 | 74% |
| 256x256 | 1,000 | 20,078 | 34,099 | 70% |
| 256x256 | 10,000 | 50,171 | 72,952 | 45% |

Full-state hashing still consumes approximately 50% to 84% of these optimized
batches, so incremental authenticated state commitment is the next priority.

### Paged state commitment slice

Canonical hash format 5 introduced fixed 256-resource
tile and cell pages, separate deterministic Merkle trees, and a root that binds
time, counts, page geometry, and the compiled ruleset. The live simulation
caches page hashes, rehashes only dirty pages, and parallelizes page encoding on
native builds. A from-scratch implementation remains the conformance oracle.
Checkpoint, replay, and host-state versions advanced to reject histories using
the previous state commitment.

Selected results after this slice:

| Board | Cells | Initial baseline | Current actions/s | Total improvement |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 28,598 | 81,192 | 184% |
| 128x128 | 1,000 | 45,201 | 172,840 | 282% |
| 256x256 | 1,000 | 20,078 | 137,690 | 586% |
| 256x256 | 10,000 | 50,171 | 132,434 | 164% |

At 10,000 cells, work outside the resolver now accounts for roughly 71% of the
end-to-end batch. Delta-driven host projection and removal of duplicate state
copies therefore move ahead of further resolver micro-optimization.

### Delta-driven projection slice

The engine now applies the canonical action delta plus conservative passive-
physics invalidations directly to its host view, and `blob_game` applies the
same invalidation list to its UI-facing mirror. This distinction matters
because passive physics advances before the reversible action delta's source
frontier. The live path no longer clones the entire canonical state, cell map,
coordinate maps, and world after every event batch. Full reconstruction
remains available for import and checkpoint restore. Per-tick host metadata
synchronization also visits only cells that are ready to decide.

Selected results after this slice:

| Board | Cells | Paged-hash actions/s | Current actions/s | Slice improvement |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 81,192 | 81,228 | ~0% |
| 128x128 | 1,000 | 172,840 | 173,624 | ~0% |
| 256x256 | 1,000 | 137,690 | 142,207 | 3% |
| 256x256 | 10,000 | 132,434 | 143,620 | 8% |

The synchronized-wait benchmark changes every cell's scheduling/outcome state
on every batch, so its cell delta is intentionally dense. The main saving in
that workload is avoiding board- and map-wide projection copies. Sparse and
fragmented online batches are quantified by the completion-frontier benchmark
below.

### Mutation-journal slice

Action resolution now records the first before-value for each written resource
directly in canonical keyed storage. It clones only the corresponding
after-values. This removes full post-state materialization and comparison while
also making rollback independent of a full completion snapshot. Debug builds
still run the original full-state delta as an assertion oracle. The journal
restores all touched resources if resolution fails after partial mutation.

| Board | Cells | Projection-slice actions/s | Current actions/s | Slice improvement |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 81,228 | 87,477 | 8% |
| 128x128 | 1,000 | 173,624 | 191,436 | 10% |
| 256x256 | 1,000 | 142,207 | 161,068 | 13% |
| 256x256 | 10,000 | 143,620 | 144,191 | ~0% |

Post-state materialization is now zero in phase telemetry. At 10,000 cells,
roughly 73% of elapsed time remains outside the instrumented resolver, so the
next useful work is sparse/lazy passive physics and reducing hot-path host/Mind
memory traffic rather than further delta micro-optimization.

### Sparse passive-field slice

The simulation now maintains exact derived frontiers for nonzero signal and
diffuse-energy tiles. Signal decay, diffusion event scheduling, passive view
invalidation, and dirty hash-page marking traverse those frontiers instead of
the board. Synchronous diffusion no longer clones the full field or allocates a
fresh board-sized accumulator on every step: it uses the active sources plus a
reused accumulator whose touched destinations are cleared sparsely. Canonical
tiles remain the sole authority, and checkpoint restore rebuilds both indexes.

Run the sparse kernel against its former dense semantic oracle with:

```sh
cargo test --release -p blob_engine \
  benchmark_sparse_diffusion_against_dense_oracle -- --ignored --nocapture
```

With three equal sparse sources and 16 spreading steps, both kernels ended at
the same 611 active tiles:

| Board | Sparse milliseconds | Dense milliseconds | Speedup |
| --- | ---: | ---: | ---: |
| 128x128 | 0.282 | 1.015 | 3.6x |
| 256x256 | 0.262 | 4.273 | 16.3x |
| 512x512 | 0.261 | 34.721 | 133.0x |

This is a field-kernel diagnostic, not an end-to-end action benchmark. It
demonstrates the intended scaling property: for the same active frontier, work
is nearly independent of inactive board area. Dense fields still need a
density crossover to a contiguous parallel kernel. The remaining passive
slice is canonical lazy metabolism.

### Indexed passive-cell slice

Derived canonical-order sets now identify cells with gut contents, cells with
assimilated energy, and zero-energy cells awaiting deterministic death cleanup.
They drive digestion, metabolic exhaustion scheduling, passive projection and
hash invalidation, and death collection. Checkpoint restore derives the sets;
every gut/energy mutation maintains them; failed action resolution rebuilds
them from the immutable completion snapshot. Invalid checkpoint digestion
remainders are now rejected explicitly.

The cell kernels use a density crossover: sparse digestion follows only the
gut-bearing set, while an all-active metabolism pass traverses the canonical
cell map directly. Run the sparse digestion kernel against the former dense
oracle with:

```sh
cargo test --release -p blob_engine \
  benchmark_sparse_digestion_against_dense_oracle -- --ignored --nocapture
```

For 32 digesting cells over 32 updates:

| Population | Sparse milliseconds | Dense milliseconds | Speedup |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.058 | 0.068 | 1.2x |
| 10,000 | 0.065 | 0.806 | 12.4x |
| 50,000 | 0.085 | 16.355 | 192.3x |

This is a passive-cell kernel diagnostic. Metabolism remains dense for a
living population because assimilated energy and fractional progress are
materialized fields in the current canonical state. Making that process truly
lazy requires a versioned representation change; otherwise state hashes and
checkpoints would silently cease to describe the observable cell state.

### Copy-on-write private-memory slice

Canonical cell private memory now uses immutable shared ownership. Cloning a
completion snapshot, checkpoint state, reversible delta, or rollback frontier
increments ownership rather than allocating and copying the unchanged byte
payload. Installing a Mind's explicit next memory creates a new immutable
buffer, so earlier snapshots cannot observe the change. The Mind ABI still
receives an owned `Vec<u8>`; no shared state crosses the isolation boundary.
Canonical hashing, checkpoint, replay, and attestation encodings continue to
write the same raw bytes and therefore did not require a format-version change.

Run the memory-heavy resolver diagnostic with:

```sh
cargo test --release -p blob_engine \
  benchmark_private_memory_snapshot_churn -- --ignored --nocapture
```

For 5,000 cells carrying 2,048 private bytes over 20 fully synchronized Wait
batches, the slice changed 9.789 ms/batch (510,781 actions/s) to 9.653 ms/batch
(517,966 actions/s), about a 1.4% throughput improvement. More importantly, it
removes roughly 20.5 MB of transient payload copying per batch from the
pre-state and before/after delta copies. The dense benchmark still changes
every cell and therefore rehashes the private bytes in dirty canonical pages;
that hashing dominates the saved allocation time. Separating hot cell fields
from cold-memory commitments was the next representation-level opportunity and
is implemented in the compact layout slice below.

### Bounded completion-frontier and disabled-physics slice

Release resolution now clones cell state only for due actors and occupants of
their declared target tiles. All tiles remain immutable snapshot inputs because
action validation reads terrain, occupancy, and local energy, but a fragmented
batch no longer clones every unrelated resident cell. The first-write mutation
journal supplies rollback and delta before-values independently. Debug builds
retain a complete canonical prestate and assert that every optimized delta is
identical to the full-state oracle.

The same workload exposed two dense scans when metabolism was disabled: passive
hash invalidation searched all living cells for fractional progress, and the
metabolism kernel traversed them again to clear already-zero remainders. A
disabled digestion or metabolism rate now requires zero canonical remainder,
making both paths true O(1) no-ops. The immutable compiled-ruleset hash is also
returned from the authenticated-state cache instead of being recomputed.

Run the fragmented online diagnostic with:

```sh
cargo test --release -p blob_engine \
  benchmark_fragmented_completion_snapshot -- --ignored --nocapture
```

For 50,000 resident cells carrying 2,048 private bytes each, with only eight
Wait actions due per batch:

| Stage | Milliseconds/batch | Prestate ms | Passive ms |
| --- | ---: | ---: | ---: |
| Full cell snapshot | 10.359 | 2.000 | not yet isolated |
| Bounded cell frontier | 8.788 | 0.268 | 5.66 |
| Disabled-physics fast path | 2.708 | 0.259 | 0.001 |

The combined reduction is 74%, or about 3.8x throughput. At the final stage,
authenticated state hashing takes 2.436 ms—about 90% of batch time—because one
dirty 256-cell page encodes roughly 512 KiB of mostly unchanged private memory.
The cell-leaf commitment slice below addresses that measured bottleneck.

### Cell-leaf and cold-memory commitment slice

Canonical hash format 6 commits each live cell through a domain-separated leaf.
Each 256-key cell page contains the ordered leaf commitments rather than the
full cell encodings. A second cached domain-separated commitment covers private
memory; hot cell changes reuse it, while a Mind decision invalidates it only
when the returned memory bytes actually differ. The live cache updates dirty
leaves and affected pages for sparse batches, with a parallel full-leaf
crossover for dense batches. The from-scratch state hash independently rebuilds
both levels and remains the conformance oracle.

On the 50,000-resident/8-due fragmented diagnostic, hash time fell from 2.436
ms to 0.052 ms and total resolution fell from 2.708 ms to 0.291 ms. This slice
is about 9.3x faster by itself; combined with the preceding snapshot and passive
fast paths, the batch is about 35.6x faster than the original 10.359 ms.

The memory-heavy dense diagnostic also improved rather than regressed: 5,000
cells with 2,048 private bytes fell from 9.653 to 7.546 ms/batch, reaching
662,646 actions/s. The standard synchronized end-to-end benchmark measured
347,050 actions/s at 64x64/100 cells, 282,109 at 128x128/1,000 cells, and
148,439 at 256x256/10,000 cells. At 30,000 cells it rose from 43,997 to 73,014
actions/s; host/Mind work outside the resolver is now 88% of elapsed time.

### Ordered host-decision merge slice

The authoritative host previously built a sorted decision vector and then, for
every ready cell, linearly searched that vector to determine whether it needed
a fallback Wait. A fully ready population therefore performed quadratic
`ready_cells * decisions` comparisons before committing any action. The host
now merges the two canonical actor-ordered frontiers once. Supplied decisions
remain actor-ordered, missing Minds still produce exact fallback Wait
commitments, and an out-of-frontier decision is rejected.

The capacity diagnostic now also records non-authoritative host phases for
ready-frontier discovery, observation projection, Mind execution, action
commit, passive invalidation, replay recording, and host projection.

| Board | Cells | Before actions/s | Current actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 3,000 | 215,535 | 257,566 | 20% |
| 128x128 | 10,000 | 148,098 | 232,236 | 57% |
| 256x256 | 10,000 | 148,439 | 237,663 | 60% |
| 256x256 | 30,000 | 73,014 | 212,353 | 191% |

At 30,000 cells, batch time fell from 410.88 to 141.27 ms. Observation
projection is now the largest measured host phase at roughly 40% of elapsed
time, followed by action commit at 9% and host-view synchronization at 7%.
That makes observation construction and a projection-free verification host
the next evidence-backed targets.

### Reusable isolated-observation slice

Observation construction previously allocated fresh slot and private-memory
vectors for every ready cell, retained the complete observation population,
and traversed each neighborhood once for neighbor cues and again for tile
fields. It now fills one reusable scratch observation per native Mind instance,
completely overwriting all fields before the next invocation. Neighbor and tile
fields are produced in one traversal. Only compact cell/randomness work remains
queued, while the submitted-WASM host reuses the same projection scratch before
serializing each independently owned ABI input.

The reuse is host-local only: no scratch state is canonical, no buffer is shared
between simultaneous Mind invocations, and a conformance test projects two
different observers through the same scratch input and compares the result with
a fresh projection. This preserves anonymous local visibility and private-memory
isolation.

| Board | Cells | Before actions/s | Current actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 3,000 | 257,566 | 260,178 | 1% |
| 128x128 | 10,000 | 232,236 | 258,124 | 11% |
| 256x256 | 30,000 | 212,353 | 236,239 | 11% |

At 30,000 cells, batch time fell from 141.27 to 126.99 ms. The projection-only
portion of the refined telemetry is about 27% of elapsed time; private-random
derivation and compact work preparation now appear in host overhead rather than
being folded into projection. Observation construction remains the largest
single host phase, but authenticated hashes, intent validation, action commit,
and decision-work preparation are now individually within roughly 9–14%.

### Bounded parallel Mind-input pipeline

For teams with multiple isolated native Mind instances, the host now assigns
contiguous actor-ordered chunks to those instances and runs private-random
derivation, scratch observation construction, reset, and decision execution in
parallel. Each worker has exclusive mutable access to one Mind and one scratch
input. Results are sorted by actor before commitment, so worker scheduling is
noncanonical. The default crossover is 256 ready cells and can be changed or
disabled as host policy; small games retain the serial path.

On a controlled 256x256/30,000-cell synchronized-Wait run:

| Native Minds | Actions/s | Milliseconds/batch | Parallel observation batches |
| ---: | ---: | ---: | ---: |
| 1 | 245,710 | 122.10 | 0/20 |
| 2 | 269,102 | 111.48 | 20/20 |
| 4 | 290,584 | 103.24 | 20/20 |
| 8 | 327,512 | 91.60 | 20/20 |

Eight Minds improve throughput by 33% over the same one-Mind executable and
reduce batch latency by 25%. Observation plus private-random preparation falls
from roughly 36% of elapsed wall time to 6%. A 100-cell run stays below the
crossover (`0/200` parallel observation batches) and showed no measurable
scheduling penalty.

The submitted-WASM host applies bounded parallel chunks to observation plus ABI
serialization, then moves each owned byte buffer into its persistent plugin
worker rather than cloning it. A pool now activates at most one worker per 64
invocations rather than dispatching every fragmented batch across its full
capacity. At the current 841-cell setup limit, this raises a 16-worker-capacity
pool from roughly 6.6–7.1k to 9.7–10.4k actions/s and makes it about 16% faster
than the one-worker median. Pristine WASM instance construction remains the
dominant online cost and is the next target for overlap/batching work.

### Atomic ordered bulk-commit slice

The host previously committed each ready decision separately. A synchronized
frontier therefore repeated canonical cell-tree lookups, dirty-set insertion,
completion-time tree lookup, and request/private-memory cloning for every
actor. It also could return a fatal error after earlier actors had already been
mutated.

The resolver now accepts one strictly actor-ordered decision slice. It
preflights the complete frontier before mutation, using a native parallel
crossover at 2,048 decisions while collecting errors back in canonical actor
order. It then applies prepared actions in one pass, groups actors by completion
time, and uses full-cell hash invalidation at the same density crossover as
hashing. Unchanged private-memory bytes keep their existing canonical `Arc`
allocation. Input requests and memory remain owned by the host and are moved
directly into replay commitments after receipts return. Semantic action
rejection remains a successful commitment to the minimum Wait; only fatal
ABI/invariant errors abort the entire batch.

| Native Minds | Before actions/s | Current actions/s | Before ms/batch | Current ms/batch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 245,710 | 267,027 | 122.10 | 112.35 |
| 8 | 327,512 | 421,782 | 91.60 | 71.13 |

At 30,000 cells this slice adds 9% throughput with one Mind and 29% with eight.
The measured commit phase fell from roughly 12.0 to 3.8 ms in the one-Mind run
and from 19.1 to 4.5 ms in the eight-Mind run. At eight Minds the remaining
large phases are authenticated hashes at about 20%, intent validation at 18%,
host projection at 11%, and prestate materialization at 10%.

### Dense authenticated-commitment slice

Dense batches previously materialized every refreshed per-cell leaf into a
`BTreeMap`, then traversed that tree again to construct 256-key page hashes.
The incremental cache now stores derived private-memory and leaf commitments in
key-indexed optional slots. Dense refresh workers hash one canonical key page
at a time and emit its leaf cache entries and page hash together, while sparse
updates retain direct single-key invalidation. This changes only the cache
representation: the authoritative cell map, page boundaries, leaf order,
Merkle construction, and hash format 6 byte contract are unchanged. Native
hash encoding uses ring's hardware-accelerated SHA-256 implementation; WASM
retains the RustCrypto implementation, and conformance compares both against
the same fixed canonical output.

On a paired 256x256/30,000-cell synchronized-Wait run, the eight-Mind case
increased from 411,431 to 480,332--484,812 actions/s (17--18%), reducing batch
latency from 72.92 to 61.88--62.46 ms. Authenticated hashing fell from 19.9% of
wall time to 6--7%. A controlled one-Mind run reached 299,625 actions/s at
100.13 ms/batch, compared with 267,027 actions/s after the preceding bulk-
commit slice.

The next resolver-side target is intent validation, now roughly 19--21% of the
eight-Mind run. End-to-end one-Mind throughput is instead dominated by serial
observation projection, so server and browser execution policy should choose
between more isolated Mind instances, parallel matches, and projection-free
verification according to frontier size.

### Snapshot-free intent and linear-journal slice

Completion resolution previously cloned every due actor and target occupant
into a separately searchable cell snapshot, even though validation completes
before any canonical cell mutation. Candidate construction then searched that
snapshot twice per actor. The resolver now validates directly against the
immutable canonical cells, carrying only the snapshot attack mass and target
guard bit needed after mutation begins. The tile snapshot remains authoritative
for simultaneous occupancy, terrain, and energy reads.

The mutation journal now captures actor-ordered before-states while completing
pending actions in the same linear traversal of the canonical cell map.
Additional victims, births, and interrupted cells remain in a sparse ordered
side map. Delta generation merges those streams and walks post-state cells in
order instead of performing a tree lookup for every changed actor. These are
derived execution structures only; reports, rollback behavior, deltas, hashes,
and completion-snapshot semantics are unchanged.

On a controlled 256x256/30,000-cell synchronized-Wait run:

| Native Minds | Before actions/s | Current actions/s | Before ms/batch | Current ms/batch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 299,625 | 362,164 | 100.13 | 82.84 |
| 8 | 479,867 | 674,133--697,222 | 62.52 | 43.03--44.50 |

The eight-Mind path gains 40--45% throughput over the immediately preceding
executable. Prestate materialization falls from 11.8% of wall time to under 1%,
and intent validation falls from 18.8% to roughly 10%. The fragmented
50,000-resident/8-due/2-KiB-private-memory diagnostic remains sparse at 0.262
ms/batch, including 0.015 ms of hashing and 0.002 ms of delta construction.

No resolver phase now exceeds roughly 11% of this dense end-to-end run. The
largest named costs are host projection, observation construction,
delta generation, commit, validation, and hashing; further work should optimize
shared data movement rather than add more resolver conflict machinery.

### Host-visible no-op projection slice

The compatibility host view previously reprojected every changed canonical
cell twice: it checked coordinate maps even when the canonical position was
unchanged, then copied private memory and scalar fields into the public `Cell`
view. A synchronized Wait frontier therefore copied about 60 MiB of unchanged
2-KiB memories and performed tens of thousands of unnecessary hash-map lookups
per batch.

Projection now skips coordinate work when a delta proves the position is
unchanged, retains `Vec` capacity when memory really changes, and linearly
merges already-sorted passive/delta invalidation keys. Native Mind chunks also
compute a non-authoritative host-visible-no-op hint while their isolated input
is still available. A Wait with no signal, unchanged private memory, and no
prior guard state can bypass scalar/memory reprojection, provided the canonical
delta and passive frontier prove that no attack or passive physics changed a
visible field. Submitted overrides compute the same hint from the current host
view. The hint never suppresses event invalidation and never enters canonical
state, Mind input, replay, hashes, or checkpoints.

On the 256x256/30,000-cell synchronized-Wait workload, the eight-Mind path rose
from 674,133--697,222 to 787,690--806,863 actions/s, reducing batch latency from
43.03--44.50 to 37.18--38.09 ms. Host projection fell from roughly 17--18% of
wall time before this work to about 2%. The one-Mind path remains approximately
flat around 360k actions/s because serial observation construction dominates;
the removed projection copying is exchanged for the necessary private-memory
equality check inside that single Mind loop. With multiple Minds the check is
distributed across isolated workers.

At this point the largest eight-Mind phases are observation construction at
roughly 14%, delta generation and commit near 12%, validation near 11%, and
hashing near 9%. Host projection is no longer a primary dense-game bottleneck.

### On-demand integrity mode

State commitments are verification output rather than simulation input. The
resolver therefore supports an explicit host-only `OnDemand` integrity mode
that keeps incremental dirty metadata current but omits the pre- and post-state
SHA-256 commitments from each batch. Physics, action outcomes, deltas, Mind
observations, private randomness, and canonical ordering are identical to
`Verified` mode. An explicit `state_hash()` or authoritative runtime-hash call
still commits the current state, including every change since the last hash.

Unverified reports carry absent hashes rather than a sentinel value. Replay
recorders reject them, and the engine refuses to start or resume replay bundle
or stream recording while on-demand mode is selected. Verified remains the
engine default and is required for browser replay, server re-execution, and
leaderboard attestation. `blob_rl::BlobEnv` selects on-demand mode for local
training and trusted rollouts.

On a paired 256x256/30,000-cell synchronized-Wait run:

| Native Minds | Verified actions/s | On-demand actions/s | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 473,518 | 509,042 | 7.5% |
| 8 | 867,671 | 1,074,917 | 23.9% |

The eight-Mind on-demand result is 27.91 ms/batch. Its hash phase is zero; the
largest named costs become observation construction, delta generation, action
commit, and validation. Use the fourth benchmark argument to compare policies:

```sh
cargo run --release -p blob_engine --bin capacity_benchmark -- \
  256 30000 8 on-demand
```

### Batched observation and private-randomness slice

Large decision frontiers previously performed a canonical `BTreeMap` search
for each observer and every visible neighboring occupant. The engine now builds
one read-only, batch-scoped key index and shares it across observation workers.
Dense recent key spaces use direct indexed lookup; sparse long-running key
spaces fall back to the canonical tree rather than allocating in proportion to
the largest historical key. The index borrows immutable canonical cells, is
discarded after projection, and never enters a Mind input or permits peer
access.

Complete slot projection now walks the compiled target row and static offsets
directly instead of repeating bounds checks and target arithmetic for every
slot. Native, submitted-WASM, and RL projection use the same path. RL tensor
collection additionally reuses one isolated input buffer across its frontier,
removing a 2-KiB private-memory allocation per cell while still copying the
current cell's bytes before policy code sees them.

Decision randomness retains the exact domain-separated SHA-256 definition.
Native hosts use the hardware-accelerated backend and clone a match-scoped hash
prefix that has already absorbed the domain and secret; WASM retains the
portable backend. Fixed golden vectors cover both the match secret and derived
cell bytes. Telemetry now separates index construction, cell projection,
private-random derivation, and Mind execution.

On the 256x256/30,000-cell synchronized-Wait workload:

| Integrity | Native Minds | Before actions/s | Current actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| On-demand | 1 | 509,042 | 767,290 | 50.7% |
| On-demand | 8 | 1,074,917 | 1,113,459 | 3.6% |
| Verified | 1 | 473,518 | 670,073 | 41.5% |
| Verified | 8 | 867,671 | 884,600 | 2.0% |

The one-Mind on-demand profile now assigns roughly 34% to observation
projection, 5% to private randomness, and under 1% to index construction. An
RL-specific 5,000-cell diagnostic produced identical tensors and action masks
in 9.01 ms instead of 16.82 ms, a 1.87x observation-frontier speedup:

```sh
cargo test --release -p blob_rl \
  benchmark_batched_rl_observation_projection -- --ignored --nocapture
```

### Explicit retained-memory slice

The Mind v5 ABI now returns a tagged private-memory operation: `Retain` or
`Replace(bytes)`. A retained result carries no private-memory payload back over
the native/WASM boundary, performs no equality scan during commitment, keeps
the actor's existing canonical `Arc<[u8]>`, and enters replay as a one-byte
tag. It does not lend canonical memory to the Mind: inputs remain isolated,
owned copies and fresh WASM instances remain mandatory. `Replace([])` is
distinct from `Retain`, and both variants have converter, resolver, and replay
conformance coverage.

The capacity benchmark accepts a fifth `retain` or `replace` argument for a
same-binary comparison. With 2-KiB private state on the 256x256/30,000-cell
synchronized-Wait workload:

| Integrity | Native Minds | Replace actions/s | Retain actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| On-demand | 1 | 613,430 | 738,495 | 20.4% |
| On-demand | 8 | 1,017,774 | 1,292,211 | 27.0% |
| Verified | 8 | 821,214 | 996,816 | 21.4% |

```sh
cargo run --release -p blob_engine --bin capacity_benchmark -- \
  256 30000 8 on-demand retain
```

For a 2-KiB unchanged state, each replay commitment is also 2,056 bytes
smaller (the omitted bytes plus their length prefix). Actual gains depend on
how often a Mind changes memory; stateful Minds still use bounded `Replace`.

## Submitted-WASM server verification

Run with:

```sh
cargo test --release -p blob_game \
  benchmark_reference_wasm_verification_capacity -- --ignored --nocapture
```

This benchmark uses `simple_mind`, exact reference-ABI serialization, and a
fresh pristine Extism/Wasmtime instance for every decision. It excludes module
compilation and initial lazy setup, but includes the rest of authoritative game
execution. Its mixed movement/wait behavior fragments completions and is more
representative of an online match than synchronized waits.

| Board | Actual cells | Workers/team | Actions/s | Milliseconds/event batch |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 1 | 3,756 | 5.20 |
| 64x64 | 100 | 16 | 3,470 | 5.63 |
| 64x64 | 841 | 1 | 6,153 | 26.39 |
| 64x64 | 841 | 16 | 5,148 | 31.54 |
| 128x128 | 841 | 16 | 4,206 | 38.60 |

After paged commitments, delta-driven projection, and the mutation journal, the
same local benchmark measured:

| Board | Actual cells | Workers/team | Actions/s | Milliseconds/event batch |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 1 | 4,935 | 3.96 |
| 64x64 | 100 | 16 | 4,254 | 4.60 |
| 64x64 | 841 | 1 | 7,781 | 20.87 |
| 64x64 | 841 | 16 | 6,543 | 24.81 |
| 128x128 | 841 | 16 | 6,321 | 25.68 |

With reusable projection, bounded preparation chunks, in-worker randomness
derivation, and ownership transfer of serialized inputs, the latest run
measured:

| Board | Actual cells | Workers/team | Actions/s | Milliseconds/event batch |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 1 | 8,385 | 2.33 |
| 64x64 | 100 | 16 | 9,025 | 2.17 |
| 64x64 | 841 | 1 | 8,648 | 18.77 |
| 64x64 | 841 | 16 | 10,458 | 15.52 |
| 128x128 | 841 | 16 | 10,221 | 15.88 |

These fragmented frontiers are too small to amortize dispatch across all 16
pristine-WASM workers. The active-worker crossover keeps the compiled pool but
selects only as many workers as the current ready frontier can support.

The batch-scoped observation index, direct compiled-slot traversal, and
match-prefixed native randomness derivation also accelerate preparation for
submitted WASM without relaxing pristine-instance execution:

| Board | Actual cells | Workers/team | Actions/s | Milliseconds/event batch |
| --- | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 1 | 9,758 | 2.00 |
| 64x64 | 100 | 16 | 10,017 | 1.95 |
| 64x64 | 841 | 1 | 10,394 | 15.62 |
| 64x64 | 841 | 16 | 11,918 | 13.62 |
| 128x128 | 841 | 16 | 12,407 | 13.09 |

This is an approximately 14--22% gain over the preceding submitted-WASM
measurements. Pristine instance restoration/execution remains dominant.

### Pooled pristine-WASM allocation slice

The default submitted-Mind host now uses Wasmtime's pooling allocator. It
reuses bounded, decommitted allocation slots and memory mappings underneath
fresh Extism plugins; it does not reuse a live plugin or guest instance. Guest
memory is reset by instance destruction/recreation, and the stateful
memory/global canary plus worker/hash parity tests remain mandatory. The
`memory.use_pooling_allocator` host setting can disable pooling for portability
or controlled comparison.

Four same-binary runs, including two with the allocator order reversed,
produced the following ranges:

| Board | Cells | Workers | On-demand actions/s | Pooling actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64x64 | 100 | 1 | 8,118--9,231 | 10,672--11,518 | 16--37% |
| 64x64 | 100 | 16 | 8,575--9,816 | 11,785--12,444 | 20--45% |
| 64x64 | 841 | 1 | 9,562--9,873 | 11,490--12,568 | 16--31% |
| 64x64 | 841 | 16 | 10,826--11,498 | 27,527--28,654 | 149--163% |
| 128x128 | 841 | 16 | 10,154--11,460 | 26,575--28,233 | 138--178% |

The high-worker gain is the expected removal of contended virtual-memory
mapping work during parallel instance creation. Pool sizes and per-module
memory/table bounds are deliberately finite, while the on-demand path remains
available for modules or platforms outside those limits.

### Pipelined WASM preparation and batched dispatch slice

The submitted-WASM host now sends one contiguous batch to each active worker
instead of one request and response message per cell. For frontiers of at least
256 cells, observation projection and exact ABI serialization for each chunk
run in parallel with pristine-instance execution of the other chunks. Cap'n
Proto output is flattened once into its exact final capacity. Results are still
sorted by canonical actor key, and every individual decision still constructs
a fresh Extism plugin and guest instance.

A same-binary threshold A/B run measured:

| Board | Cells | Workers | Allocator | Barrier actions/s | Pipelined actions/s | Improvement |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 64x64 | 1,000 | 16 | pooling | 28,429 | 29,179 | 2.6% |
| 128x128 | 5,000 | 16 | pooling | 30,080 | 32,074 | 6.6% |
| 64x64 | 1,000 | 16 | on-demand | 9,840 | 10,269 | 4.4% |
| 128x128 | 5,000 | 16 | on-demand | 7,190 | 7,401 | 2.9% |

Single-worker and sub-threshold cases retain the direct path. The old fixed
ring-20 setup guard was also replaced by a board-derived bound, so the 1,000-
and 5,000-cell benchmark cases now contain their requested populations.
Per-cell placement logging is aggregated to avoid making large online match
startup proportional to terminal I/O.

### Restricted Extism-compatible executor slice

The submitted-Mind host can now select an `extism_compat` executor that loads
the same Extism PDK artifact directly into Wasmtime. It caches immutable
compiled code and a store-independent link plan. Each decision still creates a
new Store, instance, guest linear memory, host byte arena, allocation map,
input/output state, and resource limiter. No live guest state is reset or
reused.

The host implements only the deterministic Extism input and byte-memory
contract. WASI, configuration, variables, HTTP, custom host functions, and
logging are rejected when the module is admitted. Manifest memory bounds and
epoch deadlines remain active. Exact differential tests cover every maintained
Mind, the mutable-global isolation canary, one versus four workers, malformed
imports, memory-range enforcement, and nonterminating code.

The same checks now form the named `extism_pdk_deterministic_v1` admission
profile. Match-verification format 2 commits that profile separately from the
Mind ABI and artifact hashes. Upload admission parses the artifact without
compiling it; local Extism and compatible pools repeat the gate before spawning
workers. This is match-setup work and is excluded from steady-state capacity.

A same-binary run on 2026-08-22 measured:

| Board | Cells | Workers | Allocator | Extism actions/s | Compatible actions/s | Improvement |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 64x64 | 100 | 1 | pooling | 11,035 | 19,590 | 78% |
| 64x64 | 100 | 16 | pooling | 11,373 | 19,805 | 74% |
| 64x64 | 1,000 | 1 | pooling | 12,722 | 24,753 | 95% |
| 64x64 | 1,000 | 16 | pooling | 28,270 | 44,959 | 59% |
| 128x128 | 5,000 | 16 | pooling | 29,276 | 47,260 | 61% |
| 64x64 | 1,000 | 1 | on-demand | 9,935 | 22,725 | 129% |
| 64x64 | 1,000 | 16 | on-demand | 9,881 | 23,496 | 138% |
| 128x128 | 5,000 | 16 | on-demand | 6,823 | 16,809 | 146% |

The large pooled cases improve end-to-end authoritative throughput by roughly
1.6--1.9x; the large on-demand cases improve by roughly 2.3--2.5x. This is an
experimental host-policy option rather than a new Mind target: developers keep
using an Extism PDK and the canonical Cap'n Proto Mind ABI. Stock Extism remains
the default and differential oracle. Maintained AssemblyScript and TinyGo
`wasm-unknown` canaries now cover two non-Rust PDKs under the same profile;
their setup-only admission tests do not change these steady-state measurements.

### Shared compatible-runtime slice

The restricted executor previously compiled an independent Wasmtime Engine,
Module, link plan, pooling allocator, and epoch ticker in every plugin worker.
A team pool now builds that immutable core once and shares it across its worker
threads. Each call still allocates a new Store, limiter, host byte arena,
allocation map, instance, memory, tables, and globals. Pool dimensions scale
with the maximum concurrent worker count, preserving the former aggregate slot
capacity. Stock Extism remains worker-local because its public compiled type can
contain non-thread-safe host user data.

Two release runs of the setup diagnostic measured:

| Workers | Per-worker compilation ms | Shared compilation ms | Startup speedup |
| ---: | ---: | ---: | ---: |
| 1 | 13.85--25.27 | 12.78--12.99 | 1.07--1.98x |
| 4 | 37.35--43.07 | 11.65--12.90 | 3.21--3.34x |
| 16 | 156.89--159.69 | 11.93--12.38 | 12.68--13.38x |

```sh
cargo test --release -p blob_game \
  plugin_pool::tests::benchmark_shared_compatible_runtime_startup \
  -- --exact --ignored --nocapture
```

The follow-up steady-state run confirmed full 16-worker pooled concurrency. The
64x64/1,000-cell case measured 48,371 actions/s versus 44,959 before this slice,
and the 128x128/5,000-cell case measured 50,557 versus 47,260. These 7--8%
single-run gains are consistent with better compiled-code and allocator
locality, but the primary guaranteed improvement is eliminating 15 redundant
compilations, engine/link plans, and tickers from a 16-worker team pool.

### Fresh-call arena and stage-profile slice

A release-only diagnostic now splits a no-op compatible call into fresh Store
setup, instance creation, export lookup, guest execution, result handling, and
implicit teardown. With pooling, one run measured 1.09 microseconds total:
0.27 microseconds for Store setup, 0.21 for instantiation, 0.12 for export
lookup, 0.03 for the guest call, and 0.02 for result inspection. The remaining
roughly 0.44 microseconds is loop/timing and destruction. On-demand allocation
measured 1.52 microseconds total. Fresh instance construction is therefore no
longer the dominant cost of a real maintained Mind call.

The compatible host's fresh byte arena now uses 128 inline bytes and its active
and free allocation sets use four inline records each. All three spill to the
heap without changing the bounded Extism memory contract. A test forces both
arena and metadata spillover. The 512-byte copy diagnostic improved from 2.26
to 1.52 microseconds per pooled call, or about 33%, and from 2.30 to 2.12
microseconds with the on-demand allocator. No Store, guest instance, input,
output, arena, allocation record, or mutable byte is reused between calls.

```sh
cargo test --release -p blob_game \
  extism_compat::tests::benchmark_fresh_call_stages \
  -- --exact --ignored --nocapture
cargo test --release -p blob_game \
  extism_compat::tests::benchmark_extism_byte_contract \
  -- --exact --ignored --nocapture
cargo test --release -p blob_game \
  extism_compat::tests::benchmark_maintained_mind_call \
  -- --exact --ignored --nocapture
```

The maintained `simple_mind` diagnostic uses a representative eight-slot,
2,016-byte observation and produces a 96-byte decision. It measured about
23.0 microseconds per pooled action: 22.8 microseconds inside fresh compatible
execution and only 0.15 microseconds decoding the returned decision. Full-game
runs remained inside their normal run-to-run variance (about 47--52 thousand
actions/s for the large 16-worker pooled cases). The next material target is
guest-side Cap'n Proto observation decode and policy execution, not weakening
the pristine-instance boundary.

### Compact slot-wire slice (Mind ABI v6)

The former slot schema represented each optional elevation, energy, plant,
signal, and neighbor channel with pointer-backed Cap'n Proto objects. ABI v6
uses one canonical visibility bitmap and inline scalar fields instead. This
does not alter the ergonomic `LocalObservation` type or any visibility
semantics: a bit distinguishes hidden from visible zero. Unknown bits,
neighbor details without neighbor presence, progress without activity, and
nonzero hidden values are rejected. The ABI version and schema hash changed;
legacy artifact compatibility is intentionally not retained at this stage.

The representative eight-slot native diagnostic changed from 2,176 bytes and
about 1.47 microseconds/decode to 1,152 bytes and 0.45 microseconds/decode: 47%
fewer bytes and roughly 69% less native decode time. The maintained-Mind WASM
diagnostic changed from 2,016 to 1,120 input bytes and from about 23.0 to 16.75
microseconds per pooled action, a 27% reduction while retaining a fresh Store
and guest instance.

Two end-to-end release runs measured compatible 16-worker pooled capacity at
54,166--57,041 actions/s for 1,000 cells and 55,374--55,635 actions/s for 5,000
cells. The preceding runs were roughly 47--51 thousand actions/s, so the large
authoritative improvement is about 10--15% after simulation and scheduling
overhead. Stock Extism also improved because it consumes the same smaller ABI.

### Indexed authoritative cell-store slice

The authoritative live simulation now stores cells in direct key-indexed slots
and keeps canonical active-key traversal separately. Each slot owns a boxed
cell, so deleting a historical key leaves one pointer-sized hole rather than a
full `CellState`. Hashes, checkpoints, replay, and equality traverse active keys
and therefore remain independent of physical slot capacity. Sparse passive
updates retain their key-indexed paths, and dense mutable traversal is used
only while the live key space remains dense.

Observation batches borrow the authoritative index directly. This removes the
temporary batch-scoped index, including its allocation and population pass,
without exposing any additional state to a Mind. Ordered completion mutation
also changed from a tree merge to one direct lookup per due actor.

The maintained release diagnostic measured 2,000,000 pseudorandom lookups over
100,000 live cells in 293.5 ms through `BTreeMap` and 7.70 ms through indexed
storage, a 38.1x lookup-kernel speedup. This is not an end-to-end claim.

On the 256x256/30,000-cell synchronized-Wait workload, repeated release runs
measured:

| Integrity | Native Minds | Previous actions/s | Indexed actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| On-demand | 1 | 767,290 | 873,485--910,732 | 13.8--18.7% |
| On-demand | 8 | 1,113,459 | 1,612,822--1,628,532 | 44.9--46.3% |
| Verified | 1 | 670,073 | 768,968 | 14.8% |
| Verified | 8 | 884,600 | 1,287,537 | 45.5% |

The 5,000-cell RL observation diagnostic measured 8.16 ms versus the preceding
9.01 ms batched result. Observation-index telemetry is now zero because no
frontier allocation remains.

```sh
cargo test --release -p blob_engine \
  resolution::reference::tests::benchmark_indexed_cell_lookup \
  -- --exact --ignored --nocapture
cargo run --release -p blob_engine --bin capacity_benchmark -- \
  256 30000 8 on-demand retain
```

### Compact hot/cold cell-layout slice

The live cell record previously stored the full 104-byte `PendingAction` and
private-memory/outcome state inline with position, mass, energy, digestion,
metabolism, and scheduling fields. Completion journals and deltas therefore
cloned a 192-byte record for every due actor even though pending actions are
immutable after commitment and private memory usually does not change.

`CellState` now keeps numeric physics, position, guard, ready time, a shared
pending-action pointer, and the small frequently replaced outcome inline.
Private Mind memory lives in a copy-on-write cold record. The authoritative hot
record is 88 bytes, the cold record is 16 bytes, and an installed pending action
is shared immutably. Snapshot mutation detaches only private memory when it
actually changes. Each Mind still receives an owned byte buffer; no state is
shared between cells or guest invocations.

Canonical field order and bytes are unchanged. Hashes, checkpoints, replay,
deltas, browser playback, and the Mind ABI therefore retain their existing
formats and golden vectors.

Relative to the immediately preceding indexed-store results, repeated
256x256/30,000-cell synchronized-Wait runs measured:

| Integrity | Native Minds | Indexed actions/s | Hot/cold actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| On-demand | 1 | 873,485--910,732 | 1,090,975--1,102,054 | 19.8--26.2% |
| On-demand | 8 | 1,612,822--1,628,532 | 2,296,228--2,316,262 | 41.0--43.6% |
| Verified | 1 | 768,968 | 950,597--956,650 | 23.6--24.4% |
| Verified | 8 | 1,287,537 | 1,651,085--1,838,740 | 28.2--42.8% |

The memory-heavy 5,000-cell resolver diagnostic now reaches 2.254 ms/batch and
2.22 million actions/s, compared with the older recorded 9.653 ms/batch and
517,966 actions/s before the indexed and hot/cold representation work. The
50,000-resident/eight-due fragmented diagnostic measures 0.288 ms/batch.

Pristine compatible-WASM execution remains guest dominated: the 5,000-cell,
16-worker pooled runs measured 54.2--54.4 thousand actions/s, close to the
preceding 55.4--55.6 thousand range. This slice is therefore a large native/RL
and resolver win, not a material change to the arbitrary-WASM ceiling.

### Projection-free trusted/server execution

`ReferenceHostMode::MetadataOnly` removes the renderer projection from trusted
verification runs. The canonical resolver, isolated Mind inputs, private
randomness, per-batch integrity hashes, commitments, and replay frames remain
identical. Only team/random dispatch metadata follows births and deaths; host
cell/world fields and passive projection invalidations are not maintained. The
authoritative runtime hash domain is now `blob.authoritative.runtime.v2` and
commits only canonical state plus metadata that can influence a future Mind.

Repeated 256x256/30,000-cell synchronized-Wait runs measured:

| Integrity | Native Minds | Projected actions/s | Metadata-only actions/s | Improvement |
| --- | ---: | ---: | ---: | ---: |
| On-demand | 1 | 1,093,776 | 1,128,491 | 3.2% |
| On-demand | 8 | 2,412,520 | 2,493,300 | 3.3% |
| Verified | 8 | 1,823,432 | 1,936,835 | 6.2% |

The gain is intentionally bounded: projection was about 2--5% of these batches,
while observation construction, ordered commit/delta work, and verified hashing
remain authoritative costs. Metadata-only initialization also releases the
engine's duplicate projected private-memory buffers. The renderer-facing Game
view is not updated and must not be used for scoring; online scoring should read
canonical state joined with the engine's private team map.

```sh
cargo run --release -p blob_engine --bin capacity_benchmark -- \
  256 30000 8 verified retain metadata-only
```

### Dense parallel passive kernels

Native simulations now use conservative host-only crossovers for dense passive
work: at least 75% of the resource frontier must be active, with defaults of
16,384 cells for digestion and 65,536 tiles for plant growth, signal decay, and
diffusion. WASM and smaller/sparser frontiers retain the serial oracle. The
crossovers can be forced or disabled with
`ReferenceSimulation::set_passive_parallel_thresholds` without changing any
canonical byte.

Dense diffusion is a deterministic two-pass kernel. Sources independently plan
outgoing shares, destinations gather those shares through derived reverse
adjacency, overflow results are selected in canonical tile order, and only then
are tiles updated in parallel. No atomics or schedule-dependent reductions are
used. Digestion and signal decay use conservative overflow preconditions before
parallel mutation; adversarial high-range states fall back to the serial path.

With `RAYON_NUM_THREADS=8`, eight combined passive steps over 200,000 cells and
262,144 tiles measured 160.9 ms versus 316.7 ms through the forced serial path,
or 1.97x overall. Within that run, digestion improved 6.62x, signal decay 1.93x,
and diffusion 3.43x. Plant growth was 1.13x. Metabolism remains serial because
the deterministic cell-to-tile reduction did not break even at this scale.

```sh
RAYON_NUM_THREADS=8 cargo test --release -p blob_engine \
  resolution::reference::tests::benchmark_dense_parallel_passive_kernels \
  -- --exact --ignored --nocapture
```

```sh
cargo test --release -p blob_engine \
  resolution::reference::tests::benchmark_cell_layout_size \
  -- --exact --ignored --nocapture
cargo test --release -p blob_engine \
  resolution::reference::tests::benchmark_private_memory_snapshot_churn \
  -- --exact --ignored --nocapture
```

## Interpreting the numbers

The decision interval floor permits at most one decision per living cell per
nominal simulation time unit. A conservative capacity estimate is therefore:

```text
required actions/second = living cells * simulated-time speed
```

Variable-duration actions can reduce the decision rate, while splits can raise
the population. Completion fragmentation raises fixed per-batch costs, so the
synchronized native results are ceilings rather than promises. Replay
streaming, rendering, network/storage work, hostile or computationally heavy
Minds, and browser-WASM execution are not included.
