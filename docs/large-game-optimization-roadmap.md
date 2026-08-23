# Large-game optimization roadmap

The target is to make ordinary event work proportional to the actions and
resources that changed, not to total board area or population. Every optimized
path must remain bit-for-bit equivalent to the serial reference semantics and
worker-count invariant.

## Priorities

1. **Measurement and board-sized scratch removal — completed**
   - Record non-authoritative phase timings for passive updates, snapshot
     materialization, validation, resolution, hashing, and delta generation.
   - Use claim-proportional occupancy contention storage for sparse batches,
     retaining a dense fast path for dense batches.
   - Add large-sparse-board and worker-parity benchmarks.

2. **Incremental authenticated state commitment — completed**
   - Divide canonical tiles and cells into fixed pages with canonical page
     hashes and a deterministic Merkle root.
   - Rehash only dirty pages and retain periodic full-hash audits in
     conformance tests.
   - Version hashes, checkpoints, and replay formats together.
   - Canonical hash format 6 adds domain-separated per-cell leaves and separate
     private-memory commitments. Sparse updates rehash only changed leaves and
     affected pages; dense updates use a parallel full-leaf crossover.
   - The derived live cache stores memory and leaf commitments in key-indexed
     slots. Dense workers fuse leaf creation with canonical page encoding,
     avoiding intermediate ordered-tree construction without changing the
     authoritative cell map or hash bytes. Native encoding uses an optimized
     SHA-256 backend while WASM retains the portable backend under the same
     golden vectors.
   - Host execution can select explicit on-demand integrity for trusted RL and
     local rollouts. Dirty commitment metadata remains exact, so a requested
     current/final hash commits all skipped batches. Batch reports expose the
     lack of pre/post commitments, replay recording rejects them, and verified
     per-batch hashing remains the default for server and browser workflows.

3. **Delta-driven host, UI, and replay projection — implemented**
   - The canonical simulation is the sole authoritative state owner; host and
     game state are explicitly derived views.
   - Live batches apply the canonical action delta plus conservative passive-
     physics invalidations. Full view reconstruction is retained only for
     initialization and checkpoint restore.
   - Birth team inheritance and death attribution use only affected cells,
     while per-tick host metadata synchronization is limited to ready cells.
   - Ready cells and supplied Mind decisions are combined with one canonical
     ordered merge. Missing decisions become fallback Waits without the former
     quadratic ready-cell-by-decision search.
   - Non-authoritative host telemetry separates ready-frontier discovery,
     observation projection, Mind execution, commit, passive invalidation,
     replay recording, and host-view synchronization.
   - Local observation construction is a single neighborhood traversal and
     reuses one fully overwritten slot/private-memory scratch buffer per native
     Mind. The host retains only compact private per-cell work metadata until
     invocation instead of retaining the complete observation frontier. The
     submitted-WASM path uses the same scratch projection before producing each
     isolated serialized input buffer.
   - Authoritative cells now use direct key-indexed slots plus a separate
     canonical active-key traversal. Observation projection borrows that index
     directly, eliminating both repeated tree searches and the former
     frontier-scoped index allocation. Dead historical keys retain only
     pointer-sized holes. Slot projection walks compiled target rows directly.
     RL observation and action decoding reuse scratch inputs instead of
     allocating private-memory and slot buffers per cell.
   - Private decision randomness keeps the same domain-separated SHA-256 bytes
     while native hosts use an accelerated backend and clone a match-scoped
     prehashed prefix. WASM retains the portable oracle and fixed golden vectors
     bind both implementations.
   - Teams with multiple pristine Mind instances process sufficiently large
     decision frontiers in bounded parallel chunks. Each worker owns its Mind
     and scratch observation, cell-scoped private randomness is derived inside
     that chunk, and decisions are restored to canonical actor order afterward.
     Frontiers below the configurable native crossover remain serial. WASM
     projection/serialization uses the same bounded chunking and transfers
     serialized inputs into plugin workers without cloning their byte buffers.
     A WASM pool activates at most one worker per 64 invocations, retaining
     compiled capacity for large frontiers without oversharding small batches.
   - Ready decisions are committed through one strictly actor-ordered batch.
     The resolver preflights every fatal condition before mutation, groups
     completion scheduling by time, uses dense hash invalidation for dense
     frontiers, and accepts an explicit `Retain` result without allocating or
     comparing returned private-memory bytes. `Replace` remains an owned,
     bounded update. The host moves the same operation into replay commitments
     instead of cloning it around per-cell calls. Native preflight crosses to
     read-only parallel planning at 2,048
     decisions, then selects any error in canonical actor order before applying
     mutations.
   - Live host projection skips unchanged canonical positions, preserves host
     memory allocation, and linearly merges sorted invalidation streams. Mind
     workers derive a host-only no-op hint for pure Wait decisions while their
     isolated input is available; projection still runs whenever the action
     delta or passive frontier changes any host-visible field. The hint is not
     authoritative and never crosses the Mind ABI.
   - Trusted/server execution can select `ReferenceHostMode::MetadataOnly`
     before canonical initialization. It skips passive projection discovery and
     all renderer-compatible `World`/`Cell` synchronization while retaining
     only live-cell team identity, private randomness lineage, and decision
     sequence. Births inherit team identity and deaths remove their metadata;
     verified hashes, commitments, replay, Mind isolation, and canonical
     physics are unchanged. The submitted-WASM verification entry points select
     this mode by default.

4. **Mutation journal and dirty-resource deltas — completed**
   - Record every cell/tile write in a canonical first-write journal that owns
     the before-value directly. This decouples rollback and reversible delta
     generation from the immutable completion snapshot.
   - Release resolution snapshots all tiles but only due actors and the
     occupants of their declared target tiles. It no longer materializes the
     full resident cell population for a fragmented completion batch.
   - Construct reversible deltas from touched resources without cloning or
     comparing a complete post-state.
   - On action-resolution failure, restore all touched resources, key
     allocation, scheduling indexes, metrics, and hash caches from the
     first-write journal. Passive time advancement remains a distinct phase and
     is not undone by a later action-resolution failure.
   - In debug/conformance builds, compare every journal delta with the original
     full-state oracle across all action families.
   - Completion validation reads canonical cells before mutation rather than
     cloning a second cell snapshot. Snapshot-only attack mass and guard state
     travel with the validated intent. Due actors are journaled and transitioned
     in one ordered cell-map traversal; sparse additional writes remain in a
     side map and merge canonically for rollback and delta construction.

5. **Sparse and lazy passive physics — frontier/index slices implemented**
   - Exact derived active-signal and diffuse-energy frontiers now make signal
     decay, diffusion scheduling, passive projection invalidation, and dirty
     hash-page marking proportional to active field tiles. Checkpoint restore
     derives the indexes from canonical tiles; the indexes never enter hashes
     or replay state.
   - Synchronous diffusion now reads only active sources and uses one reusable
     incoming buffer with sparse destination clearing. A dense test oracle
     verifies bit-for-bit equivalence as the frontier spreads.
   - Exact derived digestion, metabolism, and zero-energy death sets now drive
     passive cell updates, exhaustion scheduling, projection invalidation,
     dirty hash pages, and death cleanup. Every energy mutation maintains the
     sets, checkpoint restore derives them, and resolution rollback rebuilds
     them from its immutable completion snapshot.
   - Sparse digestion traverses only gut-bearing cells. When every cell is
     metabolically active, a density crossover retains direct canonical-map
     traversal rather than adding derived-index lookup overhead.
   - A zero digestion or metabolism rate now has a canonical zero-remainder
     invariant and an O(1) passive fast path. Disabled physics no longer scans
     a dense living population for changes that cannot occur.
   - Dense native hosts now cross to deterministic parallel kernels for plant
     growth, digestion, signal decay, and diffusion. Cell and tile thresholds
     are host-only policy and never enter rules, hashes, checkpoints, or replay;
     WASM retains the serial oracle. Dense diffusion plans source flux in
     parallel, gathers through derived reverse adjacency without atomics, then
     commits in tile order. Overflow selection remains canonical.
   - Metabolism deliberately remains serial after measurement: parallel
     per-cell accrual followed by deterministic tile-deposit reduction did not
     break even through 200,000 cells. A combined 200,000-cell/262,144-tile
     passive workload is nevertheless about 2x faster with the other dense
     kernels.
   - True lazy metabolism remains deferred. Exact metabolism changes both cell
     energy and the diffuse reservoir at the cell's current tile before
     diffusion, while verified hashes and checkpoints commit fully materialized
     values. A correct lazy representation therefore requires a versioned
     cell timestamp/base-energy format, an exhaustion index, deferred per-tile
     deposits, and explicit materialization barriers; it is not a storage-only
     optimization.

6. **Compact hot-state layout — implemented**
   - Canonical cell private memory now uses immutable shared ownership.
     Completion snapshots, reversible deltas, checkpoints, and rollback state
     share unchanged bytes, while each Mind invocation still receives a
     distinct owned copy at the isolation boundary. Canonical encodings and
     hashes remained byte-identical for that storage-only slice; hash format 6
     intentionally versions the commitment structure described below.
   - Tree-based hot cell lookup has been replaced by stable indexed slots plus
     a canonical active-key traversal. Canonical bytes and equality ignore
     physical slot capacity. Mutation preflight now updates due actors through
     direct lookup, and observation batches allocate no secondary index.
   - The inline live-cell record now contains the frequently read numeric,
     position, scheduling, guard, and small outcome fields. Immutable pending
     actions are shared separately, and private Mind memory lives in a
     copy-on-write cold record. This reduces the hot record from 192 to 88 bytes
     while retaining one authoritative value for every field.
   - Cold private memory now has its own cached domain-separated commitment.
     Hot cell leaves bind that commitment without rereading unchanged bytes,
     and only actual Mind-memory changes invalidate it.
   - Consider a content-addressed arena only if cross-cell deduplication proves
     worthwhile; copy-on-write already removes intra-cell snapshot churn.

7. **Strict-isolation-compatible WASM acceleration — restricted executor prototype implemented**
   - Compatible-executor workers share one immutable Engine, Module,
     store-independent link plan, and deadline ticker per team pool. Wasmtime's
     pooling allocator reuses bounded, decommitted instance slots, memory
     mappings, and allocation metadata. Every decision still creates a new
     Store and guest instance; no live linear memory, table, global, allocator,
     limiter, or host context is shared between cells.
   - Shared pool dimensions preserve the former aggregate per-worker capacity,
     guest linear memory is capped by the manifest's page limit, and on-demand
     allocation remains configurable for portability and direct A/B
     measurement. Stateful memory/global canaries and worker/hash parity gate
     the optimized path.
   - Stock Extism retains one compiled descriptor per worker because its public
     `CompiledPlugin` can contain non-Send/Sync host-function user data. It is
     not shared through an unsafe wrapper.
   - Further live-store or serialization-buffer reuse is allowed only if a
     sealed pristine restore can prove that no hidden mutable state survives.
   - Fresh compatible calls now keep the ordinary host output arena and up to
     four allocation/free records inline, with tested transparent spillover.
     This removes common host-allocator churn without pooling any per-Mind
     bytes. Release stage diagnostics show that pooled Store creation,
     instantiation, and export lookup together cost only about 0.6 microseconds;
     guest ABI decode and policy work are now the larger per-action target.
   - Mind ABI v6 replaces pointer-backed optional slot scalars with an inline
     visibility bitmap and values. Hidden versus visible zero remains exact,
     all malformed bit/value combinations are rejected, and no new observation
     channel is exposed. Representative input size fell about 44%, maintained
     compatible calls improved about 27%, and large pooled authoritative runs
     improved roughly 10--15% without weakening pristine instances.
   - Host preparation now pipelines actor-ordered observation/ABI chunks with
     worker execution and sends one request/response per active worker rather
     than per cell. Exact output buffers remain owned per invocation; batching
     exposes no peer input or shared guest state.
   - A selectable Extism-compatible Wasmtime executor caches the compiled guest
     and immutable link plan while constructing a new Store, instance, guest
     memory, host byte arena, allocation map, and limiter per decision. It
     implements only deterministic PDK byte-memory imports and rejects WASI,
     configuration, variables, networking, custom hosts, and logging during
     module admission.
   - Match-verification format 2 binds the named
     `extism_pdk_deterministic_v1` capability profile independently of executor
     implementation. Native upload admission parses and validates exact typed
     imports/exports plus memory/table shape before retaining or compiling an
     artifact; both Extism and compatible local execution use the same gate.
   - Every maintained Rust Mind and the mutable-global isolation canary produce
     byte-identical commitments and state hashes under stock Extism and the
     restricted executor. Direct tests bind deadline interruption, import
     rejection, deterministic memory primitives, and worker invariance.
   - Maintained AssemblyScript and TinyGo `wasm-unknown` artifacts now exercise
     the official language PDKs, structural admission, canonical output, and
     differential execution. Standard Go WASI output remains deliberately
     outside the profile. Stock Extism remains a selectable compatibility and
     differential-testing oracle.
   - The compatible host recognizes the optional typed reactor `_initialize`
     export used by TinyGo and executes it under the deadline inside every
     fresh instance before the Mind entry point.
   - Prefer parallel matches over large per-match worker pools until batch
     measurements show otherwise.

8. **Spatial chunk execution for a single large match**
   - Route intents to deterministic resource-owning chunks.
   - Resolve chunk interiors in parallel and border claims in a canonical
     second phase.
   - Build explicit conflict components only for genuinely multi-resource or
     cross-boundary interactions.

9. **Optional batchable Mind backend for extreme populations**
   - Define a constrained pure policy/IR whose independent lanes each receive
     only one cell's observation, explicit memory, and randomness.
   - Require scalar-oracle equivalence; never expose batch peers or shared
     mutable state.
   - Retain arbitrary isolated WASM as the general, slower backend.

## Invariants

- Minds remain anonymous, isolated, and unable to observe scheduling, worker,
  shard, or stable public identity.
- Every action resolves against the same immutable completion snapshot.
- Canonical ordering and integer arithmetic remain platform-independent.
- Hashes and replay commitments are independent of thread count and physical
  data layout.
- Browser and optimized native implementations conform to the serial oracle.
