# Project review — 2026-09-05

Reviewed revision: `6dd66718d03507108d38844b623563a6f82fbfe0`, on the active `extism` integration branch. The working tree was clean when review began. This document records the pre-cleanup baseline; see [the implementation follow-up](cleanup-2026-09-05.md) for fixes and remaining work.

The project has a substantial simulation and research foundation. Its strongest parts are the single authoritative resolver, explicit cell isolation, canonical replay contracts, retained reference oracles, and unusually careful experiment interpretation. The largest immediate risks are at allocation and artifact boundaries: bounded checkpoint inputs can panic, combined demonstration datasets can leak validation samples, and model compatibility is inconsistent across commands. Those should precede another architecture experiment.

I read the commit history, project handoff, major design/test/performance roadmaps, and recent feeding experiment narrative. Code inspection concentrated on canonical restoration/storage, resolver and host boundaries, WASM execution/admission, RL observations/model/training/evaluation/artifacts, and browser replay/viewer organization. This is a broad targeted review, not a claim that every line or every experiment was independently revalidated.

## Findings

P1 means a high-priority correctness or availability defect. P2 means a material compatibility, reproducibility, or scaling defect. Findings 1–4 have executable reproductions; 5–7 follow from code inspection.

### 1. P1 — Checkpoint dimensions bypass the allocation limits

[checkpoint.rs:1102](/Users/adam/code/blobs/blob_engine/src/resolution/checkpoint.rs:1102), [reference.rs:2720](/Users/adam/code/blobs/blob_engine/src/resolution/reference.rs:2720), [neighborhood.rs:320](/Users/adam/code/blobs/blob_engine/src/resolution/neighborhood.rs:320).

`max_tiles` bounds the decoded tile vector, but the independently decoded width and height reach topology compilation before their product is checked against that vector or the limit. Compilation allocates a neighbor array from those dimensions. The integrity digest does not prevent this: it is an unkeyed checksum that a malformed-input producer can recompute, and semantic/state hash checks happen after reconstruction.

**Reproduction:** modify a valid one-tile checkpoint to have width `2^60`, height `1`, repair its outer checksum, and decode with `max_tiles = 1`. The 930-byte input panics with `capacity overflow` instead of returning an error. This probe deliberately chose a capacity overflow; it did not attempt a huge allocation. Smaller adversarial dimensions could request excessive memory before rejection.

**Change:** before topology construction, require nonzero dimensions, checked area, `area == decoded_tiles.len()`, and `area <= max_tiles`. Bound derived neighbor storage as well. Add structured checkpoint mutations to the decoder regression corpus.

### 2. P1 — A single cell key can drive an unbounded dense allocation

[reference.rs:104](/Users/adam/code/blobs/blob_engine/src/resolution/reference.rs:104), [reference.rs:2892](/Users/adam/code/blobs/blob_engine/src/resolution/reference.rs:2892), [checkpoint.rs:1468](/Users/adam/code/blobs/blob_engine/src/resolution/checkpoint.rs:1468).

`CellStore::insert` resizes a vector to the numeric cell key. Checkpoint limits constrain live cell count, not key magnitude; restore inserts cells before even checking that `next_cell_key` exceeds their keys. A syntactically valid cell/occupant pair can therefore request storage unrelated to the input size or live population.

**Reproduction:** replace the matching cell key and tile occupant of a one-cell checkpoint with `u64::MAX - 1`, repair its outer checksum, and decode with `max_cells = 1`. The 1,025-byte input panics with `capacity overflow` before returning a validation error.

There is also a legitimate long-run cost: dead historical keys leave pointer-sized holes, and mutable iteration traverses that storage. Memory can grow with cumulative births even when live population stays bounded.

**Change:** validate keys before insertion and make allocation fallible. Prefer a sparse/paged lookup or a derived mapping from canonical keys to dense live slots; simply requiring `key < max_cells` would reject valid old survivors. Cover sparse high keys and long birth/death churn in tests, including other key-indexed caches.

### 3. P1 — Combined datasets can train on their own validation samples

[behavior_cloning.rs:736](/Users/adam/code/blobs/blob_rl/src/behavior_cloning.rs:736), [existing partition test:3741](/Users/adam/code/blobs/blob_rl/src/behavior_cloning.rs:3741).

The split ranks source seeds separately within each dataset and holds out a fraction of that dataset's seeds. Ranking is deterministic, but a seed's membership changes with the other seeds present. There is no global training/validation disjointness check. Existing coverage compares datasets with identical seed sets, which misses the defect.

**Reproduction:** with split seed `42` and validation fraction `0.5`, dataset A containing seeds `{3,2}` holds out `3`; dataset B containing `{2,4}` holds out `2`. An actual small behavior-cloning run accepted both. Because the probe derived both datasets from the same generated corpus, seed-2 samples were literally shared between A's training partition and B's validation partition.

**Change:** freeze one partition assignment across the union of all source seeds, then apply it to every dataset. Reject unusable partitions rather than silently changing a seed's role. Check overlap against inherited model lineage as well: a new adapter's holdout can already have trained its parent. Audit existing combined-corpus reports before using their validation accuracy as evidence; the reproduction does not establish that any particular published run was contaminated.

### 4. P2 — Schema admission and model loading disagree

[behavior_cloning.rs:2306](/Users/adam/code/blobs/blob_rl/src/behavior_cloning.rs:2306), [feeding_layout_evaluation.rs:67](/Users/adam/code/blobs/blob_rl/src/bin/feeding_layout_evaluation.rs:67), [training.rs:1037](/Users/adam/code/blobs/blob_rl/src/training.rs:1037), [migration loader:123](/Users/adam/code/blobs/blob_rl/src/bin/behavior_clone.rs:123).

The common verifier accepts multiple model schemas, but discards the schema in its path-only API. Feeding, contact, micro-combat, recovery, correction, and PPO initialization paths then load the file directly into the current network. The migration dispatch exists only inside the behavior-cloning binary, so other consumers cannot share it. Schema 30 is also admitted by the common range check even though that binary explicitly declares it non-migratable.

**Reproduction:** a hash-consistent schema-24 legacy record passes `verify_behavior_clone_artifact`; the current evaluator loading sequence fails with `missing field foraging_adapter_fc`.

**Change:** expose one library API returning a verified, migrated policy and its execution identity. Use it everywhere; explicitly reject unsupported versions. Test every supported schema through the shared loader, with exact migration-output checks. Avoid requiring another training run merely to convert an otherwise usable policy.

### 5. P2 — Policy identity does not fully identify executed behavior

[BehaviorCloningArtifact:497](/Users/adam/code/blobs/blob_rl/src/behavior_cloning.rs:497), [observation router:57](/Users/adam/code/blobs/blob_rl/src/observation.rs:57), [tensor router:825](/Users/adam/code/blobs/blob_rl/src/model.rs:825), [handoff:85](/Users/adam/code/blobs/docs/project-handoff.md:85).

The same saved weights and cloning metadata can produce different actions after a router semantic change. The current artifact binds configuration and weight bytes but has no explicit router/observation/action execution version. The handoff correctly acknowledges that historical artifacts need an external code/container digest to disambiguate the corrected diffuse-energy behavior.

This matters for both evaluation reuse and parent/control identity: a matching model hash is insufficient to establish that a previously qualified policy is the same executable policy now being run.

**Change:** complete this already-listed roadmap item before further adapter experiments. Bind router semantics, observation/action encoding, memory format, model schema, and execution build identity. Reject incompatible evaluation reuse and preserve explicit historical semantics where reproducibility is promised.

### 6. P2 — Local build provenance can remain stale after source edits

[build.rs:17](/Users/adam/code/blobs/blob_rl/build.rs:17).

The build script computes revision/dirty state, but declares rerun dependencies only on `.git/HEAD`, `.git/index`, and an environment override. Editing a tracked source file without touching those files can rebuild the crate while reusing the earlier embedded revision. Branch refs, packed refs, and worktree Git-directory indirection are also not covered by the hard-coded paths. A generic `+dirty` suffix does not distinguish two different dirty source trees even when recalculated.

**Change:** use an explicit immutable build identity for scientific artifacts, preferably a source-tree/content digest and executable/container digest. If local automatic provenance remains supported, resolve the actual Git paths and track the relevant inputs. Test a source-only edit and a worktree build. This finding is based on the declared dependency graph, not a full rebuild experiment.

### 7. P2 — Sparse match data expands into dense browser work and history

[match-explorer.js:121](/Users/adam/code/blobs/blob_web/viewer/match-explorer.js:121), [match-explorer.js:167](/Users/adam/code/blobs/blob_web/viewer/match-explorer.js:167), [match-explorer.js:196](/Users/adam/code/blobs/blob_web/viewer/match-explorer.js:196).

`buildIndex` walks the whole event history synchronously, recomputes metrics by scanning all tiles/cells per event, and clones the full world every 128 events. Each seek also clones a full checkpoint. Input limits cap area and events independently, without bounding their product or retained checkpoint memory.

A 256×256 history with 10,000 events passes the current limits but implies roughly 655 million tile visits for environment metrics and about 5.2 million retained tile objects across checkpoints, before cell/maps/signal-array overhead. Those are algorithmic counts, not measured browser memory or timings. Larger supported boards amplify the issue; the two browser smoke tests use only 12 events.

**Change:** update aggregate metrics from patches, bound checkpoint cache memory, and use the existing replay-segment API for large histories. Move indexing off the UI thread and introduce cancellation. Add a realistic long-history browser budget test before treating local exploration as ready for large experiments.

## Organization assessment

The top-level workspace separation is useful and should remain. `blob_engine` is authoritative; the ABI, game host, training, and browser packages have recognizable responsibilities. A broad rewrite would put well-tested physics at unnecessary risk.

Inside those crates, several files and APIs now mix stable infrastructure with rapidly changing experiments:

- `resolution/reference.rs` has 9,803 lines, with its main test module starting at line 6,472. It owns cell/tile storage, state types, topology, scheduling, passive physics, commit planning, resolution, and test oracles. Extract storage and restore validation first, because findings 1–2 show a concrete boundary problem; then separate passive kernels, commit planning, and resolution while retaining the reference oracles.
- `behavior_cloning.rs` combines splitting, sampling, loss construction, recurrent preparation, adapter freezing, metrics, publication, and verification. Move dataset partition/provenance and policy loading into reusable library modules first. Finding 4 is a direct consequence of keeping shared compatibility logic in a CLI.
- `training.rs` has nearly 3,000 lines before its main test module; `env.rs` has over 2,300. Separate rollout collection/reward attribution from curriculum assignment and artifact orchestration. Preserve exact-resume tests during extraction.
- The RL crate exports almost every research subsystem publicly and has 48 binaries. Consolidate repeated argument handling, backend dispatch, policy loading, artifact publication, and bounded I/O. Keep experiment-specific behavior behind narrow modules; a single CLI can wait until the library contracts stabilize.
- There are 50 sweep directories, while the main RL roadmap has grown to 4,937 lines. Preserve that history as evidence, but split operational instructions, artifact contracts, and experiment history. The short handoff is the right entry point; add an experiment index with hypothesis, disposition, artifact identity, and superseding result.

Do these extractions alongside fixes and contract tests. File size alone is not a correctness finding, and inline tests account for a meaningful part of the totals.

## Roadmap changes

The current diagnosis is disciplined: a legal collision-aware teacher establishes feeding viability; the corrected router removes a semantic error; dense-layout contention remains; target-head-only training failed the mechanistic gate. Deferring further ecological runs was appropriate. The proposed private-randomness target residual is a reasonable next hypothesis, not yet an established solution.

Recommended order:

1. **Repair boundary correctness:** checkpoint allocation safety, globally disjoint dataset partitions, one model loader, and executable policy identity. Recheck affected validation reports and preserve the four reproductions as permanent regressions.
2. **Run the proposed target-residual experiment:** retain zero-output migration, permutation equivariance, row isolation, and frozen-parent checks. Freeze disjoint data assignments before collection. Require target agreement and contention reduction, inspect whether private randomness actually changes greedy targets, and retain the same-effort teacher control before the five-layout ecological matrix. Evaluate feeding/combat retention as well as the corrected local decision.
3. **Make final evaluation independent:** the test roadmap already admits there is no proof that final holdout seeds avoided training, promotion, early stopping, and checkpoint selection. Introduce a persistent seed-use ledger across parent lineage and evaluation runs. Repeatedly consulted qualification suites are development evidence; reserve a separate final suite.
4. **Qualify long-run resource use:** bound evaluator histories, browser indexing, and lifetime-key storage; include population churn, not just large static populations. Measure both peak memory and useful simulated horizon on 256/512/1024 profiles.
5. **Close deployed-Mind and online gaps:** explicitly plan a current learned-checkpoint-to-WASM deployment/parity milestone if learned policies are intended to compete as submissions. Retain anonymous ABI and quantized private memory through export. Then finish service operations already named in the handoff.

The online milestone also needs an execution-budget contract. The compatible executor uses a wall-clock epoch ticker ([extism_compat.rs:95](/Users/adam/code/blobs/blob_game/src/extism_compat.rs:95)); the current host returns invocation errors for the batch ([plugin_pool.rs:364](/Users/adam/code/blobs/blob_game/src/plugin_pool.rs:364)). No per-call deterministic fuel budget is installed in the inspected path. A Mind near the deadline can succeed on one host/load and abort on another. Decide which failures are deterministic game events versus retryable infrastructure failures, bind the runtime/fuel policy, and add adversarial executor parity tests before claiming unconditional submitted-WASM reproducibility. Current benign-Mind worker parity does not cover this case.

The full cross-language conformance workflow is scheduled/manual, while ordinary PR jobs build Rust WASM in a separate job from native tests. Runtime/admission changes deserve an artifact-backed execution gate before merge. Also add a real browser test of generated WASM glue and WebCrypto attestation verification: the current Chromium tests exercise the local JSON viewer, not that signed replay path.

## Validation and limits

- `cargo test --workspace --all-features --quiet`: passed, with the repository's explicitly ignored tests still ignored.
- `python3 scripts/check_schema_registry.py`: passed; 59 version constants and browser mirrors agree.
- `npm test --prefix browser-tests`: both Chromium tests passed. The first attempt could not bind the local server inside the sandbox; the permitted rerun completed.
- Temporary Rust probes reproduced the two checkpoint panics, cross-dataset validation leakage, and legacy model load failure. These probes asserted the observed defects and therefore passed; they are not claims that the intended invariants passed.
- Probe sources were removed from the project after execution and retained locally at `/private/tmp/blob-review-20260905/engine-probes.rs` and `/private/tmp/blob-review-20260905/rl-probes.rs` for follow-up. No implementation fixes were applied.
- Full cross-language conformance was not rerun: TinyGo is not installed on this host. The handoff records the same limitation. No GPU parity run, long fuzz campaign, remote experiment rerun, or fresh throughput benchmark was performed.

The passing suite supports the existing deterministic examples and regression contracts. It does not negate the reproduced boundary failures or establish final policy competence.
