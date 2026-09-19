# Testing and fuzzing roadmap

## Current confidence boundary

The deterministic resolver has strong example and randomized regression
coverage for action resolution, conservation, ordering, worker-count
invariance, checkpoints, deltas, replay chains, and server re-execution. The
Mind ABI has malformed-input and size-limit coverage, and RL has deterministic
evaluation, event-time GAE, exact update-boundary resume, sweep recovery, and
telemetry tests.

Maintained `cargo-fuzz` harnesses now cover the canonical replay family and the
Mind ABI with tight allocation limits and bounded pull-request smoke runs. A
stateful resolver harness now covers small worlds and up to 64 commands, including
checkpoint restoration, conservation, hash/delta oracles and serial/parallel
equivalence. Structural Wasm admission now has a bounded generated-module and
arbitrary-byte harness. These do not yet cover executable Wasm differential
fuzzing, imitation and ecological artifacts, large/long-running resolver histories,
or crash-state persistence.
Deterministic garbage, truncation, and randomized
tests remain useful regressions rather than substitutes for those campaigns.

The Chromium gate now builds real wasm-bindgen glue and native-signed fixtures.
It checks native/browser replay hashes, bounded seeking, BigInt/typed-array
exports, Ed25519 acceptance and tamper rejection through actual WebCrypto,
replay-manifest binding, malformed bytes and unavailable crypto. This closes
the earlier JSON-viewer-only browser coverage gap; guest execution and other
browser engines remain separate qualification work.
Fresh Rust, AssemblyScript and Go guests now also pass the existing 18
guest/runtime conformance gates (23 tests, including three normally ignored
language tests); see [the pinned-build evidence](language-conformance-2026-09-11.md).
This fixture-based executor comparison does not replace executable Wasm fuzzing.

Learned-policy deployment now has a [portable inference and WASM gate](learned-mind-deployment.md).
It checks exact deployment-native/WASM decisions and private memory, stock/compatible
executor samples, unrelated-invocation isolation, and one/four-worker signed server
replay. CI generates a synthetic nonzero network explicitly; a separate retained
run uses the trained frozen parent. Burn/export numerical fidelity has its own
recorded tolerance and cannot be conflated with exact WASM parity. Direct PPO
export, cross-host numerical qualification and deterministic fuel remain open.

## Missing deterministic tests

### Implemented viability-sweep gates

The following requirements are implemented and retained here as regression
expectations rather than outstanding work:

- Baseline-versus-baseline episodes must be reproducible from the same scenario
  and seed, and changing a seed must be able to change a stochastic profile's
  trajectory.
- Every baseline must receive only a canonical anonymous `ReferenceMindInput`.
  The pacifist viability profile must never emit Attack, including when a
  neighbor is visible.
- Non-learning reports must account for every requested seed, distinguish
  win/loss/timeout, record resolver time as well as environment steps, and bind
  the complete scenario and compiled rules hashes.
- Scenario sweep overrides must reject unknown and non-scenario keys, preserve
  reward/PPO/opponent settings, alter the experiment and scenario hashes, and
  continue to pair identical seeds across variants.
- Scenario validation covers world area, starting population, resource counts
  and values, initial/core energy relationships, and arithmetic overflow before
  a sweep run is admitted.
- Sweep aggregation exposes every action family's selection, success, and
  interruption rates plus the tracked energy compartments; schema tests fail
  when a compact aggregate drops a promised field.

### Required before final RL claims

- A final-holdout evaluator must prove that its seeds were not used for
  training, promotion, early stopping, or best-checkpoint selection.
- Reward-component accounting should sum exactly to the scalar reward stored
  in each transition, including terminal death/team rewards and rewards that
  accrue while another cell acts.
- Training should be checked across multiple rollout cut points, episode
  resets, pool promotion/eviction boundaries, and terminal evaluation
  boundaries. The existing exact-resume test covers one representative split.
- CPU and GPU policy-only checkpoint loading should produce the same greedy
  actions within an explicit numerical tolerance on a fixed observation
  corpus; exact optimizer resume remains backend-specific.
- The execution-contract regression now rejects trainer-binary and OCI-digest
  substitution across retries. A future container integration test should also
  compare the supplied OCI digest with the runtime/orchestrator's observed image
  identity, rather than trusting the command-line claim alone.

## Coverage-guided fuzz targets

### P0: untrusted byte boundaries

1. **Canonical replay family (initial harness implemented).** Feed arbitrary bytes and adversarial limits to
   `ReplayBatchEvent`, replay archives, segments, manifests, bundles,
   checkpoints, match manifests, attestations, and leaderboard publications.
   The oracle is: never panic or allocate beyond limits; accepted values must
   round-trip canonically and preserve their submitted hash.
2. **Mind ABI (initial harness implemented).** Fuzz Cap'n Proto input/decision decoding, including truncated
   segments, oversized lists, non-canonical visibility bits, invalid actions,
   memory-update combinations, and randomness lengths. Accepted decisions must
   survive encode/decode and resolver preflight without identity leakage.
3. **Wasm admission and Extism compatibility (structural harness implemented).**
   `wasm_admission` covers raw bytes and valid generated PDK modules with
   admission/rejection oracles, all import signatures, capability and import-kind
   rejection, export types, memory/table limits and initialization metadata.
   Its 12,480 deterministic cases separately prove generated Wasm validity, and
   custom-section insertion must preserve accepted inspection. CI runs bounded
   sanitizer smoke and retains corpora. Execution is deliberately outside this
   structural harness. Remaining work: use `wasm-smith` plus mutated
   Extism-PDK modules to exercise import admission, signatures, allocation
   ranges, pointer/length pairs, output ownership, traps, fuel/deadline limits,
   and pristine-instance reset. Extism and the compatible executor should
   agree for the admitted deterministic subset.
4. **Imitation artifacts.** Fuzz demonstration JSON/MessagePack manifests,
   sample lengths, masks, action labels, source seed/cell trajectory keys,
   recurrent-memory headers and dimensions, payload paths, declared
   representability counts, validation partitions, sampling strategies, and
   behavior-cloning unroll bounds, chunk boundaries, metadata/model hashes.
   Decoders must bound allocation, reject path substitution and non-finite
   observations, never leak one source seed across training and validation or
   one cell's recurrent state into another row, never propagate gradients
   across a detached chunk boundary, preserve canonical memory between every
   decision, and never relabel lossy private-memory decisions as exact.
5. **Ecological characterization artifacts.** Mutate embedded rules/scenarios,
   identity and result hashes, seed suites, distance accounting, censored
   travel runs, effort matrices, impossible guards, and absent/unreachable food
   sources. Reports must fail closed on tampering and remain reproducible across
   worker counts and integrity modes. Property tests should compare travel and
   attack summaries against direct resolver traces and shortest-path distances
   against a small exhaustive oracle.

### P1: stateful semantic properties

6. **Resolver command sequences (bounded harness implemented).** Generate small valid worlds and sequences
   of arbitrary decisions, clock advances, checkpoints, and restores. After
   every accepted step assert occupancy consistency, mass-energy conservation,
   canonical ordering, incremental-hash equality, delta reversibility, commit
   order invariance, and serial/parallel equality. `fuzz/src/lib.rs` shares the
   harness with 128 deterministic sequences; `fuzz/generate_resolver_seeds.py`
   produces action-family, passive-profile and sparse historical-key seeds.
   The latter exercise COW hash branches against canonical hashing and restore.
   CI retains corpus/findings.
   The initial deterministic run found an idle-cell checkpoint admission bug:
   idle ready times may precede the current clock. This is fixed with a focused
   checkpoint/continuation regression. See `fuzz-campaigns-2026-09-11.md` for
   campaign bounds and results. Mixed decisions within permuted batches and
   atomic ordered decisions now have 640 explicit sequences, five focused fatal
   preflight/continuation cases and a further 35,212 passing sanitizer inputs.
   The atomic oracle checks signals and private-memory updates as well as action
   receipts. Native deterministic tests additionally cover mixed frontiers at
   2,047/2,048/2,049 decisions under one/four workers, plus competing fatal errors
   at early/middle/late positions and exact valid continuation after rejection.
   Larger-frontier fuzzing and longer histories remain useful extensions.
7. **Neighborhood compiler (bounded harness implemented).** Generate dimensions, boundary modes, offsets,
   observation masks, and per-action masks. Compilation must reject invalid
   topology without panicking; accepted topologies must never target an
   out-of-range tile or expose a disallowed field/action. The shared harness
   checks geometry with an independent i128 oracle and preserves all masks;
   deterministic cases and a 300-second sanitizer campaign pass. It found and
   fixed huge empty-world iteration and signed coordinate overflow edge cases.
   Follow-up `observations` and `movement` harnesses check actual masked field
   values/absence, ABI and scratch-buffer behavior, host-key/private-peer
   anonymity, action permissions, elevation and occupancy-dependent corners.
   Both bounded sanitizer campaigns and their independent deterministic oracles
   pass. Observation coverage now includes all ten actions, accepted/rejected
   commitments and exact early/middle/late progress transitions, with 110
   additional explicit fixtures and a further 712,110 sanitizer inputs passing.
   Longer mixed histories remain useful extensions.
8. **RL environment sequences.** Generate valid scenario/rules profiles and
   masked action sequences. Observations and rewards must remain finite,
   reported action counts must equal commitments, telemetry must conserve its
   tracked compartments, and checkpoint/restore must preserve the next
   frontier exactly.

### P2: orchestration and persistence

9. **Sweep control files.** Fuzz TOML/JSON specs, normalized hashes, duplicate
   variants/seeds, path encodings, control-file limits, and edited manifests.
   Invalid inputs must publish no partial immutable plan and must never escape
   the requested output root through a generated variant name.
10. **Crash-state model.** Model executor/trainer crashes between each atomic
   publication step. Recovery must launch at most one trainer per run, never
   replace an inconsistent result, and aggregate only a complete verified
   matrix.
11. **Online artifact store.** Fuzz interrupted writes, directory enumeration,
   stale locks, duplicate submissions, symlink-like path substitutions where
   supported, and tampered signed records under tight byte limits.

## Harness conventions

- Keep fuzz crates outside the release workspace so normal builds do not pull
  in `libfuzzer-sys`.
- Seed corpora from existing golden vectors, every canonical action family,
  smallest/largest valid neighborhoods, checkpoints with pending actions, and
  replay/attestation examples.
- Turn every minimized crash or invariant violation into a deterministic
  regression test before closing it.
- Run short sanitizer-backed fuzz jobs in CI and longer decoder/resolver jobs
  on a scheduled host; publish corpus and crash artifacts by code revision.
