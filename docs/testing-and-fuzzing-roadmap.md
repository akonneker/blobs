# Testing and fuzzing roadmap

## Current confidence boundary

The deterministic resolver has strong example and randomized regression
coverage for action resolution, conservation, ordering, worker-count
invariance, checkpoints, deltas, replay chains, and server re-execution. The
Mind ABI has malformed-input and size-limit coverage, and RL has deterministic
evaluation, event-time GAE, exact update-boundary resume, sweep recovery, and
telemetry tests.

Maintained `cargo-fuzz` harnesses now cover the canonical replay family and the
Mind ABI with tight allocation limits and bounded pull-request smoke runs. They
establish the untrusted-byte foundation, but do not yet cover Wasm admission,
imitation and ecological artifacts, long resolver command sequences, or
crash-state persistence. Deterministic garbage, truncation, and randomized
tests remain useful regressions rather than substitutes for those campaigns.

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
3. **Wasm admission and Extism compatibility.** Use `wasm-smith` plus mutated
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

6. **Resolver command sequences.** Generate small valid worlds and sequences
   of arbitrary decisions, clock advances, checkpoints, and restores. After
   every accepted step assert occupancy consistency, mass-energy conservation,
   canonical ordering, incremental-hash equality, delta reversibility, commit
   order invariance, and serial/parallel equality.
7. **Neighborhood compiler.** Generate dimensions, boundary modes, offsets,
   observation masks, and per-action masks. Compilation must reject invalid
   topology without panicking; accepted topologies must never target an
   out-of-range tile or expose a disallowed field/action.
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
