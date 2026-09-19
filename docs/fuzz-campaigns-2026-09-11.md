# Bounded fuzz campaigns — 2026-09-11

Toolchain: `nightly-2026-08-01-aarch64-apple-darwin`, cargo-fuzz 0.13.2.
The fuzz crate is locked separately from the release workspace. Missing locked
dependencies were fetched before execution; the initial offline build failure
was not a fuzz finding. Work is on the uncommitted cleanup tree at base 6dd6671.

All runs use four Cargo build jobs, disabled incremental compilation, offline
dependencies, AddressSanitizer, 300 seconds of fuzz execution, 1,024 MiB RSS and
10-second per-input limits. Existing corpus/discovery files remain in
`fuzz/corpus`; findings remain in `fuzz/artifacts`. Logs and source SHA-256 values
are retained under `training-output/fuzz-2026-09-11`.

| Target | Input cap | Result |
| --- | --- | --- |
| replay_formats | 8,192 bytes | Exit 0; 5,099,097 runs in 301 seconds; cov 2,164, features 4,272; 685 corpus entries; final RSS 904 MiB |
| mind_abi | 4,096 bytes | Exit 0; 68,351,323 runs in 301 seconds; cov 692, features 1,053; 179 corpus entries; final RSS 515 MiB |
| resolver_commands | 516 bytes, 64 commands, 25 tiles | Exit 0; 33,207 runs in 301 seconds; cov 7,866, features 41,679; 1,922 corpus entries; final RSS 588 MiB; two Rayon workers |
| wasm_admission | 8,192 bytes; no JIT or execution | Exit 0; 2,072,543 runs in 301 seconds; cov 7,374, features 19,037; 4,428 corpus entries; final RSS 647 MiB |
| resolver_commands, sparse-history follow-up | 516 bytes, 64 commands, 25 tiles | Exit 0; 35,485 runs in 301 seconds; cov 8,313, features 45,318; 1,797 corpus entries; final RSS 607 MiB; two Rayon workers |
| neighborhood | 256 bytes, ordinary worlds at most 8×8 | Exit 0; 12,749,252 runs in 301 seconds; cov 879, features 2,654; 288 corpus entries; final RSS 530 MiB |
| observations | 256 bytes, 25 tiles, eight slots | Exit 0; 806,938 runs in 301 seconds; cov 2,786, features 7,228; 365 corpus entries; final RSS 475 MiB |
| movement | 64 bytes, 25 tiles, one Move | Exit 0; 2,322,875 runs in 301 seconds; cov 2,365, features 4,459; 136 corpus entries; final RSS 489 MiB |
| resolver_commands, fused-hash follow-up | 516 bytes, 64 commands, 25 tiles | Exit 0; 34,510 runs in 301 seconds; cov 8,557, features 47,134; 1,849 corpus entries; final RSS 633 MiB; two Rayon workers |
| observations, activity/progress follow-up | 256 bytes, 25 tiles, eight slots | Exit 0; 712,110 runs in 301 seconds; cov 4,057, features 11,855; 860 corpus entries; final RSS 480 MiB |
| resolver_commands, mixed/atomic batches | 516 bytes, 64 commands, 25 tiles | Exit 0; 35,212 runs in 301 seconds; cov 8,631, features 50,027; 2,157 corpus entries; final RSS 677 MiB; two Rayon workers |

These are bounded sanitizer campaigns, not proof of absence of defects. Replay
normalizes selected outer envelopes/checksums to reach nested payload decoders;
the Mind ABI target checks semantic accepted-value round trips.

## Stateful resolver harness and discovered regression

The new shared harness checks occupancy in both directions, conserved total
mass-energy, canonical key order, incremental hashes against full canonical
hashing, sparse delta round trips, reversed ready-actor commitment order, and
serial/parallel batch and passive-update equality. It restores only one branch
through serialized checkpoints so subsequent operations compare rebuilt indexes
with the continuously maintained branch. Report delta hashes are checked at the
correct post-passive-update boundary. Inputs cover all ten action families,
invalid slots/actors/memory lengths, busy actors, invalid clock advances, pending
checkpoints, bounded/wrapped neighborhoods and passive-process profiles.

The initial 128-sequence deterministic run found that restoring an idle cell
failed after an otherwise valid clock advance. `is_ready_at` permits an idle
ready time at or before now, but restore validation required equality. Restore
now preserves past idle ready times and rejects future idle times; pending action
timestamp checks remain strict. The regression exercises a pending neighbor,
explicit clock advance, completed neighbor action, exact checkpoint bytes/state/
hashes, resumed decisions and the next batch, plus overdue pending rejection.
No simulation rules, checkpoint encoding or hash format changed.

The deterministic harness and focused regression pass. Full workspace validation
passes 596 tests (32 ignored), workspace all-target/all-feature Clippy, engine/web
WASM library Clippy, formatting and 59 schema constants/mirrors. Fuzz-crate
all-target Clippy also passes. All three sanitizer campaigns finished without a
reported finding; the idle-checkpoint bug was found in the initial deterministic
sequence run and fixed before the resolver sanitizer campaign. The resolver
campaign's libFuzzer length-growth limit reached 324 bytes; the separate
deterministic cases exercise the full 516-byte, 64-command bound.

## Structural Wasm admission

Added a target over arbitrary bytes, repaired Wasm headers, and generated valid
core modules. The generated oracle covers all 13 permitted PDK imports, wrong
parameter/result types, foreign namespaces, forbidden capabilities, non-function
imports, missing/wrongly typed reference exports, initializer signatures, start
functions, memory/table counts and shared/64-bit memory rejection. An inert
custom section must preserve an accepted inspection exactly.

The shared deterministic test checks 12,480 configurations. Each generated module
first passes the Wasm validator, so expected admission failures are not merely
syntax errors. Both fuzz-library tests and all-target Clippy pass; formatting,
schema registry and whitespace checks pass. Production code and previously
validated workspace semantics are unchanged. CI runs the new target and retains
its corpus/findings alongside the earlier targets.

The campaign uses the same pinned tools and time/RSS/input bounds above. Evidence,
command arrays and source hashes are under
`training-output/fuzz-2026-09-11/wasm-admission`. This does not cover runtime
allocation/output ownership, fuel/deadline enforcement, pristine reset, or
Extism-versus-compatible-executor differential execution. `wasm-smith`-generated
executable campaigns remain future work.

The admission campaign finished successfully with no finding. Its libFuzzer
length-growth limit reached 1,596 bytes during the bounded run, below the 8 KiB
configured maximum. The starting generator supplied 832 raw-module/selector
files; libFuzzer deduplicated and expanded this corpus during execution.

## Sparse history and neighborhood geometry

Resolver seeds now include initial keys spaced across pages and a historical
next key of 1,024, with a 4,096-slot checkpoint limit. Subsequent branch cloning,
mutation and serialized restore exercise the compact COW hash tree against the
full canonical oracle. This campaign reached a 332-byte growth limit; the
deterministic command sequences separately reach the full 516-byte input bound.

The neighborhood harness uses an independent i128 geometry oracle to check
bounded/wrapped targets, coordinate/index round trips, all observation/action
masks, invalid offsets/costs/duplicates/radii and extreme i8 offsets. Ordinary
worlds are at most 8×8 with at most 33 proposed slots. Allocation-free, zero-slot
cases additionally cover usize::MAX/i64::MAX virtual dimensions and area
overflow. The deterministic test checks 3,888 combinations, 128 arbitrary
selectors and four targeted invalid cases; 488 files seed the sanitizer run.
The campaign reached its full 256-byte input limit.

Review and deterministic checks exposed two compiler edge cases: zero-slot
neighborhoods still iterated every virtual tile, and signed coordinate
conversion/addition failed near or beyond i64::MAX. Empty neighborhoods now
skip the empty loop. Production coordinate resolution uses checked bounded
addition or reduced unsigned wrapping, without wide division on the normal
path. Engine regressions cover both boundary modes and extreme dimensions.
No rules, replay/checkpoint encodings or hash formats changed.

Both follow-up sanitizer campaigns completed without a finding. The full
workspace passes 599 tests (33 ignored); workspace and engine/web WASM Clippy,
all three shared fuzz-harness tests, schema checks and formatting pass. The
first full test build hit a rustc 1.95.0 incremental-cache "unstable fingerprints"
ICE, not a failed assertion; the retry with CARGO_INCREMENTAL=0 passed. Both
logs, command arrays and source hashes are retained under
`training-output/fuzz-2026-09-11/neighborhood-and-history`.

This initial scope checks compiled geometry and mask preservation. The next
follow-up below adds actual field exposure and occupancy-dependent corner rules.
Longer mixed resolver histories remain additional coverage work.

## Mind observation visibility and local movement

The `observations` harness projects bounded worlds with arbitrary independent
field/action masks, occupied and empty slots, wrapped self aliases, private
memory and ready/Wait/Guard activity. It checks actual field values and absence,
mass buckets, canonical ABI round trips, direct/batch projection and scratch
replacement. Renaming every host key and changing other cells' private memory
must preserve the observer's exact serialized input. Any permitted cell cue may
reveal presence; the occupancy mask is not an umbrella permission for the other
independent cues. All 128 field-permission combinations across four profiles,
plus 256 arbitrary cases, pass. The generator supplies 512 seeds.

The `movement` harness commits and resolves one Move, comparing admission and
outcome with a separate small-coordinate oracle. It covers slot masks, invalid
slots, world boundaries, wrapped self targets, occupied destinations, elevation
differences and all three orthogonal-corner policies. It also checks conservation,
full hashes and delta round trips. Deterministic tests exhaust 512 occupancy
patterns under both boundaries and three corner policies, plus 512 arbitrary
inputs. The generator supplies 360 seeds, including thin wrapped boards.

Both 300-second campaigns completed without a finding and reached their input
caps. Evidence and the original campaign source snapshots/hashes are under
`training-output/fuzz-2026-09-11/local-contracts`. After strengthening the batch
scratch check and mechanically extracting observation projection into its own
resolver child module, a further 30-second observation run completed 76,335
inputs in 31 seconds without a finding (cov 2,798, features 7,283, RSS 482 MiB).
All five shared harness tests and all-target Clippy pass. CI now retains both
new corpora. The activity/progress follow-up below expands this initial scope;
longer progress histories and executable guest isolation remain additional work.

## Final dense-hash follow-up

The end-to-end capacity comparison exposed a verified-mode regression, leading
to fused live-cell/page hashing, stable map reuse and fewer repeated membership/
page operations. The updated resolver campaign above passes with no finding;
its length-growth limit reached 340 bytes. All five deterministic harnesses,
600 workspace tests (33 ignored), workspace/WASM Clippy and nine Chromium tests
pass. The current hash code retains compact history and exact format-7 roots.
See `capacity-2026-09-11.md`; final sanitizer logs and source hashes are under
`training-output/hash-refresh-2026-09-11`.

## Full activity and progress cues

Observation fuzzing now commits all ten action families and admits semantic
rejections. Time advances to six points before the first pending event, including
immediately before and at the one-third/two-thirds progress transitions. An
independent comparison of canonical timestamps determines the progress bucket;
canonical action/rejection state determines the activity. The existing field,
ABI, batch, scratch and privacy checks apply after passive advancement too.

There are 110 explicit fixtures beyond the previous 768 deterministic cases:
60 accepted action/time combinations, two idle/guard states and 48 forbidden/
invalid target/time combinations. Assertions prove all eight activity values,
all three progress values, every accepted action family and four rejected target
families are actually observed. The seed generator writes 622 named fixtures
without deleting prior discoveries. The full 300-second campaign passed 712,110
inputs and reached the 256-byte cap. Source snapshots, hashes and logs are under
`training-output/fuzz-2026-09-11/observation-activities`.

This follow-up changes only the harness and seeds. It does not resolve actions
or claim arbitrary long-lived progress histories; the world remains bounded by
the earliest event, 25 tiles, eight local slots and 16 private bytes per cell.

## Mixed and atomic decision batches

Resolver header bit 6 now varies each actor's action, target, effort, payload and
memory content within the same frontier. The existing reverse-order comparison
uses those exact actor/action pairs. A sixth command exercises the ordered
decision API with signal sidecars, memory retention/replacement, semantic action
rejection and fatal validation failures. An independently advanced disposable
clone supplies the sequential receipt/state oracle. An invalid later decision
must leave both real branches unchanged, even if earlier oracle commitments
already changed signals, memory or escrow. Duplicate/reversed order must fail.

Deterministic coverage adds 640 mixed/atomic sequences and five focused fatal
batch fixtures followed by valid continuation. The latter cover oversized
memory, invalid signal channels, unknown actors, duplicates and reverse order;
semantic rejection still commits its bounded private memory. Eight shared fuzz
tests and all-target Clippy pass. The seed generator now supplies 241 named
resolver fixtures. The first oversized-memory fixture accidentally retained the
larger default ruleset limit; correcting that fixture exposed no engine defect.

The further 300-second sanitizer run passed 35,212 inputs without a finding.
Its length-growth limit reached 348 bytes; existing deterministic arbitrary
sequences exercise all 64 commands. Evidence and source snapshots are under
`training-output/fuzz-2026-09-11/mixed-atomic-batches`. Small worlds do not reach
the production 2,048-decision parallel-preflight threshold; they do exercise
parallel resolution/passive paths. Larger frontiers and longer histories remain
separate coverage work.

Native deterministic coverage subsequently expands the parallel-preflight gate
from Wait-only to all ten actions at 2,047/2,048/2,049 decisions and one/four
workers. It compares the former sequential oracle, derived frontiers, hashes
and the next resolution. A second test injects two different fatal errors at
early/middle/late positions, requires the earliest actor's error regardless of
worker completion order, then verifies clean continuation after rejection.
The focused group passes four tests (one benchmark ignored), as does engine
all-target/all-feature Clippy. This closes the deterministic large-frontier gap;
large-frontier coverage-guided fuzzing remains separate work.

## Passive-module extraction checks

After mechanically moving 439 lines of passive helpers into
`resolution/reference/passive.rs`, 600 workspace tests, eight shared fuzz tests,
workspace/native/WASM Clippy, schema/format checks, nine rebuilt Chromium tests
and three explicit language tests pass. Additional short checks replayed 4,056
resolver corpus inputs in 40 seconds (corpus initialization consumed the requested
30-second budget; no subsequent mutation claimed) and ran 74,168 observation
inputs in 31 seconds, without findings. Logs, commands and normalized source
equivalence are under `training-output/passive-extraction-2026-09-11`.
