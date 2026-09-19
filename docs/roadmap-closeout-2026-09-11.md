# Six-hour roadmap closeout — 2026-09-11

Work window: 04:32:41–10:32:41 UTC (00:32:41–06:32:41 America/New_York).
Implementation and validation are complete. Scheduled continuation is paused,
confirmed after the deadline; no owned compute jobs remain. Shutdown is recorded
at the end of the [work log](roadmap-work-2026-09-11.md).
Changes remain uncommitted on `extism`, based on
`6dd66718d03507108d38844b623563a6f82fbfe0`; no push or deployment was performed.
This report covers the accumulated review cleanup and this window's follow-up.

## Original review disposition

| Finding | Implemented result | Remaining boundary |
| --- | --- | --- |
| Checkpoint dimensions bypass allocation limits | Validate dimensions, checked area and decoded tile count before topology allocation; fallible reservation | Hosts must select limits appropriate to their resource budget |
| Historical cell keys drive dense storage | Validate lifetime keys, use sparse cell/scheduler pages and compact live hash storage | Cold hash restore and arbitrary old empty-page reactivation still visit history |
| Combined datasets overlap training and validation | One seed partition across the corpus union, inherited cloning ledger and explicit historical audit | Project-wide final evaluation registry and PPO ledger propagation remain open |
| Schema admission and loading disagree | Shared verified policy loading and explicit migrations for supported schemas | Historical policies execute current routing and need fresh ecological qualification |
| Weights do not identify executed behavior | Bind execution semantics and source identity; reject incompatible resume/reuse | Source identity alone does not attest the runtime container or numerical backend |
| Dirty-source build identity can remain stale | Content provenance, source dependency tracking and Git worktree/ref handling | Keep immutable binary/container evidence for scientific reproduction |
| Sparse browser histories cause dense repeated work | Reversible patch history, incremental metrics, bounded retained values and cancellation | Very long histories still benefit from segmented loading and a worker |

Details and original reproductions remain in the [baseline review](code-review-2026-09-05.md)
and [cleanup report](cleanup-2026-09-05.md).

## Roadmap results

- **Target selection:** implemented the zero-initialized target residual with
  exact migration, slot permutation, row isolation and frozen-parent training
  gates. Matched 22/88/256-update treatments failed exact target agreement.
  The margin/randomness diagnostic also failed to establish a useful mechanism.
  No ecological qualification or policy promotion followed. See the
  [residual report](target-residual-2026-09-11.md) and
  [margin diagnostic](target-margin-diagnostic-2026-09-11.md).
- **Storage and hashing:** ordinary appends, live-page refresh and trusted
  planner restoration reuse compact historical commitments. In the eight-live-
  cell/four-million-key fixture, peak RSS fell from 319.55 to 4.22 MiB and 100
  active-metabolism batches from 360.27 to 5.71 ms. The final dense Wait matrix
  still measures a 5.5–6.5% verified-throughput cost against the original base;
  on-demand medians range from -2.0% to +0.1%. All 64 final matrix runs match
  canonical hashes and counts. The traversal-cache prototype was rejected and
  reverted after its performance gate. See [capacity evidence](capacity-2026-09-11.md).
- **Correctness and organization:** fixed idle-checkpoint admission, huge empty-
  world neighborhood iteration and signed coordinate overflow. Extracted storage,
  hash-tree, observation and passive-kernel responsibilities while retaining
  independent canonical/numeric oracles. Shared bounded artifact reads now
  enforce existing byte caps on the opened stream across 23 RL modules.
- **Coverage and navigation:** added bounded resolver, Wasm-admission,
  neighborhood, observation and movement fuzzing; strengthened mixed/atomic and
  large parallel preflight oracles. Real browser WASM/WebCrypto and fresh guest
  builds close earlier local coverage gaps. The [experiment index](experiment-index.md)
  records all 50 retained sweep directories without promoting plans to results.

## Final validation

| Gate | Result |
| --- | --- |
| Workspace, all features | 604 passed, 33 ignored |
| Workspace all-target/all-feature Clippy | Passed with warnings denied |
| Engine/web WASM library Clippy | Passed |
| Separate fuzz crate deterministic tests | 8 passed; Clippy passed |
| Chromium, rebuilt WASM and native signed fixtures | 9 passed |
| Fresh Rust/AssemblyScript/TinyGo guest/runtime gates | 18 gates, 23 tests passed, including 3 normally ignored language tests |
| Formatting and schema mirrors | Passed; 59 constants/mirrors |
| Final paired capacity matrix | 64/64 runs complete; all final states match |

The full conformance shell script was not rerun end to end. The table identifies
the suites actually executed; guest tests overlap some workspace coverage and
must not be added to its count as unique tests. Bounded sanitizer campaigns
and their exact scopes, input counts and limits are recorded in the
[fuzz report](fuzz-campaigns-2026-09-11.md). Structural admission fuzzing does not
execute arbitrary guest programs. No GPU parity or production service deployment
is claimed. Large generated evidence stays local under ignored `training-output/`.

## Next work in order

1. Establish effective private-random/slot interaction on fresh development
   validation seeds before another matched adapter treatment. Preserve reserved
   confirmation seeds and the inherited ledger; keep ecology gated on exact
   target improvement and retained control behavior.
2. Address remaining verified throughput and cold-restore cost with measured
   alternatives and canonical oracles. Include changing populations and bounded
   evaluator histories before extrapolating the static capacity fixtures.
3. Extend artifact fuzzing to structural allocation, correction pairs and model
   migrations; close the separate model-record verify/load race. The bounded
   stream helper does not establish a decoded-heap or multi-file snapshot budget.
4. Define deterministic guest execution budgets and adversarial executor parity,
   then qualify learned-policy WASM export. Keep production upload, compilation,
   queues, quotas, retention, signing and leaderboard work as explicit milestones.

Organization work is also unfinished. `reference.rs` still contains tile storage,
checkpoint mutation tracking, restore validation, commitment planning and event
resolution; its test module begins at line 5,978 in this tree. Behavior cloning
still combines sequence preparation, evaluation, loss construction, optimization
and publication before its tests at line 2,432. The next extractions should follow
those ownership boundaries: tile storage/restore first, then commitment planning;
sequence preparation and artifact publication separately from the training loop.
Keep public paths stable and retain the existing checkpoint continuation,
ordered-batch atomicity, recurrent row isolation, frozen-parameter and exact-resume
oracles. File size alone is not a reason to rewrite the simulation or training
algorithms.
