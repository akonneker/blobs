# Six-hour roadmap work — 2026-09-11

Authorized window: 04:32:41–10:32:41 UTC (00:32:41–06:32:41 Eastern).
Follow-up: `blobs-six-hour-roadmap-work`, every ten minutes in this task. At the
end of the window, stop new implementation, report results, and pause the follow-up.

## Starting state

Branch `extism`, HEAD `6dd6671`, with earlier cleanup changes still uncommitted.
Do not discard them. The last full gate passed 588 tests, workspace Clippy,
engine/web WASM library Clippy, formatting, and schema registry consistency.
TinyGo conformance remains unavailable on this host.

## Work order

1. Implement the handoff's cell-private random/local-slot target residual.
   Preserve legacy model imports with exact-zero migration; provide a mode that
   updates only the new residual; maintain execution identity and seed lineage.
   Require migration, slot-permutation, row-isolation, and frozen-parent tests.
2. Inspect local parent/correction artifacts and lineage. Run matched bounded
   control/treatment tests only when their prerequisites are verified. No ecological
   matrix before exact target agreement improves on disjoint validation seeds.
3. Continue historical hash-update performance and meaningful test/fuzz coverage
   if the experiment needs unavailable inputs, or after its mechanistic gates.

## Progress

- 04:35 UTC: Follow-up configured; reviewed current handoff and model/cloning paths.
  Beginning target-residual implementation. No experiment has started.

- 04:42 UTC: Implemented target residual, cloning schema 37/training schema 46,
  legacy layout migration, adapter-only training, and pre-load backend seeding.
  Four new focused tests passed. Full workspace tests and Clippy are running;
  one Clippy single-element-loop finding in a fixture was fixed. Release CPU
  experiment tools are building. Metadata inventory found all 16 parent ancestors
  and their corpora (52 seeds); saved pair seeds are disjoint from the parent.
  See target-residual-2026-09-11.md. No ecological run is authorized by evidence yet.

- 04:44 UTC: Full workspace gate passed 592 tests (31 ignored); Clippy and schema
  checks pass. The parent inventory resolves 18 historical corpora. Preparing a
  sequential 22-update CPU control/treatment pair using the saved five layout pairs,
  with confirmation seeds 1434999901/1434999902 reserved. The existing pair seeds
  are development validation already seen in earlier architecture work, not fresh
  final confirmation. Release build is still running.

- 04:45 UTC: All five correction-pair verifiers passed (1,453 active changes total).
  The sequential 22-step control/treatment runner is active, PID recorded in
  `training-output/target-residual-2026-09-11/status.json` (initial PID 85785;
  tool session 58311). Inspect status/logs before starting anything else; do not
  duplicate it. Each arm has a 20-minute subprocess timeout and four Rayon threads.
  Immutable copied executables and their SHA-256 values are in that directory,
  along with the exact command arrays and run_pair.py. Completion writes
  comparison.json. On failure, preserve logs and diagnose before a new output path.
  Next: read treatment initial/final held-out Move exact accuracy and the control's
  retention; do not count reduced cross-entropy alone as passing the mechanism gate.
  Confirmation seeds are reserved in both outputs, not used for training/evaluation.

- 04:46 UTC correction: the first probe failed before optimization. The historical
  schema-16 effort corpora hit an equality-to-current-schema check in the loader.
  The original run directory and logs are preserved; it is no longer running.
  Implemented a schema-16-onward compatibility check and a hash-bound regression
  covering 16/17/18, malformed private memory, tampering, and schema-15 rejection.
  Validate and rebuild before launching a fresh v2 output directory.

- Compatibility regression also exposed schema 17 missing from the file loader's
  admission list despite being supported by payload validation. Both now use one
  shared supported-schema predicate (11 through current 18); the correction type
  remains restricted to 16 onward. Running the full RL library and import/lineage
  integration suites against the correction before rebuilding final probe tools.

- 04:50 UTC: Compatibility fix validated (291 library tests, 1 all-layout migration
  test, 2 seed-lineage tests; 4 ignored; Clippy clean). Rebuilt immutable CPU tools.
  The active probe is now `training-output/target-residual-2026-09-11-v2/`.
  Read its status.json and logs before doing new experiment work; the old directory
  is a preserved failed attempt. Tool session for the v2 runner is recorded in the
  task tool output. Continue from this status rather than restarting completed checks.

- 04:50 UTC result: v2 finished both arms successfully (status complete; no live
  training job remains). Control preserves 100% held-out Move exact agreement.
  Treatment remains 63.2466% (1,239/1,959) before and after 22 updates; its validation
  loss changes only 0.9414745 to 0.9414268. This dose fails the mechanism gate;
  do not launch ecological qualification. Control artifact SHA-256:
  `1d70be95dae1ccb23325fffed7acb33b479a7e1816054f7c3f5d566792786709`;
  treatment: `3a3a844d92a06355430ee07196057cc72e06643a35fe0906eb8adc3a3b92321e`.
  Full audit succeeded: 53 training seeds including 1433000101; validation 1433000202;
  confirmation 1434999901/1434999902 untouched.

## Current continuation

- 05:03 UTC: Bounded 88/256-update control/treatment probes completed with the
  immutable v2 binary. Controls retain 100%; treatment stays 63.2466% at 88 and
  slips to 63.1955% at 256. No dose passed. Stop the dose sweep; no ecological run.
  Evidence: `training-output/target-residual-2026-09-11-doses/comparison.json`.
- Implementing the next storage-roadmap slice: incremental cell-tree append and
  shared traversal for batch hash updates, including all-cell invalidation.
  Existing hashes remain exact. Initial engine tests/native+WASM Clippy pass.
  Focused release comparison: 64 refreshes of 65,536 historical pages with eight
  live pages took 0.188 ms using updates versus 760.631 ms rebuilding; this excludes
  cell commitment generation and is not whole-simulation throughput.
- Added structural work-count tests to catch accidental historical-page rebuilding.
  Review found a deferred-birth boundary case: update a refilled old frontier before
  pruning/extension, or an empty historical prefix could be reconstructed. That
  ordering is fixed and tested for both full and partial old tree shapes.
  Completing final workspace gates and documentation before choosing the next slice.

- 05:15 UTC: Hash-update slice complete. Final gates passed 595 workspace tests
  (32 ignored), workspace Clippy, engine/web WASM library Clippy, formatting,
  whitespace, and 59 schema constants/mirrors. Final-code focused benchmark:
  64 refreshes, 65,536 historical pages, eight live pages: 0.208 ms updates versus
  1,180.220 ms cold rebuild (under concurrent build load). See hash-update-2026-09-11.md.
- Starting the next testing-roadmap task: pinned nightly-2026-08-01 cargo-fuzz 0.13.2,
  existing replay_formats corpus, 300 seconds of fuzz execution, 1 GiB RSS limit,
  10-second per-input timeout, 8 KiB max input. Tool session 15080; output log
  /private/tmp/blob-roadmap-replay-fuzz.log. Check completion/crashes before starting
  another run. Preserve/minimize any finding and add a deterministic regression.

- Fuzz startup initially stopped before execution because `arbitrary` was absent
  from the local cache. Fetched the harness's locked dependencies and restarted;
  this was a build prerequisite, not a fuzz finding. The active session replaces
  15080; use the same log to determine the actual run state.

- 05:38 UTC: Replay campaign finished with exit 0: 5,099,097 inputs/301 seconds.
  Mind ABI campaign also finished with exit 0: 68,351,323 inputs/301 seconds.
  Logs/source hashes are in training-output/fuzz-2026-09-11; see
  fuzz-campaigns-2026-09-11.md for bounds and limitations.
- Added a shared resolver command harness, 128 deterministic 64-command cases,
  action-family seed generator, sanitizer target and CI smoke/artifact retention.
  It compares serial/parallel and checkpoint-restored/continuously maintained
  branches, hashes, conservation, occupancy, commit order and reversible deltas.
- The initial deterministic harness found a real restore bug: idle cells whose
  ready time precedes the checkpoint clock were rejected. Validation now follows
  is_ready_at semantics (idle ready_at <= now) while rejecting future idle and
  invalid pending times. The focused checkpoint/continued-event regression and
  deterministic harness pass; fuzz-crate all-target Clippy passes.
- Resolver sanitizer campaign is active, session 3234, log
  /private/tmp/blob-roadmap-resolver-fuzz.log. Bounds: 300 seconds, 1 GiB RSS,
  10-second input timeout, 516-byte input, 64 commands, 25 tiles, two Rayon workers.
  Workspace tests session 31758, native Clippy 37816 and WASM lib Clippy 88737
  are active; logs use /private/tmp/blob-roadmap-{workspace-tests,clippy,wasm-clippy}.log.
- 05:41 UTC: Workspace gates finished successfully: 596 tests (32 ignored),
  workspace all-target/all-feature Clippy, engine/web WASM library Clippy,
  formatting, whitespace and 59 schema constants/mirrors. The resolver campaign
  remains active; no new gate job is needed unless another implementation changes.
- 05:43 UTC: Resolver campaign finished successfully: 33,207 inputs/301 seconds,
  coverage 7,866, 1,922 corpus entries, final RSS 588 MiB. No sanitizer finding.
  All logs are retained in training-output/fuzz-2026-09-11 with command arrays
  and source hashes. No fuzz or gate process remains active. The testing slice
  and idle-checkpoint fix are complete. Implementation remains uncommitted.

## Next continuation

- 06:53 UTC: Added structural Wasm admission fuzzing without JIT execution.
  Generated contract test passes all 12,480 configurations across 13 PDK imports,
  16 admission modes, initialization/start flags and memory/table limits. A
  separate validator confirms negative cases are valid Wasm; arbitrary bytes and
  inert custom-section transformations exercise parser robustness/stability.
  Fuzz crate tests (including the existing 128 resolver sequences), all-target
  Clippy and corpus generation pass. No production code changed in this slice.
- Active sanitizer campaign: session 16036, log
  /private/tmp/blob-roadmap-wasm-admission-fuzz.log; pinned nightly/tool version,
  300 seconds, 1 GiB RSS, 10-second input timeout, 8 KiB max input. 832 generated
  raw-module/selector seed files were supplied; libFuzzer may deduplicate them.
  Check completion/findings before starting more fuzz work. Updated CI includes
  the target and existing artifact retention. Runtime/executor differential
  coverage remains open; this does not execute arbitrary guests.
- Read-only preparation for the next RL diagnostic: compare the trained residual's
  contribution to corrected-target versus policy-target margins, stratified by
  source seed/layout, and measure slot-feature differences (especially dx/dy).
  Use the verified `load_behavior_clone` and `load_demonstrations` APIs. Rebuild
  recurrent histories per (source_seed, source_cell), canonicalizing with
  encode_policy_memory/decode_policy_memory after each decision as cloning does.
  The baseline can be the trained model with a fresh zero-head target residual
  copied via with_target_residual_from; this holds all inherited parameters fixed.
  Report hashes and diagnostic scope; do not mutate schema-37 feature semantics
  or launch another dose/ecological sweep based only on this analysis. No such
  diagnostic executable or job has been created yet.
- 06:58 UTC: Structural admission campaign finished with exit 0: 2,072,543
  inputs/301 seconds, coverage 7,374, 4,428 corpus entries, final RSS 647 MiB.
  No finding. Campaign log, exact command, source hashes and validation logs are
  retained under training-output/fuzz-2026-09-11/wasm-admission. This slice is
  complete; no fuzz or other owned job remains active. Next: implement the
  read-only target-margin diagnostic described above, then use its evidence to
  select subsequent roadmap work. Keep confirmation seeds untouched.
- 07:12 UTC: Added blob_rl/examples/target_residual_diagnostics.rs. It verifies
  model/dataset hashes, reconstructs each cell's canonical recurrent history in
  batches of 64, compares the trained residual against its zero-output replacement,
  and asserts identical recurrent outputs. It emits per-Move target margins,
  adapter score shifts, feature differences and seed/layout identities. This is
  a read-only development diagnostic, not a qualification report or training run.
  Release CPU build session 50096 and example Clippy session 86495 are active;
  logs /private/tmp/blob-target-diagnostic-{build,clippy}.log. No diagnostic run
  has started yet. Check these before launching the saved 256-update treatment
  against the five verified treatment corpora; use a bounded runner and retain
  binary/source hashes, exact commands and row-level output.
- Release build and example Clippy passed. Diagnostic runner is active in
  training-output/target-margin-diagnostic-2026-09-11 (session 72094), with an
  immutable copied binary/source, SHA-256 values, exact command array and
  run.py/status.json. Four Rayon workers; 20-minute subprocess timeout. No
  training occurs. Check this run before starting another; its output is rows.json.
- 07:19 UTC: Diagnostic complete. The v1 margin-only run and v2 randomness/gain
  probe both passed in seconds; no owned job remains. V2 evidence is under
  training-output/target-margin-diagnostic-2026-09-11-v2. All 10,240 reconstructed
  decisions have identical baseline/trained recurrent outputs; all 3,880 Move
  baseline targets reproduce recorded policy targets. Validation reproduces
  1,238/1,959 trained correct targets versus 1,239 with zero residual.
- Every active validation correction gains relative score (mean 0.11797 logits)
  but none crosses its original preference (mean gap 1.57750). Random blocks are
  unique/nonzero within each corpus. A within-row random-feature rotation changes
  the residual pair shift only weakly (validation median 0.0000439). All active
  slot pairs differ only in dx/dy/distance. Offline gains 0/1/2/4/8/16/32/64 do
  not improve either split; gain 16 fixes 390 validation corrections but loses
  475 previously correct choices. See target-margin-diagnostic-2026-09-11.md.
  Release build, final example Clippy, formatting and whitespace checks pass.
  No model, schema, input semantics or confirmation seeds changed.
- Next engineering slice: investigate compact cell hash-tree restoration from
  trusted planner seeds. IncrementalStateHashCache::from_seed still reconstructs
  CellMerkleCache over historical pages from only live leaves. Inspect
  hashing.rs::IncrementalStateHashSeed and checkpoint.rs::PlannerHashSeed before
  changing representation; preserve exact hash roots and bounded live/history
  storage. Do not rerun the rejected training/gain sweep. An eventual new target
  architecture must demonstrate useful private-random/slot interaction first.
- 07:36 UTC: Implemented trusted compact cell-tree seeds. Branches use Arc COW;
  restored caches recheck all current/previous live-page commitments while sharing
  unchanged branches and reusing collapsed historical digests. Planner allocation
  accounting includes shared cell branches, and small boards retain seeds when
  cell history spans multiple pages. Public checkpoint/hash formats are unchanged.
  New regressions cover no historical-page rebuild on restore, COW isolation,
  deletion refresh/pruning, seed sharing, resumed pending actions, full-hash parity
  and rejection of changed canonical cells under a stale seed. Engine lib passed
  86 tests (18 ignored); engine Clippy passed. A release benchmark of 64 complete
  trusted restores on a 4x4 board with 65,536 historical cell pages and 2 live
  cells took 0.565 ms shared versus 740.311 ms cold; hash allocations 3,008 bytes.
  This is a deliberately history-heavy workload, not general game throughput.
- Full workspace tests session 84033, workspace Clippy 24971, engine/web WASM
  library Clippy 31163 and shared fuzz-library regressions 90021 are active.
  Logs: /private/tmp/blob-hash-restore-{workspace,workspace-clippy,wasm-clippy,fuzz-tests}.log.
  Release benchmark log: /private/tmp/blob-hash-restore-benchmark.log. Check gates
  before completing this slice; preserve prior edits and do not duplicate jobs.
- 07:39 UTC: Trusted hash restoration slice complete. Final gates passed 598
  workspace tests (33 ignored), workspace all-target/all-feature Clippy,
  engine/web WASM library Clippy, shared fuzz-library regressions, formatting,
  whitespace and 59 schema constants/mirrors. Logs, benchmark command and final
  source hashes are in training-output/hash-restore-2026-09-11. No jobs remain.
  See hash-restore-2026-09-11.md. All implementation remains uncommitted.
- Next testing-roadmap slice: extend bounded resolver seeds to span sparse
  historical key pages (to stress the new COW tree), then add neighborhood
  compiler fuzz coverage with an independent geometric target/mask oracle.
  Keep runtime/input/world dimensions tightly bounded. Existing complete
  campaigns need not be rerun wholesale; run only the changed/new harness.
- 08:08 UTC: Added sparse historical-key profiles to resolver fuzzing (initial
  next key 1,024, sparse live keys, checkpoint key bound 4,096). Added neighborhood
  compiler fuzzing with an independent i128 geometric oracle, mask/admission
  checks, extreme i8 offsets, empty/oversized neighborhoods and allocation-free
  huge virtual worlds. The generated deterministic geometry test covers 3,888
  combinations plus 128 arbitrary selectors; all three fuzz-library tests pass.
- Review exposed two compiler edge cases: empty neighborhoods still iterated
  every virtual tile, and signed coordinate conversion/addition failed near or
  beyond i64::MAX. Empty topologies now skip the empty loop. Coordinate resolution
  uses checked bounded addition or reduced unsigned wrapping, avoiding overflow
  without wide-integer arithmetic on the production path. The independent oracle
  and an engine regression cover usize::MAX dimensions, both boundaries, signed
  extreme offsets and world-area overflow. Engine/fuzz Clippy pass.
- Active sequential sanitizer campaigns, session 22279: updated resolver first
  (/private/tmp/blob-history-resolver-fuzz.log), then neighborhood
  (/private/tmp/blob-neighborhood-fuzz.log). Each 300 seconds, 1 GiB RSS, 10-second
  per-input timeout; input caps 516/256 bytes. Stop and investigate on a finding;
  the shell will not start the second target if the first fails. Full workspace
  tests 23912 and workspace Clippy 25478 are active; engine/web WASM Clippy passed.
  Logs /private/tmp/blob-neighborhood-{workspace-tests,workspace-clippy,wasm-clippy}.log.
  Check these jobs before launching any more work.
- 08:10 UTC: The full workspace test build hit rustc 1.95.0's "unstable
  fingerprints" incremental-compilation ICE while compiling match-explorer.
  The failure log is preserved at /private/tmp/blob-neighborhood-workspace-tests.log.
  Retrying with CARGO_INCREMENTAL=0; log
  /private/tmp/blob-neighborhood-workspace-tests-no-incremental.log. This is a
  compiler-cache failure, not a failed assertion; no source workaround was added.
  Workspace Clippy and WASM Clippy passed. The sequential fuzz campaigns continue.

- 08:19 UTC: Sparse-history resolver and neighborhood slice completed: 35,485
  and 12,749,252 sanitizer inputs in 301 seconds each, exit 0. Full workspace
  retry passed 599 tests (33 ignored), all three fuzz-harness tests and lints
  passed. Evidence/logs in training-output/fuzz-2026-09-11/neighborhood-and-history.
- Next slice in progress: generated browser WASM and real Chromium Ed25519
  verification. Pinned wasm-bindgen-cli 0.2.100 installed under
  /private/tmp/blob-browser-tools; package build and native signed fixture pass.
  Added browser_fixture example, four browser tests and CI preparation. Browser
  suite session 20444 running with permitted local HTTP binding; log
  /private/tmp/blob-browser-wasm-tests.log. Native example Clippy passed. Review
  test outcomes before completion; all edits remain uncommitted.
- 08:20 UTC: All nine Chromium tests passed, including four generated-WASM
  tests with actual Ed25519 WebCrypto. Native replay/seek hashes and BigInt
  exports match; wrong keys, altered signatures/messages, wrong replay binding,
  malformed bytes, seek budget and unavailable crypto behave as expected.
  Final preparation-script and formatting checks remain before recording the
  browser slice complete. No production verification behavior changed.
- 08:25 UTC: Browser slice complete: preparation script, package, fixture,
  nine Chromium tests, native example Clippy, formatting and shell syntax pass.
  Evidence/source/generated hashes in training-output/browser-verification-2026-09-11.
  Next slice: observation visibility and anonymity harness, at most 25 tiles,
  eight slots and 16 private bytes per cell. All 128 permission combinations
  across four profiles plus 256 arbitrary inputs pass; fuzz all-target Clippy
  passes. Sanitizer campaign session 89013 active, 300 seconds / 1 GiB / 256 bytes,
  log /private/tmp/blob-observations-fuzz.log. Scratch-buffer test was strengthened
  after campaign build started; rerun deterministic tests and hash final source.
- 08:34 UTC: Observation campaign passed 806,938 inputs; movement campaign
  passed 2,322,875 inputs (301 seconds each, 1 GiB bounds). All five shared
  deterministic harness tests and fuzz Clippy pass. Sources/logs retained under
  training-output/fuzz-2026-09-11/local-contracts. Observation projection was
  extracted into reference/observation.rs: four blocks / 470 lines are verified
  text-identical; existing public method/type paths remain. Focused six boundary
  tests pass and the final strengthened observation harness passed another
  76,335 sanitizer inputs in 31 seconds.
- Active extraction gates: workspace tests then Clippy session 92360, WASM
  library Clippy 3446. Logs /private/tmp/blob-observation-extraction-{workspace,
  workspace-clippy,wasm-clippy}.log. Shared fuzz tests 35402 passed; sanitizer
  29771 completed successfully. Check these before declaring extraction complete.
- Next measurement slice: compare the existing end-to-end capacity benchmark
  against the clean base archive /private/tmp/blobs-capacity-baseline-20260911.
  Archive is HEAD 6dd6671; no baseline build or measurements have started. Add
  an untimed final canonical-hash audit to the same benchmark source in both
  trees before comparing, use bounded serial alternating runs and avoid timing
  while other builds/fuzz jobs are active. Retain commands, binaries and hashes.
- 08:37 UTC: Observation extraction complete. Final gates: 599 workspace tests
  (33 ignored), workspace all-target/all-feature Clippy, engine/web WASM Clippy,
  five shared fuzz tests, rebuilt browser package/fixture and nine Chromium
  tests, formatting and schemas. Evidence in
  training-output/observation-extraction-2026-09-11. Capacity benchmark now has
  an untimed canonical/full-hash and live-count audit; identical benchmark source
  compiled in baseline/current trees, copied immutable executables in
  training-output/capacity-2026-09-11. Both release builds and benchmark Clippy
  pass. No timing matrix has started yet.
- 08:43 UTC: End-to-end capacity matrix completed 64 serial runs over 128/256/
  512/1024 boards, four Mind/Rayon workers, two million Wait actions per run,
  three measured repetitions after per-case warm-up. All final hashes/counts
  match across versions and integrity modes. Initial current verified throughput
  is 41.5% lower at 2,000 cells and 14.1–17.1% lower on larger profiles;
  on-demand mode is within about 2.4%. Evidence and runner:
  training-output/capacity-2026-09-11-v2. The first attempt is preserved under
  capacity-2026-09-11: macOS time -l failed on a sandboxed sysctl after the binary
  passed. V2 measures child peak RSS with os.wait4, requiring no escalation.
- Dense hash refresh is the current optimization target. Reusing stable leaf
  map allocations alone recovers ~2% throughput; trial evidence is under
  training-output/hash-refresh-2026-09-11/leaf-reuse. Current code additionally
  fuses live-cell leaf and page hashing, using only live pages and retaining the
  old compact history tree. New regression covers stable allocation addresses,
  sparse pages, same-count birth/death replacement and empty populations.
  Focused hash tests pass 7 tests (1 manual benchmark ignored). Active paired
  timing session 3130 compares fused code with pre-optimization cleanup at
  2,000/50,000 cells; log /private/tmp/blob-hash-refresh-fused-pages.log and
  training-output/hash-refresh-2026-09-11/fused-pages/status.json. No broader
  gates yet on this candidate. Do not overlap builds/tests with timed cases.
- 09:11 UTC: Final hash/capacity slice complete. Full comparison against the
  original base passes all 64 hash/count audits. Final verified throughput is
  4.6–6.9% lower than base, versus 14–42% before this optimization. On-demand
  cases are 0.9–4.0% lower in the final run. Keep the remaining tradeoff explicit.
  Final evidence: training-output/capacity-2026-09-11-v3; report
  docs/capacity-2026-09-11.md. Trials and the earlier regression matrix remain.
- All final gates pass: 600 workspace tests (33 ignored), workspace all-target/
  all-feature Clippy, engine/web WASM Clippy, five shared fuzz tests, nine Chromium
  tests with rebuilt generated WASM/fixtures, formatting and 59 schema mirrors.
  The fresh resolver sanitizer run completed 34,510 inputs in 301 seconds, exit
  0 (cov 8,557; features 47,134; final RSS 633 MiB). Logs, commands and hashes are
  under training-output/hash-refresh-2026-09-11. No tests or fuzz jobs remain.
- The new lifetime_capacity example also passed release builds/Clippy and 64
  paired measurements against the original base. Eight live cells are spread
  across 8/65,536/1,048,576/4,000,000 canonical historical keys, with idle/active
  metabolism. All initial/final hashes match. At 4m keys with active metabolism,
  peak RSS is 319.55 -> 4.22 MiB; 100 verified batches take 360.27 -> 5.71 ms;
  cold canonical restore plus first hash takes 24.91 -> 2.77 ms. The fixture models
  post-churn states; it does not time creating history or executing Minds. Cold
  restore still has an O(history) cost. Evidence and runner are under
  training-output/lifetime-capacity-2026-09-11. No measurements remain active.
- Next bounded testing-roadmap task: expand fuzz/src/observations.rs beyond
  ready/Wait/Guard to all action activities, accepted/rejected commitments and
  early/middle/late progress. Advance only before the earliest pending completion
  so no action resolves or exceeds the small-world bound. Keep an independent
  progress oracle, exact ABI/scratch comparisons and host-key/peer-memory
  noninterference through canonical restore; the idle observer may have ready_at
  before the new clock, which is valid. Update deterministic cases and seeds,
  then run only the changed harness under the same 300-second/1-GiB bounds.
  Do not repeat completed training doses, capacity matrices or unchanged campaigns.

### Observation activity follow-up, 09:33 UTC

- Expanded the bounded observation harness to all ten action families, rejected
  commitments, idle guard cues and six times around the early/middle/late
  boundaries. Time advances strictly before the first event, preserving the
  25-tile/eight-slot bound. The oracle derives cues from canonical pending
  requests and timestamps; exact ABI, field permissions, direct/batch scratch
  replacement and host-key/peer-memory noninterference remain checked.
- Added 110 explicit coverage fixtures and seeds (622 generated seeds total).
  All six shared harness tests and fuzz all-target Clippy pass. No production
  source changed. The 300-second/1-GiB observation campaign completed 712,110
  inputs in 301 seconds, cov 4057/features 11855/RSS 480 MiB, without a finding.
  Commands, source snapshots and log are retained under
  training-output/fuzz-2026-09-11/observation-activities. Do not duplicate it.

### Mixed/atomic resolver batches and language conformance, 09:43 UTC

- Resolver header bit 6 now selects mixed per-actor actions. A sixth operation
  checks atomic ordered batches against individual commitments on a disposable
  clone, covering sidecars, retained/replaced memory, invalid ordering, duplicate/
  unknown/busy actors and oversized memory. Fatal errors must leave both real
  branches unchanged; semantic rejection still commits memory. Existing hash,
  conservation, delta, permutation, serial/parallel and restore checks remain.
- Added 640 explicit mixed/atomic sequences and five late fatal/order cases with
  valid continuation. The first oversized-memory fixture mistakenly used the
  default memory limit; corrected the fixture to the harness's 16-byte limit.
  This was a test setup error, not an engine defect. All eight shared tests and
  all-target fuzz Clippy pass. The 300-second/1-GiB resolver campaign passes
  35,212 inputs in 301 seconds, cov 8631/features 50027/RSS 677 MiB, no finding.
  Evidence: training-output/fuzz-2026-09-11/mixed-atomic-batches.
- Found older AS/Go canary artifacts, but no TinyGo and the installed Go 1.27 is
  outside the build script's supported range. Official SHA-256-verified archives
  for CI-pinned TinyGo 0.41.1 and Go 1.25.1 are now extracted under
  /private/tmp/blob-conformance-tools-20260911. Fresh language builds ran
  with temporary dependency caches and completed successfully. All seven Rust
  guest/canary WASMs were rebuilt too. All 18 guest/runtime conformance gates
  passed (23 tests, including the three normally ignored language tests), with
  nine artifact hashes, source hashes and no skipped-artifact output. Evidence:
  training-output/language-conformance-2026-09-11; report
  docs/language-conformance-2026-09-11.md. No language build or guest gate remains
  running. This is fixture conformance, not executable Wasm fuzzing.

### Passive-kernel organization and experiment index, 09:54 UTC

- Moved 439 lines of passive arithmetic/topology into
  blob_engine/src/resolution/reference/passive.rs. Only parent-module visibility
  and formatting changed; normalized bodies compare exactly. The resolver owns
  scheduling/frontiers/journals and retains all existing numeric/parallel tests.
  Source before/after and equivalence evidence are retained under
  training-output/passive-extraction-2026-09-11.
- Post-extraction: 600 workspace tests pass (33 ignored), workspace all-target/
  all-feature Clippy, engine/web WASM Clippy, all eight shared fuzz tests, fuzz
  Clippy, formatting and schema checks pass. Short sanitizer checks pass:
  resolver replayed 4,056 corpus inputs in 40 seconds (initialization consumed
  the requested 30-second budget; no post-initialization mutation claimed),
  observations ran 74,168 inputs in 31 seconds. Browser WASM/native fixtures were
  rebuilt and nine Chromium tests pass. All three explicit language tests pass
  again on the extracted host. No validation jobs remain active.
- Added docs/experiment-index.md with recorded dispositions/successors for all
  35 narrated sweeps and evidence/status for the other 15 raw-matrix/plan
  directories. All 50 directories, local links and anchors are checked. It
  records the current failed residual/gain result, frozen parent identity and
  exposed/reserved seed boundary. Linked from README, handoff and the long RL
  narrative. Historical immutable sweep records remain intact.

Check the 10:32:41 UTC deadline first. All nine earlier full campaigns
above are complete; do not repeat them without new changes or evidence. Preserve/
minimize future findings and add deterministic regressions. No training probe
remains active; the target-residual
dose sweep failed and is stopped. Hash-update work is complete and validated.
Trusted hash restoration, resolver history and neighborhood compiler coverage
are complete. Browser WASM verification, observation extraction and both
capacity qualifications are complete. The observation-activity follow-up is
complete. Native parallel-preflight coverage now includes all ten actions at
2,047/2,048/2,049 decisions with one/four workers, comparing the former sequential
oracle, derived indexes, hashes and the next resolution. A new test injects two
different fatal errors at early/middle/late positions, requires deterministic
first-error selection, then validates clean continuation. The focused group
passes four tests (one benchmark ignored); engine all-target/all-feature Clippy
passes. This adds one test beyond the last full 600-test workspace run. Logs are
under the passive-extraction evidence root. No test jobs remain active.

### Final bounded-read fix and rejected cursor trial, 10:12 UTC

- The immutable sparse-page traversal cursor passed its two focused oracles and
  Clippy but failed the performance gate: 48 serial paired cases gave medians
  +0.69%/-1.51% verified and -0.45%/-1.52% on-demand across small/large profiles.
  All hashes matched. Reverted only cell_storage.rs and paged_slots.rs exactly
  to their pre-trial snapshots; both candidate sources/binaries, before files,
  source hashes and measurements remain in training-output/page-cursor-2026-09-11.
  Those two additional trial tests were also reverted. No trial job is active.
- Found stat/reopen reads that could bypass artifact byte caps under file growth
  or replacement. New blob_rl/src/artifact_io.rs opens once, checks that handle's
  metadata, reads at most limit+1 bytes and rejects oversize before decoding.
  Wired 23 RL modules while preserving limits, format/hash checks and validation.
  Three regressions cover exact/over limits, endless/growing input and opened-file
  identity after path replacement. Initial RL tests: 294 pass, four ignored.
  Workspace Clippy passes. Final full workspace tests completed: 604 passed,
  33 ignored, with two test threads; evidence under
  training-output/artifact-io-2026-09-11. Formatting, schema and new-doc links pass.
  No more implementation should begin before finishing final validation.
- Prepared training-output/capacity-2026-09-11-v4 for a final engine comparison
  after the passive extraction. Baseline is the original archived revision;
  current is freshly rebuilt from the restored final engine, replacing the
  rejected prototype in target/release too. The build and all 64 serial cases
  completed before the 10:22:41 UTC cutoff without concurrent compute. All
  canonical final hashes and counts match the original baseline. Verified
  medians are 5.5–6.5% lower; on-demand medians range from -2.0% to +0.1%.
  The rebuilt binary is byte-identical to the retained pre-cursor-trial binary.
  Session 62228 exited successfully and was collected; no owned test, training,
  fuzz or timing job remains active. The capacity report and handoff now use
  this final matrix. Earlier measurements remain historical evidence.
Structural Wasm admission fuzzing is complete for its initial scope. Admission is
public in blob_engine/src/mind_runtime.rs (inspect_mind_artifact), so a standalone
fuzz target can exercise it without a JIT. Execution compatibility checks live in
the binary-only blob_game/src/extism_compat.rs; inspect that boundary before
adding executable-module differential coverage.
Available toolchain: nightly-2026-08-01, cargo-fuzz 0.13.2. Use strict runtime/RSS/
input bounds and retain findings. Do not reopen ecological qualification without
a new mechanism result. Unseeded/public checkpoint restoration and arbitrary
old empty-page reactivation still visit history; seeded planner restoration and
ordinary appends/all-cell invalidation now reuse compact branches.

### Closeout audit, 10:25 UTC

- Added docs/roadmap-closeout-2026-09-11.md with the original seven findings'
  dispositions, this window's implementation/experiment results, exact validation
  scope and ordered remaining work. The handoff links it directly. Historical
  cleanup validation is explicitly labeled as initial rather than current.
- Final engine evidence matches all 29 recorded source/config hashes; the bounded
  artifact-reader evidence matches all 25 recorded source hashes. The final engine
  binary matches the retained pre-cursor-trial binary exactly. Current schema 37's
  inherited schema-36 ledger contract is clarified in the seed-lineage guide.
- Changed-file fingerprints, tracked diff and Git status are retained under
  training-output/roadmap-closeout-2026-09-11. No source changes followed the final
  validation. All owned compute sessions are complete and collected. Scheduled
  work will be paused at the 10:32:41 UTC deadline; no further implementation or
  compute campaigns are being started in this window.

### Six-hour window closed, 10:32 UTC

- The 10:32:41 UTC deadline has passed. Heartbeat
  `blobs-six-hour-roadmap-work` is PAUSED, confirmed in its persisted config
  after the app update (2026-09-11T10:33:19.177316+00:00).
- No owned compute jobs remain. Implementation stopped before closeout; no
  commit, push or deployment was performed. Final work remains uncommitted on
  `extism`. Use docs/roadmap-closeout-2026-09-11.md and docs/project-handoff.md
  for the results, validation limits and next roadmap tasks.
