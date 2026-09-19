# Project handoff

Updated 2026-09-19. The `codex/learned-agent-minds` branch is the active integration branch.
The six-hour roadmap run has completed implementation and validation; see the
[closeout](roadmap-closeout-2026-09-11.md) and [work log](roadmap-work-2026-09-11.md)
for results, remaining work and evidence. The cleanup and learned-Mind follow-ups
are now collected on this branch. Dated experiment notes retain their original
revision and working-tree provenance. This document is the short operational index; the linked design documents and
versioned sweep notes remain the detailed evidence.
The [experiment index](experiment-index.md) records dispositions and successors
for all 50 retained sweep directories, including plans without result evidence.

Build caches, Python bytecode, local environments, training outputs and raw sweep
artifacts are ignored. Frozen binaries, weights and detailed experiment evidence
under `training-output/` remain local; Git contains the source, verification tools,
experiment summaries and existing bounded sweep reports. A fresh clone must build
its binaries and generate its own artifacts; it does not include the archived
trained candidates. The separate remote `extism` history is preserved.

## Review cleanup

The [2026-09-05 cleanup](cleanup-2026-09-05.md) closes the immediate checkpoint,
combined-corpus split, model-loading, provenance, and viewer-indexing defects.
New cloning artifacts bind execution semantics; exact resume and evaluation reuse
require matching source builds. Cloning schema 36 now carries a [cumulative seed ledger](seed-lineage.md),
with an explicit audit path for historical parents. The [lifetime-storage cleanup](lifetime-storage.md)
releases dead cell/scheduler pages and compacts the live hash cache without changing
canonical bytes. [Incremental cell-tree updates](hash-update-2026-09-11.md) now
reuse the historical prefix across births and full-cell invalidation. [Trusted
planner restores](hash-restore-2026-09-11.md) now share compact cell-tree branches.
Cold/unseeded restores and arbitrary reactivation of old empty hash regions still
visit historical pages.
The [capacity follow-up](capacity-2026-09-11.md) reduced a dense verified-hash
regression to 5.5–6.5% below the original baseline on measured Wait profiles.
An eight-live-cell/four-million-historical-key fixture uses 4.22 MiB peak RSS
versus 319.55 MiB originally; active verified events also avoid historical-page
rebuilds. Cold restore retains an O(history) cost.
Observation projection and passive arithmetic now live in separate resolver
child modules. The passive extraction preserves 439 lines of arithmetic after
visibility/formatting normalization; the existing numeric and parallel oracles
remain in place.
The [artifact read follow-up](artifact-io-2026-09-11.md) closes a size-check/reopen
gap across 23 RL modules by enforcing existing byte caps on the opened stream.

## Learned deployment follow-up

The [checkpoint-to-WASM path](learned-mind-deployment.md) now exports supported
cloning checkpoints through a shared portable inference runtime. A frozen parent
runs as an ordinary WASM Mind, with exact native-deployment/WASM decision and
memory parity, worker-invariant replay and server attestation checks. Export
records its distinct numerical execution identity and the measured one-quantum
memory differences from Burn. The final trained artifact is
`training-output/learned-mind-2026-09-12/frozen-final/mind.wasm`; it passes 2,010
exact parity decisions and ten signed replay fixtures, and runs through the
ordinary game CLI. The synthetic CI fixture passes separately. This closes the
initial packaging gap; robust feeding/combat qualification and deterministic fuel
policy remain open.
The six-hour heartbeat remains paused; this is subsequent user-authorized work.

## Non-negotiable operational principle

Each cell is an independent Mind invocation. It receives only its bounded local
observation, inherited private memory, and engine-derived cell-private random
bytes. It has no public cell identity, team identity, absolute coordinates,
global clock, shared seed, shared memory, host calls, or implicit colony
channel. Multi-cell coordination must pass through visible actions, terrain,
four environmental signal channels, or lineage-local memory inherited during a
split.

Keep privileged canonical state confined to characterization, verification,
and explicitly labeled search upper bounds. A controller qualifies as a legal
Mind witness only when it replays through the ordinary Mind ABI.

## Implemented foundation

- The fixed-point event-time simulation resolves simultaneous local actions in
  canonical order, conserves mass-energy, supports all ten action families,
  and reproduces across worker counts.
- Canonical checkpoints, replay segments, incremental hashes, server-side
  re-execution, signed attestations, and browser verification form the trusted
  submission kernel.
- Mind ABI v8, structural WASM admission, stock Extism compatibility, and the
  restricted Wasmtime executor preserve fresh-instance isolation. Rust,
  AssemblyScript, and TinyGo examples exercise the PDK boundary.
- Recurrent PPO and behavior cloning include deterministic resume, curricula,
  specialist routing, competence gates, self-play infrastructure, immutable
  artifacts, counterfactual evaluation, and observation-legal micro-combat
  search.
- Large-world simulation paths, dense passive-field kernels, trusted-mode hash
  controls, batched inference, Docker CPU/GPU runners, and scale gates are in
  place. The optimized passive benchmark is approximately 8.9 times faster
  than its retained serial oracle; consult the performance history before
  extrapolating to a different workload.
- The local match explorer can seek through complete histories, inspect cells
  and outcomes, switch field layers, and graph team/cell metrics. It is ready
  for local analysis but not yet a hosted leaderboard service.

## Current RL finding

The [deployed combat comparison](contact-deployed-2026-09-12.md) now runs the exact
portable Minds and fails all seven frozen candidates: zero attacks, damage and
kills across 168 episodes. All episodes offer legal occupied-target attacks;
the candidates choose Move on those opportunities. A maintained aggressive
baseline passes with 526 attacks, 2,508 applied damage and 22 kills. Four feeding
sentinel episodes reproduce their archived trial reports exactly. The next
blocking skill is combat action selection. Both the base-head fit and the
[local-context adapter](interaction-slot-adapter-2026-09-12.md) fail feeding
retention. The [per-slot residual](interaction-relational-2026-09-12.md) now
preserves per-slot feature relationships and passes deployment checks for its first pair,
but fails development retention. The [coverage follow-up](interaction-coverage-2026-09-12.md)
adds 44 episodes and two training seeds: combat agreement stabilizes at
97.6–98.6%, while feeding retention still fails on training and development.
The [sampling follow-up](interaction-sampling-2026-09-12.md) now improves
feeding by losing combat; all three treatments fail, including combat on
training rows. The [direct-context probe](interaction-context-2026-09-13.md) now compares
zero, actual and shuffled residual context at three budgets. None of 27
training checkpoints passes; direct memory gives no consistent improvement.
The [hard-example follow-up](interaction-hard-2026-09-13.md) also fails all
24 residual checkpoints, while a standardized linear observation-plus-memory
control fits 128/128 rows. Gradient checks pass at batches 2 and 128. The
[matched standardization check](interaction-standardized-2026-09-13.md) now
fits 128/128 in all three residual runs by 512 updates, with exact raw-control
regression. Next is zero-state-preserving normalization and full-corpus
training: fixture-derived centering otherwise produces extreme inputs for
2,146 zero-memory states.
A Move-target-only adapter cannot directly acquire Attack selection. Self-play
remains gated.

The [fresh-development feeding matrix](fresh-feeding-2026-09-12.md) now passes for
all three sampled-utility initializations. Adjacent-food survival is 94.73–95.00%,
versus 84.45% for the frozen parent and 87.66–88.05% for learned controls;
contested Move outcomes fall from 70.96% for the parent to 29.88–34.23%.
On-food survival remains 100%, with no illegal decisions or safety aborts.
This is 140 unique development episodes on two seeds fresh at preflight and now
recorded as exposed development evidence, not final
confirmation. Every episode reaches 262,144 quanta, with no survivor-count
loss after waiting opponents die at 102,400. A separate host assessment preserves
canonical match-victory semantics. The earlier same-seed study verified its
canonical prefix; the fresh cohort records the boundary using that unchanged
evaluator. Ring survival remains slightly below learned controls despite the
aggregate pass. The combat failure above sets the next training slice. The preceding failure
and diagnosis below explain how this candidate was developed.

The collision-aware maintained forager proves the five feeding layouts are
physically viable. A learned-policy failure previously attributed to an
adjacent `Consume` decision was actually an expert-router semantic bug: diffuse
field energy is plant food, not directly consumable cell food. The corrected
router selects foraging from current plant capacity or loose energy, retains
temporarily exhausted plants, and routes ordinary diffuse-only tiles to
exploration.

With unchanged weights, the correction removes the off-plant `Consume` error
and gives 100% plant retention. Ring adjacent survival rises to 92.8%, but line
and checkerboard remain near 75%. Transition diagnostics show that 80.6% of
line and 85.4% of checkerboard plant-directed Moves are `Contested`. The
remaining dense-layout defect is synchronized target selection.

Schema-18 correction corpora now support verified `move-target-control` and
`move-target-treatment` pairs. Across five layouts they contain 1,453 active
same-effort target changes on otherwise identical policy-induced recurrent
prefixes. The pair verifier proves that observation, private trajectory,
policy action, and every non-action label field are identical.

Training only the inherited target-query head does not solve the seam. Doses
from 22 through 1,024 updates lower cross-entropy but never improve held-out
greedy targets; accuracy eventually falls from 63.2% to 60.0%. This treatment
is rejected. Incomplete ecological runs were stopped and no result artifact
was published.

## Immediate next slice

The [target residual implementation](target-residual-2026-09-11.md) now provides
a zero-initialized, permutation-equivariant target residual whose input is
the raw cell-private random block paired independently with each raw local slot.
Its adapter-only training mode keeps all inherited parameters bit-identical. This adds no shared state or identity: every output row remains
a pure function of one cell's legal Mind input and private memory.

Migration, active slot-permutation, row-isolation, and frozen-parent gates pass.
Matched 22/88/256-update probes preserve the controls but leave treatment Move
exact agreement at 63.25%/63.25%/63.20%. The dose sweep is rejected and ecological
evaluation remains gated. The [margin diagnostic](target-margin-diagnostic-2026-09-11.md)
now shows that increasing this fitted residual's gain trades corrected choices
for losses, and relative scoring depends only weakly on the tested randomness
intervention. The [controlled interaction probe](target-interaction-2026-09-12.md)
now shows that scaling dx/dy raises synthetic fixed-pair validation from 50.24%
to 99.15%. Varying candidate sets expose an exact 92.22% independent-score ceiling
for that synthetic quantile-teacher task; a teacher-informed set-context positive
control reaches 96.36% and passes the random-intervention gate. This is not a bound
on the full contextual policy or evidence of improved ecological performance.
The [learned set follow-up](target-set-2026-09-12.md) compares max and additive
summaries; its fits and the initial relational revisions used a CPU dependency
configuration later found to produce biased batched gradients on ARM. The
[numerical fix](backend-numerics-2026-09-12.md) disables NdArray SIMD while retaining
multithreaded matrix operations. Reciprocal, analytic loss-gradient and real
optimizer regressions pass; the full all-feature workspace passes 610 tests
(33 ignored), and workspace Clippy passes. Treat the earlier failed fits as
historical behavior, not clean architecture-limit evidence.
The [corrected target comparison](target-relational-2026-09-12.md) reruns 54 fits:
additive scoring passes 3/9 gates, relational 5/9, context×quantile product 7/9,
hard-label sampled utility 9/9, marginal-distribution utility 8/9 and rank control
9/9. The hard-label utility candidate reaches 98.29–98.90% mixed-food exact
choices and 94.73–96.32% joint intervention accuracy, from a 42.92% untrained
uniform-sampling baseline. Its native single-cell/32-slot p95 is 0.124 ms.
This is synthetic development evidence, not a promoted policy or 32-slot learned
behavior. The [portable sampler follow-up](sampled-target-deployment-2026-09-12.md)
now closes decoder arithmetic and ordinary-ABI adaptation: 9,223 native/WASM
choices and full integer weight vectors match, with 165 isolation sequences.
The same portable decoder preserves all nine learned-utility gate results exactly.
Its separate execution contract does not enable sampling for old greedy weights.
The [real-observation transfer](feeding-utility-transfer-2026-09-12.md) now passes
all three interval-loss initializations: fresh treatment agreement is
97.99–98.83%, with 98.03–99.12% unchanged-Move retention. All matched controls
pass, and six full-f32 diagnostic records reload with identical parameters and
evaluations. Cross-entropy alone passes only one of three treatments. These are
conditional offline results. The [composite deployment](feeding-composite-deployment-2026-09-12.md)
now retains all six learned utilities with the exact parent weights, preserving
34,794 recorded sampled choices under portable inference. The first matched
control/treatment pair passes 4,896 native/WASM decisions and twenty signed
replay fixtures; all checked outputs beyond Move target are identical to the
parent on the same input. The [three-arm feeding matrix](feeding-ecology-2026-09-12.md) now passes for
all three initializations. The [sustained assessment](sustained-feeding-2026-09-12.md)
also passes all three through 262,144 quanta, with no late survivor-count loss.
The [fresh-development cohort](fresh-feeding-2026-09-12.md) also passes all three
on seeds 1435500201/1435500202. Carry its supplemental evaluation seed ledger
into subsequent work; the frozen exports retain their original ledgers.
Before training
another adapter, use the [seed-lineage checks](seed-lineage.md)
to inherit its parent ledger or audit the historical chain, and reserve untouched
final confirmation seeds. Legacy audits conservatively mark all old corpus seeds
as exposed, so the next stage needs new validation seeds.

The implementation gates are complete: exact-zero migration, slot permutation,
row isolation, frozen inherited parameters, execution identity and the cumulative
schema-37 seed ledger are covered. The remaining experiment gates are:

1. Expand disjoint training-seed coverage for the fixed per-slot interaction
   residual, with matched combat/feeding labels on frozen deployed prefixes.
   The [relational residual](interaction-relational-2026-09-12.md) is implemented:
   exact-zero migration and first-pair WASM checks pass, but all three treatments
   fail development retention despite passing training-row thresholds. Keep
   the architecture and parent fixed for the coverage experiment; preserve
   the all-run gate, then require new-seed evaluation and deployed combat.
   Do not select the strongest initialization from development results.
   Carry the seed audit (61 training, 24 validation, two reserved confirmation
   seeds); 1435600201/1435600202 remain exposed and excluded from training.
   Deployed-Mind combat evaluation is implemented; all seven
   frozen feeding candidates fail its activity/kill gate despite legal occupied
   attack opportunities and a passing maintained baseline. The Move-target-only
   utility cannot directly fix action kind. Fresh-development sustained feeding
   passes all 140 episodes, but one uses 4,014 of 4,096 host steps; retain explicit
   safety-cap checks. Keep final confirmation seeds
   1434999901/1434999902 untouched. Broader seed, visibility, neighborhood-capacity
   and resource-distribution coverage remain open.
2. Keep execution identities and evidence attached to each imported artifact.
   The current composite's feeding result does not qualify unrelated legacy
   checkpoints, combat behavior or future migrations. Retain historical
   code/container digests for reproduction and untouched confirmation seeds.
3. Qualify fixed-opponent combat while checking feeding
   retention, then consider self-play. Keep evaluation concurrency bounded. A
   historical 256x256 evaluator retaining complete histories used roughly 3–9
   GiB per process on `fitty`; do not run six such workers on its 60 GiB host.

## Remaining product work

- Establish robust feeding and combat baselines, then introduce self-play only
  after fixed-opponent competence and retention gates pass.
- Tune physics and victory criteria through paired viability sweeps rather than
  co-adapting rules and policies in one experiment.
- Expand long-horizon 256/512/1024 training without retaining unnecessary full
  histories in evaluation workers.
- Finish production upload authentication, sandboxed source compilation,
  durable job queues, quotas, object storage/retention, season/ruleset pinning,
  signing-key operations, leaderboard APIs, and `tinker.ninja` embedding.
- Run the maintained fuzz targets for resolver conflicts, checkpoint/replay
  parsing, ABI conversion, artifact validation, and adversarial WASM. Extend
  fuzzing to the new correction-pair and model-migration formats.

## Evidence and validation

The [September 11 fuzz campaigns](fuzz-campaigns-2026-09-11.md) completed replay,
Mind ABI, stateful resolver and structural Wasm admission runs. The admission
target also passes 12,480 generated contract cases without executing guest code.
The new resolver harness found and led to a
fix for restoring idle cells after the clock advances; checkpoint bytes and hash
formats are unchanged. Sparse-history resolver and neighborhood campaigns also
pass, with 35,485 and 12,749,252 inputs respectively. Geometry checks led to fixes
for huge empty-world iteration and signed coordinate overflow. Follow-up
observation/movement campaigns pass 806,938 / 2,322,875 inputs, checking actual
field masking, anonymous serialization and local movement rules. A further
712,110 observation inputs pass after adding every activity and exact progress
boundaries, including rejected commitments. Mind projection
now lives in `resolution/reference/observation.rs`; the 470 extracted lines are
text-identical and the public API is preserved. Current gates
pass 607 workspace tests (33 ignored),
workspace Clippy, engine/web WASM library Clippy, formatting and schema checks.
The separate fuzz crate passes all eight deterministic harness tests and Clippy,
including mixed/atomic decision batches; the further resolver campaign passes
35,212 inputs with no finding.
The full test build required an incremental-cache-disabled retry after a rustc
ICE; no source workaround was needed.
The Chromium suite now passes nine tests, including the generated WASM package,
native replay-hash parity and real WebCrypto Ed25519 verification/tamper cases.
Run `npm run prepare:wasm --prefix browser-tests` before `npm test`; CI installs
the pinned wasm-bindgen CLI and prepares fixtures automatically.
Fresh Rust, AssemblyScript and Go builds also pass all 18 guest/runtime gates
(23 tests, including the three normally ignored language tests). See
[language conformance](language-conformance-2026-09-11.md) for pinned tools,
artifact hashes, execution bounds and remaining executable-fuzzing gaps.

The compact experiment narrative and hashes are in
[`sweeps/feeding-dagger-round5-v1/README.md`](../sweeps/feeding-dagger-round5-v1/README.md).
Large generated artifacts remain intentionally ignored under
`training-output/feeding-dagger-round5-v1`. The remote CPU image for the latest
pair collection is
`sha256:1b54eb9200baa44822dcbd84e9575c2478b97ff4d28f74454b5e1819039a49bb`
with label
`f4c7f30e2803-working-tree-20260902-move-target-correction-v1`.

The final local pre-commit gate for this handoff is:

```sh
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
scripts/conformance.sh
npm run prepare:wasm --prefix browser-tests
npm test --prefix browser-tests
```

If host-only dependencies make the complete conformance script unavailable,
record the exact skipped component; do not silently replace it with a narrower
claim.

Handoff verification on 2026-09-05 passed workspace tests with all features,
workspace-wide all-target Clippy with warnings denied, formatting, schema
registry consistency, diff whitespace checks, and both real-Chromium match
explorer tests. The conformance script passed through the native, resolver,
replay, RL, web, WASM-build, and sample-match checks, then stopped at the
cross-language Go canary because TinyGo was not then installed on this host.
The AssemblyScript example built successfully. The September 11 follow-up
built fresh Rust/AssemblyScript/TinyGo artifacts with pinned temporary tools
and passed every guest/runtime gate, including the formerly blocked Go canary.
The full conformance script was not rerun end to end; the current claims above
identify the suites actually executed.

## Detailed references

- [`simulation-resolution-design.md`](simulation-resolution-design.md)
- [`rl-training-and-rules-tuning.md`](rl-training-and-rules-tuning.md)
- [`physics-vs-training-diagnostics.md`](physics-vs-training-diagnostics.md)
- [`large-game-optimization-roadmap.md`](large-game-optimization-roadmap.md)
- [`testing-and-fuzzing-roadmap.md`](testing-and-fuzzing-roadmap.md)
- [`mind-audit-and-colony-roadmap.md`](mind-audit-and-colony-roadmap.md)
- [`docker-training.md`](docker-training.md)
