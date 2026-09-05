# Project handoff

Updated 2026-09-05. The `extism` branch is the active integration branch.
This document is the short operational index; the linked design documents and
versioned sweep notes remain the detailed evidence.

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

Add a zero-initialized, permutation-equivariant target residual whose input is
the raw cell-private random block paired independently with each raw local slot.
Give it an adapter-only training mode so all inherited parameters remain
bit-identical. This adds no shared state or identity: every output row remains
a pure function of one cell's legal Mind input and private memory.

Before another ecological run:

1. Add a migration from the current model schema that initializes the residual
   to exact zero and prove pre/post-migration outputs are identical.
2. Prove slot permutation permutes residual target logits while leaving global
   outputs unchanged.
3. Prove changing one batch row's private random bytes cannot affect another
   row.
4. Bind the expert-routing semantic version to new behavior-clone metadata;
   old artifacts currently require the documented code/container digest to
   disambiguate routing behavior.
5. Train matched target-adapter control/treatment arms and require improved
   exact target agreement on disjoint seeds before running any match matrix.
6. Qualify only the first mechanistically active dose on the five-layout
   matrix, with bounded concurrency. A 256x256 evaluator retaining complete
   histories used roughly 3–9 GiB per process on `fitty`; do not run six in
   parallel on its 60 GiB host.

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
cross-language Go canary because TinyGo is not installed on this host. The
AssemblyScript example built successfully. Re-run the full script on a host
with the pinned TinyGo toolchain before claiming complete cross-language PDK
conformance for this commit.

## Detailed references

- [`simulation-resolution-design.md`](simulation-resolution-design.md)
- [`rl-training-and-rules-tuning.md`](rl-training-and-rules-tuning.md)
- [`physics-vs-training-diagnostics.md`](physics-vs-training-diagnostics.md)
- [`large-game-optimization-roadmap.md`](large-game-optimization-roadmap.md)
- [`testing-and-fuzzing-roadmap.md`](testing-and-fuzzing-roadmap.md)
- [`mind-audit-and-colony-roadmap.md`](mind-audit-and-colony-roadmap.md)
- [`docker-training.md`](docker-training.md)
