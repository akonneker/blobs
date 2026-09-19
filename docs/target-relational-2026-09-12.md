# Relational and sampled target selection — 2026-09-12

The [portable sampler follow-up](sampled-target-deployment-2026-09-12.md) now
closes the decoder arithmetic/ABI gate and preserves all nine selected-utility
results. The remaining transfer is training on actual production observations.

The hard-label sampled utility model passes all nine synthetic gates on the
corrected CPU backend. Mixed-food target accuracy rises from the untrained
uniform decoder’s 42.92% to 98.29–98.90%, with 94.73–96.32% joint intervention
accuracy. It is the candidate for the next production-input gate; no policy
checkpoint is promoted from this synthetic result.

This comparison follows the [learned-set probe](target-set-2026-09-12.md).
The first four revisions exposed a [CPU reciprocal defect](backend-numerics-2026-09-12.md).
Revision 5 repeats all 54 fits with the corrected backend. All runs are retained;
none exports policy weights or establishes ecological competence.

## Controlled protocol

Every fit uses 16,384 training rows, 4,096 development-validation rows, 1,024
AdamW updates, batch size 128, learning rate 0.005 and norm-1 gradient clipping.
The unchanged gate requires **every** task and initialization to reach 95% exact
teacher choices and 90% jointly correct original/intervened choices among rows
whose teacher choice changes. Evaluation also asserts finite scores and bit-exact
slot and batch-row permutation. Random intervention flips bit 7 of private byte 7,
shifting the first little-endian u64 quantile by one half.

The three tasks use nonempty subsets of four cardinal destinations, all eight
Moore destinations with equal food, and eight destinations with independent food
levels 1–3. The teacher chooses uniformly by private quantile among best-food
legal destinations. These are local synthetic features, not complete Mind ABI
observations or a simulation of feeding costs.

Training seed 1435200101, development-validation seed 1435200202 and initialization
seeds 1435200301/302/303 are now exposed. Every revision deliberately reuses them
for diagnosis. Reserved confirmation seeds 1434999901 and 1434999902 remain unused.

The shared score network uses tile-scaled dx/dy and food/3, a 3→16→16 tanh encoder,
a 64-unit ReLU hidden layer and a zero-initialized scalar head. No network receives
host identities or tensor row indices. The final layout has 6,369 nominal
parameters, including unused parameters and input columns in some controls;
effective capacity is not matched by that count.

| Arm | Additional computation |
| --- | --- |
| LearnedSum | Sorted sum of encoded legal candidates, divided by slot capacity, repeated per destination. |
| Relational | For each destination, learn 3→16→16 tanh features of every other candidate's dx/dy/food difference; mask illegal competitors, then sorted sum/capacity. Self-comparisons are included. |
| RelationalProduct | Add a learned projection of the relational context multiplied by the decoded private quantile to the score hidden layer. That projection starts at exact zero. |
| QuantileUtility | LearnedSum network with all random inputs zero; fit hard sampled teacher labels, then use a categorical decoder driven by the private u64. |
| MarginalUtility | Same network and decoder as QuantileUtility, but supervise the uniform probability distribution over best-food legal candidates. Distribution labels never enter network features. |
| RankContext | Teacher-informed positive control with best-food membership, rank midpoint and reciprocal candidate count. Those features are zero in all learned arms. |

The first three arms and rank control take the greedy maximum of learned scores.
The utility arms learn probabilities and leave random selection to the decoder.
The native prototype decoder computes relative exponential weights in f64,
rounds them to integers scaled by 2^48, and locates the private u64 in the
cumulative integer weights using u128 multiplication. Input candidates are in
canonical geometry order. Equal logits reproduce exact teacher u64 boundaries;
tests cover every nonempty eight-slot subset and all candidate counts through 32.
The decoder rejects invalid legal logits and has a 32-slot bound. It is not yet a
portable numerical contract or WASM implementation.

The utility models start at 100% on equal-food tasks because the zero output head
and decoder already implement uniform choice. That is an explicit decoder prior,
not a learned skill. The mixed-food improvement over initial uniform sampling is
the relevant evidence that the utility network learned something.

## Retained development revisions

| Evidence directory suffix | Change | Gates passed out of 9 |
| --- | --- | --- |
| v1 | Basic relative comparisons | Sum 3, relational 4, rank control 9 |
| v2 | Explicit context×quantile product | Product 6; preceding arms unchanged |
| v3 | Sampled utility with hard labels | Utility 6; preceding arms unchanged |
| v4 | Teacher-marginal distribution loss | Marginal utility 5; preceding arms unchanged |
| v5 | Corrected CPU backend, all models/losses/budgets retained | See corrected results below |

Directories are `training-output/target-relational-2026-09-12-v1` through `v5`.
The historical runs used the defective CPU reciprocal path. Common arms have
identical validation, loss checkpoints and summary diagnostics across v1–v4;
the verifier checks this. Their failed fits are not clean architecture-limit
estimates. The marginal arm drifting from its already-optimal uniform policy
provided the diagnostic that led to the numerical fix.

## Corrected results

Ranges span three initializations on the same development data. All 54 models
pass the finite-score and exact permutation assertions.

| Task | Arm | Exact choices | Joint intervention | Gates / 3 |
| --- | --- | ---: | ---: | ---: |
| FourEqual | LearnedSum | 98.19–98.44% | 95.67–96.20% | 3 |
| FourEqual | Relational | 98.73–98.93% | 96.90–97.37% | 3 |
| FourEqual | RelationalProduct | 99.22–99.46% | 97.80–98.37% | 3 |
| FourEqual | QuantileUtility | 100.00–100.00% | 99.97–99.97% | 3 |
| FourEqual | MarginalUtility | 100.00–100.00% | 100.00–100.00% | 3 |
| FourEqual | RankContext | 98.63–98.78% | 96.47–96.74% | 3 |
| EightEqual | LearnedSum | 90.84–91.26% | 81.81–82.42% | 0 |
| EightEqual | Relational | 92.94–93.26% | 86.02–86.73% | 0 |
| EightEqual | RelationalProduct | 97.27–98.07% | 95.16–96.15% | 3 |
| EightEqual | QuantileUtility | 99.58–99.76% | 99.37–99.62% | 3 |
| EightEqual | MarginalUtility | 100.00–100.00% | 100.00–100.00% | 3 |
| EightEqual | RankContext | 97.09–97.46% | 94.93–95.59% | 3 |
| EightMixed | LearnedSum | 93.43–94.58% | 75.55–80.45% | 0 |
| EightMixed | Relational | 96.88–97.83% | 88.82–92.18% | 2 |
| EightMixed | RelationalProduct | 96.73–98.12% | 87.86–93.45% | 1 |
| EightMixed | QuantileUtility | 98.29–98.90% | 94.73–96.32% | 3 |
| EightMixed | MarginalUtility | 96.90–99.73% | 88.36–99.14% | 2 |
| EightMixed | RankContext | 98.73–98.97% | 95.27–96.36% | 3 |

The marginal utility arm passes eight of nine gates. Its first mixed-food fit
reaches 96.90% exact choices but only 88.36% joint intervention accuracy. Fixing
the reciprocal defect restores its stationary equal-food behavior but does not
make distribution supervision uniformly better under this budget. Retain it as
an ablation, rather than selecting its best initialization.

Additive scoring passes 3/9 gates, basic relational scoring 5/9, the explicit
product 7/9, hard-label sampled utility 9/9, and the engineered rank control 9/9.
The sampled utility result separates learning destination utility from computing
a quantile-to-rank mapping in a neural score. It does not prove that the summary
is necessary, or that this is the smallest sufficient utility network.

## Cost and capacity

The corrected run's single-cell, 32-slot p95 costs are 0.120 ms for LearnedSum,
0.720 ms for Relational, 0.711 ms for RelationalProduct, and 0.124 ms for the
sampled utility including its native decoder. All clear the predeclared local
10 ms gate. With 32 cells and 32 slots, the corresponding p95 costs are 1.040,
15.826, 15.761 and 0.899 ms. These local timings include tensor construction and
output materialization, five warmups and 101 measured calls per configuration.
They are not portable performance guarantees or before/after speedup estimates.

A relational pair-feature tensor alone occupies 65,536 bytes for one 32-slot cell
and 2,097,152 bytes for 32 cells; these are not peak-RSS measurements. The utility
arms use no pair-feature tensor. Exact 32-slot permutation is tested separately
for the active relative/product modules. All training tasks still have at most
eight slots. Thirty-two-slot timing and shape support do not establish learned
behavior at that capacity, ordinary ABI legality, or deterministic WASM fuel.

## Reproduction and verification

Implementation is `blob_rl/examples/target_set_probe.rs`, with separate
`target_set_probe/{relational,benchmark,sampling}.rs` modules. Every run retains a
pre-training plan, per-fit progress, complete report, source copies, the exact
executable, command/environment/binary hash, log and exit status. Revision 5 also
copies its manifest and lockfile. Recompiling an old snapshot requires restoring
`probe.rs` to its original filename, `target_set_probe.rs`, for its self-hash.

```
cargo test --locked -p blob_rl --no-default-features --features ndarray --test backend_numerics --example target_set_probe
cargo build --release --locked -p blob_rl --no-default-features --features ndarray --example target_set_probe
RAYON_NUM_THREADS=4 target/release/examples/target_set_probe --output training-output/target-relational-new-run
python3 scripts/verify_target_probe.py training-output/target-relational-2026-09-12-v5
```

The verifier checks archived source/binary hashes, expected model coverage,
accuracy/count/gate arithmetic and reported cost-gate arithmetic. Its optional
`--unchanged-from` checks common training results across v1–v4. Those historical
runs need `--historical-lock training-output/backend-numerics-2026-09-12/before-Cargo.lock`.
It does not recreate fitted weights or independently repeat timings. The older
learned-set verifier additionally binds to live source hashes, so it requires its
matching historical checkout; do not rewrite old evidence to match this source.

Validation after the dependency fix: the full workspace with all features passes
610 tests, with 33 intentionally ignored; the probe passes 13 focused tests and
the numerical integration suite passes two. Workspace Clippy with all targets and
features passes with warnings denied. Schema validation still covers 63 constants.

## Next gate

Use QuantileUtility as the candidate, preserving the hard-label loss and bounded
local-input discipline. First establish the categorical decoder's portable
numerical contract and canonical geometry ordering through the ordinary Mind
input, including malformed/empty masks and up to 32 slots. Keep the existing
greedy deployment contract explicit; a sampled policy needs a distinct identity.
Then fit on the retained collision-sensitive control/treatment observations with
corrected CPU arithmetic, frozen inherited heads and audited seed lineage. Check
matched target agreement, retention and new development seeds before spending
the untouched confirmation seeds or running ecological match matrices.

The remaining gaps to a functional learned agent include this production-input
transfer, deployment parity for the chosen decoder, robust feeding/combat and
long-horizon retention. Synthetic target selection is one prerequisite, not that
finished agent.
