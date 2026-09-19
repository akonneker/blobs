# Deployed feeding ecology — 2026-09-12

The deployed sampled utility now improves feeding on actual evolving
trajectories. The first initialization passes the complete, declared three-arm
matrix: adjacent-food survival rises from **85.43% for the frozen parent** and
**87.93% for learned control** to **94.88% for treatment**. Contested Moves fall
from **70.35% / 67.67%** to **28.91%**. On-food survival remains **100%** in all
three arms. No actions fail the observable legality check and no episode hits
its host safety limit.

This follows [composite deployment](feeding-composite-deployment-2026-09-12.md).
The result is a feeding development gate, not final confirmation, combat
competence or general learned-agent robustness. Both remaining archived
initializations also pass the same matrix in a separate extension: treatment
survival is **94.73–94.88%** across all three initializations, with **100% on-food
survival** throughout.

## Evaluator and fixed experiment

`blob_rl::feeding_deployed::evaluate_deployed_feeding` executes `DeployedPolicy`
through `BlobEnv`'s ordinary reference-Mind input path. Each invocation receives
its canonical cell-private randomness and private memory; returned decisions
and memory go back through the resolver. Host diagnostic counters never enter
policy features. This uses the exact native portable execution contract already
qualified against WASM, rather than running a WASM guest for every ecology
invocation.

`feeding_deployed_evaluation` binds each trial to its export manifest SHA,
weight SHA, execution identity, ABI identity and source configuration. It checks
confirmation reservations before constructing an environment or creating output.
Outputs are immutable. Each episode records the initial canonical state hash,
a hash of its complete initial anonymous decision frontier, effective environment
hash, rules hash and final state hash. These initial identities agree across
all paired arms, including private draws and memory.

The source remains `blob_rl/config/combat_warm_start_256.toml`, SHA-256
`4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9`:

- 256×256 worlds, 256 cells per team; line, checkerboard, ring, loose-random and
  random founder layouts.
- Both on-food and adjacent-food stages, with the existing deterministic food
  placement and waiting opponent.
- Development seeds **1435400201/1435400202**; these already informed utility
  development and are not untouched confirmation.
- Fixed parent weights and archived control/treatment weights, with no training,
  source-configuration changes or physics tuning.
- Existing 262,144-quanta deadline and 4,096-step host cap; one trial process at
  a time, four Rayon threads. Each trial covers both stages.

The plan was saved before evaluation. Treatment must pass every existing
feeding-promotion check in all ten layout/seed trials, have a strictly lower
aggregate adjacent-Move contested fraction and more adjacent survivors than
both parent and control, and lose at most one percentage point of on-food
survival against either comparator. The verifier additionally requires no
illegal decisions or safety aborts. The frozen parent remains a separate
baseline because the learned control is not globally identical to it.

## First-initialization results

Initialization 1435400301 was chosen by its existing order before runtime
qualification, not by taking the best ecological score. All 30 trials / 60
episodes completed in about three minutes.

| Adjacent-food metric | Frozen parent | Learned control | Treatment |
| --- | ---: | ---: | ---: |
| Survival | 85.43% | 87.93% | 94.88% |
| Contested / completed Moves | 70.35% | 67.67% | 28.91% |
| Consumed energy / initial cell | 211.16 | 215.94 | 231.66 |

Survival averaged across the two seeds for each layout:

| Layout | Frozen parent | Learned control | Treatment |
| --- | ---: | ---: | ---: |
| Line | 75.00% | 84.57% | 94.73% |
| Checkerboard | 75.78% | 76.17% | 90.23% |
| Ring | 92.77% | 94.92% | 94.34% |
| Loose-random | 83.98% | 84.38% | 96.09% |
| Random | 99.61% | 99.61% | 99.02% |

The gains are concentrated in collision-sensitive layouts. Treatment is slightly
below control on ring and below both comparators on random, so this is not a
claim of improvement in every layout. Contention fractions aggregate all
completed Moves, not only initial food-directed Moves. Food acquisition is
measured from actual consume outcomes, excluding requested but unavailable food.

## Limits and next gate

After the first matrix passed, both remaining archived control/treatment pairs
were evaluated without changing weights, data, rules or thresholds. Their ten
parent trials were reused with verified report hashes and matching initial
frontier identities. There are **70 unique trials / 140 unique episodes** across
the three matrices; reused parent episodes are not additional observations.

| Initialization suffix | Control survival | Treatment survival | Control contention | Treatment contention | Treatment food / cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| 301 | 87.93% | 94.88% | 67.67% | 28.91% | 231.66 |
| 302 | 88.87% | 94.84% | 58.25% | 32.25% | 231.80 |
| 303 | 88.48% | 94.73% | 65.40% | 32.10% | 231.48 |

All three treatments pass every declared comparison and every individual
feeding-promotion trial. The initialization extension reuses the same two
development seeds; it adds initialization robustness, not an independent
confirmation cohort. The first pair retains its prior WASM qualification; all
three pairs here run in the native portable deployment runtime.

All episodes terminate with an ordinary extermination win
at **102,400 quanta**, when the waiting opponents die. The configured deadline
is an upper bound, not the observed duration. The three arms share this stopping
condition, so their comparison is valid; it does not demonstrate survival to
262,144 quanta. A sustained-feeding assay must continue to its time boundary
without ending merely because an inactive opponent disappears, and must label
that assessment separately from canonical match victory.

The [sustained-feeding follow-up](sustained-feeding-2026-09-12.md) now closes
this duration gap for all three initializations: all 140 unique episodes reach
262,144 quanta, with no survivor-count loss after opponent extinction. Its
separate assessment boundary preserves canonical match-victory semantics.

Keep confirmation seeds **1434999901/1434999902 untouched**. The next priority
is a fresh-development cohort, then fixed-opponent combat and retention before self-play. Visibility and
full-capacity learned competence also remain open.

## Evidence

The first matrix is retained at
`training-output/feeding-ecology-2026-09-12-v1`: pre-run plan, exact executable and
hash, source copies, per-trial commands/status/logs, immutable trial reports,
summary and report hashes. Its parent comparison trials can be reused only with
matching report hashes, configuration and initial-state/frontier identities.

```sh
python3 scripts/verify_feeding_ecology.py training-output/feeding-ecology-2026-09-12-v1
python3 scripts/verify_feeding_ecology.py training-output/feeding-ecology-2026-09-12-init1435400302
python3 scripts/verify_feeding_ecology.py training-output/feeding-ecology-2026-09-12-init1435400303
```

The verifier checks matrix completeness, exact pairing, report arithmetic,
legality/safety counts and the declared comparison gate. It does not replay the
whole matrix or grant a production promotion. The existing feeding-promotion
report validator also runs inside every trial.

Validation passes **620 workspace tests (33 ignored)**, all-target/all-feature
workspace Clippy with warnings denied, formatting, the **64-constant** schema
registry and diff whitespace checks. A new regression compares deployed Wait
against the existing baseline at identical initial states and decision
frontiers. A real CLI rejection check proves reserved confirmation seeds fail
before creating output or running the environment. Logs and final verifier
outputs are retained with the first matrix.
