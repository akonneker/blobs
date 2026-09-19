# Fresh-development feeding cohort — 2026-09-12

All three frozen sampled-utility initializations pass the fresh-development
feeding gate. Adjacent-food treatment survival is **94.73–95.00%**, versus
**84.45%** for the parent and **87.66–88.05%** for learned controls. All **140
unique episodes** reach the full 262,144-quanta deadline, with zero illegal
decisions, safety aborts or post-extinction survivor-count loss.

This study evaluates the unchanged frozen parent and all three archived
sampled-utility control/treatment pairs on development seeds
**1435500201/1435500202**. They were absent from the audited candidate lineage
and searched local experiment records before this cohort. They are now exposed
development seeds; final confirmation seeds **1434999901/1434999902 remain
reserved**. Future descendants must carry this cohort's development exposure
alongside the frozen exports' earlier ledgers; the immutable exports themselves
are not rewritten after evaluation.

## Prospective comparison

All three initialization plans and their hashes were written before the first
trial. The comparison retains the preceding [sustained feeding study](sustained-feeding-2026-09-12.md):
256×256 worlds, 256 cells per team, five founder layouts, on-food/adjacent-food
stages, 262,144-quanta deadline and a 4,096-step host safety cap. Evaluation uses
one process with four Rayon threads. No weights, rules, resource placement,
energy settings, thresholds or runtime arithmetic are changed.

The evaluator is copied byte-for-byte from the qualified sustained study:
SHA-256 `ce0eec0f4c3c7d566bf993eb9f1df99195cc1012714407390ced38eafac94cc2`.
Configuration SHA-256 remains
`4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9`.
All six learned utilities run in the native portable deployment runtime. Only
the first control/treatment pair has the preceding complete WASM qualification;
this cohort does not add guest-execution coverage for the other pairs.

The gate requires every individual treatment trial to pass the existing feeding
threshold report, aggregate adjacent-food survival to improve and contested-Move
fraction to decrease against both comparators, and at most one percentage point
of on-food survival regression against either comparator. All treatment episodes
must reach the deadline. Every arm must have zero illegal decisions and safety
aborts. A failed comparison does not skip later predeclared candidates; an
infrastructure failure stops execution and retains its evidence.

Parent trials are computed once and reused for the other initializations only
when the copied reports are byte-identical and their source matrix plan matches
the predeclared hash. The full cohort has **70 unique trials / 140 unique
episodes**; shared parent results are not counted as new observations.

## Seed and evidence checks

The seed audit binds all seven export manifests by hash, including the cumulative
training, validation and confirmation ledgers of all six composite exports.
It also includes the parent's recorded development seeds and the preceding
feeding seeds. A pre-run search checks JSON/TOML/Markdown records under local
`training-output` and `docs`, including ignored files. This is evidence about
declared local lineage and records, not a global guarantee about undeclared
external experiments.

The verifier rejects overlapping, duplicate or invalid seeds, omitted candidate
pairs, failed historical searches, altered manifests/audits and mislabeled arms
or initializations. All three arms and all three initializations must have
identical initial state, private-decision frontier, environment and rules hashes
for each layout/seed/stage. Reporting recomputes the original feeding thresholds,
survival and movement arithmetic, safety classification and comparison gates.

Unlike the preceding same-seed continuation study, these fresh episodes do not
have separate canonical reports for a direct prefix comparison. Plans explicitly
declare `recorded_boundary_only`: extinction time and cumulative counters are
checked for consistency, and the exact previously validated evaluator supplies
the boundary semantics. No full per-event replay is claimed for this cohort.

## Results

Adjacent-food aggregates over the ten layout/seed trials per arm:

| Initialization suffix | Control survival | Treatment survival | Control contention | Treatment contention | Control food / initial cell | Treatment food / initial cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 301 | 87.66% | 95.00% | 66.43% | 29.88% | 488.66 | 528.36 |
| 302 | 88.05% | 94.73% | 58.90% | 33.11% | 490.55 | 527.08 |
| 303 | 87.93% | 94.80% | 64.45% | 34.23% | 489.94 | 527.49 |

The shared parent has 84.45% survival, 70.96% contention and 472.38 consumed
energy per initial cell. Every arm has 100% on-food survival and 561.08 consumed
energy per initial cell. All three treatments pass every individual trial's
feeding thresholds and every predeclared aggregate comparison.

The gain is not uniform across layouts. Ring treatment survival is
94.53–94.92%, versus 94.73–95.70% for learned controls; each treatment is slightly
below its matched control there. Checkerboard treatment survival is
88.87–90.23%, versus 74.41% for the parent and 74.41–75.20% for controls. Thus the
aggregate gate passes without claiming superiority in every layout.

Every episode reports `Timeout` / `SimTimeDeadline` at exactly 262,144 quanta.
Each recorded opponent-extinction boundary occurs at 102,400 quanta, and every
individual episode has zero subsequent survivor-count change. The maximum step
count is **4,014 / 4,096**, for initialization 301's checkerboard treatment at
seed 1435500201, adjacent-food. The cap has only 82 steps of margin in that case;
broader evaluations must retain explicit safety-abort checks.

The serial matrices took 359.7, 244.3 and 241.3 seconds, about **14.1 minutes**
in total. These results support transfer to this two-seed development cohort,
not a statistical claim about all seeds or environments.

## Evidence and validation

Evidence lives in `training-output/fresh-feeding-2026-09-12-v1`. The root contains
the pre-run seed audit and cohort plan; `matrices/1435400301`,
`matrices/1435400302` and `matrices/1435400303` retain frozen plans, evaluator
executables, source snapshots, per-trial commands/logs/reports and summaries.
The root summary verifies every predeclared plan hash, pairing across all seven
Minds and the exact reused-parent reports; it includes report hashes and a
140-episode inventory. `evaluation-seed-ledger.json` retains the supplemental
post-evaluation ledger: 57 training, 22 validation and two reserved confirmation
seeds. This is an exposure record to carry alongside future lineage, not a
rewritten export or automatically imported training artifact.

```sh
python3 -B scripts/verify_fresh_feeding.py training-output/fresh-feeding-2026-09-12-v1
python3 -B scripts/test_feeding_cohort.py
```

The fixed study runner is `scripts/run_fresh_feeding.py`; it refuses an existing
output directory or prior local exposure of its proposed seeds. The ten
self-contained lineage tests run in CI and the local conformance script.
Rust simulation and policy code are unchanged from the preceding 622-test
workspace validation. All six earlier canonical/sustained matrices still pass
the updated verifier. Seven deliberate report changes are rejected: premature
deadline, claimed assessment win, inconsistent boundary time/steps/consumed
energy, and false own-extinction or safety-abort outcomes.
An actual duplicate-run invocation is also rejected before modifying the cohort.
Final Python/shell syntax, formatting, schema-registry and diff checks pass;
validation scripts and logs are retained with the evidence.

## Remaining scope

The fresh-development feeding gate is complete. The [deployed combat follow-up](contact-deployed-2026-09-12.md)
now evaluates this composite contract through canonical Mind calls. All seven
frozen candidates fail: zero attacks despite legal occupied-target opportunities.
The maintained aggressive baseline passes, while four rerun feeding sentinel
episodes reproduce their archived reports exactly. Combat action selection or
routing needs acquisition and joint feeding/combat qualification before self-play.
Two fresh seeds are a limited development cohort;
broader seed, resource, visibility and neighborhood-capacity coverage remains
open. No final confirmation or production promotion is claimed.
