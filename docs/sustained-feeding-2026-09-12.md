# Sustained feeding assessment — 2026-09-12

All three frozen sampled-utility initializations pass the sustained feeding
gate: **140 unique episodes reach 262,144 quanta**, with no survivor-count loss
after the waiting opponents die at 102,400 quanta. Adjacent-food treatment
survival remains **94.73–94.88%**, versus **85.43%** for the frozen parent and
**87.93–88.87%** for learned controls. On-food survival remains 100%.

The feeding evaluator now supports an explicit sustained assessment that
continues after the waiting opponents die. Canonical matches still end on
extermination. The new assessment stops at the configured time deadline, own-team
extinction or the existing host safety limit; it does not award an assessment win
for eliminating opponents.

This closes the duration limitation identified in the
[previous feeding matrix](feeding-ecology-2026-09-12.md), whose episodes all ended
at 102,400 quanta despite a 262,144-quanta configured deadline. The current
experiment keeps the same frozen parent, all three archived learned
control/treatment pairs, rules, layouts and development seeds.

## Boundary semantics

`BlobEnv` has a private, per-call assessment boundary. Existing training,
baseline and canonical-Mind entry points always choose the canonical match
boundary. Only the feeding assessment entry point can ignore opponent extinction
as a terminal event. The boundary is not stored in checkpoints, attached to
rulesets, passed into observations or exposed to the policy. Canonical engine,
replay and victory formats are unchanged.

The assessment returns once at the exact event where the final opponent dies.
It records the state hash, time, step count, surviving training cells, decision
counters and movement/consume telemetry at that boundary. Subsequent steps
advance to actual training decision frontiers, the time deadline or own-team
extinction; they do not create repeated empty decision steps merely because
opponents are absent.

`FeedingAssessmentMode` is explicit in both plan and result. Deployed feeding
report schema **2** distinguishes `canonical_match` from `sustained_feeding` and
adds the opponent-extinction observation. The default CLI mode remains canonical:

```sh
feeding_deployed_evaluation --assessment-mode sustained-feeding ...
```

The existing feeding threshold report remains a comparison of survival, food
acquisition and safety. A passing threshold report in assessment mode is not a
canonical match win or an automatic policy promotion.

## Fixed comparison and provenance

The assessment runs the same 256×256 worlds, 256 cells per team, five founder
layouts, on-food/adjacent-food stages and development seeds
**1435400201/1435400202**. Source configuration SHA-256 remains
`4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9`.
The deadline is 262,144 quanta and the host cap is 4,096 decision steps. No model
parameters, action rules, energy values or resource placement are changed.

All three initializations were scheduled up front. Parent trials from the first
matrix were reused in the remaining matrices only with identical report
hashes. There are 70 unique trials / 140 unique episodes across all three
matrices; shared parent episodes are not counted as new evidence.

Each sustained episode is checked against its corresponding earlier canonical
trial. The initial state, initial private-decision frontier, effective environment
and rules hashes must agree. At opponent extinction, its state hash, time, step
count, surviving cells and every recorded decision/movement/consume counter must
exactly equal the old terminal report. This checks the earlier terminal state
and recorded counters exactly before comparing the continuation. It does not replace a complete per-event replay of
every ecology episode.

The pre-run comparison gate retains the original feeding criteria: every
individual treatment trial passes the existing feeding thresholds, aggregate
adjacent survival improves and contested-Move fraction falls against both
comparators, and on-food survival loses at most one percentage point against
either comparator. Sustained treatment episodes must also reach the deadline.
All arms must have zero illegal decisions and safety aborts. Post-extinction
survivor changes, consumed energy and decision counts are reported separately.

## Results

All three treatments pass every declared comparison and all ten individual
layout/seed feeding threshold reports. All 140 unique episodes end at exactly
262,144 quanta with `Timeout` / `SimTimeDeadline`; none ends in an assessment
win, own-team extinction or safety abort. Illegal-decision counts are zero.
All initial identities and recorded opponent-extinction boundaries match the
corresponding canonical reports exactly.

Adjacent-food aggregates at the full deadline:

| Initialization suffix | Control survival | Treatment survival | Control contention | Treatment contention | Control food / initial cell | Treatment food / initial cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 301 | 87.93% | 94.88% | 67.67% | 28.91% | 490.29 | 527.70 |
| 302 | 88.87% | 94.84% | 58.25% | 32.25% | 494.89 | 527.73 |
| 303 | 88.48% | 94.73% | 65.40% | 32.10% | 492.97 | 527.04 |

The shared parent has 85.43% survival, 70.35% contention and 477.71 consumed
energy per initial cell. Every arm has 100% on-food survival and 561.11 consumed
energy per initial cell. Survival and contention match the earlier, shorter
canonical results; the added interval contributes sustained consumption.
Across unique episodes, the continuation adds **5,332,080 decisions** and
**10,664,668 consumed energy**, with zero survivor-count change in every episode.

The largest host-step count is **4,009 / 4,096**, for initialization 302's
checkerboard treatment at seed 1435400202, adjacent-food. This is a narrow
remaining margin, not a reason to reinterpret a future safety abort as a
completed assessment. Preserve explicit cap reporting on the fresh cohort.
The three serial matrices take 509.9, 243.5 and 239.5 seconds respectively
(about 16.5 minutes total); the first overlaps workspace validation.

## Evidence and verification

The versioned evidence directories are:

- `training-output/sustained-feeding-2026-09-12-v1`
- `training-output/sustained-feeding-2026-09-12-init1435400302`
- `training-output/sustained-feeding-2026-09-12-init1435400303`

Each contains a pre-run plan, exact evaluator executable and hashes, source
copies, per-trial commands/status/logs, immutable reports, canonical-prefix
references and a checked summary. Each validation directory retains the final
verifier and report hashes. The first also retains the all-matrix audit,
140-episode inventory, test/Clippy logs and eight negative verifier checks.
Use the same verifier for both modes:

```sh
python3 scripts/verify_feeding_ecology.py training-output/sustained-feeding-2026-09-12-v1
python3 scripts/verify_feeding_ecology.py training-output/sustained-feeding-2026-09-12-init1435400302
python3 scripts/verify_feeding_ecology.py training-output/sustained-feeding-2026-09-12-init1435400303
```

Boundary regressions verify an identical canonical prefix, continued assessment
after opponent death, actual deadline termination, own-team extinction, host
safety termination and the unchanged canonical behavior of later normal calls.
The pre-existing deployed-Wait/baseline regression also remains in place.

Validation passes **622 workspace tests (33 ignored)**, workspace Clippy across
all targets/features with warnings denied, formatting, the 64-constant schema
registry and diff whitespace checks. The evidence verifier accepts a valid
completed report and rejects eight deliberate changes to initial identities,
boundary state/counters/time, terminal time, claimed outcome or assessment mode.

## Remaining scope

These are the same exposed development seeds used in the earlier utility and
feeding studies. Sustained feeding establishes longer survival on these
trajectories, not independent confirmation or general competence. The first
control/treatment pair retains its prior WASM qualification; all six learned
utilities here run in the native portable deployment runtime.
The [fresh-development follow-up](fresh-feeding-2026-09-12.md) now passes all three
initializations on two previously unexposed local seeds, with 140 further unique
episodes reaching the full deadline. Next qualify deployed-Mind fixed-opponent
combat and feeding retention before self-play. Broader visibility, neighborhood-capacity and
resource-distribution coverage remain open. Final confirmation seeds
**1434999901/1434999902 remain reserved**.
