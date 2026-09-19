# Interaction training coverage — 2026-09-12

**Adding two training seeds stabilizes combat agreement but does not recover
feeding retention. All three treatments fail; none is promoted.** Training
feeding retention also falls below the gate on the expanded corpus. The [error-balanced sampling follow-up](interaction-sampling-2026-09-12.md)
now completes that intervention: feeding improves at the expense of combat,
and all treatments still fail. Its next step is a training-only separability
and optimization probe.

## Controlled intervention

The predeclared study pools training seeds 1435600101, 1435700101 and
1435700102. The two new seeds were checked against the inherited ledger and
local evidence before collection. The frozen collector adds 44 episodes,
479 combat rows and 5,154 feeding rows with zero safety aborts. Training now
contains **890 combat and 7,422 feeding rows**, versus 411 and 2,268 before.
Development remains the same 586 combat and 5,110 feeding rows from seeds
1435600201/202. The cumulative ledger is 63 training, 24 validation and two
reserved confirmation seeds; 1434999901/1434999902 remain untouched.

The [per-slot residual](interaction-relational-2026-09-12.md), frozen parent and
Move utility, teacher labels, three initializations, optimizer and update
budget are unchanged: 3,418 residual parameters, 1,024 AdamW updates at 0.005,
128-row batches with 64 uniformly sampled combat and 64 feeding rows. Each
matched pair uses identical sampling streams. The larger corpus receives fewer
average presentations per row under this fixed budget. NdArray SIMD remains
disabled. No validation row enters training.

Collection uses the archived collector and the ordinary frozen Mind's actual
prefixes. Combat teacher labels are recorded without executing them. Feeding
labels retain the frozen kind. Runtime, physics, ABI and assessment boundaries
are unchanged.

## Results

All controls retain 100% combat and feeding agreement. Treatment results on
development seeds 201 / 202:

| Initialization | Combat agreement | Teacher-Attack agreement | Feeding retention |
| --- | ---: | ---: | ---: |
| 301 | 97.93% / 98.65% | 99.64% / 100.00% | 94.70% / 98.28% |
| 302 | 98.28% / 97.64% | 100.00% / 98.97% | 94.70% / 98.28% |
| 303 | 97.93% / 98.65% | 99.64% / 100.00% | 94.66% / 98.28% |

Every treatment passes the combat thresholds but misses 99% feeding retention
on both development seeds. Combat agreement is more consistent across
initializations than in the single-seed study; feeding remains essentially
unchanged. On training rows, combat agreement is **97.64%** in all three runs,
while feeding retention is **97.80%, 97.82% and 97.74%**. The expanded corpus
therefore exposes a fitting problem as well as the development failure.

All 177 / 177 / 178 development feeding errors are unwanted Attacks. Seed 201
accounts for 132 / 132 / 133 errors, concentrated in Line (61), Checkerboard
(53) and Ring (18). Training has 163 / 162 / 168 feeding errors, all Move labels
predicted as Attack. The new training seed 1435700102 contributes 109–111 of
those errors. Almost all training feeding rows already have legal occupied
Attack opportunities (7,413 of 7,422), so merely balancing that legality flag
would not isolate the failure.

There are six conflicting groups among complete nonrandom training
observations: 16 rows and a seven-error lower bound for a function of only
those features. Adding the recorded private recurrent memory removes these
exact conflicts; exact full policy inputs also have no conflicts. This does
not prove that memory is sufficient to generalize, nor that these few groups
explain the much larger error count. The residual has no direct memory input,
but the frozen parent scores already depend on memory. Do not interpret the
observation-only lower bound as an impossibility result for the full Mind.

## Verification and code

The fit loader and independent replay command now share explicit ordered
training partitions and per-seed corpus roots. They reject empty, duplicate,
reclassified or reserved training seeds and missing explicit roots. Legacy
single-seed plans retain their original loading order. Re-running all six
previous single-seed fits reproduces **every result record and weight byte**.

All **34,176 validation predictions** match the independently loaded portable
models and the fit's recorded choices, with zero reported Burn/portable kind
mismatches. Another **49,872 training predictions** are independently replayed.
Every nested parent and Move-utility byte is preserved; zero-residual output
migration is bit-identical on all **14,008** corpus rows.

Audit checks bind the collector, sources, corpora, partition ledger, fixed
plan fields and exported weights. Four in-memory mutation checks reject a
reordered partition, forged training count, swapped validation predictions
and forged treatment gate. The prediction check compares the entire ordered
sequence, not only aggregate agreement. The legacy diagnostic report also
reproduces exactly after multi-seed support.

Validation includes partition tests, the active residual model test, both
backend numerical regressions, all-target/all-feature RL Clippy, formatting,
the 66-constant schema registry and whitespace checks. These six new fitted
artifacts have **native evidence only**. Previous fitted models' WASM checks
do not qualify these weights. No new development cohort, ecological rollout
or promotion is claimed.

Evidence lives in `training-output/interaction-coverage-2026-09-12-v1`.
Recompute the checks with `scripts/verify_interaction_coverage.py`; inspect
`scripts/diagnose_interaction_fit.py` and
`scripts/diagnose_interaction_coverage.py` for the read-only error breakdowns.
The root retains plans, source/binary hashes, corpora, fits, independent replay,
regression evidence, diagnoses and mutation checks.

## Next gate

The following intervention has now completed in the
[sampling follow-up](interaction-sampling-2026-09-12.md), with all treatments
failing. This records the proposal that preceded that result.

Predeclare a fixed training-only hard-error sampling partition using the
retained training predictions, and compare it with uniform sampling at the
same budget and architecture across all three matched initializations.
Freeze the error partition before fitting, share it across paired arms, and
keep development errors out of both selection and sampling. Retain all runs
and the existing 95% combat / 90% teacher-Attack / 99% feeding thresholds.
This tests whether the uniform objective underfits the recurring feeding
errors; it is not a promise that reweighting will solve them. If training
retention still fails, inspect optimization and legal recurrent context before
collecting more data blindly. Only after all existing gates pass should a new
development cohort and actual deployed combat/feeding outcomes be spent.
