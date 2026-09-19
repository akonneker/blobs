# Interaction error-balanced sampling — 2026-09-12

**The sampling intervention improves feeding retention by losing combat
agreement. All three treatments fail the combined gate; none is promoted.**
The same tradeoff appears on training rows. Uniform sampling reruns reproduce
all six previous result records and weight files exactly. The
[direct-context diagnostic](interaction-context-2026-09-13.md) now also fails
to recover joint training retention; its next step is a small hard-example
fitting and conditioning check.

## Frozen intervention

This follows the [coverage study](interaction-coverage-2026-09-12.md) without
new collection, seeds, model inputs or parameters. The ordered training corpus
remains 890 combat and 7,422 feeding rows from three training seeds. The
5,696 development rows remain excluded from training. The cumulative ledger
stays at 63 training, 24 validation and two reserved confirmation seeds;
1434999901/1434999902 are untouched.

Before fitting, the study freezes the union of feeding training errors from
all three coverage treatments: **168 hard rows and 7,254 remaining rows**.
Membership is recomputed from source training predictions and frozen feeding
labels; no development prediction selects a row. The pool, source replay,
corpus audit, ledger and plan have recorded digests.

Each update samples 64 combat rows uniformly, 32 hard feeding rows and 32
remaining feeding rows, all with replacement. Both control and treatment arms
use the same frozen pools and sampling stream. Each hard row consequently has
about 43 times the sampling probability of each remaining feeding row. This
is a deliberate fixed objective change, not an estimate of natural encounter
frequency. The model, frozen parent and Move utility, labels, initializations
301/302/303, 1,024-update budget, batch size, AdamW configuration and learning
rate 0.005 remain unchanged. All runs are retained. NdArray SIMD stays disabled.

The comparator is the coverage study's uniform sampler, rerun through the new
trainer with its original plan. Every result record and weight byte matches
all six archived uniform fits. No best initialization is selected.

## Results

All controls retain 100% combat and feeding agreement. Development treatment
results on seeds 1435600201 / 1435600202:

| Initialization | Combat agreement | Teacher-Attack agreement | Feeding retention |
| --- | ---: | ---: | ---: |
| 301 | 57.59% / 42.57% | 57.50% / 41.92% | 99.40% / 99.96% |
| 302 | 72.76% / 65.20% | 73.21% / 65.64% | 97.79% / 99.66% |
| 303 | 86.21% / 74.32% | 87.50% / 74.91% | 97.67% / 99.20% |

The unchanged gate requires at least 95% combat agreement, 90% teacher-Attack
agreement and 99% feeding retention on both seeds, with zero portable/Burn
kind mismatches. Treatment 301 passes feeding retention but loses most of its
combat agreement; the other two fail both combat and first-seed feeding.
Uniform treatments previously reached 97.64–98.65% development combat
agreement while missing feeding retention.

The tradeoff is already visible on training rows:

| Initialization | Combat agreement | Feeding retention | Hard-pool feeding correct |
| --- | ---: | ---: | ---: |
| 301 | 70.90% | 99.87% | 159 / 168 |
| 302 | 83.82% | 99.56% | 137 / 168 |
| 303 | 89.10% | 99.07% | 100 / 168 |

The remaining 7,254 feeding rows incur only one, two and one errors. Training
feeding errors fall from 163 / 162 / 168 under uniform sampling to 10 / 33 / 69.
This confirms that prioritizing the fixed pool changes the learned boundary,
but does not establish that sampling alone can satisfy both objectives.
One tested sampling ratio is not evidence that every possible ratio must fail.

The underlying observation conflicts are unchanged from the coverage study:
six groups, 16 rows, and seven unavoidable label disagreements for a function
of only the complete nonrandom observation. Adding recorded private recurrent
memory removes those exact conflicts. These few conflicts do not explain the
full error count, and the frozen parent already depends on memory. Neither
these counts nor this failed fit prove an architectural impossibility.

## Code and verification

`blob_rl/examples/interaction_fit/sampling.rs` owns the two feeding strata. It
rejects empty, unsorted, duplicate, out-of-range or all-row hard pools. Tests
check exactly 32 samples from each disjoint stratum. The legacy uniform path
preserves its random-number stream and reproduces the archived fits.

The trainer binds the pool and source evidence before fitting.
`scripts/verify_interaction_head.py` recomputes pool membership solely from
training rows, verifies that the complete prior plan is preserved except the
declared sampling fields, and checks both full prediction sequences and
aggregate counts. `scripts/verify_interaction_sampling.py` also verifies the
uniform rerun and reports training accuracy by stratum. Six in-memory mutation
checks reject reordered training seeds, forged training counts, swapped
predictions, forged gates, changed pool membership and a changed update budget.

All **34,176 development predictions** reproduce through independently loaded
portable models, with zero reported Burn/portable kind mismatches. Another
**49,872 training predictions** are independently replayed. The entire nested
parent and Move-utility bytes remain unchanged. Exact-zero output migration
passes on all **14,008** corpus rows.

Validation passes the sampling and partition tests, active residual model
comparison, two backend numerical regressions, all-target/all-feature RL
Clippy, formatting, the 66-constant schema registry and whitespace checks.
The new fits have native evidence only; previous WASM qualification does not
transfer to these weights. No ecological rollout or promotion is claimed.

Evidence: `training-output/interaction-sampling-2026-09-12-v1`, including the
preflight, immutable pool and plan, six fits, six uniform reproductions, source
and binary hashes, independent replay, diagnoses and verification results.
Recheck with `python3 -B scripts/verify_interaction_sampling.py` and the evidence
root as its argument.

## Next gate

The [direct-context follow-up](interaction-context-2026-09-13.md) has completed
the probe below: none of 27 training checkpoints meets the combined thresholds.
See that successor for the current next step.

The next step is a training-only separability and optimization probe. Compare
observation-only features with the existing cell-private recurrent state under
matched diagnostic capacity and optimization budgets. First establish whether
the available inputs can fit both objectives without the present tradeoff;
then require the unchanged development gates before extending a deployed
residual. Do not supply team identity, episode labels, global state or shared
memory as model inputs. Private state may help distinguish histories, but its
ability to generalize remains unproven. Avoid another blind collection or
sampling-dose sweep. Fresh development and actual deployed combat/feeding
outcomes remain downstream gates; self-play stays gated.
