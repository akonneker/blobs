# Hard-example fitting and gradient checks — 2026-09-13

**The current residual cannot fit the frozen 128-row fixture at the declared
budget, but a standardized linear classifier using observations and actual
private memory separates all 128 rows.** Gradient checks pass through the
encoder and pooling path. The [standardization follow-up](interaction-standardized-2026-09-13.md)
now isolates that input transform and fits all 128 rows in all three runs.
Its next step addresses zero-history states before full-corpus training.

## Frozen fixture and neural fits

The fixture follows the [direct-context probe](interaction-context-2026-09-13.md).
It contains 64 feeding examples from the previously frozen 168-row training
error pool. Selection uses the first initialization seed and source row indices
in SHA256 order. Each feeding example is paired greedily with the nearest
unused teacher-Attack combat row, measured by squared distance on the complete
canonical nonrandom observation, with ties resolved by source index. Memory is
excluded from the distance. Every row is checked against the original training
corpus; development examples never enter selection or fitting.

There are no contradictory labels for identical complete observation, private
memory and legal-mask inputs in this fixture. Some cross-label observations
are identical when memory is omitted. The frozen labels are Attack for all
64 combat examples and Move for all 64 feeding examples.

The pre-run plan keeps the 7,514-parameter residual, frozen parent offsets,
AdamW at 0.005, zero weight decay and norm clipping at 1. Observation and actual
memory arms share initial weights for each of the three existing seeds. Every
update uses the same complete batch of 128 rows, with no sampling RNG.
Checkpoints at 128, 512, 2,048 and 4,096 updates are all retained. The success
criterion is exact 128/128 agreement, for this training diagnostic only.

At 4,096 updates, correct choices out of 64 per domain are:

| Initialization | Observation: combat / feeding | Actual memory: combat / feeding |
| --- | ---: | ---: |
| 301 | 41 / 64 | 48 / 64 |
| 302 | 50 / 56 | 44 / 64 |
| 303 | 60 / 50 | 47 / 64 |

None of the 24 checkpoints fits every row. Final full-batch cross-entropy is
0.247–0.333 for observation arms and 0.206–0.308 for memory arms. The loss does
not decrease monotonically at the retained checkpoints in every fit. Actual
memory reaches full feeding retention but still loses combat rows.

A separate label-coded score control correctly decodes all 128 requested
kinds through the inherited parent transform and legal mask. It bypasses
learning and validates label/decoder plumbing only; it is not a learned Mind.

## Gradients, activations and refactor checks

Directional finite differences agree with autodiff for each of the four
parameter blocks, including both slot encoders and pooled summaries, at batch
sizes **2 and 128**. The independent scalar loss uses central differences
along each block's normalized gradient. The checks exercise active heads and
memory, rather than the zero-output initialization where hidden gradients are
expected to vanish. They are bounded numerical checks on the tested inputs,
not a proof about every activation boundary or trained checkpoint.

The retained final models do not show wholesale saturation or a dead-ReLU
collapse. Present-slot first-layer tanh saturation at absolute output 0.99
ranges from 0.38% to 19.04%, and second-layer saturation from 0.85% to 23.68%.
Between 46.2% and 81.0% of final hidden ReLUs are active. Restored coordinate
magnitudes in this fixture are at most one tile; large raw coordinate values
are not the immediate explanation here. These measurements do not establish
that the optimization problem is well conditioned.

Scoring and scalar evaluation now share `context_fit/evaluation.rs`; the row
schema and bounded JSON reader are separated from whole-corpus loading in
`interaction_fit/row.rs`. Activation tracing preserves the original scalar
arithmetic. A separate invocation reproduces every one of the previous
context probe's **27 replay records and 224,424 predictions exactly**.

All **3,072 new checkpoint predictions** also reproduce from saved weights,
with zero scalar/Burn kind mismatches. The saved-weight pass recomputes the
activation statistics and checks full-batch loss against independent scalar
cross-entropy. Six semantic mutation checks reject altered fixture membership,
validation seeds inserted into training, a failed decoder control, unmatched
initialization, forged loss and forged counts.

## Supplemental linear separability control

After the neural fits failed, a separate supplemental plan fixed a ridge
coefficient of 0.001 and two direct binary classifiers. Both standardize their
inputs using this fixture alone. One uses the complete canonical nonrandom
observation; the other appends actual incoming private memory. Their targets
are +1 for Attack and -1 for Move. A double-precision dual ridge solve replaces
iterative neural fitting. There is no parameter search or development access.

| Inputs | Nonconstant features | Combat correct | Feeding correct | Total |
| --- | ---: | ---: | ---: | ---: |
| Observation | 69 | 54 / 64 | 50 / 64 | 104 / 128 |
| Observation + memory | 197 | 64 / 64 | 64 / 64 | 128 / 128 |

The memory classifier's minimum signed training margin is 0.779. Solver
residuals are below 1e-9. A separate verification pass reloads saved coefficients,
recomputes predictions and checks the ridge normal equations without refitting.

This establishes separability of this fixture using its recorded legal inputs.
It does **not** isolate the cause of the neural failure: the control changes
representation, normalization, solver and objective, and omits the frozen
parent score offset. It also does not show that these histories predict labels
on another seed, that a deployed Mind will generate the same prefixes, or that
binary imitation produces useful combat outcomes. The linear weights are
explicitly diagnostic and have no deployment qualification.

## Evidence and next gate

Evidence lives in `training-output/interaction-hard-2026-09-13-v1`: the frozen
fixture and plan, six fits, 24 checkpoints, saved-weight replay, activation
measurements, decoder control, source hashes, gradient-test log, prior-record
regression, mutation checks, and separately predeclared linear plan and results.
Recheck with `scripts/verify_interaction_hard.py`,
`scripts/test_interaction_hard.py`, and
`scripts/diagnose_hard_separability.py --verify`, each given the evidence root.

Model/partition tests, backend numerical regressions, finite-difference checks,
all-target/all-feature RL Clippy, formatting, the 66-constant schema registry
and whitespace checks pass. No runtime ABI or policy envelope changes. No new
simulation seed, development cohort, episode or promotion is claimed. Reserved
confirmation seeds 1434999901/1434999902 remain untouched.

The [standardization follow-up](interaction-standardized-2026-09-13.md) has
completed the following proposed comparison successfully on the fixture; see
that successor for the current next gate.

Next: compare raw versus standardized actual-memory features in the same
residual on this exact fixture, with matched initializations, full batches,
optimizer and budget. Freeze training-only means/scales before fitting and
retain every run. This isolates one conditioning change suggested by the
linear control; it does not assume that normalization is the sole cause.
If the residual can then fit the fixture, return to the full training corpus
before spending the unchanged development gates. Actual deployed combat and
feeding outcomes remain downstream requirements; self-play stays gated.
