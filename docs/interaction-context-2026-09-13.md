# Direct private-context diagnostic — 2026-09-13

**Direct access to existing private recurrent memory does not resolve the
training tradeoff in this probe. None of nine fits, at any of 27 checkpoints,
meets both combat and feeding thresholds.** No development corpus was loaded,
no model was promoted, and no deployment format changed. The
[hard-example follow-up](interaction-hard-2026-09-13.md) now finds that a linear
observation-plus-memory control separates the fixture while the residual
still fails; the next intervention isolates feature conditioning.

## Question and controls

The [sampling experiment](interaction-sampling-2026-09-12.md) improved feeding
retention by losing combat agreement, including on training rows. This probe
asks whether exposing incoming private state directly to the residual improves
fitting, and whether more updates resolve the failure.

The predeclared model retains the two 16-unit slot encoders, masked sorted
mean/32 and max, and 32-unit ReLU head. A 128-value context block is concatenated
with the 38 nonrandom header features and 32 slot summaries. Each arm has
**7,514 nominal parameters**, with the output head initialized to zero:

- **Observation:** the residual's context block is zero.
- **Memory:** it receives that row's actual incoming private recurrent memory.
- **Shuffled:** it receives a fixed training-row donor's memory.

All arms keep the frozen parent's scores, which already depend on actual
private state. Observation therefore means no *direct residual* context; it
is not a memoryless full policy. Its zero context also removes active input
degrees of freedom despite the equal parameter count. Real versus shuffled
context is the comparison with equally active context dimensions. The shuffled
arm is an offline control, not an executable legal Mind: donor indices and
other rows' private state must never become runtime inputs.

The donor permutation uses only the existing first initialization seed and
row indices, with every fixed point removed. It never consults labels or
development data. Row order is all training combat rows followed by all
training feeding rows, with each domain preserving the established seed and
layout order. The verifier recomputes the complete permutation.

All three initializations receive identical initial weight bytes across arms.
Each batch has 64 uniform combat and 64 uniform feeding examples. Batch-stream
digests agree across arms at every checkpoint. Labels remain combat teacher
kinds and frozen feeding kinds. AdamW uses learning rate 0.005, zero weight
decay and norm clipping at 1, with checkpoints at 512, 1,024 and 2,048 updates.
The budget and checkpoints were declared before fitting; every result is kept.
NdArray SIMD remains disabled.

The corpus remains **890 combat and 7,422 feeding rows** from the three existing
training seeds. The ledger stays at 63 training, 24 validation and two reserved
confirmation seeds. No new seeds, collection episodes or development
predictions are introduced. Confirmation seeds 1434999901/1434999902 remain
untouched.

## Training results

Final 2,048-update training agreement:

| Initialization | Observation: combat / feeding | Actual memory: combat / feeding | Shuffled: combat / feeding |
| --- | ---: | ---: | ---: |
| 301 | 97.64% / 98.13% | 97.64% / 97.87% | 38.31% / 99.54% |
| 302 | 97.64% / 98.10% | 97.98% / 97.84% | 97.98% / 97.74% |
| 303 | 97.87% / 97.84% | 97.75% / 98.13% | 97.98% / 97.84% |

The diagnostic thresholds are the existing 95% combat, 90% teacher-Attack and
99% feeding agreement thresholds, applied only to training rows here. None
of the 27 checkpoints meets all three. Actual memory gives no consistent
improvement across initializations. Doubling the previous update budget also
fails to reach joint retention. The first shuffled run trades away combat
for feeding; it is retained as evidence of sensitivity to the context inputs,
not selected as a useful model.

This does not prove that private memory is irrelevant or that the task is
unlearnable. It establishes failure for this architecture, optimizer, feature
scales, corpus and declared budgets. No claim about generalization follows
from these training-only results.

## Input diagnosis

Incoming memory is zero on 2,146 of 8,312 training rows, but **none of the 168
previously identified hard feeding rows has zero memory**. No memory channel
is constant. Across the complete corpus, channel standard deviations range
from 0.380 to 0.731 (median approximately 0.437); 24.42% of components have
absolute value at least 0.999. These statistics do not support an explanation
based simply on missing, constant or uniformly tiny memory values. They do
not rule out saturation or poor conditioning inside the learned encoder.

The six groups with identical complete nonrandom observations and conflicting
labels have cross-label maximum memory differences of approximately
0.045–0.186. The histories are numerically distinguishable, but that fact alone
does not show that the residual can exploit the differences or generalize.

## Verification and organization

The diagnostic has a separate example, `context_separability_probe.rs`, and
separate training/scalar modules under `blob_rl/examples/context_fit`. It writes
raw diagnostic weight files, not deployed policy manifests or envelopes.
The scalar implementation uses explicit f32 linear loops and host `f32::tanh`;
the inherited parent retains its existing scalar probability transform. This
is a diagnostic numerical path, not a newly qualified portable contract.

All **224,424 checkpoint predictions** have zero scalar/Burn kind mismatches.
A separate saved-weight invocation reloads all 27 checkpoints and reproduces
every ordered prediction and count. Source and binary hashes, training-file
hashes, donor indices, initializations and sampling streams are bound to the
pre-run plan. The audit verifies all counts and thresholds independently.
Six in-memory mutation checks reject changed donors, validation seeds inserted
into training, unmatched batches, unmatched initialization, forged counts and
a missing checkpoint.

Model tests cover active scalar/batched agreement, row independence, slot
permutation, excluded fresh random bytes, exact-zero output and malformed
weights/context. Partition tests, both backend numerical regressions,
all-target/all-feature RL Clippy, formatting, the 66-constant schema registry
and whitespace checks pass.

Evidence: `training-output/interaction-context-2026-09-13-v1`. Recompute the
main audit with `scripts/verify_interaction_context.py`, the memory statistics
with `scripts/diagnose_interaction_context.py`, and mutation checks with
`scripts/test_interaction_context.py`, each taking the evidence root.
`memory-diagnosis-final.json` is the reproducible extended statistics report.

## Next gate

The [hard-example follow-up](interaction-hard-2026-09-13.md) has completed the
fixture, gradient checks and a supplemental linear separability control below.
See that successor for the current next step.

Build a small deterministic training-only fixture containing hard feeding
examples and nearby combat examples. Test whether the residual can deliberately
fit that fixture, with a known-correct label/decoder sanity control. Inspect
feature scales, encoder activation saturation and gradients, including finite
difference checks through the slot pooling path. This should distinguish a
basic fitting or conditioning failure from a larger-corpus limitation before
another model expansion or collection run. Freeze the fixture and budget
before evaluating it; keep development data out of selection and training.

Only a successful training intervention should proceed to the unchanged
all-run development gates and then ordinary deployed combat and feeding
outcomes. Existing private memory remains a legal candidate input, but this
probe gives no basis to expand the deployed format for it yet. Self-play
remains gated.
