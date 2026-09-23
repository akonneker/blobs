# Standardized memory on the hard fixture — 2026-09-13

**Standardizing the residual's actual-memory input resolves the measured
hard-fixture fitting failure: all three runs reach 128/128 correct choices
by 512 updates and retain that result at 2,048 and 4,096 updates.** Raw-memory
controls reproduce every previous checkpoint and weight byte. This is a
training-only result; full-corpus fitting and deployment remain unqualified.

## Isolated intervention

The [hard-example study](interaction-hard-2026-09-13.md) found that a standardized
linear classifier could separate this fixture, while the residual failed.
This follow-up changes only the residual's memory input transform. It retains
the exact 64 combat and 64 feeding rows, labels, 7,514-parameter architecture,
initializations 301/302/303, full batches, frozen parent scores, AdamW at 0.005,
zero weight decay, norm clipping at 1 and 4,096-update budget. Checkpoints
remain 128, 512, 2,048 and 4,096; every run is retained.

Population means and standard deviations are computed from the frozen fixture
alone, in f64, then rounded once to f32. The residual receives `(memory-mean)/scale`
with f32 subtraction and division. Constant channels would use scale 1; this
fixture has none. There is no clipping. The original parent always receives
the unmodified incoming private memory, and its recorded scores remain fixed.

The fixture's channel standard deviations range from **0.0000488 to 0.476**,
with median **0.00272**. The previous full-corpus memory statistics were much
broader, partly because they included zero-history states. Those aggregate
statistics did not describe the variation among these hard examples.

The plan, fixture, normalizer, source study and ledger are hash-bound before
fitting. The independent audit recomputes all means, scales and transformed
f32 context bytes. Raw and standardized arms have identical initial weights.
All 12 raw-memory checkpoint records and weight files reproduce the preceding
study exactly, including predictions, loss and activation statistics.

## Results

| Initialization | Raw final combat, feeding (each of 64) | Standardized at 512 | Standardized final | Final standardized loss |
| --- | ---: | ---: | ---: | ---: |
| 301 | 48, 64 | 128 / 128 | 128 / 128 | 0.0001161 |
| 302 | 44, 64 | 128 / 128 | 128 / 128 | 0.0000829 |
| 303 | 47, 64 | 128 / 128 | 128 / 128 | 0.0000773 |

At 128 updates, standardized runs score 100, 103 and 99 of 128. Each then
reaches exact fitting at the first retained 512-update checkpoint. Because
architecture, labels, parent offsets, initialization, optimizer and batches
are matched, this experiment shows that this conditioning change is sufficient
to remove the failure on this fixture. It does not establish that normalization
alone will produce useful combat behavior on new trajectories.

All **3,072 checkpoint predictions** reproduce from saved weights, with zero
scalar/Burn kind mismatches. The independent scalar pass also recomputes
activation statistics and full-batch loss. The label-coded decoder sanity
control remains 128/128; it bypasses learning and is not a candidate Mind.

## Full-training range audit

A read-only audit applies the frozen transform to the existing 8,312 training
rows; it does not fit models or load development corpora.

| Input cohort | Rows | Largest absolute standardized component |
| --- | ---: | ---: |
| Hard fixture | 128 | 11.27 |
| All training rows | 8,312 | 20,505.51 |
| Training rows with nonzero memory | 6,166 | 28.19 |

The fixture has no zero-memory rows. All **2,146** zero-memory states in the
larger corpus exceed absolute context value 1,024 after centering; no nonzero
memory row exceeds 64. This is a concrete transfer issue: the fixture-derived
means encode an established recurrent state, so centering an initial zero
state creates extreme values. The descriptive range buckets are not new
qualification gates.

These measurements support explicit handling of zero history before returning
to full-corpus fitting. They do not justify silently clipping inputs or changing
the current retained experiment after seeing its results.

## Code and evidence

`context_fit/normalization.rs` validates the frozen means/scales and performs
the transform without mutating incoming memory. Shared diagnostic scoring and
scalar evaluation accept an optional residual-context override; parent inference
continues to use the original row memory. The trainer records separate hashes
for original memory and standardized residual context, both independently
reconstructed by Python.

Tests cover exact transformation, no clipping, constant-channel behavior,
input immutability and rejection of bad dimensions, scales and nonfinite
values. Existing model, partition, backend numerical and finite-difference
checks pass, including gradients at batch sizes 2 and 128. All-target/all-feature
RL Clippy, formatting, the 66-constant schema registry and whitespace checks
pass. Six semantic mutation checks reject changed means, scales, residual
context, parent memory, standardized counts and update budget.

Evidence: `training-output/interaction-standardized-2026-09-13-v1` contains the
pre-run plan, frozen normalizer and fixture, six fits, 24 checkpoints, source
and executable hashes, scalar replay, raw-control regression, independent
context hashes, mutation checks and transfer-range audit. Recheck with
`scripts/verify_interaction_standardized.py`,
`scripts/test_interaction_standardized.py` and
`scripts/diagnose_standardized_transfer.py`, each taking the evidence root.

No new seeds, simulation episodes, development predictions, runtime contract
or deployed artifact are introduced. Reserved confirmation seeds
1434999901/1434999902 remain untouched. These diagnostic weights are unpromoted.

## Next gate

Update 2026-09-23: the [full-corpus successor](interaction-zero-state-2026-09-23.md)
has completed the protocol below. Zero-state range handling passes, but all three
normalized fits fail the joint training gate. No development or deployment
evaluation was spent; environment calibration and action value are next.

Predeclare a zero-state-preserving transform: zero incoming memory produces a
zero residual-context block; nonzero memory receives the frozen standardization.
Compare raw and transformed inputs on the full training corpus, preserving
actual parent memory and the previous full-corpus study's architecture,
initializations, labels, sampling, optimizer and budget. Audit the transformed
ranges before fitting and retain all runs. This tests transfer beyond the
small fixture while addressing the measured initial-state problem.

Only after joint training retention passes should the unchanged all-run
development gates be spent, followed by ordinary deployed combat and feeding
evaluation. Self-play remains downstream of those qualifications.
