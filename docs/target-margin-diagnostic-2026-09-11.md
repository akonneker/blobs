# Target-residual margin diagnostic — 2026-09-11

The 256-update residual shifts every active correction toward its teacher target,
but the shifts are too small to change those choices. Amplifying this fitted
correction then damages as many or more previously correct choices. This rejects
simple gain amplification as a remedy for the failed dose sweep on these data.

Follow-up: the [2026-09-12 interaction probe](target-interaction-2026-09-12.md)
isolates direction scaling on a synthetic fixed-pair task and establishes a
separate candidate-set limitation of independent quantile-teacher scoring.

## Scope and validation

`blob_rl/examples/target_residual_diagnostics.rs` is a read-only CPU diagnostic.
It loads the saved schema-37 residual-only treatment and five correction corpora
through the shared hash-verifying loaders. Its baseline is the same trained model
with only the residual replaced by a zero-output residual. It reconstructs each
recorded (source seed, source cell) history, starting from its declared initial
memory, and encodes/decodes private memory after every decision. Both branches
produce bit-identical recurrent outputs throughout all 10,240 decisions.

All 3,880 teacher-Move rows reproduce the recorded policy's target with the zero
residual. The trained branch reproduces the earlier validation target count:
1,238 of 1,959, versus 1,239 with no residual. These are target-only comparisons
conditioned on teacher Move; they are not full action/effort/ecological metrics.
The example builds and passes Clippy; formatting and whitespace checks pass.
Production model, training and simulation semantics are unchanged.

Seeds 1433000101 and 1433000202 are the previously exposed training/development
validation seeds. This diagnostic neither uses final confirmation seeds nor
creates a promoted policy. It does not establish ecological performance.

## Observed margins

| Active correction measure | Training: 733 rows | Development validation: 720 rows |
| --- | ---: | ---: |
| Mean original policy-over-teacher gap | 1.542962 logits | 1.577497 logits |
| Mean residual teacher-over-policy shift | 0.117502 logits | 0.117973 logits |
| Active pairs shifted in the desired direction | 733 | 720 |
| Active targets actually corrected | 0 | 0 |
| Median gain needed to cross the original policy target alone | 12.75× | 12.89× |

Every active pair differs only in slot direction and/or distance (features 2, 3,
4). For 797 of the 1,453 pairs, only dx/dy differ. Mean slot-feature distance is
0.01583; dx/dy are encoded in units of 1/127. This establishes the narrow feature
seam, but does not by itself prove feature scaling is the sole cause.

Every corpus contains 2,048 distinct, nonzero private-random blocks. Randomness
was not accidentally zeroed. Rotating the 32 random features within each row,
holding recorded memory fixed and subtracting the baseline under the same
intervention, changes the validation residual pair shift by a median 0.0000439
and mean 0.0010669 logits. Only 217/720 active validation rows change by more than
0.0001 logits. This one intervention shows weak random-conditioned relative
scoring on these examples; it does not prove complete randomness independence.

## Offline gain sensitivity

Using retained legal-target logits, evaluate
`baseline + gain * (trained - baseline)` for each target. No model is edited or
trained; the complete fixed grid is reported, with no validation-selected policy.

| Gain | Training correct / 1,921 | Validation correct / 1,959 | Validation active corrections fixed | Previously correct validation choices lost |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,188 | 1,239 | 0 | 0 |
| 1 | 1,188 | 1,238 | 0 | 1 |
| 2 | 1,188 | 1,238 | 0 | 1 |
| 4 | 1,182 | 1,236 | 8 | 11 |
| 8 | 1,182 | 1,184 | 62 | 117 |
| 16 | 1,181 | 1,154 | 390 | 475 |
| 32 | 1,176 | 1,136 | 461 | 564 |
| 64 | 1,176 | 1,136 | 461 | 564 |

The results are consistent with a weakly randomness-conditioned directional
correction that cannot discriminate enough between conflicting target choices.
That is an inference from this diagnostic, not an architectural impossibility
proof. A next architecture proposal should first demonstrate effective private-
random/slot interaction on a controlled ranking test, preserve existing model
semantics through explicit migration, then use fresh paired development seeds.
Do not reopen the rejected dose sweep or run an ecological matrix from this result.

## Retained evidence

`training-output/target-margin-diagnostic-2026-09-11-v2` contains immutable copied
source/binary hashes, exact command, bounded runner/status/log, hash-bound input
identities, every Move row's margins and legal logits, `analyze.py`, and summaries
by seed and layout. Runtime: 3.45 seconds with four Rayon workers, under a
20-minute timeout. The earlier v1 directory preserves the first margin-only run.
