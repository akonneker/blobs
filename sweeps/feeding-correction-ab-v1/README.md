# Feeding correction versus rehearsal A/B V1

This is the first causal retraining test after the stage-5 recovery and stage-6
exact-input agreement evidence. Both arms continue from the V4 hard-context
control behavior clone with seed 42 for one epoch at `1e-5`. Each receives
exactly 14,806 sample presentations, 64 optimizer updates, the same action-loss
mass, recurrent-unroll limit, action balancing, and authoritative context
router. The control uses only the original 12-dataset rehearsal mixture. The
treatment reallocates 2,116 presentations (14.3%) to the on-food and
adjacent-food policy-correction datasets.

The evaluation seeds
`960000101,960000202,960000303,960000404,960000505,960000606,960000707,960000808`
are disjoint from training and correction collection. A separate exact-input
probe uses seed `970000101`; the schema-15 margin probe uses `990000101`.

## Result

The bounded correction dose is ineffective. Although the two model files have
different hashes, every measured greedy behavior is identical.

| metric | rehearsal control | correction treatment |
|---|---:|---:|
| sample presentations | 14,806 | 14,806 |
| optimizer updates | 64 | 64 |
| action-loss mass | 14,806.0002 | 14,805.9995 |
| on-food exact agreement | 37.5% | 37.5% |
| adjacent-food exact agreement | 30.0% | 30.0% |
| on-food intake / initial cell | 48.000 | 48.000 |
| adjacent-food intake / initial cell | 23.242 | 23.242 |
| on-food survival | 0% | 0% |
| adjacent-food survival | 0% | 0% |

Both post-training policies still make all 1,280 observed `Consume -> Attack`
errors in the on-food probe. In adjacent-food they each make 650
`Consume -> Attack`, 468 `Move -> Attack`, and 315 `Move -> Consume` errors.
The treatment did not cross a single greedy action boundary on the paired
probe.

This does not refute correction training. It rejects this particular dose:
one epoch at `1e-5` with a 14.3% correction share.

The follow-up schema-15 margin probe distinguishes two regimes:

| teacher → policy error | rehearsal mean advantage | correction mean advantage |
|---|---:|---:|
| on-food Consume → Attack | 0.2971 | 0.2783 |
| adjacent Consume → Attack | 0.4501 | 0.4322 |
| adjacent Move → Consume | 12.1511 | 11.9910 |
| adjacent Move → Attack | 10.7939 | 10.6349 |

The weak correction dose moves every margin in the intended direction but does
not cross a decision boundary. Consume errors are plausibly dose-limited: all
are within 0.5 logits, and 256/1,280 corrected on-food errors are within 0.05.
The movement errors are confidently wrong by roughly 11--12 logits and should
not be lumped into the same escalation. The next experiment should use a small
paired Consume-focused exposure grid while movement receives a separate
representation and dataset analysis.

An initial implementation dry run was discarded from scientific interpretation
because recurrent chunk composition produced 61 control versus 63 treatment
optimizer updates. Behavior-cloning schema 23 adds an exact
`optimizer_steps_per_epoch` contract with fail-closed feasibility validation.
The results above are the corrected rerun with 64 updates in both arms.

## Bound artifacts

| artifact | rehearsal control | correction treatment |
|---|---|---|
| behavior-clone metadata | `fdcb7d583361e85b7175b2cd5aed0b3ad651e9b1414ce020da2bc7eb1c3eafaa` | `13df661533ab863c90a4a634a00dbcf9fb9c4ede4f97d04de2d71b3c2a02b330` |
| model | `f5f8101542cd511dcdd654959166fd72ce8a073cc44123c68caf66cad04d9038` | `18486f745a2df82269331c015d5ba4521a8649746a036b7c19015406b260e116` |
| feeding evaluation | `4da9fae794e28893be597d26bcd0f6aa4cd53272a62bed5568cfe30292c961b5` | `7665364277764a9ff5976c9c6fe64d5306e173a43029660325d46aaa542444b0` |
| on-food agreement manifest | `18e1930281ae272af6c6056bc327a67d6934e211160df3bdb3128de89019fb44` | `b6adc05b0598df2480434b7aa903c76f9f89979202f08ee43ecd7abf64fa17c5` |
| adjacent agreement manifest | `29ba1c6e4bd47f90eb79b22257414c822b5984a44b31451a68fe4a77382652a6` | `55de54c72175924b84e63e4bf624ed931342fa87a0700f860769982e14e674ae` |
| schema-15 on-food margin probe | `c8a0101e3f3c11fee7573a4a600f43fdcd6c9ec5b3b7f71f7ad752c8aac37dc2` | `b395c33047fc4166092b02b2896585b4abafef9884e65cc02d3c9697a4a353d2` |
| schema-15 adjacent margin probe | `faf07e7c3894e49f19bcdaa326373171f3bc74d84d79f980e7204f7b13ea5083` | `0f9198be586a61a4798e52d5dbfd26fd597dda7105ef7553f0d3dbefc629d101` |

Local artifacts are retained under
`training-output/feeding-correction-ab-v2-control-parent`. The reusable paired
runner is `scripts/run_feeding_correction_ab.sh`.
