# Per-slot interaction residual — 2026-09-12

**The per-slot residual is implemented, zero-migration checks pass, and the
first matched pair passes native/WASM deployment checks. All three trained
treatments still fail the development retention gate, so none is promoted.**
They pass the offline thresholds on training rows but lose feeding retention
on development rows. The [fixed-architecture coverage follow-up](interaction-coverage-2026-09-12.md)
now adds two training seeds: combat stabilizes, but feeding retention still
fails on both training and development rows. See that successor for the next gate.

## Representation and deployment

The [max-summary study](interaction-slot-adapter-2026-09-12.md) found that
component-wise maxima erase which food, occupant, activity and geometry
features belong to the same slot. The new residual encodes each complete
33-feature slot through shared 16-unit tanh layers before pooling. It restores
dx/dy to tile units, masks absent slots, and concatenates a sorted sum divided
by 32 and a component-wise maximum. Those 32 summaries and 38 nonrandom header
features feed a 32-unit ReLU layer and a 10-kind output head: **3,418 new
parameters**. The output head starts at exactly zero.

Only Interaction observations receive the resulting kind-score adjustment.
It is added before the inherited probability transform and legal-mask decoder,
so target, effort, amount and signal choices still use the existing conditional
heads for the resulting kind. The learned Move utility still samples legal
targets after signal-cost reservation. Every original parent and utility byte
remains unchanged. Changed kinds can change subsequent trajectories; frozen
weights alone do not establish behavioral retention.

The explicit `BLCMR001` envelope contains the exact `BLCMP001` bytes plus the
residual weights. Its contract is
`blob.policy.interaction-slot-encoder-f32-libm-move-q48.v1`. Older greedy and
Move-utility envelopes keep their existing contracts and behavior. Decoders
reject bad lengths, unknown envelopes and nonfinite parameters. The envelope
has its own registered version constant; the registry now checks 66 constants.
No Mind ABI, physics or termination semantics change.

An exactly zero adjustment skips addition, preserving the original score bits.
All output vectors match on **8,375** recorded observations. Zero-residual
replays reproduce all **24 combat episodes and two sustained-feeding episodes**
from the archive. The feeding trial differs only in its intentional execution
contract metadata; complete episode records and promotion metrics match.
These are repeated development episodes, not a new biological cohort.

## Matched fit and results

The pre-run plan reuses the same frozen 1435400301 feeding treatment, training
seed 1435600101, development seeds 1435600201/202, paired labels, masks and real
private recurrent memories. No development examples enter training. The seed
ledger remains 61 training, 24 validation and two reserved confirmation seeds;
1434999901/1434999902 are untouched. Deployment fixture seeds 72101/72102 were
already exposed in the inherited ledger.

Each paired run initializes its new hidden layers from seed 1435600301, 302 or
303, with a zero output head, and uses that seed's identical batch stream in
both arms. Each fit has 1,024 AdamW updates, learning rate 0.005, zero weight
decay, norm clipping at 1 and 128-row batches balanced 64 combat/64 feeding.
Controls retain the frozen kind everywhere. Treatments use the maintained
aggressive teacher kind only on combat rows. NdArray SIMD stays disabled.

The gate is unchanged: each arm must achieve 95% combat-label agreement and
99% feeding-kind retention on both development seeds, with 90% teacher-Attack
agreement for treatments and zero portable/Burn kind mismatches. All controls
score 100% combat and feeding agreement. Treatments:

| Initialization | Combat agreement, seed 201 / 202 | Teacher-Attack agreement | Feeding retention |
| --- | ---: | ---: | ---: |
| 301 | 90.34% / 83.78% | 91.79% / 84.19% | 94.94% / 98.59% |
| 302 | 98.28% / 97.97% | 99.64% / 99.31% | 94.70% / 98.28% |
| 303 | 95.17% / 92.23% | 96.79% / 93.13% | 94.90% / 98.32% |

All three treatments reach **97.32% combat agreement and 99.07–99.12% feeding
retention on the training rows**, passing the offline thresholds there. This
is a training/development generalization gap in the measured cohort, not proof
that more data alone will solve it. Model capacity, coverage and optimization
remain possible contributors. A fixed-architecture coverage experiment is the
next controlled intervention; do not select initialization 302 from these
outcomes and call it qualified.

All **34,176 validation choices** match between Burn and the portable runtime.
A separate executable reloads every fitted envelope, checks the entire nested
parent/utility bytes, reproduces every reported validation count and separately
replays **16,074 training-row predictions**. It never trains or modifies labels.

## Bounded WASM qualification

The first predeclared control/treatment pair is built through the ordinary
learned-Mind plugin and qualified regardless of its learning result:

| Mind | Native/WASM decisions | ABI fixture decisions | Signed replay cases |
| --- | ---: | ---: | ---: |
| 301 control | 2,418 | 528 | 10 |
| 301 treatment | 2,620 | 528 | 10 |

There are zero complete decision or memory mismatches. Each artifact also
passes 32 stock-Extism comparisons, 32 cell-isolation checks, malformed-input
rejection and replay checks at worker counts 1 and 4. Move-target-only
inherited-retention checks are explicitly marked inapplicable to this new
kind-changing format. These are bounded deployment fixtures, not deterministic
fuel or gameplay qualification. **The other four fits have native evidence
only.** No trained candidate proceeds to fresh-seed ecology or promotion.

## Code, evidence and next gate

The portable implementation lives in `blob_policy/src/relational.rs`;
`runtime.rs` shares output decoding and adds the pre-transform residual hook.
The composite runtime shares its signal-preserving Move utility application.
Training models and verified corpus loading have separate shared modules under
`blob_rl/examples/relational_fit` and `interaction_fit`.

Evidence: `training-output/interaction-relational-2026-09-12-v1` contains the
pre-run plan, source/executable digests, inherited ledger, zero export, six
trained exports, independent predictions, zero-migration trials and two WASM
build/qualification bundles. Recheck with `scripts/verify_relational.py`.

Validation includes 30 policy tests, 290 RL library tests (four ignored), two
backend numerical regressions, three fit-model tests, and Clippy across policy,
RL and game targets/features. Tests cover exact-zero outputs and complete
ordinary decisions across contexts and slot counts, active residual isolation,
slot permutation, row independence, excluded private draws, malformed envelopes,
and distinguishing food/occupant pairings that share identical raw maxima.

The [coverage follow-up](interaction-coverage-2026-09-12.md) has now completed
the additional-seed collection and matched fixed-residual fits. All treatments
still fail feeding retention; the next study targets training-only sampling. Keep the current
development rows out of training. Require all-run retention before spending a
new development cohort, and actual deployed combat activity/damage/kills plus
feeding retention before self-play.
