# Local-context combat adapter — 2026-09-12

**Training the existing local-context adapter substantially improves the failed
base-head fit, but all three treatments still miss the retention gate.** Attack
label agreement reaches 95.5–97.5%; feeding kind retention reaches 94.7–98.2%.
No fit advances to live combat or WASM qualification. Error analysis identifies
information lost by the adapter's component-wise slot maximum. The subsequent
[per-slot residual study](interaction-relational-2026-09-12.md) implements that
representation change and passes deployment checks, but still fails development
retention; the next experiment expands training coverage.

## Controlled comparison

The [previous head study](interaction-head-2026-09-12.md) supplies the same
frozen 1435400301 feeding treatment, paired observations, legal masks, actual
private recurrent memories, labels, training seed 1435600101 and development
seeds 1435600201/1435600202. These are reused development observations, not a
new independent cohort. No additional seeds are consumed. Its seed ledger
remains 61 training, 24 validation and two reserved confirmation seeds;
1434999901/1434999902 remain untouched.

A plan written before fitting changes only the trained parameter block:
existing `interaction_slot_adapter_fc` and `interaction_slot_adapter_head`,
71→16 ReLU→10, **1,322 parameters**. The input is 38 nonrandom header features
plus the component-wise maximum of 33 raw slot features. The existing input
layer has nonzero weights and the output head is exactly zero initially.
Both layers initialize exactly from the parent. The interaction base head,
other adapters, trunk, recurrent state computation, targeting, effort, amounts,
signals and complete learned feeding utility remain fixed.

The three matched batch streams 1435600301/302/303 use the same 1,024 updates,
128-row batches (64 combat, 64 feeding), learning rate 0.005, AdamW without
weight decay, norm clipping at 1, and masked cross-entropy as the previous
study. Controls retain parent kinds everywhere; treatments substitute the
maintained aggressive teacher kind on combat rows only. There is no best-run
selection, changed label rule, longer training budget or relaxed gate.

The gate remains 95% combat label agreement, 90% teacher-Attack agreement for
treatments, 99% feeding kind retention, and zero portable/Burn kind mismatches
on each development seed in every run. This evaluates action kinds on frozen
prefixes, not realized attacks or survival under a changed policy.

## Results

All three controls retain 100% of combat and feeding kinds. Treatments:

| Batch stream | Combat agreement, seed 201 / 202 | Teacher-Attack agreement | Feeding retention |
| --- | ---: | ---: | ---: |
| 301 | 95.52% / 94.59% | 97.14% / 95.53% | 94.74% / 98.17% |
| 302 | 95.86% / 94.59% | 97.50% / 95.53% | 94.78% / 98.25% |
| 303 | 95.52% / 94.93% | 97.14% / 95.88% | 94.70% / 98.09% |

All six fits produce zero portable/Burn disagreements across **34,176**
validation choices. A separate executable reloads the saved composites and
reproduces every accuracy count. Independent byte checks verify that only the
declared two layers changed, including bit-exact preservation of the entire
feeding utility. The six complete exports remain unpromoted diagnostic artifacts.

## Residual error diagnosis

Every remaining feeding error is a false Attack choice. Across both development
seeds, runs 301/302/303 make 179/176/182 such errors. The largest common slices
are seed 201's line layout (60 errors per run), checkerboard (52–53) and ring
(18). Random-layout examples have no errors in this small corpus.

Across all **8,375** training/development rows, the 71-dimensional local input
has **19 identical-feature groups with conflicting kind labels**, containing
94 rows. Any deterministic function of *only* those local features must miss
at least 36 labels. Each treatment's feeding failures include **50 rows whose
local features exactly match a training combat Attack example**.

Using the complete nonrandom header and canonically sorted complete slot
vectors yields **zero identical-input label conflicts** in this corpus.
Sorting removes slot-order identity while retaining each slot's geometry and
associated food, occupant and activity features. Component-wise maxima erase
those associations: the maximum food value and a visible occupant need not
belong to the same slot.

These are feature-level findings, not a proof that the complete deployed Mind
cannot meet the gate. Its frozen base/context offsets and private recurrent
state also affect scores; the local-feature lower bound excludes them. The
absence of exact full-input conflicts also does not prove that a particular
model will generalize. Error analysis uses exposed development rows only and
does not relabel or train on them.

The next experiment should preserve per-slot feature relationships in a learned
interaction action-kind residual, with permutation invariance, an exactly zero
initial effect and frozen inherited parameters. A new deployment representation
needs explicit format/execution identity and native/WASM fidelity checks. Keep
the matched feeding retention gate, then require new-seed evaluation and actual
combat damage/kills before promotion or self-play. Repeating this unchanged
max-summary fit is not the next roadmap task.

## Implementation and evidence

The existing fit runner now shares `interaction_fit/model.rs` between base-head
and slot-adapter modes. Observation loading, matched sampling, objectives,
portable evaluation, export and gates remain common. The independent Rust
verifier derives the allowed parameter range from layer shapes. The Python
verifier supports hash-bound reuse of an audited source corpus and does not
count reused episodes as new evidence.

Evidence: `training-output/interaction-slot-adapter-2026-09-12-v1` contains the
pre-run plan, inherited seed audit, exact trainer/verifier binaries and sources,
six exports, saved predictions, recomputed verification and error diagnosis.
The original 66 collection episodes and two feeding sentinels are referenced
from the preceding study; no new ecological episodes were run here.

Validation includes feature permutation/private-draw isolation and zero-head
roundtrip tests, backend numerical regressions, Clippy across all targets and
features, formatting and the 65-constant schema registry. A numerical refactor
regression repeats all six original base-head fits and requires every complete
result record and saved weight hash to match the archive exactly.
