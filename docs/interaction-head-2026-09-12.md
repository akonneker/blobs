# Combat score diagnosis and controlled head fit — 2026-09-12

**The inherited interaction action-kind head strongly prefers Move. A small
head-only fit learns some Attack choices but fails feeding retention in all
three matched runs. No candidate is promoted.** The subsequent
[local-context adapter study](interaction-slot-adapter-2026-09-12.md) improves
agreement but still fails the retention gate; it identifies slot-feature
information loss as the next representation problem.

## Routed-score diagnosis

`FrozenPolicy::forward_with_trace` exposes the selected observation context,
frozen hidden representation, base action-kind scores and adapter contributions.
Ordinary inference retains the original arithmetic and output format. A test
compares every output bit with and without tracing in all three contexts.
The read-only diagnostic returns each original complete Mind decision unchanged.

All seven frozen Minds repeat their original 24-episode combat evaluation
records exactly: 168 repeated episodes and 11,020 recorded decisions. These
use already exposed development seeds 1435500201/1435500202, not training data
or another independent cohort.

Routing is deterministic in this deployed runtime. Every legal occupied-target
opportunity routes to Interaction. For the parent's 1,122 such observations:

| Contribution to Move minus Attack | Mean | Range |
| --- | ---: | ---: |
| Interaction base head | 25.966 | 21.704–34.653 |
| Context adapter | −9.379 | −9.3795–−9.3787 |
| Slot adapter | 0 | 0–0 |
| Resulting raw score | 16.586 | 12.324–25.275 |

The maintained aggressive teacher chooses Attack on 1,078 of those inputs.
Attack is legal throughout that slice. These margins are below the runtime's
probability floor threshold; the observed suppression is already present in the
raw head scores. The diagnostic identifies a strong inherited score preference,
not a missing learned router. Anonymous occupied neighbors still do not expose
team identity or guarantee that a particular attack is tactically useful.

Evidence: `training-output/contact-action-trace-2026-09-12-v1`, with frozen
executable/source snapshots, plan, all rows and independently checked summary.
Recheck with `scripts/verify_contact_action_trace.py`.

## Predeclared controlled intervention

The frozen 1435400301 feeding treatment supplies ordinary live recurrent
prefixes. This is the previously WASM-qualified pair's treatment; the new fits
have no WASM qualification. Its Move utility remains byte-identical.

Before collection, the study fixes training seed **1435600101**, development
seeds **1435600201/1435600202**, and three batch-sampling seeds
**1435600301/1435600302/1435600303**. The audit inherits the supplemental
fresh-feeding ledger, checks disjointness and searches historical local records.
The resulting ledger has 61 training, 24 validation and two reserved confirmation
seeds. Confirmation **1434999901/1434999902 remains untouched**. Carry the
study's `seed-audit.json` into future work; its new development seeds are exposed.

Collection includes the ordinary contact/skirmish suite and all five sustained
feeding layouts, totaling 66 frozen-policy episodes. It records only Interaction
observations, with a predeclared cap of the first 2,048 per episode. These rows
retain actual cell-private recurrent state, observations, legal kind masks and
both labels. Labels are never executed during collection. Host episode keys
are diagnostic metadata, never model features.

| Partition | Combat rows | Feeding rows |
| --- | ---: | ---: |
| Training | 411 | 2,268 |
| Development 1435600201 | 290 | 2,489 |
| Development 1435600202 | 296 | 2,621 |

Only the existing 128→10 interaction base head is trained: **1,290 parameters**,
initialized exactly from the parent. Recurrent/trunk weights, both adapters,
other context heads, targets, effort, amounts, signals and feeding utility stay
fixed. The matched runs vary batch sampling, not initialization.

Each arm receives 1,024 AdamW updates, learning rate 0.005, no weight decay,
gradient norm clipping at 1, and batches of 64 combat plus 64 feeding rows.
Control labels retain the frozen kind everywhere. Treatment substitutes the
maintained aggressive teacher kind only on combat rows. It uses masked
cross-entropy and the same batch stream as its control. NdArray SIMD remains
disabled; numerical regressions pass.

The predeclared gate requires each arm on both development seeds to reach
95% combat label agreement, 99% feeding kind retention and zero portable/Burn
kind mismatches. Treatments also require 90% agreement on teacher-Attack rows.
This is an offline gate on frozen prefixes, not demonstrated combat competence.

## Result: all treatments fail

All three controls score 100% combat and feeding kind agreement. Treatments:

| Sampling run | Combat agreement, seed 201 / 202 | Teacher-Attack agreement | Feeding retention |
| --- | ---: | ---: | ---: |
| 301 | 65.17% / 66.89% | 65.71% / 67.70% | 56.17% / 51.24% |
| 302 | 65.17% / 69.26% | 65.71% / 70.10% | 55.12% / 50.44% |
| 303 | 65.17% / 67.23% | 65.71% / 68.04% | 55.12% / 51.05% |

All 34,176 validation choices across the six fits agree between Burn and the
portable runtime. A separate executable reloads every saved composite, verifies
all inherited bytes outside the declared head and the complete utility, and
recomputes every reported accuracy count through ordinary portable forward
inference. Failed fits are retained as diagnostic exports, without ecological
promotion or inherited WASM claims.

This bounded experiment rejects this head-only fit. It does not prove that the
frozen representation is incapable of separation or that another training
budget cannot work. The next controlled experiment should use the existing
interaction slot adapter's direct local-state features while preserving the
same parent and retention controls. If it passes, action targets, effort,
payloads and realized damage/kills still need deployed evaluation. Self-play
remains gated by functional feeding and combat qualification.

## Code and verification

`evaluate_feeding_mind` permits read-only instrumentation through the existing
feeding boundary and telemetry. The deployed-policy wrapper keeps its API and
execution contract. Mind reset errors fail before decisions. Two full archived
checkerboard feeding sentinel episodes match exactly after this refactor.

Evidence is under `training-output/interaction-head-2026-09-12-v1`: pre-run plan,
seed audit, collector/trainer/verifier binaries and source hashes, paired rows,
six full-precision exports, result records, independent portable predictions,
feeding sentinels and verification. `scripts/verify_interaction_head.py` checks
provenance, disjoint splits, saved predictions, counts and all gate decisions.

Validation includes 26 policy tests, 289 RL library tests with four ignored,
two backend numerical regressions, the added feeding reset-error test,
Clippy across all targets/features, formatting and the 65-constant schema registry.
