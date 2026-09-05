# Feeding DAgger round 4 V1

This slice follows the selected round-three policy onto a longer, two-seed
on-policy frontier. The 16,384-decision correction corpus contains 340
teacher-Guard/policy-Move kind errors, 554 teacher-Consume/policy-Move kind
errors, and 734 target-only errors. The policy agrees exactly on 90.1% of the
corpus. Teacher target labels are strongly directional: the first eight target
counts are 5,772, 1,560, 1,308, 292, 254, 722, 566, and 562.

## What the diagnostics ruled out

The overlap auditor finds 3,755 exact observable states with no conflicting
action-family or target labels. Projecting to the eight original raw foraging
features produces 2,405 states and only three family-conflicting states (20
samples). Thus exact Mind-visible aliasing is not the explanation for this
frontier, although the narrow legacy projection does discard useful context.

Target-query-only training with square-root or full target balancing reduced
Move exact accuracy instead of correcting the directional errors. These arms
are rejected. A target residual prototype also failed to improve exact
agreement and was removed rather than expanding the runtime model without
evidence.

The existing foraging residual remains useful. Sixteen adapter-only epochs at
`1e-2` improve held-out Consume action-kind accuracy from 88.9% to 98.9% while
preserving 100% Move-kind accuracy and 93.3% Move exact accuracy. Training the
complete foraging head reaches only 97.9% Consume accuracy, so the residual is
the selected kind correction.

The interaction head exposes a different seam. An unconstrained linear-head
dose can learn 9.4% of Guard labels, but Move accuracy falls to 95.7%; capped
or weaker doses select no Guards. This is a representation boundary rather
than a reason to accept a broad regression.

## Local context residuals

Policy schema 28 adds separate zero-initialized interaction and exploration
action-kind residuals. Each is a 16-unit local MLP over the frozen recurrent
state plus the complete non-random observation header. It cannot see another
cell's memory, stable identity, or private randomness, and it does not change
the Mind ABI. Foraging keeps its existing compact residual. The new paths add
5,684 parameters to the large profile (about 1.7%) and are intentionally
isolated by `--context-adapter-only interaction|exploration` training.

A two-dose interaction treatment reduces the mean Guard-to-Move wrong-kind
margin from 19.9215 to 17.1222 logits on a fresh 8,192-decision probe. It also
reduces Consume-to-Move errors to seven at a 1.0102-logit mean margin, but no
Guard crosses the argmax boundary. The retained test proves that one context
adapter step changes only the selected residual; action targets, memory, and
all other model families remain byte-for-byte fixed.

## Directional rollout result

The combined foraging-plus-interaction candidate improves both tested seeds.
On-food survival is 38.7% on each seed with 156.699 consumed energy per initial
cell. Adjacent-food survival is 62.5% and 61.3%, with 189.676 and 191.570
consumed energy per initial cell. Round three achieved 33.6% on-food survival
and 58.2--59.0% adjacent survival, with 144.652 and 183.289--185.992 intake.

The direction is favorable, but neither survival result reaches the 80%
promotion gate and the interaction treatment still selects no Guard actions.
This candidate is therefore evidence, not a promoted warm start. Combat
retention is deliberately deferred. The next feeding slice should optimize the
measured wrong-kind margin directly or collect a narrower Guard boundary
curriculum, then require fresh-seed argmax crossover without Move regression
before paying the cost of full rollout and combat qualification.

## Bound artifacts

| artifact | SHA-256 |
|---|---|
| two-seed correction manifest | `3f63cdaee21ecfbcb9ef87bbdfdf4d570433ea82efc3fb1f3b9f8f0ff28ea562` |
| two-seed correction payload | `b146e771a36286ad5c646150c1379e114a7451102a0efeadc89f157a355cfd4b` |
| foraging residual metadata | `b8d68428abb0b95ff9bd1755f2d52df2bd5cc887e287f8754fcb931ea6a16e13` |
| foraging residual model | `1ea2bf796504ef479fc48474dedb88b0c2ed2620edbe3be2975e2ace0b8b28c3` |
| interaction residual metadata | `2542974d3f524d435b3dc428169b613f17356b3988417b98560f7f04800842a4` |
| interaction residual model | `901624bc1d20c5fd9e272c305485ac38951db6951735a93486c43978ecb7d2b4` |
| fresh margin-probe manifest | `d38c927f73bba9dc05d577e7006455f933ed709e4090157fffd96c04405acade` |
| fresh margin-probe payload | `c06b4248b1e9698e4aad28ed4a264c70acb2ce459294e2644dd3d0408090c4c9` |

Local immutable evidence is retained under
`training-output/feeding-dagger-round4-v1`.
