# Feeding DAgger round 2 V1

This slice starts from the selected 20% first-round Consume-correction clone
and tests the next on-policy frontier: leave the tile after the starting plant
is depleted. Four fresh collection seeds (`1020000101` through `1020000404`)
produce 8,192 exact-input decisions. The first 4,096 are already-correct
Consume decisions and the following 4,096 are `Move` labels where the parent
continues to Consume. The latter error begins at a mean 11.131-logit policy
advantage.

Evaluation seeds are disjoint from correction collection and margin probes.
No candidate in this slice is promoted.

## Ordinary rehearsal and dose response

The normalized 20% pair uses two epochs at `5e-5`, 29,612 total sample
presentations, and 128 optimizer updates in each arm. Both ordinary rehearsal
and the correction treatment regress to zero survival: on-food intake is 57.0
and adjacent-food intake is 23.930 energy per initial cell. The treatment does
move the post-depletion deficit from 11.950 to 9.668 logits, but also creates a
new `Consume -> Attack` deficit of 0.641 logits.

Increasing the correction share confirms a monotonic but inefficient response.

| correction share | post-depletion Move deficit | on-food survival | adjacent survival | adjacent intake |
|---:|---:|---:|---:|---:|
| 0% | 11.950 | 0% | 0% | 23.930 |
| 20% | 9.668 | 0% | 0% | 23.930 |
| 33.3% | 8.411 | 0% | 0% | 26.120 |
| 50% | about 7.5 | 0% | 6.4% | 73.333 |

This rejects further blind extrapolation of the mixed-rehearsal dose curve.
Ordinary rehearsal is not neutral at this parent: it pushes the already narrow
Consume boundary back toward Attack.

## Stage-local head adaptation

Behavior-cloning schema 24 adds optional `--action-kind-expert-only`. After
each optimizer step, only the selected observation-routed action-kind head is
retained; the recurrent trunk, the other two action-kind experts, the phase
gate, target/effort/amount/signal heads, and value head remain exactly at the
verified parent. The mode requires `--initial-behavior-clone` and changes no
Mind input, ABI, physics, or runtime routing.

The first attempted specialization exposed an unstated routing assumption.
Post-depletion states still contain local loose or diffuse energy and therefore
remain in the **foraging** context; they are not exploration states. An
exploration-head A/B is consequently an exact no-op with respect to the added
correction corpus. The foraging-head pair does learn, but ordinary rehearsal
still regresses both arms to zero survival. Its treatment reduces the Move
deficit from 11.343 to 10.268 logits while changing the new Consume-to-Attack
deficit from 0.098 to 0.247.

## Correction-only boundary

Removing unrelated rehearsal and adapting only the foraging action-kind head
shows that the targets are not intrinsically contradictory. Eight epochs keep
all prefix Consume kinds correct and reduce the Move deficit from 11.131 to
0.930 logits. One more epoch crosses the Move boundary, but too broadly:
25% of held-out Consume kinds become Move, rollout intake falls from 249 to 48,
and survival remains zero. The cells move extensively but leave before the
plant is depleted.

The sharp transition localizes the remaining failure to within-foraging state
discrimination. The current frozen representation plus a linear action-kind
head does not expose enough consumability/depletion information to separate
late Consume from post-depletion Move at a useful boundary. The next model
slice should give the foraging head an observation-local residual/adapter over
raw plant, loose, diffuse, gut-capacity, and action-legality features, while
keeping the same Mind-visible inputs and authoritative routing. The exact
round-two correction trajectory is the regression test: it must retain the
complete Consume prefix, select the teacher's Move suffix, and improve fresh
rollout survival before combat retention is rerun.

## Bound artifacts

| artifact | SHA-256 |
|---|---|
| correction manifest | `6fef5d3f55a61e2ca727c68e445a1170a7047cec22d7d7f40c070cf2fc08d617` |
| correction payload | `4854145c237251bee35b6c4342a86503d14831a3540a2b9a9b57c12cb763cecf` |
| normalized control metadata | `a30463e0ed9416803a06a8daf9b30d271c899a5ff8b9954d0ff9c6370fefe553` |
| normalized treatment metadata | `01f23bcadc728b85120b5e6a4fa63784091ba7539d42284e06d5c31c849bcb08` |
| 50% dose metadata | `020d902ddcb3a4229840ea5e01c91d88e36b236beee66a96f384fac8fe7203fb` |
| foraging-head control metadata | `229febf80a4dea86d1b9a88d429cedd7255ec78599ec3fdd701a99321a9a0104` |
| foraging-head treatment metadata | `f28dfc2de81923f801f7317ba45ce3807459a1211e99fbcb5930e23dcfd2d5b7` |
| correction-only eight-epoch metadata | `630edd38c4cab31d6f757761079b29a4e9c9aed16bb09fe81d613179e88c5173` |
| correction-only crossing metadata | `ec909bb784e475ed9e880b9ce9bfd7b881010fcbe7a813163b0fea0b28f22119` |

Local artifacts are retained under `training-output/feeding-dagger-round2-*`.
The paired runners accept `BLOB_FEEDING_ACTION_KIND_EXPERT_ONLY` and
`BLOB_CONSUME_ACTION_KIND_EXPERT_ONLY` for reproducible stage-local runs.
