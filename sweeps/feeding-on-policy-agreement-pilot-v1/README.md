# Feeding on-policy agreement pilot V1

This stage-6 pilot labels the exact zero-randomness observations reached by the
two frozen authoritative-router V4 policies with the memoryless
`collision_aware_forager`. Seeds `950000101` and `950000202` are disjoint from
all prior demonstration, held-out evaluation, and recovery-pilot seeds. Each
stage admits at most 4,096 complete-prefix decisions; early policy termination
may produce fewer.

## Result

| policy and stage | samples | exact full-choice agreement | largest teacher → policy error |
|---|---:|---:|---|
| control, on-food | 4,096 | 1,536 (37.5%) | Consume → Attack: 2,560 |
| control, adjacent-food | 4,096 | 1,243 (30.3%) | Consume → Attack: 1,326 |
| diversified, on-food | 3,584 | 512 (14.3%) | Consume → Attack: 3,072 |
| diversified, adjacent-food | 3,286 | 25 (0.8%) | Consume → Attack: 1,445 |

The control's adjacent-food errors also include 645 Move → Consume and 882
Move → Attack choices. The diversified adjacent policy adds 1,016 Move →
Attack, 137 Consume → Move, and 151 Wait → Move errors. Its 25 agreements are
all Wait choices; it emits no Consume decisions in either feeding stage.

These are exact-input comparisons, not teacher-trajectory validation. The
frozen policy and teacher see the same anonymous observation, private recurrent
memory, action mask, and zero random block at every labeled state. Only the
policy controls the trajectory. The evidence therefore localizes the observed
failure to on-policy choice disagreement rather than expert routing, physical
impossibility, or an observation mismatch.

The published payloads are directly usable as DAgger-style behavior-cloning
inputs. The next experiment must compare a fixed correction share against an
equal-presentation ordinary-rehearsal control, retain the same optimizer steps,
and evaluate on a new seed partition. Because the treatment terminates early,
the paired training design must equalize presentations explicitly rather than
mistaking its smaller raw dataset for less weight.

## Bound artifacts

| artifact | manifest SHA-256 | payload SHA-256 |
|---|---|---|
| control on-food | `c3d4b8c4655b59d35868c3dba4fe57cc966c3d3f8654466fcc50ee180c2f27ba` | `2bdf648b948fce2ea28f76fce4c4c0e2a4105a4911c44f3566da6950e149897d` |
| control adjacent-food | `42e9cdf9521da15d16da65e84a8bc7cca4e958615fbbbbe204733170aba316e7` | `2bd57a1bb1aef03329debc0e7bae99dbbae12aa59e23ede989c64abe151f3a69` |
| diversified on-food | `21e20ab60f226ee72bed4dbf954d61360571f38abf41535ab9b7ed11912e8389` | `31ae97b7f34ca17e03b2763b4d82f0670c17d58c47ab534d65c174d74fff08fc` |
| diversified adjacent-food | `12bdd2c452c369e28af38030e896695fd720909fb184780ef1c60765ce70d2db` | `0c1bd6fc7d7d476253ada4f36369f602e8de1257cf0c059ca3c6677002245ee0` |

The on-food scenario hash is
`da9d227b5d8d0773e6cb6c0507523abec0884740d5b3b53729ca251588c3e8a6`;
the adjacent-food scenario hash is
`0768331af2f625114c393f81ee895312f1d7904d272f2c8656b62fc12f49b0ee`.
Local immutable datasets are retained beside each V4 behavior clone under
`corrections-pilot-on-food` and `corrections-pilot-adjacent-food`.
