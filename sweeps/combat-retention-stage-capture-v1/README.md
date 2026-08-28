# World-time combat-retention stage capture

This paired diagnostic revisits seeds 47–49 from the specialist-distillation
holdout. Both arms use the same qualified warm start, four-stage curriculum,
PPO settings, held-out suites, and 524,288 simulation-quanta budget per
environment. A complete competency evaluation and immutable checkpoint are
now triggered after the contact and skirmish portions of every curriculum
cycle. Only coefficient-0.05 stage-local specialist distillation differs.

Each run published eight scheduled measurements plus the terminal measurement,
compared with the much coarser update-16/update-32 view used by the earlier
holdout. Evaluation happens at the first PPO boundary whose minimum
per-environment clock crosses the requested frontier, so bounded rollout
overshoot remains explicit in `competency-timeline.csv`.

| Seed | Control qualifying updates | Distillation qualifying updates | Control best | Distillation best | Terminal combat |
| ---: | :--- | :--- | ---: | ---: | :--- |
| 47 | 12, 13 | 12, 13, 21, 28 | 13 | 13 | neither |
| 48 | 4, 5, 27, 28 | 4, 5 | 28 | 4 | neither |
| 49 | 5, 12 | 5, 32 | 12 | 32 | distillation only |

The finer schedule found at least one jointly feeding- and combat-qualified
checkpoint in all 3/3 control runs and all 3/3 distillation runs. In
particular, control seed 49—classified as a failure by the former sparse
schedule—contained qualified policies around 102,144 and 214,016 minimum
simulation quanta. Checkpoint timing was therefore a real source of false
negative run classifications.

Distillation did not improve the probability of ever finding a qualified
policy and did not consistently retain combat. Each arm produced eight jointly
qualified measurements across its 27 total measurements. Distillation helped
seed 47 reacquire kills in later cycles and made seed 49 combat-qualified at
the terminal boundary, but it prevented the late reacquisition seen in control
seed 48. Terminal combat retention moved from 0/3 to 1/3, which is too small
and inconsistent to promote coefficient 0.05 into the maintained default.

The practical result is to keep world-time stage capture and continue storing
the best immutable qualified frontier, while leaving specialist distillation
opt-in. The next retention experiment should rehearse fixed qualified combat
examples during *all* intervening PPO stages, or enforce a bounded functional
regression constraint after updates. Stage-local rehearsal only repairs a
skill when combat data is already present and does not reliably prevent other
stages from overwriting it.

Mean sampled throughput was 116.5 actions/s for control and 119.5 actions/s for
distillation. The paired difference was +3.0 actions/s with a 95% Student-t
half-width of 25.3 actions/s. These serial wall-clock measurements are noise,
not evidence of a performance improvement.
