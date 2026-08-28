# Combat curriculum cadence sweep v1

This immutable paired experiment compares two-cycle balanced rehearsal,
four-cycle balanced rehearsal, and four-cycle skirmish-heavy rehearsal at
training seeds 42, 43, and 44. All runs share the same behavior clone,
qualification evidence, physics, reward, PPO configuration, evaluation seeds,
and 524,288-quanta-per-environment budget. `manifest.json` binds the scientific
plan; `execution-contract.json` binds the exact trainer and warm-start CLI
arguments; `aggregate.json` contains generic paired terminal metrics.

| Schedule | Joint-qualified seeds | Mean skirmish damage | Mean skirmish kills | Mean independent survival (on / adjacent) |
|---|---:|---:|---:|---:|
| two-cycle balanced | 3/3 | 1,269 | 13.3 | 92.7% / 83.7% |
| four-cycle balanced | 2/3 | 1,364* | 12.0* | 91.7% / 83.3%* |
| four-cycle skirmish-heavy | 3/3 | 1,389 | 10.7 | 97.6% / 88.9% |

`*` Four-cycle balanced means are conditional on its two qualifying seeds and
must not be read as a three-seed policy mean. Seed 44 demonstrated temporal
interference: update 16 had 16 skirmish kills but failed adjacent-food retention
at 69.8%; update 32 restored retention to 83.3% but had zero skirmish kills.

Every qualifying policy scored 16 wins, 0 losses, and eight timeouts in the
fixed full-match suite. The sweep therefore does not justify changing the
maintained schedule or adding blind PPO exposure. It motivates a constrained
multi-skill retention or Pareto-archive slice.
