# Specialist-distillation coefficient sensitivity

This strict three-seed sweep varies only the stage-local ecology and combat
distillation coefficients. Every run uses the partitioned teacher-inference
implementation, seeds 42/43/44, the same qualified initial policy, four-stage
cadence, fixed evaluation suite, and 524,288 simulation quanta per environment.

| Coefficient | Joint qualification | Mean actions/s | Mean quanta/s |
| ---: | ---: | ---: | ---: |
| 0.05 | 3/3 | 307.2 | 51,311.9 |
| 0.10 | 3/3 | 314.4 | 52,743.8 |
| 0.20 | 2/3 | 402.7 | 68,001.9 |

`0.05` is the lowest tested coefficient that preserves qualification across
all three seeds. Seeds 42 and 43 qualified at update 16 under every coefficient,
before specialist choice could distinguish the later trajectories. On the
diagnostic seed 44, `0.05` qualified at update 32 with 100% on-food survival,
88.5% adjacent-food survival, eight skirmish kills, and 1,384 skirmish damage.
The existing `0.10` setting qualified one update later with 100% / 86.5%, eight
kills, and 1,216 damage.

The `0.20` seed-44 run reproduced the two-specialist interference pattern but
did not combine the competencies: update 16 retained 16 skirmish kills while
failing feeding retention at 81.2% / 69.8%; update 32 passed feeding retention
at 100% / 85.4% but had zero skirmish kills. This indicates that a stronger
functional target is not monotonically safer and can overconstrain the
student's reconciliation of the two specialists.

The throughput numbers are diagnostic only. All coefficients execute the same
network topology and number of teacher rows, runs were ordered rather than
interleaved, and the large `0.20` increase is therefore wall-clock noise or
machine-load variation rather than an expected coefficient-dependent speedup.
`aggregate.json` was regenerated from the immutable per-run records after all
nine runs completed.
