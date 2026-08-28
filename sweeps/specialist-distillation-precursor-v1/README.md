# Damage-precursor combat teacher

This paired five-seed experiment tests whether a high-damage checkpoint can
serve as a temporary combat teacher before any checkpoint passes the discrete
combat gate. Both arms use coefficient `0.05`, seeds 45–49, the same qualified
initial policy, four-stage cadence, evaluation suite, and 524,288 simulation
quanta per environment. The precursor arm alone permits a frontier checkpoint
with at least one unit of resolver-attributed skirmish damage when no qualified
combat teacher exists. A qualified teacher always takes precedence.

| Seed | Qualified-only | Damage precursor | Qualified-only best | Precursor best |
| ---: | :---: | :---: | ---: | ---: |
| 45 | qualified | qualified | update 16 | update 16 |
| 46 | failed | failed | — | — |
| 47 | qualified | qualified | update 32 | update 32 |
| 48 | qualified | qualified | update 16 | update 16 |
| 49 | failed | failed | — | — |

Both arms qualified 3/5, with identical best competency evidence on every
successful seed. Seeds 45, 47, 48, and 49 already had qualified combat teachers
whenever a combat teacher was needed, so qualified-first selection made their
scientific outcome unchanged.

Seed 46 exercised the new path. Its update-16 checkpoint dealt 1,464 skirmish
damage but scored zero kills, and was explicitly activated as a precursor for
later combat stages. The fallback did not create a kill. At update 32 it raised
adjacent-food survival from 87.5% to 88.5%, but skirmish damage fell from 1,480
to 728. At the terminal update, survival rose from 88.5% to 89.6% while damage
fell from 848 to 376. Functional imitation of a damaging-but-zero-kill policy
therefore reinforced the wrong side of the combat boundary rather than serving
as a useful stepping stone.

The option remains disabled by default and available only through the explicit
`combat_precursor_min_skirmish_damage` threshold. This experiment rejects
enabling it in maintained training profiles or spending additional sweeps on
the threshold: every positive threshold at or below 1,464 selects the same
seed-46 teacher, while a higher threshold reduces to the qualified-only arm.

Mean wall-clock rates were 280.0 actions/s for qualified-only and 305.5 for the
precursor arm, but an anomalously slow first control run dominates that ordered
comparison. Throughput is recorded diagnostically and is not a scientific
effect of the selector.
