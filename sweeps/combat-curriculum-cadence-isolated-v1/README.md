# Isolated combat-curriculum cadence holdout

This strict five-seed sweep fixes rollout retention, contact, and skirmish
episode horizons at 16,384, 16,384, and 32,768 canonical quanta in both arms.
In the schema-35 implementation, contact and skirmish limits were also shared
with held-out combat evaluation; feeding evaluation already used its separate
promotion horizon. Total on-food, adjacent-food, contact, and skirmish training
exposure is also identical. Only cycle length and the corresponding per-cycle
rollout windows differ.

| Seed | Four-cycle | Two-cycle | Four-cycle best | Two-cycle best |
| ---: | :---: | :---: | ---: | ---: |
| 45 | qualified | failed | update 16 | — |
| 46 | failed | failed | — | — |
| 47 | qualified | qualified | update 16 | update 16 |
| 48 | qualified | failed | update 16 | — |
| 49 | failed | qualified | — | update 32 |

Four-cycle qualified 3/5 and isolated two-cycle qualified 2/5. The two-cycle
schedule shifts which seeds discover and retain combat behavior but does not
increase reliability. In particular, seed 46 no longer qualifies when it uses
the common shorter horizon, proving that its qualification in the historical
bundle was a horizon effect. Seed 49 moves the other way and qualifies only
under two-cycle cadence, while seeds 45 and 48 are lost.

The failure frontiers still show temporal interference rather than inadequate
feeding. Two-cycle seed 45 has eight skirmish kills at update 16 while failing
retention, then passes retention with zero kills at update 32. Seeds 46 and 48
finish with passing feeding evidence and zero kills. The current four-cycle
schedule remains the better holdout choice, but its 3/5 rate is not strong
enough to justify large-world promotion.

The schema-35 configuration used the same contact/skirmish horizon fields for
training rollouts and held-out combat evaluation. Keeping them equal made this
comparison fair. Training artifact schema 36 subsequently separated those
concepts so future curriculum and gate experiments cannot be confounded this
way.
