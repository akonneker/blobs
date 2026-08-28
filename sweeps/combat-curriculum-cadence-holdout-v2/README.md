# Coupled cadence/horizon holdout

This five-seed comparison repeats the historical two-cycle and four-cycle
balanced schedule bundles on seeds 45–49 using the same newly rebuilt,
independently qualified initial policy. Both arms have identical total stage
exposure over 524,288 quanta per environment. As in the original profiles,
two-cycle rollout stage-episode limits are twice the four-cycle limits.
Contact and skirmish limits also affected held-out combat evaluation in the
schema-35 implementation; feeding retention evaluation already had its own
independent horizon.

| Seed | Two-cycle bundle | Four-cycle bundle | Two-cycle best | Four-cycle best |
| ---: | :---: | :---: | ---: | ---: |
| 45 | failed | qualified | — | update 16 |
| 46 | qualified | failed | update 32 | — |
| 47 | qualified | qualified | update 32 | update 16 |
| 48 | failed | qualified | — | update 16 |
| 49 | failed | failed | — | — |

The two-cycle bundle qualified 2/5 and four-cycle qualified 3/5. The apparent
seed-46 rescue could not be assigned to cadence because two-cycle evaluation
also allowed twice as much canonical time per combat episode. This result is a
valid comparison of the two historical profile bundles, but not a clean
cadence experiment. The corrected isolated sweep fixes all three stage-episode
horizons across both arms.

The first publication attempt is retained separately as a pre-simulation
failure: its `/tmp` warm-start dependency had expired. This `v2` execution is
bound to rebuilt behavior-clone metadata SHA-256
`4244392609b76655df5a6884c8375494a8259ec8a19325f87871764f065eec45`
and passing feeding artifact hash
`3fb341d3d2b77f2f00fe7c5d54b5deb7b73490f660d0393872462dd5196809b8`.
A byte-identical local copy is retained outside version control at
`training-output/warmstarts/skirmish-schema35` so future contracts need not
depend on an ephemeral path.
