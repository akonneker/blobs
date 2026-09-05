# Diversified-oracle warm-start A/B V1

This paired seed-42 experiment compares the maintained twelve-dataset control
with the three-oracle diversity treatment. Both arms use visible-neighbor
routing, 32 epochs, the same architecture and optimizer, and exactly 14,806
presentations per epoch (473,792 total). Feeding retains ten of fifteen mixture
shares. The control gives five shares to two maintained contact datasets; the
treatment gives one share each to those datasets and the three admitted oracle
shards.

## Result

The diversified treatment is rejected. It learns a much more force-efficient
attack policy, but completely erases Consume and dies in every feeding and
micro-combat rollout. It does not solve either elimination objective.

| held-out metric | visible-routing control | diversified treatment |
| --- | ---: | ---: |
| Consume action-kind accuracy | 98.0% | 1.5% |
| Move action-kind accuracy | 99.6% | 99.3% |
| Attack action-kind accuracy | 65.6% | 54.5% |
| phase-gate accuracy | 100% | 100% |
| on-food success / survival | 8/8 / 100% | 0/8 / 0% |
| adjacent-food success / survival | 8/8 / 46.5% | 0/8 / 0% |
| contact/skirmish attacks | 1,380 | 284 |
| contact/skirmish damage | 1,304 | 6,480 |
| contact/skirmish kills | 24 | 72 |
| micro survival objectives | 32/32 | 0/32 |
| micro elimination objectives | 0/16 | 0/16 |

The control is itself not promotion-ready: although every feeding episode
registers success and all on-food cells survive, only 46.5% of adjacent-food
cells survive. It does establish that visible-neighbor routing can retain
social Consume and improves the asymmetric survival suite to 32/32. Neither
arm demonstrates elimination competence.

## Weight-normalization confound

The fixed presentation budget exposed a remaining optimizer confound. Family
weights are capped at a 4x ratio but were not renormalized afterward. The
control therefore received 473,792 total action-loss mass, while the treatment
received only 331,278. Its rare Wait family and changed family distribution
reduced Consume mass from 157,931 to 87,843 and also made the phase-gate loss
relatively stronger. Presentation count and optimizer steps are identical, but
effective action-loss scale is not.

Behavior-cloning schema 20 fixes this by renormalizing post-cap weights to a
mean of one over the actual epoch presentations. Future paired arms therefore
have action-loss mass equal to their common presentation budget while retaining
the configured relative cap. This V1 result remains a decisive rejection of
the produced treatment model, but is not the final causal estimate of oracle
diversification. A normalized V2 rerun is required before changing the mixture
again.

## Bound evidence

- Control behavior clone: `24248528267d82d1f93fea31d497f107e8ef5ebc1cf32c5a87a53137d79759a3`
- Treatment behavior clone: `bd07502e75606c0dfcd5d6132c6c75b9ecd9e735aafa1ec7458d11d349022c4f`
- Control feeding: `67710305f70a25d61db86ced935a36d642c9d807cd7b790d36e0adac879ac0e2`
- Treatment feeding: `4b8289db74a9c898ef54284efec521900a357ce665c1923347475995ec669424`
- Control contact: `da16a8e3bdcdbfe58efaaeb6a5d20723997dd5c054eb3b2f47c8661e11a27faa`
- Treatment contact: `bcd4daa39465ba841462c92afc28e82c2d6ba3204499d234d87999742d67b60b`
- Control micro-combat: `a4be2c411519f3bc14a928942a116541ad9db66f720743b60b28cc9fa049efdd`
- Treatment micro-combat: `3e3b03fae00a6477acffc04f42eec2c2b8a7a4b1ea87d04c87aa9900ca77ee10`

Artifacts are retained under
`training-output/warmstarts/combat-256-visible-routing-control-v1` and
`training-output/warmstarts/combat-256-diverse-oracles-v1`.
