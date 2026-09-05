# Diversified-oracle warm-start A/B V2

This is the exact schema-20 rerun of
[`oracle-diversity-ab-v1`](../oracle-diversity-ab-v1/README.md). Both arms use
seed 42, visible-neighbor routing, 32 epochs, the same architecture and
optimizer, and exactly 14,806 presentations per epoch (473,792 total). The
only intentional change from V1 is normalization of post-cap action-family
weights to a mean of one over the presented labels.

## Result

Normalization removes the V1 loss-scale confound but does not rescue the
diversified treatment. Total action-loss mass now equals the common
presentation budget in both arms (within floating-point accumulation error),
and both execute 1,952 optimizer steps. The treatment still erases Consume and
dies in every feeding and asymmetric-survival rollout.

The combat transfer is real rather than a wholly failed model. The treatment
attacks in every contact/skirmish episode, deals 5.4 times the control's
damage, and solves the local two-versus-one elimination objective on all eight
held-out seeds. It does so suicidally: it fails every one of the 32 survival
episodes and the one-versus-one defender-breach objective.

| held-out metric | visible-routing control | diversified treatment |
| --- | ---: | ---: |
| total action-loss mass | 473,792.006 | 473,791.991 |
| optimizer steps | 1,952 | 1,952 |
| Consume action-kind accuracy | 98.0% | 1.1% |
| Move action-kind accuracy | 99.6% | 99.3% |
| Attack action-kind accuracy | 65.6% | 68.2% |
| phase-gate accuracy | 100% | 100% |
| on-food success / survival | 8/8 / 100% | 0/8 / 0% |
| adjacent-food success / survival | 8/8 / 46.5% | 0/8 / 0% |
| contact/skirmish attacks | 1,380 | 588 |
| contact/skirmish damage | 1,304 | 7,028 |
| contact/skirmish kills | 24 | 48 |
| micro survival objectives | 32/32 | 0/32 |
| micro elimination objectives | 0/16 | 8/16 |

The control exactly reproduces its V1 supervised and rollout metrics, which
also checks deterministic reproducibility across the schema change. It remains
unfit for promotion because adjacent-food survival is only 46.5% and neither
elimination objective succeeds.

## Decision

The three-oracle treatment is rejected as a general warm start, but its local
elimination capability is retained as evidence for the next architecture.
Another scalar mixture-weight adjustment is not the highest-value experiment:
equal presentation count, optimizer work, total loss mass, aggregate feeding
shares, and phase-gate accuracy are already controlled.

The current observable router has only `isolated` and `visible-neighbor`
experts. Social feeding and combat therefore still share the same recurrent
expert whenever another cell is visible. The replacement three-way architecture
and corpus audit are recorded in
[`foraging-interaction-exploration-router-v1`](../foraging-interaction-exploration-router-v1/README.md).
It uses only Mind-visible anonymous evidence and gives visible attack/guard
activity precedence over current food, then separates other current-food,
neighbor, and exploration observations. Its 17,128-sample audit finds all
three contexts populated with zero contradictory route states. The subsequent
A/B should reuse this exact V2 mixture, seed, presentation budget, and held-out
suites.

## Bound evidence

- Control behavior clone: `9b9b8aa90930a0e8f3395719d3315213162232e15e07d66bbe4a12cd94f083f6`
- Treatment behavior clone: `a0f1f4e91cd496a3cd168f1d4ac08ce20db4f5181d2fb30910c659b7d159e955`
- Control model: `a53941ebdbffaa1d91e81ecce12f6812f30a5fd6752d47e0c2ed6cba12106b8c`
- Treatment model: `ee70aeb01428ca43d47f4cef784a24b781430696ff1c43b931c28454d827c505`
- Control feeding: `4b31b700232e689a8154645c48daf20f024f0eefd8187fc353511f7c3f14b7c3`
- Treatment feeding: `37f0cc262fbfaefe360ac8f43574e4eaa09e45ed9342961e1f8ff73fee2ae557`
- Control contact: `d715e757758b5657dc0db873f8c40482fa144c565102815e7e90247ba1272069`
- Treatment contact: `4a865135fb82b83d60561663f0294a049348533c44d69ec1b259201d99de42be`
- Control micro-combat: `17c85015fb8cae5be3c639c4c9003f81da2bf77115c8f1b7cc8c89f0de6eeaa4`
- Treatment micro-combat: `c1463f63ed81a4039002458a6137bb05ab8496ad81aea867521096677c34cbf6`

Artifacts are retained under
`training-output/warmstarts/combat-256-visible-routing-control-v2` and
`training-output/warmstarts/combat-256-diverse-oracles-v2`.
