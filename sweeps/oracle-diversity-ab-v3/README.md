# Three-context diversified-oracle warm-start A/B V3

This paired seed-42 experiment repeats the normalized V2 control and
three-oracle treatment with the schema-21 foraging/interaction/exploration
architecture. Both arms retain 32 epochs, 14,806 presentations per epoch,
473,792 total action-loss mass, 1,952 optimizer steps, the same dataset weights,
and the same held-out feeding, contact, and micro-combat seeds.

## Result

The architecture is rejected. Each routed expert learns its assigned labels,
but the learned soft gate fails to select the foraging expert. The control's
held-out Consume accuracy is 99.6% inside the assigned foraging expert but only
1.9% after gating; the treatment is 99.0% versus 20.2%. Both policies make zero
on-food consumes and lose every feeding cell.

| held-out metric | three-context control | diversified treatment |
| --- | ---: | ---: |
| total action-loss mass | 473,792.006 | 473,791.991 |
| optimizer steps | 1,952 | 1,952 |
| foraging gate accuracy | 2.8% | 22.5% |
| interaction gate accuracy | 100% | 98.4% |
| exploration gate accuracy | 100% | 100% |
| Consume assigned-expert accuracy | 99.6% | 99.0% |
| Consume end-to-end accuracy | 1.9% | 20.2% |
| Move end-to-end accuracy | 99.6% | 99.3% |
| Attack end-to-end accuracy | 93.8% | 88.6% |
| on-food success / survival | 0/8 / 0% | 0/8 / 0% |
| adjacent-food success / survival | 8/8 / 0% | 0/8 / 0% |
| contact/skirmish attacks | 952 | 308 |
| contact/skirmish damage | 2,012 | 6,400 |
| contact/skirmish kills | 44 | 72 |
| micro survival objectives | 32/32 | 0/32 |
| micro elimination objectives | 0/16 | 0/16 |

The control's adjacent-food success is not competence: only 120 consumes occur
across 2,048 initial cells, intake is 0.938 energy per initial cell, and every
cell dies. The diversified treatment retains high-damage attack behavior but
loses the V2 treatment's 8/8 local two-versus-one elimination result.

## Diagnosis and decision

The deterministic corpus router itself is coherent: its prior audit found all
three contexts populated with zero contradictory route labels. The failure is
the separately learned soft gate. Its branch is trained only by the balanced
auxiliary context loss, while action-kind experts receive the correct hard
supervision. Increasing dataset mixture weights cannot repair this separation:
the desired Consume classifier already exists and is simply not selected.

The authoritative follow-up is recorded in
[`oracle-diversity-ab-v4`](../oracle-diversity-ab-v4/README.md). It makes the
audited observation predicate the action-kind router at inference and training;
the learned gate remains diagnostic-only. End-to-end and assigned-expert
accuracy then agree exactly, but rollout feeding still fails because the clones
leave the teacher-trajectory state distribution. The next correction must
therefore target states reached by the learned policy rather than adjust the
gate or static mixture again.

## Bound evidence

- Control behavior clone: `08685f8239a414c2841a3e0f608da4923d2c614b5a5150ebbf1b7a5551e36627`
- Treatment behavior clone: `750c28110a538fdc02adde10e42b6c0e7645fcfb9f89090b351b7cd3f3441233`
- Control model: `6d95302ab5f651ee557d36ba8c3fcfeec29ef0914eb6dfa6680bac085b79702f`
- Treatment model: `95c54e844515dc05f8ed75a34611ad8acf8341f18ab0ea210ee0da072cd43c95`
- Control feeding: `9ceead497d7b9745cd68655292418011790101746e50dc9d89207ef00aafcea0`
- Treatment feeding: `07847b91cc800072b69a84c478d0663af19989ae7402db46941fee93b30115ae`
- Control contact: `1e5e724d208cb3aa8825f1d24d143fb16c0661fd0b79cd03309c629505138b21`
- Treatment contact: `07458301baa02f2eb985ca6d83202da53d44c3a41b4e0d79f415e7167c773293`
- Control micro-combat: `ab113fb7bd291b3966b121a3b94b21d5ae3f23310e56ea721581eddeae67bb64`
- Treatment micro-combat: `cb7c21f027a103c5d974afe5d39c999bc3ac0e1ca1013f0634033932ad001ddc`

Artifacts are retained under
`training-output/warmstarts/combat-256-three-context-control-v3` and
`training-output/warmstarts/combat-256-three-context-diverse-oracles-v3`.
