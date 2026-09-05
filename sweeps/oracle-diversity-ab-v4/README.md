# Authoritative-context diversified-oracle A/B V4

This paired seed-42 experiment replaces V3's learned soft expert selection
with the audited anonymous observation predicate. The learned gate remains
diagnostic-only. Both arms are freshly trained as behavior-cloning schema 22;
no V3 model is reinterpreted under changed inference semantics. Dataset weights,
32 epochs, 14,806 presentations per epoch, total action-loss mass, optimizer,
and every held-out rollout seed remain fixed.

Training-artifact schema 44 and behavior-cloning schema 22 fail closed on the
older soft-routing inference contract even though the recorded tensor shape is
unchanged.

## Result

Authoritative routing fixes the gate defect but does not make either warm start
promotion-ready. End-to-end action-kind accuracy now exactly equals the
assigned expert's accuracy even when the diagnostic soft gate is wrong.

| held-out metric | hard-context control | diversified treatment |
| --- | ---: | ---: |
| total action-loss mass | 473,792.006 | 473,791.991 |
| optimizer steps | 1,952 | 1,952 |
| Consume expert / end-to-end accuracy | 99.6% / 99.6% | 99.0% / 99.0% |
| diagnostic foraging-gate accuracy | 2.8% | 22.5% |
| Move end-to-end accuracy | 99.6% | 99.1% |
| Attack end-to-end accuracy | 56.2% | 68.2% |
| on-food consumes | 6,144 | 0 |
| on-food intake / initial cell | 48.000 | 0.000 |
| on-food survival | 0% | 0% |
| adjacent-food consumes | 2,856 | 0 |
| adjacent-food intake / initial cell | 22.312 | 0.000 |
| adjacent-food survival | 0% | 0% |
| contact/skirmish damage | 2,256 | 6,400 |
| contact/skirmish kills | 32 | 72 |
| micro survival objectives | 32/32 | 0/32 |
| micro elimination objectives | 0/16 | 0/16 |

The control demonstrates that the selector works: V3 made zero on-food
consumes, while V4 makes 6,144 under otherwise identical learned supervision.
It nevertheless stops after roughly three consumes per initial cell and every
cell dies. The treatment makes no feeding consumes despite 99.0% held-out
Consume accuracy. Its contact and micro-combat behavior is effectively the
same suicidal high-damage policy as V3.

## Diagnosis and decision

The remaining failure is rollout covariate shift, not expert selection or
aggregate loss scale. Seed-held-out labels are evaluated on teacher-generated
trajectories. Once the learned policy changes its gut, energy, activity,
private recurrent memory, or location differently from the teacher, it enters
states absent or underrepresented in that validation partition. The control's
three-consume plateau and the treatment's zero-consume fresh-seed behavior are
direct evidence that static accuracy is not a sufficient retention gate.

Authoritative routing should remain: it is deterministic, Mind-visible, and
repairs a measured failure at negligible runtime and parameter cost. The next
bounded slice should add feeding DAgger-style corrections: run the frozen clone
on disjoint seeds, have the maintained collision-aware forager label the exact
states the clone actually reaches, bind both clone and teacher provenance into
an immutable dataset, and retrain with a fixed correction share. Admission
must require improved on-policy action agreement plus held-out feeding survival,
not merely teacher-trajectory validation accuracy.

The correction collector must preserve cell-private recurrent histories and
must not expose teacher decisions, scenario identity, team identity, or
host-only state to the learned Mind. A paired control should spend the same
presentations on ordinary rehearsal so the correction benefit is not confused
with extra optimizer work.

## Bound evidence

- Control behavior clone: `23639b55983ae83f6c85a5c9a6c519cbcacde36c22fed5904364535332759a7f`
- Treatment behavior clone: `0c9c86dfc033723fb88801a41d4ef9124b428041b9a8252d2e9e97451fe2a4dd`
- Control model: `745fbd98dc7e31e96fc39fd1b583e0852fb77744511032e47a63ba3cd72c4a93`
- Treatment model: `1accb7fdc86767c371d54d251c58820aa2ea300395370c01690adf3f4db64285`
- Control feeding: `0734fc6e4508d9957ae0468ab5c43b6e6f693ba890406e5d4c34a46ff887ba40`
- Treatment feeding: `8064b933017a8ad3954f7fa982f3d21df9cde1effbfa5aea5912553f2bbeb790`
- Control contact: `74b0b5e95c2c2c2ae8c993723f6f1780ad9bb57a0af1be85c96c4198d1ef4981`
- Treatment contact: `94a5c1021350c4ffee40dfac42a2ab02445e26340d645491f0e45bb31c9b7627`
- Control micro-combat: `94b914ac27a9f28948eb9a3a5d2fcb4b3615d6ac353f42c23dfb83b24c9b9290`
- Treatment micro-combat: `cbb6883e86010f4ef270518bb4566ae827b7257497df14bcbcb854a2d8c2c52d`

Artifacts are retained under
`training-output/warmstarts/combat-256-hard-context-control-v4` and
`training-output/warmstarts/combat-256-hard-context-diverse-oracles-v4`.
