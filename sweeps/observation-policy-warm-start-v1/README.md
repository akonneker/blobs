# Observation-policy warm-start A/B

This bounded seed-42 experiment tests whether a successful anonymous-observation
combat oracle teaches target, effort, and timing better than the maintained
aggressive contact teacher.

Both arms use the same model, optimizer, validation split, 32 epochs, ten
multi-layout feeding datasets, and two contact datasets. The control gives the
contact datasets 2.5 shares each. The treatment preserves the 15-share total
and two-thirds feeding allocation while splitting the five-share combat budget
as 1.25/1.25 heuristic contact and 2.5 observation-policy oracle. The oracle is
the successful 16-seed searching-pursuer policy and contributes 112 distinct
exact decisions: 64 Attack, 32 Move, and 16 Wait.

The treatment has 14,904 training samples per epoch versus 14,806 in the
control because the current sampler defines an epoch as the complete eligible
training corpus before applying mixture weights. This is a 0.66% presentation
count difference and a residual experimental confound. The behavioral changes
are much larger, but a follow-up should add an explicit fixed epoch-presentation
budget before using close outcomes for selection.

## Result

The treatment substantially changes force application but is rejected. It
uses fewer attacks in held-out contact/skirmish evaluation while dealing more
than twice the damage and producing twice the kills. That improvement does not
generalize to the asymmetric micro suite: survival success collapses, and no
elimination objective is solved. Both arms fail the independent 256-cell
feeding gate with zero consumes.

| held-out metric | maintained control | oracle treatment |
| --- | ---: | ---: |
| 256-cell on-food success | 0/8 | 0/8 |
| 256-cell adjacent-food success | 0/8 | 0/8 |
| contact/skirmish attacks committed | 1,332 | 544 |
| contact/skirmish attacks succeeded | 1,320 | 484 |
| contact/skirmish damage | 2,716 | 5,952 |
| contact/skirmish kills | 28 | 56 |
| micro survival successes | 22/32 | 0/32 |
| micro elimination successes | 0/16 | 0/16 |
| micro attacks | 416 | 297 |
| micro damage | 1,408 | 2,800 |
| micro kills | 8 | 8 |

The supervised diagnostics explain part of the failure. Control held-out
Consume accuracy is 76.0%; treatment Consume accuracy is 1.1%. The treatment's
combat expert still predicts Attack at 100%, but its local gate routes only 55%
of Attack labels to that expert. In addition, oracle Move and Wait decisions
are currently assigned to the general/feeding expert by the local-action-family
router. Repeatedly resampling 112 narrow pursuit decisions at a 2.5 share can
therefore overwrite feeding behavior without teaching a general combat state
representation.

This rejects simply increasing the oracle dataset weight. The next experiment
should first define and audit a locally observable combat-context predicate,
route complete short combat sequences without host scenario metadata, add
oracles from several distinct opponents/objectives, and use an explicit fixed
presentation budget. Feeding remains a hard gate.

## Bound evidence

- Control behavior clone: `845a3b2606e456de7e601faffb22b731bf7242ca5079e648858095be2f25e6ae`
- Treatment behavior clone: `2f7a790ff056e3b8dd043c7f4a4cf7d58c931ea971dd2baba8deedef5b1d87e9`
- Control feeding: `e97534920a33eaca0c359892c50041ae763fdf6492a95677fb155cbad5703068`
- Treatment feeding: `0e985caeef376d1f188368cd24f28b87707ff4b4eb78b52f83d41717bb119838`
- Control contact: `e679416dc29e90185cd13403d195912069e17f7121354604f6efd261937a0e60`
- Treatment contact: `aaa04759794a0fe042732b6111f341472efd92ef9db207d1a7835ee219ebffb1`
- Control micro-combat: `42247ee14a7ec9c2a21a6315dd78d3fbd296731bf53888a576e8bf0f4d3cef72`
- Treatment micro-combat: `f0875510443e0eb3f7cbb5efa3b685cc02a8eef315945756423ad2edb03e0cc6`

The control is retained at
`training-output/warmstarts/combat-256-local-routing-base-v1`; the treatment is
retained at `training-output/warmstarts/combat-256-observation-oracle-v1`.

## Follow-up controls

The next implementation slice added two controls before rerunning this A/B:

- behavior-cloning schema 19 binds an exact `epoch_sample_budget`; the combat
  builder defaults both arms to 14,806 presentations per epoch;
- `visible-neighbor-context` routes a decision from the anonymous observation
  alone, so Move, Wait, Consume, and Attack in a visible interaction share one
  expert without consulting scenario identity or the selected label.

`demonstration-overlap` audited the exact 13 input corpora with that router:
17,036 exact samples, 839 distinct observable states, and zero contradictory
expert routes. It routes 80/112 oracle decisions to the interaction expert;
the remaining 32 are pursuit moves before any opponent is visible and remain
with the isolated/general expert. Dense feeding is intentionally split by the
same observable condition: all line/ring samples, none of the checkerboard
samples, 978/1,638 loose-random samples, and 108/1,638 random samples use the
interaction expert in each feeding stage. This turns the expert split into an
auditable isolated/social boundary and ensures that social Consume behavior is
rehearsed in the same expert that receives contact sequences.
