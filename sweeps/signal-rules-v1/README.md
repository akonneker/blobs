# Signal-rule control characterization

This is a local deterministic, unverified control matrix: three paired seeds,
four maintained opponents, and mirrored seats produce 24 episodes per variant.
It is a screening result, not a statistical tuning conclusion or leaderboard
artifact. The authoritative raw report is `control-matrix.json`.

| Variant | W-L-T | Win rate | Signal deposits | Signal energy | Mean field | Mean observable variation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 11-8-5 | 45.8% | 7,623 | 7,623 | 0.84 | 13.28 |
| signal-disabled | 9-11-4 | 37.5% | 0 | 0 | 0.00 | 0.00 |
| cost-4 | 10-12-2 | 41.7% | 2,558 | 10,232 | 13.63 | 210.84 |
| cost-16 | 4-20-0 | 16.7% | 502 | 8,032 | 51.15 | 803.97 |
| persistent | 10-11-3 | 41.7% | 1,276 | 1,276 | 50.58 | 680.31 |
| slow-decay | 8-13-3 | 33.3% | 4,060 | 4,060 | 4.54 | 69.62 |
| fast-decay | 10-8-6 | 41.7% | 9,712 | 9,712 | 0.17 | 2.69 |
| current-tile-only | 11-10-3 | 45.8% | 6,775 | 6,775 | 0.91 | 0.00 |
| cardinal-visibility | 11-9-4 | 45.8% | 7,137 | 7,137 | 0.88 | 6.98 |
| expensive-disruption | 12-7-5 | 50.0% | 8,062 | 8,062 | 0.78 | 12.40 |
| sparse-plants | 16-7-1 | 66.7% | 1,835 | 1,835 | 0.41 | 6.46 |
| dense-plants | 7-6-11 | 29.2% | 24,137 | 24,137 | 1.71 | 27.36 |

Preliminary interpretation:

- Baseline beats the signal-disabled control by only two episodes. The
  current-tile-only and cardinal variants match baseline wins, so this small
  matrix does not yet show that neighbor-visible gradients improve outcomes.
- Persistent signals produce a large standing field while suppressing repeat
  deposits through the colony Mind's local-retention logic, but do not improve
  wins. This is the clearest saturation warning in the batch.
- A 16-energy quantum is strongly harmful. This is consistent with signaling
  becoming a direct energy tax, not evidence about communication quality.
- Sparse plants favor the colony strongly, while dense plants create many more
  deposits and timeouts. Resource scarcity appears to be a more promising seam
  for coordinated behavior than simply increasing signal persistence.
- Terrain edits erased little signal energy in this batch. Terrain disruption
  needs a targeted scenario with signals deliberately placed on active
  excavation/deposition sites before its strategic effect can be evaluated.

The next experiment should increase seeds and compare learned policies against
the same fixed controls, preserving `signal-disabled`, `current-tile-only`, and
`persistent` as negative/saturation controls. Judge communication using outcome
improvement and ablation sensitivity together, not field energy alone.
