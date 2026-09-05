# Frozen-policy feeding-recovery pilot V1

This directional two-seed pilot applies the schema-1 stage-5 recovery evaluator
to both authoritative-router V4 behavior clones. Seeds `940000101` and
`940000202` are disjoint from the demonstration seeds and the prior feeding,
layout, contact, and micro-combat evaluation suites.

Both arms use `blob_rl/config/combat_warm_start_256.toml`, the memoryless
`collision_aware_forager`, the original evaluation deadline, a first fork after
one learned-policy decision frontier, an 8,192-quanta interval, at most two
forks per stage/seed, and an 8,192-quanta minimum remaining horizon.

## Result

| metric | hard-context control | diversified treatment |
|---|---:|---:|
| on-food policy objective | 2/2 | 0/2 |
| on-food policy survival | 0/2 | 0/2 |
| on-food teacher recovery | 4/4 | 2/2 |
| adjacent-food policy objective | 2/2 | 0/2 |
| adjacent-food policy survival | 0/2 | 0/2 |
| adjacent-food teacher recovery | 4/4 | 2/2 |
| recovery verdict | pass | pass |

The treatment produced only one eligible frontier per stage/seed because it
terminated before the next 8,192-quanta interval. Those early states were still
fully recoverable: after the learned policy had moved but consumed nothing, the
teacher consumed and retained at least 229 cells in every branch. The control
had already consumed at its sampled frontiers, but its own continuation killed
every cell; the teacher retained at least 235 cells in every branch.

This evidence rejects an early physical or information-boundary explanation
for these failures. At the sampled frontiers, the same isolated cells can still
feed and survive when controlled by a legal Mind. It supports proceeding to
exact-input learned/teacher disagreement and DAgger-style correction coverage.
It does not show that every later state is recoverable, and two seeds are not a
promotion-quality confidence sample.

## Bound artifacts

- Shared compiled ruleset:
  `3d6f1676c57ba75f07ffc7814c9c1b0121aaf0ce77c05c7ffb6858c1b3661855`
- Shared ordered scenario-suite hash:
  `e1a80e1801ea5b2d5de5f0d12f14e9b00a7032854e4faaf824cb8603dc242171`
- Control recovery artifact:
  `6a68f92869eb6c9bab425722489034c326a3505bafa8efe750469c209489be04`
- Treatment recovery artifact:
  `f8946ef1ab5fee891ec2c8c486f0c08280a156b3ff117b6e1cef46db660fffa1`
- Control model:
  `745fbd98dc7e31e96fc39fd1b583e0852fb77744511032e47a63ba3cd72c4a93`
- Treatment model:
  `1accb7fdc86767c371d54d251c58820aa2ea300395370c01690adf3f4db64285`

The immutable local artifacts are retained at:

- `training-output/warmstarts/combat-256-hard-context-control-v4/feeding-recovery-pilot.json`
- `training-output/warmstarts/combat-256-hard-context-diverse-oracles-v4/feeding-recovery-pilot.json`
