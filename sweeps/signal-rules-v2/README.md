# Durable-signal maintained-Mind control sweep

This report reruns the 12-variant, three-seed paired signal matrix after the
maintained colony was changed from one-quantum sidecars to semantic strengths:
plant 4x, threat 8x, construction 4x, and frontier 2x. Co-occurring facts use
the explicit four-channel `Signal` action when affordable; otherwise the
primary behavior keeps the highest-priority affordable sidecar.

The run is deterministic local evidence only. It contains 288 episodes, is
not server verified, and is too small for parameter selection.

| Variant | W-L-T | Explicit | Sidecars | Signal energy | Mean field | Observable variation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5-18-1 | 153 | 1,588 | 8,222 | 9.93 | 151.74 |
| signal disabled | 9-11-4 | 0 | 0 | 0 | 0 | 0 |
| current tile only | 4-19-1 | 162 | 1,761 | 8,820 | 10.12 | 0 |
| cardinal visibility | 5-17-2 | 156 | 2,040 | 9,862 | 9.28 | 70.63 |
| cost 4 | 5-19-0 | 8 | 499 | 7,360 | 38.37 | 600.84 |
| cost 16 | 5-17-2 | 0 | 188 | 6,496 | 33.05 | 522.13 |
| persistent | 7-15-2 | 54 | 548 | 3,040 | 114.49 | 1,514.29 |
| slow decay | 7-15-2 | 107 | 1,259 | 6,120 | 19.01 | 276.77 |
| fast decay | 3-21-0 | 109 | 1,506 | 7,304 | 1.76 | 27.68 |
| expensive disruption | 6-17-1 | 152 | 1,570 | 8,134 | 9.86 | 150.84 |
| sparse plants | 11-13-0 | 44 | 531 | 3,002 | 6.61 | 96.85 |
| dense plants | 5-13-6 | 298 | 5,524 | 24,140 | 9.08 | 139.50 |

## Interpretation

The revised policy exercises the intended mechanics: baseline controls include
153 explicit multichannel actions, all four semantic channels, durable fields,
passive decay, and 42 energy erased by terrain edits. This is a materially
better signal-system integration test than the v1 colony, whose one-quantum
deposits generally vanished during a standard action.

It is not a better game policy. The signal-disabled control beats baseline in
this seed suite, and hiding neighboring fields does not hurt results. Persistent
and slow-decay variants reduce repeat deposits but do not overcome the control.
The stronger policy also spends 8,222 energy in baseline matches and gives up
153 complete decisions to combined writes. Because energy is health and mass,
these are changes to survival, inertia, and action opportunity as well as
communication.

The v1-to-v2 outcome drop therefore must not be read as evidence that signals
are intrinsically harmful. It demonstrates that unconditional heuristic
beacons are too expensive and that raw win rate cannot distinguish field
utility from emission cost and foregone actions. The next evaluation should
separate sidecar strength, emission cadence, and dedicated-action opportunity
cost, then let a learned policy decide when communication has future value.
