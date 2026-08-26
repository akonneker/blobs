# One-quantum signal-visibility ablation v1

This experiment keeps the one-quantum sidecar policy, physical deposits,
conserved energy transfer, passive decay, mass/health changes, rules, and 12
mirrored seeds fixed. Only neighboring signal observations change. The v2
aggregate also binds the matching no-deposit control from the policy ablation.

Paired improvement treats win > timeout > loss.

| Visibility | W-L-T | vs full (+/-/=) | vs disabled (+/-/=) | Observable variation |
|---|---:|---:|---:|---:|
| Full Moore-8 | 51-23-22 | 0/0/96 | 16/3/77 | 11.92 |
| Cardinal four | 51-24-21 | 0/1/95 | 16/4/76 | 6.20 |
| No neighbors | 50-25-21 | 0/3/93 | 15/5/76 | 0.00 |

Removing diagonal observations changes one outcome. Removing every neighboring
signal observation changes three, all in favor of full visibility. The
aggressive and defensive matchups are identical across all three visibility
profiles; differences occur only against simple and explorer controls.

The larger one-quantum benefit survives without neighbor communication. The
no-neighbor profile still improves 15 paired outcomes and worsens 5 relative to
the no-deposit colony. Therefore most of the earlier gain cannot be attributed
to neighboring cells reading the signal field. The leading mechanisms are the
conserved transfer out of assimilated health/mass, lower inertia and altered
action timing, or another same-cell physical interaction. Neighbor visibility
may help at the margin—full visibility is never worse than hidden in these
pairs—but only 3 of 96 episodes differ, which is too little for a tuning claim.

This is an important rules-design seam: a nominal communication action doubles
as controlled mass shedding. A learned policy may exploit it even when the
channel carries no useful information. That dual use is consistent with the
project's mass-energy principle, not automatically an exploit or a defect.
Future training should report the marginal visibility effect separately by
retaining hidden-neighbor controls. An explicit unreadable energy-dump action
would be an optional rule family only if experiments need to separate shedding
from communication as independently selectable behaviors.

The authoritative analysis is `aggregate-v2.json`. `aggregate.json` is the
earlier visibility-only analysis retained immutably. All artifacts remain local
and unverified.
