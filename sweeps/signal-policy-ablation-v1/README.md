# Colony signal-policy ablation v1

This experiment holds the default rules, scenario, opponents, and twelve
mirrored seeds fixed while changing only the colony's local signal-deposit
policy. Each profile is a separate hash-bound control-matrix candidate. The
alternate native profiles receive the same isolated per-cell Mind input and no
shared state; only the full multichannel profile is the maintained Wasm
entrypoint.

The aggregate validates all five source reports, their common evaluation
identity, and all 96 paired episode keys before comparing outcomes. Paired
improvement treats win > timeout > loss.

| Policy | W-L-T | Paired vs disabled (+/-/=) | Explicit | Sidecars | Signal energy | Mean field |
|---|---:|---:|---:|---:|---:|---:|
| disabled | 47-37-12 | 0/0/96 | 0 | 0 | 0 | 0.00 |
| one-quantum sidecar | 51-23-22 | 16/3/77 | 0 | 32,347 | 32,347 | 0.75 |
| semantic sidecar | 39-55-2 | 8/26/62 | 0 | 6,243 | 26,988 | 6.87 |
| cadenced semantic sidecar | 41-48-7 | 8/20/68 | 0 | 3,796 | 18,326 | 3.96 |
| full multichannel | 34-60-2 | 4/27/65 | 407 | 5,700 | 27,748 | 6.99 |

## What this separates

- Disabled versus one quantum measures the combined effect of cheap ephemeral
  sidecars, including communication, health/mass shedding, and resulting
  inertia changes.
- One quantum versus semantic sidecar isolates stronger, longer-lived deposits
  while retaining primary actions. Strong deposits perform substantially
  worse despite fewer emissions.
- Semantic versus cadenced semantic sidecars tests emission frequency.
  Cadencing recovers some outcomes and reduces signal expenditure by 32%, but
  remains worse than disabled.
- Semantic sidecar versus full multichannel adds the opportunity cost of
  dedicated combined writes. The full profile spends 407 complete actions and
  has five fewer wins and five more losses.

The surprising useful seam is the one-quantum profile: it improves 16 paired
episodes and worsens only 3, primarily converting losses into timeouts. Because
the minimum field normally expires over a standard action interval, this is
not yet evidence of communication. Faster asynchronous neighbors may observe
the transient deposit, but the conserved transfer also makes emitters lighter,
faster, and less healthy. A neighbor-visibility ablation of this exact profile
is required to separate those mechanisms.

All reports are `local_deterministic`, `server_verified=false`, and
`replay_committed=false`. They are diagnostic controls, not leaderboard or
parameter-selection evidence.
