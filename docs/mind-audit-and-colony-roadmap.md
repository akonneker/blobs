# Mind audit and separation-safe colony roadmap

This audit distinguishes behavioral proxies from ABI/runtime canaries. A proxy
is intentionally simple and useful as an experimental control; it is not
presented as a competitive strategy.

## Maintained artifact audit

| Artifact | Useful as | Decision | Important limitation |
|---|---|---|---|
| `simple_mind` | Minimum survival, food-seeking, movement, and affordability baseline | Keep | No memory, reproduction, combat, signals, or terrain behavior. Food acquisition now precedes its low-energy guard fallback. |
| `aggressive_mind` | Combat-pressure and high-effort movement stress proxy | Keep | The ABI intentionally exposes no team identity. It attacks any locally occupied target, including an unmarked ally, and should not be interpreted as a competent team policy. |
| `defensive_mind` | Guard-heavy, low-mobility population control | Keep, but stop describing it as a wall/territorial strategy | It guards and reproduces; it does not build, communicate, identify allies, or patrol. Food acquisition now precedes its low-energy guard fallback. |
| `explorer_mind` | Stateful private-memory, cursor movement, and split-memory proxy | Keep | Its four-byte memory is a plant encounter counter and movement cursor, not a spatial map. |
| `colony_mind` | Stateful heterogeneous colony prototype and future planning baseline | Keep and develop | It coordinates through lossy local mechanisms and currently uses a bounded lineage-local map rather than a shared global map. |
| AssemblyScript and Go wait Minds | Cross-language Extism PDK admission and executor conformance | Keep as canaries only | `Wait` has no behavioral value. These artifacts must not enter strategy comparisons. |
| `isolation_canary` | Fresh-guest and explicit-memory isolation regression test | Keep as a canary only | Deliberately artificial behavior. |
| `invalid_mind_canary` | Admission rejection regression test | Keep as a canary only | Deliberately invalid and never a playable Mind. |

None of the four old Rust policies needs to be deleted: each covers a distinct
and cheap control condition. The important correction is classification. In
particular, `defensive_mind` and `explorer_mind` should not be used as evidence
that territorial defense or mapping works.

All Rust examples now use ABI-v7 exact affordability helpers. Attack payloads
and split allocations reserve both the action effort and minimum survival
energy, and final decisions are checked against the anonymous commit frontier.
This prevents ruleset changes from turning a proxy's preferred action into a
deterministic resolver rejection. Contention can still frustrate, contest, or
interrupt a valid action.

### Online execution cost

The release-only compatibility-executor diagnostic now compares
`simple_mind` with `colony_mind` on the same eight-slot input. On the current
Apple Silicon development machine, the latest pooling run measured about
16.7 microseconds per simple decision and 18.5 microseconds per colony
decision (roughly 59,900 versus 53,900 calls/second per serial worker). The
on-demand allocator measured 20.9 and 21.8 microseconds. The colony output is
520 bytes versus 96 bytes because it persists its private map, rolling
visitation window, and bounded outcome evidence. These microbenchmarks are
noisy; the important result is that per-plant tactical planning did not regress
the previous roughly 19.9-microsecond colony call. The 64-byte output increase
is the worst-case cost of adding four construction-state bytes to each of the
fixture's 16 private plant records.

This is a guest-call diagnostic, not a whole-simulation throughput claim. It
does show that the first planning slice adds about 5% to the pooled call on
this fixture; fresh guest setup remains much larger than policy arithmetic.
Map growth affects serialization size, so scenario benchmarks must also cover
full 16-entry maps. Avoiding unchanged memory replacements and compact map
deltas are future ABI/performance experiments, but any optimization must keep
memory cell-private.

## Colony Mind v1

`colony_mind` is one WASM policy with five private roles:

- **Feeders** consume, follow mapped plants or plant beacons, and split when
  their reserve can support a child and both cells' survival margins.
- **Explorers** dead-reckon a relative position from successful moves, record
  locally observed plants, perform a rotating sweep, and emit occasional
  frontier signals.
- **Builders** follow inherited maps or plant beacons, excavate source terrain
  outside a plant's immediate ring, carry it inward, and deposit around the
  plant one ring position at a time.
- **Defenders** move toward inherited plant records or build/plant signals,
  attack an adjacent occupant not carrying a recognized colony-role marker,
  and otherwise guard near a plant.
- **Attackers** prioritize adjacent attack windups, attack other adjacent
  occupants, and follow local threat signals.

Split children inherit a serialized copy of the parent's map with a corrected
relative position and an assigned role. There is no reference or pointer to a
shared map. The parent and child diverge independently after the split.

Child roles are now selected by a bounded local deficit estimator rather than
a fixed rotation. A reproducing cell scores demand using its own role and
private plant records, visible plants and ring elevations, nearby role markers,
anonymous signal levels, unmarked neighboring occupants, and stable action-space
parameters. An unmapped lineage favors exploration; a flat known plant ring
favors construction; a locally completed ring with build activity favors
defense; and strong threat evidence favors attackers. Signal magnitude is
log-bucketed, marker counts are capped, and a private cursor breaks equal-score
ties. This is neither a census nor authenticated coordination: spoofed markers
can redirect allocation, deliberately preserving that ecological pressure.

The first learned world model is deliberately small. Cells record observed
plant growth/capacity ranges, rank mapped plants by travel distance and learned
productivity, measure their own move success/failure history to select travel
effort, and treat `TerrainLimit` as evidence to advance to another wall
position. Exact action-cost coefficients, metabolism, terrain mass, and child
minimums are already local ABI inputs and are used directly rather than
needlessly re-estimated.

Exploration now uses an exact 8×8 rolling visitation bitmap stored in eight
private bytes. Successful movement recenters the bitmap and discards history
that falls outside the window, avoiding both modulo-coordinate aliases and the
eventual saturation of a global Bloom filter. Empty reachable targets not marked
visited are preferred after food and role-specific objectives. A split child
receives a copied bitmap shifted into its own local frame and then diverges from
the parent. The representation tracks visits, not canonical world coverage;
trusted telemetry therefore reports only aggregate bitmap density and never
merges these windows into a Mind-visible map.

The outcome model stores bounded success, setback, and contention counts for
movement, attack, guard, consume, split, excavate, deposit, and signal families.
Counts decay before saturation and reliability uses a prior rather than treating
one observation as certainty. Cell-local interval snapshots also estimate gut
progress, same-tile signal retention, and gross assimilated-energy loss while
guarded or unguarded. Split children inherit the lineage's aggregate evidence
but not the parent's pending snapshot.

These estimates now affect policy: movement contention selects standard rather
than gentle effort, unreliable or contested attacks risk less payload, sustained
attack success permits a larger commitment, learned digestion progress bounds a
new bite, persistent local signals suppress redundant emissions, and guarded
versus unguarded loss evidence modestly adjusts defender demand. Terrain-limit
outcomes remain the strongest directly observed saturation evidence.

Two effects cannot honestly be identified from the current ABI. A successful
attack does not reveal raw, mitigated, applied, or overkill damage to its Mind,
and an energy-loss interval does not reveal the counterfactual loss had the cell
not guarded. The policy therefore learns attack *resolution reliability* and
gross guarded/unguarded loss evidence—not an invented damage-transfer coefficient
or causal guard-mitigation value. Exposing either would require an explicit rule
and ABI decision about locally observable outcome detail.

## Operational constraints, not implementation shortcuts

The colony must preserve the project's central rule: each cell Mind is an
isolated agent with only its own state, bounded local observations, private
randomness, and explicit private memory. That has concrete consequences:

1. **There is no authenticated ally identity.** Role markers are
   unauthenticated local hints. An opponent can use the same marker, and an
   initial or otherwise unmarked ally is indistinguishable from an enemy.
   Friendly fire and marker spoofing are valid ecological pressures, not host
   services to route around.
2. **Signals are anonymous scalar fields.** Four energy channels can form
   plant, threat, construction, and frontier gradients. They cannot carry an
   arbitrary map, sender identity, or a trusted order. Emitting them costs
   energy and the exact affordability check can suppress a signal while
   retaining its physical action.
3. **Coordinates are lineage-local.** Successful moves update a private
   dead-reckoned frame. A split child can inherit that frame. Independently
   seeded cells have unrelated origins, and a wrapping world can create
   aliases. The current ABI provides no honest way to merge these into one
   canonical global map.
4. **“Find all plants” is not generally provable.** Without known bounds,
   global coordinates, or a trusted completion oracle, a cell cannot prove it
   has exhausted an unknown or wrapped world. The implementable goal is
   sustained coverage and a bounded record of every plant encountered by a
   lineage.
5. **A wall is terrain, not ownership.** Builders deposit on their current
   tile. Whether the elevation is actually impassable depends on a ruleset
   limit not exposed as a direct parameter. Builders therefore infer saturated
   positions from outcomes. Walls can also block colony movement and be altered
   by any cell with terrain actions.
6. **A role is behavior, not privilege.** All roles receive the same local ABI
   and action set. Role differentiation exists only in private memory and the
   policy's choice of action.

Adding global team memory, an unspoofable team identifier, an engine-maintained
map, or a hidden messaging bus would make the requested strategy easier, but
would violate strict cell separation. If richer communication is desired, it
should be introduced as an explicit configurable world rule with physical
cost, range, contention, observability, and adversarial access.

## Next development slices

1. **Completed:** trusted scenario telemetry now reports role counts, valid
   private maps, lineage-record totals, host-reconstructed unique plant
   discovery, elevated and actually impassable wall coverage, occupied ring
   tiles, defenders on station, builders carrying material, and energy in all
   four signal channels. It is versioned and serializable but is never a Mind
   input or canonical state.
2. **Completed:** deterministic
   resolver scenarios cover bounded and wrapped plant rings, one- versus
   two-layer wall semantics, a builder's full construction cycle, deliberate
   defender relief through a one-tile gate, spoofed markers, colliding
   plant/threat gradients, unrelated founder coordinate frames, and a
   stationed defender guarding beside a recognized colony cell. Telemetry also
   distinguishes raised terrain from an actual barrier in a ruleset where no
   practical elevation delta becomes impassable.
3. **Completed:** fixed role rotation has been replaced with bounded local
   demand estimates. Wall state, private plant knowledge, nearby signals,
   visible occupants, and capped role-marker hints can change the next split
   child's role without a population-wide census. Resolver coverage confirms
   that a plant feeder actually materializes the locally requested builder.
4. **Completed:** exploration now combines its slot cursor with an 8×8 rolling
   visitation bitmap. The window recenters on successful moves, prefers locally
   unvisited targets, is shifted for a split child's private frame, and costs
   eight bytes without increasing the ABI memory cap. Telemetry schema 2 reports
   aggregate and explorer-only bitmap density without treating it as global
   coverage.
5. **Completed to the current ABI's observability boundary:** bounded private
   estimators cover resolution reliability and contention for eight families,
   gut progress, gross guarded/unguarded loss intervals, terrain saturation,
   and same-tile signal retention. Smoothed evidence now adjusts movement
   effort, attack commitment, bite size, signal emission, and defender demand.
   Exact attack energy transfer and causal guard mitigation remain intentionally
   unclaimed because those outcomes are not exposed to the acting Mind.
6. **Completed:** tactical construction progress is stored per private plant
   record, so completing or contesting one site cannot redirect another.
   Builders retain deliberate gates and explicit outward sources, rotate pits
   after locally observed contention, skip every productive tile known before
   action selection, and stop a plant's plan after two layers. Local anonymous
   threat evidence repositions defenders to the south relief gate. Resolver and
   policy scenarios cover multi-plant deposits, productive wall/source tiles,
   per-plant pit pressure, and threat-driven gate control. An unseen plant still
   cannot be protected before it enters a cell's bounded observation.
7. **Completed as a smoke baseline:** the exact native entrypoints used by all
   five maintained Wasm Minds now run through a deterministic comparative
   matrix. Six paired regimes cover 16x16 and 32x32 boards, normal and sparse
   plants, wrapped and bounded edges, blocked diagonals, and higher action
   costs. Each seed is mirrored with the colony in both placement seats; the
   host inverts outcomes and telemetry labels for the second game without
   exposing identity or shared state to either Mind. The current three-seed
   matrix is deliberately too small for tuning claims.

The construction scenarios exposed and fixed several real planner defects. Builders
previously excavated any non-plant tile outside the ring, which could strand
them in a pit beside a raised wall; they now travel to explicit source points.
They also reused those pits for every wall pass; private memory now tracks the
wall layer per mapped plant and moves each layer's sources farther outward.
Repeated contested/frustrated source actions rotate a bounded flank offset,
while successful source work relaxes that pressure. Productive tiles observed
before the action are excluded from both deposits and excavation. Local routing avoids
observed elevation changes above one unit. Under the default one-unit movement
limit, the first eight deposits form a visually complete but still passable
ring. The second pass raises seven tiles to impassable height while
deliberately leaving the south cardinal traversable for defender relief. Its
exterior source is offset so the gate is not paired with an impassable
excavation pit.

The adversarial coordination slice confirms the intended limits: markers can
be spoofed, signals can collide without becoming messages, and independently
seeded coordinate frames can only be reconciled by trusted telemetry—not by a
Mind. Threat signals are likewise spoofable evidence: they can reposition a
defender to the relief gate, but they do not authenticate an attacker or issue
a trusted command. Local demand-driven role allocation, rolling exploration,
bounded outcome estimation, tactical construction, and a comparative smoke
harness are now complete at the prototype level. The corrected event-time
matrix contains 144 mirrored episodes. The colony won 1/36 against simple,
21/36 against aggressive, 9/36 against defensive, and 18/36 against explorer,
with 10, 1, 8, and 0 timeouts respectively. Of all episodes, 125 ended by
extermination and 19 at the public event-time deadline; none reached the host
decision-frontier safety guard. Three maps are still nowhere near enough to
estimate strength: these numbers are diagnostic baselines, not
parameter-selection evidence.

The current immutable report is
`sweeps/colony-controls-event-frontier-v3/control-matrix-event-frontier.json`. It labels itself
`local_deterministic`, `server_verified=false`, and `replay_committed=false`.
Native maintained entrypoints are deterministic and are regression-checked
against the Wasm Extism-compatible executor, but native evaluation is not a
substitute for executing submitted bytes inside the verifier. The prior
`colony-controls-v1` report is retained as a preliminary artifact, but its
timeout was measured in whichever Mind occupied team zero's decision
frontiers. That makes timeout comparisons scientifically confounded even
though it used mirrored seats; use the event-frontier-v3 report instead. The older
72-episode report is also retained and remains one-seat-only.

The recorded/live viewer slice is complete. The matrix runner can atomically
replace a compact progress report after each completed matchup, while retaining
full episodes and telemetry only in the immutable final artifact. The local
observatory presents opponent summaries, regime records, mirrored-seat splits,
progress, and the unusually large seat gap. Recorded and live routes use the
same Shadow-DOM component intended for later tinker.ninja embedding, but it
rejects trusted-status claims and explicitly says the matrix is not leaderboard
evidence.

The highest-value next slice is binding browser replay attestation to a
displayed run and then adding server re-execution of the submitted Wasm. Only
that server path may set trusted leaderboard status. After that, expand the
paired seed suite and add confidence intervals before tuning colony constants
or introducing learned weights.

The first winning-criteria sensitivity slice is also complete. A strict policy
grid can counterfactually score only canonical deadline draws from the horizon
matrix, with every compartment weight and minimum margin explicit. Across 22
draws, all six tested formulas assign the colony a loss; changing gut weight
from zero to full does not flip an outcome. This is diagnostic evidence about
the current Minds, not evidence that biomass should replace draws. Objective-
aware training and broader seeds remain necessary to measure turtling pressure.
