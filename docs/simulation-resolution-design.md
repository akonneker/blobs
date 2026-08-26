# Simulation State and Action Resolution

Status: implemented reference semantics with native parallel resolution
Scope: canonical simulation core, mind ABI, deterministic replay, and parallel execution

## 1. Purpose

This document defines the operational model for cells and the resolution of their
actions. Its most important goal is to preserve the project's core principle:

> A cell's decision may depend only on immutable mind code, bounded private
> state, its local observation and inbox, and private deterministic randomness.
> Every influence from another cell must pass through an explicit, local,
> replayable game mechanic.

The resolution model must also be efficient enough for native training, server
verification, and browser WASM. In particular, a failed action should not trigger
recursive retries or make a distant action succeed or fail merely because of
global iteration order.

This is a semantic design first. Worker counts, scheduling order, hash-map
iteration order, and native versus WASM execution must not affect a match.

## 2. Design goals

1. **Local causality.** Core actions touch a fixed-radius neighborhood.
2. **No accidental cascades.** Success or failure is not recursively
   re-evaluated after another result in the same resolution batch.
3. **Physical rather than enum priority.** Action kind does not grant a global
   ordering advantage.
4. **Determinism.** Identical rules, artifacts, initial state, and private random
   stream produce identical events and state hashes.
5. **Parallel evaluation.** Independent work is discoverable without changing
   semantics.
6. **Energy accounting.** Energy is never silently created or destroyed in a
   closed ruleset.
7. **Ruleset exploration.** Formulae and parameters can vary without weakening
   the separation or replay guarantees.
8. **Useful failure.** Failed actions have explicit time, energy, and observation
   consequences so minds and RL policies can learn from them.

Non-goals for the first reference ruleset:

- atomic group actions;
- unbounded pushing or vacancy propagation;
- long-range targeting;
- continuous rigid-body collision simulation;
- differentiating through the simulation engine;
- guaranteeing that every conflict component is small under every future
  ruleset.

## 3. Normative invariants

The terms **MUST**, **SHOULD**, and **MAY** are used normatively.

### 3.1 Decision isolation

- A mind invocation MUST contain exactly one cell's observation and private
  state.
- Arbitrary WASM state MUST be pristine before every invocation. Persistent
  state crosses the boundary only as an explicit `Retain` or
  `Replace(private_memory)` result and is stored by the game. `Retain` exposes
  no reference, pointer, or allocation identity to the Mind.
- A mind MUST NOT receive a public cell ID, team ID, global coordinate, global
  tick, population count, shared seed, or another cell's private state.
- Random values MUST be privately derived from the match secret, cell lineage,
  and that cell's decision sequence. The derivation and seed are not exposed.
- Mind decisions read an immutable observation snapshot. Mind execution can
  therefore run in parallel even when the cells may later interact.

### 3.2 Resolution locality

- Every core action MUST declare a maximum spatial footprint fixed by the
  ruleset.
- Reads from a resolution snapshot are immutable and do not create causal
  dependencies between actions.
- No action may consume a vacancy, energy, terrain change, or other resource
  produced by another action completing in the same batch.
- Failure is final for that completion. The engine MUST NOT retry, redirect, or
  recursively re-evaluate an action within the batch.
- Exact-time completions are simultaneous: a cell alive immediately before the
  batch completes its due action even if it receives lethal damage in that same
  batch. A death at an earlier simulated time cancels later actor-bound actions;
  detached effects such as an already-launched attack payload persist.

The third rule is the main barrier against board-wide movement chains. It is
referred to below as **snapshot-gated consumption**.

### 3.3 Deterministic parallelism

- Workers compute intents, claims, and deltas; they do not mutate canonical
  state directly.
- Reductions and merges MUST use a canonical integer algorithm.
- Floating-point arithmetic MUST NOT determine authoritative outcomes.
- Internal IDs MAY break otherwise indistinguishable bookkeeping ties only when
  the result cannot systematically favor a mind. The preferred reference rule
  leaves indivisible remainders in the world or makes equal claims collide,
  avoiding an ID tie-break.

## 4. Authoritative state

### 4.1 World state

Each tile contains independent fields:

```text
Tile {
    elevation
    occupant: CellKey?
    plant_energy
    plant_capacity
    plant_growth_rate
    plant_growth_remainder
    loose_energy
    diffuse_energy
    diffusion_remainder
    signal_energy[4]
    signal_decay_remainder[4]
}
```

`CellKey` is an engine-private stable handle. It can appear in replays and server
diagnostics but not in mind input. Plant energy and loose energy are separate so
regurgitation or death does not replace a plant.

Authoritative arrays SHOULD be dense and indexed by canonical tile index. Sparse
side structures are appropriate for cells, pending actions, and active field
regions.

### 4.2 Cell state

```text
Cell {
    key                         // engine-private
    position
    core_mass
    assimilated_energy          // health, available fuel, and mass
    gut_energy                  // mass, but not health or available fuel
    digestion_remainder
    metabolism_remainder
    carried_material
    marker
    posture
    private_memory[fixed_size]
    bounded_inbox
    age_time
    decision_sequence
    ready_at
    pending_action?
    last_outcome
}
```

The initial inertial mass model is:

```text
total_mass = core_mass
           + assimilated_energy
           + gut_energy
           + carried_material_mass
           + action_escrow_mass
```

Whether action escrow remains attached to the cell is action-specific, but it
must be counted exactly once.

`last_outcome` is a small private report such as `success`, `frustrated`,
`contested`, or `interrupted`, plus quantities directly experienced by the cell.
It MUST NOT identify another cell or reveal non-local state. This feedback is
important for hand-written minds and RL without weakening separation.

### 4.3 Locally observable activity

Slow physical actions need local counterplay. An adjacent cell SHOULD expose a
coarse physical cue derived from authoritative state:

```text
NeighborCue {
    marker
    apparent_mass_bucket
    activity: Ready | Moving | AttackWindup | Guarding | Feeding
            | Splitting | ManipulatingTerrain | OtherBusy
    progress_bucket: Early | Middle | Late
}
```

The cue is not a message from the other mind. It is a local observation of an
ongoing physical process, produced and replayed by the engine. It MUST NOT expose
the neighbor's `CellKey`, exact energy, exact completion timestamp, private
memory, or attack target. Rulesets may coarsen the activity categories further,
but the reference ruleset telegraphs an attack wind-up.

Only actions already committed before the observation snapshot are visible. Two
cells deciding at the same simulated time do not see each other's new choices.

### 4.4 Pending action

```text
PendingAction {
    actor
    kind_and_parameters
    origin
    fixed_target_or_direction
    started_at
    completes_at
    effort_spent
    payload_escrow
    maximum_footprint
}
```

Duration and commitment are fixed when the action starts from the actor's mass,
load, effort tier, terrain, and the frozen ruleset. Later damage does not
retroactively change its completion time.

An action separates:

- **effort**, irreversibly committed to attempting it; and
- **payload**, energy or material whose destination depends on the outcome.

This distinction permits failures to cost something without silently destroying
a split child allocation or attack projectile.

## 5. Time and decision model

The engine uses unsigned fixed-point simulated time and an event queue.

1. Advance to the next event time `t`.
2. Apply passive processes due before `t`, using deterministic time deltas.
3. Resolve all actions completing at `t` as one batch.
4. Apply death cleanup and field updates for that batch.
5. Build observations for all cells ready at `t` from the resulting immutable
   state.
6. Invoke their minds in parallel.
7. Validate and commit each returned decision, charge initial effort or escrow,
   calculate duration, and schedule its completion.

Actions MUST have positive duration. A newly selected action cannot resolve in
the same event cycle in which it was selected.

Cells are not invoked while an action is pending. Digestion, metabolism,
diffusion, plant growth, and signal decay depend on simulated time rather than
decision count.

### 5.1 Time granularity and decision-rate floor

Presentation frames, old-style ticks, and authoritative time are separate. The
proposed representation is a `u64` count with 1024 arithmetic quanta per nominal
simulation time unit. The event queue jumps directly between occupied
timestamps; it does not iterate through empty quanta, so fine arithmetic
representation does not itself add simulation work.

Action formulae use fixed-point integer arithmetic, then round duration upward
to a coarser `completion_bucket`. The proposed reference values are:

```text
arithmetic_quanta_per_time_unit = 1024
completion_bucket               = 64    // 1/16 time unit
decision_interval_floor         = 1024  // 1 time unit
guard_duration                  = 1024
```

Separating arithmetic precision from completion buckets matters for parallelism.
If every distinct mass produced a unique timestamp, completion batches would
fragment and exact-time conflicts would become rare. Sixteen completion slots
per time unit preserve meaningful inertial differences while keeping useful
batches. Values such as 8, 16, and 32 slots should be benchmarked because this
choice changes both performance and game behavior. It is therefore part of the
hashed ruleset.

Separately, every primary action obeys a positive `decision_interval_floor`:

```text
duration = max(decision_interval_floor, calculated_duration)
```

This floor bounds WASM invocations per cell per simulated time and is therefore
an online resource limit as well as a physical parameter. The proposed
one-time-unit value should be validated with mind-call benchmarks. `Guard` uses
exactly this minimum duration and is the fastest reference action.

## 6. Decision contract and proposed core actions

```text
Decision {
    primary_action
    memory_update: Retain | Replace(private_memory)
    optional_signal
}
```

The optional signal sidecar is bounded, local, and separately costed. It does
not allow a second physical action. A Mind that needs to write several channels
at once can instead choose the explicit `Signal` primary action.

| Primary action | Intended footprint | Principal state effect |
| --- | --- | --- |
| `Wait` | actor | Recover or deliberately spend time |
| `Move(direction, effort)` | actor, origin, adjacent target | Relocate if target was empty in the completion snapshot and uncontested |
| `Attack(direction, effort, payload)` | actor, adjacent target | Deliver committed mass-energy to a coordinate and scatter damage locally |
| `Guard(effort)` | actor | Activate guard immediately and remain guarded until another primary action is committed |
| `Consume(amount)` | actor, current tile | Move plant/loose energy into gut, limited by bite rate and gut capacity |
| `Split(direction, child_energy, marker, child_memory)` | actor, adjacent target | Create a child using escrowed assimilated energy |
| `Regurgitate(direction, amount)` | actor, adjacent target | Move gut energy to loose energy on the target tile |
| `Signal([channel_amount; 4])` | actor, current tile | Atomically deposit independent strengths in all four anonymous local channels |
| `Excavate` | actor, current tile | Convert local terrain into carried material |
| `DepositTerrain` | actor, current tile | Convert carried material into elevation at the current location |

This table is an action-family proposal, not a frozen ABI.

`Guard` changes posture at action commitment rather than completion. It schedules
the cell's next decision after `decision_interval_floor`. The posture persists
through that interval and remains until the cell commits a different primary
action; choosing `Guard` again maintains it. Committing any other action clears
guard immediately. Consequently, if `A` commits an attack at time `t` and `B`
commits guard at `t`, an attack from `A` that completes later observes `B` as
guarding. If an attack completes at the exact instant `B` becomes ready, due
completions resolve before new decisions, so `B` cannot retroactively guard.

Actions unavailable under a ruleset SHOULD remain representable in a stable ABI
and be disabled through capability flags and action masks. A semantic change to
information flow, target tracking, conservation, or action footprint requires a
versioned kernel or ABI, not merely a parameter change.

### 6.1 Configurable local neighborhoods

The ABI should target a local slot rather than hard-code compass directions into
every action:

```text
LocalSlot {
    dx: signed_small_int
    dy: signed_small_int
    distance_cost
}

NeighborhoodSpec {
    slots[bounded_count]
    observed_slots_by_channel
    target_slots_by_action
    diagonal_corner_rule
    boundary_rule
}
```

A factorized physical action then contains a verb, `Self` or a `LocalSlot`, an
effort tier, and any action-specific amount or payload. For example, a ruleset
can permit eight movement targets, four attack targets, eight regurgitation
targets, and self-only terrain deposition without defining new action variants.

The ruleset compiler MUST:

- enforce a small maximum slot count and spatial radius;
- reject action targets outside the action's declared slots;
- derive every action's maximum resource footprint;
- precompute `neighbor_tile[tile][slot]` with a boundary sentinel or wrapped
  index;
- precompute action masks and fixed-point distance/cost multipliers;
- include slot ordering, masks, and boundary behavior in the canonical ruleset
  hash.

The public ruleset manifest describes slot meaning. It is identical for every
cell and therefore does not violate decision isolation. A fixed ABI can carry a
bounded list plus valid/action masks; neural observation and action shapes are
compiled for a particular ruleset.

Interesting initial choices include:

- four orthogonal versus eight surrounding neighbors;
- different neighborhoods for sensing, movement, attack, splitting, signaling,
  and material transfer;
- equal diagonal cost versus an integer approximation of `sqrt(2)`;
- allowing or forbidding diagonal corner-cutting between occupied or elevated
  orthogonal tiles;
- bounded versus wrapping world edges;
- which physical activity and field channels are observable at each slot.

The initial implementation SHOULD support a square grid, self plus a bounded
offset table, per-action slot masks, diagonal cost, and boundary policy. It
SHOULD NOT initially support arbitrary graphs, hexagonal rendering, line of
sight, unbounded radii, variable-sized cell footprints, or user-supplied
resolution code. Named semantic kernels remain necessary for choices such as
snapshot-gated occupancy versus pushing or `all_collide` versus impulse
arbitration.

### 6.2 Implemented canonical Mind ABI v8

The checked-in `reference_mind.capnp` schema is the sole versioned Mind
boundary. Its input contains only one ready cell's own
mass/energy/gut/guard/outcome state, its current authoritative tile, canonically
ordered relative slots filtered by the ruleset observation masks, action masks
and bounds, explicit private memory, and 32 bytes of private random output.
Current-tile state is not represented as a synthetic slot: `Consume` acts at
the current location, while every slot is a distinct relative target.

The output is a complete decision containing one primary action, an optional
anonymous four-channel local signal emission, and an explicit private-memory
operation: retain the current bytes or replace them with bounded owned bytes.
This is necessary because arbitrary WASM state is pristine for every call; a
bare action would provide no legal persistence channel. All
standard actions currently implemented by the semantic kernel are represented
without adapter defaults: Wait, Move, Attack, Guard, Consume, Split,
Regurgitate, Signal, Excavate, and DepositTerrain. ABI version 8 adds a
variable-strength amount to the optional one-channel signal sidecar and an
explicit four-channel signal vector. It retains ABI v7's locally
observable plant capacity and growth rate plus the cell's metabolism remainder
and the frozen metabolism rate, terrain capabilities, and anonymous
signal-field channels. It also exposes the active effort-cost ratios and
per-family cost coefficients. A Mind can therefore derive exact affordability
from its own mass/energy and each observed slot's distance without any identity,
peer state, or global information channel.
Identity-bearing messages and inboxes remain absent because they would require
different information-flow semantics; they must not be accepted and silently
ignored.

Learned recurrent policies use this same boundary rather than a privileged
trainer channel. Their versioned, signed-16-bit-quantized recurrent vector is
read from the acting cell's private bytes and returned with `Replace`; split
children receive the same newly computed vector. Empty, malformed, or
architecture-mismatched bytes reset to zero. GPU/CPU batching remains row-
separable and receives no cell identity, team identity, peer memory, batch
aggregate, or cross-row attention. Host cell keys exist only long enough to
route each output back to its originating private state. Thus recurrence
changes policy capacity without weakening the strict decision-isolation
principle.

Cap'n Proto conversion is allocation-bounded and rejects truncated input,
noncanonical slot ordering, impossible action masks, malformed private random
data, inconsistent rejection outcomes, unknown effort bits, and oversized
private state. The exact schema source and an independent ABI version are bound
by `reference_mind_abi_hash`. Projection tests prove that translated local
situations with different engine-private cell keys encode identically.

ABI v8 retains v6's inline optional-observation representation plus
a canonical visibility bitmap. This preserves the distinction between hidden
and visible zero, including independent neighbor detail masks, while removing
the former pointer-backed optional object for every scalar. Unknown bits,
neighbor detail without neighbor presence, progress without activity, and
nonzero hidden fields are rejected. Neighborhood size, ordering, visibility,
and action masks remain ruleset-configurable; this is a wire-layout change, not
a new information channel or a hard-coded neighborhood policy.

The semantic resolver projects this input and maps every primary action
losslessly to `ActionRequest`. It atomically installs both the action and the
bounded explicit memory update; rejected actions still apply that update
exactly once. `Retain` preserves the existing canonical allocation, while
`Replace([])` deliberately clears it. Replay format version 10 records the
tagged update, variable-strength optional signal, and explicit signal vector in
the decision commitment, and authoritative native/browser replay applies it
before resolution. Live native dispatch uses the `ReferenceMind`/
`ReferenceMindFactory` boundary, while every hosted WASM team must export
`reference_mind_function`. Both paths receive the same canonical projection and
commit the returned memory operation without defaults supplied by the host.

Every WASM decision runs in a fresh store/instance. A trap, missing export,
malformed decision, noncanonical action, or oversized memory fails the tick
before authoritative state is synchronized; it is not converted into Wait.
The host drains all already-dispatched worker results and rolls every ready
cell's private decision sequence back, so a failed batch cannot consume private
randomness or poison a later retry. The Game host has one ABI and does not mix
execution modes. Host checkpoint version 11 stores only the canonical reference
runtime and rejects every prior host-checkpoint version. Duplicate team
registration is rejected instead of replacing a pool or creating a second
starting cluster.

The default executor remains Extism. The selectable `extism_compat` executor
loads the same PDK artifact directly in Wasmtime and implements only Extism's
deterministic byte-arena imports: input reads, bounded allocation/free/length,
bounded load/store, output, and error. It rejects WASI, configuration,
variables, HTTP, custom host functions, and logging at module admission. All
workers in one compatible pool share only its immutable Engine, Module,
InstancePre link plan, and deadline ticker. Every invocation creates a new
Store, instance, guest linear memory, host byte arena, allocation map, input,
output, error state, and resource limiter. Pooling resources are sized for the
pool's maximum concurrent worker count. Thus executor selection changes host
overhead, not the Mind isolation or ABI. Manifest memory and time limits are
enforced in both modes. Stock Extism retains one compiled descriptor per worker
because its public compiled type can contain non-Send/Sync host user data; the
host does not bypass that boundary with an unsafe wrapper.

The host byte arena and its allocation/free metadata use small per-call inline
storage for the common case and transparently spill when a Mind needs more.
This is a physical allocation optimization inside the newly constructed
`CompatState`; it neither preserves bytes between calls nor changes offsets,
limits, range validation, or any Mind-visible behavior.

## 7. Failure model

Failure is data, not control flow. Every committed action has exactly one terminal
outcome:

| Outcome | Meaning | Default consequence |
| --- | --- | --- |
| `Rejected` | Malformed, unavailable, or unaffordable at decision time | Replace with a minimum-duration wait; do not charge requested payload |
| `Success` | Preconditions and all exclusive claims succeeded | Apply declared delta |
| `Frustrated` | A required local condition is already false in the completion snapshot, whether because of an earlier action or pre-existing state | Effort remains spent; action-specific payload is returned or deposited |
| `Contested` | Multiple actions that are individually valid in the same completion snapshot make incompatible exclusive claims at that timestamp | All equal-time claims collide in the reference ruleset; effort remains spent |
| `Interrupted` | Actor died strictly before an actor-bound completion | Cancel completion and dispose of escrow through the death rule; already-detached effects are not interrupted |

The reference conflict rule is intentionally symmetric: if more than one valid
action tries to create occupancy on the same empty tile at exactly the same time,
none succeeds. Alternative rulesets may compare physical arrival impulse, but
must not introduce action-enum priority.

For example, if `A` enters an empty tile at `t = 1` and `B` attempts to enter it
at `t = 2`, `B` is frustrated because the tile is occupied in `S_2`. If `A` and
`B` both complete moves into the tile at `t = 1` and it was empty in `S_1`, both
are contested. The actions may have started at different times; their completion
timestamp and claims determine contention. A cell receives only the outcome and
locally experienced quantities, not an authoritative explanation of which other
cell caused a frustration.

Suggested payload behavior:

- Move has effort but no payload. A blocked move spends its effort locally.
- An attack payload is reserved during the visible wind-up and detaches at
  completion. If the actor dies earlier, the actor-bound attack is interrupted
  and its escrow joins the death spill. Once a payload has detached, it cannot be
  recalled; if it hits no cell, it lands as loose/diffuse energy rather than
  disappearing. The reference adjacent attack impacts when it detaches, while a
  future ranged-projectile kernel could schedule a separate persistent event.
- Split energy remains escrowed with the parent until completion. On frustration
  it returns after the action; on earlier parent death it joins the death spill.
- Regurgitated gut energy remains escrowed with the actor until completion. A
  failed placement returns it; on earlier death it joins the gut spill.
- Consume has no payload. It receives a deterministic allocation from energy
  present in the completion snapshot.

Exact effort and recovery fractions belong to the ruleset and conservation
ledger.

## 8. Multi-pass resolution

Let `S_t` be the immutable state immediately before all completions at time `t`.
Let `A_t` be those due actions.

### Pass 1: Local validation

In parallel, convert each action into a normalized intent using only `S_t`:

- confirm the actor exists in `S_t`;
- verify range, terrain, capacity, and action-specific parameters;
- classify snapshot-known frustrations immediately;
- capture target coordinates or target occupants according to the action's
  documented semantics;
- emit a bounded list of resource claims and commutative contributions.

An action that fails here does not enter a conflict component.

### Pass 2: Claim indexing

Claims use compact canonical keys:

```text
ResourceKey = Occupancy(tile)
            | CellState(cell)
            | LooseEnergy(tile)
            | PlantEnergy(tile)
            | Terrain(tile)
            | Inbox(cell)
            | Signal(tile)
```

Each claim declares an access algebra:

- `ReadSnapshot`: immutable; never connects actions.
- `Add`: commutative integer contribution.
- `Min/Max`: commutative deterministic reduction.
- `BoundedTake`: proportional or otherwise canonical allocation from a frozen
  quantity.
- `Exclusive`: mutually incompatible structural claim.

Claims are bucketed by resource key using deterministic radix sorting or dense
tile buckets. Pairwise action comparison is forbidden.

### Pass 3: Exact conflict graph

Construct a graph only for exclusive, multi-resource decisions:

- vertices are still-valid intents;
- an edge joins two intents only if their exclusive write claims conflict;
- immutable reads and commutative reductions do not add edges.

Connected components are independent and may be assigned to workers. Most
actions need no explicit component at all: attacks are additive cell/field
contributions, consumption is a bounded allocation per tile, and snapshot-blocked
moves have already terminated.

The graph can be represented implicitly by resource buckets plus a disjoint-set
union structure. It does not need pointer-heavy per-action adjacency lists.

### Pass 4: Component resolution

Resolve each component from `S_t` into a private delta buffer. The reference
occupancy rule is:

1. The target must be empty in `S_t`.
2. Exactly one valid completion may claim the target.
3. The actor must occupy its recorded origin in `S_t`.
4. A source tile becoming empty in this batch cannot satisfy another action.

Alternative physical arbitration, such as greatest arrival impulse with equal
impulses colliding, can be a ruleset kernel while retaining steps 1, 3, and 4.

### Pass 5: Deterministic reduction and commit

Merge component deltas and commutative contributions by canonical resource key.
Then:

1. apply successful structural occupancy changes;
2. apply cell energy, damage, gut, and posture deltas;
3. determine deaths after all same-time contributions;
4. spill dead cells' remaining compartments at their resulting positions;
5. apply terrain, loose-energy, diffuse-energy, inbox, and signal deltas;
6. record outcomes and schedule recovery/next readiness;
7. emit canonical events and a state hash.

An action due from a cell alive in `S_t` contributes even when the combined
same-time damage is lethal. This eliminates iteration-order cancellation.

### 8.1 Locality guarantee for the reference actions

For the reference action set, changing or removing one completion can affect only
resources in that action's fixed footprint and actions making a direct exclusive
claim on one of those resources. Another action's changed outcome is never used
as fresh input during the batch, so the dependency cannot be followed again.

This bounds the causal depth of one completion batch even if a resource bucket
has many contributors. It does not claim that influence can never travel far over
simulated time: later cells may observe a changed local state, act on it, and pass
the influence onward. That finite-speed, multi-decision propagation is intended
emergent behavior. The prohibited case is an arbitrarily long hidden computation
inside one nominally local resolution step.

## 9. Pre-action potential graph

Before a mind has selected an action, the exact conflict graph is unknowable. A
separate **potential interaction graph** may nevertheless provide a broad-phase
certificate:

- a vertex is a ready cell or already-pending action;
- two vertices are potentially adjacent when their maximum ruleset footprints
  can overlap;
- cells separated by more than twice the maximum footprint radius are certified
  non-interacting for that completion horizon.

This graph can be maintained incrementally from occupancy changes or derived
from spatial bins. It is useful for cache-local observation building, assigning
nearby pending events to the same worker, and proving that spatial regions need
no shared resolution resources.

It must not be mistaken for a causal graph. A long line of adjacent cells forms
one transitive potential component even when every actual action is independent.
Using those broad components to schedule mind calls would unnecessarily
serialize the workload. Minds already read snapshots and can all run in
parallel; group them primarily for WASM artifact reuse and memory locality.

For an event-driven implementation, exact action footprints are known after
commit and may be registered in buckets indexed by `completes_at`. At time `t`,
the engine can retrieve the due footprint buckets directly, revalidate them
against `S_t`, and build only the exact conflicts that remain.

## 10. Thought experiments

### 10.1 Convoy moving toward an empty tile

```text
before: A B C .
intent: > > >
after:  A B . C
```

Only `C` sees an empty target in `S_t`. `A` and `B` are independently frustrated.
They do not consume vacancies produced by the same batch. The computation is
linear work with no sequential dependency chain.

Over later decisions the empty space can propagate backward one tile at a time.
That propagation occurs through new observations and explicit simulated time,
not recursion inside one resolution.

### 10.2 Convoy ending at a wall or stationary cell

```text
before: A B C #
intent: > > >
after:  A B C #
```

All three actions are locally frustrated from `S_t`. The final blockage does not
cause a graph walk from `C` back to `A`; each result is computed independently.

### 10.3 Swap and rotation

```text
A B       A wants B's tile; B wants A's tile
```

Both targets are occupied in `S_t`, so both moves fail. The same rule rejects a
ring rotation. If swapping is ever desirable, it should be an explicit bounded
mechanic, not an emergent exception in ordinary movement.

### 10.4 Two cells enter one empty tile

```text
A . B
  ^
```

Both pass snapshot validation and share one exclusive occupancy claim. In the
reference ruleset both collide and remain at their origins. This component has
two actions and one contested resource. A physical-impulse kernel could instead
select a unique stronger arrival while exact ties still collide.

### 10.5 Split and move target the same empty tile

The child placement and move are the same class of occupancy claim. Neither gets
priority because one is `Split` and the other is `Move`; both collide under the
reference rule. Different completion times resolve naturally in time order.

### 10.6 A blocker is killed as another cell moves toward it

The destination was occupied in `S_t`, so the move is frustrated even if the
occupant dies in the batch. The corpse energy is deposited during commit and the
vacancy can be observed and used by a later decision. Kill-to-vacancy-to-move
cannot chain within one batch.

### 10.7 A chain of simultaneous attacks

```text
A attacks B; B attacks C; C attacks D
```

All impacts due at `t` are computed from `S_t` and reduced additively. If `B` is
killed by `A`, the attack already completing from `B` still contributes. There
is no death-cancellation walk down the chain. An attack still winding up when its
actor dies at an earlier time is interrupted; a future detached projectile would
persist independently.

### 10.8 Consume and regurgitate on one tile

The occupying consumer can take only energy present in `S_t`. Energy regurgitated
onto its tile at `t` is added during commit and is not consumable until a later
completion. This prevents a same-time regurgitate-consume conveyor from becoming
an unbounded computation chain. If a later ruleset permits multiple cells to
consume one field, it must use a canonical bounded allocation and leave any
indivisible remainder on the tile.

### 10.9 A cell loses mass while an action is pending

Its scheduled completion time does not change. Mass loss changes the duration
and cost of its next action. Rescheduling the current action would propagate
timing changes through the event queue and make damage resolution much more
expensive and difficult to replay.

### 10.10 Very large local conflict

Many cells can still attack one cell or regurgitate onto one tile. These are
high-degree resource buckets but not sequential graph components: contributions
are reduced in parallel and combined canonically. A genuinely multi-resource
exclusive action can create a large component, so the engine retains a
deterministic serial fallback and records component-size telemetry.

### 10.11 Guard and a telegraphed attack

At `t = 0`, adjacent `A` and `B` receive observations. `A` commits an attack that
will complete at `t = 2.25`; `B` commits guard. Neither saw the other's new
choice, but guard becomes active immediately. At `t = 1`, `B` becomes ready and
observes `A` in `AttackWindup` with coarse progress. If `B` guards again, posture
remains continuous and the attack samples a guarded `B` at `t = 2.25`. If `B`
commits another action at `t = 1`, guard clears immediately.

If a quick attack and guard both have duration `1`, the attack completion at
`t = 1` resolves before `B` receives its next decision. The guard selected at
`t = 0` still applies, but `B` cannot see the attack and add a retroactive guard
at the instant of impact.

## 11. Complexity and worker scheduling

For `N` completions with maximum `K` claims per action, where `K` is ruleset
bounded:

- normalization and validation: `O(N)` work;
- claim emission: `O(KN)` work and storage;
- dense/radix claim indexing: `O(KN)` expected or bounded integer-key work;
- component construction: proportional to actual exclusive collisions, not all
  neighboring pairs;
- deterministic merge: proportional to touched resources.

The convoy examples remain `O(N)` total and highly parallel rather than becoming
an `O(N)` critical path.

Workers SHOULD receive components or resource buckets using estimated cost, not
just vertex count. They write thread-local deltas and event fragments. Work
stealing is safe because canonical merge order, not worker order, determines the
result.

Spatial chunks with read-only halos are useful for observation assembly and
field stencils. Cross-chunk actions are routed by canonical target tile. No
worker should lock cells or tiles one at a time during authoritative resolution.

The engine SHOULD expose these metrics in benchmarks and replay diagnostics:

- actions by terminal outcome;
- claims and touched resources per action family;
- exact component count, maximum size, and size histogram;
- resource-bucket contention histogram;
- validation, graph, component, reduction, mind, and field-update time;
- fraction of work handled without an explicit component;
- deterministic state hash per event batch or replay interval.

### 11.1 Passive-field density and lazy metabolism

Sparse passive work follows exact derived frontiers. When at least 75% of a
large native cell or tile domain is active, host-only thresholds may select
parallel dense kernels. This selection is non-authoritative: the serial and
parallel paths must produce identical cells, tiles, indexes, errors, hashes,
and replay frames. Diffusion uses source planning followed by destination
gathering through reverse adjacency; worker-order atomics are not permitted.
WASM uses the serial path.

Metabolism is not currently lazy. Canonical `assimilated_energy` and
`metabolism_remainder` are materialized at simulation time `now`, and spent
energy is deposited at the cell's current tile before any same-time diffusion.
An exact derived indexed min-heap now maintains absolute exhaustion deadlines:
sparse energy changes replace one entry, dense digestion rebuilds the heap in
linear time, and ordinary metabolic accrual preserves each absolute deadline.
The index is reconstructed after restore or rollback and never enters canonical
state. A true lazy scheme cannot merely attach a timestamp to cell energy: it
also needs deferred, position-sensitive tile deposits. At minimum, a future
version would require:

- per-cell base energy, remainder, and last-materialized time;
- a derived minimum exhaustion index updated after every energy mutation;
- deferred metabolic deposits that materialize before movement, death, tile
  observation, and each diffusion boundary;
- full logical materialization for verified hashes, checkpoints, reversible
  deltas, and replay unless those formats are deliberately versioned to commit
  the lazy representation itself.

Consequently, laziness is most promising for on-demand-integrity RL rollouts
with long intervals between diffusion or inspection barriers. It cannot remove
the dense verified-hash cost under the current canonical format. The derived
exhaustion-time prerequisite is now implemented and measured; deferred tile
deposits and materialization barriers remain the next semantic design slice
before changing the state representation.

## 12. RL consequences

Action duration changes the learning interval, so discounting should use
simulated time (`gamma^delta_time`) rather than number of decisions. Rollouts and
reward rates should likewise be measured in simulated time.

The bounded private `last_outcome` lets a policy distinguish a collision from a
successful move without granting global knowledge. The reward function should
not treat every failed action as terminal or omit elapsed-time costs. Otherwise,
policies may learn to spam blocked actions or exploit differences between fast
failure paths and successful paths.

Snapshot-gated consumption intentionally changes the learned coordination
problem: a convoy must react over time instead of obtaining atomic group motion
for free. Explicit signaling or learned spacing can improve throughput without
introducing a hidden team-level primitive.

## 13. Replay and server verification

A replay event for an action should record:

- ruleset and ABI hash;
- simulated start and completion time;
- private actor handle used only by the engine/replay;
- normalized action parameters;
- committed effort and payload ledger entries;
- terminal outcome and locally observable result;
- canonical resource deltas;
- periodic full-state or chunk-state hashes.

The server re-executes the same deterministic pipeline using the submitted mind
artifact and server-derived private random samples. The match secret and sample
derivation inputs never enter the Mind ABI. Browser results remain provisional
until native verification reproduces the canonical hashes. Worker partitioning and
parallel execution order are deliberately absent from the replay contract.

### 13.1 Canonical hash contract

The reference implementation uses SHA-256 with domain-separated, versioned
canonical encodings. It does not use Rust's `Hash` trait, debug output, memory
layout, or platform-sized integer bytes. Integer values are encoded explicitly
in little-endian order, collection lengths and tile indices use `u64`, options
have explicit presence tags, and enum variants have stable hand-assigned tags.

Three related hashes have distinct purposes:

- the **semantic ruleset hash** covers the reference kernel version, local-slot
  ordering, observation/action masks, boundary and corner policies, and every
  numerical rule parameter;
- the **compiled ruleset hash** binds the semantic ruleset hash to the compiled
  world dimensions, which determine edge and wrapping behavior;
- the **state hash** binds the compiled ruleset hash to simulated time, the cell
  key allocator, every tile reservoir and occupant, and every authoritative cell
  field including private memory, pending actions, escrow, and last outcome.

The separate authoritative runtime hash binds that canonical state to match
progress, the secret-randomness commitment, and the minimal private dispatch
metadata that can affect later Mind calls: cell-to-team ownership, random
lineage, and decision sequence. Renderer-only projections such as displayed
age, loaded flags, inboxes, and host pheromone copies are deliberately excluded;
their authoritative counterparts either live in canonical state or are not part
of the reference Mind semantics.

Every reference completion batch emits both its compiled ruleset hash and its
post-commit state hash. Golden vectors pin the byte contract. Changing that
contract requires a hash-format or semantic-kernel version increment rather than
silently accepting divergent replay histories.

### 13.2 Replay batch frames and chain

Replay wire format version 8 records one completion batch per canonical binary
frame. A frame contains its sequence number, compiled ruleset hash, previous
event hash, previous post-state hash, pre-resolution state hash, post-resolution
state hash, completion time, and the action commitments made at the preceding
decision boundary. Each commitment binds its private actor handle, full
normalized request, optional anonymous signal emission, tagged private-memory
update, start/completion times, acceptance or rejection, effort, and payload
escrow. The frame also contains terminal outcomes, resource claims,
deaths, births, and the reversible state delta. Split requests therefore retain
their marker and bounded private child memory at both commitment and completion
rather than being reduced to an action discriminant.

The chain begins with a domain-separated genesis hash over the compiled ruleset
and initial state. Each event hash commits to the complete frame body and the
previous event/state links. Decoding verifies the frame hash, stable ordering,
sequence, monotonic completion time, ruleset identity, and both chain links.
Configurable limits bound frame bytes, collection counts, and private-memory
allocation before untrusted browser submissions are accepted into memory.

This integrity check detects mutation, truncation, reordering, and splicing, but
it does not make a client-authored run authoritative. A dishonest client can
construct a self-consistent false chain. Leaderboard trust still requires the
server to re-execute the submitted mind and reproduce the recorded pre/post
state hashes.

### 13.3 Reversible deltas and indexed archives

Each completion frame carries a sparse canonical delta from the batch's
pre-resolution state to its post-resolution state. It records the before/after
time and cell-key allocator plus only tiles and cells whose authoritative state
changed. Tile entries are strictly ordered by tile index, cell entries by cell
key, and unchanged or duplicate entries are rejected. Applying a delta verifies
the complete source value of every changed resource before mutating anything;
the same payload can therefore be applied forward or backward without accepting
a partially matching state.

The pre-resolution state is captured after passive time advancement and before
the due actions resolve. Consequently, a delta is an exact reversible record of
one resolution batch, but deltas alone do not reconstruct passive changes across
the interval between batches. Efficient arbitrary-time replay will use periodic
canonical checkpoints (or deterministic re-execution from an earlier
checkpoint) in addition to these batch deltas.

Replay frames can be packaged in a bounded indexed archive. Its canonical
header commits to the ruleset, initial state, and a contiguous offset/length
entry for every frame; a trailing domain-separated SHA-256 hash covers the
entire archive body. The index provides constant-time access to an event frame
without scanning earlier frames. Archive parsing bounds total bytes, event
count, frame size, and nested collection allocations, then validates the full
event/state hash chain. As with individual frames, archive integrity is not
proof of honest execution: the authoritative server must still re-execute a
submission under the frozen compiled ruleset.

### 13.4 Server re-execution and attestation

The verification envelope is distinct from the Mind ABI. It may contain
server-private competition slots and artifact identifiers, but none of those
values enter a cell observation. The canonical match-verification manifest
binds a hashed match identifier, the exact Mind ABI schema hash, sorted
slot-to-artifact hashes, the versioned Mind runtime profile, compiled ruleset
and initial canonical-state hashes, the initial authoritative runtime hash, the
replay-manifest root, and its final chain cursor. Match-verification format 2
adds the runtime-profile field and rejects format 1. The runtime hash commits to
host-only state needed to reproduce future observations and randomness,
including team assignment, inboxes, lineages, decision sequences, adapter
defaults, and a one-way commitment to the private match secret.

The server reconstructs that initial frontier and invokes the submitted Mind
artifact itself. For WASM Minds this uses the same pristine-instance path as a
normal reference game. Each resulting normalized commitment list and complete
resolution report is rebuilt into the canonical event chain and compared with
the submitted segment before proceeding. Verification operates one bounded
segment at a time, so neither the server nor a browser needs the entire match in
memory.

Only completion of every segment produces a `VerifiedMatchManifest` token. The
Ed25519 signing API accepts that token rather than a merely parseable manifest,
making it difficult for service code to attest a structurally valid but
unexecuted client claim. The signed envelope carries a public-key-derived key ID
and can be checked independently by leaderboard and replay clients.

### 13.5 Browser replay boundary

The browser package parses the compact replay manifest, resolves a global event
to one independently fetchable content-addressed segment, restores that
segment's checkpoint, and re-executes only a caller-bounded prefix. Segment
descriptors and hashes are exposed before the payload is fetched; render loops
can then read dense tile arrays and compact cell metadata without copying the
entire match into browser memory. Each subsequent step exposes the sparse tile
and cell keys in its canonical delta, allowing incremental rendering instead of
copying every dense array per event.

Attestation parsing, key-ID derivation, signing-message construction, and wire
encoding live in the WASM-compatible resolver library and are shared with the
native server signer. Native code uses `ring` for Ed25519; the browser wrapper
passes the exact parsed signature and domain-separated message to WebCrypto.
The wrapper also compares the signed replay-manifest hash with the replay being
displayed. The expected server public key still needs an authenticated
distribution channel; accepting a key delivered beside an untrusted replay
would authenticate only the submitter's own key.

Scores are not trusted client fields. A leaderboard service must derive them
from the server-verified terminal state under a frozen season/scoring version,
then atomically associate that derived score, the attestation, and replay root.

### 13.6 Online service state machine

Submission admission accepts exact artifact bytes rather than caller-supplied
artifact hashes. It bounds per-artifact and aggregate bytes before retaining
them, requires strictly ordered unique competition slots, structurally validates
each core WebAssembly module against `extism_pdk_deterministic_v1`, derives every
artifact hash itself, and checks the verification and replay manifests against
the server-generated match ID hash, frozen Mind ABI, frozen runtime profile,
actual initial runtime hash, and exact artifact bindings. The service-visible
match ID is domain-separated envelope metadata and remains absent from every
Mind invocation.

Runtime profile `extism_pdk_deterministic_v1` requires the exact
`reference_mind_function: () -> i32` export and admits only typed Extism input,
bounded byte-arena allocation/free/length, load/store, output, and error
functions. All imports must be functions under `extism:host/env`; WASI,
configuration, variables, HTTP, logging, custom host functions, imported
memories/tables/globals, components, shared or 64-bit memory/table indexes, and
custom memory pages are rejected before compilation. Module-defined mutable
state and start functions remain legal because every decision receives a fresh
instance. A module may also export the conventional reactor initializer
`_initialize: () -> ()`; when present, the host invokes it under the decision
deadline inside every fresh instance before `reference_mind_function`. This is
required by TinyGo's non-WASI reactor output and never creates persistent guest
state. The profile caps declared memories and tables to the executor's resource
limits. Executor brand, allocator, worker count, and scheduling are host policy
and do not enter the manifest.

After the execution host instantiates those accepted artifacts, the stored
verification session consumes only trusted server-produced completion batches.
It loads one immutable content-addressed segment at a time, advances only after
the preceding segment is complete, and returns the existing
`VerifiedMatchManifest` type-state token only after the entire replay root has
been reproduced.

A scoring policy receives the verified terminal `ReferenceSimulation` and
explicitly trusted server context. Before invoking it, the service compares the
live compiled-ruleset and terminal-state hashes with the verified manifest's
final cursor. The policy returns an integer score and a frozen policy hash;
floating-point values and client-provided score claims never enter the
publication contract.

The leaderboard publication is separately Ed25519-signed and binds the signing
key ID, season hash, score-policy hash, integer score, terminal-state hash, and
complete signed match attestation. Its append-only filesystem store verifies
both signatures and the referenced replay manifest before making the whole
record visible with one atomic hard link. Concurrent duplicate publication is
idempotent; a different record for the same season and verified manifest fails
closed. Leaderboard scans are bounded and sort equal scores by verification
manifest hash, not arrival or filesystem iteration order.

## 14. Ruleset boundary

Safe parameter or formula-kernel variation includes:

- mass-to-duration and mass-to-cost curves;
- effort tiers and recovery times;
- attack payload, retained-mass leverage, and scatter fractions;
- digestion, bite, gut capacity, metabolism, plant growth, and diffusion rates;
- collision arbitration (`all_collide` versus a bounded physical-impulse rule);
- enabled action families and signal limits.

The following require a versioned semantic kernel or ABI because they change the
causal model:

- allowing same-batch vacancy or resource chaining;
- target identity tracking instead of coordinate targeting;
- unbounded action range or footprint;
- action cancellation by same-time damage;
- exposing authoritative kinship or global information;
- group-atomic decisions;
- non-local energy reservoirs or communication.

Leaderboard seasons freeze all kernels, parameters, artifacts, world-generation
rules, and scoring rules under a canonical hash.

`ReferenceRuleset` is the strict serializable physics profile. Hosted callers
must choose their boundary policy and mass thresholds explicitly; the engine
constructor does not rewrite them. RL TOML accepts partial `[env.rules]`
overrides, expands them over its documented hosted default before execution,
and stores the complete expanded profile plus compiled hash in immutable
artifacts. Reward shaping is intentionally not part of the physics hash.

## 15. Confirmed reference semantics and remaining parameters

The following choices are accepted for the first reference ruleset:

1. **Movement:** snapshot-gated occupancy, including failed swaps and only one
   convoy step per decision wave.
2. **Exact collision:** symmetric `all_collide`, with physical impulse
   arbitration available as a later ruleset kernel.
3. **Same-time death:** every action due from a cell alive in `S_t` completes,
   followed by death cleanup.
4. **Attack targeting:** a coordinate is fixed at action start; at completion the
   attack affects the occupant of that coordinate in `S_t`, if any, rather than
   tracking a cell identity.
5. **Failure costs:** effort is always spent; non-launched split/regurgitate
   payload returns on frustration; a launched payload is never recalled.
6. **Guard:** posture activates at commitment and persists until a different
   primary action is committed. Guard has the minimum primary-action duration.
7. **Terrain:** deposition occurs at the actor's current location.
8. **Pushing:** omitted. A future pushing kernel must have a fixed propagation
   limit or an energy-decreasing bound.
9. **Consumption:** `Consume(amount)` takes plant energy first and then loose
   energy from the actor's current tile, capped by the requested amount, bite
   capacity, remaining gut capacity, and the completion snapshot. Same-batch
   deposits are not consumable.
10. **Digestion and regurgitation:** digestion transfers gut energy into
    assimilated energy at a fixed time-based rate with an authoritative
    fractional remainder. `Regurgitate` escrows gut energy, deposits it as loose
    energy on a reachable local target, and returns the payload to the gut on
    frustration.
11. **Plant growth:** each growing plant has a canonical capacity, rate, and
    fractional remainder. Simulated-time growth converts diffuse energy on the
    same tile into plant energy, so it neither creates energy nor depends on
    host-only metadata. Progress is not banked while the source is empty or the
    plant is full.
12. **Metabolism:** every living cell pays a fixed simulated-time upkeep rate
    from assimilated energy. Spent energy becomes diffuse energy at the cell's
    current tile, with a canonical fractional remainder, so metabolism is
    partition-invariant and energy-conserving. Metabolic exhaustion is an
    explicit event: exhaustion before a pending completion interrupts that
    action, while an exact tie permits the action to complete before death
    cleanup. Passive ordering is plant growth, digestion, metabolism, signal
    decay, then a due diffusion step; therefore digestion can sustain a cell at
    an exhaustion boundary and newly diffuse energy cannot feed plant growth
    until a later interval.
13. **Terrain/material:** `Excavate` lowers the actor's current tile by one
    elevation unit and transfers the hashed `terrain_mass_per_elevation` into
    carried material. `DepositTerrain` escrows that same carried mass and raises
    the actor's current tile by one unit. Signed elevation bounds reject the
    action, and interrupted material escrow joins the ordinary local death
    spill. Elevation-derived terrain mass is included in the conservation
    ledger.
14. **Signals:** `signal_emission_cost` is a conserved emission quantum, not a
    fixed per-action fee. A decision may deposit any positive multiple on one
    anonymous channel alongside a non-Signal primary action, or use the
    explicit `Signal([amount; 4])` primary action to deposit independent
    strengths atomically on all four channels. The two forms cannot be combined.
    Their total amount is paid first from assimilated energy and placed on the
    actor's current tile; a sidecar's primary action is then validated against
    the remaining energy. Signal energy decays at a fixed simulated-time rate
    into local diffuse energy. Successful excavation or terrain deposition
    clears every channel and decay remainder on that tile, transferring the
    erased signal energy to local diffuse energy. Observations expose only
    locally masked channel strengths, never sender identity or an inbox.
15. **Diffusion:** diffuse energy moves on hashed `diffusion_targets` at fixed
    absolute-time intervals. Every step reads one immutable field snapshot and
    distributes equal integer shares to canonical distinct neighbors; an
    indivisible remainder stays at its source, and sub-share reservoirs do not
    schedule no-op events or bank fractional progress. Diffusion is therefore
    conservative, local, independent of action iteration order, and invariant
    to unrelated event-time partitioning.

Parameters still to be frozen include the time scale and decision-rate floor,
the number of apparent-mass and action-progress observation buckets and the
guard damage/cost formula. Bite capacity, gut capacity, digestion rate, the
fixed metabolism rate, terrain conversion, signal decay, diffusion, and action
costs are implemented as provisional ruleset parameters and still need gameplay
tuning. Mass-dependent metabolism remains a possible versioned formula
variant, not an unstated property of the initial fixed-rate kernel.

## 16. Migration and validation sequence

The migration is now through the parallel claim/component stage:

1. Introduce normalized intents, immutable resolution snapshots, terminal
   outcome events, and delta buffers while retaining synchronous ticks.
2. Add executable tests for every thought experiment in section 10, including
   reversed input order and varied worker counts.
3. Replace global sorting with claim buckets, commutative reducers, and the
   reference occupancy collision rule.
4. Add state hashes and assert identical results for serial, parallel, native,
   and WASM-compatible core execution.
5. Introduce fixed-point event time, pending actions, effort/payload escrow, and
   passive time-based processes.
6. Add the potential broad phase and incremental due-event claim registry only
   after profiling shows their maintenance cost is justified.
7. Benchmark adversarial layouts: full convoys, checkerboard collisions, attack
   stars, dense split attempts, and hot signal/energy tiles.

Steps 1–5 are complete. Step 6 now includes an exact derived due-action index;
the broader potential-interaction graph remains deferred until profiling shows
that maintaining it beats direct local validation. The serial reference
resolver remains the oracle for every optimized resolver.

### 16.1 Conformance gate

The first conformance gate is executable through `scripts/conformance.sh`. It
runs the reference thought experiments and golden vectors, then adds seeded
multi-batch worlds whose actions are committed in forward and reverse order.
Every generated batch must produce identical reports, states, and hashes;
conserve total energy-equivalent; preserve the tile/cell occupancy bijection;
and reverse/reapply its sparse delta exactly.

The gate also treats replay input as hostile. It rejects every truncated prefix
of representative frames and archives, enforces every nested collection limit,
rejects duplicate or noncanonical event resources, exercises malformed deltas,
and feeds a deterministic garbage corpus to both bounded decoders. It also
checks the strict Mind ABI directly: changing engine-private cell IDs, team IDs,
lineages, or decision counters cannot change serialized input when the bounded
observation, private memory, and supplied random samples are equal. The schema
contains no public identity, coordinate, tick, or shared-seed field and rejects
random sample payloads of the wrong size.

Finally, the gate strictly lints the reference library for the WASM target,
rebuilds the exact WASM artifacts used by integration tests, and compares full
projected and canonical state across worker counts. A malicious stateful WASM canary returns a
different action after its first invocation; it must nevertheless return its
first-invocation result for every cell because each call receives a pristine
store and instance. The compatible pool shares one immutable compiled core;
stock Extism caches a worker-local compiled descriptor. Neither recompiles per
decision. Native engine tests likewise compare full state with one and four
mind workers using cell-scoped random samples.

The reference replay driver now uses the same WASM-compatible resolver core for
native and browser builds. The `blob_web` package exposes bounded manifest and
attestation parsing, content-addressed segment lookup, checkpoint seek,
authoritative stepping, and render-oriented state access. Conformance builds an
optimized `wasm32-unknown-unknown` module and native package tests reproduce
canonical hashes across checkpoint and segment boundaries. An automated test
that instantiates the generated module in an actual browser runtime is still
needed. Browser runs remain non-authoritative regardless: the server must invoke
the submitted Mind itself and reproduce both its commitments and the canonical
resolver hashes.

## 17. Implementation status

The first isolated reference milestone is implemented under
`blob_engine::resolution`:

- `neighborhood.rs` validates bounded local offsets, observation masks,
  per-action target masks, diagonal corner policy, boundary policy, and dense
  precomputed neighbor tables;
- `reference.rs` defines fixed-point event time, inertial duration/effort
  parameters, action escrow, immediate guard posture, local activity cues,
  terminal outcomes, resource claims, snapshot validation, symmetric occupancy
  contention, deterministic commit, death interruption, and conservation
  accounting for Move, Split, Guard, Wait, Attack, Consume, and Regurgitate. It
  also applies partition-invariant time-based digestion, fixed-rate metabolism,
  signal decay, synchronous diffuse transport, and conservative plant growth,
  plus plant-first bounded consumption, gut capacity, regurgitation escrow, and
  terrain/material conversion. Metabolic exhaustion and diffusion steps
  participate in the event queue. Native resolution reuses the reversible
  all tiles plus only due actors and their target occupants as its immutable
  completion snapshot, validates large intent sets in ordered parallel work,
  journals first-write before-values for exact rollback and reversible deltas,
  indexes exclusive occupancy claims in dense tile buckets, and
  resolves the current one-resource components without locks. A derived exact
  completion index avoids scanning every cell for each event. Growing-plant
  indexing keeps passive updates proportional to configured plant tiles rather
  than total board area. Exact derived signal and diffuse-energy frontiers do
  the same for sparse fields; synchronous diffusion uses active sources and a
  reusable incoming buffer with sparse clearing. Derived active-digestion,
  active-metabolism, and zero-energy sets similarly drive passive cell work,
  exhaustion scheduling, and death cleanup, with direct-map traversal retained
  for dense populations. Disabled digestion and metabolism rates enforce zero
  canonical remainder and take O(1) passive fast paths. Canonical private
  memory is immutable copy-on-write
  storage internally, so snapshots and deltas share unchanged cold bytes; the
  Mind boundary still receives an owned copy. A Mind may explicitly return
  `Retain` to avoid sending those unchanged bytes back, but never receives a
  reference to canonical storage. None of these indexes or scratch
  buffers enter authoritative state;
- `tests/reference_resolution.rs` makes the thought experiments executable and
  checks commit-order independence and energy conservation;
- completion validation reads the immutable canonical cell map before any cell
  writes and retains only action-specific snapshot facts needed later. The tile
  snapshot still gates simultaneous occupancy, terrain, and bounded-energy
  behavior. Ordered due-cell journal capture and pending-action completion share
  one linear traversal; sparse extra writes merge back into canonical delta
  order. These structures do not enter hashes, checkpoints, or replay;
- `tests/resolution_conformance.rs` runs deterministic generated multi-batch
  worlds through the serial oracle and checks order independence, conservation,
  state invariants, hashes, and reversible deltas;
- the native host combines ready actors and returned Mind decisions with a
  canonical linear merge, filling absent decisions with exact Wait commitments
  without exposing peer decisions or scheduling information. Host-only phase
  telemetry remains outside canonical state and replay. The compatibility host
  projection may tag a pure Wait as host-visible-no-op when its isolated input
  proves private memory is unchanged and no signal/guard transition occurs.
  Canonical deltas and passive invalidations override that hint for attacks,
  metabolism, movement, death, or any other visible change. The hint is neither
  observable by a Mind nor serialized;
- `tests/parallel_resolution.rs` checks byte-identical reports, deltas, and
  states under one, two, and four resolver workers, exposes component telemetry,
  and provides an ignored release throughput diagnostic for threshold tuning;
- `hashing.rs` defines the versioned SHA-256 canonical encoding, semantic and
  compiled ruleset hashes, and authoritative state hashes. Hash format 6 uses
  domain-separated cell leaves and separately cached private-memory
  commitments, with sparse dirty-leaf updates and a dense parallel crossover.
  The live cache uses derived key-indexed commitment slots, and dense workers
  produce leaf entries and their canonical page hash in one pass. This layout
  is non-authoritative and never enters checkpoints or replay; native and WASM
  SHA-256 backends are pinned to identical bytes by the from-scratch oracle and
  golden vectors. Completion reports carry the
  compiled ruleset and post-commit state hashes, and golden-vector
  tests pin the encoding;
- `replay.rs` defines bounded canonical batch frames, a genesis/event hash
  chain, replay recording, structural verification, stable-order validation,
  explicit rejection of tampering, reordering, and chain splicing, plus a
  bounded canonical archive with constant-time indexed frame lookup. Version 8
  records the exact accepted or rejected decision commitments at each boundary,
  including the tagged private-memory update, escrow, and scheduled completion
  receipts;
- `replay_bundle.rs` pairs that archive with sparse, independently decodable
  canonical checkpoints. Each checkpoint is keyed by its applied-event count
  and must match the archive's ruleset and the state hash at that exact chain
  position. The versioned bundle has bounded decoding, its own integrity hash,
  and logarithmic nearest-checkpoint seek;
- `replay_segment.rs` defines bounded content-addressed stream segments. A
  segment carries its global chain cursor, canonical starting checkpoint,
  indexed event payload, and ending cursor. Compact manifests chain segment
  descriptors by sequence, event hash, state hash, and completion time, so
  segment bytes can live in object storage and be fetched independently;
- `replay_driver.rs` restores a segment checkpoint and deterministically
  re-executes its commitment stream through the reference resolver. It compares
  every commit receipt, batch report, delta, and state hash, supports prefix
  playback and direct checkpoint seek, and carries actions that begin in one
  segment and complete in another without inventing a segment-boundary action;
- `match_manifest.rs` defines the bounded canonical match-verification envelope
  binding exact Mind artifacts, ABI, and runtime profile to the initial runtime,
  compiled rules, replay root, and final cursor. Artifact slots are
  server-envelope metadata and remain absent from every Mind input;
- `attestation.rs` owns the WASM-compatible bounded attestation wire format,
  key-ID derivation, and domain-separated signing message. Native signing and
  browser verification therefore cannot drift to independent encodings;
- native `server_verification.rs` compares server-generated commitments and
  reports one segment at a time. Its type-state completion token gates Ed25519
  attestation, and the `blob_game` verification tick exposes the same data after
  pristine-instance execution of an actual submitted WASM module;
- native `mind_runtime.rs` validates untrusted core-WASM structure, exact typed
  imports/exports, and resource shape against the manifest-bound runtime profile;
- native `online_service.rs` bounds, validates, and re-hashes submitted artifact bytes,
  verifies one stored replay segment at a time, gates score derivation on the
  verified live terminal state, creates a separately signed canonical score
  publication, and atomically installs complete replay/attestation/score
  associations in an append-only leaderboard store;
- `blob_web` provides the read-only browser package boundary. It can seek into
  independently fetched replay segments, re-execute their commitment stream,
  expose tile/cell snapshots, and provide exact Ed25519 inputs to WebCrypto. It
  contains neither a signing path nor authority to publish leaderboard data;
- `tests/replay_conformance.rs` exercises truncation, allocation limits,
  duplicate resources, malformed deltas, empty archives, and deterministic
  arbitrary decoder input. Canonical event collections require strict ordering;
- `delta.rs` defines sparse canonical before/after tile and cell deltas with
  atomic source validation and exact forward/backward application. Batch
  reports and replay frames carry these deltas;
- the resolution module compiles for `wasm32-unknown-unknown`; native-only host
  modules and dependencies are excluded from that target;
- the canonical Mind v8 ABI contains only one cell's bounded local projection
  and 32 bytes of private random output. It carries canonical
  gut/metabolism/outcome/activity/action-space data plus locally observable
  plant, terrain, and anonymous signal fields, and returns an exact reference
  action, optional signal, and an explicit `Retain` or `Replace` private-memory
  operation. Public cell/team
  identity and shared randomness are absent;
- decision randomness is derived from a private match secret, engine-private
  lineage, and per-cell invocation sequence. Online constructors accept an
  independently generated server secret; deterministic local constructors
  derive one from their configured world seed;
- RL observations are the same canonical anonymous `ReferenceMindInput` used by
  WASM Minds, padded to 32 local slots (1094 finite features). The policy uses
  conditional heads for 264 physical action/effort/slot choices, five bounded
  payload/amount tiers, 16 four-bit channel patterns, and five quantized signal
  strengths. Ordinary actions are masked to no signal or a one-channel pattern;
  explicit Signal may select any nonempty channel subset. The ABI itself permits
  independent channel amounts. Packed
  per-observation masks make every conditional choice commit-legal without
  constructing their Cartesian product. PPO and behavior cloning sum only the
  selected factors' log probabilities. Engine-private
  `CellId` values are host handles only and never enter the observation;
- native minds are reset before every decision, while untrusted WASM decisions
  use fresh stores/instances created from per-worker compiled modules. Exact
  worker-parity and stateful-isolation canaries are part of the conformance gate.
  Native hosts may retain allocation capacity in a per-Mind observation scratch
  buffer, but every semantic field, local slot, private-memory byte, and private
  random sample is replaced before invocation. Submitted-WASM inputs remain
  independently serialized buffers, so this allocation optimization does not
  create a communication channel between cells. When multiple pristine Mind
  instances exist, a host may run their actor-ordered chunks concurrently;
  every chunk owns one instance and scratch buffer, derives randomness only for
  its assigned cells, and returns owned decisions that are canonically sorted
  before commitment. Worker count and scheduling remain outside hashes and
  replay. A submitted-WASM pool may keep more compiled workers than it activates
  for a small frontier; inactive workers receive no input or cell state;

The reference resolver now has an explicit authoritative-driver boundary:

- `Engine` is a non-generic canonical runtime. It owns canonical time, pending
  actions, gut contents, outcomes, and the canonical state hash; accepts only
  `ReferenceMind` instances or exact `ReferenceMindDecision` overrides; and has
  no alternate resolver, action adapter, or direction-only action path;
- a ready frontier is installed through one strictly actor-ordered canonical
  commitment batch. Fatal ABI errors, invalid signals, unknown/unready actors,
  ordering violations, timestamp overflow, or resource-field overflow are
  preflighted before any mutation. Ordinary semantic rejection is not fatal:
  it still installs the specified minimum Wait and private-memory update. Batch
  grouping and dense dirty-cache invalidation are derived execution details and
  never enter replay semantics;
- the RL environment uses reference resolution by default, requests actions
  only from cells that are ready at the canonical event time, and advances
  autonomous opponent-only batches until the training team reaches its next
  decision frontier. This prevents variable-duration actions from stalling a
  rollout on an empty observation set;
- CLI games accept only exact action/private-memory decisions through the
  reference ABI while retaining fresh-instance WASM isolation. A missing
  export, trap, malformed output, or noncanonical action fails closed and rolls
  back private decision sequences;
- maintained example Minds all export the reference ABI directly. Terrain,
  messaging, and pheromone actions are deliberately absent until their
  canonical semantics exist; they are not silently converted into Wait;
- the `World`/`Cell` view is a projection for UI and host integration.
  Plant and loose energy share one projected tile value, but remain distinct in
  canonical state. Plant rate, capacity, and fractional progress are canonical
  rather than host-side profiles. The authoritative runtime hash additionally
  commits the private host metadata that can affect later decisions;
- trusted/server hosts may select metadata-only execution before canonical
  initialization. This retains isolated Mind dispatch, hashes, commitments,
  replay, team inheritance, and private randomness counters, but does not copy
  batch changes into renderer-shaped cells, coordinate maps, or world arrays;
- canonical checkpoints now have a versioned, platform-independent byte format
  with bounded decoding and an integrity hash. Restoration validates the exact
  ruleset, dimensions, occupancy, event time, pending actions, gut contents,
  state hash, and all allocation limits;
- live game saves embed that canonical checkpoint in a versioned host envelope
  alongside private team and decision-randomness state. Restore
  rebuilds every core-derived projection field and verifies the authoritative
  runtime hash before another mind can run. Host checkpoint version 11 is
  intentionally forward-only; older envelopes and unversioned saves are
  rejected;
- reference-mode Engine and Game callers can opt into live replay recording
  with a fixed checkpoint interval. Exports always include a verified tail
  checkpoint, while resumed recordings discard incidental export boundaries so
  the eventual replay is byte-identical to an uninterrupted run. Recording
  state is carried through live game saves, and worker-assignment parity is
  tested for the complete bundle bytes;
- online callers can instead opt into bounded replay streaming. The engine
  rotates on configured event-count or encoded-byte thresholds, retains only
  the active suffix plus at most one sealed segment, and exposes ownership-based
  draining. If that segment is not drained, the next tick is rejected before a
  mind runs, private randomness advances, or authoritative state changes. Game
  checkpoint version 11 preserves the manifest, active suffix, rotation limits,
  and an optional undrained segment; exact save/resume output is tested;
- native hosts can persist drained artifacts through `FileReplayStore`.
  Segments and manifests are immutable objects addressed by their canonical
  hashes. Writes are bounded, revalidated under store-local limits, fsynced,
  and installed idempotently. Per-match publication verifies newly appended
  segment objects, rejects history rewrites, and updates an integrity-protected
  `CURRENT` pointer with compare-and-swap semantics under a cross-process lock.
  Failed publication leaves the prior pointer valid; an unreferenced immutable
  object can be safely retried or garbage-collected later.

The checkpoint index locates a valid restart state and the following event
range. Applying only stored resolution deltas would skip commitments and passive
time, so authoritative playback instead restores that checkpoint and runs the
version-6 commitment stream through `ReferenceReplayDriver`. Commitments made
before a segment boundary are already represented by pending actions in the
next segment checkpoint; they are not duplicated in the later segment.

This physics replay proves that a commitment stream reproduces its claimed
canonical states. It does not prove that an untrusted client obtained those
commitments by invoking the submitted Mind. Leaderboard verification must also
run the frozen Mind artifact with server-derived private randomness and compare
its normalized commitments to this stream before a result becomes trusted.

`FileReplayStore` assumes a server-controlled filesystem with POSIX-style
atomic rename and hard-link behavior. Its directory lock deliberately fails
closed; an abnormal process exit while publishing may require removal of that
match's stale `.publish-lock` directory. A database or object-store adapter
should use that system's native conditional-write/lease primitive instead of
copying this locking mechanism.

`FileLeaderboardStore` has the same server-controlled POSIX-volume assumption.
It does not maintain a mutable rank index: immutable signed entries are the
source of truth and the initial implementation performs a bounded scan and
deterministic sort. A production database adapter should atomically insert a
unique `(season_hash, verification_manifest_hash)` record and maintain ranking
indexes transactionally, while preserving the signed publication bytes as the
portable audit record.

The native parallel claim/component executor is active for large batches, while
the serial resolver remains the replay and browser oracle. Online match setup
binds each submitted artifact to both the canonical ABI hash and
`extism_pdk_deterministic_v1` in the frozen verification manifest.

The next online slice is a transport and worker layer over this
framework-neutral service core: authenticated upload endpoints, job lifecycle
and cancellation, compilation, wall-clock/fuel/memory quotas,
and production season configuration. It also needs authenticated signing-key
distribution, a database/object-store adapter, retention/garbage collection
for unreferenced objects, and a browser parser for the signed leaderboard
publication. The browser path still needs a real-runtime integration test.

Further optimization should be driven by the telemetry and adversarial
benchmarks. The first measured baseline is recorded in
`docs/performance-baseline.md`, and the ordered implementation plan is in
`docs/large-game-optimization-roadmap.md`. The baseline shows that the large-batch resolver is no
longer the dominant end-to-end cost. The next targets are intent-validation and
completion-snapshot overhead, a strict-isolation-compatible alternative
to constructing a fresh WASM store for every decision, and large passive-field
stencils. Parallelism should normally be applied across independent matches;
small fragmented action batches do not currently repay a large per-match
worker pool. Commutative attack reduction remains useful for adversarial dense
batches, while a union-find graph is unnecessary until a ruleset adds an
exclusive action that claims more than one resource. Current numerical physics,
terrain, signal, diffusion, and metabolic defaults remain provisional testable
parameters rather than a frozen leaderboard ruleset.
