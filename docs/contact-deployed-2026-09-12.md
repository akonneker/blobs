# Deployed combat evaluation — 2026-09-12

**The frozen feeding candidates fail the existing combat gate.** Across 168
episodes, the parent and all three learned control/treatment pairs select zero
attacks, deal zero damage and score zero kills. This is not a safety abort or a
missing legal-action opportunity: every episode has a legal attack on an
observable occupied target. The maintained aggressive baseline passes the same
suite with **526 attacks, 2,508 applied damage and 22 resolver-attributed kills**.

The [fresh-development feeding result](fresh-feeding-2026-09-12.md) remains valid.
Four rerun feeding sentinel episodes reproduce their archived trial reports
exactly. Combat acquisition is now the blocking skill; self-play remains gated.

## Implementation and frozen comparison

`contact_deployed::evaluate_contact_mind` evaluates ordinary `ReferenceMind`
calls through the canonical environment entry point. It uses the existing
contact/skirmish environment configuration, metric aggregation, report validator
and activity/kill thresholds. It does not use the sustained-feeding boundary or
the older Burn greedy decoder.

Schema-1 deployed contact evidence includes initial/final state hashes, initial
private-decision frontier hashes, effective environment/rules identities,
decision kinds and legality, full attack/movement/consume telemetry, terminal
reason, elapsed quanta, host-step budget, applied damage and attributed kills.
It embeds the existing schema-3 contact report. No policy receives host identity
or diagnostic counters.

`deployed_artifact` now shares bounded export loading between feeding and combat
commands. It checks manifest and weight hashes, format, execution contract,
Mind ABI and confirmation reservations before creating a trial directory.
Empty or duplicate evaluation seeds and malformed confirmation ledgers fail.

The pre-run plan fixes all seven exported Minds, the unchanged configuration
SHA-256 `4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9`,
and seeds **1435500201/1435500202**. These seeds were already exposed by feeding;
this is combat development characterization, not another fresh cohort.
Final confirmation seeds **1434999901/1434999902 remain reserved**.

Each Mind runs 24 episodes: contact 1v1 and skirmish 4v4, starting energies
60/100/180, aggressive/defensive opponents, and two seeds. Deadlines are 32,768
and 65,536 quanta respectively; the configured host cap is 65,536 steps.
The canonical rules end exterminations normally. All candidates run even when
earlier candidates fail.

The existing combat gate requires at least one successful damaging attack in
each population, no safety-aborted variants, and at least one skirmish kill
(the contact kill floor is zero). These are modest acquisition thresholds,
not comprehensive combat mastery. All frozen candidates fail them.

## Results and diagnosis

Every frozen Mind has zero wins, attacks, applied damage and kills. There are
no illegal decisions or safety aborts. Parent/control episodes have 12 losses
and 12 timeouts per Mind. Treatments 301 and 303 have 14 losses and 10 timeouts;
treatment 302 has 12 losses and 12 timeouts. Thus the Move-target change does
not establish combat retention or competence.

After the failure, a read-only diagnostic wrapper reran the frozen policies and
the maintained aggressive baseline. Every complete frozen-policy evaluation
record—including final states, telemetry and outcomes—matches the original
combat comparison exactly. The wrapper returns each Mind's decision unchanged.

| Mind | Decisions | Legal occupied-target opportunities | Moves on those opportunities | Attacks |
| --- | ---: | ---: | ---: | ---: |
| Parent | 1,548 | 1,122 | 1,122 | 0 |
| Control 301 | 1,553 | 1,108 | 1,108 | 0 |
| Control 302 | 1,553 | 1,108 | 1,108 | 0 |
| Control 303 | 1,548 | 1,122 | 1,122 | 0 |
| Treatment 301 | 1,599 | 800 | 800 | 0 |
| Treatment 302 | 1,608 | 560 | 560 | 0 |
| Treatment 303 | 1,611 | 496 | 496 | 0 |
| Aggressive baseline | 1,134 | 1,066 | 0 | 526 |

An opportunity means that an observable occupied slot admits a payload-one
attack at some supported effort under the existing commit-legality check,
including concurrent-signal reservation. The anonymous input does not expose
team identity, and legality does not imply tactical value. The baseline's
applied damage and attributed kills independently demonstrate that this exact
suite permits productive attacks.

The first diagnostic counted all legal targets, including potentially empty
tiles. It is retained as version 1. Version 2 adds explicit occupied-target
counters and supports the table above. Neither diagnostic adds an independent
candidate cohort: both repeat the same 168 frozen-policy episodes, plus the
same 24 baseline episodes.

The current composite changes only Move targets and preserves the parent's
action kind on the same input. That adapter cannot directly teach Attack
selection. The subsequent [raw-score diagnosis and controlled head fit](interaction-head-2026-09-12.md) identifies a strong Move preference in the inherited interaction
base head. A head-only fit learns some Attack choices but fails feeding
retention in all three runs. The subsequent [local-context adapter study](interaction-slot-adapter-2026-09-12.md) improves agreement but still fails
retention and exposes information lost by component-wise slot maxima. No failed
fit is promoted; the next residual must preserve per-slot feature relationships.

## Evidence and validation

- `training-output/contact-deployed-2026-09-12-v1`: frozen plan, exact combat and
  feeding executables, source snapshots, 168 episodes, four feeding sentinels,
  checked summary and report hashes.
- `training-output/contact-opportunities-2026-09-12-v2`: diagnostic executable,
  source/plan hashes, full records, opportunity counters and positive control.
  Version 1 remains archived separately.

```sh
python3 -B scripts/verify_contact_deployed.py training-output/contact-deployed-2026-09-12-v1
python3 -B scripts/verify_contact_opportunities.py training-output/contact-opportunities-2026-09-12-v2
```

The combat verifier recomputes raw metrics, rates, stage gates and terminal
classification; it checks initial pairing across all seven Minds, inherited
feeding identities and exact sentinel reproduction. The opportunity verifier
checks provenance, counter arithmetic, paired initial inputs and unchanged full
candidate evaluation records. Neither claims a full archived per-event replay
or new WASM qualification. All seven Minds execute in the native portable runtime.

Validation passes **289 RL library tests (four ignored)** and all-target,
all-feature RL Clippy with warnings denied. Four new tests cover canonical
baseline state/metric equivalence, passive-Mind rejection, altered deployment
identities/weights and seed reservation/shape checks. Formatting and the
**65-constant** schema registry pass.

## Next slice

Inspect the inherited routing and action-kind scores on these actual occupied
contact observations, then collect paired combat examples using the maintained
baseline and the policy's own recurrent prefixes. Fit a narrowly scoped combat
action-selection intervention with matched controls and explicit feeding
retention data. Requalify the deployed policy on both feeding and combat before
self-play; do not tune only the Move-target utility or consume final
confirmation seeds. Carry the supplemental development exposure ledger forward.
