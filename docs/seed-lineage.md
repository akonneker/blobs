# Behavior-cloning seed lineage

Cloning schema 36 introduced a cumulative `seed_ledger` containing training,
validation, and reserved confirmation seeds. Each child embeds the verified
parent ledger in its configuration and extends it with its own corpus roles.
The parent artifact hash binds that inheritance; paths are not part of identity.
Current schema 37 retains this contract while adding the target residual.

## New runs and descendants

Use `--confirmation-seed 900,901` (or repeat the flag) to reserve an untouched
confirmation cohort when creating a root policy. Choose actual seeds for the
experiment; the example values are not a project-wide reserved suite. Reservations
are optional for diagnostic runs, but declare them before a promotion experiment.

An ordinary `--initial-behavior-clone` import automatically carries the parent's
ledger. Historical roles remain fixed even when the child changes `--seed`,
`--validation-fraction`, dataset order, eligible-row filtering, or corpus membership.
The fraction controls how many new seeds fill the requested validation share;
it cannot move a historical seed to a different role. Actual counts may therefore
differ from the requested ratio. Empty required partitions are rejected rather
than repaired by reassigning old seeds.

Removing a seed from one stage does not remove it from lineage. Adding new
confirmation reservations is allowed only for seeds absent from training and
validation history. Confirmation seeds cannot appear in any input demonstration
manifest or sample, including rows filtered out by `--exact-round-trip-only`.
Disabling validation rejects a corpus containing an inherited validation seed.

Publication checks the reported partitions against the actual input corpora.
Loading schema-36-or-later metadata checks the cumulative ledger against its inherited
roles and recorded training/validation seed lists. The ledger is capped at
65,536 seeds and the full metadata at 1 MiB; oversized artifacts fail publication.

## Existing policies

Schemas 22–29 and 31–35 remain available for evaluation through the shared model
loader. They cannot be ordinary training parents because their metadata does not
prove a complete seed history.

To bridge one into the new format, explicitly add these options to the ordinary
behavior-cloning command:

```text
--initial-behavior-clone /path/to/legacy-parent
--audit-legacy-lineage
--lineage-artifact /path/to/earlier-ancestor
--lineage-dataset /path/to/original-corpus
```

Repeat the ancestor and corpus flags as needed. Supply every ancestor named by
an `initial_artifact_sha256` link, until reaching a root or an existing schema-36
ledger, and every original corpus used by the pre-ledger artifacts. The immediate
parent is already supplied by `--initial-behavior-clone`. The audit verifies one corpus at a time, retaining only its manifest, and checks
artifact/model hashes and demonstration manifest/payload hashes; it rejects
missing evidence, cycles, and contradictory roles. Audit artifact and corpus
hashes are retained in the child's metadata.

Every historical corpus seed is conservatively classified as exposed training
history, including old validation seeds and filtered samples. This avoids
repeating the old cross-corpus split leak or treating model-selection evidence
as new validation. The bridge therefore needs new validation seeds in its current
training corpora. The bridge is a new immutable artifact; historical files and
reported results are not rewritten or retroactively qualified.

## Scope

This ledger guards declared behavior-cloning ancestry and demonstration inputs.
It is not a global experiment registry: it cannot detect undeclared weights,
manual model selection, outside evaluations, or training performed by another
pipeline. In particular, reserve final confirmation seeds before selection and
use them only for the final evaluation; repeated confirmation runs can still
influence human decisions. PPO-wide provenance and a project-wide record of final
confirmation use remain follow-up work.

## Validation

The RL suite passed 294 tests (four existing ignored tests). New integration
coverage trains three generations with changing corpora and split settings,
checks inherited confirmation reservations and filtered-row rejection, rejects
ledger tampering and missing historical evidence, and bridges a complete legacy
chain using fresh validation seeds. The audit tests were rerun after the final
change to release each historical corpus payload before reading the next one.
CLI checks cover the new options and reject an audit without a parent.
