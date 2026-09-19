# Format and schema registry

`registry.json` is the machine-checked inventory of current Rust-owned wire,
storage, replay, verification, RL-artifact, telemetry, and browser API
versions. `scripts/check_schema_registry.py` enforces three properties:

1. every matching source constant is registered exactly once;
2. the registered value equals the compiled source value; and
3. named browser mirrors use the same current value as their Rust producer.

A version bump must update its decoder tests, golden vectors where applicable,
the registry, and any browser mirror in the same change. Registering a version
does not imply backward compatibility. Canonical binary and security-boundary
decoders are forward-only unless their source explicitly implements a migration
path. Viewer code may intentionally admit documented older additive report
schemas, while still naming the current producer version.

The registry does not replace semantic hashes. Mind ABI, rules, artifacts, and
replays continue to bind their canonical bytes independently.

The September 2026 cleanup advances game saves to 12 (checkpoint lifetime-key
limits), cloning artifacts to 35 (execution/build identity), and training
artifacts to 45 (execution identity and strict source-build resume). Canonical
checkpoint format 6 is unchanged. Cloning imports use the shared migration loader
for schemas 22–29 and 31–37; schema 30 remains unsupported. Legacy imports require
fresh qualification under current execution semantics. Older game saves and
training artifacts are not implicitly migrated.

Cloning schema 36 adds the cumulative seed ledger, inherited roles, confirmation
reservations, and explicit training-seed identities. Legacy parents need the
[historical lineage audit](../docs/seed-lineage.md) before further cloning.

Cloning schema 37 adds the shared private-random/local-slot target residual and
`target_residual_only` adaptation. Schemas 34–36 load through the pre-residual
layout with a zero-output migration; earlier supported layouts also initialize
the residual to zero. Training schema 46 binds the new model/optimizer layout;
older training states are not exact-resume compatible. Mind ABI, observation,
action, private-memory encodings, and expert routing are unchanged.

Learned deployment uses weight format 1 (`BLPOL001`), export/build/verification
manifest schema 1, and `blob.policy.scalar-f32-libm.v1` execution semantics.
`blob_policy` versions are included in the registry. This adds a deployment
artifact without changing Mind ABI, training weights, or canonical replay bytes;
see [learned Mind deployment](../docs/learned-mind-deployment.md).

Deployed contact evaluation schema 1 binds canonical Mind execution to per-episode
initial state/private-frontier identities, terminal state, action telemetry and
resolver-attributed damage/kills. It embeds the existing schema-3 contact report
and uses its activity/kill thresholds. This is distinct from sustained feeding;
normal canonical extermination and deadline semantics apply.

Deployed feeding evaluation schema 2 records native portable-policy ecology,
including paired initial state and private-frontier hashes, authoritative action
telemetry and validated feeding-promotion metrics. It explicitly distinguishes
canonical match evaluation from sustained feeding assessment and binds the
opponent-extinction observation for exact earlier-trajectory checks. It is development evidence,
not an automatic policy promotion; see [feeding ecology](../docs/feeding-ecology-2026-09-12.md).

The per-slot interaction residual uses its own `BLCMR001` envelope and
`RELATIONAL_WEIGHT_FORMAT_VERSION = 1`, containing an exact nested `BLCMP001`
plus 3,418 finite f32 parameters. Its execution contract is
`blob.policy.interaction-slot-encoder-f32-libm-move-q48.v1`; old envelopes retain
their existing semantics. Deployment verification reports may explicitly mark
Move-target-only inherited-retention checks as inapplicable to kind-changing
residuals. See [the study](../docs/interaction-relational-2026-09-12.md).
