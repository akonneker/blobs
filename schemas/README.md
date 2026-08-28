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
