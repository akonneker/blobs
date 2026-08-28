# Cadence holdout launch failure

This published execution did not run scientifically. All ten trainers failed
before environment construction because the execution contract referenced a
qualified warm start under `/tmp` that had been cleared between sessions. The
durable failed statuses and stderr logs are retained rather than rewritten.

The warm start was rebuilt and requalified, producing different immutable
artifact hashes. A new manifest and execution contract were therefore
published as `combat-curriculum-cadence-holdout-v2`; failed `v1` records were
not retried or rebound to different inputs.
