# Compact test and evidence harness

The first reporting pilot is implemented in `scripts/run_harness.py`. It runs
existing producers, keeps complete logs and writes a bounded summary. It does
not train a model, change a qualification threshold, or replace the full CI and
conformance suites. The initial process supervisor supports POSIX hosts (macOS/Linux).

## Commands

From the repository root:

```sh
python3 -B scripts/run_harness.py --group core
python3 -B scripts/run_harness.py --group rl-probes
python3 -B scripts/run_harness.py --group hard \
  --study training-output/interaction-hard-2026-09-13-v1
python3 -B scripts/run_harness.py --group standardized \
  --study training-output/interaction-standardized-2026-09-13-v1
```

Each invocation creates a new ignored directory under `training-output/harness/`.
Use `--out training-output/harness/<new-name>` for an explicit destination;
existing output directories are refused. Study reports cannot be written into
the archived study. `--timeout <seconds>` bounds each command, not the whole group.
The default is 1,800 seconds. A timeout kills the command's process group, including
its children. Independent later steps still run; an interrupt marks them not-run.

| Group | Coverage | Explicit exclusions |
| --- | --- | --- |
| `core` | Release-mode library tests for interface, engine and portable policy | Integration tests, RL, WASM, browser, fuzz, ignored tests |
| `rl-probes` | NdArray backend numerics and seed-lineage integrations; context, hard-fixture, target-set and feeding-utility example unit tests | Full RL/workspace suite, GPU, WASM, ecology |
| `hard` | Full hard-fixture verification, including prior replay and linear control | New training and model qualification |
| `standardized` | Full normalizer, parent-memory, saved-fit and raw-control regression audit | New training, full-corpus transfer and model qualification |

Rust groups set `CARGO_INCREMENTAL=0`, disable output color, and use four test
threads. They use locked dependencies and explicit suite selections. The Cargo
adapter checks suite identities, selected counts, individual test records and
completion totals; filtered, zero-executed, ignored-only, truncated, duplicate or
unknown-format results cannot pass. This deliberately supports the selected
stable libtest output rather than arbitrary test runners. Unknown formats remain
unverified and retain their logs for diagnosis.

## Results and exits

- `manifest.json`: source content hashes (including eligible untracked/dirty files),
  commands, working directory, tool versions, feature arguments, selected environment
  settings, a digest of the complete inherited environment and timeouts. No cache
  is reused. No credentials or arbitrary environment values are dumped.
- `<step>.log`: complete combined stdout/stderr in observed order, with byte count
  and SHA-256 in the report. Parser exceptions have separate adapter-error logs.
- `study.json`: the full verifier result, bound study inputs and scientific summary.
  Historical verification files, weights, binaries and source snapshots are unchanged.
- `report.json`: all step statuses, test names/counts, failures, rerun argv, artifact
  identities, coverage and output-size measurements. Every reported failure remains
  here even when the display omits it.
- `summary.txt`: the same compact view printed to stdout. Passing displays are
  bounded to 2 KiB; failures to 6 KiB including header, with at most five witnesses.
- `status.json`: atomic stage updates and a five-second liveness heartbeat during
  long commands. It is not evidence of scientific progress.

Exit 0 means the requested command group executed and its evidence verified.
Exit 1 indicates a command, verification or required scientific-gate failure;
exit 2 indicates invalid invocation/setup; exit 130 records interruption. Inspect
the separate execution, verification and scientific fields, not only the exit.
A valid negative study can therefore exit 0 and clearly say `science=rejected`.
Add `--require-scientific-pass` to a study invocation to make that rejection exit 1.

For the historical fixture studies, the displayed scientific result is descriptive:
all target-arm initializations must fit both 64-row domains at the final declared
checkpoint. Target arms are `memory` for hard and `standardized` for standardized.
Every checkpoint is still verified and retained. The display includes both arms'
final exact-fit counts, the worst domain/seed and its margin. Full paired deltas
are in `study.json`/`report.json`. This does not promote a model or redefine a
historical development gate. Raw controls are allowed to fail fitting while their
regression evidence passes.

Supply `--plan-sha256 <expected-digest>` to pin the intended study plan. Otherwise
the runner binds the plan it observes at startup. The adapter checks the producer's
identity, planned checkpoint inventory, summary arithmetic and retained input
hashes. It also checks for source changes during the run. An unchanged end hash
does not establish an atomic filesystem snapshot: use immutable evidence and do
not edit source concurrently. Historical readers verify the transitive evidence
chain; some still report a legacy assertion with its full traceback.

## Failure witnesses and validation

Hard/standardized checks use stable invariant IDs, first-difference field paths,
expected/actual values and seed/arm/checkpoint context. Prediction failures add
the fixture row and legal mask. The plain verifier CLIs still emit their original
successful JSON shape; `--harness-output` is an additive adapter interface.
Optimized Python (`-O`/`PYTHONOPTIMIZE`) is rejected because historical dependencies
still use assertions. The migrated checks themselves are unconditional.

```sh
python3 -B scripts/test_harness.py
python3 -B scripts/test_interaction_hard.py \
  training-output/interaction-hard-2026-09-13-v1
python3 -B scripts/test_interaction_standardized.py \
  training-output/interaction-standardized-2026-09-13-v1
```

The first command is self-contained and runs in CI. The other two require local
archived evidence and verify the exact invariant for each semantic mutation.
Producer logs and study files have a 32 MiB adapter parse limit; exceeding it
fails verification while retaining the complete output on disk.

Calibration and the future full-corpus normalization run need their own adapters
once their plans/producers exist. Cache reuse, baseline diffs and counterexample
minimization remain deferred. Output byte counts measure display reduction, not
token consumption or billing; those require separate usage measurements.

## First real pilot — 2026-09-23

| Run | Verified result | Full producer output | Display |
| --- | --- | ---: | ---: |
| Core libraries | 108 passed, 18 ignored, all 3 requested suites accounted for | 13,705 bytes | 595 bytes (95.7% smaller) |
| RL probes | 32 passed, no ignored tests, all 6 suites accounted for | 4,819 bytes | 881 bytes (81.7% smaller) |
| Standardized fixture | Evidence valid; final exact fit 3/3 target initializations | Existing verification JSON: 11,241 bytes | 712 bytes |
| Hard fixture with required fit gate | Evidence valid; final exact fit 0/3; wrapper correctly exits 1 | Existing verification JSON: 11,054 bytes | 795 bytes |

The study adapters write their full result to an artifact instead of stdout.
Their original result objects equal the archived verification objects exactly.
The hard study's expected scientific rejection is not a new regression. All twelve
semantic mutations are rejected with their expected invariant IDs. The smaller RL
group falls short of the proposed 90% display-reduction target; its fixed coverage,
identity and scope overhead is retained instead of hiding that information.

Evidence directories are `training-output/harness/pilot-{core,rl-probes,hard,standardized}-2026-09-23`.
These measurements describe the first pilot runs; source identities are recorded
in their manifests. They establish display-byte savings, not measured token savings.
