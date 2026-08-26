# Blob browser replay package

`blob_web` is the read-only browser boundary for replay exploration and public
match verification. It deliberately does not expose Mind execution, a signing
key, or a way to create a trusted leaderboard result.

## Build

Install `wasm-bindgen-cli` version `0.2.100`, matching the Rust crate, and run:

```sh
scripts/build_browser_package.sh
```

This produces ES-module glue, the compiled module, and the higher-level wrapper
in `blob_web/pkg`. The raw release WASM build is part of repository conformance;
generating the JavaScript glue is kept out of conformance so the CLI is not an
implicit development dependency.

## Replay

```js
import { initializeBlobWeb, BrowserReplay } from "/blob_web/pkg/index.js";

await initializeBlobWeb();
const replay = new BrowserReplay(await fetchBytes(manifestUrl));
const segmentIndex = replay.segmentIndexForEvent(500n);
const descriptor = JSON.parse(replay.segmentDescriptorJson(segmentIndex));
const segment = await fetchBytes(`/replays/segments/${descriptor.segmentHash}`);
replay.loadSegment(segmentIndex, segment, 500n, 4096);

while (replay.currentSequence < 520n) {
  const event = replay.step();
  render(replay, event);
}
```

The manifest is a compact index: the UI can locate and fetch only the segment
containing the desired event. Each segment starts with an authoritative
checkpoint, so seek work is bounded to at most one segment. `loadSegment`
verifies the segment against its content-addressed manifest descriptor before
restoring or replaying it.

JavaScript-visible Rust `u64` values are `BigInt`. JSON methods encode `u64`
fields as decimal strings. `tileOccupants()` uses `2^64 - 1` as its empty-tile
sentinel. Plant inspectors can read energy, capacity, growth rate, and
fractional growth progress through the corresponding `tilePlant*()` arrays.
Diffuse transport progress and each of the four anonymous signal channels are
available through `tileDiffusionRemainder()` and `tileSignalEnergy(channel)`.
Tile arrays are intended for render loops; `cellsJson()` carries the
less regular cell metadata used by inspectors, including assimilated/gut energy
and the canonical metabolism remainder. After the first full snapshot,
`event.changedTiles()` and `event.changedCells()` let a renderer refresh only
resources touched by a step; `event.deaths()` and the alternating
parent/child values in `event.births()` support entity lifecycle updates.
Pass those sparse arrays to `replay.tilesJson(indices)` and
`replay.cellsByKeyJson(keys)` to retrieve current values without copying the
full world snapshot.

## Public attestation verification

```js
import { verifyPublishedMatch } from "/blob_web/pkg/index.js";

const result = await verifyPublishedMatch(
  attestationBytes,
  serverPublicKey,
  replay,
);
if (!result.verified) throw new Error(result.reason);
```

Rust performs bounded canonical parsing and checks that the public key hashes
to the signed key ID. Browser WebCrypto verifies the Ed25519 signature over the
exact domain-separated message returned by WASM. The same portable encoder and
parser are used by the native server signer. `verifyPublishedMatch` additionally
checks that the signed replay-manifest hash is the replay being displayed.

The caller must obtain the expected server public key through a trusted channel;
a key supplied beside an untrusted replay proves nothing about server identity.

Successful browser replay proves that the published commitment stream produces
the claimed states. The server attestation is what proves that the server also
re-ran the frozen Mind artifacts with private server-derived randomness. A
browser-submitted score or locally generated replay is never authoritative on
its own.

`BrowserAttestation.mindRuntimeProfile` exposes the manifest-bound capability
profile (currently `extism_pdk_deterministic_v1`) alongside `mindAbiHash` and
the exact artifact bindings. It identifies the admitted guest contract, not the
server's executor implementation or worker policy.

`fetchBytes` and `render` above are application-specific helpers, not package
exports.

## Local telemetry viewer

Run the dependency-free viewer from the repository root:

```sh
scripts/serve_telemetry_viewer.sh
```

Then open `http://localhost:4173`. It loads a sample colony report by default
and accepts dropped or selected JSON files. The report envelope is versioned; its
`latest` object is the serializable `ColonyScenarioTelemetry` sample, while
timeline, spatial, action, and adversarial-check sections are optional viewer
extensions. Telemetry schema 2 added aggregate occupancy of cell-private rolling
visitation windows. Schema 3 adds aggregate outcome reliability, contention,
gut-progress, gross energy-loss, and signal-retention evidence. Visitation
density is useful for diagnosing exploration memory, but it is not canonical
board coverage and must not be presented as such. Guarded loss is observational,
not a causal mitigation measurement.

Telemetry schema 4 adds the explicit Signal action family. Older telemetry is
rejected rather than treating signal actions as another family.

The UI is implemented as the Shadow-DOM custom element
`<blob-telemetry-viewer>`. A later tinker.ninja page can import
`viewer/telemetry-viewer.js` and set either its `src` attribute or its `data`
property without inheriting or leaking page styles. Browser-loaded reports are
always shown as local and unverified, even if their JSON claims otherwise. An
enclosing application may call `setVerifiedTelemetry(report, verification)`
only after `verifyPublishedMatch` returns a successful result for the same
run. The component rejects unsuccessful verification results, but the server
attestation and enclosing site remain the security boundary; the component is
only presentation.

The same observatory serves the maintained-Mind comparison page at
`http://localhost:4173/?view=controls`. By default its recorded route reads
`sweeps/colony-controls-event-frontier-v3/control-matrix-event-frontier.json`. Override that path
with `CONTROL_MATRIX_REPORT=/absolute/report.json` when starting the server.
The comparison UI is the independent Shadow-DOM component
`<blob-control-matrix-viewer>` from `viewer/control-matrix-viewer.js`; keeping
it separate prevents aggregate policy evaluation from being confused with an
attested replay.

For live progress, run the matrix with a mutable status destination:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin control-matrix -- sweeps/colony-controls-event-frontier-v3/manifest.json \
  --max-parallel 4 \
  --output sweeps/colony-controls-event-frontier-v3/control-matrix-next.json \
  --live-output sweeps/colony-controls-event-frontier-v3/control-matrix-live.json
```

Open `http://localhost:4173/?view=controls&live=1` while it runs. The status
file is atomically replaced after each completed matchup and served with
`Cache-Control: no-store`. It contains only compact matchup/profile/seat
aggregates rather than repeating full episode telemetry; the detailed final
report remains immutable. `CONTROL_MATRIX_LIVE_REPORT` can point the local
route at a different status file. Both report components label browser-loaded
data local and unverified, and the comparison component rejects JSON claiming
server verification or a replay commitment. Final schema-3 reports also show
the raw per-side terminal core, assimilated, gut, carried, and escrow
compartments for event-time draws. The viewer deliberately does not turn those
measurements into a biomass score.

The winning-criteria lab is available at
`http://localhost:4173/?view=adjudication`. It reads the immutable horizon
sensitivity report by default and compares explicit gut, carried-material,
escrow, and minimum-margin policies. Set
`ADJUDICATION_SENSITIVITY_REPORT=/absolute/report.json` to display another
analysis. The population studies are available through the same component:

- `?view=adjudication&study=fragmentation`
- `?view=adjudication&study=density`
- `?view=adjudication&study=crowding`

Their server routes can be redirected with
`ADJUDICATION_FRAGMENTATION_REPORT`, `ADJUDICATION_DENSITY_REPORT`, and
`ADJUDICATION_CROWDING_REPORT`. The navigation and table adapt to either
multiple horizons or multiple population variants. Every page is deliberately
labeled local and counterfactual: it cannot turn a post-hoc score into a
verified match outcome.
