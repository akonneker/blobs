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
