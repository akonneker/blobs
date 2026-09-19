import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = new URL("../../", import.meta.url);
let expected;
let resources;

test.beforeAll(async () => {
  resources = new Map();
  const files = [
    ["index.js", "blob_web/pkg/index.js", "text/javascript"],
    ["blob_web.js", "blob_web/pkg/blob_web.js", "text/javascript"],
    ["blob_web_bg.wasm", "blob_web/pkg/blob_web_bg.wasm", "application/wasm"],
    ...["manifest", "other-manifest", "attestation", "bad-signature", "bad-message",
      "public-key", "wrong-key", "segment-0", "segment-1"].map((name) => [
      `${name}.bin`, `browser-tests/fixtures/generated/${name}.bin`, "application/octet-stream",
    ]),
  ];
  for (const [name, path, contentType] of files) {
    try {
      resources.set(`/wasm-test/${name}`, { body: await readFile(new URL(path, root)), contentType });
    } catch (cause) {
      throw new Error(`Missing browser fixture ${fileURLToPath(new URL(path, root))}. Run npm run prepare:wasm --prefix browser-tests first.`, { cause });
    }
  }
  expected = JSON.parse(await readFile(new URL("browser-tests/fixtures/generated/expected.json", root), "utf8"));
});

test.beforeEach(async ({ page }) => {
  // Serve only an explicit list of generated files; no production server changes
  // and no mocked WASM exports or signature verification are involved.
  await page.route("**/wasm-test/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/wasm-test/") {
      await route.fulfill({ contentType: "text/html", body: "<!doctype html><title>WASM replay test</title>" });
    } else if (resources.has(path)) {
      await route.fulfill(resources.get(path));
    } else {
      await route.fulfill({ status: 404, body: "Unknown test resource" });
    }
  });
  await page.goto("/wasm-test/");
  await page.evaluate(async () => {
    window.api = await import("/wasm-test/index.js");
    await window.api.initializeBlobWeb();
    window.bytes = async (name) => new Uint8Array(await (await fetch(`/wasm-test/${name}.bin`)).arrayBuffer());
  });
});

test("generated WASM replays native segments, exposes BigInts, and seeks backward", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const replay = new api.BrowserReplay(await bytes("manifest"));
    try {
      const manifestHash = replay.manifestHash;
      const eventCount = replay.eventCount;
      const segmentCount = replay.segmentCount;
      const types = [typeof eventCount];
      const sequences = [];
      const eventHashes = [];
      for (let index = 0; index < segmentCount; index++) {
        const descriptor = JSON.parse(replay.segmentDescriptorJson(index));
        replay.loadSegment(index, await bytes(`segment-${index}`), BigInt(descriptor.startEvent), 2);
        while (replay.currentSequence < BigInt(descriptor.endEvent)) {
          const event = replay.step();
          try {
            sequences.push(event.sequence.toString());
            eventHashes.push(event.stateHash === replay.stateHash);
            types.push(typeof event.sequence, typeof replay.currentTime);
            if (!(event.changedCells() instanceof BigUint64Array)) throw new Error("wrong changed-cell array type");
            if (!(event.changedTiles() instanceof Uint32Array)) throw new Error("wrong changed-tile array type");
            JSON.parse(replay.cellsByKeyJson(event.changedCells()));
            JSON.parse(replay.tilesJson(event.changedTiles()));
          } finally {
            event.free();
          }
        }
      }
      const finalStateHash = replay.stateHash;
      replay.loadSegment(0, await bytes("segment-0"), 0n, 0);
      const initialStateHash = replay.stateHash;
      const occupants = replay.tileOccupants();
      const emptySentinel = occupants.includes((1n << 64n) - 1n);
      const cells = JSON.parse(replay.cellsJson());
      replay.loadSegment(1, await bytes("segment-1"), eventCount, 2);
      return { apiVersion: api.browser_api_version(), manifestHash, eventCount: eventCount.toString(), segmentCount,
        initialStateHash, finalStateHash, seekHash: replay.stateHash, sequences, eventHashes,
        types: [...new Set(types)], occupantsBigInt: occupants instanceof BigUint64Array,
        emptySentinel, cellKeysAreStrings: cells.every((cell) => typeof cell.key === "string"),
        lastSegment: replay.segmentIndexForEvent(eventCount - 1n) };
    } finally {
      replay.free();
    }
  });
  expect(result).toMatchObject({ ...expected, apiVersion: 1, seekHash: expected.finalStateHash,
    types: ["bigint"], occupantsBigInt: true, emptySentinel: true, cellKeysAreStrings: true,
    lastSegment: expected.segmentCount - 1 });
  expect(result.sequences).toEqual(Array.from({ length: Number(expected.eventCount) }, (_, index) => String(index)));
  expect(result.eventHashes).toEqual(Array(Number(expected.eventCount)).fill(true));
});

test("real WebCrypto verifies the native signature and rejects altered inputs", async ({ page }) => {
  const results = await page.evaluate(async () => {
    const replay = new api.BrowserReplay(await bytes("manifest"));
    const other = new api.BrowserReplay(await bytes("other-manifest"));
    const cases = [
      ["attestation", "public-key", replay],
      ["attestation", "wrong-key", replay],
      ["bad-signature", "public-key", replay],
      ["bad-message", "public-key", replay],
      ["attestation", "public-key", other],
    ];
    try {
      const results = [];
      for (const [attestation, key, target] of cases) {
        const result = await api.verifyPublishedMatch(await bytes(attestation), await bytes(key), target);
        results.push({ verified: result.verified, reason: result.reason });
        result.attestation.free();
      }
      return results;
    } finally {
      replay.free();
      other.free();
    }
  });
  expect(results).toEqual([
    { verified: true, reason: null },
    { verified: false, reason: "signing-key-mismatch" },
    { verified: false, reason: "invalid-signature" },
    { verified: false, reason: "invalid-signature" },
    { verified: false, reason: "replay-manifest-mismatch" },
  ]);
});

test("malformed bytes and over-budget seeks fail without changing the loaded state", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const replay = new api.BrowserReplay(await bytes("manifest"));
    const segment = await bytes("segment-0");
    replay.loadSegment(0, segment, 0n, 0);
    const initialHash = replay.stateHash;
    const errors = [];
    const rejects = (operation) => {
      try { const value = operation(); value?.free?.(); errors.push(null); }
      catch (error) { errors.push(String(error)); }
    };
    try {
      rejects(() => new api.BrowserReplay(new Uint8Array([1, 2, 3])));
      rejects(() => new api.BrowserAttestation(new Uint8Array([1, 2, 3])));
      rejects(() => replay.loadSegment(0, segment.subarray(0, segment.length - 1), 0n, 0));
      rejects(() => replay.loadSegment(0, segment, 2n, 1));
      rejects(() => replay.loadSegment(0, segment, 3n, 3));
      rejects(() => replay.segmentIndexForEvent(replay.eventCount));
      return { errors, hash: replay.stateHash, initialHash, sequence: replay.currentSequence.toString() };
    } finally {
      replay.free();
    }
  });
  expect(result.errors).toHaveLength(6);
  for (const error of result.errors) expect(error).toEqual(expect.any(String));
  expect(result.hash).toBe(expected.initialStateHash);
  expect(result.hash).toBe(result.initialHash);
  expect(result.sequence).toBe("0");
});

test("unavailable WebCrypto returns an explicit unverified result", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const attestation = await bytes("attestation");
    const key = await bytes("public-key");
    // Isolated capability fault injection; the positive/negative signature test
    // above always uses Chromium's actual Ed25519 implementation.
    Object.defineProperty(globalThis, "crypto", { configurable: true, value: undefined });
    const result = await api.verifyMatchAttestation(attestation, key);
    try {
      return { verified: result.verified, reason: result.reason };
    } finally {
      result.attestation.free();
    }
  });
  expect(result).toEqual({ verified: false, reason: "webcrypto-unavailable" });
});
