import initWasm, {
  BrowserAttestation,
  BrowserReplay,
  browser_api_version,
} from "./blob_web.js";

export { BrowserAttestation, BrowserReplay, browser_api_version };

export async function initializeBlobWeb(wasmSource) {
  return initWasm(wasmSource);
}

export async function verifyMatchAttestation(attestationBytes, publicKeyBytes) {
  const attestation = new BrowserAttestation(attestationBytes);
  if (!attestation.matchesPublicKey(publicKeyBytes)) {
    return { verified: false, reason: "signing-key-mismatch", attestation };
  }

  try {
    const key = await globalThis.crypto.subtle.importKey(
      "raw",
      publicKeyBytes,
      { name: "Ed25519" },
      false,
      ["verify"],
    );
    const verified = await globalThis.crypto.subtle.verify(
      { name: "Ed25519" },
      key,
      attestation.signature(),
      attestation.signingMessage(),
    );
    return {
      verified,
      reason: verified ? null : "invalid-signature",
      attestation,
    };
  } catch (error) {
    return {
      verified: false,
      reason: "webcrypto-unavailable",
      error,
      attestation,
    };
  }
}

export async function verifyPublishedMatch(
  attestationBytes,
  publicKeyBytes,
  replay,
) {
  const result = await verifyMatchAttestation(
    attestationBytes,
    publicKeyBytes,
  );
  if (!result.verified) return result;
  if (result.attestation.replayManifestHash !== replay.manifestHash) {
    return { ...result, verified: false, reason: "replay-manifest-mismatch" };
  }
  return result;
}
