#!/usr/bin/env python3
"""Build a frozen export as an immutable Extism PDK Mind (no training)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.export_directory.resolve()
    with (output / "export.json").open("rb") as source:
        manifest_bytes = source.read(1024 * 1024 + 1)
    if len(manifest_bytes) > 1024 * 1024:
        raise SystemExit("export manifest exceeds byte budget")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or manifest.get("weight_format") != 1 or manifest.get("execution_contract") not in {"blob.policy.scalar-f32-libm.v1", "blob.policy.move-utility-scalar-f32-libm-q48-signal-reserved.v1", "blob.policy.interaction-slot-encoder-f32-libm-move-q48.v1"}:
        raise SystemExit("unsupported deployment artifact contract")
    with (output / "weights.bin").open("rb") as source:
        weights = source.read(16 * 1024 * 1024 + 1)
    if len(weights) > 16 * 1024 * 1024 or sha(weights) != manifest["weights_sha256"]:
        raise SystemExit("exported weights do not match manifest")
    expected_magic = (b"BLCMR001" if manifest["execution_contract"] == "blob.policy.interaction-slot-encoder-f32-libm-move-q48.v1" else b"BLCMP001" if manifest["execution_contract"] == "blob.policy.move-utility-scalar-f32-libm-q48-signal-reserved.v1" else b"BLPOL001")
    if not weights.startswith(expected_magic) or len(weights) != manifest["weight_bytes"]:
        raise SystemExit("weight envelope does not match execution contract/size")
    if (output / "mind.wasm").exists() or (output / "mind-build.json").exists():
        raise SystemExit("Mind output already exists; use a new export")
    inputs = [root / "Cargo.toml", root / "Cargo.lock", Path(__file__).resolve()]
    for crate in ["blob_interface", "blob_policy", "minds/learned_mind"]:
        inputs.extend(p for p in (root / crate / "src").rglob("*") if p.is_file())
        inputs.extend(p for p in [root / crate / "Cargo.toml", root / crate / "build.rs"] if p.exists())
    inputs.extend(p for p in (root / "blob_interface/interface").rglob("*") if p.is_file())
    sources = {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(set(inputs))}
    command = ["cargo", "build", "--locked", "--release", "--target", "wasm32-unknown-unknown", "-p", "learned_mind", "--message-format=json"]
    with tempfile.TemporaryDirectory(prefix="blob-mind-build-") as temporary:
        frozen = Path(temporary) / "weights.bin"
        frozen.write_bytes(weights)
        env = dict(os.environ, BLOB_POLICY_WEIGHTS=str(frozen))
        result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE)
        artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        for artifact in artifacts:
            if artifact.get("reason") == "compiler-message":
                sys.stderr.write(artifact["message"].get("rendered") or artifact["message"]["message"])
        result.check_returncode()
        files = [Path(f) for artifact in artifacts if artifact.get("reason") == "compiler-artifact" and artifact.get("target", {}).get("name") == "learned_mind" for f in artifact["filenames"] if f.endswith(".wasm")]
        if len(files) != 1:
            raise SystemExit("expected one compiled learned Mind WASM")
        wasm = files[0].read_bytes()
    if any(sha((root / p).read_bytes()) != digest for p, digest in sources.items()):
        raise SystemExit("deployment source changed during build")
    report = {"schema_version": 1, "export_manifest_sha256": sha(manifest_bytes), "weights_sha256": sha(weights), "wasm_sha256": sha(wasm), "wasm_bytes": len(wasm), "command": command, "rustc": subprocess.check_output(["rustc", "-Vv"], text=True), "source_sha256": sources}
    with (output / "mind.wasm").open("xb") as f:
        f.write(wasm)
    with (output / "mind-build.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ["wasm_sha256", "wasm_bytes", "weights_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
