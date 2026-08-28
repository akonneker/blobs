#!/usr/bin/env python3
"""Fail when a persisted-format version drifts outside the central registry."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "schemas" / "registry.json"
SOURCE_ROOTS = ["blob_engine", "blob_interface", "blob_game", "blob_rl", "blob_web"]
VERSION = re.compile(
    r"^(?:pub )?const "
    r"(?P<symbol>[A-Z0-9_]*(?:_SCHEMA_VERSION|_FORMAT_VERSION|_API_VERSION|"
    r"_STATE_VERSION|_KERNEL_VERSION|_ABI_VERSION)|PUBLICATION_VERSION|POINTER_VERSION)"
    r": u(?:16|32|64) = (?P<version>[0-9]+);$",
    re.MULTILINE,
)


def fail(messages: list[str]) -> None:
    for message in messages:
        print(f"schema registry: {message}", file=sys.stderr)
    raise SystemExit(1)


def source_versions() -> dict[tuple[str, str], int]:
    found: dict[tuple[str, str], int] = {}
    for root_name in SOURCE_ROOTS:
        for path in (ROOT / root_name).rglob("*.rs"):
            if "target" in path.parts or path.name.endswith("_capnp.rs"):
                continue
            relative = path.relative_to(ROOT).as_posix()
            for match in VERSION.finditer(path.read_text(encoding="utf-8")):
                key = (relative, match.group("symbol"))
                if key in found:
                    fail([f"duplicate source constant {relative}:{key[1]}"])
                found[key] = int(match.group("version"))
    return found


def main() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    errors: list[str] = []
    if value.get("schema_version") != 1 or not isinstance(value.get("formats"), list):
        fail(["registry envelope must be schema 1 with a formats list"])

    registered: dict[tuple[str, str], int] = {}
    for entry in value["formats"]:
        try:
            key = (entry["source"], entry["symbol"])
            version = entry["version"]
        except (KeyError, TypeError):
            errors.append(f"malformed entry {entry!r}")
            continue
        if key in registered:
            errors.append(f"duplicate registry entry {key[0]}:{key[1]}")
        if not isinstance(version, int) or version < 1:
            errors.append(f"invalid version for {key[0]}:{key[1]}")
            continue
        registered[key] = version
        for mirror in entry.get("mirrors", []):
            mirror_path = ROOT / mirror["source"]
            pattern = re.compile(
                rf"\b{re.escape(mirror['symbol'])}\s*=\s*{version}\s*;"
            )
            if not mirror_path.is_file() or not pattern.search(
                mirror_path.read_text(encoding="utf-8")
            ):
                errors.append(
                    f"browser mirror {mirror['source']}:{mirror['symbol']} is not {version}"
                )

    discovered = source_versions()
    for key, actual in sorted(discovered.items()):
        expected = registered.get(key)
        if expected is None:
            errors.append(f"unregistered source constant {key[0]}:{key[1]}={actual}")
        elif expected != actual:
            errors.append(
                f"version mismatch {key[0]}:{key[1]} source={actual} registry={expected}"
            )
    for key in sorted(registered.keys() - discovered.keys()):
        errors.append(f"registry entry has no source constant {key[0]}:{key[1]}")

    if errors:
        fail(errors)
    print(f"schema registry: {len(registered)} version constants and browser mirrors agree")


if __name__ == "__main__":
    main()
