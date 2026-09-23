"""Content identities for the local pilot; deliberately no result cache."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot(root):
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
    files = {}
    for raw in sorted(set(names.split(b"\0")) - {b""}):
        name = os.fsdecode(raw)
        path = root / name
        files[name] = {"sha256": sha(path) if path.is_file() else None,
                       "symlink": os.readlink(path) if path.is_symlink() else None,
                       "executable": bool(path.stat().st_mode & 0o111) if path.exists() else False}
    return {"sha256": digest(files), "files": files}


def environment(root, variables):
    tools = {"python": {"version": sys.version, "path": sys.executable}}
    for name, args in (("cargo", ["--version"]), ("rustc", ["-vV"])):
        path = shutil.which(name, path=variables.get("PATH"))
        try:
            result = subprocess.run([name, *args], cwd=root, env=variables, text=True,
                                    capture_output=True, timeout=15, check=False)
            tools[name] = {"path": path, "exit": result.returncode,
                           "version": result.stdout.strip(), "error": result.stderr[:500]}
        except (OSError, subprocess.TimeoutExpired) as error:
            tools[name] = {"path": path, "error": str(error)}
    # Bind inherited settings without publishing credentials or arbitrary values.
    return {"platform": platform.platform(), "tools": tools,
            "environment_sha256": digest(dict(variables)),
            "settings": {key: variables.get(key) for key in
                         ("CARGO_INCREMENTAL", "CARGO_TERM_COLOR", "RUST_TEST_THREADS",
                          "RAYON_NUM_THREADS", "RUSTFLAGS", "CARGO_BUILD_TARGET")}}
