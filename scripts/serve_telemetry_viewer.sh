#!/bin/sh
set -eu

VIEWER_PORT=${VIEWER_PORT:-4173}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

exec python3 "$SCRIPT_DIR/telemetry_viewer_server.py" --port "$VIEWER_PORT"
