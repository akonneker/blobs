#!/usr/bin/env python3
"""Serve the local observatory plus recorded and live comparison reports."""

from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class ViewerHandler(SimpleHTTPRequestHandler):
    report_routes: dict[str, Path] = {}

    def end_headers(self) -> None:
        if urlsplit(self.path).path.startswith("/reports/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = self.report_routes.get(urlsplit(self.path).path)
        if route is None:
            super().do_GET()
            return
        self._send_report(route, include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        route = self.report_routes.get(urlsplit(self.path).path)
        if route is None:
            super().do_HEAD()
            return
        self._send_report(route, include_body=False)

    def _send_report(self, path: Path, *, include_body: bool) -> None:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"Report not found: {path}")
            return
        except OSError as error:
            self.send_error(500, f"Could not read report: {error}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if include_body:
            self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIEWER_PORT", "4173")))
    args = parser.parse_args()

    repository = Path(__file__).resolve().parent.parent
    viewer = repository / "blob_web" / "viewer"
    recorded_default = repository / "sweeps" / "colony-controls-event-frontier-v3" / "control-matrix-event-frontier.json"
    live_default = repository / "sweeps" / "colony-controls-event-frontier-v3" / "control-matrix-live.json"
    sensitivity_default = repository / "sweeps" / "colony-horizon-v1" / "adjudication-sensitivity-v2.json"
    fragmentation_sensitivity_default = repository / "sweeps" / "colony-population-fragmentation-v1" / "adjudication-sensitivity-v2.json"
    density_sensitivity_default = repository / "sweeps" / "colony-population-density-v1" / "adjudication-sensitivity-v2.json"
    crowding_sensitivity_default = repository / "sweeps" / "colony-population-crowding-v1" / "adjudication-sensitivity-v2.json"
    match_explorer_default = viewer / "sample-match.json"
    ViewerHandler.report_routes = {
        "/reports/match-explorer.json": Path(
            os.environ.get("MATCH_EXPLORER_REPORT", match_explorer_default)
        ).resolve(),
        "/reports/control-matrix.json": Path(
            os.environ.get("CONTROL_MATRIX_REPORT", recorded_default)
        ).resolve(),
        "/reports/control-matrix-live.json": Path(
            os.environ.get("CONTROL_MATRIX_LIVE_REPORT", live_default)
        ).resolve(),
        "/reports/adjudication-sensitivity.json": Path(
            os.environ.get("ADJUDICATION_SENSITIVITY_REPORT", sensitivity_default)
        ).resolve(),
        "/reports/adjudication-fragmentation.json": Path(
            os.environ.get(
                "ADJUDICATION_FRAGMENTATION_REPORT",
                fragmentation_sensitivity_default,
            )
        ).resolve(),
        "/reports/adjudication-density.json": Path(
            os.environ.get("ADJUDICATION_DENSITY_REPORT", density_sensitivity_default)
        ).resolve(),
        "/reports/adjudication-crowding.json": Path(
            os.environ.get("ADJUDICATION_CROWDING_REPORT", crowding_sensitivity_default)
        ).resolve(),
    }
    handler = partial(ViewerHandler, directory=viewer)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Blob observatory: http://{args.host}:{args.port}", flush=True)
    print(f"Recorded matrix: {ViewerHandler.report_routes['/reports/control-matrix.json']}", flush=True)
    print(f"Live matrix: {ViewerHandler.report_routes['/reports/control-matrix-live.json']}", flush=True)
    print(f"Match explorer: {ViewerHandler.report_routes['/reports/match-explorer.json']}", flush=True)
    print(
        f"Adjudication sensitivity: {ViewerHandler.report_routes['/reports/adjudication-sensitivity.json']}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
