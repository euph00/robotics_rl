#!/usr/bin/env python3
"""Read-only, localhost experiment dashboard using only Python's standard library."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
PUBLIC = Path(__file__).resolve().parent / "public"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
MEDIA_EXTENSIONS = {".mp4", ".webm", ".png", ".jpg", ".jpeg", ".gif", ".csv", ".json", ".txt", ".pdf"}
STATE_KEYS = {"schema_version", "id", "name", "kind", "algorithm", "skill", "seed", "status", "started_at",
              "updated_at", "finished_at", "transitions", "budget", "summary", "metrics", "artifacts", "accounting", "inherited_transitions"}
DEFAULT_PROJECT = {
    "schema_version": 1, "phase": "setup", "summary": "First milestone: one physically valid SAC + HER system.",
    "milestones": [{"id": key, "label": label, "status": "pending"} for key, label in (
        ("environment", "Environment"), ("reference", "Physical reference"), ("learning", "SAC + HER"),
        ("evaluation", "Full-task evaluation"), ("demo", "Recorded demonstration"))],
}


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    try:
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            return None
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def finite_json(value):
    """External writers may emit NaN; keep the public API valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def read_records(path, limit=800):
    """Read a bounded JSONL tail, tolerating a live writer's incomplete last row."""
    try:
        if path.is_symlink():
            return []
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 2 * 1024 * 1024))
            if handle.tell():
                handle.readline()
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            continue
    return finite_json(rows[-limit:])


def read_metrics(directory):
    rows = read_records(directory / "metrics.jsonl")
    if not rows:
        path = directory / "metrics.csv"
        try:
            if not path.is_symlink() and path.stat().st_size <= 2 * 1024 * 1024:
                with path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        for key, value in list(row.items()):
                            try:
                                row[key] = float(value)
                            except (ValueError, TypeError):
                                pass
                        rows.append(row)
                rows = finite_json(rows[-800:])
        except OSError:
            pass
    # Preserve endpoints while limiting response size.
    if len(rows) > 240:
        rows = [rows[round(i * (len(rows) - 1) / 239)] for i in range(240)]
    return rows


def safe_artifact(runs_root, run_id, relative):
    """Return a real file strictly inside this run's real artifacts directory."""
    if not SAFE_ID.fullmatch(run_id):
        return None
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "artifacts":
        return None
    if relative.suffix.lower() not in MEDIA_EXTENSIONS:
        return None
    try:
        run_root = (runs_root / run_id).resolve()
        run_root.relative_to(runs_root.resolve())
        if (run_root / "artifacts").is_symlink():
            return None
        artifacts = (run_root / "artifacts").resolve()
        artifacts.relative_to(run_root)
        target = (run_root / relative).resolve()
        target.relative_to(artifacts)
        return target if target.is_file() else None
    except (OSError, ValueError):
        return None


def get_runs(runs_root):
    runs, events = [], []
    if not runs_root.is_dir():
        return runs, events
    directories = sorted((item for item in runs_root.iterdir() if item.is_dir() and not item.is_symlink()
                          and SAFE_ID.fullmatch(item.name)), key=lambda item: item.name, reverse=True)
    for directory in directories:
        raw = read_json(directory / "status.json")
        if not raw:
            continue
        run = {key: value for key, value in raw.items() if key in STATE_KEYS}
        run["id"] = directory.name
        # Charts use scalar measurements; repeated nested diagnostics and lineage
        # remain in the source logs and latest metrics, not every chart point.
        run["series"] = [{key: value for key, value in row.items()
                          if not isinstance(value, (dict, list))}
                         for row in read_metrics(directory)]
        run["artifacts"] = []
        for item in raw.get("artifacts", []):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            if safe_artifact(runs_root, directory.name, item["path"]) is not None:
                run["artifacts"].append({key: item[key] for key in ("path", "label", "kind") if key in item})
        if raw.get("status") == "running" and raw.get("host") == socket.gethostname() and isinstance(raw.get("pid"), int) and raw["pid"] > 0:
            try:
                os.kill(raw["pid"], 0)
            except ProcessLookupError:
                run["status"] = "interrupted"
                run["summary"] = "Process exited without a final status. " + str(raw.get("summary", ""))
            except PermissionError:
                pass
        for event in read_records(directory / "events.jsonl", limit=8):
            clean_event = {key: event[key] for key in ("timestamp", "level", "message") if key in event}
            clean_event["run_id"] = directory.name
            events.append(clean_event)
        runs.append(run)
    runs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return finite_json(runs), events[:12]


class Resources:
    def __init__(self, workspace):
        self.workspace = workspace
        self.lock = threading.Lock()
        self.timestamp = 0
        self.cached = {}

    def get(self):
        with self.lock:
            if time.monotonic() - self.timestamp < 15:
                return self.cached
            result = {"logical_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}
            try:
                result["load_1m"] = round(os.getloadavg()[0], 1)
                memory = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, value = line.split(":", 1)
                    memory[key] = int(value.strip().split()[0]) * 1024
                result.update(memory_total_bytes=memory["MemTotal"], memory_available_bytes=memory["MemAvailable"])
            except (OSError, ValueError, KeyError):
                pass
            disk = shutil.disk_usage(self.workspace)
            result.update(disk_free_bytes=disk.free, disk_total_bytes=disk.total)
            try:
                process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True)
                result["gpus"] = [{"name": parts[0].strip(), "memory_used_mib": float(parts[1]),
                                   "memory_total_mib": float(parts[2]), "utilization": float(parts[3])}
                                  for parts in csv.reader(io.StringIO(process.stdout)) if len(parts) == 4]
            except (OSError, ValueError, subprocess.SubprocessError):
                result["gpus"] = []
            self.cached, self.timestamp = result, time.monotonic()
            return result


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runs_root):
        self.runs_root = Path(runs_root).resolve()
        self.resources = Resources(REPO)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "PickNavReachDashboard/1"

    def _headers(self, status, content_type, size, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; object-src 'none'; frame-ancestors 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/state":
            project = dict(DEFAULT_PROJECT)
            raw = read_json(self.server.runs_root / "project.json")
            if raw:
                project.update({key: raw[key] for key in ("schema_version", "phase", "summary", "milestones", "updated_at", "run_labels") if key in raw})
            runs, events = get_runs(self.server.runs_root)
            state = {"timestamp": utcnow(), "project": project, "runs": runs, "events": events,
                     "resources": self.server.resources.get()}
            payload = json.dumps(finite_json(state), allow_nan=False, separators=(",", ":")).encode()
            headers = {"Vary": "Accept-Encoding"}
            if "gzip" in [part.strip() for part in self.headers.get("Accept-Encoding", "").split(",")]:
                payload = gzip.compress(payload, compresslevel=1)
                headers["Content-Encoding"] = "gzip"
            self._headers(200, "application/json; charset=utf-8", len(payload), headers)
            if not head:
                self.wfile.write(payload)
            return
        static = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
        if path in static:
            target = PUBLIC / static[path]
        elif path.startswith("/artifacts/"):
            parts = path.split("/", 3)
            target = safe_artifact(self.server.runs_root, parts[2], parts[3]) if len(parts) == 4 else None
        else:
            target = None
        if target is None or not target.is_file():
            self.send_error(404)
            return
        size = target.stat().st_size
        start, end, status = 0, size - 1, 200
        headers = {"Accept-Ranges": "bytes"}
        if self.headers.get("Range"):
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"])
            if not match or not any(match.groups()):
                self._headers(416, "text/plain", 0, {"Content-Range": "bytes */%s" % size})
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                start = max(0, size - int(right))
            if start > end or start >= size:
                self._headers(416, "text/plain", 0, {"Content-Range": "bytes */%s" % size})
                return
            status = 206
            headers["Content-Range"] = "bytes %s-%s/%s" % (start, end, size)
        length = max(0, end - start + 1)
        self._headers(status, mimetypes.guess_type(str(target))[0] or "application/octet-stream", length, headers)
        if not head:
            try:
                with target.open("rb") as handle:
                    handle.seek(start)
                    remaining = length
                    while remaining:
                        block = handle.read(min(128 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, fmt, *args):
        # Polling does not need a growing access log; errors remain visible.
        if args and not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--runs", type=Path, default=REPO / "runs")
    args = parser.parse_args()
    server = DashboardServer(("127.0.0.1", args.port), args.runs)
    print("Dashboard: http://127.0.0.1:%s (read-only)" % server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
