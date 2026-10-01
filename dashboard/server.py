#!/usr/bin/env python3
"""Read-only, localhost experiment dashboard using only Python's standard library."""
from __future__ import annotations

import argparse
import csv
import gzip
import html
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
from urllib.parse import quote, unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
PUBLIC = Path(__file__).resolve().parent / "public"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
REPORTS = {
    "project_state.md": REPO / "docs/project_state.md",
    "experiment_log.md": REPO / "docs/experiment_log.md",
    "phase2_candidate.json": REPO / "configs/phase2_candidate.json",
    "phase2_candidate02.json": REPO / "configs/phase2_candidate02.json",
    "candidate03.json": REPO / "configs/candidate03.json",
    "project_spec.md": REPO / "docs/project_spec.md",
}

MEDIA_EXTENSIONS = {".mp4", ".webm", ".png", ".jpg", ".jpeg", ".gif", ".csv", ".json", ".txt", ".pdf"}
STATE_KEYS = {"schema_version", "id", "name", "kind", "algorithm", "skill", "seed", "status", "started_at",
              "updated_at", "finished_at", "transitions", "budget", "summary", "metrics", "artifacts", "accounting", "inherited_transitions", "planned_cases"}
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



def safe_report(name):
    """Only explicitly published project reports; no arbitrary repository files."""
    target = REPORTS.get(name)
    try:
        if (target is None or target.resolve() != target.absolute() or not target.is_file()
                or target.stat().st_size > 1024 * 1024):
            return None
        return target
    except OSError:
        return None


def report_link(target, runs_root):
    if target in REPORTS and safe_report(target):
        return "/reports/" + quote(target)
    config_reports = {"../configs/" + name: name for name in ("phase2_candidate.json", "phase2_candidate02.json")}
    if target in config_reports and safe_report(config_reports[target]):
        return "/reports/" + config_reports[target]
    if target.startswith("../runs/"):
        parts = target.split("/", 3)
        if len(parts) == 4 and safe_artifact(runs_root, parts[2], parts[3]):
            return "/artifacts/" + quote(parts[2]) + "/" + quote(parts[3], safe="/")
    if urlsplit(target).scheme in ("https", "http"):
        return target
    return None


def render_report(target, runs_root):
    """Render the small Markdown subset used by published reports, escaping HTML."""
    def inline(text):
        output = []
        for token in re.split(r"(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*)", text):
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            if link:
                label, destination = link.groups()
                url = report_link(destination, runs_root)
                output.append('<a href="' + html.escape(url, quote=True) + '" rel="noopener">'
                              + html.escape(label) + '</a>' if url else html.escape(label))
            elif token.startswith('`') and token.endswith('`'):
                output.append('<code>' + html.escape(token[1:-1]) + '</code>')
            elif token.startswith('**') and token.endswith('**'):
                output.append('<strong>' + html.escape(token[2:-2]) + '</strong>')
            else:
                output.append(html.escape(token))
        return ''.join(output)

    lines = target.read_text(encoding="utf-8").splitlines()
    blocks, paragraph = [], []
    def flush():
        if paragraph:
            blocks.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
        elif line.startswith('```'):
            flush(); code = []; index += 1
            while index < len(lines) and not lines[index].strip().startswith('```'):
                code.append(lines[index]); index += 1
            blocks.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
        elif re.match(r'^#{1,3} ', line):
            flush(); heading, title = line.split(' ', 1)
            blocks.append(f'<h{len(heading)}>' + inline(title) + f'</h{len(heading)}>')
        elif line.startswith('|') and index + 1 < len(lines) and re.fullmatch(r'[| :\-]+', lines[index + 1].strip()):
            flush()
            headers = line.strip('|').split('|')
            rows = ['<thead><tr>' + ''.join('<th>' + inline(cell.strip()) + '</th>' for cell in headers) + '</tr></thead><tbody>']
            index += 2
            while index < len(lines) and lines[index].strip().startswith('|'):
                rows.append('<tr>' + ''.join('<td>' + inline(cell.strip()) + '</td>' for cell in lines[index].strip().strip('|').split('|')) + '</tr>')
                index += 1
            blocks.append('<div class="table-scroll"><table>' + ''.join(rows) + '</tbody></table></div>')
            continue
        else:
            paragraph.append(line)
        index += 1
    flush()
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Project report · Pick Nav Reach</title><link rel="stylesheet" href="/style.css">'
            '</head><body><main class="report"><p><a href="/">← Dashboard</a></p><article>'
            + ''.join(blocks) + '</article></main></body></html>').encode()


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


OVERVIEW = Path(__file__).resolve().parent / "overview.json"
OBJECT_NAMES = {14: "foam", 5: "banana", 2: "can", 7: "apple", 10: "mug"}
_EVALUATION_CACHE = {}


def _alive(raw):
    if raw.get("status") != "running" or raw.get("host") != socket.gethostname():
        return raw.get("status") == "running"
    try:
        os.kill(int(raw.get("pid", -1)), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def failure_label(row):
    error = str(row.get("error") or row.get("failed_stage") or "unknown")
    stage, _, detail = error.partition(":")
    detail = detail.strip().lower()
    if "lost" in detail or "drop" in detail:
        detail = "object dropped"
    elif "horizon" in detail:
        detail = "ran out of decisions"
    return f"{stage.strip()}: {detail}" if detail else stage.strip()


def rubric_points(rows):
    """Per-trial course rubric estimate (strict reading: sim step = physics tick)."""
    if not rows:
        return None
    items = {"lift_within_3000": 0., "lift_10cm": 0., "hold_50_steps": 0., "pose_close": 0.,
             "reach_10cm": 0., "reach_within_15000": 0., "carry_stability": 0.}
    for row in rows:
        lift, reach, pose = row.get("first_lift_physics_tick"), row.get("first_complete_reach_physics_tick"), row.get("pick_pose_error") or {}
        error = str(row.get("error") or "").lower()
        items["lift_within_3000"] += 6 * bool(lift is not None and lift <= 3000)
        items["lift_10cm"] += 6 * bool(lift is not None)
        items["hold_50_steps"] += 6 * bool(row.get("pick_hold_ok"))
        items["pose_close"] += 6 * bool(pose and pose.get("position_m", 1) < .02 and pose.get("orientation_rad", 1) < .1)
        items["reach_10cm"] += 6 * bool(row.get("success"))
        items["reach_within_15000"] += 6 * bool(row.get("success") and reach is not None and reach <= 15000)
        items["carry_stability"] += 4 * bool((row.get("stage_ticks") or {}).get("carry") is not None
                                            and not (error.startswith("carry") and ("lost" in error or "drop" in error)))
    items = {key: round(value / len(rows), 2) for key, value in items.items()}
    if "first_lift_physics_tick" not in rows[0]:
        return None  # older result files lack the timing fields
    return {"per_trial": round(sum(items.values()), 1), "items": items}


def _model_source(path):
    parts = Path(str(path)).parts
    return parts[parts.index("runs") + 1] if "runs" in parts and parts.index("runs") + 1 < len(parts) else str(path)


def evaluation_summary(directory):
    """Summary of one scripts/eval.py run, cached by result-file size and mtime."""
    config = read_json(directory / "config.json")
    results = directory / "results.jsonl"
    if not config or "execution" not in config or "set" not in config or not results.is_file():
        return None
    stat = results.stat()
    status = read_json(directory / "status.json") or {}
    key = (stat.st_mtime_ns, stat.st_size, status.get("status"))
    cached = _EVALUATION_CACHE.get(directory.name)
    if cached and cached[0] == key:
        return cached[1]
    rows = read_records(results, limit=5000)
    by_object, by_set, failures = {}, {}, {}
    for row in rows:
        for table, name in ((by_object, OBJECT_NAMES.get(row.get("object_idx"), str(row.get("object_idx")))),
                            (by_set, row.get("set", "?"))):
            entry = table.setdefault(name, {"cases": 0, "success": 0, "timed": 0})
            entry["cases"] += 1
            entry["success"] += bool(row.get("success"))
            entry["timed"] += bool(row.get("timed"))
        if not row.get("success"):
            label = failure_label(row)
            failures[label] = failures.get(label, 0) + 1
    execution = config["execution"]
    summary = {
        "id": directory.name, "label": config.get("label") or status.get("name") or directory.name,
        "set": config["set"], "planned": config.get("cases"), "completed": len(rows),
        "success": sum(bool(r.get("success")) for r in rows), "timed": sum(bool(r.get("timed")) for r in rows),
        "by_object": by_object, "by_set": by_set,
        "failures": dict(sorted(failures.items(), key=lambda item: -item[1])),
        "rubric": rubric_points(rows), "execution": execution,
        "controller": execution.get("controller", "learned"),
        "models": {name: _model_source(path) for name, path in (config.get("checkpoints") or {}).items()},
        "status": "running" if _alive(status) else status.get("status", "complete"),
        "started_at": status.get("started_at"), "updated_at": status.get("updated_at") or status.get("finished_at"),
    }
    _EVALUATION_CACHE[directory.name] = (key, summary)
    return summary


def get_overview(runs_root):
    curated = read_json(OVERVIEW) or {}
    evaluations, active, history = [], [], []
    directories = [item for item in runs_root.iterdir() if item.is_dir() and not item.is_symlink()
                   and SAFE_ID.fullmatch(item.name)] if runs_root.is_dir() else []
    for directory in directories:
        status = read_json(directory / "status.json") or {}
        summary = evaluation_summary(directory)
        if summary:
            evaluations.append(summary)
        elif status:
            history.append(dict({key: status.get(key) for key in ("name", "kind", "status", "summary", "updated_at")}, id=directory.name))
        if status and _alive(status):
            if summary:
                progress, unit = summary["completed"] / max(summary["planned"] or 1, 1), f"{summary['completed']}/{summary['planned']} cases"
            else:
                budget = status.get("budget") or 0
                progress = (status.get("transitions") or 0) / budget if budget else None
                unit = f"{status.get('transitions') or 0:,}/{budget:,} decisions" if budget else status.get("summary", "")
            active.append({"id": directory.name, "name": status.get("name") or directory.name, "kind": status.get("kind"),
                           "progress": progress, "detail": unit, "started_at": status.get("started_at")})
    evaluations.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    history.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    by_id = {item["id"]: item for item in evaluations}
    for arm in curated.get("arms", []):
        arm["results"] = {}
        for split, run_id in (arm.get("runs") or {}).items():
            item = by_id.get(run_id)
            if item:
                arm["results"][split] = {key: item[key] for key in ("id", "success", "timed", "completed", "planned", "rubric")}
    for recording in curated.get("recordings", []):
        raw = read_json(runs_root / str(recording.get("run", "")) / "status.json") or {}
        recording["media"] = [item for item in raw.get("artifacts", []) if isinstance(item, dict)
                              and item.get("kind") == "video" and isinstance(item.get("path"), str)
                              and safe_artifact(runs_root, recording["run"], item["path"]) is not None]
    return finite_json({"timestamp": utcnow(), "overview": curated, "evaluations": evaluations,
                        "active": active, "history": history})


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
        if path == "/api/overview":
            state = get_overview(self.server.runs_root)
            state["resources"] = self.server.resources.get()
            payload = json.dumps(state, allow_nan=False, separators=(",", ":")).encode()
            headers = {"Vary": "Accept-Encoding"}
            if "gzip" in [part.strip() for part in self.headers.get("Accept-Encoding", "").split(",")]:
                payload = gzip.compress(payload, compresslevel=1)
                headers["Content-Encoding"] = "gzip"
            self._headers(200, "application/json; charset=utf-8", len(payload), headers)
            if not head:
                self.wfile.write(payload)
            return
        if path == "/api/state":
            project = dict(DEFAULT_PROJECT)
            raw = read_json(self.server.runs_root / "project.json")
            if raw:
                project.update({key: raw[key] for key in ("schema_version", "phase", "summary", "milestones", "updated_at", "run_labels", "featured_run") if key in raw})
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
        elif path.startswith("/reports/"):
            target = safe_report(path[len("/reports/"):])
            if target is not None and target.suffix == ".md":
                payload = render_report(target, self.server.runs_root)
                self._headers(200, "text/html; charset=utf-8", len(payload))
                if not head:
                    self.wfile.write(payload)
                return
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
