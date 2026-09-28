"""Small durable training logger consumed by ``dashboard/server.py``.

No training dependency is imported. All paths default to the repository's runs/.
"""
from __future__ import annotations

import json
import os
import re
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
DEFAULT_MILESTONES = [
    {"id": "environment", "label": "Environment", "status": "pending"},
    {"id": "reference", "label": "Physical reference", "status": "pending"},
    {"id": "learning", "label": "SAC + HER", "status": "pending"},
    {"id": "evaluation", "label": "Full-task evaluation", "status": "pending"},
    {"id": "demo", "label": "Recorded demonstration", "status": "pending"},
]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.%s.tmp" % (path.name, os.getpid(), threading.get_ident()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def update_project(root: Optional[Path] = None, **fields: Any) -> Dict[str, Any]:
    """Merge project-level state. Supply the full milestones list when updating it."""
    path = Path(root or DEFAULT_ROOT) / "project.json"
    state = {"schema_version": 1, "phase": "setup", "summary": "Implementation pending.",
             "milestones": [dict(item) for item in DEFAULT_MILESTONES]}
    if path.exists():
        state.update(json.loads(path.read_text(encoding="utf-8")))
    state.update(fields)
    state["updated_at"] = timestamp()
    atomic_json(path, state)
    return state


class RunLogger:
    """One writer per run. Call metric periodically, and finish on exit.

    status.json is replaced atomically; metrics/events are newline-delimited JSON.
    Resume an existing run with resume=True. Artifacts must be under artifacts/.
    """

    def __init__(self, run_id: str, root: Optional[Path] = None,
                 resume: bool = False, **fields: Any) -> None:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a short filesystem-safe identifier")
        self.directory = Path(root or DEFAULT_ROOT) / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "status.json"
        self._lock = threading.RLock()
        if self.path.exists() and not resume:
            raise FileExistsError("Run already exists; use a new ID or resume=True")
        self.state = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else
                      {"schema_version": 1, "id": run_id, "name": run_id,
                       "kind": "train", "status": "running", "started_at": timestamp(),
                       "transitions": 0, "metrics": {}, "artifacts": []})
        self.state.update(fields)
        self.state.update(id=run_id, pid=os.getpid(), host=socket.gethostname())
        self.update()

    def _append(self, name: str, record: Dict[str, Any]) -> None:
        line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def update(self, **fields: Any) -> None:
        with self._lock:
            self.state.update(fields)
            self.state["updated_at"] = timestamp()
            atomic_json(self.path, self.state)

    def metric(self, transitions: Optional[int] = None, **values: Any) -> None:
        """Record measured metrics; success_rate is a fraction in [0, 1].

        Use scope='skill' or scope='full_task' to label what success measures.
        Count real environment transitions, never HER relabels, on the x axis.
        """
        with self._lock:
            if transitions is not None:
                self.state["transitions"] = int(transitions)
            record = dict(values, timestamp=timestamp(), transitions=self.state["transitions"])
            self._append("metrics.jsonl", record)
            self.state.setdefault("metrics", {}).update(values)
            self.update()

    def event(self, message: str, level: str = "info", **fields: Any) -> None:
        with self._lock:
            self._append("events.jsonl", dict(fields, timestamp=timestamp(), level=level, message=message))

    def artifact(self, path: str, label: str, kind: str = "file") -> None:
        relative = Path(path)
        target = (self.directory / relative).resolve()
        root = (self.directory / "artifacts").resolve()
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "artifacts":
            raise ValueError("Artifact path must be relative and inside artifacts/")
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError("Artifact escapes artifacts/ directory") from None
        if not target.is_file():
            raise FileNotFoundError(target)
        with self._lock:
            artifacts = [item for item in self.state.get("artifacts", []) if item["path"] != path]
            artifacts.append({"path": path, "label": label, "kind": kind})
            self.update(artifacts=artifacts)

    def finish(self, status: str = "complete", summary: Optional[str] = None) -> None:
        if status not in {"complete", "failed", "stopped"}:
            raise ValueError("Invalid final run status")
        fields = {"status": status, "finished_at": timestamp()}
        if summary is not None:
            fields["summary"] = summary
        self.update(**fields)
        self.event(summary or "Run " + status, "error" if status == "failed" else "info")
