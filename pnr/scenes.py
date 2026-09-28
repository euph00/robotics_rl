"""Versioned scene distribution and frozen, content-disjoint layout partitions.

The independent distribution uses the upstream generator and all original maze
geometry. Only the previously coupled/fixed RNG inputs change. Geometry checks
are not physical success checks: no policy is run to admit a layout, and neither
route length nor later physical failures may remove a declared evaluation case.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .env import UPSTREAM_COMMIT, _load_maze_helpers, _preserve_global_rng

SEED_MODE = "independent-maze-v1"
BANK_VERSION = "scene-bank-v1"
PARTITION_SEED = 20260928
DEFAULT_COUNTS = {"train": 64, "validation": 12, "milestone_test": 20, "final_reserve": 20}
HELDOUT_PARTITIONS = frozenset(("milestone_test", "final_reserve"))
GEOMETRY = {
    "maze_rows": 12, "maze_columns": 16, "left_room_columns": 6,
    "grid_xy_m": 1.0, "wall_height_m": 2.0, "maze_center_xyz": [7.0, 0.0, 0.0],
    "grid_density": 0.8, "p_greedy": 0.6, "all_border_walls": True,
    "goal_x_m": 17.5, "goal_z_m": 0.9,
    "world_cell_transform": "x = column - 3.5; y = 5.5 - row",
}


def map_hash(maze_map):
    """Same canonical int8 row-major byte hash used by the native environment."""
    maze_map = np.asarray(maze_map)
    if maze_map.shape != (12, 22) or not np.isin(maze_map, (0, 1)).all():
        raise ValueError("Expected the original 12x22 binary maze including its left room")
    return hashlib.sha256(maze_map.astype(np.int8).tobytes()).hexdigest()


@dataclass(frozen=True)
class MazeLayout:
    maze_seed: int
    maze_map: np.ndarray
    entrance_cell: tuple[int, int]
    exit_cell: tuple[int, int]

    @property
    def maze_hash(self):
        return map_hash(self.maze_map)

    @property
    def goal_pos(self):
        return np.array([17.5, 5.5 - self.exit_cell[0], 0.9], dtype=np.float64)

    @property
    def wall_centers(self):
        rows, cols = np.where(self.maze_map == 1)
        return np.column_stack((cols - 3.5, 5.5 - rows)).astype(np.float32)

    def as_dict(self):
        route = shortest_grid_route(self.maze_map, self.entrance_cell, self.exit_cell)
        return {
            "maze_seed": self.maze_seed, "seed_mode": SEED_MODE,
            "maze_hash": self.maze_hash, "maze_map": self.maze_map.tolist(),
            "maze_entrance_cell": list(self.entrance_cell),
            "maze_exit_cell": list(self.exit_cell), "goal_pos": self.goal_pos.tolist(),
            "geometry_check": {
                "gate_to_gate_connected": bool(route),
                "gate_to_gate_shortest_grid_length_m": len(route) - 1 if route else None,
                "meaning": "Free-cell centerline connectivity only; not a physical carrying guarantee",
            },
        }


def _generate_map(in_row, out_row, generator_seed):
    helpers = _load_maze_helpers()
    with _preserve_global_rng():
        maze = helpers.generate_maze_map(
            12, 16, (in_row, 0), (out_row, 15), grid_density=0.8,
            add_x_min_wall=True, add_x_max_wall=True,
            add_y_min_wall=True, add_y_max_wall=True, p_greedy=0.6,
            seed=int(generator_seed))
        return helpers.add_left_room_to_maze(maze, room_width=6)


def generate_layout(maze_seed: int) -> MazeLayout:
    """Independent mode v1: RandomState(seed) gates and generator(seed) interior.

    Using a local, explicitly named RNG makes the gates independent of object
    sampling. The generator still runs with its own original RNG implementation.
    No process-global learner RNG state is consumed or replaced.
    """
    maze_seed = int(maze_seed)
    if not 0 <= maze_seed < 2**32:
        raise ValueError("maze_seed must fit an unsigned 32-bit seed")
    in_row, out_row = map(int, np.random.RandomState(maze_seed).randint(1, 11, size=2))
    return MazeLayout(maze_seed, _generate_map(in_row, out_row, maze_seed),
                      (in_row, 6), (out_row, 21))


def shortest_grid_route(maze_map, start, goal):
    """Deterministic four-connected BFS used solely for declared geometry checks."""
    maze_map = np.asarray(maze_map)
    start, goal = tuple(start), tuple(goal)
    queue, parent = deque([start]), {start: None}
    if maze_map[start] or maze_map[goal]:
        return []
    while queue:
        cell = queue.popleft()
        if cell == goal:
            route = []
            while cell is not None:
                route.append(cell)
                cell = parent[cell]
            return route[::-1]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidate = (cell[0] + dr, cell[1] + dc)
            if (0 <= candidate[0] < maze_map.shape[0]
                    and 0 <= candidate[1] < maze_map.shape[1]
                    and not maze_map[candidate] and candidate not in parent):
                parent[candidate] = cell
                queue.append(candidate)
    return []


def course_legacy_hashes():
    """Exclude all 100 possible gate pairs under upstream's fixed internal seed."""
    return sorted({map_hash(_generate_map(i, o, 5)) for i in range(1, 11) for o in range(1, 11)})


def manifest_digest(manifest):
    content = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_scene_bank(partition_seed=PARTITION_SEED, counts: Mapping[str, int] | None = None,
                     object_indices=()):
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS) or any(int(n) != n or n < 0 for n in counts.values()):
        raise ValueError("Declare nonnegative counts for all four fixed partitions")
    counts = {name: int(counts[name]) for name in DEFAULT_COUNTS}
    object_indices = [int(i) for i in object_indices]
    if len(set(object_indices)) != len(object_indices) or any(not 0 <= i < 15 for i in object_indices):
        raise ValueError("Object indices must be distinct entries from the original 15 assets")
    excluded = course_legacy_hashes()
    seen = set(excluded)
    rng = np.random.default_rng(int(partition_seed))
    partitions, rejected = {}, []
    for partition, count in counts.items():
        records = []
        for _ in range(max(1, count) * 1000):
            if len(records) == count:
                break
            candidate = int(rng.integers(0, 2**31))
            layout = generate_layout(candidate)
            record = layout.as_dict()
            reason = ("duplicate_or_legacy_map" if layout.maze_hash in seen else
                      "disconnected_gate_cells" if not record["geometry_check"]["gate_to_gate_connected"] else None)
            if reason:
                rejected.append({"maze_seed": candidate, "maze_hash": layout.maze_hash,
                                 "partition": partition, "reason": reason})
                continue
            seen.add(layout.maze_hash)
            record["layout_id"] = f"{partition}_{len(records):03d}"
            records.append(record)
        if len(records) != count:
            raise RuntimeError(f"Bounded geometry sampling exhausted for {partition}")
        partitions[partition] = records
    bank = {
        "version": BANK_VERSION, "status": "frozen_layout_partition",
        "upstream_commit": UPSTREAM_COMMIT, "seed_mode": SEED_MODE,
        "partition_seed": int(partition_seed), "partition_counts": counts,
        "geometry": GEOMETRY, "object_indices": object_indices,
        "object_selection_status": "declared" if object_indices else "pending_separate_pretraining_feasibility_selection",
        "rules": {
            "candidate_rng": "numpy.default_rng(partition_seed), integers(0, 2**31)",
            "partition_order": list(DEFAULT_COUNTS),
            "admission": "content hash unique across all partitions and legacy maps; gate-cell connectivity",
            "route_length_filter": False, "learned_outcome_filter": False,
            "physical_failure_policy": "Every declared attempted case remains in the denominator; never silently replace it",
            "training_usage": "train only; validation may select checkpoints/methods",
            "milestone_test_usage": "once a complete candidate is frozen; no training or method selection",
            "final_reserve_usage": "sealed for the later independent algorithm/gripper/seed comparison",
            "course_regression": "separate course-legacy-v1 evaluation; never substitute for held-out layouts",
            "initial_settling": "native defaults unchanged; report setup ticks separately",
        },
        "excluded_course_legacy_hashes": excluded,
        "geometry_rejections": rejected, "partitions": partitions,
    }
    bank["manifest_sha256"] = manifest_digest(bank)
    validate_scene_bank(bank)
    return bank


def validate_scene_bank(bank):
    if bank.get("version") != BANK_VERSION or bank.get("seed_mode") != SEED_MODE:
        raise ValueError("Unknown scene-bank distribution version")
    if bank.get("manifest_sha256") != manifest_digest(bank):
        raise ValueError("Scene-bank manifest digest mismatch")
    if set(bank["partitions"]) != set(DEFAULT_COUNTS):
        raise ValueError("Missing scene-bank partition")
    seen = set(bank["excluded_course_legacy_hashes"])
    for partition in DEFAULT_COUNTS:
        records = bank["partitions"][partition]
        if len(records) != bank["partition_counts"][partition]:
            raise ValueError("Partition count mismatch")
        for record in records:
            digest = map_hash(record["maze_map"])
            if digest != record["maze_hash"] or digest in seen:
                raise ValueError("Duplicate or incorrect maze content hash")
            seen.add(digest)
            layout = generate_layout(record["maze_seed"])
            if layout.maze_hash != digest:
                raise ValueError("Layout seed does not reproduce its frozen content")
            if (list(layout.entrance_cell) != record["maze_entrance_cell"]
                    or list(layout.exit_cell) != record["maze_exit_cell"]
                    or layout.goal_pos.tolist() != record["goal_pos"]):
                raise ValueError("Layout goal/gate geometry mismatch")
    return bank


def load_scene_bank(path):
    return validate_scene_bank(json.loads(Path(path).read_text()))


def scene_records(bank, partition="train", *, allow_heldout=False):
    """Explicit access guard against accidentally using held-out layouts to train."""
    if partition not in DEFAULT_COUNTS:
        raise ValueError(f"Unknown partition: {partition}")
    if partition in HELDOUT_PARTITIONS and not allow_heldout:
        raise ValueError("Held-out partitions require explicit allow_heldout=True evaluation access")
    return list(bank["partitions"][partition])


def case_specs(bank, partition="train", *, object_indices=None, allow_heldout=False):
    """Cross selected objects with layouts; each case has a fixed point-cloud seed."""
    objects = bank["object_indices"] if object_indices is None else [int(i) for i in object_indices]
    if not objects:
        raise ValueError("Declare the object set before constructing physical cases")
    if len(set(objects)) != len(objects) or any(not 0 <= i < 15 for i in objects):
        raise ValueError("Object indices must be distinct entries from the original 15 assets")
    result = []
    for layout in scene_records(bank, partition, allow_heldout=allow_heldout):
        for object_idx in objects:
            case_id = f'{layout["layout_id"]}_object_{object_idx:02d}'
            point_seed = int.from_bytes(hashlib.sha256(
                f'{bank["partition_seed"]}:{case_id}:points-v1'.encode()).digest()[:4], "little")
            result.append({"case_id": case_id, "layout_id": layout["layout_id"], "partition": partition,
                           "maze_hash": layout["maze_hash"], "object_idx": object_idx,
                           "env_kwargs": {"seed": point_seed, "object_idx": object_idx,
                                          "maze_seed": layout["maze_seed"], "seed_mode": SEED_MODE}})
    return result


def bank_summary(bank):
    summary = {"manifest_sha256": bank["manifest_sha256"], "counts": bank["partition_counts"],
               "geometry_rejections": len(bank["geometry_rejections"]),
               "object_indices": bank["object_indices"], "route_length_filter": False,
               "physical_evaluations": 0, "development_route_lengths_m": {}}
    for partition in ("train", "validation"):
        lengths = [r["geometry_check"]["gate_to_gate_shortest_grid_length_m"] for r in bank["partitions"][partition]]
        summary["development_route_lengths_m"][partition] = {
            "mean": float(np.mean(lengths)) if lengths else None,
            "max": max(lengths) if lengths else None, "min": min(lengths) if lengths else None}
    summary["heldout_hashes"] = {partition: [r["maze_hash"] for r in bank["partitions"][partition]]
                                  for partition in ("milestone_test", "final_reserve")}
    return summary
