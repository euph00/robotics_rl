#!/usr/bin/env python3
"""Build the dashboard's maze illustration from recorded learned-carry trajectories.

Reads per-call trajectories written by scripts/record.py (one delivered episode and,
optionally, one that failed in the carry), rebuilds the maze (no rollout), and computes
the A* waypoint path the classical planner would follow from the same start.
Writes dashboard/maze_example.json.

  python scripts/dashboard_maze.py --recording runs/mj_recordings01 --set dev \
      --success validation_003_object_05 [--drop validation_007_object_14]
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from pnr.env import PickNavReachEnv
from pnr.planning import maze_astar
from scripts.eval import OBJECT_NAMES, select_cases

EVERY = 5   # native calls per plotted point


def carry_trace(recording, case_id, root_xy):
    rows = [json.loads(line) for line in (recording / 'artifacts' / f'{case_id}_trajectory.jsonl').open()]
    carry = [r for r in rows if r['stage'] == 'carry'][::EVERY]
    if not carry:
        raise ValueError(f'{case_id} has no carry stage')
    xy = [(root_xy + np.asarray(r['qpos'][:2])).round(3).tolist() for r in carry]
    # Object height relative to its height at the start of the carry: in-hand slip.
    z0 = carry[0]['object_position'][2]
    slip_mm = [round((r['object_position'][2] - z0) * 1000, 2) for r in carry]
    return xy, slip_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--set', default='dev')
    parser.add_argument('--success', required=True, help='Delivered case (same maze as --drop)')
    parser.add_argument('--drop', help='Optional case that failed during the carry, same maze')
    args = parser.parse_args()
    cases = {c['case_id']: c for c in select_cases(args.set)}
    case = cases[args.success]
    if args.drop and cases[args.drop]['layout_id'] != case['layout_id']:
        parser.error('--success and --drop must share a maze')
    env = PickNavReachEnv(**case['env_kwargs'])
    try:
        root_xy = np.asarray(env.root_position[:2])
        success_xy, success_slip = carry_trace(args.recording, args.success, root_xy)
        drop_xy, drop_slip = carry_trace(args.recording, args.drop, root_xy) if args.drop else ([], [])
        lower, upper = env.table_aabb()
        astar = maze_astar(env.cube_positions, np.asarray(success_xy[0]), env.goal_pos[:2],
                           footprint_radius=.36, table_bounds=(lower, upper))
        name = lambda c: OBJECT_NAMES.get(cases[c]['object_idx'], str(cases[c]['object_idx']))
        data = {
            'layout': case['layout_id'], 'cell_m': 1.0, 'recording': args.recording.name,
            'walls': np.asarray(env.cube_positions)[:, :2].round(3).tolist(),
            'robot_start': root_xy.round(3).tolist(), 'goal': np.asarray(env.goal_pos[:2]).round(3).tolist(),
            'table': [list(lower[:2]), list(upper[:2])],
            'astar': np.asarray(astar).round(3).tolist(),
            'learned_success': success_xy, 'learned_drop': drop_xy or None,
            'drop_point': drop_xy[-1] if drop_xy else None,
            'slip_success_mm': success_slip, 'slip_drop_mm': drop_slip or None,
            'trace_interval_native_calls': EVERY,
            'captions': {'success': f'{name(args.success).capitalize()} · imitation + SAC carry · delivered',
                         'drop': f'{name(args.drop).capitalize()} · imitation + SAC carry · failed' if args.drop else None,
                         'astar': 'Planner: A* waypoints, stop at each',
                         'slip': f'Delivered {name(args.success)} stays within {max(map(abs, success_slip)):.0f} mm'
                                 + (f'; the {name(args.drop)} slides out and falls' if args.drop else '')},
        }
    finally:
        env.close()
    out = ROOT / 'dashboard/maze_example.json'
    out.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    print(out, len(data['walls']), 'walls', len(data['astar']), 'A* waypoints', len(success_xy), len(drop_xy), 'trace points')


if __name__ == '__main__':
    main()
