#!/usr/bin/env python3
"""Build the dashboard's maze illustration from recorded carry trajectories.

Constructs the example maze (no rollout), reads two saved learned-carry traces
from the retired confirmation test (one success, one drop), and computes the
A* waypoint path the classical planner would follow from the same start.
Writes dashboard/maze_example.json.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from pnr.env import PickNavReachEnv
from pnr.planning import maze_astar
from scripts.eval import select_cases

TRACES = ROOT / 'runs/phase2_system_confirmation_test01/artifacts'
LAYOUT = 'confirmation_20260930_013'
SUCCESS, DROP = LAYOUT + '_object_05', LAYOUT + '_object_14'


def carry_trace(case, root_xy):
    rows = [json.loads(line) for line in (TRACES / f'{case}_trajectory.jsonl').open()]
    carry = [r for r in rows if r['stage'] == 'carry']
    xy = [(root_xy + np.asarray(r['qpos'][:2])).round(3).tolist() for r in carry]
    # Object height relative to the base's starting height: a proxy for slip in the hand.
    z0 = carry[0]['object_position'][2]
    slip_mm = [round((r['object_position'][2] - z0) * 1000, 2) for r in carry]
    contact = [bool(r['bilateral_contact']) for r in carry]
    return xy, slip_mm, contact


def main():
    case = next(c for c in select_cases('test2') if c['case_id'] == SUCCESS)
    env = PickNavReachEnv(**case['env_kwargs'])
    try:
        root_xy = np.asarray(env.root_position[:2])
        success_xy, success_slip, _ = carry_trace(SUCCESS, root_xy)
        drop_xy, drop_slip, drop_contact = carry_trace(DROP, root_xy)
        lower, upper = env.p.getAABB(env.table_id)
        astar = maze_astar(env.cube_positions, np.asarray(success_xy[0]), env.goal_pos[:2],
                           footprint_radius=.36, table_bounds=(lower, upper))
        data = {
            'layout': LAYOUT, 'cell_m': 1.0,
            'walls': np.asarray(env.cube_positions)[:, :2].round(3).tolist(),
            'robot_start': (root_xy).round(3).tolist(), 'goal': np.asarray(env.goal_pos[:2]).round(3).tolist(),
            'table': [list(lower[:2]), list(upper[:2])],
            'astar': np.asarray(astar).round(3).tolist(),
            'learned_success': success_xy, 'learned_drop': drop_xy,
            'drop_point': drop_xy[-1], 'slip_success_mm': success_slip, 'slip_drop_mm': drop_slip,
            'trace_interval_native_calls': 5,
            'captions': {'success': 'Banana · learned carry · delivered',
                         'drop': 'Foam brick · learned carry · slipped out',
                         'astar': 'Planner: A* waypoints, stop at each'},
        }
    finally:
        env.close()
    out = ROOT / 'dashboard/maze_example.json'
    out.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    print(out, len(data['walls']), 'walls', len(data['astar']), 'A* waypoints',
          len(success_xy), len(drop_xy), 'trace points')


if __name__ == '__main__':
    main()
