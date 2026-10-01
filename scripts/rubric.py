#!/usr/bin/env python3
"""Estimate the course rubric's per-trial points from an eval.py results file.

Per-trial items (README "Rubrics"): pick 24 pts, navigation and reach 16 pts.
"Sim steps" is ambiguous upstream: reported both as physics ticks (strict) and
as env.step() calls (5 physics ticks each). The rubric gives no threshold for
"final pose close to target grasp"; we report the measured error and score it
with a declared 2 cm / 0.1 rad tolerance. The 60 points for the grasp,
motion-planning and navigation functions are not per-trial and not scored here.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def score(row, ticks_per_step):
    lift = row.get('first_lift_physics_tick')
    reach = row.get('first_complete_reach_physics_tick')
    pose = row.get('pick_pose_error') or {}
    error = (row.get('error') or '').lower()
    dropped_in_carry = error.startswith('carry') and ('lost' in error or 'drop' in error)
    navigated = row.get('stage_ticks', {}).get('carry') is not None
    return {
        'lift_within_3000': 6 * bool(lift is not None and lift / ticks_per_step <= 3000),
        'lift_10cm': 6 * bool(lift is not None),
        'hold_50_steps': 6 * bool(row.get('pick_hold_ok')),
        'pose_close': 6 * bool(pose and pose['position_m'] < .02 and pose['orientation_rad'] < .1),
        'reach_10cm': 6 * bool(row['success']),
        'reach_within_15000': 6 * bool(row['success'] and reach is not None and reach / ticks_per_step <= 15000),
        'carry_stability': 4 * bool(navigated and not dropped_in_carry),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='eval.py output directory')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'results.jsonl').open()]
    for label, ticks_per_step in (('sim step = physics tick', 1), ('sim step = env.step call', 5)):
        scores = [score(r, ticks_per_step) for r in rows]
        mean = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
        pick = sum(mean[k] for k in ('lift_within_3000', 'lift_10cm', 'hold_50_steps', 'pose_close'))
        nav = sum(mean[k] for k in ('reach_10cm', 'reach_within_15000', 'carry_stability'))
        print(f'{label}: pick {pick:.1f}/24, navigation+reach {nav:.1f}/16, per-trial total {pick+nav:.1f}/40')
        print('   ' + ', '.join(f'{k} {v:.2f}' for k, v in mean.items()))
    errors = [r['pick_pose_error'] for r in rows if r.get('pick_pose_error')]
    print(f'pick pose error: median {np.median([e["position_m"] for e in errors])*1000:.1f} mm, '
          f'{np.median([e["orientation_rad"] for e in errors]):.3f} rad; '
          f'upstream proximity success {np.mean([bool(r.get("upstream_success")) for r in rows]):.2f}')


if __name__ == '__main__':
    main()
