#!/usr/bin/env python3
"""Which course objects can each end effector pick? Realistic physics, all 15 YCB objects.

Allegro: the scripted power-grasp teacher on the hand rig (lift 15 cm with thumb + finger).
Fetch gripper: the planner's antipodal pick (grasp, lift, hold validation) on course seeds.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def allegro(args):
    idx, trials = args
    from pnr.hand_env import HandGraspEnv
    from pnr.hand_teacher import run_teacher
    env = HandGraspEnv(objects=[idx], seed=0, physics='realistic', horizon=160)
    wins = sum(bool(run_teacher(env, idx, seed=700 + k)[1].get('success')) for k in range(trials))
    return 'allegro', idx, env.rig.object_name, wins, trials


def gripper(args):
    idx, trials = args
    from pnr.env import PickNavReachEnv
    from pnr.reference import ReferenceController
    wins, name = 0, None
    for k in range(trials):
        env = PickNavReachEnv(seed=1000 + k, object_idx=idx, physics='realistic')
        name = env.object_name
        controller = ReferenceController(env, seed=1000 + k, speed_scale=3., grasp_proposal_mode='clearance-v2', holding_speed_scale=2.)
        try:
            wins += bool(controller.execute_grasp(controller.prepare_grasp()))
        except (ValueError, RuntimeError):
            pass
        env.close()
    return 'gripper', idx, name, wins, trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=6)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/object_survey_realistic.json')
    args = parser.parse_args()
    jobs = [(f, (idx, args.trials)) for idx in range(15) for f in (allegro, gripper)]
    with ProcessPoolExecutor(30) as pool:
        rows = list(pool.map(lambda job: job[0](job[1]), jobs)) if False else [f.result() for f in [pool.submit(fn, a) for fn, a in jobs]]
    table = {}
    for effector, idx, name, wins, trials in rows:
        table.setdefault(idx, {'object': name})[effector] = f'{wins}/{trials}'
    args.output.write_text(json.dumps(table, indent=1))
    for idx in sorted(table):
        print(f"{idx:2d} {table[idx]['object']:24s} allegro {table[idx]['allegro']:>5s}   gripper {table[idx]['gripper']:>5s}")


if __name__ == '__main__':
    main()
