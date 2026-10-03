#!/usr/bin/env python3
"""Which objects can each end effector pick? Realistic physics; all 15 YCB objects by default,
or --objects (enlarged variants included, pnr/env.py OBJECT_VARIANTS).

Allegro: the scripted grasp teachers on the hand rig (lift 15 cm with thumb + finger), top-down
power grasp and side power grasp (the latter from a side pregrasp), reported separately.
Fetch gripper: the planner's antipodal pick (grasp, lift, hold validation) on course seeds; with
--random-pose the object starts at a seeded offset and yaw, as eval.py --random-object-pose.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def allegro(args, kind='power'):
    idx, trials, _ = args
    from pnr.hand_env import HandGraspEnv
    from pnr.hand_teacher import run_teacher
    env = HandGraspEnv(objects=[idx], seed=0, physics='realistic', horizon=200, side_objects=(idx,) if kind == 'side' else ())
    wins = sum(bool(run_teacher(env, idx, seed=700 + k, kind=kind)[1].get('ever_success')) for k in range(trials))
    return f'allegro_{kind}', idx, env.rig.object_name, wins, trials


def allegro_side(args):
    return allegro(args, 'side')


def gripper(args):
    idx, trials, random_pose = args
    from pnr.env import PickNavReachEnv
    from pnr.reference import ReferenceController
    wins, name = 0, None
    for k in range(trials):
        env = PickNavReachEnv(seed=1000 + k, object_idx=idx, physics='realistic',
                              object_pose_seed=(5000 + k) if random_pose else None)
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
    parser.add_argument('--objects', type=int, nargs='+', default=list(range(15)))
    parser.add_argument('--random-pose', action='store_true')
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/object_survey_realistic.json')
    args = parser.parse_args()
    jobs = [(f, (idx, args.trials, args.random_pose)) for idx in args.objects for f in (allegro, allegro_side, gripper)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda job: job[0](job[1]), jobs)) if False else [f.result() for f in [pool.submit(fn, a) for fn, a in jobs]]
    table = {}
    for effector, idx, name, wins, trials in rows:
        table.setdefault(idx, {'object': name})[effector] = f'{wins}/{trials}'
    args.output.write_text(json.dumps(table, indent=1))
    for idx in sorted(table):
        row = table[idx]
        print(f"{idx:3d} {row['object']:28s} allegro top {row['allegro_power']:>5s}  side {row['allegro_side']:>5s}   gripper {row['gripper']:>5s}")


if __name__ == '__main__':
    main()
