#!/usr/bin/env python3
"""Record full-task episodes of the Allegro on the Fetch (pnr/hand_system.py) as dashboard videos.

Close-up while the hand grasps, chase view for the rest. --policy teacher uses the scripted grasp;
otherwise a train_hand.py run and step.

  python scripts/record_hand_task.py --policy teacher --objects 10 --output runs/hand_task_recordings01
  python scripts/record_hand_task.py --policy hand_f_01 --step 4000000 --objects 2 7 10 --output ...
  python scripts/record_hand_task.py --policy hand_w_wide_tc_s0 --step 3500000 --set holdout --random-object-pose \
      --cases confirmation_20261001_000_object_00 --output ...       # protocol cases, as eval_hand.py runs them
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pnr.env import PickNavReachEnv  # noqa: E402  (sets the headless rendering defaults first)
from pnr.hand_system import HandPlannerSystem, HandPolicy, TeacherPolicy  # noqa: E402
import imageio  # noqa: E402

from pnr.telemetry import RunLogger  # noqa: E402
from pnr.video import webm_copy  # noqa: E402

NAMES = {0: 'master chef can', 2: 'soup can', 5: 'banana', 7: 'apple', 10: 'mug', 14: 'foam brick'}
CHASE = dict(distance=6, pitch=-75)


def record(env_kwargs, name, policy, output, every, size):
    env = PickNavReachEnv(**env_kwargs, physics='realistic', effector='allegro', hand_mount='wrist')
    frames, step = [], env.step
    system = HandPlannerSystem(env, policy, seed=env.seed, speed_scale=3., holding_speed_scale=2.)   # as eval_hand.py

    def recording_step(action):
        result = step(action)
        if env.step_count % every == 0:
            close = system.stage in ('pregrasp', 'grasp', 'hold')
            frames.append(env.render_rgb('grasp', *size, distance=1.1) if close else env.render_rgb('follow', *size, **CHASE))
        return result
    env.step = recording_step
    result = system.run()
    env.close()
    path = output / 'artifacts' / f'{name}.mp4'
    imageio.mimwrite(path, frames, fps=30, quality=7, macro_block_size=8)
    webm_copy(path)
    grasp_frames = [i for i, _ in enumerate(frames)][:max(1, len(frames) // 6)]
    imageio.imwrite(output / 'artifacts' / f'{name}_preview.png', frames[grasp_frames[-1]])
    outcome = 'delivered' if result['success'] else f"failed at {result.get('failed_stage')}"
    return name, outcome, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy', required=True, help='teacher, or a train_hand.py run id')
    parser.add_argument('--step', type=int)
    parser.add_argument('--objects', type=int, nargs='+', default=[10])
    parser.add_argument('--seed', type=int, default=0, help='Course scene seed')
    parser.add_argument('--every', type=int, default=4, help='Native calls per frame')
    parser.add_argument('--size', type=int, nargs=2, default=(640, 360))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--set', help="eval.py case set (e.g. holdout); with --cases, replaces --objects/--seed")
    parser.add_argument('--cases', nargs='+', help='case_id values from --set')
    parser.add_argument('--random-object-pose', action='store_true', help="The protocol's seeded random object start per case")
    args = parser.parse_args()
    if args.set:
        import scripts.eval as E
        by_id = {c['case_id']: c for c in E.select_cases(args.set, random_object_pose=args.random_object_pose)}
        episodes = [(by_id[c]['env_kwargs'], c, by_id[c]['object_idx']) for c in args.cases]
    else:
        episodes = [(dict(seed=args.seed, object_idx=idx), f'object_{idx:02d}_seed_{args.seed}', idx) for idx in args.objects]
    policy = TeacherPolicy() if args.policy == 'teacher' else HandPolicy(ROOT / 'runs' / args.policy, args.step)
    (args.output / 'artifacts').mkdir(parents=True, exist_ok=False)
    logger = RunLogger(args.output.name, root=args.output.parent, kind='recording',
                       name=args.label or f'Allegro on the Fetch · {policy.source}', summary=f'{len(episodes)} episodes')
    outcomes = []
    for env_kwargs, name, idx in episodes:
        name, outcome, result = record(env_kwargs, name, policy, args.output, args.every, args.size)
        logger.artifact(f'artifacts/{name}.webm', f'{NAMES.get(idx, idx)} · {outcome}', kind='video')
        outcomes.append(f'{NAMES.get(idx, idx)} {outcome}')
        print(json.dumps({'object_idx': idx, 'outcome': outcome, 'ticks': result['metrics']['physics_ticks'],
                          'error': result.get('error')}), flush=True)
    logger.finish(summary=', '.join(outcomes))


if __name__ == '__main__':
    main()
