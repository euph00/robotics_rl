#!/usr/bin/env python3
"""Record full-task episodes of the Allegro on the Fetch (pnr/hand_system.py) as dashboard videos.

Close-up while the hand grasps, chase view for the rest. --policy teacher uses the scripted grasp;
otherwise a train_hand.py run and step.

  python scripts/record_hand_task.py --policy teacher --objects 10 --output runs/hand_task_recordings01
  python scripts/record_hand_task.py --policy hand_f_01 --step 4000000 --objects 2 7 10 --output ...
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


def record(idx, seed, policy, output, every, size):
    env = PickNavReachEnv(seed=seed, object_idx=idx, physics='realistic', effector='allegro')
    frames, step = [], env.step
    system = HandPlannerSystem(env, policy, seed=seed, speed_scale=3., holding_speed_scale=2.)

    def recording_step(action):
        result = step(action)
        if env.step_count % every == 0:
            close = system.stage in ('pregrasp', 'grasp', 'hold')
            frames.append(env.render_rgb('grasp', *size, distance=1.1) if close else env.render_rgb('follow', *size, **CHASE))
        return result
    env.step = recording_step
    result = system.run()
    env.close()
    name = f'object_{idx:02d}_seed_{seed}'
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
    args = parser.parse_args()
    policy = TeacherPolicy() if args.policy == 'teacher' else HandPolicy(ROOT / 'runs' / args.policy, args.step)
    (args.output / 'artifacts').mkdir(parents=True, exist_ok=False)
    logger = RunLogger(args.output.name, root=args.output.parent, kind='recording',
                       name=args.label or f'Allegro on the Fetch · {policy.source}', summary=f'{len(args.objects)} episodes')
    outcomes = []
    for idx in args.objects:
        name, outcome, result = record(idx, args.seed, policy, args.output, args.every, args.size)
        logger.artifact(f'artifacts/{name}.webm', f'{NAMES.get(idx, idx)} · {outcome}', kind='video')
        outcomes.append(f'{NAMES.get(idx, idx)} {outcome}')
        print(json.dumps({'object_idx': idx, 'outcome': outcome, 'ticks': result['metrics']['physics_ticks'],
                          'error': result.get('error')}), flush=True)
    logger.finish(summary=', '.join(outcomes))


if __name__ == '__main__':
    main()
