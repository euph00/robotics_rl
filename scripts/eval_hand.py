#!/usr/bin/env python3
"""Full-task evaluation of the Allegro on the Fetch (pnr/hand_system.py) on eval.py's case sets.

Same cases, physics, planned motion speeds and success/deadline accounting as
`scripts/evaluate_mujoco.sh`'s planner runs (the parallel gripper, planned speed 3, holding 2), so the two
compare case by case: complete = native complete_task_success; timed = complete within
15,000 physics ticks including the scene's settling ticks.

  python scripts/eval_hand.py --policy hand_f_01 --step 6000000 --set course --objects 2 7 10 --output runs/hand_eval_course01
  python scripts/eval_hand.py --policy teacher --set course --objects 10 --output runs/hand_eval_teacher01
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.eval as E  # noqa: E402  (imports pnr first: rendering defaults, same module state as eval.py)
from pnr.env import PickNavReachEnv  # noqa: E402
from pnr.hand_system import HandPlannerSystem, HandPolicy, TeacherPolicy  # noqa: E402
from pnr.telemetry import RunLogger, atomic_json  # noqa: E402

DEADLINE = 15000
_POLICY = None


def initialize_worker(policy, step):
    global _POLICY
    import torch
    torch.set_num_threads(1)
    _POLICY = TeacherPolicy() if policy == 'teacher' else HandPolicy(ROOT / 'runs' / policy, step)


def evaluate_case(case, speed_scale, holding_speed_scale, horizon, fingers='frozen', hand_mount='wrist', cradle=False):
    started, env, first = time.monotonic(), None, {'reach': None}

    def on_step(controller, observation, info):
        if first['reach'] is None and info.get('complete_task_success'):
            first['reach'] = env.physics_steps
    try:
        env = PickNavReachEnv(**case['env_kwargs'], physics='realistic', effector='allegro', hand_mount=hand_mount)
        system = HandPlannerSystem(env, _POLICY, horizon=horizon, on_step=on_step, seed=env.seed, speed_scale=speed_scale,
                                   holding_speed_scale=holding_speed_scale, fingers_after_grasp=fingers, cradle=cradle)
        result = system.run()
    except Exception as exc:  # construction or unexpected errors count as failures
        result = {'success': False, 'failed_stage': 'environment_setup' if env is None else 'unexpected',
                  'error': f'{type(exc).__name__}: {exc}'}
    ticks = env.physics_steps if env is not None else 0
    total = ticks + (env.settle_steps if env is not None else 0)
    if env is not None:
        env.close()
    success = bool(result.get('success'))
    hand = result.get('stages', {}).get('hand_grasp', {})
    return {'case_id': case['case_id'], 'set': case['set'], 'layout_id': case['layout_id'], 'object_idx': case['object_idx'],
            'success': success, 'timed': bool(success and total <= DEADLINE), 'failed_stage': result.get('failed_stage'),
            'error': result.get('error'), 'physics_ticks': ticks, 'total_ticks_including_settling': total,
            'first_lift_physics_tick': result.get('first_lift_physics_tick'),
            'first_complete_reach_physics_tick': first['reach'], 'hand_decisions': hand.get('decisions'),
            'hand_held': hand.get('held'), 'drop_events': result.get('metrics', {}).get('drop_events'),
            'wall_contact_steps': result.get('wall_contact_steps'), 'wall_seconds': time.monotonic() - started}


def summarise(rows):
    def block(items):
        n = len(items)
        return {'cases': n, 'success': sum(r['success'] for r in items), 'timed': sum(r['timed'] for r in items),
                'success_rate': sum(r['success'] for r in items) / max(n, 1), 'timed_rate': sum(r['timed'] for r in items) / max(n, 1)}
    failures = {}
    for row in rows:
        if not row['success']:
            label = (row.get('error') or str(row.get('failed_stage'))).split(':')[0]
            failures[label] = failures.get(label, 0) + 1
    return {'overall': block(rows), 'by_object': {E.OBJECT_NAMES.get(i, str(i)): block([r for r in rows if r['object_idx'] == i])
                                                  for i in sorted({r['object_idx'] for r in rows})},
            'by_set': {s: block([r for r in rows if r['set'] == s]) for s in sorted({r['set'] for r in rows})},
            'failures': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy', required=True, help='teacher, or a train_hand.py run id')
    parser.add_argument('--step', type=int)
    parser.add_argument('--set', choices=sorted(E.SETS), default='course')
    parser.add_argument('--objects', type=int, nargs='+', default=[2, 7, 10])
    parser.add_argument('--layout-limit', type=int)
    parser.add_argument('--random-object-pose', action='store_true', help='Seeded object start offset and yaw per case')
    parser.add_argument('--speed-scale', type=float, default=3., help='Planned motion speed (as evaluate_mujoco.sh)')
    parser.add_argument('--holding-speed-scale', type=float, default=2., help='Planned motion speed while holding')
    parser.add_argument('--horizon', type=int, default=300, help='Hand-policy decisions for the grasp')
    parser.add_argument('--fingers-after-grasp', choices=('policy', 'frozen'), default='frozen',
                        help='After the grasp: servo targets fixed, or the hand policy keeps setting them')
    parser.add_argument('--hand-mount', choices=('wrist', 'dorsal'), default='wrist')
    parser.add_argument('--carry-posture', choices=('cradle', 'upright'), default='upright',
                        help='Carry with the grasp orientation kept, or palm-up under the object (needs a policy trained to roll it there)')
    parser.add_argument('--allow-reserve', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.set == 'reserve' and not args.allow_reserve:
        parser.error('The final reserve is sealed; pass --allow-reserve only for a final claim')
    if args.output.exists():
        parser.error(f'{args.output} exists')
    cases = E.select_cases(args.set, objects=args.objects, layout_limit=args.layout_limit,
                           random_object_pose=args.random_object_pose)
    args.output.mkdir(parents=True)
    atomic_json(args.output / 'config.json', {'set': args.set, 'cases': len(cases), 'label': f'hand · {args.policy}',
                                              'controller': 'hand', 'policy': args.policy, 'step': args.step,
                                              'objects': args.objects, 'random_object_pose': args.random_object_pose,
                                              'hand_mount': args.hand_mount, 'fingers_after_grasp': args.fingers_after_grasp,
                                              'carry_posture': args.carry_posture,
                                              'command': sys.argv})
    source = 'scripted teacher' if args.policy == 'teacher' else f'{args.policy}/{args.step}'
    logger = RunLogger(args.output.name, root=args.output.parent, kind='evaluation',
                       name=f'Allegro on the Fetch · {source} · {args.set}', summary=f'{len(cases)} cases')
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)), mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(args.policy, args.step)) as pool:
        futures = [pool.submit(evaluate_case, case, args.speed_scale, args.holding_speed_scale, args.horizon, args.fingers_after_grasp,
                               args.hand_mount, args.carry_posture == 'cradle') for case in cases]
        with (args.output / 'results.jsonl').open('w') as stream:
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
    summary = dict(summarise(rows), config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    atomic_json(args.output / 'summary.json', summary)
    o = summary['overall']
    print(json.dumps(summary, indent=1))
    logger.finish(summary=f"{o['success']}/{o['cases']} complete; {o['timed']}/{o['cases']} within 15k ticks")


if __name__ == '__main__':
    main()
