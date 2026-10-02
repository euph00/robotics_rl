#!/usr/bin/env python3
"""Full-task evaluation of a grasp/carry/reach system on a fixed set of mazes.

Every case starts from the original task start, with no resets between skills,
teacher corrections or object attachment. Complete-task success and success
within the 15,000-physics-tick deadline (including initial settling and the
final hold) are reported separately.

Case sets (five objects x layouts):
  validation  12 validation mazes                          (60 cases)
  test1       first milestone test mazes, now development  (100 cases)
  test2       confirmation test mazes, now development     (100 cases)
  dev         validation + test1 + test2                   (260 cases)
  holdout     fresh 20 mazes built 2026-09-29, unused for any selection (100 cases)
  reserve     sealed final_reserve mazes for the final study; requires --allow-reserve
  course      unmodified course seed mapping (as graders run it), seeds 1000-1019 (100 cases)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np

from pnr.confirmation_scenes import load_evaluation_bank
from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.general_manipulation import GeneralGraspEnv
from pnr.carry_rules import CarryRules
from pnr.policies import load_policy_checkpoint
from pnr.reference import ReferenceController
from pnr.scenes import case_specs
from pnr.system import LearnedSystem
from pnr.telemetry import RunLogger, atomic_json

OBJECTS = [14, 5, 2, 7, 10]
OBJECT_NAMES = {14: 'foam', 5: 'banana', 2: 'can', 7: 'apple', 10: 'mug'}
MAIN_BANK = ROOT / 'configs/phase2_scene_bank_expanded.json'
CONFIRMATION_BANK = ROOT / 'configs/phase2_confirmation_bank_20260930.json'
HOLDOUT_BANK = ROOT / 'configs/holdout_bank_20260929.json'
SETS = {'validation': [(MAIN_BANK, 'validation', 'validation')],
        'test1': [(MAIN_BANK, 'milestone_test', 'test1')],
        'test2': [(CONFIRMATION_BANK, 'milestone_test', 'test2')],
        'holdout': [(HOLDOUT_BANK, 'milestone_test', 'holdout')],
        'reserve': [(MAIN_BANK, 'final_reserve', 'reserve')]}
SETS['dev'] = SETS['validation'] + SETS['test1'] + SETS['test2']
SETS['course'] = []  # built from course seeds in select_cases
DEFAULT_EXECUTION = {'grasp_horizon': 600, 'carry_horizon': 1200, 'reach_horizon': 150,
                     'delta_scale': .075, 'planned_speed_scale': 1., 'deadline_physics_ticks': 15000,
                     'grasp_lift_m': .24, 'reach_tolerance_m': .08, 'grasp_proposal_mode': 'clearance-v2',
                     'holding_speed_scale': 0.,  # 0: same as planned_speed_scale
                     'controller': 'learned',  # or 'reference': classical planner end to end
                     'carry_max_turn_deg': 0., 'carry_slip_drift_mm': 0., 'carry_slip_scale': .5,
                     'grasp_mode': 'learned', 'carry_mode': 'learned',  # 'planned' swaps in the classical stage
                     'physics': 'realistic'}  # or 'course' (unlimited course motors, mu 0.25 grip)
_POLICIES = None


COURSE_SEEDS = range(1000, 1020)


def select_cases(name, objects=OBJECTS, layout_limit=None, random_object_pose=False):
    if name == 'course':
        seeds = list(COURSE_SEEDS)[:layout_limit] if layout_limit else COURSE_SEEDS
        return [{'case_id': f'course_seed_{seed}_object_{obj:02d}', 'layout_id': f'course_seed_{seed}',
                 'set': 'course', 'maze_hash': None, 'object_idx': obj,
                 'env_kwargs': {'seed': seed, 'object_idx': obj, 'seed_mode': 'course-legacy-v1'}}
                for seed in seeds for obj in objects]
    cases = []
    for bank_path, partition, label in SETS[name]:
        bank = load_evaluation_bank(bank_path)
        selected = case_specs(bank, partition, object_indices=objects, allow_heldout=True)
        if layout_limit:
            layouts = sorted({case['layout_id'] for case in selected})[:layout_limit]
            selected = [case for case in selected if case['layout_id'] in layouts]
        for case in selected:
            case['set'] = label
        cases += selected
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case identifiers across banks')
    if random_object_pose:      # seeded per case, so every system sees the same start
        for case in cases:
            case['env_kwargs']['object_pose_seed'] = zlib.crc32(case['case_id'].encode())
    return cases


def initialize_worker(checkpoints):
    global _POLICIES
    import torch
    torch.set_num_threads(1)
    _POLICIES = {name: load_policy_checkpoint(path, device='cpu') for name, path in checkpoints.items()}


def run_reference(native, execution):
    """Classical pipeline: antipodal grasp, planned descent/lift, A* carry, base/torso reach."""
    stage_ticks, first = {}, {'reach': None}

    def on_step(controller, observation, info):
        stage_ticks[controller.stage] = stage_ticks.get(controller.stage, 0) + native.substeps
        if first['reach'] is None and info.get('complete_task_success'):
            first['reach'] = native.physics_steps

    controller = ReferenceController(native, on_step=on_step, seed=native.seed,
        speed_scale=execution['planned_speed_scale'], grasp_proposal_mode=execution['grasp_proposal_mode'],
        holding_speed_scale=execution['holding_speed_scale'] or None)
    result = {'success': False, 'policy_decisions': {}}
    try:
        grasp = controller.prepare_grasp()
        result['stages_grasp'] = grasp.as_dict()
        picked = controller.execute_grasp(grasp)
        ee, orientation = controller.planner.ee_pose()
        result['pick_pose_error'] = {
            'position_m': float(np.linalg.norm(ee - (grasp.position + [0, 0, .20]))),
            'orientation_rad': float(2 * np.arccos(np.clip(abs(np.dot(orientation, grasp.quaternion)), 0, 1)))}
        if not picked:
            raise RuntimeError('grasp, lift and hold validation failed')
        controller.prepare_carry()
        if not native.contacts()['bilateral_contact']:
            raise RuntimeError('lost grasp')
        controller.carry()
        if not controller.reach()['complete_task_success']:
            raise RuntimeError('final strict physical success check failed')
        result['success'] = True
    except (ValueError, RuntimeError) as exc:
        result.update(failed_stage=controller.stage, error=f'{controller.stage}: {exc}')
    result.update(metrics=native.evaluate(),
        first_lift_physics_tick=None if controller.first_lift_action is None else controller.first_lift_action * native.substeps,
        first_complete_reach_physics_tick=first['reach'],
        stages={name: {'physics_ticks': ticks} for name, ticks in stage_ticks.items()})
    result['stages']['pick_hold'] = {'success': controller.stage_results.get('pick', {}).get('success'),
                                     'physics_ticks': stage_ticks.get('hold')}
    result['stages']['planned_approach'] = {'grasp': result.pop('stages_grasp', None)}
    return result


def run_learned(native, execution):
    """Three learned skills with planned approach and carry posture (LearnedSystem)."""
    policies = dict(_POLICIES)
    if execution['carry_max_turn_deg'] > 0 or execution['carry_slip_drift_mm'] > 0:
        policies['carry'] = CarryRules(_POLICIES['carry'], execution['carry_max_turn_deg'],
                                       execution['carry_slip_drift_mm'], execution['carry_slip_scale'])
    system = LearnedSystem(native, policies,
        grasp_horizon=execution['grasp_horizon'], carry_horizon=execution['carry_horizon'],
        reach_horizon=execution['reach_horizon'], reach_tolerance_m=execution['reach_tolerance_m'],
        grasp_proposal_mode=execution['grasp_proposal_mode'],
        controller_speed_scale=execution['planned_speed_scale'],
        skill_factories={'grasp': partial(GeneralGraspEnv.bind, goal_lift_m=execution['grasp_lift_m']),
                         'carry': partial(GeneralCarryEnv.bind, delta_scale=execution['delta_scale'])})
    if execution['holding_speed_scale'] > 0:
        system.controller.holding_speed_scale = execution['holding_speed_scale']
    learned_skill = system._skill

    def skill(name):
        # Optionally replace one learned stage with the classical controller's version.
        if execution.get(f'{name}_mode', 'learned') != 'planned':
            return learned_skill(name)
        start = system._begin(name, 'planned')
        if name == 'grasp':
            ok = system.controller.execute_grasp(system.controller.grasp)
        elif name == 'carry':
            try:
                system.controller.carry()
            except RuntimeError as exc:
                raise RuntimeError(f'carry: {exc}') from exc
            ok = bool(native.contacts()['bilateral_contact']) and native.drop_events == 0
        else:
            raise ValueError(f'No planned version of {name}')
        system._finish(name, start, ok)
        if not ok:
            raise RuntimeError(f'{name}: lost_grasp')
    system._skill = skill
    try:
        return system.run()
    except Exception as exc:
        return {'success': False, 'failed_stage': system.stage, 'error': f'{type(exc).__name__}: {exc}',
                'stages': system.stage_results, 'policy_decisions': system.policy_decisions}


def evaluate_case(case, execution):
    started = time.monotonic()
    native = None
    try:
        native = PickNavReachEnv(**case['env_kwargs'], physics=execution['physics'])
        if case['maze_hash'] is not None and native.scene_manifest()['maze_hash'] != case['maze_hash']:
            raise ValueError('Native layout differs from the case maze hash')
        if execution['controller'] == 'reference':
            result = run_reference(native, execution)
        else:
            result = run_learned(native, execution)
    except Exception as exc:  # environment construction or unexpected errors count as failures
        result = {'success': False, 'failed_stage': 'environment_setup' if native is None else 'unexpected',
                  'error': f'{type(exc).__name__}: {exc}'}
    calls = native.step_count if native is not None else 0
    maze_hash = native.scene_manifest()['maze_hash'] if native is not None else None
    ticks = native.physics_steps if native is not None else 0
    settle = native.settle_steps if native is not None else 0
    if native is not None:
        native.close()
    success = bool(result.get('success'))
    total = ticks + settle
    return {'case_id': case['case_id'], 'set': case['set'], 'layout_id': case['layout_id'],
            'object_idx': case['object_idx'], 'success': success,
            'timed': bool(success and total <= execution['deadline_physics_ticks']),
            'failed_stage': result.get('failed_stage'), 'error': result.get('error'),
            'physics_ticks': ticks, 'total_ticks_including_settling': total,
            'stage_ticks': {k: v.get('physics_ticks') for k, v in result.get('stages', {}).items()},
            'policy_decisions': result.get('policy_decisions', {}),
            'drop_events': result.get('metrics', {}).get('drop_events'),
            'grasp': result.get('stages', {}).get('planned_approach', {}).get('grasp'),
            'pick_pose_error': result.get('pick_pose_error'),
            'maze_hash': maze_hash, 'source_action_calls': calls,
            'first_lift_physics_tick': result.get('first_lift_physics_tick'),
            'pick_hold_ok': result.get('stages', {}).get('pick_hold', {}).get('success'),
            'first_complete_reach_physics_tick': result.get('first_complete_reach_physics_tick'),
            'upstream_success': result.get('metrics', {}).get('success'),
            'wall_seconds': time.monotonic() - started}


def failure_label(row):
    if row['success']:
        return 'success'
    error = (row.get('error') or '').split(':')
    return ':'.join(part.strip() for part in error[:2]) if error else str(row.get('failed_stage'))


def summarize(rows):
    def block(items):
        n = len(items)
        return {'cases': n, 'success': sum(r['success'] for r in items), 'timed': sum(r['timed'] for r in items),
                'success_rate': sum(r['success'] for r in items) / max(n, 1),
                'timed_rate': sum(r['timed'] for r in items) / max(n, 1)}
    by_set, by_object = defaultdict(list), defaultdict(list)
    for row in rows:
        by_set[row['set']].append(row)
        by_object[OBJECT_NAMES.get(row['object_idx'], str(row['object_idx']))].append(row)
    successes = [r['total_ticks_including_settling'] for r in rows if r['success']]
    return {'overall': block(rows), 'by_set': {k: block(v) for k, v in sorted(by_set.items())},
            'by_object': {k: block(v) for k, v in by_object.items()},
            'failures': dict(Counter(failure_label(r) for r in rows if not r['success']).most_common()),
            'median_ticks_successful': float(np.median(successes)) if successes else None}


def format_summary(summary):
    o = summary['overall']
    lines = [f"overall {o['success']}/{o['cases']} complete ({o['success_rate']:.1%}), "
             f"{o['timed']}/{o['cases']} within deadline ({o['timed_rate']:.1%}); "
             f"median ticks {summary['median_ticks_successful']}"]
    lines += [f"  {k:10s} {v['success']:3d}/{v['cases']:<3d} complete  {v['timed']:3d} timed" for k, v in summary['by_set'].items()]
    lines += [f"  {k:10s} {v['success']:3d}/{v['cases']:<3d} complete  {v['timed']:3d} timed" for k, v in summary['by_object'].items()]
    lines += [f"  failure {k}: {v}" for k, v in summary['failures'].items()]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidate', type=Path, help='Candidate JSON with checkpoints and execution settings')
    for skill in ('grasp', 'carry', 'reach'):
        parser.add_argument('--' + skill, type=Path, help='Model ZIP overriding the candidate')
    parser.add_argument('--set', choices=sorted(SETS), default='dev')
    parser.add_argument('--layout-limit', type=int, help='First N layouts of each bank (quick checks)')
    parser.add_argument('--objects', type=int, nargs='+', default=OBJECTS)
    parser.add_argument('--allow-reserve', action='store_true')
    parser.add_argument('--random-object-pose', action='store_true', help='Seeded object start offset and yaw per case')
    parser.add_argument('--workers', type=int, default=44)
    parser.add_argument('--output', type=Path, required=True, help='New run directory, e.g. runs/eval_x')
    parser.add_argument('--label', default='')
    for key, value in DEFAULT_EXECUTION.items():
        kind = str if isinstance(value, str) else type(value)
        parser.add_argument('--' + key.replace('_', '-'), type=kind, default=None)
    args = parser.parse_args()
    if args.set == 'reserve' and not args.allow_reserve:
        parser.error('The final reserve is sealed; pass --allow-reserve only for a final claim')

    execution, checkpoints = dict(DEFAULT_EXECUTION), {}
    if args.candidate:
        candidate = json.loads(args.candidate.read_text())
        execution.update(candidate.get('execution', {}))
        checkpoints = {k: v['frozen_path'] for k, v in candidate['checkpoints'].items()}
    for key in DEFAULT_EXECUTION:
        if getattr(args, key) is not None:
            execution[key] = getattr(args, key)
    for skill in ('grasp', 'carry', 'reach'):
        if getattr(args, skill):
            checkpoints[skill] = str(getattr(args, skill).resolve())
    if execution['controller'] not in ('learned', 'reference'):
        parser.error('--controller must be learned or reference')
    if execution['controller'] == 'reference':
        checkpoints = {}
    elif set(checkpoints) != {'grasp', 'carry', 'reach'}:
        parser.error('Provide --candidate or all three checkpoints')

    cases = select_cases(args.set, args.objects, args.layout_limit, args.random_object_pose)
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory already exists')
    logger = RunLogger(output.name, root=output.parent, kind='evaluation', planned_cases=len(cases),
                       name=args.label or f'Full task · {args.set}', summary=f'{len(cases)} original-start cases')
    config = {'set': args.set, 'cases': len(cases), 'checkpoints': checkpoints, 'execution': execution,
              'label': args.label, 'command': sys.argv}
    atomic_json(output / 'config.json', config)
    initialize_worker(checkpoints)  # fail fast on unloadable checkpoints
    rows, started = [], time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)), mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(checkpoints,)) as pool, \
                (output / 'results.jsonl').open('w') as stream:
            futures = [pool.submit(evaluate_case, case, execution) for case in cases]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if len(rows) % 10 == 0 or len(rows) == len(cases):
                    o = summarize(rows)['overall']
                    logger.metric(transitions=len(rows), scope='full_task', eval_episodes=len(rows),
                                  success_rate=o['success_rate'], success_rate_within_deadline=o['timed_rate'])
        summary = summarize(rows)
        summary.update(config=config, wall_seconds=time.monotonic() - started)
        atomic_json(output / 'summary.json', summary)
        text = format_summary(summary)
        print(text, flush=True)
        o = summary['overall']
        logger.finish(summary=f"{o['success']}/{o['cases']} complete; {o['timed']}/{o['cases']} within 15k ticks")
    except BaseException as exc:
        logger.finish('failed', str(exc))
        raise


if __name__ == '__main__':
    main()
