#!/usr/bin/env python3
"""Evaluate the phase-2 original-gripper system from uninterrupted original starts.

Development partitions are train/validation; course_legacy remains separate.
Held-out evaluation requires an explicit flag AND a previously frozen evaluation
plan with verified checkpoint and scene-bank hashes. --freeze-only prepares that
plan without running any physics or policy evaluations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.general_manipulation import GeneralGraspEnv
from pnr.policies import load_policy_checkpoint
from pnr.scenes import HELDOUT_PARTITIONS, case_specs, load_scene_bank
from pnr.system import LearnedSystem
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import freeze_checkpoints, plain, resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS = [14, 5, 2, 7, 10]
PLAN_VERSION = 'phase2-original-gripper-system-evaluation-v3'
PREVIOUS_PLAN_VERSION = 'phase2-original-gripper-system-evaluation-v2'
LEGACY_PLAN_VERSION = 'phase2-original-gripper-system-evaluation-v1'
_WORKER_POLICIES = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plan_digest(plan):
    data = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def select_cases(bank, partition, objects, *, allow_heldout=False, legacy_seeds=(42,),
                 layout_limit=None, case_limit=None):
    objects = [int(value) for value in objects]
    if not objects or len(set(objects)) != len(objects) or any(not 0 <= value < 15 for value in objects):
        raise ValueError('Provide distinct supplied object indices')
    if partition in HELDOUT_PARTITIONS:
        if not allow_heldout:
            raise ValueError('Held-out evaluation requires --allow-heldout')
        if layout_limit is not None or case_limit is not None:
            raise ValueError('Held-out evaluation must include the complete frozen partition')
        if set(objects) != set(DEFAULT_OBJECTS):
            raise ValueError('Held-out evaluation requires all five predeclared phase-2 objects')
    if partition == 'course_legacy':
        if layout_limit is not None:
            raise ValueError('Use --legacy-seeds rather than --layout-limit for course regression')
        cases = [
            {'case_id': f'course_legacy_seed_{int(seed)}_object_{obj:02d}',
             'layout_id': None, 'partition': 'course_legacy', 'maze_hash': None, 'object_idx': obj,
             'env_kwargs': {'seed': int(seed), 'object_idx': obj, 'seed_mode': 'course-legacy-v1'}}
            for seed in legacy_seeds for obj in objects]
    else:
        cases = case_specs(bank, partition, object_indices=objects, allow_heldout=allow_heldout)
        if layout_limit is not None:
            if layout_limit < 1:
                raise ValueError('--layout-limit must be positive')
            layouts = {r['layout_id'] for r in bank['partitions'][partition][:layout_limit]}
            cases = [case for case in cases if case['layout_id'] in layouts]
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError('--case-limit must be positive')
        cases = cases[:case_limit]
    if not cases or len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Case list must be nonempty with no duplicates')
    return cases


def verify_frozen_plan(plan, bank, *, allow_heldout=False, verify_checkpoints=True):
    if plan.get('version') not in (PLAN_VERSION, PREVIOUS_PLAN_VERSION, LEGACY_PLAN_VERSION) or plan.get('status') != 'frozen':
        raise ValueError('Evaluation requires a versioned frozen plan')
    if plan.get('plan_sha256') != plan_digest(plan):
        raise ValueError('Frozen evaluation plan digest mismatch')
    if plan.get('scene_bank_sha256') != bank['manifest_sha256']:
        raise ValueError('Frozen plan and scene bank digests differ')
    partition = plan['partition']
    if partition in HELDOUT_PARTITIONS:
        expected = select_cases(bank, partition, plan['object_indices'], allow_heldout=allow_heldout)
        if expected != plan['cases']:
            raise ValueError('Held-out case list differs from the complete frozen partition')
    else:
        expected = select_cases(bank, partition, plan['object_indices'],
                                legacy_seeds=plan.get('legacy_seeds', [42]))
        available = {case['case_id']: case for case in expected}
        if not plan['cases'] or any(available.get(case['case_id']) != case for case in plan['cases']):
            raise ValueError('Development cases do not match the declared bank')
    if len({case['case_id'] for case in plan['cases']}) != len(plan['cases']):
        raise ValueError('Duplicate frozen evaluation cases')
    for flag, expected in {'original_start': True, 'inter_skill_resets': False,
                           'teacher_at_inference': False, 'original_gripper': True,
                           'retain_all_failures': True}.items():
        if plan.get(flag) is not expected:
            raise ValueError('Frozen plan must preserve the original uninterrupted physical task')
    settings = plan['execution']
    required = {'delta_scale', 'planned_speed_scale', 'deadline_physics_ticks',
                'grasp_horizon', 'carry_horizon', 'reach_horizon'}
    if plan['version'] != LEGACY_PLAN_VERSION: required.add('grasp_lift_m')
    if plan['version'] == PLAN_VERSION: required.add('reach_tolerance_m')
    if set(settings) != required or any(not np.isfinite(value) or value <= 0 for value in settings.values()):
        raise ValueError('Frozen execution settings must declare positive horizons, scales and deadline')
    if any(int(settings[key]) != settings[key] for key in required - {'delta_scale', 'planned_speed_scale', 'grasp_lift_m', 'reach_tolerance_m'}):
        raise ValueError('Frozen horizons and deadline must be integer counts')
    if not .15 <= settings.get('grasp_lift_m', .20) <= .30:
        raise ValueError('Grasp lift subgoal must be between15 and30cm')
    if settings.get('reach_tolerance_m', .04) not in (.04, .10):
        raise ValueError('Reach tolerance must be native 0.10 m or legacy 0.04 m')
    if set(plan['checkpoints']) != {'grasp', 'carry', 'reach'}:
        raise ValueError('Frozen plan needs exactly three checkpoint records')
    for name, record in plan['checkpoints'].items():
        if not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64:
            raise ValueError(f'{name} checkpoint digest missing')
        if verify_checkpoints and sha256_file(record['frozen_path']) != record['sha256']:
            raise ValueError(f'{name} frozen checkpoint hash mismatch')
    return plan


def execution_settings(plan):
    """Expand verified legacy plans without silently changing their behavior."""
    settings = dict(plan['execution'])
    settings.setdefault('grasp_lift_m', .20)
    settings.setdefault('reach_tolerance_m', .04)
    return settings


def deadline_metrics(result, settling_ticks, deadline=15000):
    """Separate final retained success from first-reach and complete-hold timing."""
    success = bool(result.get('success', False))
    ticks = int(result.get('physics_ticks', 0))
    reach = result.get('first_complete_reach_physics_tick')
    first_lift = result.get('first_lift_physics_tick')
    grasped_lift_action = result.get('first_grasped_lift_action')
    grasped_lift = None if grasped_lift_action is None else int(grasped_lift_action) * 5
    within = lambda value: bool(success and value is not None and value <= deadline)
    return {
        'deadline_physics_ticks': int(deadline), 'initialization_physics_ticks': int(settling_ticks),
        'total_physics_ticks_including_initialization': ticks + int(settling_ticks),
        'first_grasped_lift_physics_tick': grasped_lift,
        'first_lift_physics_tick_including_initialization': None if first_lift is None else first_lift + settling_ticks,
        'first_grasped_lift_physics_tick_including_initialization': None if grasped_lift is None else grasped_lift + settling_ticks,
        'first_complete_reach_physics_tick_including_initialization': None if reach is None else reach + settling_ticks,
        'task_success': success,
        'success_within_deadline': within(ticks),
        'success_within_deadline_including_initialization': within(ticks + settling_ticks),
        'reached_within_deadline': within(reach),
        'reached_within_deadline_including_initialization': within(None if reach is None else reach + settling_ticks),
        'deadline_primary_definition': 'Original task start through successful final 60-call retained hold; settling reported separately',
    }


def summarize_results(results, planned_cases):
    n = len(planned_cases)
    by_id = {result['case_id']: result for result in results}
    if len(by_id) != len(results) or set(by_id) - {case['case_id'] for case in planned_cases}:
        raise ValueError('Results contain duplicate or undeclared cases')
    successes = sum(bool(result.get('success')) for result in results)
    on_time = sum(bool(result.get('success_within_deadline')) for result in results)
    complete_costs = all(result.get('cost_accounting_complete', True) for result in results)
    return {
        'scope': 'full_task', 'planned_cases': n, 'completed_cases': len(results),
        'missing_case_ids': [case['case_id'] for case in planned_cases if case['case_id'] not in by_id],
        'successes': successes, 'success_rate': successes / n,
        'successes_within_deadline': on_time, 'success_rate_within_deadline': on_time / n,
        'successes_within_deadline_including_initialization': sum(bool(r.get('success_within_deadline_including_initialization')) for r in results),
        'first_reach_within_deadline': sum(bool(r.get('reached_within_deadline')) for r in results),
        'source_action_calls': sum(r.get('source_action_calls', 0) for r in results),
        'physics_ticks': sum(r.get('physics_ticks', 0) for r in results),
        'initialization_physics_ticks': sum(r.get('initialization_physics_ticks', 0) for r in results),
        'total_physics_ticks_including_initialization': sum(r.get('total_physics_ticks_including_initialization', 0) for r in results),
        'cost_accounting_complete': complete_costs, 'retain_all_failures': True,
        'failed_cases': [{'case_id': r['case_id'], 'stage': r.get('failed_stage'), 'error': r.get('error')}
                         for r in results if not r.get('success')],
    }


def validate_policy_interfaces(policies):
    for name, observation_dim, goal_dim, action_dim in (('grasp', 54, 3, 4), ('carry', 580, 2, 2), ('reach', 50, 3, 3)):
        policy = policies[name]
        if (policy.action_space.shape != (action_dim,)
                or policy.observation_space['observation'].shape != (observation_dim,)
                or policy.observation_space['achieved_goal'].shape != (goal_dim,)
                or policy.observation_space['desired_goal'].shape != (goal_dim,)):
            raise ValueError(f'{name} checkpoint does not match the declared phase-2 observation/action interface')


def initialize_worker(records, device='cpu', torch_threads=1):
    global _WORKER_POLICIES
    import torch
    torch.set_num_threads(torch_threads)
    _WORKER_POLICIES = {name: load_policy_checkpoint(record['frozen_path'], device=device)
                        for name, record in records.items()}
    validate_policy_interfaces(_WORKER_POLICIES)


def failure_result(case, exc, stage='environment_setup', native=None):
    result = {'kind': 'learned_system', 'scope': 'full_task', 'success': False, 'status': 'failed',
              'failed_stage': stage, 'error': f'{type(exc).__name__}: {exc}',
              'source_action_calls': native.step_count if native is not None else 0,
              'physics_ticks': native.physics_steps if native is not None else 0,
              'cost_accounting_complete': native is not None, 'stages': {},
              'first_complete_reach_physics_tick': None, 'case_id': case['case_id']}
    if native is not None:
        result['metrics'] = native.evaluate()
        result['scene_manifest'] = native.scene_manifest()
    else:
        result['cost_accounting_note'] = 'Failure before a live native environment was available; any partial construction cost is unknown'
    return result


def evaluate_case(case, execution, artifact_dir, video=False):
    """Each worker owns its client and records every failed original-start case."""
    if _WORKER_POLICIES is None:
        raise RuntimeError('Policy worker was not initialized')
    artifact_dir = Path(artifact_dir)
    case_id = case['case_id']
    trace_path = artifact_dir / f'{case_id}_trajectory.jsonl'
    native = writer = system = None
    media, rendering_error = [], None
    result = None
    started = time.monotonic()
    with trace_path.open('w') as trace:
        try:
            native = PickNavReachEnv(**case['env_kwargs'])
            if case['maze_hash'] is not None and native.scene_manifest()['maze_hash'] != case['maze_hash']:
                raise ValueError('Native layout differs from its frozen case hash')
            if video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(artifact_dir / f'{case_id}.mp4', fps=12, codec='libx264',
                                            quality=7, macro_block_size=8)
            def observe(current, observation, info):
                if native.step_count % 5 == 0:
                    trace.write(json.dumps({'source_action': native.step_count, 'physics_tick': native.physics_steps,
                        'stage': current.display_stage, 'controller': current.control_source,
                        'qpos': observation['qpos'].tolist(), 'object_position': observation['object_pos'].tolist(),
                        'object_quaternion': observation['object_xyzw'].tolist(),
                        'goal_distance': info['dist_to_goal'], 'bilateral_contact': info['bilateral_contact']}) + '\n')
                if writer is not None and native.step_count % 4 == 0:
                    frame = native.render_rgb(camera='follow', width=640, height=360, shadow=False, pitch=-78)
                    from PIL import Image, ImageDraw
                    canvas = Image.fromarray(frame)
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 640, 25), fill=(15, 23, 42))
                    draw.text((8, 7), f'{case_id} | {current.display_stage} | {native.physics_steps} ticks', fill='white')
                    writer.append_data(np.asarray(canvas))
            system = LearnedSystem(native, _WORKER_POLICIES,
                grasp_horizon=execution['grasp_horizon'], carry_horizon=execution['carry_horizon'],
                reach_horizon=execution['reach_horizon'], on_step=observe,
                reach_tolerance_m=execution.get('reach_tolerance_m', .04),
                controller_speed_scale=execution['planned_speed_scale'],
                skill_factories={'grasp': partial(GeneralGraspEnv.bind, goal_lift_m=execution.get('grasp_lift_m', .20)),
                    'carry': partial(GeneralCarryEnv.bind, delta_scale=execution['delta_scale'])})
            result = system.run()
            result['cost_accounting_complete'] = True
        except Exception as exc:
            result = failure_result(case, exc, system.stage if system is not None else 'environment_setup', native)
            if system is not None:
                result.update(stages=system.stage_results, policy_decisions=system.policy_decisions,
                              source_calls_by_controller=system.calls_by_source)
        finally:
            if writer is not None:
                try:
                    writer.close()
                    from pnr.video import webm_copy
                    import imageio.v2 as imageio
                    movie = artifact_dir / f'{case_id}.mp4'
                    portable = webm_copy(movie)
                    preview = artifact_dir / f'{case_id}_preview.png'
                    imageio.imwrite(preview, native.render_rgb(camera='follow', width=640, height=360,
                                                               shadow=False, pitch=-78))
                    media = [{'path': 'artifacts/' + portable.name, 'kind': 'video'},
                             {'path': 'artifacts/' + movie.name, 'kind': 'file'},
                             {'path': 'artifacts/' + preview.name, 'kind': 'image'}]
                except Exception as exc:
                    rendering_error = f'{type(exc).__name__}: {exc}'
            settling = native.settle_steps if native is not None else 0
            if native is not None:
                native.close()
    result.update(case_id=case_id, case=case, partition=case['partition'], execution=execution,
                  wall_seconds_including_setup=time.monotonic() - started, media=media,
                  rendering_error=rendering_error)
    result.update(deadline_metrics(result, settling, execution['deadline_physics_ticks']))
    atomic_json(artifact_dir / f'{case_id}.json', json.loads(json.dumps(result, default=plain)))
    return result


def freeze_sources(directory):
    hashes = {}
    for source in sorted((ROOT / 'pnr').glob('*.py')) + [Path(__file__).resolve(), ROOT / 'scripts/evaluate_system.py']:
        relative = source.relative_to(ROOT)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256_file(target)
    atomic_json(directory / 'manifest.json', hashes)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for skill in ('grasp', 'carry', 'reach'):
        parser.add_argument('--' + skill, help='Checkpoint ZIP or existing run/generation')
    parser.add_argument('--bank', type=Path, default=ROOT / 'configs/phase2_scene_bank.json')
    parser.add_argument('--partition', choices=['train', 'validation', 'milestone_test', 'final_reserve', 'course_legacy'])
    parser.add_argument('--object-indices', type=int, nargs='+', default=DEFAULT_OBJECTS)
    parser.add_argument('--legacy-seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--layout-limit', type=int)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--allow-heldout', action='store_true')
    parser.add_argument('--frozen-config', type=Path)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--grasp-horizon', type=int, default=600)
    parser.add_argument('--grasp-lift-m', type=float, default=.24)
    parser.add_argument('--carry-horizon', type=int, default=1200)
    parser.add_argument('--reach-horizon', type=int, default=150)
    parser.add_argument('--reach-tolerance-m', type=float, choices=(.04, .10), default=.10,
                        help='Native task acceptance (0.10 m), or explicit legacy 0.04 m; final hold required')
    parser.add_argument('--delta-scale', type=float, default=.075)
    parser.add_argument('--planned-speed-scale', type=float, default=1.)
    parser.add_argument('--deadline-physics-ticks', type=int, default=15000)
    parser.add_argument('--video', action='store_true', help='Record the first declared case, including failure')
    parser.add_argument('--algorithm-label', default='SAC+HER phase-2 candidate')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.torch_threads < 1:
        parser.error('Use1..8 workers and at least one Torch thread')
    if args.workers > 1 and args.device != 'cpu':
        parser.error('Parallel evaluation workers require --device cpu')
    bank = load_scene_bank(args.bank)
    if args.frozen_config:
        overrides = {'--object-indices', '--legacy-seeds', '--grasp-horizon', '--carry-horizon',
                     '--reach-horizon', '--delta-scale', '--planned-speed-scale',
                     '--deadline-physics-ticks', '--algorithm-label', '--grasp-lift-m', '--reach-tolerance-m'}
        if overrides & {argument.split('=')[0] for argument in sys.argv[1:]}:
            parser.error('Execution settings and object selection cannot override a frozen plan')
        original = json.loads(args.frozen_config.read_text())
        verify_frozen_plan(original, bank, allow_heldout=args.allow_heldout)
        if original['partition'] in HELDOUT_PARTITIONS:
            if not original.get('source_hashes'):
                parser.error('Held-out frozen plan must include archived evaluation source hashes')
            for relative, expected in original['source_hashes'].items():
                source = (ROOT / relative).resolve()
                if not source.is_relative_to(ROOT) or sha256_file(source) != expected:
                    parser.error('Evaluation source changed since freezing; prepare a new reviewed plan before held-out execution')
        if args.partition is not None and args.partition != original['partition']:
            parser.error('--partition cannot change the frozen plan')
        if args.case_limit is not None or args.layout_limit is not None or any(getattr(args, name) for name in ('grasp', 'carry', 'reach')):
            parser.error('Frozen evaluation forbids case subsets or checkpoint overrides')
        partition, cases, execution = original['partition'], original['cases'], original['execution']
        execution = execution_settings(original)
        objects, algorithm = original['object_indices'], original['algorithm']
        paths = {name: Path(record['frozen_path']) for name, record in original['checkpoints'].items()}
        legacy_seeds = original.get('legacy_seeds', [42])
    else:
        partition = args.partition or 'validation'
        if partition in HELDOUT_PARTITIONS and not args.freeze_only:
            parser.error('Held-out execution requires --allow-heldout and a previously prepared --frozen-config; use --freeze-only first')
        cases = select_cases(bank, partition, args.object_indices, allow_heldout=args.allow_heldout,
                             legacy_seeds=args.legacy_seeds, layout_limit=args.layout_limit, case_limit=args.case_limit)
        if any(getattr(args, name) is None for name in ('grasp', 'carry', 'reach')):
            parser.error('Provide all three checkpoints, or --frozen-config')
        paths = {name: resolve_checkpoint(getattr(args, name)) for name in ('grasp', 'carry', 'reach')}
        execution = {key: getattr(args, key) for key in ('grasp_horizon', 'carry_horizon', 'reach_horizon',
                     'delta_scale', 'planned_speed_scale', 'deadline_physics_ticks', 'grasp_lift_m', 'reach_tolerance_m')}
        objects, algorithm, legacy_seeds = args.object_indices, args.algorithm_label, args.legacy_seeds
    artifact_dir = args.output / 'artifacts'
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory; evaluations never overwrite earlier attempts')
    artifact_dir.mkdir(parents=True)
    records = freeze_checkpoints(paths, artifact_dir / 'checkpoints')
    if args.frozen_config:
        if any(records[name]['sha256'] != original['checkpoints'][name]['sha256'] for name in records):
            raise ValueError('A frozen source checkpoint changed while being copied')
    source_hashes = freeze_sources(artifact_dir / 'source')
    shutil.copy2(args.bank, artifact_dir / 'scene_bank.json')
    plan = {'version': PLAN_VERSION, 'status': 'frozen', 'partition': partition, 'algorithm': algorithm,
            'scene_bank_sha256': bank['manifest_sha256'], 'object_indices': objects,
            'legacy_seeds': legacy_seeds, 'cases': cases, 'execution': execution,
            'checkpoints': records, 'source_hashes': source_hashes,
            'original_start': True, 'inter_skill_resets': False, 'teacher_at_inference': False,
            'original_gripper': True, 'retain_all_failures': True,
            'parent_plan_sha256': original['plan_sha256'] if args.frozen_config else None}
    plan['plan_sha256'] = plan_digest(plan)
    verify_frozen_plan(plan, bank, allow_heldout=args.allow_heldout)
    atomic_json(artifact_dir / 'evaluation_plan.json', plan)
    if args.freeze_only:
        print(json.dumps({'status': 'frozen_without_evaluation', 'cases': len(cases),
                          'plan': str(artifact_dir / 'evaluation_plan.json'), 'sha256': plan['plan_sha256']}), flush=True)
        return
    logger = RunLogger(args.output.name, root=args.output.parent, kind='evaluation', algorithm=algorithm,
        name=f'Phase2 full system · {partition}', partition=partition, planned_cases=len(cases),
        summary=f'{len(cases)} declared original-start cases; all failures retained; deadline reported separately.')
    logger.artifact('artifacts/evaluation_plan.json', 'Frozen evaluation plan')
    results = []
    def record_result(result):
        results.append(result)
        summary = summarize_results(results, cases)
        summary.update(partition=partition, algorithm=algorithm, plan_sha256=plan['plan_sha256'],
                       checkpoint_hashes={name: record['sha256'] for name, record in records.items()})
        atomic_json(artifact_dir / 'summary.json', summary)
        logger.metric(transitions=summary['source_action_calls'], scope='full_task',
                      success_rate=summary['success_rate'], success_rate_within_deadline=summary['success_rate_within_deadline'],
                      eval_episodes=len(results), planned_cases=len(cases), physics_ticks=summary['physics_ticks'])
        for media in result.get('media', []):
            logger.artifact(media['path'], result['case_id'], kind=media['kind'])
        logger.event(f'{result["case_id"]}: '+('success' if result['success'] else 'failed'),
                     failed_stage=result.get('failed_stage'), error=result.get('error'))
        print(json.dumps({'case_id': result['case_id'], 'success': result['success'],
                          'success_within_deadline': result['success_within_deadline'],
                          'failed_stage': result.get('failed_stage'), 'physics_ticks': result['physics_ticks']}), flush=True)
    try:
        # Validate checkpoint interfaces once before launching any physical case.
        initialize_worker(records, args.device, args.torch_threads)
        if args.workers == 1:
            for i, case in enumerate(cases):
                record_result(evaluate_case(case, execution, artifact_dir, args.video and i == 0))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                     initializer=initialize_worker, initargs=(records, 'cpu', args.torch_threads)) as pool:
                futures = {pool.submit(evaluate_case, case, execution, str(artifact_dir), args.video and i == 0): case
                           for i, case in enumerate(cases)}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = failure_result(futures[future], exc, 'worker_execution')
                        result.update(deadline_metrics(result, 0, execution['deadline_physics_ticks']))
                        atomic_json(artifact_dir / f'{result["case_id"]}.json', result)
                    record_result(result)
        from pnr.figures import system_matrix
        system_matrix(results,cases,artifact_dir/'validation_matrix.png',title=f'Full-system {partition}')
        logger.artifact('artifacts/validation_matrix.png','Object and maze outcomes; timing includes settling','image')
        logger.artifact('artifacts/summary.json', 'Full-task and deadline results')
        summary = summarize_results(results, cases)
        logger.finish(summary=f'{summary["successes"]}/{len(cases)} complete; {summary["successes_within_deadline"]}/{len(cases)} within deadline.')
    except KeyboardInterrupt:
        logger.finish('stopped', 'Interrupted; completed results and full declared denominator retained.')
        raise
    except BaseException as exc:
        logger.finish('failed', str(exc))
        raise


if __name__ == '__main__':
    main()
