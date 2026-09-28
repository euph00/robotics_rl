#!/usr/bin/env python3
"""Bounded real-physics manipulation profile; no training or success claims."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from pnr.policies import load_policy_checkpoint
from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json


def function_rows(profile):
    rows = []
    for (filename, line, function), (primitive, calls, exclusive, cumulative, callers) in pstats.Stats(profile).stats.items():
        rows.append({'file':filename, 'line':line, 'function':function,
                     'primitive_calls':primitive, 'calls':calls,
                     'exclusive_seconds':exclusive, 'inclusive_seconds':cumulative})
    return sorted(rows, key=lambda row:row['inclusive_seconds'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='grasp_simulator_profile')
    parser.add_argument('--checkpoint', default='runs/sac_grasp_demoseed01/checkpoints/best/model.zip')
    parser.add_argument('--decisions', type=int, default=100)
    args = parser.parse_args()
    if not 2 <= args.decisions <= 100:
        raise ValueError('This diagnostic permits at most 100 physical decisions')
    torch.set_num_threads(1)
    logger = RunLogger(args.run_id, kind='validation', skill='grasp', algorithm='cProfile',
        name='Physical simulator cost profile', budget=args.decisions,
        summary='Bounded frozen-policy physics profiling; no learning or competence evaluation.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    env = None
    started = time.perf_counter()
    records = []
    profiler = cProfile.Profile()
    checkpoint = ROOT/args.checkpoint
    frozen = artifacts/'profile_policy.zip'
    shutil.copyfile(checkpoint, frozen)
    sha256 = hashlib.sha256(frozen.read_bytes()).hexdigest()
    atomic_json(logger.directory/'config.json', vars(args) | {
        'training':False, 'checkpoint_sha256':sha256, 'device':'cpu', 'torch_threads':1,
        'scope':'env.step only; policy inference, physical initial setup and resets excluded',
        'starts':f'{args.decisions//2} easy .008m and {args.decisions-args.decisions//2} full .17m pregrasp decisions',
        'limitations':'One process while other training runs remain active; profiler overhead included.'})
    try:
        model = load_policy_checkpoint(frozen, device='cpu')
        env = ManipulationSkillEnv(skill='grasp', scene_seed=42, object_idx=14,
                                   seed=31415, horizon=100, evaluation=True)
        setup_calls = env.setup_action_calls
        setup_seconds = time.perf_counter()-started
        # BulletClient exposes builtins through functools.partial, which cProfile
        # folds into callers. These wrappers separately measure the native calls.
        bullet_times = {}
        profiling_active = False
        def wrap_bullet(name):
            original = getattr(env.native.p, name)
            summary = bullet_times.setdefault(name, {'calls':0, 'seconds':0.})
            def measured(*args, **kwargs):
                if not profiling_active:
                    return original(*args, **kwargs)
                begin = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    summary['calls'] += 1
                    summary['seconds'] += time.perf_counter()-begin
            setattr(env.native.p, name, measured)
        for name in ('saveState','restoreState','removeState','stepSimulation',
                     'performCollisionDetection','getClosestPoints','calculateJacobian',
                     'resetJointState','getLinkState','getJointStates',
                     'getContactPoints','getBasePositionAndOrientation'):
            wrap_bullet(name)
        for group, count in enumerate((args.decisions//2, args.decisions-args.decisions//2)):
            observation, reset_info = env.reset(seed=31415+group,
                                                options={'full_pregrasp':bool(group)})
            for group_step in range(count):
                action, _ = model.predict(observation, deterministic=True)
                before_calls, before_ticks = env.native.step_count, env.native.physics_steps
                begin = time.perf_counter()
                profiling_active = True
                profiler.enable()
                try:
                    observation, reward, terminated, truncated, info = env.step(action)
                finally:
                    profiler.disable()
                    profiling_active = False
                elapsed = time.perf_counter()-begin
                records.append({'decision':len(records)+1, 'start_height_m':reset_info['start_height'],
                    'group_step':group_step, 'step_seconds':elapsed, 'action':action.tolist(),
                    'action_rejected':bool(info['action_rejected']),
                    'local_ik_iterations':env.local_ik.last_iterations,
                    'bilateral_contact':bool(info['bilateral_contact']),
                    'terminated':bool(terminated), 'truncated':bool(truncated),
                    'source_action_calls':env.native.step_count-before_calls,
                    'physics_ticks':env.native.physics_steps-before_ticks})
                if terminated or truncated:
                    observation, reset_info = env.reset(seed=31415+group+100+group_step,
                                                        options={'full_pregrasp':bool(group)})
                if len(records) % 25 == 0:
                    logger.metric(len(records), scope='performance', metric_source='profiling',
                        decisions_per_second=len(records)/sum(r['step_seconds'] for r in records))
        profiler.dump_stats(str(artifacts/'env_step.prof'))
        with (artifacts/'profile.txt').open('w') as output:
            stats = pstats.Stats(profiler, stream=output)
            stats.strip_dirs().sort_stats('cumulative').print_stats(65)
            stats.sort_stats('tottime').print_stats(35)
        rows = function_rows(profiler)
        step_seconds = sum(r['step_seconds'] for r in records)
        counts = {'policy_decisions':len(records),
                  'demonstration_source_action_calls':sum(r['source_action_calls'] for r in records),
                  'demonstration_physics_ticks':sum(r['physics_ticks'] for r in records),
                  'setup_source_action_calls':setup_calls,
                  'setup_physics_ticks':setup_calls*env.native.substeps,
                  'initialization_physics_ticks':env.native.settle_steps}
        # This is performance validation, not training demonstrations. The
        # explicit totals below prevent these extra interactions disappearing.
        counts['profile_source_action_calls'] = counts.pop('demonstration_source_action_calls')
        counts['profile_physics_ticks'] = counts.pop('demonstration_physics_ticks')
        counts['total_source_action_calls'] = counts['setup_source_action_calls']+counts['profile_source_action_calls']
        counts['total_physics_ticks'] = counts['setup_physics_ticks']+counts['profile_physics_ticks']+counts['initialization_physics_ticks']
        report = {'training':False, 'competence_evaluation':False,
            'checkpoint':str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256':sha256,
            'counts':counts, 'setup_and_policy_load_seconds':setup_seconds,
            'profiled_step_seconds':step_seconds, 'profiled_decisions_per_second':len(records)/step_seconds,
            'step_time_ms_quantiles':dict(zip(('p10','p50','p90','p99'),
                (1000*np.quantile([r['step_seconds'] for r in records],[.1,.5,.9,.99])).tolist())),
            'action_rejections':sum(r['action_rejected'] for r in records),
            'bilateral_contact_fraction':float(np.mean([r['bilateral_contact'] for r in records])),
            'resets':env.reset_count, 'wall_seconds':time.perf_counter()-started,
            'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('pnr/env.py','pnr/skills.py','pnr/local_control.py','pnr/planning.py')},
            'profile_functions':rows, 'bullet_method_timings':bullet_times,
            'interpretation_limit':'A bounded instrumented simulator workload; no training or success claims. Inclusive costs overlap.'}
        atomic_json(artifacts/'profile.json', report)
        atomic_json(artifacts/'decisions.json', records)
        logger.artifact('artifacts/profile.json', 'Simulator timings and physical interaction counts')
        logger.artifact('artifacts/profile.txt', 'Python and Bullet call cost profile')
        logger.artifact('artifacts/decisions.json', 'Actual profiled physical decisions')
        logger.finish(summary=f"{len(records)} real decisions, {1000*step_seconds/len(records):.1f} ms/decision with profiling. No learning.")
        print(json.dumps({key:value for key,value in report.items() if key not in ('profile_functions','bullet_method_timings')}, indent=2), flush=True)
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
