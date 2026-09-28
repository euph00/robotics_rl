#!/usr/bin/env python3
"""Counted physical multi-object grasp demonstrations; all failures retained."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect_case(config):
    import numpy as np
    from pnr.general_manipulation import GeneralGraspEnv, ContactGraspTeacher
    from pnr.env import PickNavReachEnv
    from pnr.reference import ReferenceController
    from pnr.telemetry import atomic_json
    from scripts.generate_grasp_demos import save_dataset
    started = time.monotonic()
    object_idx, scene_seed = config['object_idx'], config['scene_seed']
    prefix = f'object_{object_idx:02d}_pointseed_{scene_seed}'
    output = Path(config['output'])
    metadata = {'object_idx': object_idx, 'scene_seed': scene_seed, 'episodes': 0,
                'setup_source_action_calls': 0, 'setup_physics_ticks': 0,
                'initialization_physics_ticks': 0, 'demonstration_source_action_calls': 0,
                'demonstration_physics_ticks': 0, 'perturbation_source_action_calls': 0,
                'policy_decisions': 0, 'case_status': 'running'}
    episodes, env, native = [], None, None
    try:
        native = PickNavReachEnv(seed=scene_seed, object_idx=object_idx)
        controller = ReferenceController(native, seed=config['seed'])
        metadata['initialization_physics_ticks'] = native.settle_steps
        env = GeneralGraspEnv(native=native, controller=controller,
                              seed=config['seed'], horizon=config['horizon'], evaluation=True)
        metadata.update(setup_source_action_calls=env.setup_action_calls,
                        setup_physics_ticks=env.setup_action_calls*env.native.substeps,
                        initialization_physics_ticks=env.native.settle_steps,
                        scene=env.native.scene_manifest(), object_name=env.native.object_name,
                        proposal=env.controller.grasp.as_dict(),
                        observation_version=env.observation_version)
        teacher = ContactGraspTeacher(env)
        rng = np.random.default_rng(config['seed']+scene_seed+100*object_idx)
        for episode_idx in range(config['repetitions']*len(env.snapshots)):
            seed = config['seed']+scene_seed+100*object_idx+episode_idx
            env.reset(seed=seed)
            height, snapshot = env.snapshots[episode_idx % len(env.snapshots)]
            env.native.restore_state(snapshot, reset_counters=True)
            env.goal = env.base_goal.copy()
            lift = float(rng.uniform(*config['lift_range']))
            env.goal[2] += lift-.20
            env.goal[:2] += rng.uniform(-.015, .015, 2)
            jitter = np.r_[rng.uniform(-config['jitter_xy'], config['jitter_xy'], 2), 0.]
            perturbation_error = None
            before = env.native.step_count
            if np.any(jitter):
                ee, orientation = env.controller.planner.ee_pose()
                try:
                    env.controller.move_ee(ee+jitter, orientation, stage='training_start_perturbation', closed=False)
                except (ValueError, RuntimeError) as error:
                    perturbation_error = f'{type(error).__name__}: {error}'
            perturbation_calls = env.native.step_count-before
            metadata['perturbation_source_action_calls'] += perturbation_calls
            observation = env._observation()
            transitions = []
            initial_object = env.native.observe()['object_pos'].copy()
            first_goal = None
            before = env.native.step_count
            for step in range(config['horizon']):
                action, phase = teacher.action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                info = copy.deepcopy(info)
                info['TimeLimit.truncated'] = bool(truncated and not terminated)
                transitions.append({'episode': episode_idx, 'step': step,
                                    'observation': copy.deepcopy(observation), 'action': action.copy(),
                                    'next_observation': copy.deepcopy(next_observation),
                                    'reward': float(reward), 'terminated': bool(terminated),
                                    'truncated': bool(truncated), 'info': info, 'teacher_phase': phase})
                if info['is_success'] and first_goal is None:
                    first_goal = step+1
                observation = next_observation
                if terminated or truncated:
                    break
            metrics = env.native.evaluate()
            passed = bool(first_goal is not None and metrics['lifted'] and metrics['bilateral_contact']
                          and metrics['stable_hold_steps'] >= 50 and metrics['drop_events'] == 0
                          and transitions[-1]['info']['goal_distance'] < env.tolerance)
            summary = {'episode': episode_idx, 'object_idx': object_idx,
                       'object_name': env.native.object_name, 'scene_seed': scene_seed, 'seed': seed,
                       'start_height_m': float(height), 'goal_lift_m': lift, 'start_jitter_m': jitter.tolist(),
                       'perturbation_error': perturbation_error,
                       'perturbation_source_action_calls': perturbation_calls,
                       'successful_physical_lift_and_hold': passed,
                       'first_goal_decision': first_goal, 'decisions': len(transitions),
                       'source_action_calls': env.native.step_count-before,
                       'physics_ticks': (env.native.step_count-before)*env.native.substeps,
                       'initial_object_position': initial_object.tolist(), 'desired_goal': env.goal.tolist(),
                       'final_goal_distance_m': transitions[-1]['info']['goal_distance'],
                       'action_rejections': sum(t['info']['action_rejected'] for t in transitions),
                       'final_metrics': metrics}
            episodes.append({'summary': summary, 'transitions': transitions})
            metadata.update(episodes=len(episodes), policy_decisions=sum(len(e['transitions']) for e in episodes),
                            demonstration_source_action_calls=sum(e['summary']['source_action_calls'] for e in episodes),
                            demonstration_physics_ticks=sum(e['summary']['physics_ticks'] for e in episodes),
                            wall_seconds=time.monotonic()-started)
            save_dataset(output/(prefix+'.pkl'), episodes, metadata)
            atomic_json(output/(prefix+'.json'), metadata | {'episodes': [e['summary'] for e in episodes]})
        metadata['case_status'] = 'complete'
    except BaseException as error:
        metadata.update(case_status='failed', error=f'{type(error).__name__}: {error}')
        if native is not None and not metadata['setup_source_action_calls']:
            metadata['setup_source_action_calls'] = native.step_count
            metadata['setup_physics_ticks'] = native.step_count*native.substeps
    finally:
        metadata['wall_seconds'] = time.monotonic()-started
        save_dataset(output/(prefix+'.pkl'), episodes, metadata)
        summary = metadata | {'episodes': [e['summary'] for e in episodes], 'pickle_file': prefix+'.pkl'}
        atomic_json(output/(prefix+'.json'), summary)
        if native is not None:
            native.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='phase2_grasp_demos01')
    parser.add_argument('--objects', nargs='+', type=int, default=[14, 5, 2, 7, 10])
    parser.add_argument('--scene-seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--jitter-xy', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=61000)
    parser.add_argument('--lift-range', nargs=2, type=float, default=[.20, .20])
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 <= args.jitter_xy <= .02 or not .15 <= args.lift_range[0] <= args.lift_range[1] <= .30:
        raise ValueError('Use at most eight workers and two centimeters start jitter')
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    from pnr.telemetry import RunLogger, atomic_json
    from scripts.generate_grasp_demos import save_dataset
    logger = RunLogger(args.run_id, kind='data', skill='grasp',
                       algorithm='Physical training demonstrations',
                       name='Phase 2 · multi-object grasp demonstrations',
                       budget=len(args.objects)*len(args.scene_seeds)*args.repetitions*4*args.horizon,
                       summary='Real approach/close/lift transitions over the declared five-object bank; no policy training.')
    artifacts = logger.directory/'artifacts'
    artifacts.mkdir(exist_ok=True)
    configuration = vars(args) | {'output': str(artifacts), 'training_only': True,
                                 'policy_observation': GeneralObservationDescription,
                                 'no_hidden_maze_evaluation': True}
    source_files = ['scripts/phase2_grasp_demos.py', 'pnr/general_manipulation.py', 'pnr/planning.py',
                    'pnr/skills.py', 'pnr/env.py', 'pnr/reference.py', 'pnr/local_control.py']
    configuration['source_hashes'] = {}
    for name in source_files:
        content = (ROOT/name).read_bytes()
        configuration['source_hashes'][name] = hashlib.sha256(content).hexdigest()
        (artifacts/name.replace('/', '__')).write_bytes(content)
    atomic_json(logger.directory/'config.json', configuration)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(collect_case, configuration | {'object_idx': index, 'scene_seed': seed})
                       for index in args.objects for seed in args.scene_seeds]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                episodes = [e for r in results for e in r['episodes']]
                logger.metric(sum(r['policy_decisions'] for r in results), scope='training_data',
                              cases_completed=len(results), episodes=len(episodes),
                              successful_episodes=sum(e['successful_physical_lift_and_hold'] for e in episodes),
                              success_rate=sum(e['successful_physical_lift_and_hold'] for e in episodes)/max(1, len(episodes)),
                              setup_source_action_calls=sum(r['setup_source_action_calls'] for r in results),
                              perturbation_source_action_calls=sum(r['perturbation_source_action_calls'] for r in results),
                              demonstration_source_action_calls=sum(r['demonstration_source_action_calls'] for r in results))
                atomic_json(artifacts/'cases.json', {'configuration': configuration, 'results': results})
                print(json.dumps(result), flush=True)
        all_episodes = []
        for result in results:
            with (artifacts/result['pickle_file']).open('rb') as handle:
                all_episodes.extend(pickle.load(handle)['episodes'])
        for index, episode in enumerate(all_episodes):
            episode['summary']['dataset_episode'] = index
            for transition in episode['transitions']:
                transition['episode'] = index
        metadata = {'configuration': configuration, 'cases': results,
                    'collection_status': 'complete', 'observation_version': 'antipodal-relative-grasp-v1',
                    'demonstration_source_action_calls': sum(r['demonstration_source_action_calls'] for r in results),
                    'demonstration_physics_ticks': sum(r['demonstration_physics_ticks'] for r in results),
                    'setup_source_action_calls': sum(r['setup_source_action_calls'] for r in results),
                    'setup_physics_ticks': sum(r['setup_physics_ticks'] for r in results),
                    'initialization_physics_ticks': sum(r['initialization_physics_ticks'] for r in results),
                    'perturbation_source_action_calls': sum(r['perturbation_source_action_calls'] for r in results),
                    'policy_decisions': sum(len(e['transitions']) for e in all_episodes),
                    'all_attempts_retained': True}
        digest = save_dataset(artifacts/'demonstrations.pkl', all_episodes, metadata)
        atomic_json(artifacts/'manifest.json', metadata | {'dataset_sha256': digest,
                    'episodes': [e['summary'] for e in all_episodes]})
        logger.artifact('artifacts/demonstrations.pkl', 'Actual multi-object replay transitions')
        logger.artifact('artifacts/manifest.json', 'Physical outcomes, counts, observation contract')
        successful = sum(e['summary']['successful_physical_lift_and_hold'] for e in all_episodes)
        logger.finish(summary=f'Collected {len(all_episodes)} physical episodes: {successful} lift/hold successes; every failed attempt retained.')
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise


GeneralObservationDescription = ('Original50 measured state values + live worldXYZ antipodal target-minus-EE + proposal width; '
    'canonical achieved/desired object goals unchanged; no object identity, teacher actions or hidden stage state.')

if __name__ == '__main__':
    main()
