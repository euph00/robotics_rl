#!/usr/bin/env python3
"""Bounded physical object audit, retaining every attempted reference/policy case.

The reference and frozen policy branch from the same physically reached pregrasp.
Costs count actual execution in both branches and setup once. This is a diagnostic,
not an uninterrupted full-task or held-out navigation evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plain(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def raised_contact_proposals(points, normals, aperture=.098, limit=24, table_height=None):
    """Experimental proposal-frame clearance; leaves contact points unchanged.

    Finger pads extend 31.5 mm below the grasp frame. Opposed contacts may be
    below that frame while still lying on their pads. Limit vertical displacement
    to 25 mm; preserve full collision checking and physical execution downstream.
    """
    import numpy as np
    from pnr.planning import antipodal_grasps, Grasp
    source = antipodal_grasps(points, normals, aperture=aperture, limit=256,
                             table_height=None)
    selected = []
    for grasp in source:
        position = grasp.position.copy()
        if table_height is not None:
            position[2] = max(position[2], table_height + .037)
            if position[2] - np.min(grasp.contacts[:, 2]) > .025:
                continue
        candidate = Grasp(position, grasp.quaternion, grasp.width, grasp.score,
                          grasp.contacts)
        if any(np.linalg.norm(candidate.position - other.position) < .008
               and abs(np.dot(candidate.quaternion, other.quaternion)) > .99
               for other in selected):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def table_fallback_proposals(points, normals, aperture=.098, limit=24, table_height=None):
    from pnr.planning import antipodal_grasps
    original = antipodal_grasps(points, normals, aperture, limit, table_height)
    return original or raised_contact_proposals(points, normals, aperture, limit, table_height)


def probe_case(config):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from pnr.env import PickNavReachEnv
    from pnr.planning import antipodal_grasps
    import pnr.reference as reference_module
    from pnr.reference import ReferenceController
    from pnr.skills import ManipulationSkillEnv
    from pnr.telemetry import atomic_json

    torch.set_num_threads(1)
    if config['proposal_mode'] == 'raised_contact':
        reference_module.antipodal_grasps = raised_contact_proposals
    elif config['proposal_mode'] == 'table_fallback':
        reference_module.antipodal_grasps = table_fallback_proposals
    index = config['object_idx']
    started = time.monotonic()
    result = {'object_idx': index, 'scene_seed': config['scene_seed'],
              'proposal_mode': config['proposal_mode'], 'reference': {},
              'frozen_grasp': {}, 'setup_action_calls': 0,
              'reference_action_calls': 0, 'frozen_grasp_action_calls': 0, 'frozen_reach_action_calls': 0}
    env = None
    output = Path(config['output']) / f'object_{index:02d}.json'
    try:
        env = PickNavReachEnv(seed=config['scene_seed'], object_idx=index)
        obs = env.observe()
        aabb = np.array(env.p.getAABB(env.object_id))
        canonical = env.object_canonical_mesh.bounds
        filtered = antipodal_grasps(obs['object_pc'], obs['object_normals'],
                                    table_height=env.table_height)
        unfiltered = antipodal_grasps(obs['object_pc'], obs['object_normals'])
        raised = raised_contact_proposals(obs['object_pc'], obs['object_normals'],
                                         table_height=env.table_height)
        result.update(object_name=env.object_name, scene=env.scene_manifest(),
                      mass_kg=env.p.getDynamicsInfo(env.object_id, -1)[0],
                      settled_extent_m=(aabb[1]-aabb[0]).tolist(),
                      mesh_extent_m=(canonical[1]-canonical[0]).tolist(),
                      point_height_range_above_table_m=(np.array([
                          obs['object_pc'][:, 2].min(), obs['object_pc'][:, 2].max()])
                          - env.table_height).tolist(),
                      proposals_table_filtered=len(filtered),
                      proposals_without_table_filter=len(unfiltered),
                      proposals_raised_contact=len(raised),
                      initialization_physics_ticks=env.settle_steps)
        controller = ReferenceController(env, seed=42)
        try:
            grasp = controller.prepare_grasp()
            result['prepared_pregrasp'] = True
            result['selected_grasp'] = grasp.as_dict()
        except Exception as error:
            result.update(prepared_pregrasp=False,
                          preparation_error=f'{type(error).__name__}: {error}')
            return result
        finally:
            result['setup_action_calls'] = env.step_count
        pregrasp = env.save_state()
        before = env.step_count
        try:
            passed = controller.execute_grasp()
            result['reference'] = {'success': passed,
                                   'pick': controller.stage_results.get('pick'),
                                   'metrics': env.evaluate(),
                                   'wall_contact_calls': controller.wall_contact_steps,
                                   'robot_table_contact_calls': controller.table_contact_steps}
            if passed and config.get('reference_transport'):
                controller.prepare_carry()
                result['reference_transport'] = {'carry_posture': env.evaluate()}
                if not env.contacts()['bilateral_contact'] or env.drop_events:
                    raise RuntimeError('Reference carry posture lost object retention')
                controller.carry()
                result['reference_transport']['arrival'] = env.evaluate()
                reach_before = env.step_count
                try:
                    reach = ManipulationSkillEnv.bind(env, controller, skill='reach', horizon=100)
                    reach_model = SAC.load(str(ROOT/'runs/milestone1_validation01/artifacts/checkpoints/reach.zip'), env=reach, device='cpu')
                    observation = reach._observation()
                    reach_steps = []
                    reached = False
                    for decision in range(100):
                        action, _ = reach_model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = reach.step(action)
                        reach_steps.append({'decision': decision, 'action': action.tolist(), 'goal_distance': info['goal_distance'],
                                            'bilateral_contact': info['bilateral_contact'], 'action_rejected': info['action_rejected']})
                        reached = bool(info['is_success'] and not terminated and not env.drop_events)
                        if reached or terminated or truncated or env.drop_events:
                            break
                    held = []
                    if reached:
                        for _ in range(60):
                            state, _ = controller.tick(controller._closed_target(), 'hold')
                            held.append(float(state['object_pos'][2]))
                    metrics = env.evaluate()
                    result['frozen_reach'] = {'success': bool(reached and metrics['complete_task_success'] and held and np.ptp(held) < .02),
                                               'metrics': metrics, 'trace': reach_steps,
                                               'height_range_final_hold_m': float(np.ptp(held)) if held else None,
                                               'wall_contact_calls': controller.wall_contact_steps,
                                               'robot_table_contact_calls': controller.table_contact_steps}
                finally:
                    result['frozen_reach_action_calls'] = env.step_count-reach_before
        except Exception as error:
            result['reference'] = {'success': False, 'error': f'{type(error).__name__}: {error}',
                                   'metrics': env.evaluate(), 'stage': controller.stage}
        finally:
            result['reference_action_calls'] = env.step_count-before-result['frozen_reach_action_calls']

        env.restore_state(pregrasp)
        before = env.step_count
        try:
            skill = ManipulationSkillEnv.bind(env, controller, skill='grasp',
                                               horizon=config['horizon'])
            model = SAC.load(config['checkpoint'], env=skill, device='cpu')
            observation = skill._observation()
            steps = []
            reached = False
            for step in range(config['horizon']):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = skill.step(action)
                steps.append({'step': step, 'action': action.tolist(), 'reward': reward,
                              'object_position': env.observe()['object_pos'].tolist(),
                              'goal_distance': info['goal_distance'],
                              'bilateral_contact': info['bilateral_contact'],
                              'action_rejected': info['action_rejected'],
                              'terminated': terminated, 'truncated': truncated})
                reached = bool(info['is_success'] and info['bilateral_contact']
                               and not terminated and env.drop_events == 0)
                if reached or terminated or truncated or env.drop_events:
                    break
            held = []
            if reached:
                for _ in range(50):
                    state, _ = controller.tick(controller._closed_target(), 'hold')
                    held.append(float(state['object_pos'][2]))
            metrics = env.evaluate()
            passed = bool(reached and metrics['lifted'] and metrics['bilateral_contact']
                          and metrics['stable_hold_steps'] >= 50 and metrics['drop_events'] == 0
                          and np.ptp(held) < .02)
            result['frozen_grasp'] = {'success': passed, 'reached_skill_goal': reached,
                                     'decisions': len(steps), 'metrics': metrics,
                                     'height_range_final_hold_m': float(np.ptp(held)) if held else None,
                                     'action_rejections': sum(t['action_rejected'] for t in steps),
                                     'trace': steps}
        except Exception as error:
            result['frozen_grasp'] = {'success': False, 'error': f'{type(error).__name__}: {error}',
                                     'metrics': env.evaluate()}
        finally:
            result['frozen_grasp_action_calls'] = env.step_count-before
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['actual_action_calls'] = sum(result[key] for key in (
            'setup_action_calls', 'reference_action_calls', 'frozen_grasp_action_calls', 'frozen_reach_action_calls'))
        result['actual_action_physics_ticks'] = result['actual_action_calls']*5
        result['wall_seconds'] = time.monotonic()-started
        atomic_json(output, json.loads(json.dumps(result, default=plain)))
        if env is not None:
            env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='phase2_manipulation_audit01')
    parser.add_argument('--objects', type=int, nargs='+', default=list(range(15)))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--scene-seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--proposal-mode', choices=('original', 'raised_contact', 'table_fallback'), default='original')
    parser.add_argument('--reference-transport', action='store_true')
    parser.add_argument('--checkpoint', default=str(ROOT/'runs/milestone1_validation01/artifacts/checkpoints/grasp.zip'))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.horizon <= 200:
        raise ValueError('Bounded diagnostic requires 1–8 workers and 1–200 decisions')
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    from pnr.telemetry import RunLogger, atomic_json
    logger = RunLogger(args.run_id, kind='validation', skill='grasp',
                       name='Phase 2 · supplied object manipulation audit',
                       algorithm='Physical reference + frozen SAC+HER probe',
                       budget=len(args.objects),
                       summary='All attempted objects retained; diagnostic only, no training or hidden mazes.')
    artifacts = logger.directory/'artifacts'
    artifacts.mkdir(exist_ok=True)
    configuration = vars(args) | {'output': str(artifacts),
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'protocol': 'Physical original-start preparation; reference/policy branches share saved pregrasp; count actual setup once plus both execution branches.',
        'policy_inference': 'Frozen deterministic; no teacher actions.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(logger.directory/'config.json', configuration)
    (artifacts/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(probe_case, configuration | {'object_idx': index}): index
                       for index in args.objects}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                compact = {key: value for key, value in result.items() if key not in ('scene', 'frozen_grasp')}
                compact['frozen_grasp'] = {key: value for key, value in result.get('frozen_grasp', {}).items() if key != 'trace'}
                print(json.dumps(compact, default=plain), flush=True)
                logger.metric(sum(r.get('frozen_grasp', {}).get('decisions', 0) for r in results),
                              scope='multi_object_diagnostic', objects_completed=len(results),
                              reference_successes=sum(r.get('reference', {}).get('success', False) for r in results),
                              frozen_grasp_successes=sum(r.get('frozen_grasp', {}).get('success', False) for r in results),
                              actual_action_calls=sum(r['actual_action_calls'] for r in results),
                              actual_physics_ticks=sum(r['actual_action_physics_ticks']+r.get('initialization_physics_ticks', 0) for r in results))
                atomic_json(artifacts/'summary.json', {'configuration': configuration,
                                                      'results': sorted(results, key=lambda r: r['object_idx'])})
                logger.event(f"Object {result['object_idx']} {result.get('object_name', '')}: prepared={result.get('prepared_pregrasp', False)}, reference={result.get('reference', {}).get('success', False)}, frozen policy={result.get('frozen_grasp', {}).get('success', False)}")
        logger.artifact('artifacts/summary.json', 'All supplied-object physical results and costs')
        logger.artifact('artifacts/probe_source.py', 'Executed audit source')
        logger.finish(summary=f'Completed {len(results)} object probes; reference {sum(r.get("reference", {}).get("success", False) for r in results)}, frozen grasp {sum(r.get("frozen_grasp", {}).get("success", False) for r in results)} successes. No training.')
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
