#!/usr/bin/env python3
"""Original-start apple retention branches; no production control changes.

This development diagnostic stops after200 native navigation calls. Its result
is retained transport, never full-task success. Every native call and contact is
recorded, with unchanged meshes, force, gains, friction and five physics ticks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation

from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.general_manipulation import GeneralGraspEnv
from pnr.policies import load_policy_checkpoint
from pnr.system import LearnedSystem, StageFailure
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import freeze_checkpoints, plain
from scripts.phase2_evaluate_system import freeze_sources, plan_digest, sha256_file

BRANCHES = ('original', 'first_contact_pause', 'post_lift_pause', 'half_transition')
_POLICIES = None


def initialize_worker(records):
    global _POLICIES
    import torch
    torch.set_num_threads(1)
    _POLICIES = {name: load_policy_checkpoint(record['frozen_path'], device='cpu')
                 for name, record in records.items()}


class FirstContactPause:
    """Probe-only wrapper; pause once at the first bilateral policy decision."""
    def __init__(self, wrapped, system, events):
        self.wrapped, self.system, self.events = wrapped, system, events
        self.action_space = wrapped.action_space
        self.invoked = False

    def _observation(self):
        return self.wrapped._observation()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.wrapped.step(action)
        if self.invoked or terminated or truncated or not info['bilateral_contact']:
            return observation, reward, terminated, truncated, info
        self.invoked = True
        system, native = self.system, self.system.native
        before = native.step_count
        target = native.agent_qpos.copy()
        target[native.finger_slots] = 0.
        event = {'kind': 'first_contact_pause', 'start_call': before,
                 'trigger': 'first policy decision reporting bilateral contact',
                 'lifted_at_trigger': info['lifted'], 'object_height_at_trigger': info['object_height'],
                 'target_qpos': target.tolist(), 'native_calls': 30,
                 'before_target_error_m': float(np.linalg.norm(self.wrapped.measured_grasp_target()
                                                      -system.controller.planner.ee_pose()[0]))}
        previous = system.stage, system.control_source
        system.stage, system.control_source = 'first_contact_pause', 'hold'
        try:
            for _ in range(30):
                system.controller.tick(target, 'first_contact_pause')
        finally:
            system.stage, system.control_source = previous
        observation = self.wrapped._observation()
        info.update(native.evaluate())
        distance = float(np.linalg.norm(observation['achieved_goal']-observation['desired_goal']))
        success = bool(distance < self.wrapped.tolerance and info['bilateral_contact'])
        self.wrapped._episode_success |= success
        info.update(is_success=success, episode_success=self.wrapped._episode_success,
                    goal_distance=distance, diagnostic_contact_pause_calls=30)
        terminated |= bool(native.observe()['object_pos'][2] < native.table_height-.08)
        reward = float(self.wrapped.compute_reward(observation['achieved_goal'], observation['desired_goal'], info))
        event.update(end_call=native.step_count, retained=bool(info['bilateral_contact'] and not native.drop_events))
        self.events.append(event)
        return observation, reward, terminated, truncated, info


def contact_record(native, controller, observation, info, system):
    ee, orientation = controller.planner.ee_pose()
    rotation = Rotation.from_quat(orientation)
    obj = np.asarray(observation['object_pos'])
    points = native.p.getContactPoints(native.robot_id, native.object_id)
    contacts = []
    for point in points:
        if point[9] <= 0:
            continue
        contacts.append({'robot_link': int(point[3]),
            'finger': (list(native.finger_links).index(point[3]) if point[3] in native.finger_links else None),
            'robot_point_world': list(point[5]), 'object_point_world': list(point[6]),
            'object_point_hand': rotation.inv().apply(np.asarray(point[6])-ee).tolist(),
            'normal_world': list(point[7]), 'normal_hand': rotation.inv().apply(point[7]).tolist(),
            'normal_force': float(point[9]), 'lateral_friction_forces': [float(point[10]),float(point[12])]})
    fingers = []
    for finger in range(2):
        selected = [point for point in contacts if point['finger'] == finger]
        forces = np.array([point['normal_force'] for point in selected])
        positions = np.array([point['object_point_hand'] for point in selected])
        centroid = np.average(positions, axis=0, weights=forces) if len(selected) else None
        fingers.append({'normal_force': float(forces.sum()), 'contact_count': len(selected),
            'force_weighted_centroid_hand': None if centroid is None else centroid.tolist(),
            'pad_edge_clearance_xz_m': None if centroid is None else float(min(.0315-abs(centroid[0]),.012-abs(centroid[2])))})
    linear, angular = native.p.getBaseVelocity(native.object_id)
    return {'source_action': native.step_count, 'physics_tick': native.physics_steps,
        'stage': system.display_stage, 'controller': system.control_source,
        'qpos': observation['qpos'].tolist(), 'qvel': observation['qvel'].tolist(),
        'target_qpos': native.last_target.tolist(), 'ee_world': ee.tolist(), 'ee_quaternion': orientation.tolist(),
        'object_world': obj.tolist(), 'object_quaternion': observation['object_xyzw'].tolist(),
        'object_in_hand': rotation.inv().apply(obj-ee).tolist(),
        'object_quaternion_in_hand': (rotation.inv()*Rotation.from_quat(observation['object_xyzw'])).as_quat().tolist(),
        'object_linear_velocity': list(linear), 'object_angular_velocity': list(angular),
        'finger_positions': observation['qpos'][native.finger_slots].tolist(),
        'finger_targets': native.last_target[native.finger_slots].tolist(),
        'bilateral_contact': info['bilateral_contact'], 'lifted': info['lifted'],
        'drop_events': info['drop_events'], 'contacts': contacts, 'fingers': fingers}


def worker(task):
    case, branch, out = task
    out = Path(out)
    name = case['case_id']+'_'+branch
    native = system = None
    rows, interventions = [], []
    started = time.monotonic()
    result = {'diagnostic_id': name, 'case': case, 'branch': branch,
              'retained_transport': False, 'full_task_evaluation': False,
              'scope': 'original_start_to_first_200_native_navigation_calls',
              'original_start': True, 'inter_skill_resets': False,
              'teacher_at_inference': False, 'source_action_calls': 0,
              'physics_ticks': 0, 'initialization_physics_ticks': 0}
    with (out/(name+'.log')).open('w') as log, (out/(name+'_contacts.jsonl')).open('w') as trace:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            native = PickNavReachEnv(**case['env_kwargs'])
            if native.scene_manifest()['maze_hash'] != case['maze_hash']:
                raise ValueError('Frozen layout hash mismatch')
            def observe(current, observation, info):
                row = contact_record(native, current.controller, observation, info, current)
                rows.append(row)
                trace.write(json.dumps(row, allow_nan=False)+'\n')
            system = LearnedSystem(native, _POLICIES, grasp_horizon=600, on_step=observe,
                grasp_proposal_mode='legacy', controller_speed_scale=1.,
                skill_factories={'grasp': partial(GeneralGraspEnv.bind, goal_lift_m=.24)})
            if branch == 'first_contact_pause':
                def bind_grasp(n, controller, horizon):
                    return FirstContactPause(GeneralGraspEnv.bind(n, controller, horizon=horizon,
                                                                 goal_lift_m=.24), system, interventions)
                system.skill_factories['grasp'] = bind_grasp
            with system._observe_native_steps():
                start = system._begin('planned_approach', 'planned')
                grasp = system.controller.prepare_grasp()
                system._finish('planned_approach', start, True, grasp=grasp.as_dict())
                result['selected_grasp'] = grasp.as_dict()
                system._skill('grasp')
                if branch == 'post_lift_pause':
                    start = system._begin('post_lift_pause', 'hold')
                    target = native.agent_qpos.copy();target[native.finger_slots] = 0.
                    for _ in range(30):
                        system.controller.tick(target, 'post_lift_pause')
                    retained = bool(native.contacts()['bilateral_contact'] and native.drop_events == 0)
                    system._finish('post_lift_pause', start, retained)
                    interventions.append({'kind':'post_lift_pause','start_call':start[0],
                        'end_call':native.step_count,'native_calls':30,'target_qpos':target.tolist(),'retained':retained})
                    if not retained:
                        raise StageFailure('post_lift_pause: lost grasp')
                system._held_pause('pick_hold', 50)
                result['after_pick_hold'] = rows[-1]
                start = system._begin('planned_carry_posture', 'planned')
                system.controller.speed_scale = .5 if branch == 'half_transition' else 1.
                result['transition_speed_scale'] = system.controller.speed_scale
                try:
                    system.controller.prepare_carry()
                finally:
                    system.controller.speed_scale = 1.
                retained = bool(native.contacts()['bilateral_contact'] and native.drop_events == 0)
                system._finish('planned_carry_posture', start, retained)
                result['after_carry_posture'] = rows[-1]
                if not retained:
                    raise StageFailure('planned_carry_posture: lost grasp')
                start = system._begin('carry_diagnostic', 'learned')
                carry = GeneralCarryEnv.bind(native, system.controller, horizon=40, delta_scale=.075)
                observation = carry._observation()
                terminated = False
                for decision in range(40):
                    action, _ = _POLICIES['carry'].predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = carry.step(action)
                    system.policy_decisions['carry'] += 1
                    if terminated or native.drop_events:
                        break
                retained = bool(not terminated and native.drop_events == 0 and info['bilateral_contact']
                                and native.step_count-start[0] == 200)
                system._finish('carry_diagnostic', start, retained,
                               policy_decisions=system.policy_decisions['carry'],
                               reason=info.get('failure_reason') or 'completed_200_native_calls')
                if not retained:
                    raise StageFailure('carry_diagnostic: '+str(info.get('failure_reason') or 'lost grasp'))
                result['retained_transport'] = True
        except Exception as exc:
            result.update(error=f'{type(exc).__name__}: {exc}',
                          failed_stage=system.stage if system is not None else 'environment_setup')
            if system is not None and system._active_stage_start is not None:
                system._finish(system.stage, system._active_stage_start, False, error=str(exc))
        finally:
            result['interventions'] = interventions
            if system is not None:
                result.update(stages=system.stage_results, policy_decisions=system.policy_decisions,
                    source_calls_by_controller=system.calls_by_source,
                    wall_contact_calls=system.wall_contact_calls,
                    robot_table_contact_calls=system.robot_table_contact_calls)
            if native is not None:
                result.update(scene_manifest=native.scene_manifest(), metrics=native.evaluate(),
                    source_action_calls=native.step_count, physics_ticks=native.physics_steps,
                    initialization_physics_ticks=native.settle_steps,
                    total_physics_ticks_including_initialization=native.physics_steps+native.settle_steps,
                    cost_accounting_complete=True)
                if rows:
                    result['final_trace'] = rows[-1]
                if 'after_pick_hold' in result:
                    anchor = np.array(result['after_pick_hold']['object_in_hand'])
                    result['motion_by_stage'] = {stage: {
                        'source_action_calls':len(selected),
                        'peak_shift_from_pick_hold_m':float(max(np.linalg.norm(np.array(row['object_in_hand'])-anchor) for row in selected)),
                        'first_nonbilateral_call':next((row['source_action'] for row in selected if not row['bilateral_contact']),None),
                        'final_object_in_hand':selected[-1]['object_in_hand'],
                        'final_fingers':selected[-1]['fingers']}
                        for stage in dict.fromkeys(row['stage'] for row in rows)
                        if (selected := [row for row in rows if row['stage']==stage])}
                assert len(rows) == native.step_count
                assert sum(system.calls_by_source.values()) == native.step_count
                native.close()
            else:
                result.update(cost_accounting_complete=False,
                              cost_accounting_note='No live environment; any partial setup cost is unknown')
            result['wall_seconds'] = time.monotonic()-started
            result = json.loads(json.dumps(result, default=plain, allow_nan=False))
            atomic_json(out/(name+'.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, default=ROOT/'runs/phase2_system_recovery_bc08_01/artifacts/evaluation_plan.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use1–8 CPU workers')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory; every attempt is retained')
    source = json.loads(args.source_plan.read_text())
    desired = [f'validation_{i:03d}_object_07' for i in (6,0,8)]
    cases = [next(case for case in source['cases'] if case['case_id']==key) for key in desired]
    out = args.output/'artifacts';out.mkdir(parents=True)
    records = freeze_checkpoints({k:Path(v['frozen_path']) for k,v in source['checkpoints'].items()},out/'checkpoints')
    assert all(records[k]['sha256']==v['sha256'] for k,v in source['checkpoints'].items())
    hashes = freeze_sources(out/'source')
    shutil.copy2(__file__,out/'source'/'scripts'/Path(__file__).name)
    hashes['scripts/'+Path(__file__).name] = sha256_file(Path(__file__))
    plan = {'protocol':'apple-retention-original-start-branches-v1','source_plan_sha256':source['plan_sha256'],
            'cases':cases,'branches':BRANCHES,'planned_attempts':12,'workers':args.workers,
            'physics_unchanged':True,'inter_skill_resets':False,'native_calls_per_pause':30,
            'carry_native_call_budget':200,'grasp_horizon':600,'grasp_lift_m':.24,
            'delta_scale':.075,'grasp_proposal_mode':'legacy','checkpoints':records,
            'source_hashes':hashes,'full_task_evaluation':False}
    plan['plan_sha256'] = plan_digest(plan);atomic_json(out/'diagnostic_plan.json',plan)
    logger = RunLogger(args.output.name,root=args.output.parent,kind='diagnostic',
        name='Apple retention:12 controlled branches',summary='Original start, unchanged physics; retention through200 navigation calls.')
    logger.artifact('artifacts/diagnostic_plan.json','Frozen diagnostic branches')
    results=[]
    tasks=[(case,branch,str(out)) for case in cases for branch in BRANCHES]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker,initargs=(records,)) as pool:
        for future in as_completed([pool.submit(worker,task) for task in tasks]):
            result=future.result();results.append(result)
            summary={'planned_attempts':12,'completed_attempts':len(results),
                'retained_through_200_calls':sum(r['retained_transport'] for r in results),
                'full_task_evaluation':False,'all_failures_retained':True,
                'physics_ticks':sum(r['physics_ticks'] for r in results),
                'initialization_physics_ticks':sum(r['initialization_physics_ticks'] for r in results),
                'total_physics_ticks':sum(r.get('total_physics_ticks_including_initialization',0) for r in results),
                'results':results}
            atomic_json(out/'summary.json',summary)
            logger.metric(transitions=sum(r['source_action_calls'] for r in results),
                completed_attempts=len(results),planned_attempts=12,
                retained_transport=sum(r['retained_transport'] for r in results),physics_ticks=summary['total_physics_ticks'])
    logger.artifact('artifacts/summary.json','Every branch, contact measurements and all costs')
    logger.finish(summary=f"{summary['retained_through_200_calls']}/12 retained through200 navigation calls; not full-task evaluation.")
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__ == '__main__':
    main()
