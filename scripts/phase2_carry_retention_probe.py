#!/usr/bin/env python3
"""Counted TRAIN/legacy-only physical transport diagnostics; no learned claims.

Original meshes, forces, friction and five-tick controls remain unchanged.
Optional action slew and turn slowdown are explicitly labeled teacher probes,
not silently installed in a learner or evaluation controller.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation

from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv, route_action
from pnr.general_manipulation import GeneralGraspEnv, ContactGraspTeacher
from pnr.planning import maze_astar
from pnr.reference import ReferenceController
from pnr.scenes import load_scene_bank, scene_records
from pnr.telemetry import RunLogger, atomic_json

MODES = {'capture': 0., 'general020': .020, 'general030': .030, 'general040': .040, 'stationary': 0., 'reference': .045, 'general045': .045, 'general060': .060, 'general075': .075,
         'slew075': .075, 'turnslow075': .075, 'slip_reactive075': .075}


def worker(task):
    case, mode, grasp_source, out, horizon, grasp_checkpoint, carry_posture, closure_margin_mm, observe_subticks, capture_grip, close_distance_mm = task
    out = Path(out)
    name = f'{case.get("diagnostic_id", case["layout_id"])}_object_{case["object_idx"]:02d}_{grasp_source}_{mode}_margin{int(round(closure_margin_mm*1000)):04d}um_gate{int(round(close_distance_mm*1000)):05d}um'
    env = None
    controller = None
    rows = []
    subtick_rows = []
    carry_start = None
    started = time.monotonic()
    result = {'case_id': name, 'kind': 'physical_retention_diagnostic', 'case': case, 'mode': mode,
              'grasp_source': grasp_source, 'delta_scale': MODES[mode], 'success': False,
              'teacher_at_inference': True, 'learned_policy_evaluation': grasp_source == 'learned',
              'complete_learned_system': False, 'navigation_controller': 'physical_teacher',
              'grasp_checkpoint': grasp_checkpoint, 'carry_posture': carry_posture,
              'closure_margin_mm': closure_margin_mm, 'teacher_close_distance_mm': close_distance_mm,
              'closure_protocol': 'originalzero' if closure_margin_mm == 0 else 'measuredopening-minus-margin-frozen-at-carry-start',
              'source_action_calls': 0, 'physics_ticks': 0, 'initialization_physics_ticks': 0}
    with (out / f'{name}.log').open('w') as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            env = PickNavReachEnv(seed=case.get('scene_seed',42), object_idx=case['object_idx'], maze_seed=case['maze_seed'])
            controller = ReferenceController(env, seed=case.get('scene_seed',42))
            stage = 'setup'
            previous_velocity = np.zeros(2)
            current_action = None
            current_limiter_scale = 1.
            payload_anchor = None
            carry_finger_targets = None
            decision = -1
            original_step = env.step
            if observe_subticks:
                original_physics_step = env.p.stepSimulation
                def observed_physics_step():
                    observe_tick = stage == 'carry' and len(subtick_rows) < 500
                    if observe_tick:
                        before = np.array([j[1] for j in env.p.getJointStates(env.robot_id, env.joint_indices[:2])])
                    returned = original_physics_step()
                    if observe_tick:
                        velocity = np.array([j[1] for j in env.p.getJointStates(env.robot_id, env.joint_indices[:2])])
                        cp = env.p.getContactPoints(env.robot_id, env.object_id)
                        forces = [sum(c[9] for c in cp if c[3] == link) for link in env.finger_links]
                        subtick_rows.append({'source_action': env.step_count+1,
                            'physics_tick': env.physics_steps + len(subtick_rows)%5 + 1,
                            'within_action_tick': len(subtick_rows)%5+1,
                            'base_velocity': velocity.tolist(),
                            'base_acceleration': ((velocity-before)/env.physics_timestep).tolist(),
                            'finger_normal_forces': forces,
                            'object_linear_velocity': list(env.p.getBaseVelocity(env.object_id)[0])})
                    return returned
                env.p.stepSimulation = observed_physics_step
            def observed(target):
                nonlocal previous_velocity
                if carry_finger_targets is not None and stage in ('carry', 'transport_hold'):
                    target = np.asarray(target).copy()
                    target[env.finger_slots] = carry_finger_targets
                obs, info = original_step(target)
                ee, ee_quat = controller.planner.ee_pose()
                ee_rotation = Rotation.from_quat(ee_quat)
                relative = ee_rotation.inv().apply(obs['object_pos'] - ee)
                relative_quat = (ee_rotation.inv() * Rotation.from_quat(obs['object_xyzw'])).as_quat()
                velocity = obs['qvel'][:2].copy()
                acceleration = (velocity - previous_velocity) / (env.substeps * env.physics_timestep)
                previous_velocity = velocity
                contact_points = env.p.getContactPoints(env.robot_id, env.object_id)
                forces = [sum(c[9] for c in contact_points if c[3] == link) for link in env.finger_links]
                walls = env.p.getContactPoints(env.robot_id, env.maze_id) + env.p.getContactPoints(env.object_id, env.maze_id)
                table = env.p.getContactPoints(env.robot_id, env.table_id)
                linear, angular = env.p.getBaseVelocity(env.object_id)
                rows.append({'call': env.step_count, 'physics_tick': env.physics_steps,
                    'stage': controller.stage if stage == 'setup' else stage, 'decision': decision,
                    'action': current_action, 'carry_limiter_scale': current_limiter_scale,
                    'payload_shift_from_carry_start_m': None if payload_anchor is None else float(np.linalg.norm(relative-payload_anchor)),
                    'base_xy': controller.base_position().tolist(),
                    'base_velocity': velocity.tolist(), 'base_acceleration': acceleration.tolist(),
                    'object_xyz': obs['object_pos'].tolist(), 'ee_xyz': ee.tolist(),
                    'object_in_hand_xyz': relative.tolist(), 'object_in_hand_quat': relative_quat.tolist(),
                    'object_linear_velocity': list(linear), 'object_angular_velocity': list(angular),
                    'finger_positions': obs['qpos'][env.finger_slots].tolist(),
                    'commanded_finger_targets': np.asarray(target)[env.finger_slots].tolist(),
                    'bilateral_contact': info['bilateral_contact'], 'finger_contact': info['finger_contacts'],
                    'finger_normal_forces': forces, 'total_object_normal_force': info['normal_force'],
                    'drop_events': info['drop_events'], 'object_height': info['object_height'],
                    'wall_contact': any(c[9] > 0 for c in walls),
                    'wall_contact_details': [{'body': 'robot' if c[1] == env.robot_id else 'object',
                        'link': int(c[3]), 'force': float(c[9]), 'body_point': list(c[5]),
                        'wall_point': list(c[6])} for c in walls if c[9] > 0],
                    'robot_table_contact': any(c[9] > 0 for c in table)})
                return obs, info
            env.step = observed
            grasp = controller.prepare_grasp()
            result['selected_grasp'] = grasp.as_dict()
            result['grasp_generation'] = controller.stage_results.get('grasp_generation', {})
            if grasp_source == 'reference':
                if not controller.execute_grasp(grasp):
                    raise RuntimeError('Reference grasp/lift failed')
            else:
                grasp_env = GeneralGraspEnv.bind(env, controller, horizon=200)
                teacher = ContactGraspTeacher(grasp_env, close_distance=close_distance_mm/1000.)
                if grasp_source == 'learned':
                    import torch
                    from pnr.policies import load_policy_checkpoint
                    torch.set_num_threads(1)
                    predictor = load_policy_checkpoint(grasp_checkpoint['frozen_path'], device='cpu')
                obs = grasp_env._observation()
                for _ in range(200):
                    action, _ = (predictor.predict(obs, deterministic=True) if grasp_source == 'learned'
                                 else teacher.action(obs))
                    if action[-1] < 0 and 'first_close_command' not in result:
                        measured_ee = controller.planner.ee_pose()[0]
                        result['first_close_command'] = {'before_source_call': env.step_count,
                            'ee_xyz': measured_ee.tolist(), 'object_xyz': env.observe()['object_pos'].tolist(),
                            'proposal_target_error_m': float(np.linalg.norm(grasp_env.measured_grasp_target()-measured_ee))}
                    obs, _, terminated, truncated, info = grasp_env.step(action)
                    if info['is_success'] and info['bilateral_contact']:
                        break
                    if terminated or truncated:
                        raise RuntimeError('Contact-teacher grasp failed: ' + str(info.get('failure_reason')))
                else:
                    raise RuntimeError('Contact-teacher grasp exhausted horizon')
                controller.hold(50, 'hold')
            controller.prepare_carry()
            if carry_posture != 'original':
                ee, orientation = controller.planner.ee_pose()
                if carry_posture == 'centered':
                    ee = np.r_[controller.base_position() + [.12, 0.], 1.10]
                elif carry_posture == 'pinch_x':
                    orientation = Rotation.from_matrix(np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])).as_quat()
                controller.move_ee(ee, orientation, 'carry_posture', closed=True, step_size=.006)
                controller.hold(50, 'hold')
                controller.planner.set_payload()
            if not env.contacts()['bilateral_contact']:
                raise RuntimeError('Carry posture lost contact')
            result['carry_start'] = {'source_action_calls': env.step_count, 'metrics': env.evaluate(),
                                      'trace': rows[-1]}
            footprint=[]
            base_xy=controller.base_position()
            for body,label in [(env.robot_id,'robot'),(env.object_id,'object')]:
                for link in range(-1,env.p.getNumJoints(body)):
                    if not env.p.getCollisionShapeData(body,link):
                        continue
                    lower,upper=env.p.getAABB(body,link)
                    footprint.append({'body':label,'link':link,
                        'link_name':env.p.getJointInfo(body,link)[12].decode() if link>=0 else label,
                        'lower_world':list(lower),'upper_world':list(upper),
                        'lower_base_xy':(np.asarray(lower[:2])-base_xy).tolist(),
                        'upper_base_xy':(np.asarray(upper[:2])-base_xy).tolist()})
            result['carry_start']['collision_envelope']=footprint
            carry_start = env.step_count
            payload_anchor = np.asarray(rows[-1]['object_in_hand_xyz'])
            if mode == 'slip_reactive075':
                result['speed_limiter'] = {
                    'protocol': 'latched payload displacement in measured hand frame',
                    'threshold_m': .002, 'fast_delta_m': .075, 'slow_delta_m': .020,
                    'observed_at': 'before each carry policy decision',
                    'object_identity_used': False,
                    'initial_payload_in_hand_m': payload_anchor.tolist(),
                    'trigger': None}
            measured_opening = env.agent_qpos[env.finger_slots].copy()
            if closure_margin_mm > 0:
                carry_finger_targets = np.clip(measured_opening - closure_margin_mm/1000., 0., .05)
            result['initial_measured_finger_positions'] = measured_opening.tolist()
            result['carry_finger_target_positions'] = [0., 0.] if carry_finger_targets is None else carry_finger_targets.tolist()
            if capture_grip:
                import imageio.v2 as imageio
                ee, ee_quaternion = controller.planner.ee_pose()
                object_position, object_quaternion = env.p.getBasePositionAndOrientation(env.object_id)
                hand_rotation = Rotation.from_quat(ee_quaternion)
                can_axis = Rotation.from_quat(object_quaternion).apply([0.,0.,1.])
                pinch_axis = hand_rotation.apply([0.,1.,0.])
                contact_points = env.p.getContactPoints(env.robot_id, env.object_id)
                contacts = [{'robot_link': int(c[3]), 'object_world_position': list(c[6]),
                    'object_hand_position': hand_rotation.inv().apply(np.asarray(c[6])-ee).tolist(),
                    'normal_world': list(c[7]), 'normal_force': c[9]} for c in contact_points]
                result['grip_geometry'] = {'ee_world': ee.tolist(), 'ee_quaternion': ee_quaternion.tolist(),
                    'object_world': list(object_position), 'object_quaternion': list(object_quaternion),
                    'object_center_in_hand': hand_rotation.inv().apply(np.asarray(object_position)-ee).tolist(),
                    'object_local_z_axis_world': can_axis.tolist(), 'pinch_axis_world': pinch_axis.tolist(),
                    'can_axis_to_vertical_deg': float(np.degrees(np.arccos(np.clip(abs(can_axis[2]),0,1)))),
                    'can_axis_to_pinch_deg': float(np.degrees(np.arccos(np.clip(abs(np.dot(can_axis,pinch_axis)),0,1)))),
                    'pinch_axis_to_world_x_deg': float(np.degrees(np.arccos(np.clip(abs(pinch_axis[0]),0,1)))),
                    'contacts': contacts, 'finger_pad_half_extents_in_hand_m': [.0315,.006,.012]}
                for view, yaw, pitch in [('side',0,-5),('front',90,-5),('top',0,-89)]:
                    imageio.imwrite(out/f'{name}_{view}.png', env.render_rgb(camera='grasp',target=ee,
                        distance=.42,yaw=yaw,pitch=pitch,width=800,height=600,shadow=False))
            result['object_mass_kg'] = env.p.getDynamicsInfo(env.object_id, -1)[0]
            result['object_dynamics'] = list(env.p.getDynamicsInfo(env.object_id, -1))
            stage = 'carry'
            if mode == 'capture':
                success = env.contacts()['bilateral_contact'] and not env.drop_events
                result['capture_only'] = True
            elif mode == 'stationary':
                controller.hold(3500, 'hold')
                success = env.contacts()['bilateral_contact'] and not env.drop_events
            elif mode == 'reference':
                controller.carry()
                success = env.contacts()['bilateral_contact'] and not env.drop_events
            else:
                carry = GeneralCarryEnv.bind(env, controller, horizon=horizon, delta_scale=MODES[mode])
                obs = carry._observation()
                goal = obs['desired_goal'] * carry.goal_scale
                path = maze_astar(env.cube_positions, controller.base_position(), goal,
                                  table_bounds=env.p.getAABB(env.table_id))
                result['route'] = [point.tolist() for point in path]
                previous_action = np.zeros(2)
                success = False
                for decision in range(horizon):
                    action = route_action(path, obs['achieved_goal'] * carry.goal_scale, goal, carry.delta_scale)
                    if mode == 'slip_reactive075':
                        payload_shift = np.asarray(rows[-1]['object_in_hand_xyz']) - payload_anchor
                        if result['speed_limiter']['trigger'] is None and np.linalg.norm(payload_shift) > .002:
                            result['speed_limiter']['trigger'] = {
                                'before_decision': decision, 'before_source_action_call': env.step_count,
                                'before_physics_tick': env.physics_steps,
                                'carry_source_action_calls': env.step_count-carry_start,
                                'payload_shift_vector_m': payload_shift.tolist(),
                                'payload_shift_norm_m': float(np.linalg.norm(payload_shift)),
                                'bilateral_contact': bool(rows[-1]['bilateral_contact']),
                                'finger_normal_forces': rows[-1]['finger_normal_forces'],
                                'base_xy': rows[-1]['base_xy']}
                        if result['speed_limiter']['trigger'] is not None:
                            current_limiter_scale = .020/.075
                            action *= current_limiter_scale
                    elif mode == 'slew075':
                        # Teacher-only proposal: needs observed controller memory if later learned.
                        change = action - previous_action
                        action = previous_action + change * min(1., .18 / max(np.linalg.norm(change), 1e-9))
                    elif mode == 'turnslow075':
                        alignment = float(np.dot(action, previous_action) / max(np.linalg.norm(action)*np.linalg.norm(previous_action), 1e-9))
                        if alignment < .95:
                            action *= .55
                    previous_action = action.copy()
                    current_action = action.tolist()
                    obs, _, terminated, truncated, info = carry.step(action)
                    if terminated:
                        raise RuntimeError('Carry step failed: ' + str(info['failure_reason']))
                    if info['is_success']:
                        success = True
                        break
                    if truncated:
                        raise RuntimeError('Carry teacher exhausted horizon')
                if not success:
                    raise RuntimeError('Carry teacher did not reach goal')
            stage = 'transport_hold'
            result['carry_goal_action'] = env.step_count
            controller.hold(60, 'hold')
            result['success'] = bool(success and env.contacts()['bilateral_contact'] and not env.drop_events
                                     and env.evaluate()['lifted'])
            if not result['success']:
                result['error'] = 'Lost retained lift during final transport hold'
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if controller is not None:
                result['grasp_generation'] = controller.stage_results.get('grasp_generation', {})
            if env is not None:
                result.update(scene=env.scene_manifest(), final_metrics=env.evaluate(),
                              source_action_calls=env.step_count, physics_ticks=env.physics_steps,
                              initialization_physics_ticks=env.settle_steps,
                              carry_source_action_calls=0 if carry_start is None else env.step_count - carry_start)
                env.close()
        result['wall_seconds'] = time.monotonic() - started
        if subtick_rows:
            result['subtick_base_acceleration_peak_m_s2'] = float(np.linalg.norm([row['base_acceleration'] for row in subtick_rows], axis=1).max())
            with (out / f'{name}_subticks.jsonl').open('w') as trace:
                for row in subtick_rows:
                    trace.write(json.dumps(row, separators=(',', ':')) + '\n')
        transport = [row for row in rows if row['stage'] in ('carry', 'transport_hold')]
        result['first_transport_contact_loss'] = next((row for row in transport if not row['bilateral_contact']), None)
        result['first_transport_wall_contact'] = next((row for row in transport if row['wall_contact']), None)
        result['first_transport_drop'] = next((row for row in transport if row['drop_events']), None)
        if transport:
            initial_relative = np.asarray(transport[0]['object_in_hand_xyz'])
            relative = np.asarray([row['object_in_hand_xyz'] for row in transport])
            result['max_payload_translation_in_hand_m'] = float(np.linalg.norm(relative - initial_relative, axis=1).max())
            result['max_base_acceleration_m_s2'] = float(np.linalg.norm([row['base_acceleration'] for row in transport], axis=1).max())
            result['transport_wall_contact_calls'] = sum(row['wall_contact'] for row in transport)
            result['transport_missing_contact_calls'] = sum(not row['bilateral_contact'] for row in transport)
        with (out / f'{name}_trajectory.jsonl').open('w') as trace:
            for row in rows:
                trace.write(json.dumps(row, separators=(',', ':')) + '\n')
        atomic_json(out / f'{name}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', default='configs/phase2_scene_bank.json')
    parser.add_argument('--scene-seed', type=int, default=42, help='Point-cloud and planner seed')
    parser.add_argument('--development-layouts', nargs='+', help='Explicit TRAIN/validation layout IDs only')
    parser.add_argument('--objects', type=int, nargs='+', default=[2, 14])
    parser.add_argument('--train-layouts', type=int, default=2)
    parser.add_argument('--skip-legacy', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=['reference', 'general045', 'general060', 'general075'])
    parser.add_argument('--grasp-source', choices=['reference', 'contact', 'learned'], default='reference')
    parser.add_argument('--grasp-checkpoint', type=Path)
    parser.add_argument('--close-distances-mm', type=float, nargs='+', default=[18.],
                        help='Contact-teacher closure trigger distance; no change to learned grasp checkpoints')
    parser.add_argument('--carry-posture', choices=['original', 'centered', 'pinch_x'], default='original')
    parser.add_argument('--closure-margins-mm', type=float, nargs='+', default=[0.],
                        help='0 retains original zero target; positive margin is subtracted from measured opening')
    parser.add_argument('--controls-legacy-only', action='store_true',
                        help='Restrict non-tomato control objects to the course-legacy scene')
    parser.add_argument('--capture-grip', action='store_true', help='Save carry-start grip geometry and three camera views')
    parser.add_argument('--observe-subticks', action='store_true', help='Read-only240Hz dynamics observation for first500carryticks')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--horizon', type=int, default=1400)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 <= args.train_layouts <= 2:
        parser.error('Use1..8 CPU workers and at most the first2TRAIN layouts')
    if any(not np.isfinite(margin) or not 0 <= margin <= 3 for margin in args.closure_margins_mm):
        parser.error('This bounded closure diagnostic supports margins0..3mm')
    if any(not np.isfinite(distance) or not 5 <= distance <= 20 for distance in args.close_distances_mm):
        parser.error('Closure-distance probe supports5..20mm gates')
    if args.grasp_source != 'contact' and args.close_distances_mm != [18.]:
        parser.error('Nondefault closure-distance gates require contact teacher')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new diagnostic output directory')
    if (args.grasp_source == 'learned') != (args.grasp_checkpoint is not None):
        parser.error('--grasp-source learned requires --grasp-checkpoint, and other sources forbid it')
    bank = load_scene_bank(args.bank)
    layouts = [{'layout_id': 'course_legacy42', 'maze_seed': None}] if not args.skip_legacy else []
    layouts += [{'layout_id': row['layout_id'], 'maze_seed': row['maze_seed']}
                for row in scene_records(bank, 'train')[:args.train_layouts]]
    if args.development_layouts:
        available = {row['layout_id']: row for partition in ('train','validation') for row in scene_records(bank,partition)}
        if len(args.development_layouts)>2 or any(name not in available for name in args.development_layouts):
            parser.error('Choose at most2 declared TRAIN/validation layout IDs; heldout layouts forbidden')
        layouts = [{'layout_id': name, 'maze_seed': available[name]['maze_seed']} for name in args.development_layouts]
    cases = [dict(layout, object_idx=obj, scene_seed=args.scene_seed) for layout in layouts for obj in args.objects
             if not args.controls_legacy_only or obj == 2 or layout['maze_seed'] is None]
    out = args.output / 'artifacts'; out.mkdir(parents=True)
    checkpoint_record = None
    if args.grasp_checkpoint:
        frozen = out / 'grasp_checkpoint.zip'
        shutil.copy2(args.grasp_checkpoint, frozen)
        checkpoint_record = {'source': str(args.grasp_checkpoint), 'frozen_path': str(frozen.resolve()),
                             'sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()}
    shutil.copy2(Path(__file__), out/'probe_source.py')
    logger = RunLogger(args.output.name, root=args.output.parent, kind='reference', skill='carry',
                       name='Carry retention diagnostic', algorithm='Physical teacher diagnostic',
                       summary='Original physics; fixed selected objects; declared development layouts only.')
    atomic_json(args.output / 'config.json', dict(vars(args), output=str(args.output),
                grasp_checkpoint=str(args.grasp_checkpoint) if args.grasp_checkpoint else None,
                bank_sha256=bank['manifest_sha256'], cases=cases, selected_test_objects_unchanged=True))
    tasks = [(case, mode, args.grasp_source, str(out), args.horizon, checkpoint_record, args.carry_posture, margin, args.observe_subticks, args.capture_grip, gate)
             for case in cases for mode in args.modes for margin in args.closure_margins_mm for gate in args.close_distances_mm]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            summary = {'kind': 'physical_retention_diagnostic', 'learned_policy_evaluation': False,
                       'planned_attempts': len(tasks), 'completed_attempts': len(results),
                       'successes': sum(row['success'] for row in results),
                       'source_action_calls': sum(row['source_action_calls'] for row in results),
                       'physics_ticks': sum(row['physics_ticks'] for row in results),
                       'initialization_physics_ticks': sum(row['initialization_physics_ticks'] for row in results),
                       'results': results}
            atomic_json(out / 'summary.json', summary)
            logger.metric(transitions=summary['source_action_calls'], scope='physical_teacher_diagnostic',
                          eval_episodes=len(results), physics_ticks=summary['physics_ticks'],
                          reference_success_rate=summary['successes']/len(results))
            print(json.dumps({key: result.get(key) for key in ('case_id', 'success', 'error', 'physics_ticks',
                  'max_payload_translation_in_hand_m', 'transport_wall_contact_calls')}), flush=True)
    logger.artifact('artifacts/summary.json', 'Physical retention diagnostic summary')
    logger.finish(summary=f'{summary["successes"]}/{len(tasks)} physical teacher transport probes passed; not learned evaluations.')


if __name__ == '__main__':
    main()
