#!/usr/bin/env python3
"""Replay an exposed validation trace and compare teacher actions; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from stable_baselines3 import SAC
    from pnr.general_manipulation import ContactGraspTeacher, make_grasp_case
    from pnr.telemetry import RunLogger, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-dataset', type=Path, required=True)
    parser.add_argument('--object-idx', type=int, required=True)
    parser.add_argument('--scene-seed', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = next(r for r in json.loads(args.evaluation.read_text())['records']
                  if r['object_idx'] == args.object_idx and r['scene_seed'] == args.scene_seed)
    if source['partition'] != 'validation':
        raise ValueError('This diagnostic labels an already exposed validation case explicitly')
    dataset = pickle.loads(args.training_dataset.read_bytes())
    if args.scene_seed in {c['scene_seed'] for c in dataset['metadata']['cases']}:
        raise ValueError('Validation point seed overlaps training')
    logger = RunLogger(args.run_id, kind='validation', skill='grasp', budget=240,
        name='Grasp validation action diagnosis', algorithm='Frozen actor trace versus physical teacher',
        summary='Diagnostic only on an already exposed case; no learning and no replay data export.')
    out = logger.directory/'artifacts'; out.mkdir()
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(no_learning=True, validation_diagnosis_only=True,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        evaluation_sha256=hashlib.sha256(args.evaluation.read_bytes()).hexdigest())
    atomic_json(logger.directory/'config.json', configuration)
    env = None
    try:
        env = make_grasp_case(args.object_idx, args.scene_seed, seed=71000, horizon=120,
            evaluation=True, goal_lift_m=source['goal_lift_m'],
            grasp_proposal_mode=source['grasp_proposal_mode'])
        native = env.native
        setup = {'source_action_calls': env.setup_action_calls,
                 'physics_ticks': env.setup_action_calls*native.substeps,
                 'initialization_physics_ticks': native.settle_steps}
        atomic_json(out/'setup_accounting.json', setup)
        model = SAC.load(args.checkpoint, env=env, device='cpu')
        teacher = ContactGraspTeacher(env, approach_control='ray-tilted')

        def geometry():
            ee, quat = env.controller.planner.ee_pose()
            obj = native.observe()['object_pos']
            delta = env.measured_grasp_target()-ee
            return {'ee_position': ee.tolist(), 'ee_quaternion': np.asarray(quat).tolist(),
                'object_position': obj.tolist(), 'target_delta_m': delta.tolist(),
                'target_distance_m': float(np.linalg.norm(delta)),
                'object_com_hand_frame_m': Rotation.from_quat(quat).inv().apply(obj-ee).tolist(),
                'finger_opening_m': float(np.sum(native.agent_qpos[native.finger_slots])),
                'contacts': native.contacts()}

        records = []
        for mode in ['recorded_actor_trace', 'physical_teacher']:
            env.reset(seed=71000, options={'full_pregrasp': True})
            env.goal = env.base_goal.copy(); obs = env._observation()
            rows, held = [], []; reached = False
            for i in range(len(source['trajectory']) if mode == 'recorded_actor_trace' else 120):
                teacher_action, teacher_phase = teacher.action(obs)
                actor_action, _ = model.predict(obs, deterministic=True)
                action = (np.asarray(source['trajectory'][i]['action'], dtype=np.float32)
                          if mode == 'recorded_actor_trace' else teacher_action)
                before = geometry()
                obs, _, terminated, truncated, info = env.step(action)
                after = geometry()
                row = {'decision': i+1, 'executed_action': action.tolist(),
                    'actor_action': actor_action.tolist(), 'teacher_action': teacher_action.tolist(),
                    'teacher_phase': teacher_phase, 'before': before, 'after': after,
                    'action_rejected': info['action_rejected'], 'goal_distance': info['goal_distance']}
                if mode == 'recorded_actor_trace':
                    row['recorded_object_position_error_m'] = float(np.linalg.norm(
                        np.asarray(source['trajectory'][i]['object_position'])-after['object_position']))
                    row['saved_actor_action_max_error'] = float(np.max(np.abs(actor_action-action)))
                rows.append(row)
                reached = bool(info['is_success'] and not terminated and not native.drop_events)
                if reached:
                    q = native.agent_qpos.copy(); q[native.finger_slots] = 0.
                    for _ in range(50):
                        state, _ = native.step(q); held.append(float(state['object_pos'][2]))
                    break
                if terminated or truncated or native.drop_events: break
            metrics = native.evaluate()
            passed = bool(reached and metrics['lifted'] and metrics['bilateral_contact']
                and metrics['stable_hold_steps'] >= 50 and not metrics['drop_events'] and np.ptp(held) < .02)
            records.append({'mode': mode, 'success': passed, 'trace': rows, 'metrics': metrics,
                'source_action_calls': native.step_count, 'physics_ticks': native.physics_steps})
            logger.metric(sum(len(r['trace']) for r in records), scope='diagnostic_only',
                          evaluated_branches=len(records), success_rate=sum(r['success'] for r in records)/len(records))
        rotation = Rotation.from_quat(env.controller.grasp.quaternion)
        comparison = []
        for case in dataset['metadata']['cases']:
            if case['object_idx'] != args.object_idx or 'proposal' not in case: continue
            r = Rotation.from_quat(case['proposal']['quaternion'])
            comparison.append({'scene_seed': case['scene_seed'], 'proposal': case['proposal'],
                'angular_distance_degrees': float((rotation*r.inv()).magnitude()*180/np.pi)})
        result = {'case': source | {'trajectory': 'See source evaluation'},
            'proposal': env.controller.grasp.as_dict(),
            'grasp_generation': env.controller.stage_results['grasp_generation'],
            'training_proposal_comparison': comparison, 'shared_setup_costs_counted_once': setup,
            'branches': records, 'total_source_action_calls': setup['source_action_calls']+sum(r['source_action_calls'] for r in records),
            'total_physics_ticks_including_settling': setup['physics_ticks']+setup['initialization_physics_ticks']+sum(r['physics_ticks'] for r in records)}
        atomic_json(out/'diagnosis.json', result)
        logger.artifact('artifacts/diagnosis.json', 'Action, contact and exact trace reproduction diagnostic')
        logger.finish(summary='Completed two counted diagnostic branches; validation states never enter training.')
        print(json.dumps({'branches': [{k: r[k] for k in ['mode','success','source_action_calls']} for r in records],
            'replay_object_error_max_m': max(r['recorded_object_position_error_m'] for r in records[0]['trace']),
            'replay_actor_error_max': max(r['saved_actor_action_max_error'] for r in records[0]['trace']),
            'training_proposal_angle_degrees': [r['angular_distance_degrees'] for r in comparison]}), flush=True)
    except BaseException as error:
        atomic_json(out/'error.json', {'error': repr(error), 'setup_costs': getattr(error, 'setup_accounting', {'source_action_calls': env.setup_action_calls, 'physics_ticks': env.setup_action_calls*env.native.substeps, 'initialization_physics_ticks': env.native.settle_steps} if env is not None else {})})
        logger.finish('failed', str(error)); raise
    finally:
        if env is not None: env.close()


if __name__ == '__main__': main()
