#!/usr/bin/env python3
"""Bounded deterministic grasp-goal diagnosis; never trains or changes a policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from stable_baselines3 import SAC
import torch

from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json
from scripts.diagnose_grasp import summarize


def inspect_replay(checkpoint):
    with (checkpoint/'replay.pkl').open('rb') as handle:
        replay = pickle.load(handle)
    n = replay.buffer_size if replay.full else replay.pos
    achieved = replay.next_observations['achieved_goal'][:n]
    infos = replay.infos[:n]
    contacts = np.array([[i.get('bilateral_contact', False) for i in row] for row in infos])
    failures = np.array([[i.get('physical_failure', False) for i in row] for row in infos])
    desired_z = replay.observations['desired_goal'][:n, :, 2]
    initial_z = float(np.median(replay.observations['achieved_goal'][0, :, 2]))
    return {'transitions':n*replay.n_envs, 'initial_object_z':initial_z,
        'max_lift_m':float(achieved[:, :, 2].max()-initial_z),
        'achieved_z_std_m':float(achieved[:, :, 2].std()),
        'achieved_xy_range_m':np.ptp(achieved[:, :, :2].reshape(-1, 2), axis=0).tolist(),
        'bilateral_contact_fraction':float(contacts.mean()), 'physical_failure_fraction':float(failures.mean()),
        'near_closed_command_fraction':float(np.mean(replay.actions[:n, :, 3] < -.8)),
        'requested_lift_distribution_m':{str(height):int((np.round(desired_z-initial_z, 2)==height).sum())
            for height in np.unique(np.round(desired_z-initial_z, 2))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, default=ROOT/'runs/sac_grasp_curriculum01')
    parser.add_argument('--generation')
    parser.add_argument('--run-id', default='grasp_curriculum_10k_diagnosis')
    parser.add_argument('--horizon', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 100:
        raise ValueError('Bounded diagnosis requires horizon <=100')
    generation = args.generation or json.loads((args.training_run/'checkpoints/latest.json').read_text())['generation']
    checkpoint = args.training_run/'checkpoints'/generation
    logger = RunLogger(args.run_id, kind='validation', skill='grasp', algorithm='SAC + HER evaluation',
        name='Grasp curriculum policy diagnosis', budget=6*args.horizon,
        summary='Comparing deterministic behavior at 7, 13 and 20 cm goals; no training.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    env = None
    cases = []
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        env = ManipulationSkillEnv(skill='grasp', scene_seed=42, object_idx=14, seed=10000,
            horizon=args.horizon, evaluation=True)
        model = SAC.load(checkpoint/'model.zip', env=env, device='cpu')
        logger.update(checkpoint=generation, checkpoint_transitions=int(model.num_timesteps), setup_action_calls=env.setup_action_calls)
        with (artifacts/'transitions.jsonl').open('w') as raw:
            for height in (.07, .13, .20):
                for seed in (10000, 10001):
                    _, reset_info = env.reset(seed=seed)
                    env.goal[2] = env.base_goal[2]-.20+height
                    observation = env._observation()
                    initial = env.native.observe()['object_pos'].copy()
                    initial_ee = env.controller.planner.ee_pose()[0]
                    steps = []
                    for step in range(args.horizon):
                        action, _ = model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = env.step(action)
                        record = {'requested_lift_m':height, 'seed':seed, 'step':step,
                            'action':action.tolist(), 'achieved_goal':observation['achieved_goal'].tolist(),
                            'desired_goal':observation['desired_goal'].tolist(),
                            'object_position':env.native.observe()['object_pos'].tolist(),
                            'ee_position':env.controller.planner.ee_pose()[0].tolist(),
                            'finger_positions':env.native.agent_qpos[env.native.finger_slots].tolist(),
                            'reward':reward, 'terminated':terminated, 'truncated':truncated,
                            **{key:info[key] for key in ('bilateral_contact', 'finger_contacts', 'is_success',
                                'goal_distance', 'action_rejected', 'stable_hold_steps', 'action_calls_delta', 'physics_ticks_delta')}}
                        steps.append(record); raw.write(json.dumps(record)+'\n')
                        if terminated or truncated:
                            break
                    raw.flush()
                    actions = np.array([step['action'] for step in steps])
                    positions = np.array([step['object_position'] for step in steps])
                    result = {'requested_lift_m':height, 'seed':seed, 'reset':reset_info,
                        'initial_object_position':initial.tolist(), 'initial_ee_position':initial_ee.tolist(),
                        **summarize(steps, initial), 'first_action':actions[0].tolist(),
                        'mean_action':actions.mean(axis=0).tolist(), 'last_action':actions[-1].tolist(),
                        'z_positive_action_fraction':float(np.mean(actions[:, 2] > .1)),
                        'z_negative_action_fraction':float(np.mean(actions[:, 2] < -.1)),
                        'action_std':actions.std(axis=0).tolist(),
                        'final_object_xy_drift_m':(positions[-1, :2]-initial[:2]).tolist(),
                        'maximum_object_xy_drift_m':float(np.max(np.linalg.norm(positions[:, :2]-initial[:2], axis=1))),
                        'maximum_ee_lift_m':max(step['ee_position'][2]-initial_ee[2] for step in steps)}
                    cases.append(result)
                    logger.metric(sum(case['decisions'] for case in cases), scope='skill', metric_source='diagnostic',
                        requested_lift_m=height, success_rate=float(np.mean([case['requested_goal_success'] for case in cases if case['requested_lift_m']==height])),
                        max_object_lift_m=result['max_object_lift_m'], bilateral_contact_fraction=result['bilateral_contact_fraction'])
                    print(json.dumps(result), flush=True)
        report = {'training':False, 'checkpoint':str(checkpoint), 'checkpoint_transitions':int(model.num_timesteps),
            'cases':cases, 'replay':inspect_replay(checkpoint), 'setup_action_calls':env.setup_action_calls,
            'policy_decisions':sum(case['decisions'] for case in cases),
            'source_action_calls':sum(case['source_action_calls'] for case in cases),
            'physics_ticks':sum(case['physics_ticks'] for case in cases), 'wall_seconds':time.perf_counter()-started,
            'interpretation_limit':'Easy physical pregrasp starts; success at 7 or 13 cm does not establish the fixed 20 cm lift or full task.'}
        atomic_json(artifacts/'diagnosis.json', report)
        logger.artifact('artifacts/diagnosis.json', 'Grasp policy goal and replay diagnosis')
        logger.artifact('artifacts/transitions.jsonl', 'Grasp policy transition records')
        logger.finish(summary='Bounded 7/13/20 cm policy diagnosis complete; no training.')
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
