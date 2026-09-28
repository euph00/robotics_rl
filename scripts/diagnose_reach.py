#!/usr/bin/env python3
"""Read-only reach-policy diagnosis on physical local and carry-end states."""
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


def replay_summary(checkpoint):
    # Only inspect checkpoints produced locally by this project; pickle is not a
    # suitable format for untrusted external artifacts.
    with (checkpoint/'replay.pkl').open('rb') as handle:
        replay = pickle.load(handle)
    count = replay.buffer_size if replay.full else replay.pos
    goals = replay.observations['desired_goal'][:count]
    achieved = replay.observations['achieved_goal'][:count]
    infos = replay.infos[:count]
    contacts = np.array([[info.get('bilateral_contact', False) for info in row] for row in infos])
    rejects = np.array([[info.get('action_rejected', False) for info in row] for row in infos])
    rewards = replay.rewards[:count]
    begins = np.zeros((count, replay.n_envs), dtype=bool)
    begins[0] = True
    begins[1:] = replay.dones[:count-1] > 0
    initial_errors = (goals-achieved)[begins]
    return {'transitions':count*replay.n_envs, 'bilateral_contact_fraction':float(contacts.mean()),
        'rejection_fraction':float(rejects.mean()), 'original_goal_reward_zero_fraction':float((rewards == 0).mean()),
        'initial_goal_error_min_xyz':initial_errors.min(axis=0).tolist(),
        'initial_goal_error_max_xyz':initial_errors.max(axis=0).tolist(),
        'desired_goal_min_xyz':goals.reshape(-1, 3).min(axis=0).tolist(),
        'desired_goal_max_xyz':goals.reshape(-1, 3).max(axis=0).tolist(),
        'achieved_goal_min_xyz':achieved.reshape(-1, 3).min(axis=0).tolist(),
        'achieved_goal_max_xyz':achieved.reshape(-1, 3).max(axis=0).tolist()}


def rollout(model, env, horizon, case, raw, course_stop=False):
    observation = env._observation()
    initial = {'achieved_goal':observation['achieved_goal'].tolist(), 'desired_goal':observation['desired_goal'].tolist(),
        'initial_error_xyz':(observation['desired_goal']-observation['achieved_goal']).tolist(),
        'initial_contact':env.native.contacts()['bilateral_contact']}
    data = []
    for step in range(horizon):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        row = {'case':case, 'step':step, 'action':action.tolist(), 'reward':reward,
            'achieved_goal':observation['achieved_goal'].tolist(), 'desired_goal':observation['desired_goal'].tolist(),
            **{key:info[key] for key in ('goal_distance', 'bilateral_contact', 'action_rejected', 'is_success',
                'complete_task_success', 'dist_to_goal', 'action_calls_delta', 'physics_ticks_delta')}}
        raw.write(json.dumps(row)+'\n'); data.append(row)
        if terminated or truncated or (course_stop and info['complete_task_success']):
            break
    raw.flush()
    return {'case':case, **initial, 'decisions':len(data),
        'ever_within_4cm_held':any(row['is_success'] for row in data),
        'ever_within_10cm_held':any(row['goal_distance'] < .1 and row['bilateral_contact'] for row in data),
        'complete_task_success':any(row['complete_task_success'] for row in data),
        'minimum_goal_distance_m':min(row['goal_distance'] for row in data),
        'final_goal_distance_m':data[-1]['goal_distance'],
        'bilateral_contact_fraction':float(np.mean([row['bilateral_contact'] for row in data])),
        'action_rejection_fraction':float(np.mean([row['action_rejected'] for row in data])),
        'mean_action':np.mean([row['action'] for row in data], axis=0).tolist(),
        'source_action_calls':sum(row['action_calls_delta'] for row in data),
        'physics_ticks':sum(row['physics_ticks_delta'] for row in data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='reach_policy_diagnosis')
    parser.add_argument('--training-run', type=Path, default=ROOT/'runs/sac_reach_pilot01')
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.horizon <= 100:
        raise ValueError('Bounded diagnosis allows at most 100 decisions per case')
    torch.set_num_threads(1)
    logger = RunLogger(args.run_id, resume=args.resume, kind='validation', skill='reach', algorithm='SAC + HER evaluation',
        name='Reach policy diagnosis', summary='Comparing saved policies and physically generated carry-end transfer; no learning.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    checkpoints = sorted((args.training_run/'checkpoints').glob('step_*/model.zip'))
    models = []
    env = None
    cases = []
    started = time.perf_counter()
    try:
        env = ManipulationSkillEnv(skill='reach', scene_seed=42, object_idx=14, seed=10000,
            horizon=args.horizon, evaluation=True)
        setup_calls = env.setup_action_calls
        models = [(path.parent.name, SAC.load(path, env=env, device='cpu')) for path in checkpoints]
        with (artifacts/'transitions.jsonl').open('w') as raw:
            for checkpoint, model in models:
                for seed in (10000, 10001, 10002):
                    env.reset(seed=seed)
                    case = {'checkpoint':checkpoint, 'distribution':'local_training_goal_bank', 'seed':seed}
                    result = rollout(model, env, args.horizon, case, raw)
                    cases.append(result)
                    logger.metric(sum(c['decisions'] for c in cases), scope='skill', metric_source='diagnostic',
                        success_rate=float(np.mean([c['ever_within_4cm_held'] for c in cases])),
                        goal_distance=result['final_goal_distance_m'])
                    print(json.dumps(result), flush=True)
            # This is a real contact-held object carried through the complete
            # original maze, not a teleported or attached curriculum object.
            env.reset(seed=10000)
            begin = env.native.step_count
            env.controller.carry()
            carry_calls = env.native.step_count-begin
            carry_snapshot = env.native.save_state()
            transfer_state = {'base_position':env.controller.base_position().tolist(),
                'object_position':env.native.observe()['object_pos'].tolist(),
                'goal_position':env.native.goal_pos.tolist(), 'metrics':env.native.evaluate(),
                'reference_carry_action_calls':carry_calls}
            for checkpoint, model in models:
                env.native.restore_state(carry_snapshot)
                # Reset discarded initialization history was validly observed
                # during reference setup. Report held-at-start transfer rather
                # than claiming full native history when evaluating this probe.
                bound = ManipulationSkillEnv.bind(env.native, env.controller, skill='reach', horizon=args.horizon)
                case = {'checkpoint':checkpoint, 'distribution':'physical_reference_carry_endpoint'}
                result = rollout(model, bound, args.horizon, case, raw, course_stop=True)
                cases.append(result)
                print(json.dumps(result), flush=True)
            env.native.remove_state(carry_snapshot)
        report = {'training':False, 'training_run':str(args.training_run), 'checkpoints':[name for name, _ in models],
            'cases':cases, 'setup_action_calls':setup_calls, 'transfer_state':transfer_state,
            'replay':replay_summary(checkpoints[-1].parent), 'wall_seconds':time.perf_counter()-started,
            'interpretation':'Policy evaluation only; carry-end states were produced by reference control, not the learned full system.'}
        atomic_json(artifacts/'diagnosis.json', report)
        logger.artifact('artifacts/diagnosis.json', 'Reach evaluation and replay diagnosis')
        logger.artifact('artifacts/transitions.jsonl', 'Reach diagnostic transitions')
        logger.metric(sum(c['decisions'] for c in cases), scope='skill', metric_source='diagnostic', cases=len(cases))
        logger.finish(summary='Saved reach policies and reference-carry transfer evaluated; no training.')
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
