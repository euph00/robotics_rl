#!/usr/bin/env python3
"""Bounded physical grasp exploration diagnosis; no learning or reward changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json


def summarize(steps, initial):
    positions = np.array([step['object_position'] for step in steps])
    goals = np.array([step['achieved_goal'] for step in steps])
    bilateral = np.array([step['bilateral_contact'] for step in steps])
    closed = np.array([step['action'][3] < -.8 for step in steps])
    def longest(values):
        run = maximum = 0
        for value in values:
            run = run+1 if value else 0
            maximum = max(maximum, run)
        return maximum
    return {'decisions':len(steps), 'requested_goal_success':any(step['is_success'] for step in steps),
        'bilateral_contact_fraction':float(bilateral.mean()), 'max_consecutive_bilateral_decisions':longest(bilateral),
        'near_closed_command_fraction':float(closed.mean()), 'max_consecutive_near_closed_commands':longest(closed),
        'action_rejection_fraction':float(np.mean([step['action_rejected'] for step in steps])),
        'max_object_lift_m':float(np.max(positions[:, 2]-initial[2])),
        'max_object_displacement_m':float(np.max(np.linalg.norm(positions-initial, axis=1))),
        'object_position_range_m':np.ptp(positions, axis=0).tolist(),
        'achieved_goal_std_m':goals.std(axis=0).tolist(),
        'achieved_goal_1cm_voxels':len(np.unique(np.floor(goals/.01).astype(int), axis=0)),
        'min_requested_goal_distance_m':min(step['goal_distance'] for step in steps),
        'maximum_lifted_hold_source_actions':max(step['stable_hold_steps'] for step in steps),
        'source_action_calls':sum(step['action_calls_delta'] for step in steps),
        'physics_ticks':sum(step['physics_ticks_delta'] for step in steps),
        'terminated':steps[-1]['terminated'], 'truncated':steps[-1]['truncated']}


def aggregate_controls(episodes):
    groups = {}
    for mode in sorted({episode['mode'] for episode in episodes}):
        values = [episode for episode in episodes if episode['mode'] == mode]
        decisions = sum(episode['decisions'] for episode in values)
        groups[mode] = {'episodes':len(values), 'decisions':decisions,
            'successful_episodes':sum(episode['requested_goal_success'] for episode in values),
            'bilateral_contact_fraction':sum(episode['decisions']*episode['bilateral_contact_fraction'] for episode in values)/decisions,
            'action_rejection_fraction':sum(episode['decisions']*episode['action_rejection_fraction'] for episode in values)/decisions,
            'largest_lift_m':max(episode['max_object_lift_m'] for episode in values),
            'mean_maximum_lift_m':float(np.mean([episode['max_object_lift_m'] for episode in values])),
            'longest_bilateral_contact_decisions':max(episode['max_consecutive_bilateral_decisions'] for episode in values),
            'longest_near_closed_commands':max(episode['max_consecutive_near_closed_commands'] for episode in values)}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='grasp_exploration_probe')
    parser.add_argument('--random-episodes', type=int, default=9)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--scene-seed', type=int, default=42)
    parser.add_argument('--object-idx', type=int, default=14)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--closed-start', action='store_true')
    parser.add_argument('--only-random', action='store_true')
    parser.add_argument('--lift-height', type=float, default=.20)
    args = parser.parse_args()
    if not 1 <= args.random_episodes <= 9 or not 1 <= args.horizon <= 100:
        raise ValueError('This diagnostic is bounded to at most 10 episodes of 100 decisions')
    episode_count = args.random_episodes + int(not args.only_random)
    logger = RunLogger(args.run_id, kind='validation', skill='grasp', algorithm='Diagnostic controls',
        name='Closed-start grasp exploration' if args.closed_start else 'Grasp exploration diagnostic', budget=episode_count*args.horizon,
        summary='Bounded uniform-action physical exploration; no learning.' if args.only_random else 'Uniform actions versus one close-then-lift episode; no learning.')
    atomic_json(logger.directory/'config.json', vars(args) | {'training':False, 'start':'physical easy .008m pregrasp',
        'controls':['independent uniform actions'] if args.only_random else ['scripted close-then-lift', 'independent uniform actions']})
    env = None
    started = time.perf_counter()
    episodes = []
    total = 0
    rng = np.random.default_rng(args.seed)
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    try:
        env = ManipulationSkillEnv(skill='grasp', scene_seed=args.scene_seed, object_idx=args.object_idx,
            seed=args.seed, horizon=args.horizon, evaluation=True)
        closed_state = None
        closed_setup_calls = 0
        if args.closed_start:
            env.reset(seed=args.seed)
            target = env.native.agent_qpos.copy()
            target[env.native.finger_slots] = 0.
            for _ in range(20):
                env.native.step(target)
                closed_setup_calls += 1
            if not env.native.contacts()['bilateral_contact']:
                raise RuntimeError('Closed curriculum start did not establish bilateral physical contact')
            closed_state = env.native.save_state()
        logger.update(setup_action_calls=env.setup_action_calls, setup_physics_ticks=env.setup_action_calls*env.native.substeps,
                      closed_start_generation_calls=closed_setup_calls)
        with (artifacts/'transitions.jsonl').open('w') as raw:
            for episode in range(int(args.only_random), args.random_episodes+1):
                mode = 'close_then_lift' if episode == 0 else 'uniform_random'
                observation, reset_info = env.reset(seed=args.seed+episode)
                if closed_state is not None:
                    env.native.restore_state(closed_state, reset_counters=True)
                env.goal[2] = env.base_goal[2]-.20+args.lift_height
                observation = env._observation()
                initial = env.native.observe()['object_pos'].copy()
                transitions = []
                for step in range(args.horizon):
                    if mode == 'uniform_random':
                        action = rng.uniform(-1, 1, 4).astype(np.float32)
                    else:
                        lift = 0. if step < 20 else float(np.clip((observation['desired_goal'][2]-observation['achieved_goal'][2])/.012, -1, 1))
                        action = np.array([0, 0, lift, -1], dtype=np.float32)
                    observation, reward, terminated, truncated, info = env.step(action)
                    record = {'episode':episode, 'mode':mode, 'step':step, 'action':action.tolist(),
                        'achieved_goal':observation['achieved_goal'].tolist(),
                        'desired_goal':observation['desired_goal'].tolist(),
                        'object_position':env.native.observe()['object_pos'].tolist(),
                        'reward':reward, 'terminated':terminated, 'truncated':truncated,
                        **{key:info[key] for key in ('bilateral_contact', 'finger_contacts', 'is_success',
                           'goal_distance', 'action_rejected', 'stable_hold_steps', 'action_calls_delta', 'physics_ticks_delta')}}
                    transitions.append(record)
                    raw.write(json.dumps(record, separators=(',', ':'))+'\n')
                    total += 1
                    if terminated or truncated:
                        break
                raw.flush()
                result = {'episode':episode, 'mode':mode, 'reset':reset_info,
                    'initial_object_position':initial.tolist(), **summarize(transitions, initial)}
                episodes.append(result)
                logger.metric(total, scope='skill', metric_source='diagnostic', mode=mode, episodes=len(episodes),
                    success_rate=float(np.mean([ep['requested_goal_success'] for ep in episodes if ep['mode']==mode])),
                    bilateral_contact_fraction=result['bilateral_contact_fraction'],
                    max_object_lift_m=result['max_object_lift_m'],
                    action_rejection_fraction=result['action_rejection_fraction'])
                logger.event(f"{mode}: success={result['requested_goal_success']}, contact={result['bilateral_contact_fraction']:.1%}, lift={result['max_object_lift_m']:.3f}m")
                print(json.dumps(result), flush=True)
        report = {'training':False, 'scene':env.native.scene_manifest(), 'setup_action_calls':env.setup_action_calls,
            'setup_physics_ticks':env.setup_action_calls*env.native.substeps, 'episodes':episodes,
            'closed_start':args.closed_start, 'closed_start_generation_calls':closed_setup_calls, 'requested_lift_m':args.lift_height,
            'policy_decisions':total, 'source_action_calls':sum(ep['source_action_calls'] for ep in episodes),
            'physics_ticks':sum(ep['physics_ticks'] for ep in episodes), 'aggregate_controls':aggregate_controls(episodes),
            'wall_seconds':time.perf_counter()-started,
            'interpretation_limit':'One easy-start physical diagnostic, not learned or complete-task success.'}
        atomic_json(artifacts/'diagnosis.json', report)
        logger.artifact('artifacts/diagnosis.json', 'Grasp exploration measurements')
        logger.artifact('artifacts/transitions.jsonl', 'Diagnostic transition records')
        random_eps = [ep for ep in episodes if ep['mode']=='uniform_random']
        summary = f"Uniform random {sum(ep['requested_goal_success'] for ep in random_eps)}/{len(random_eps)}, closed_start={args.closed_start}. No learning."
        if not args.only_random:
            summary = f"Close-then-lift {int(episodes[0]['requested_goal_success'])}/1; "+summary
        logger.finish(summary=summary)
    except BaseException as error:
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
