#!/usr/bin/env python3
"""Frozen carry checkpoints versus direct-goal TEST control on matched starts.

No training occurs. A direct-goal controller supplies only an explicitly labeled
physical feasibility diagnostic, never a learned-policy result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer

from pnr.carry import CarrySkillEnv
from pnr.telemetry import RunLogger, atomic_json


CASE_SEEDS = (10000, 10001, 10002)


def freeze(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as reader, destination.open('xb') as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
    return {'source': str(source.resolve()), 'frozen_path': str(destination.resolve()),
            'sha256': digest.hexdigest()}


def straight_line_geometry(env, start, goal, footprint=.36):
    """Conservative circle/wall/table screen; actual TEST motion is decisive."""
    count = max(2, int(np.ceil(np.linalg.norm(goal-start) / .01)) + 1)
    points = start[None, :] + np.linspace(0, 1, count)[:, None] * (goal-start)[None, :]
    walls = np.asarray(env.native.cube_positions)[:, :2]
    wall_gap = np.maximum(np.abs(points[:, None, :] - walls[None, :, :]) - .5, 0)
    wall_margin = float(np.min(np.linalg.norm(wall_gap, axis=-1)) - footprint)
    low, high = (np.asarray(item)[:2] for item in env.native.p.getAABB(env.native.table_id))
    table_gap = np.maximum(np.maximum(low - points, points - high), 0)
    table_margin = float(np.min(np.linalg.norm(table_gap, axis=-1)) - footprint)
    return {'footprint_radius_m': footprint, 'minimum_wall_margin_m': wall_margin,
            'minimum_table_margin_m': table_margin,
            'straight_line_geometry_clear': bool(min(wall_margin, table_margin) >= 0),
            'interpretation': 'Circle footprint screen; physical direct-goal TEST result is reported separately.'}


def evaluate_case(env, policy, mode, case_seed, expected_start, trace_writer, reset_options=None):
    observation, reset_info = env.reset(seed=case_seed, options=reset_options)
    current_start = {key: value.copy() for key, value in observation.items()}
    if expected_start is not None:
        for key in current_start:
            if not np.array_equal(current_start[key], expected_start[key]):
                raise RuntimeError('Matched physical reset differed for %s, case %s' % (mode, case_seed))
    initial_xy = observation['achieved_goal'].astype(float) * env.goal_scale
    goal_xy = observation['desired_goal'].astype(float) * env.goal_scale
    initial_vector = goal_xy - initial_xy
    geometry = straight_line_geometry(env, initial_xy, goal_xy)
    rows = []
    path = [initial_xy.tolist()]
    success = False
    started = time.perf_counter()
    for decision in range(env.horizon):
        error = (observation['desired_goal'] - observation['achieved_goal']).astype(float) * env.goal_scale
        previous_distance = float(np.linalg.norm(error))
        if policy is None:
            action = error / env.delta_scale
            action /= max(1., float(np.linalg.norm(action)))
        else:
            action, _ = policy.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=float)
        applied = np.clip(action, -1, 1)
        applied /= max(1., float(np.linalg.norm(applied)))
        magnitude = float(np.linalg.norm(applied))
        cosine = float(np.dot(applied, error) / (magnitude * previous_distance)) if magnitude > 1e-8 and previous_distance > 1e-8 else 0.
        observation, reward, terminated, truncated, info = env.step(action)
        position = (observation['achieved_goal'].astype(float) * env.goal_scale).tolist()
        row = {'controller': mode, 'case_seed': case_seed, 'decision': decision + 1,
            'base_x_m': position[0], 'base_y_m': position[1],
            'goal_error_x_m': float(error[0]), 'goal_error_y_m': float(error[1]),
            'action_x': float(action[0]), 'action_y': float(action[1]),
            'action_dot_goal_m': float(np.dot(applied, error)), 'action_goal_cosine': cosine,
            'projected_command_m': float(env.delta_scale * np.dot(applied, error) / max(previous_distance, 1e-12)),
            'distance_before_m': previous_distance, 'goal_distance_m': float(info['goal_distance']),
            'distance_progress_m': previous_distance - float(info['goal_distance']),
            'bilateral_contact': bool(info['bilateral_contact']), 'collision': bool(info['collision']),
            'failure_reason': info['failure_reason'], 'is_success': bool(info['is_success']),
            'terminated': bool(terminated), 'truncated': bool(truncated), 'reward': float(reward),
            'source_action_calls': int(info['action_calls_delta']), 'physics_ticks': int(info['physics_ticks_delta'])}
        if trace_writer['writer'] is None:
            trace_writer['writer'] = csv.DictWriter(trace_writer['stream'], fieldnames=list(row))
            trace_writer['writer'].writeheader()
        trace_writer['writer'].writerow(row)
        rows.append(row); path.append(position)
        success = bool(info['is_success'] and info['bilateral_contact'] and not terminated)
        if success or terminated or truncated:
            break
    trace_writer['stream'].flush()
    result = {'controller': mode, 'controller_type': 'explicit_analytical_TEST' if policy is None else 'frozen_SAC_checkpoint',
        'training': False, 'case_seed': case_seed, 'scope': 'matched_easy_carry_diagnostic',
        'start_matches_other_controllers': True, 'reset': reset_info,
        'initial_base_xy_m': initial_xy.tolist(), 'fixed_goal_xy_m': goal_xy.tolist(),
        'initial_goal_vector_m': initial_vector.tolist(), 'initial_distance_m': float(np.linalg.norm(initial_vector)),
        'initial_action': [rows[0]['action_x'], rows[0]['action_y']],
        'initial_action_goal_cosine': rows[0]['action_goal_cosine'],
        'mean_action_goal_cosine': float(np.mean([row['action_goal_cosine'] for row in rows])),
        'fraction_actions_toward_goal': float(np.mean([row['action_dot_goal_m'] > 0 for row in rows])),
        'minimum_goal_distance_m': min(row['goal_distance_m'] for row in rows),
        'final_goal_distance_m': rows[-1]['goal_distance_m'],
        'net_distance_progress_m': float(np.linalg.norm(initial_vector)) - rows[-1]['goal_distance_m'],
        'success': success, 'stop_reason': 'retained_goal' if success else rows[-1]['failure_reason'] or 'horizon',
        'bilateral_contact_fraction': float(np.mean([row['bilateral_contact'] for row in rows])),
        'collision': any(row['collision'] for row in rows),
        'policy_decisions': len(rows), 'source_action_calls': sum(row['source_action_calls'] for row in rows),
        'physics_ticks': sum(row['physics_ticks'] for row in rows),
        'wall_seconds': time.perf_counter() - started, 'base_path_xy_m': path, **geometry}
    return result, current_start


def figure(env, episodes, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    colors = {'latest_learned': '#087e76', 'best_learned': '#cd8a32', 'direct_goal_TEST': '#7165a3'}
    labels = {'latest_learned': 'Latest checkpoint', 'best_learned': 'Best checkpoint', 'direct_goal_TEST': 'Direct goal · TEST'}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for axis, seed in zip(axes, CASE_SEEDS):
        group = [episode for episode in episodes if episode['case_seed'] == seed]
        points = np.concatenate([episode['base_path_xy_m'] for episode in group])
        goal = np.asarray(group[0]['fixed_goal_xy_m']); start = np.asarray(group[0]['initial_base_xy_m'])
        points = np.vstack([points, goal, start]); low = points.min(axis=0) - .45; high = points.max(axis=0) + .45
        for x, y, *_ in env.native.cube_positions:
            if x + .5 >= low[0] and x - .5 <= high[0] and y + .5 >= low[1] and y - .5 <= high[1]:
                axis.add_patch(Rectangle((x-.5, y-.5), 1, 1, color='#e2e8e8', zorder=0))
        for episode in group:
            xy = np.asarray(episode['base_path_xy_m'])
            axis.plot(xy[:, 0], xy[:, 1], color=colors[episode['controller']], lw=1.8, label=labels[episode['controller']])
            axis.scatter(*xy[-1], color=colors[episode['controller']], s=13, zorder=4)
        axis.add_patch(Circle(goal, env.tolerance, facecolor='none', edgecolor='#899b99', lw=.8, linestyle='--'))
        axis.scatter(*start, color='#263d46', marker='o', s=20, zorder=5)
        axis.scatter(*goal, color='#263d46', marker='*', s=55, zorder=5)
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect='equal',
                 title='Start %s' % seed, xlabel='x (m)', ylabel='y (m)')
        axis.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Matched 0.3 m carry goals · frozen checkpoints and explicit TEST control', fontsize=11)
    fig.tight_layout(rect=(0, .09, 1, .93)); fig.savefig(destination, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'runs/sac_carry_corrected01')
    parser.add_argument('--run-id', default='carry_checkpoint_diagnosis')
    parser.add_argument('--horizon', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 100:
        parser.error('Diagnostic horizon must be in [1, 100]')
    torch.set_num_threads(1)
    logger = RunLogger(args.run_id, kind='validation', skill='carry', name='Carry checkpoint diagnosis',
        algorithm='Frozen SAC + explicit TEST control', budget=9*args.horizon,
        summary='Same three physical starts; latest/best checkpoints versus direct-goal TEST controller. No training.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    source = args.source_run.resolve()
    source_config = json.loads((source/'config.json').read_text())
    generation = json.loads((source/'checkpoints/latest.json').read_text())['generation']
    records = {}; env = None; episodes = []; total = 0; started = time.perf_counter()
    try:
        policies = {}
        for mode, path in [('latest_learned', source/'checkpoints'/generation/'model.zip'),
                           ('best_learned', source/'checkpoints/best/model.zip')]:
            records[mode] = freeze(path, artifacts/'checkpoints'/f'{mode}.zip')
            policies[mode] = SAC.load(records[mode]['frozen_path'], device='cpu',
                replay_buffer_class=DictReplayBuffer, replay_buffer_kwargs={}, buffer_size=1)
            records[mode]['model_training_decisions'] = int(policies[mode].num_timesteps)
        atomic_json(artifacts/'checkpoints.json', records)
        identical = records['latest_learned']['sha256'] == records['best_learned']['sha256']
        if identical:
            logger.event('Latest and best are the same %s-decision checkpoint; repeated checks are not independent policies.' % records['latest_learned']['model_training_decisions'])
        atomic_json(logger.directory/'config.json', {'source_run': str(source), 'latest_generation': generation,
            'case_seeds': list(CASE_SEEDS), 'horizon': args.horizon, 'training': False,
            'evaluation_rule': 'Stop at first current retained success or physical failure; goals stay fixed.',
            'source_config': source_config,
            'inference_loading': 'HER replay replaced by unused size-one DictReplayBuffer; saved actor/critic parameters unchanged.'})
        env = CarrySkillEnv(scene_seed=source_config['scene_seed'], object_idx=source_config['object_idx'],
            seed=source_config['seed'] + 10000, horizon=args.horizon, evaluation=True)
        logger.update(setup_action_calls=env.setup_action_calls, setup_physics_ticks=env.setup_physics_ticks)
        logger.event('Physically generated start bank: %s native calls.' % env.setup_action_calls)
        expected = {}
        with (artifacts/'transitions.csv').open('w', newline='') as stream:
            writer = {'stream': stream, 'writer': None}
            for mode in ('latest_learned', 'best_learned', 'direct_goal_TEST'):
                for seed in CASE_SEEDS:
                    episode, initial = evaluate_case(env, policies.get(mode), mode, seed, expected.get(seed), writer)
                    expected.setdefault(seed, initial)
                    episodes.append(episode); total += episode['policy_decisions']
                    group = [item for item in episodes if item['controller'] == mode]
                    logger.metric(total, scope='diagnostic', metric_source='matched_checkpoint_diagnostic',
                        evaluation_scope=mode+' · easy carry diagnostic', success_rate=float(np.mean([item['success'] for item in group])),
                        eval_episodes=len(group), mean_goal_distance=float(np.mean([item['final_goal_distance_m'] for item in group])),
                        source_action_calls=sum(item['source_action_calls'] for item in episodes),
                        physics_ticks=sum(item['physics_ticks'] for item in episodes))
                    logger.event('%s / start%s: %s, goal%.3fm, initial alignment%.2f.' %
                        (mode, seed, episode['stop_reason'], episode['final_goal_distance_m'], episode['initial_action_goal_cosine']))
                    print(json.dumps({key: value for key, value in episode.items() if key != 'base_path_xy_m'}), flush=True)
        actual_direct = {item['case_seed']: item['success'] for item in episodes if item['controller'] == 'direct_goal_TEST'}
        for episode in episodes:
            episode['direct_goal_TEST_reachable'] = actual_direct[episode['case_seed']]
        aggregates = {}
        for mode in ('latest_learned', 'best_learned', 'direct_goal_TEST'):
            group = [item for item in episodes if item['controller'] == mode]
            aggregates[mode] = {'successes': sum(item['success'] for item in group), 'episodes': len(group),
                'collisions': sum(item['collision'] for item in group),
                'mean_initial_action_goal_cosine': float(np.mean([item['initial_action_goal_cosine'] for item in group])),
                'mean_final_goal_distance_m': float(np.mean([item['final_goal_distance_m'] for item in group])),
                'source_action_calls': sum(item['source_action_calls'] for item in group),
                'physics_ticks': sum(item['physics_ticks'] for item in group)}
        report = {'training': False, 'scope': 'matched_easy_carry_diagnostic', 'checkpoints': records,
            'latest_and_best_identical': identical,
            'scene': env.native.scene_manifest(), 'setup': env.setup_manifest, 'episodes': episodes,
            'aggregate_controls': aggregates, 'policy_decisions': total,
            'evaluation_source_action_calls': sum(item['source_action_calls'] for item in episodes),
            'evaluation_physics_ticks': sum(item['physics_ticks'] for item in episodes),
            'total_source_action_calls_including_setup': env.setup_action_calls + sum(item['source_action_calls'] for item in episodes),
            'total_physics_ticks_including_setup': env.setup_physics_ticks + sum(item['physics_ticks'] for item in episodes),
            'scene_settling_physics_ticks': env.native.settle_steps,
            'total_executed_physics_ticks_including_scene_settling': env.native.settle_steps+env.setup_physics_ticks+sum(item['physics_ticks'] for item in episodes),
            'wall_seconds': time.perf_counter() - started,
            'limits': 'Three fixed short-segment starts. TEST-controller success is physical feasibility evidence only; no full-maze or learned-system claim.'}
        atomic_json(artifacts/'diagnosis.json', report)
        figure(env, episodes, artifacts/'paths.png')
        for path, label, kind in [('diagnosis.json', 'Matched carry diagnosis', 'file'),
                                  ('transitions.csv', 'Action and progress records', 'file'),
                                  ('checkpoints.json', 'Frozen checkpoint identities', 'file'),
                                  ('paths.png', 'Three matched starts · learned vs TEST', 'image')]:
            logger.artifact('artifacts/'+path, label, kind)
        summary = '; '.join('%s %s/3' % (mode, value['successes']) for mode, value in aggregates.items())
        if identical:
            summary = 'SAC at %s decisions (latest=best) %s/3; explicit direct-goal TEST %s/3' % (records['latest_learned']['model_training_decisions'], aggregates['latest_learned']['successes'], aggregates['direct_goal_TEST']['successes'])
        logger.finish(summary=summary+'. Short-segment diagnostic only.')
        print(json.dumps({'aggregate_controls': aggregates, 'physics_cost': report['total_physics_ticks_including_setup']}), flush=True)
    except BaseException as error:
        logger.finish('failed', '%s: %s' % (type(error).__name__, error))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
