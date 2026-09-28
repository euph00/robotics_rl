#!/usr/bin/env python3
"""Frozen carry policy on longer fixed segments and one full-route diagnostic.

No training or core simulator changes. Reference paths describe the physically
collected bank and post hoc geometry; the policy sees only its normal observation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer

from diagnose_carry import CASE_SEEDS, evaluate_case, freeze
from pnr.carry import CarrySkillEnv
from pnr.telemetry import RunLogger, atomic_json


def reference_corners(env, start_arc, length):
    route = np.asarray(env.controller.path)
    keep = np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-6]
    route = route[keep]
    result = []
    for before, point, after in zip(route[:-2], route[1:-1], route[2:]):
        first, second = point-before, after-point
        if abs(first[0]*second[1]-first[1]*second[0]) < 1e-6:
            continue
        index = int(np.argmin(np.linalg.norm(env.route_points-point, axis=1)))
        arc = float(env.route_arcs[index])
        if start_arc + 1e-5 < arc < start_arc + length + 1e-5:
            result.append({'xy_m': point.tolist(), 'arc_length_m': arc})
    return result


def credit_assessment(route_length, minimum_decisions):
    decisions = 629
    rows = []
    for gamma in (.98, .99, .995, .997, .998, .999):
        rows.append({'gamma': gamma, 'discount_at_629_decisions': gamma**decisions,
            'effective_horizon_decisions': 1/(1-gamma),
            'discount_half_life_decisions': float(np.log(.5)/np.log(gamma)),
            'matching_terminal_failure_cost': -1/(1-gamma),
            'undiscounted_remaining_decisions': decisions,
            'cost_of_629_failed_progress_steps': -(1-gamma**decisions)/(1-gamma),
            'cost_of_1000_failed_progress_steps': -(1-gamma**1000)/(1-gamma)})
    return {'reference_route_length_m': route_length, 'minimum_decisions_at_command_limit': minimum_decisions,
        'comparison_reference_decisions': decisions, 'values': rows,
        'assessment': [
            'At gamma0.98, a terminal improvement629 decisions away contributes only gamma**629 of its value; the effective horizon is50 decisions.',
            'HER supplies short-horizon goal successes but does not itself establish long-route planning or obstacle avoidance.',
            'If gamma increases, keeping failure cost-50 can make early terminal failure cheaper than a long string of-1 rewards. The absorbing-failure-equivalent penalty is-1/(1-gamma).',
            'A gamma/reward change requires a new recorded protocol and consistent replay rewards; unchanged real replay rewards cannot be mixed silently with new HER recomputation.',
            'Discount calculations alone do not establish an appropriate gamma. No algorithm, replay, reward, or physics change is made by this diagnostic.'
        ]}


def plot_paths(env, episodes, artifacts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    def decorate(axis, episode, full=False):
        xy = np.asarray(episode['base_path_xy_m'])
        start = np.asarray(episode['initial_base_xy_m']); goal = np.asarray(episode['fixed_goal_xy_m'])
        if full:
            low, high = np.array([-4., -6.]), np.array([18.5, 6.])
        else:
            points = np.vstack([xy, start, goal])
            low, high = points.min(axis=0)-.6, points.max(axis=0)+.6
        for x, y, *_ in env.native.cube_positions:
            if x+.5 >= low[0] and x-.5 <= high[0] and y+.5 >= low[1] and y-.5 <= high[1]:
                axis.add_patch(Rectangle((x-.5, y-.5), 1, 1, color='#e2e8e8', zorder=0))
        if full:
            route = np.asarray(env.controller.path)
            axis.plot(route[:, 0], route[:, 1], color='#879b99', lw=1, linestyle='--', label='Reference route · diagnostic only')
            table_low, table_high = env.native.p.getAABB(env.native.table_id)
            axis.add_patch(Rectangle(table_low[:2], table_high[0]-table_low[0], table_high[1]-table_low[1], color='#c8a783', alpha=.7))
        axis.plot(xy[:, 0], xy[:, 1], color='#087e76', lw=2, label='Frozen SAC policy')
        axis.scatter(*start, c='#263d46', s=22, zorder=4, label='Start')
        axis.scatter(*goal, c='#263d46', s=55, marker='*', zorder=4, label='Goal')
        axis.scatter(*xy[-1], c='#b95145' if not episode['success'] else '#087e76', s=25, marker='x', zorder=5, label='Stop')
        axis.add_patch(Circle(goal, env.tolerance, facecolor='none', edgecolor='#899b99', lw=.8, linestyle='--'))
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect='equal', xlabel='x (m)', ylabel='y (m)')
        axis.spines[['top', 'right']].set_visible(False)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for axis, episode in zip(axes.flat, episodes[:-1]):
        decorate(axis, episode)
        axis.set_title('%s · start%s\n%s, error%.2fm' % (episode['gate'], episode['case_seed'], episode['stop_reason'], episode['final_goal_distance_m']), fontsize=10)
    fig.suptitle('Fixed segment goals · one frozen carry checkpoint', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(artifacts/'segments.png', dpi=140); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    decorate(axis, episodes[-1], full=True)
    axis.set_title('Full carry diagnostic · %s after%s decisions' % (episodes[-1]['stop_reason'], episodes[-1]['policy_decisions']))
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(artifacts/'full_carry.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'runs/sac_carry_corrected01')
    parser.add_argument('--run-id', default='carry_longer_gate01')
    args = parser.parse_args()
    torch.set_num_threads(1)
    logger = RunLogger(args.run_id, kind='validation', skill='carry', name='Carry longer-goal gate',
        algorithm='Frozen SAC checkpoint', budget=1600,
        summary='Six fixed1.2/2.5m goals and one full-route diagnostic; no training or waypoint assistance.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    env = None; started = time.perf_counter(); episodes = []; total = 0
    try:
        source = args.source_run.resolve()
        config = json.loads((source/'config.json').read_text())
        generation = json.loads((source/'checkpoints/latest.json').read_text())['generation']
        checkpoint = freeze(source/'checkpoints'/generation/'model.zip', artifacts/'checkpoints/model.zip')
        model = SAC.load(checkpoint['frozen_path'], device='cpu', replay_buffer_class=DictReplayBuffer,
                         replay_buffer_kwargs={}, buffer_size=1)
        checkpoint['model_training_decisions'] = int(model.num_timesteps)
        checkpoint['model_gamma'] = float(model.gamma)
        atomic_json(artifacts/'checkpoint.json', checkpoint)
        atomic_json(logger.directory/'config.json', {'training': False, 'source_run': str(source), 'source_config': config,
            'latest_generation': generation, 'segment_lengths_m': [1.2, 2.5], 'segment_horizon': 100,
            'full_horizon': 1000, 'case_seeds': list(CASE_SEEDS), 'inference_waypoints': False})
        env = CarrySkillEnv(scene_seed=config['scene_seed'], object_idx=config['object_idx'], seed=config['seed']+10000,
                            horizon=100, evaluation=True)
        logger.update(setup_action_calls=env.setup_action_calls, setup_physics_ticks=env.setup_physics_ticks)
        logger.event('Frozen policy at%s decisions; physical bank uses%s native calls.' % (model.num_timesteps, env.setup_action_calls))
        with (artifacts/'transitions.csv').open('w', newline='') as stream:
            writer = {'stream': stream, 'writer': None}
            cases = [(length, seed) for length in (1.2, 2.5) for seed in CASE_SEEDS] + [('full', CASE_SEEDS[0])]
            for length, seed in cases:
                full = length == 'full'
                env.horizon = 1000 if full else 100
                if not full:
                    env.curriculum_level = env.segment_lengths.index(length)
                gate = 'full_carry' if full else 'segment_%sm' % length
                episode, _ = evaluate_case(env, model, gate, seed, None, writer,
                    reset_options={'evaluation_mode': 'full' if full else None})
                episode.pop('start_matches_other_controllers', None)
                episode.update(gate=gate, scope='bounded_longer_carry_diagnostic',
                    interpretation='Footprint geometry screen only; no direct-goal TEST is run in this longer gate.')
                corners = reference_corners(env, episode['reset']['start_arc_length_m'], episode['reset']['segment_length_m'])
                episode['reference_corners_on_segment'] = corners
                episode['reference_corner_count'] = len(corners)
                episode['policy_passed_corner_neighborhood_count'] = sum(
                    int(np.min(np.linalg.norm(np.asarray(episode['base_path_xy_m']) - item['xy_m'], axis=1)) < .25) for item in corners)
                episodes.append(episode); total += episode['policy_decisions']
                group = [item for item in episodes if item['gate'] == gate]
                logger.metric(total, scope='diagnostic', metric_source='frozen_carry_gate',
                    evaluation_scope=gate+' · frozen policy diagnostic',
                    success_rate=float(np.mean([item['success'] for item in group])), eval_episodes=len(group),
                    mean_goal_distance=float(np.mean([item['final_goal_distance_m'] for item in group])),
                    source_action_calls=sum(item['source_action_calls'] for item in episodes),
                    physics_ticks=sum(item['physics_ticks'] for item in episodes))
                logger.event('%s/start%s: %s, goal%.2fm, reference corners%s.' %
                    (gate, seed, episode['stop_reason'], episode['final_goal_distance_m'], len(corners)))
                print(json.dumps({key: value for key, value in episode.items() if key != 'base_path_xy_m'}), flush=True)
        credit = credit_assessment(env.route_length, env.full_carry_min_decisions)
        atomic_json(artifacts/'discount_assessment.json', credit)
        aggregates = {gate: {'successes': sum(item['success'] for item in episodes if item['gate'] == gate),
                            'episodes': sum(item['gate'] == gate for item in episodes),
                            'collisions': sum(item['collision'] for item in episodes if item['gate'] == gate)}
                      for gate in ('segment_1.2m', 'segment_2.5m', 'full_carry')}
        report = {'training': False, 'scope': 'bounded_longer_carry_diagnostic', 'checkpoint': checkpoint,
            'scene': env.native.scene_manifest(), 'setup': env.setup_manifest,
            'episodes': episodes, 'aggregate_gates': aggregates, 'credit_assessment': credit,
            'policy_decisions': total, 'evaluation_source_action_calls': sum(item['source_action_calls'] for item in episodes),
            'evaluation_physics_ticks': sum(item['physics_ticks'] for item in episodes),
            'total_source_action_calls_including_setup': env.setup_action_calls+sum(item['source_action_calls'] for item in episodes),
            'total_physics_ticks_including_setup': env.setup_physics_ticks+sum(item['physics_ticks'] for item in episodes),
            'scene_settling_physics_ticks': env.native.settle_steps,
            'total_executed_physics_ticks_including_scene_settling': env.native.settle_steps+env.setup_physics_ticks+sum(item['physics_ticks'] for item in episodes),
            'wall_seconds': time.perf_counter()-started,
            'limits': 'Six fixed segments and one original-carry-start episode. This does not evaluate original-start pickup or learned whole-system success.'}
        atomic_json(artifacts/'diagnosis.json', report)
        plot_paths(env, episodes, artifacts)
        for filename, label, kind in [('diagnosis.json', 'Longer carry diagnosis', 'file'),
            ('transitions.csv', 'Gate action and progress records', 'file'),
            ('discount_assessment.json', 'Long-route discount assessment', 'file'),
            ('segments.png', 'Fixed1.2/2.5m goals', 'image'), ('full_carry.png', 'Full carry first failure', 'image')]:
            logger.artifact('artifacts/'+filename, label, kind)
        summary = '; '.join('%s %s/%s' % (gate, value['successes'], value['episodes']) for gate, value in aggregates.items())
        logger.finish(summary=summary+'. Frozen checkpoint diagnostic only.')
        print(json.dumps({'aggregate_gates': aggregates, 'checkpoint_decisions': model.num_timesteps,
                          'gamma_098_at629': .98**629}), flush=True)
    except BaseException as error:
        logger.finish('failed', '%s: %s' % (type(error).__name__, error)); raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
