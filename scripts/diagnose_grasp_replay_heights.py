#!/usr/bin/env python3
"""Read-only online grasp replay stratification by measured physical start height.

This reports stochastic training episodes, not frozen-policy evaluation trials.
The demonstration prefix is excluded using checkpoint import accounting. Exact
snapshot EE/object offsets from that dataset identify nominal start heights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from stable_baselines3.common.save_util import load_from_pkl
from pnr.telemetry import RunLogger, atomic_json
from pnr.train import validate_replay_layout

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows, heights):
    result = []
    for height in heights:
        selected = [row for row in rows if row['start_height_m'] == height]
        total = len(selected)
        hits = sum(row['episode_success_any_hit'] for row in selected)
        retained = sum(row['retained_goal_at_end'] for row in selected)
        result.append({'start_height_m': height, 'episodes': total,
                       'successes_any_hit': hits, 'success_rate_any_hit': hits / total if total else None,
                       'retained_goal_at_end': retained, 'retained_goal_end_rate': retained / total if total else None,
                       'physical_failures': sum(row['physical_failure'] for row in selected)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default='sac_grasp_demoseed01')
    parser.add_argument('--generation')
    parser.add_argument('--window', type=int, default=8000)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = ROOT / 'runs' / args.source_run
    generation = args.generation or json.loads((source / 'checkpoints/latest.json').read_text())['generation']
    checkpoint = source / 'checkpoints' / generation
    progress = json.loads((checkpoint / 'progress.json').read_text())
    replay_path = checkpoint / 'replay.pkl'
    before = {name: sha256(checkpoint / name) for name in ('model.zip', 'replay.pkl', 'progress.json')}
    replay = load_from_pkl(replay_path)
    workers = replay.n_envs
    validate_replay_layout(replay, workers, 'grasp')
    if replay.full:
        raise ValueError('This bounded diagnostic requires a non-wrapped replay buffer')
    demos = progress['demonstrations']
    imported = int(demos['transitions_imported'])
    if imported % workers:
        raise ValueError('Demonstration prefix must occupy whole vector steps')
    prefix = imported // workers
    if replay.pos * workers != imported + progress['transitions']:
        raise ValueError('Replay/accounting mismatch; cannot infer online transition coordinates')
    demo_path = Path(demos['path'])
    if sha256(demo_path) != demos['sha256']:
        raise ValueError('Demonstration dataset changed since checkpoint import')
    dataset = pickle.loads(demo_path.read_bytes())
    offsets = {}
    for episode in dataset['episodes']:
        initial = episode['transitions'][0]['observation']['observation']
        height = float(episode['summary']['start_height_m'])
        offsets.setdefault(height, []).append(float(initial[30] - initial[37]))
    prototypes = {height: float(np.median(values)) for height, values in offsets.items()}
    rows = []
    completed_demonstrations = 0
    excluded_lower_goals = 0
    for worker in range(workers):
        starts = np.unique(replay.ep_start[:replay.pos, worker][replay.ep_length[:replay.pos, worker] > 0])
        for start in starts:
            start = int(start)
            length = int(replay.ep_length[start, worker])
            end = start + length - 1
            if end >= replay.pos:
                raise ValueError('Completed episode extends past filled replay')
            if not replay.dones[end, worker] or replay.dones[start:end, worker].any():
                raise ValueError('Episode done markers disagree with HER boundaries')
            if not (replay.ep_start[start:end + 1, worker] == start).all():
                raise ValueError('HER episode crosses another physical reset')
            if not (replay.ep_length[start:end + 1, worker] == length).all():
                raise ValueError('HER episode length is inconsistent')
            if start < prefix:
                if end >= prefix:
                    raise ValueError('Demonstration episode crosses online boundary')
                completed_demonstrations += 1
                continue
            obs = replay.observations['observation'][start, worker]
            desired = replay.observations['desired_goal'][start, worker]
            infos = replay.infos[start:end + 1, worker]
            if not np.allclose(replay.observations['desired_goal'][start:end + 1, worker], desired, atol=1e-6, rtol=0):
                raise ValueError('Desired goal changes inside an episode')
            lift = float(desired[2] - obs[37])
            level = int(infos[0]['curriculum_level'])
            if abs(lift - .20) > .001 or level < 3:
                excluded_lower_goals += 1
                continue
            offset = float(obs[30] - obs[37])
            height = min(prototypes, key=lambda value: abs(offset - prototypes[value]))
            error = abs(offset - prototypes[height])
            if error > .005:
                raise ValueError('Initial geometry does not match a measured snapshot')
            end_info = infos[-1]
            any_hit = any(bool(info.get('is_success')) for info in infos)
            if any_hit != bool(end_info['episode_success']):
                raise ValueError('Cumulative episode success disagrees with transition success')
            following = replay.next_observations['achieved_goal'][start:end + 1, worker]
            rows.append({'worker': worker, 'replay_start': start, 'episode_decisions': length,
                'online_start_decisions': (start - prefix) * workers,
                'online_end_decisions': (end + 1 - prefix) * workers,
                'start_height_m': height, 'initial_ee_minus_object_z_m': offset,
                'prototype_error_m': error, 'requested_lift_m': lift, 'curriculum_level': level,
                'episode_success_any_hit': any_hit, 'retained_goal_at_end': bool(end_info['is_success']),
                'physical_failure': bool(end_info['physical_failure']),
                'max_object_lift_m': float(np.max(following[:, 2]) - obs[37]),
                'action_rejections': sum(bool(info.get('action_rejected')) for info in infos),
                'bilateral_contact_decisions': sum(bool(info.get('bilateral_contact')) for info in infos)})
    cutoff = max(0, progress['transitions'] - args.window)
    recent = [row for row in rows if row['online_start_decisions'] >= cutoff]
    grouped = aggregate(recent, sorted(prototypes))
    if before != {name: sha256(checkpoint / name) for name in before}:
        raise ValueError('Source checkpoint changed during read-only diagnosis')
    report = {'training': False, 'new_physics_calls': 0, 'scope': 'stochastic_online_training_replay',
        'source_run': args.source_run, 'generation': generation, 'source_sha256': before,
        'online_checkpoint_decisions': progress['transitions'], 'window_start_online_decisions': cutoff,
        'window_end_online_decisions': progress['transitions'], 'selection': 'Whole completed episodes starting inside window; fixed20cm lift goal and curriculum>=3',
        'demonstration_transitions_excluded': imported, 'demonstration_episodes_excluded': completed_demonstrations,
        'lower_goal_online_episodes_excluded': excluded_lower_goals, 'height_prototypes_ee_minus_object_z_m': prototypes,
        'maximum_height_mapping_error_m': max(row['prototype_error_m'] for row in rows),
        'recent_completed_episodes': len(recent), 'recent_successes_any_hit': sum(row['episode_success_any_hit'] for row in recent),
        'recent_by_height': grouped, 'all_completed20cm_online_by_height': aggregate(rows, sorted(prototypes)),
        'interpretation': 'Success means any retained goal hit during stochastic training; end retention is reported separately. Shared training history and starts are not independent frozen-policy trials. No whole-task success claim.',
        'source_checkpoint_unchanged': True}
    logger = RunLogger(args.run_id, kind='diagnostic', name='Grasp · online start-height diagnosis', skill='grasp',
        summary=f"Recent online episodes: {report['recent_successes_any_hit']}/{len(recent)} any-hit; higher starts fail.")
    artifacts = logger.directory / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    atomic_json(artifacts / 'diagnosis.json', report)
    with (artifacts / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(grouped)); width = .34
    for shift, field, color, label in [(-.5, 'success_rate_any_hit', '#218b8b', 'Any goal hit'), (.5, 'retained_goal_end_rate', '#485d73', 'Retained at end')]:
        values = [item[field] or 0 for item in grouped]
        bars = ax.bar(x + shift * width, values, width, color=color, label=label)
        key = 'successes_any_hit' if field == 'success_rate_any_hit' else 'retained_goal_at_end'
        ax.bar_label(bars, labels=[f"{item[key]}/{item['episodes']}" for item in grouped], padding=3, fontsize=9)
    ax.set(ylim=(0, 1.18), yticks=[0, .5, 1], yticklabels=['0%', '50%', '100%'],
        xticks=x, xticklabels=[f"{item['start_height_m']*100:g} cm" for item in grouped], xlabel='Nominal pregrasp start height', ylabel='Training episode success')
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False, loc='upper right')
    fig.suptitle(f"Online grasp episodes · {cutoff//1000}k–{progress['transitions']//1000}k decisions", x=.12, ha='left')
    ax.set_title(f'20 cm goals · {imported:,} demonstration decisions excluded · shared training history', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(artifacts / 'start_heights.png', dpi=160); plt.close(fig)
    logger.artifact('artifacts/start_heights.png', 'Online grasp success by initial height', 'image')
    logger.artifact('artifacts/diagnosis.json', 'Read-only replay stratification and provenance', 'file')
    logger.artifact('artifacts/episodes.csv', 'Completed online20cm-goal episodes', 'file')
    result_text = ', '.join(f"{item['start_height_m']*100:g}cm {item['successes_any_hit']}/{item['episodes']}" for item in grouped)
    logger.finish('complete', f"Online replay any-hit by start: {result_text}; no new physics.")
    print(json.dumps({'generation': generation, 'recent_by_height': grouped, 'artifacts': str(artifacts)}, indent=2))


if __name__ == '__main__':
    main()
