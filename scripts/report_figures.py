#!/usr/bin/env python3
"""Report figures from the study's evaluations and training logs (the dashboard's own aggregation).

  python scripts/report_figures.py                 # writes docs/figures/*.png and *.pdf

results    delivered share per system and task (mean over seeds, one dot per seed; darker: within the deadline)
stages     picked vs delivered per system: where each system loses objects
algorithms PPO vs RLPD: lift-and-carry success against environment steps and training hours
training   the hand's imitation + RL pipeline: BC, PPO, then the task-carry fine-tune, per seed
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dashboard'))
import server  # noqa: E402

# The dashboard's validated light-mode palette (dashboard/public/style.css).
SOURCE = {'planned': '#3b6fd8', 'imitation': '#27a06b', 'imitation_rl': '#d97a10', 'rl': '#b8357a'}
INK, MUTED, GRID = '#1d2330', '#5f687b', '#e3e6ec'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def save(fig, out, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{suffix}')
    plt.close(fig)
    print(f'wrote {out / name}.png')


def systems(study):
    """Conditions with at least one evaluated seed: (effector, controller, source, {task: [tallies]})."""
    effectors = {e['key']: e['label'] for e in study['effectors']}
    controllers = {c['key']: c['label'] for c in study['controllers']}
    rows = []
    for c in study['conditions']:
        results = {task: [r for r in runs if r.get('cases')] for task, runs in (c.get('results') or {}).items() if runs}
        if c.get('pipeline') and any(results.values()):
            source = 'imitation_rl' if 'imitation_rl' in c['pipeline'] else 'imitation' if 'imitation' in c['pipeline'] else 'planned'
            rows.append(dict(effector=effectors[c['effector']], controller=controllers[c['controller']], source=source,
                             results=results))
    return rows


SHORT = {'Scripted': 'Scripted', 'Imitation': 'Imitation', 'Imitation + RL': 'Imit.+RL', 'RL from scratch': 'RL only'}


def separate(ax, rows):
    """Short controller names as tick labels, the end effector once under each group, a divider between groups."""
    ax.set_xticks(range(len(rows)), [SHORT.get(r['controller'], r['controller']) for r in rows], fontsize=7)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['effector'], []).append(i)
        if i and row['effector'] != rows[i - 1]['effector']:
            ax.axvline(i - .5, color=MUTED, lw=.8, ls=':')
    for effector, members in groups.items():
        ax.text(np.mean(members), -.16, effector, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=8, color=INK)


def share(tally, key):
    return tally[key] / tally['cases']


def results_figure(study, out):
    tasks = study['tasks']
    rows = systems(study)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 2.8), sharey=True)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        labels = []
        for i, row in enumerate(rows):
            seeds = row['results'].get(task['key']) or []
            labels.append(f"{row['controller']}\n{row['effector'].split()[-1].lower()}")
            if not seeds:
                ax.text(i, .02, 'n/a', ha='center', color=MUTED, fontsize=8)
                continue
            delivered = [share(t, 'success') for t in seeds]
            timed = [share(t, 'timed') for t in seeds]
            color = SOURCE[row['source']]
            ax.bar(i, np.mean(delivered), .7, color=color, alpha=.38)
            ax.bar(i, np.mean(timed), .7, color=color)
            ax.scatter([i] * len(seeds), delivered, s=10, color=INK, zorder=3)
            ax.text(i, np.mean(delivered) + .03, f'{np.mean(delivered):.0%}', ha='center', fontsize=8, color=INK)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, .5, 1], ['0%', '50%', '100%'])
        ax.set_title(task['label'], fontsize=9, color=INK)
        separate(ax, rows)
    np.atleast_1d(axes)[0].set_ylabel('held-out cases delivered')
    handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE[s['key']]) for s in study['sources']]
    fig.legend(handles, [s['label'] for s in study['sources']], loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.08))
    fig.text(.5, -.14, 'light bar: delivered · dark bar: delivered within 15,000 physics ticks · dots: training seeds',
             ha='center', color=MUTED, fontsize=8)
    save(fig, out, 'results')


def stages_figure(study, out):
    tasks = study['tasks']
    rows = systems(study)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 2.8), sharey=True)
    for ax, task in zip(np.atleast_1d(axes), tasks):
        labels = []
        for i, row in enumerate(rows):
            seeds = row['results'].get(task['key']) or []
            labels.append(f"{row['controller']}\n{row['effector'].split()[-1].lower()}")
            if not seeds:
                continue
            picked = np.mean([share(t, 'picked') for t in seeds])
            delivered = np.mean([share(t, 'success') for t in seeds])
            color = SOURCE[row['source']]
            ax.bar(i, picked, .7, color='none', edgecolor=color, hatch='////', linewidth=1)
            ax.bar(i, delivered, .7, color=color)
            given = delivered / picked if picked else 0
            ax.text(i, picked + .03, f'{given:.0%}', ha='center', fontsize=8, color=INK)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, .5, 1], ['0%', '50%', '100%'])
        ax.set_title(task['label'], fontsize=9, color=INK)
        separate(ax, rows)
    np.atleast_1d(axes)[0].set_ylabel('share of held-out cases')
    fig.text(.5, -.14, 'hatched: picked (grasped and lifted) · solid: delivered · number: delivered given picked',
             ha='center', color=MUTED, fontsize=8)
    save(fig, out, 'stages')


def algorithms_figure(study, out):
    comparison = study.get('comparison')
    if not comparison:
        return
    colors = {'ppo': SOURCE['imitation_rl'], 'rlpd': SOURCE['rl']}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, key, label in ((axes[0], 'steps', 'environment steps (millions)'), (axes[1], 'hours', 'training hours')):
        for algorithm in comparison['algorithms']:
            for k, curve in enumerate(algorithm['curves']):
                points = [p for p in curve['points'] if p.get(key) is not None]
                if not points:
                    continue
                xs = [p[key] / (1e6 if key == 'steps' else 1) for p in points]
                ax.plot(xs, [p['carried'] for p in points], color=colors.get(algorithm['key'], MUTED), lw=1.5,
                        alpha=.85, label=algorithm['label'] if k == 0 else None, marker='o', ms=2.5)
        ax.set_xlabel(label)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1], ['0%', '50%', '100%'])
    axes[0].set_ylabel('lift and carry (hand task)')
    axes[0].legend(loc='lower right')
    fig.text(.5, -.1, comparison['label'] + ' · one line per seed', ha='center', color=MUTED, fontsize=8)
    save(fig, out, 'algorithms')


def training_figure(study, runs_root, out):
    condition = next((c for c in study['conditions'] if c.get('training')), None)
    if not condition:
        return
    fig, axes = plt.subplots(1, len(condition['training']), figsize=(3.4 * len(condition['training']), 2.6), sharey=True)
    for ax, (task, runs) in zip(np.atleast_1d(axes), condition['training'].items()):
        sources = []
        for run in runs:                                  # fine-tunes' source runs, when not listed themselves
            source = str((server.read_json(runs_root / run / 'config.json') or {}).get('init_from') or '').split('/')[0]
            if source and source not in runs and source not in sources:
                sources.append(source)
        for run in sources + runs:
            config = server.read_json(runs_root / run / 'config.json') or {}
            offset = 0
            if config.get('init_from'):                   # fine-tunes continue the source run's step axis
                offset = int(str(config['init_from']).rsplit('/', 1)[1])
            rows = [r for r in server.read_records(runs_root / run / 'metrics.jsonl', limit=2000)
                    if r.get('metric_source') == 'evaluation']
            if not rows:
                continue
            tc = bool(config.get('task_carry'))
            ax.plot([(offset + r['transitions']) / 1e6 for r in rows], [r['carry_success_rate'] for r in rows],
                    color=SOURCE['imitation_rl'], lw=1.3, ls='--' if tc else '-', marker='s' if tc else 'o', ms=2.5)
            if rows[0]['transitions'] == 0 and not offset:   # step 0 of a cloned run: the imitation policy
                ax.scatter([0], [rows[0]['carry_success_rate']], s=22, color=SOURCE['imitation'], zorder=3)
        ax.set_title(next(t['label'] for t in study['tasks'] if t['key'] == task), fontsize=9, color=INK)
        ax.set_xlabel('environment steps (millions)')
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1], ['0%', '50%', '100%'])
    np.atleast_1d(axes)[0].set_ylabel('lift and carry (hand task)')
    handles = [plt.Line2D([], [], color=SOURCE['imitation'], marker='o', ls='', ms=5),
               plt.Line2D([], [], color=SOURCE['imitation_rl'], lw=1.5, marker='o', ms=2.5),
               plt.Line2D([], [], color=SOURCE['imitation_rl'], lw=1.5, ls='--', marker='s', ms=2.5)]
    fig.legend(handles, ['imitation (BC)', 'PPO, synthetic carry', 'PPO fine-tune, the task\'s retreat, tuck and carry'],
               loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.1), fontsize=8)
    save(fig, out, 'training')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs' / 'figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = server.study_progress(args.runs.resolve())
    results_figure(study, args.output)
    stages_figure(study, args.output)
    algorithms_figure(study, args.output)
    training_figure(study, args.runs.resolve(), args.output)
    tallies = {f"{c['effector']}/{c['controller']}/{task}": [(r['id'], r.get('success'), r.get('timed'), r.get('picked'), r.get('cases'))
                                                                for r in runs]
               for c in study['conditions'] for task, runs in (c.get('results') or {}).items() if runs}
    (args.output / 'tallies.json').write_text(json.dumps(tallies, indent=1))


if __name__ == '__main__':
    main()
