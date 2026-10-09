#!/usr/bin/env python3
"""The redesigned study's final evaluation (docs/project_spec.md): every frozen system once on the sealed
reserve bank (20 unseen layouts; standard 100 cases, wide 120 = 20 layouts x 2 objects x 3 seeded starts).

  python scripts/final_v2.py freeze     # writes configs/final_v2.json (systems, checkpoints, hashes, protocol)
  python scripts/final_v2.py run        # evaluates every system in the manifest (completed ones skipped)
  python scripts/final_v2.py report     # pooled delivery per cell with Wilson 95% intervals

Freeze applies the rules fixed on development data: checkpoint = the best training-time evaluation within
the 10M budget (later on ties), imitation = step 0 of the BC -> PPO run; grip after the grasp from
runs/dev_v2/grip_choice.json; upright carry; planned speeds 3 / holding 2; hand grasp <= 300 decisions.
`run` refuses a manifest whose checkpoints changed since the freeze.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs' / 'final_v2.json'
OUT = ROOT / 'runs' / 'final_v2'
OBJECTS = {'std': [14, 5, 2, 7, 10], 'wide': [0, 102]}
STARTS = {'std': 1, 'wide': 3}
SEEDS = range(5)
PROTOCOL = {'set': 'reserve', 'random_object_pose': True, 'physics': 'realistic', 'speed_scale': 3.0,
            'holding_speed_scale': 2.0, 'horizon': 300, 'carry_posture': 'upright', 'starts': STARTS,
            'objects': OBJECTS, 'primary_metric': 'delivered (complete_task_success)', 'workers': 40}


def best_step(run):
    rows = [json.loads(line) for line in (ROOT / 'runs' / run / 'metrics.jsonl').open() if '"evaluation"' in line]
    rows = [r for r in rows if r['transitions'] <= 10_000_100]
    return max(rows, key=lambda r: (r['carry_success_rate'], r['transitions']))['transitions']


def digest(run, step):
    base = ROOT / 'runs' / run / 'checkpoints' / f'step_{step:08d}'
    h = hashlib.sha256()
    for path in (Path(str(base) + '.zip'), Path(str(base) + '_vecnormalize.pkl')):
        h.update(path.read_bytes())
    return h.hexdigest()


def freeze():
    grip = {k: v['grip'] for k, v in json.loads((ROOT / 'runs/dev_v2/grip_choice.json').read_text()).items()}
    systems = []
    for task in ('std', 'wide'):
        systems.append({'id': f'gripper_planner_{task}', 'cell': 'gripper planner', 'task': task, 'controller': 'planner'})
        systems.append({'id': f'hand_scripted_{task}', 'cell': 'hand scripted', 'task': task, 'controller': 'teacher',
                        'grip': grip[f'hand_{task}']})
        for e in ('gripper', 'hand'):
            for s in SEEDS:
                for cell, run, step in (('BC', f'v2_{e}_bcppo_{task}_s{s}', 0),
                                        ('BC -> PPO', f'v2_{e}_bcppo_{task}_s{s}', None),
                                        ('PPO from scratch', f'v2_{e}_ppo_{task}_s{s}', None)):
                    step = best_step(run) if step is None else step
                    kind = {'BC': 'bc', 'BC -> PPO': 'bcppo', 'PPO from scratch': 'ppo'}[cell]
                    systems.append({'id': f'{e}_{kind}_{task}_s{s}', 'cell': f'{e} {cell}', 'task': task, 'controller': 'policy',
                                    'effector': 'gripper' if e == 'gripper' else 'allegro', 'run': run, 'step': step,
                                    'seed': s, 'grip': grip[f'{e}_{task}'], 'sha256': digest(run, step)})
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    MANIFEST.write_text(json.dumps({'frozen_on_commit': commit, 'protocol': PROTOCOL, 'systems': systems}, indent=1) + '\n')
    print(f'{len(systems)} systems frozen -> {MANIFEST.relative_to(ROOT)}')


def command(system):
    task, p = system['task'], PROTOCOL
    out = OUT / system['id']
    common = ['--set', 'reserve', '--allow-reserve', '--random-object-pose', '--workers', str(p['workers']),
              '--objects', *map(str, OBJECTS[task]), '--starts', str(STARTS[task]), '--output', str(out)]
    if system['controller'] == 'planner':
        return [sys.executable, 'scripts/eval.py', '--controller', 'reference', '--physics', p['physics'],
                '--planned-speed-scale', str(p['speed_scale']), '--holding-speed-scale', str(p['holding_speed_scale']),
                '--label', f"final v2 · {system['id']}", *common]
    policy = ['--policy', 'teacher'] if system['controller'] == 'teacher' else \
        ['--policy', system['run'], '--step', str(system['step']), '--effector', system['effector']]
    return [sys.executable, 'scripts/eval_hand.py', *policy, '--fingers-after-grasp', system['grip'],
            '--speed-scale', str(p['speed_scale']), '--holding-speed-scale', str(p['holding_speed_scale']),
            '--horizon', str(p['horizon']), '--carry-posture', p['carry_posture'], *common]


def run():
    manifest = json.loads(MANIFEST.read_text())
    for system in manifest['systems']:
        if 'sha256' in system and digest(system['run'], system['step']) != system['sha256']:
            sys.exit(f"{system['id']}: checkpoint changed since the freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    for system in manifest['systems']:
        out = OUT / system['id']
        if (out / 'summary.json').is_file():
            continue
        if out.exists():
            subprocess.run(['rm', '-rf', str(out)], check=True)
        with (OUT / f"{system['id']}.log").open('w') as log:
            subprocess.run(command(system), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        o = json.loads((out / 'summary.json').read_text())['overall']
        print(f"{system['id']:28s} {o['success']:4d} / {o['cases']} delivered", flush=True)
    report()


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report():
    manifest = json.loads(MANIFEST.read_text())
    cells = {}
    for system in manifest['systems']:
        path = OUT / system['id'] / 'summary.json'
        if path.is_file():
            o = json.loads(path.read_text())['overall']
            cells.setdefault((system['task'], system['cell']), []).append((system.get('seed'), o['success'], o['cases']))
    rows = []
    for (task, cell), seeds in cells.items():
        k, n = sum(s[1] for s in seeds), sum(s[2] for s in seeds)
        lo, hi = wilson(k, n)
        rows.append({'task': task, 'cell': cell, 'delivered': k, 'cases': n, 'rate': k / n, 'wilson95': [lo, hi],
                     'per_seed': [s[1] / s[2] for s in sorted(seeds, key=lambda s: -1 if s[0] is None else s[0])]})
        print(f"{task:4s} {cell:26s} {k / n:.2f} [{lo:.2f}, {hi:.2f}]  " + ' '.join(f'{r:.2f}' for r in rows[-1]['per_seed']))
    (OUT / 'report.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('freeze', 'run', 'report'))
    {'freeze': freeze, 'run': run, 'report': report}[parser.parse_args().action]()
