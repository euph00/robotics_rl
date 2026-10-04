#!/usr/bin/env python3
"""The tuning sweep's selection rule (scripts/tune_v2.sh), applied per end effector.

score = mean lift-and-carry of the last three training-time evaluations (3.0, 3.5, 4.0M steps); for each
factor, the level with the higher mean score over its four runs; prints the scores, main effects and
the tuned recipe, and writes them to runs/tune_v2_report.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ['- - - -', '+ - - +', '- + - +', '+ + - -', '- - + +', '+ - + -', '- + + -', '+ + + +']
FACTORS = [('learning_rate', '3e-4', '1e-4'), ('bc_log_std', '-1.2', '-0.7'),
           ('demos_per_object', '100', '300'), ('bc_interleave', '0', '200')]


def score(run):
    path = ROOT / 'runs' / run / 'metrics.jsonl'
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.open() if '"evaluation"' in line]
    last = [r['carry_success_rate'] for r in rows if r['transitions'] >= 2_999_000]
    return sum(last) / len(last) if len(last) >= 3 else None


def main():
    report = {}
    for effector in ('gripper', 'hand'):
        scores = [score(f'v2tune_{effector}_c{k + 1}') for k in range(8)]
        print(f'== {effector}')
        for k, (config, s) in enumerate(zip(CONFIGS, scores)):
            print(f'  c{k + 1} {config}  {"running" if s is None else f"{s:.3f}"}')
        if any(s is None for s in scores):
            report[effector] = {'scores': scores, 'complete': False}
            continue
        recipe, effects = {}, {}
        for f, (name, low, high) in enumerate(FACTORS):
            minus = [s for c, s in zip(CONFIGS, scores) if c.split()[f] == '-']
            plus = [s for c, s in zip(CONFIGS, scores) if c.split()[f] == '+']
            effects[name] = sum(plus) / 4 - sum(minus) / 4
            recipe[name] = high if effects[name] > 0 else low
            print(f'  {name:18s} {low:>6s} {sum(minus) / 4:.3f} | {high:>6s} {sum(plus) / 4:.3f}  -> {recipe[name]}')
        report[effector] = {'scores': scores, 'effects': effects, 'recipe': recipe, 'complete': True}
    (ROOT / 'runs' / 'tune_v2_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
