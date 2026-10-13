#!/usr/bin/env python3
"""Build the project site's data bundle (site/data/results.js) from the run records under runs/.

Measured results are read from the evaluation and training records; results of the earlier phases, whose
runs are archived or were removed with their code, are transcribed from the logs named in `source`.

  python scripts/site_data.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
OUT = ROOT / 'site' / 'data' / 'results.js'      # a script, so the page also opens from disk (file://)
SEEDS = range(5)
NAMES = {14: 'foam brick', 5: 'banana', 2: 'soup can', 7: 'apple', 10: 'mug', 0: 'master chef can', 102: 'soup can ×1.6'}
CELLS = [('gripper', 'planner', 'gripper_planner_{t}', False), ('gripper', 'imitation', 'gripper_bc_{t}_s{s}', True),
         ('gripper', 'imitation_rl', 'gripper_bcppo_{t}_s{s}', True), ('gripper', 'rl_scratch', 'gripper_ppo_{t}_s{s}', True),
         ('hand', 'scripted', 'hand_scripted_{t}', False), ('hand', 'imitation', 'hand_bc_{t}_s{s}', True),
         ('hand', 'imitation_rl', 'hand_bcppo_{t}_s{s}', True), ('hand', 'rl_scratch', 'hand_ppo_{t}_s{s}', True)]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0., c - h), 4), round(min(1., c + h), 4)]


def rows(path):
    return [json.loads(line) for line in path.open()]


def cell_block(root, ids):
    """Pooled delivery, per seed, per object and failure stage over the evaluations `ids`."""
    per_seed, pooled = [], []
    for i in ids:
        r = rows(root / i / 'results.jsonl')
        per_seed.append(round(sum(x['success'] for x in r) / len(r), 4))
        pooled += r
    k, n = sum(x['success'] for x in pooled), len(pooled)
    objects, stages = {}, {}
    for x in pooled:
        o = objects.setdefault(NAMES.get(x['object_idx'], str(x['object_idx'])), [0, 0])
        o[0] += x['success']; o[1] += 1
        stage = 'delivered' if x['success'] else (x.get('failed_stage') or 'other')
        stages[stage] = stages.get(stage, 0) + 1
    return {'delivered': k, 'cases': n, 'rate': round(k / n, 4), 'wilson95': wilson(k, n), 'per_seed': per_seed,
            'per_object': {name: round(a / b, 4) for name, (a, b) in objects.items()},
            'stages': {s: round(c / n, 4) for s, c in sorted(stages.items(), key=lambda kv: -kv[1])}}


def matrix(root, dev=False):
    """Every cell of the study on one bank; on development data the BC -> PPO evaluations carry the grip's name."""
    out = {}
    for task in ('std', 'wide'):
        for effector, controller, pattern, seeded in CELLS:
            tail = '_frozen' if dev and controller == 'imitation_rl' else ''
            ids = [pattern.format(t=task, s=s) + tail for s in SEEDS] if seeded else [pattern.format(t=task)]
            out[f'{effector}.{controller}.{task}'] = cell_block(root, ids)
    return out


def curve(run, keys=('transitions', 'carry_success_rate')):
    r = [json.loads(line) for line in (RUNS / run / 'metrics.jsonl').open() if '"evaluation"' in line]
    return [[round(x[k], 4) if isinstance(x[k], float) else x[k] for k in keys] for x in r]


def main():
    data = {}
    data['reserve'] = matrix(RUNS / 'final_v2')
    data['dev'] = matrix(RUNS / 'dev_v2', dev=True)
    data['grip_choice'] = json.loads((RUNS / 'dev_v2' / 'grip_choice.json').read_text())
    data['campaign'] = {f'{e}.{recipe}.{task}': [curve(f'v2_{e}_{recipe}_{task}_s{s}') for s in SEEDS]
                        for e in ('gripper', 'hand') for recipe in ('bcppo', 'ppo') for task in ('std', 'wide')}
    data['tuning'] = json.loads((RUNS / 'tune_v2_report.json').read_text())
    data['algorithms'] = {f'{e}.{a}': [curve(f'v2c3_{e}_{a}_s{s}', ('transitions', 'carry_success_rate', 'train_seconds'))
                                       for s in SEEDS] for e in ('gripper', 'hand') for a in ('ppo', 'rlpd')}
    data['algorithms_no_warmup'] = {e: curve(f'v2c3_{e}_rlpd_nowarmup_s0') for e in ('gripper', 'hand')}
    data['rig_feasibility'] = {'source': 'runs/hand_r_bc03 (teacher demos -> BC -> PPO on the idealised rig; can, apple, mug)',
                               'curve': curve('hand_r_bc03')}
    data['first_study'] = {
        'source': 'runs/p_* (hold-out bank, 3 seeds, 2026-10-03/04; development evidence after the redesign)',
        'cells': {name: [json.loads((RUNS / i / 'summary.json').read_text())['overall']['success_rate'] for i in ids]
                  for name, ids in {
                      'gripper.planner.std': ['p_gripper_planner_standard'], 'gripper.planner.wide': ['p_gripper_planner_wide'],
                      'gripper.imitation.std': [f'p_gripper_bc_standard_s{s}' for s in range(3)],
                      'gripper.imitation_sac.std': [f'p_gripper_bcrl_standard_s{s}' for s in range(3)],
                      'hand.scripted.std': ['p_hand_scripted_standard'], 'hand.scripted.wide': ['p_hand_scripted_wide'],
                      'hand.imitation.std': [f'p_hand_bc_standard_s{s}' for s in range(3)],
                      'hand.imitation_ppo.std': [f'p_hand_bcrl_standard_s{s}' for s in range(3)],
                      'hand.imitation_ppo.wide': [f'p_hand_bcrl_wide_s{s}' for s in range(3)]}.items()}}
    data['history'] = {   # transcribed; see `source`
        'pybullet_skills': {'source': 'docs/archive/pybullet/experiment_log.md',
                            'milestone1': 'one maze, one object: 3/3 fresh full-task runs; pure SAC+HER learned neither grasp nor carry',
                            'candidates': [{'name': 'Candidate 1', 'unseen': 78, 'cases': 100},
                                           {'name': 'Candidate 2', 'unseen': 69, 'cases': 100},
                                           {'name': 'Candidate 3', 'unseen': 80, 'cases': 100}],
                            'sac_ablation_dev': {'imitation grasp + imitation carry': 205, 'SAC grasp + SAC carry': 211,
                                                 'SAC grasp + imitation carry': 211, 'imitation grasp + SAC carry': 212, 'cases': 260},
                            'planner_course': 76, 'learned_course': 83, 'carry_drop_share': 24 / 31},
        'pybullet_hand': {'source': 'docs/archive/pybullet/hand_feasibility.md',
                          'recorded_routes': {'gripper': [40, 50], 'allegro': [20, 50]},
                          'unseen_routes': {'gripper': [18, 25], 'allegro': [14, 25]}},
        'mujoco_port': {'source': 'docs/experiment_log.md (2026-10-01)',
                        'planner_course': {'pybullet': 76, 'mujoco': 96}, 'planner_dev': {'pybullet': 190, 'mujoco': 250, 'cases': 260},
                        'holdout_course_physics': {'planner': 98, 'imitation': 92, 'imitation_sac': 92},
                        'holdout_realistic_physics': {'planner': 99, 'imitation': 77, 'imitation_sac': 77}},
        'hand_course_physics': {'source': 'docs/experiment_log.md (2026-10-02)',
                                'scripted_lift_mu025': '0/5 objects', 'ppo_lift_mu025': 'foam brick and banana only (20-33% over five objects)',
                                'frozen_grasp_carries': [1, 10], 'hand_squeeze_N': [15, 40], 'gripper_squeeze_N': 220},
        'hand_realistic': {'source': 'docs/experiment_log.md (2026-10-02)', 'ppo_scratch': 'no 15 cm lift in 24M steps',
                           'imitation_only': 0.14, 'imitation_ppo': 0.97, 'teacher': 0.72,
                           'reward_exploit': 'hovered just under the 15 cm success height: holding there paid ~7.5 per decision vs ~1.5 for carrying'},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('window.RESULTS = ' + json.dumps(data, separators=(',', ':'), ensure_ascii=False) + ';\n')
    print(f'{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
