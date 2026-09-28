#!/usr/bin/env python3
"""Fixed TRAIN-only physical check of counted handover restoration; no learning."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import torch
from pnr.learned_carry import LearnedMultiCaseCarry, PREPARATION_SETTINGS, CACHE_RESTORE_SETTINGS
from pnr.policies import load_policy_checkpoint
from pnr.scenes import load_scene_bank, case_specs
from pnr.telemetry import atomic_json


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=ROOT/'configs/phase2_scene_bank_expanded.json')
    parser.add_argument('--grasp', type=Path, default=ROOT/'runs/phase2_sac_grasp_clearance01/candidate/grasp.zip')
    parser.add_argument('--carry', type=Path, default=ROOT/'runs/phase2_carry_expanded_init01/checkpoints/bc_initialized.zip')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory; no overwritten outcomes')
    torch.set_num_threads(1)
    out = args.output/'artifacts'; out.mkdir(parents=True)
    source = args.output/'sources'; source.mkdir()
    hashes = {}
    for path in list((ROOT/'pnr').glob('*.py')) + [Path(__file__).resolve()]:
        relative = path.relative_to(ROOT); dest = source/relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        hashes[str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    atomic_json(args.output/'source_manifest.json', hashes)
    checkpoints = {}
    for name, path in [('grasp', args.grasp), ('carry', args.carry)]:
        dest = out/(name+'.zip'); shutil.copy2(path, dest)
        checkpoints[name] = {'source': str(path.resolve()), 'frozen_path': str(dest.resolve()),
                             'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    bank = load_scene_bank(args.bank)
    cases = case_specs(bank, 'train', object_indices=[14, 2, 5])[:3]
    config = {'protocol': 'learned-carry-cache-physical-preflight-v1',
        'cases': cases, 'bank_sha256': bank['manifest_sha256'], 'checkpoints': checkpoints,
        'cycles_per_case': 3, 'maximum_carry_decisions_per_cycle': 20,
        'preparation_settings': PREPARATION_SETTINGS, 'restore_settings': CACHE_RESTORE_SETTINGS,
        'selection': 'first TRAIN layout, prespecified foam/can/apple',
        'source_hashes': hashes, 'training_updates': 0, 'full_task_evaluation': False}
    atomic_json(args.output/'config.json', config)
    policy = load_policy_checkpoint(checkpoints['carry']['frozen_path'], device='cpu')
    results = []; start = time.monotonic()
    for case in cases:
        env = None
        record = {'case_id': case['case_id'], 'cycles': [], 'completed': False}
        try:
            env = LearnedMultiCaseCarry([case], checkpoints['grasp']['frozen_path'],
                checkpoints['grasp']['sha256'])
            record['initial_cache'] = env.prime_cache()
            if not record['initial_cache']['ready_for_training']:
                record['error'] = 'Preparation or initial restore validation failed'
            else:
                with (out/(case['case_id']+'_trajectory.jsonl')).open('w') as trace:
                    for cycle in range(3):
                        row = {'cycle': cycle, 'decisions': 0, 'restored': False}; record['cycles'].append(row)
                        observation, info = env.reset()
                        row['restored'] = True; row['reset_info'] = info
                        row['initial_observation'] = observation
                        for step in range(20):
                            action, _ = policy.predict(observation, deterministic=True)
                            observation, reward, terminated, truncated, info = env.step(action)
                            row['decisions'] += 1
                            trace.write(json.dumps({'cycle': cycle, 'decision': step,
                                'action': action, 'observation': observation, 'reward': reward,
                                'terminated': terminated, 'truncated': truncated, 'info': info}, default=plain)+'\n')
                            if terminated or truncated:
                                break
                        row['last_info'] = info
                record['completed'] = True
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if env is not None:
                record['cache_report'] = env.cache_report()
                record['training_state'] = env.get_training_state()
                env.close()
            results.append(record)
            payload = {'planned_cases': len(cases), 'completed_cases': len(results),
                'successful_cases': sum(r['completed'] for r in results),
                'successful_online_restores': sum(c['restored'] for r in results for c in r['cycles']),
                'training_updates': 0, 'wall_seconds': time.monotonic()-start, 'results': results}
            reports = [r['cache_report'] for r in results if 'cache_report' in r]
            payload['costs'] = {
                key: sum(r[key] for r in reports) for key in ('setup_source_action_calls',
                'setup_physics_ticks', 'initialization_physics_ticks',
                'incremental_transport_source_action_calls', 'incremental_transport_physics_ticks')}
            payload['all_source_action_calls'] = (payload['costs']['setup_source_action_calls'] +
                payload['costs']['incremental_transport_source_action_calls'])
            payload['all_physics_ticks'] = sum(payload['costs'][k] for k in (
                'setup_physics_ticks', 'initialization_physics_ticks', 'incremental_transport_physics_ticks'))
            atomic_json(out/'result.json', json.loads(json.dumps(payload, default=plain, allow_nan=False)))
            print(json.dumps({'case': case['case_id'], 'completed': record['completed'],
                              'restores': sum(c['restored'] for c in record['cycles']),
                              'error': record.get('error')}), flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != 'results'}), flush=True)


if __name__ == '__main__': main()
