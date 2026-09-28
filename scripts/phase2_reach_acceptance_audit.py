#!/usr/bin/env python3
"""Read-only historical reach audit; never reclassifies recorded full-task outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ('phase2_system_bc_ik24_01', 'phase2_system_carryreg01')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(directory):
    artifacts = directory / 'artifacts'
    paths = sorted(artifacts.glob('validation_*_object_*.json'))
    records = [(path, json.loads(path.read_text())) for path in paths]
    missed = [(path, result) for path, result in records
              if not result['success'] and result.get('metrics', {}).get('complete_task_success')]
    details = []
    for path, result in missed:
        reach = result.get('stages', {}).get('reach')
        if not reach:
            continue
        trace_path = path.with_name(path.stem + '_trajectory.jsonl')
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        trace = [row for row in trace if row['stage'] == 'reach']
        start = 0
        for name, stage in result['stages'].items():
            if name == 'reach':
                break
            start += stage['source_action_calls']
        tail = [row['object_position'] for row in trace
                if row['source_action'] >= start + reach['source_action_calls'] - 60]
        error = np.array(reach['final_achieved_goal']) - reach['desired_goal']
        first_call = result['first_complete_reach_action']
        details.append({
            'case_id': result['case_id'], 'recorded_full_task_success': result['success'],
            'failed_stage': result.get('failed_stage'),
            'recorded_final_hold_executed': 'final_hold' in result['stages'],
            'native_complete_at_end': result['metrics']['complete_task_success'],
            'native_final_goal_distance_m': result['metrics']['dist_to_goal'],
            'final_error_xyz_m': error.tolist(),
            'bind_frame_vs_native_distance_error_m': abs(float(np.linalg.norm(error)) - result['metrics']['dist_to_goal']),
            'reach_policy_decisions': reach['policy_decisions'],
            'reach_source_action_calls': reach['source_action_calls'],
            'reach_start_source_action': start,
            'first_native_complete_source_action': first_call,
            'first_native_complete_physics_tick': result['first_complete_reach_physics_tick'],
            'first_native_complete_within_reach_decision': math.ceil((first_call - start) / 3),
            'action_rejections': reach['action_rejections'],
            'drop_events': result['metrics']['drop_events'],
            'wall_contact_calls': result['wall_contact_calls'],
            'bilateral_contact_on_all_saved_reach_points': all(row['bilateral_contact'] for row in trace),
            'saved_last_60_calls_object_extent_m': float(np.linalg.norm(np.ptp(tail, axis=0))) if tail else None,
            'source_records': {str(path.relative_to(ROOT)): digest(path),
                               str(trace_path.relative_to(ROOT)): digest(trace_path)},
        })
    successful_can = [{
        'case_id': result['case_id'],
        'reach_policy_decisions': result['stages']['reach']['policy_decisions'],
        'reach_final_goal_distance_m': result['stages']['reach']['goal_distance'],
    } for _, result in records if result['case']['object_idx'] == 2 and result['success']]
    return {'run': directory.name, 'recorded_cases': len(records),
            'recorded_full_task_successes': sum(bool(result['success']) for _, result in records),
            'failed_cases_with_native_complete_at_end': len(missed),
            'native_complete_reach_failures': details,
            'successful_can_reaches': successful_can}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', default=[ROOT / 'runs' / name for name in DEFAULT_RUNS])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path; historical audit artifacts are immutable')
    report = {'kind': 'read_only_reach_acceptance_audit', 'new_source_action_calls': 0,
              'new_physics_ticks': 0, 'historical_outcomes_rewritten': False,
              'interpretation': 'Native proximity and retention history were observed; success under an earlier handover plus final held pause remains unmeasured. Historical failures stay failures.',
              'trace_sampling_note': 'Saved every 5 native calls; each reach decision repeats 3 calls. Native first-success fields record every call, not only saved trace points.',
              'runs': [audit_run(path.resolve()) for path in args.runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'runs': [
        {key: run[key] for key in ('run', 'recorded_cases', 'recorded_full_task_successes',
                                 'failed_cases_with_native_complete_at_end')} for run in report['runs']]}))


if __name__ == '__main__':
    main()
