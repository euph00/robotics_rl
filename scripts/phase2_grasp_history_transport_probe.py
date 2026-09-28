#!/usr/bin/env python3
"""TRAIN-only one-time contact settling versus full-route transport.

Each branch independently replays the original start. The intervention uses
ten ordinary grasp actions, never a snapshot refresh, and is diagnostic only.
No observation contract, production policy, dataset, or physics is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv, route_action
from pnr.general_manipulation import GeneralGraspEnv
from pnr.planning import maze_astar
from pnr.policies import load_policy_checkpoint
from pnr.scenes import load_scene_bank
from pnr.system import LearnedSystem, StageFailure
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import freeze_checkpoints, plain
from scripts.phase2_evaluate_system import freeze_sources, plan_digest, select_cases, sha256_file
from scripts.phase2_apple_retention_branches import contact_record

BRANCHES = ('original', 'contact_settle30')
_GRASP = None


def serializable(value):
    return json.loads(json.dumps(value, default=plain, allow_nan=False))


class UnusedPolicy:
    def predict(self, *args, **kwargs):
        raise AssertionError('Transport is an explicit TRAIN teacher, not a learned policy')


class ContactHistoryProbe:
    """Record first observed contact; optionally execute diagnostic teacher actions."""
    def __init__(self, wrapped, system, branch):
        self.wrapped, self.system, self.branch = wrapped, system, branch
        self.action_space = wrapped.action_space
        self.first_contact = None
        self.prefix_actions = []
        self.settling_transitions = []

    def _observation(self):
        return self.wrapped._observation()

    def step(self, action):
        if self.first_contact is None:
            self.prefix_actions.append(np.asarray(action).copy())
        result = self.wrapped.step(action)
        observation, reward, terminated, truncated, info = result
        if self.first_contact is not None or terminated or truncated or not info['bilateral_contact']:
            return result
        native, system = self.system.native, self.system
        self.first_contact = {
            'source_action_call': native.step_count,
            'observation': observation,
            'measured': contact_record(native, system.controller, native.observe(), info, system),
            'prefix_actions': self.prefix_actions,
            'event_definition': 'first completed policy decision reporting bilateral contact',
        }
        if self.branch == 'original':
            return result
        previous = system.stage, system.control_source
        system.stage, system.control_source = 'contact_settle30', 'teacher'
        start = native.step_count
        try:
            for _ in range(10):
                before_observation = self.wrapped._observation()
                before_calls = native.step_count
                action = np.array([0., 0., 0., -1.], dtype=np.float32)
                observation, reward, terminated, truncated, info = self.wrapped.step(action)
                self.settling_transitions.append({
                    'observation': before_observation, 'action': action,
                    'next_observation': observation, 'reward': reward,
                    'terminated': terminated, 'truncated': truncated, 'info': dict(info),
                    'native_calls_since_first_contact_before': before_calls-start,
                    'native_calls_since_first_contact_after': native.step_count-start,
                    'source_action_calls': native.step_count-before_calls,
                })
                if terminated or truncated or native.drop_events:
                    break
        finally:
            system.stage, system.control_source = previous
        if not terminated and not truncated and native.step_count-start != 30:
            raise AssertionError('Ten ordinary grasp decisions must execute thirty native calls')
        return observation, reward, terminated, truncated, info


def initialize_worker(record):
    global _GRASP
    import torch
    torch.set_num_threads(1)
    if sha256_file(record['frozen_path']) != record['sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    _GRASP = load_policy_checkpoint(record['frozen_path'], device='cpu')
    if _GRASP.observation_space['observation'].shape != (54,):
        raise ValueError('This diagnostic preserves the existing 54-state policy interface')


def worker(task):
    case, branch, output = task
    out = Path(output)
    key = case['case_id']+'_'+branch
    result = dict(diagnostic_id=key, case_id=case['case_id'], case=case, branch=branch,
                  success=False, retained_transport=False, original_start=True,
                  full_task_evaluation=False, partition='train', cost_accounting_complete=False)
    native = system = probe = None
    rows = []
    started = time.monotonic()
    with (out/(key+'.log')).open('w') as log, (out/(key+'_contacts.jsonl')).open('w') as trace:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            native = PickNavReachEnv(**case['env_kwargs'])
            if native.scene_manifest()['maze_hash'] != case['maze_hash']:
                raise ValueError('Frozen TRAIN layout mismatch')

            def on_step(current, observation, info):
                row = contact_record(native, current.controller, observation, info, current)
                rows.append(row)
                trace.write(json.dumps(row, default=plain, allow_nan=False)+'\n')

            system = LearnedSystem(native,
                {'grasp': _GRASP, 'carry': UnusedPolicy(), 'reach': UnusedPolicy()},
                grasp_horizon=600, on_step=on_step, grasp_proposal_mode='clearance-v2')
            system.calls_by_source['teacher'] = 0

            def bind_grasp(n, controller, horizon):
                nonlocal probe
                probe = ContactHistoryProbe(GeneralGraspEnv.bind(n, controller,
                    horizon=horizon, goal_lift_m=.24), system, branch)
                return probe

            system.skill_factories['grasp'] = bind_grasp
            with system._observe_native_steps():
                start = system._begin('planned_approach', 'planned')
                grasp = system.controller.prepare_grasp()
                system._finish('planned_approach', start, True, grasp=grasp.as_dict())
                result['selected_grasp'] = grasp.as_dict()
                system._skill('grasp')
                system._held_pause('pick_hold', 50)
                result['after_pick_hold'] = rows[-1]
                start = system._begin('planned_carry_posture', 'planned')
                system.controller.prepare_carry()
                retained = bool(native.contacts()['bilateral_contact'] and not native.drop_events)
                system._finish('planned_carry_posture', start, retained)
                result['after_carry_posture'] = rows[-1]
                if not retained:
                    raise StageFailure('planned_carry_posture: lost grasp')
                start = system._begin('teacher_carry', 'teacher')
                system.navigation_start_action = native.step_count
                carry = GeneralCarryEnv.bind(native, system.controller, horizon=1600, delta_scale=.075)
                observation = carry._observation()
                fixed_goal = observation['desired_goal'].copy()
                path = maze_astar(native.cube_positions, system.controller.base_position(),
                    native.goal_pos[:2], table_bounds=native.p.getAABB(native.table_id))
                reached = False
                last_info = {}
                for decision in range(1600):
                    action = route_action(path, observation['achieved_goal']*carry.goal_scale,
                                         native.goal_pos[:2], carry.delta_scale)
                    observation, _, terminated, truncated, last_info = carry.step(action)
                    if not np.array_equal(observation['desired_goal'], fixed_goal):
                        raise AssertionError('Carry goal changed')
                    reached = bool(last_info['is_success'] and last_info['bilateral_contact']
                                   and not terminated and not native.drop_events)
                    if terminated or truncated or native.drop_events:
                        break
                system._finish('teacher_carry', start, reached, decisions=decision+1,
                    goal_distance=last_info.get('goal_distance'),
                    reason=last_info.get('failure_reason'), original_goal=fixed_goal.tolist())
                result['teacher_carry_decisions'] = decision+1
                if not reached:
                    raise StageFailure('teacher_carry: '+str(last_info.get('failure_reason') or 'route incomplete'))
                system._held_pause('transport_final_hold', 50)
                result['retained_transport'] = True
                result['success'] = bool(system.wall_contact_calls == 0 and system.robot_table_contact_calls == 0)
                result['after_transport_hold'] = rows[-1]
        except Exception as error:
            result.update(error=repr(error), failed_stage=system.stage if system else 'environment_setup')
            if system is not None and system._active_stage_start is not None:
                system._finish(system.stage, system._active_stage_start, False, error=repr(error))
        finally:
            if probe is not None:
                result['first_contact'] = probe.first_contact
                result['settling_teacher_decisions'] = len(probe.settling_transitions)
                result['settling_source_action_calls'] = sum(t['source_action_calls'] for t in probe.settling_transitions)
                with (out/(key+'_settling_transitions.pkl')).open('wb') as stream:
                    pickle.dump(probe.settling_transitions, stream, protocol=pickle.HIGHEST_PROTOCOL)
            if system is not None:
                result.update(stages=system.stage_results, policy_decisions=system.policy_decisions,
                    source_calls_by_controller=system.calls_by_source,
                    wall_contact_calls=system.wall_contact_calls,
                    robot_table_contact_calls=system.robot_table_contact_calls,
                    proposal_diagnostics=system.controller.stage_results.get('grasp_generation'))
            if native is not None:
                result.update(source_action_calls=native.step_count, physics_ticks=native.physics_steps,
                    initialization_physics_ticks=native.settle_steps,
                    total_physics_ticks_including_initialization=native.physics_steps+native.settle_steps,
                    metrics=native.evaluate(), cost_accounting_complete=bool(system is not None
                        and len(rows) == native.step_count
                        and sum(system.calls_by_source.values()) == native.step_count))
                native.close()
            result['wall_seconds'] = time.monotonic()-started
            # Backup precedes JSON formatting; every completed branch persists now.
            with (out/(key+'_result.pkl')).open('wb') as stream:
                pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
            result = serializable(result)
            atomic_json(out/(key+'.json'), result)
    return result


def summarize(results, cases):
    pairs = []
    for case in cases:
        records = {r['branch']: r for r in results if r['case_id'] == case['case_id']}
        if len(records) != 2:
            continue
        first = [records[b].get('first_contact') for b in BRANCHES]
        matched = all(x is not None for x in first)
        differences = {}
        if matched:
            for field in ('qpos', 'qvel', 'object_world', 'object_quaternion', 'object_in_hand'):
                differences[field] = float(np.max(np.abs(np.asarray(first[0]['measured'][field])
                                                        -np.asarray(first[1]['measured'][field]))))
            a, b = [np.asarray(x['prefix_actions']) for x in first]
            action_match = a.shape == b.shape and np.array_equal(a, b)
            matched = bool(action_match and first[0]['source_action_call'] == first[1]['source_action_call']
                           and max(differences.values()) < 1e-9)
        pairs.append({'case_id': case['case_id'], 'matched_first_contact': matched,
                      'max_absolute_state_differences': differences,
                      'success': {b: records[b]['success'] for b in BRANCHES}})
    return dict(partition='train', scope='original_start_through_full_teacher_transport_and_50_call_hold',
        full_task_evaluation=False, completed_branches=len(results), planned_branches=2*len(cases),
        all_failures_retained=True, pairs=pairs,
        branch_successes={b: sum(r['success'] for r in results if r['branch'] == b) for b in BRANCHES},
        cost_accounting_complete=all(r['cost_accounting_complete'] for r in results),
        **{k: sum(r.get(k, 0) for r in results) for k in ('source_action_calls', 'physics_ticks',
            'initialization_physics_ticks', 'total_physics_ticks_including_initialization')},
        results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use one to eight CPU workers')
    directory = ROOT/'runs'/args.run_id
    if directory.exists():
        parser.error('Use a new run ID; preserve earlier attempts')
    out = directory/'artifacts'
    out.mkdir(parents=True)
    bank = load_scene_bank(ROOT/'configs/phase2_scene_bank.json')
    cases = select_cases(bank, 'train', [14, 5, 2, 7, 10], layout_limit=2)
    checkpoint = freeze_checkpoints({'grasp': args.grasp}, out/'checkpoints')['grasp']
    sources = freeze_sources(out/'source')
    for path in [Path(__file__).resolve(), ROOT/'scripts/phase2_apple_retention_branches.py']:
        destination = out/'source'/'scripts'/path.name
        destination.write_bytes(path.read_bytes())
        sources['scripts/'+path.name] = sha256_file(path)
    plan = dict(protocol='train-first-observed-contact-settling-full-transport-v1',
        cases=cases, branches=BRANCHES, checkpoint=checkpoint, source_hashes=sources,
        bank_sha256=bank['manifest_sha256'], selection='first two TRAIN layouts cross all five frozen objects',
        original_start=True, independent_branch_replay=True, inter_skill_resets=False,
        snapshot_restoration=False, contact_settling_native_calls=30,
        contact_settling_action=[0., 0., 0., -1.], grasp_repeat=3,
        grasp_lift_m=.24, grasp_proposal_mode='clearance-v2',
        teacher_carry_delta_scale=.075, teacher_carry_repeat=5, teacher_carry_horizon=1600,
        final_hold_native_calls=50, full_task_evaluation=False, physics_unchanged=True,
        production_policy_changed=False, observation_contract_changed=False, workers=args.workers)
    plan['plan_sha256'] = plan_digest(plan)
    atomic_json(out/'diagnostic_plan.json', plan)
    logger = RunLogger(args.run_id, kind='diagnostic', budget=20, skill='grasp',
        name='TRAIN first-contact settling and full transport',
        summary='Twenty independent original-start branches; matched contact audit, teacher carry, unchanged physics.')
    logger.artifact('artifacts/diagnostic_plan.json', 'Frozen TRAIN-only transport diagnostic')
    results = []
    tasks = [(case, branch, str(out)) for case in cases for branch in BRANCHES]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(checkpoint,)) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            case, branch, _ = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(case_id=case['case_id'], branch=branch, success=False,
                    error=repr(error), cost_accounting_complete=False, failed_stage='worker_exception')
                atomic_json(out/(case['case_id']+'_'+branch+'_worker_exception.json'), result)
            results.append(result)
            summary = summarize(results, cases)
            atomic_json(out/'summary.json', summary)
            logger.metric(len(results), completed_branches=len(results), planned_branches=20,
                          branch_successes=summary['branch_successes'],
                          physics_ticks=summary['total_physics_ticks_including_initialization'])
    logger.artifact('artifacts/summary.json', 'Paired transport outcomes, measured contacts, and all costs')
    logger.finish(summary='TRAIN transport diagnostic complete; no production pause or observation change.')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
