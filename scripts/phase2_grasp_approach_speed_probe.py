#!/usr/bin/env python3
"""TRAIN-only final-approach teacher speed grid and complete transport.

Three shared teacher caps vary only within the final 3 cm before bilateral
contact. All original starts, closure gates, physics and full routes are fixed.
This diagnostic does not qualify or collect a new training dataset.
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
from pnr.general_manipulation import GeneralGraspEnv, ContactGraspTeacher
from pnr.planning import maze_astar
from pnr.scenes import load_scene_bank
from pnr.system import LearnedSystem, StageFailure
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import plain
from scripts.phase2_evaluate_system import freeze_sources, plan_digest, select_cases, sha256_file
from scripts.phase2_apple_retention_branches import contact_record

CAPS = (.65, .40, .20)


def serializable(value):
    return json.loads(json.dumps(value, default=plain, allow_nan=False))


class UnusedPolicy:
    def predict(self, *args, **kwargs):
        raise AssertionError('Transport is an explicit TRAIN teacher, not a learned policy')


class NearContactTeacherProbe:
    """An explicitly identified diagnostic teacher; no learned actor override."""
    def __init__(self, wrapped, system, cap):
        self.wrapped, self.system, self.cap = wrapped, system, float(cap)
        self.action_space = wrapped.action_space
        self.ordinary = ContactGraspTeacher(wrapped, max_action=.65,
                                            close_distance=.018, approach_control='ray-tilted')
        self.near = ContactGraspTeacher(wrapped, max_action=self.cap,
                                        close_distance=.018, approach_control='ray-tilted')
        self.first_contact = None
        self.decisions = []
        self.pending = None

    def _observation(self):
        return self.wrapped._observation()

    def predict(self, observation, deterministic=True):
        native = self.system.native
        ee, _ = self.system.controller.planner.ee_pose()
        distance = float(np.linalg.norm(self.wrapped.measured_grasp_target()-ee))
        near_contact = bool(not native.contacts()['bilateral_contact'] and distance <= .03)
        teacher = self.near if near_contact else self.ordinary
        action, phase = teacher.action(observation)
        self.pending = dict(action=action.copy(), teacher_phase=phase,
            near_contact_cap_applied=near_contact, requested_cap=self.cap if near_contact else .65,
            target_error_before_m=distance, ee_before_m=ee.copy(),
            source_action_call_before=native.step_count)
        return action, None

    def step(self, action):
        if self.pending is None or not np.array_equal(action, self.pending['action']):
            raise AssertionError('Teacher prediction and executed action differ')
        self.system.control_source = 'teacher'
        result = self.wrapped.step(action)
        observation, reward, terminated, truncated, info = result
        native = self.system.native
        ee, _ = self.system.controller.planner.ee_pose()
        row = self.pending
        calls = native.step_count-row['source_action_call_before']
        row.update(ee_after_m=ee.copy(), source_action_calls=calls,
            actual_mean_ee_speed_m_per_s=float(np.linalg.norm(ee-row['ee_before_m'])/(calls*native.substeps*native.physics_timestep)),
            target_error_after_m=float(np.linalg.norm(self.wrapped.measured_grasp_target()-ee)),
            bilateral_contact=bool(info['bilateral_contact']), action_rejected=bool(info['action_rejected']))
        self.decisions.append(row)
        self.pending = None
        if self.first_contact is None and info['bilateral_contact']:
            self.first_contact = dict(source_action_call=native.step_count,
                measured=contact_record(native, self.system.controller, native.observe(), info, self.system),
                target_error_m=row['target_error_after_m'])
        return result


def initialize_worker():
    import torch
    torch.set_num_threads(1)


def worker(task):
    case, cap, output = task
    out = Path(output)
    branch = f'cap_{cap:.2f}'
    key = case['case_id']+'_'+branch
    result = dict(diagnostic_id=key, case_id=case['case_id'], case=case, branch=branch, near_contact_cap=cap,
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
                {'grasp': UnusedPolicy(), 'carry': UnusedPolicy(), 'reach': UnusedPolicy()},
                grasp_horizon=600, on_step=on_step, grasp_proposal_mode='clearance-v2')
            system.calls_by_source['teacher'] = 0

            def bind_grasp(n, controller, horizon):
                nonlocal probe
                probe = NearContactTeacherProbe(GeneralGraspEnv.bind(n, controller,
                    horizon=horizon, goal_lift_m=.24), system, cap)
                system.policies['grasp'] = probe
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
                result['grasp_teacher_decisions'] = len(probe.decisions)
                near = [d for d in probe.decisions if d['near_contact_cap_applied']]
                result['near_contact_measurements'] = dict(decisions=len(near),
                    mean_actual_ee_speed_m_per_s=float(np.mean([d['actual_mean_ee_speed_m_per_s'] for d in near])) if near else None,
                    max_actual_ee_speed_m_per_s=float(max(d['actual_mean_ee_speed_m_per_s'] for d in near)) if near else None,
                    mean_action_xyz_norm=float(np.mean([np.linalg.norm(d['action'][:3]) for d in near])) if near else None)
                with (out/(key+'_teacher_decisions.pkl')).open('wb') as stream:
                    pickle.dump(probe.decisions, stream, protocol=pickle.HIGHEST_PROTOCOL)
                atomic_json(out/(key+'_teacher_decisions.json'), serializable(probe.decisions))
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
    comparisons = []
    for case in cases:
        records = [r for r in results if r['case_id'] == case['case_id']]
        if len(records) != len(CAPS):
            continue
        proposals = [json.dumps(r.get('selected_grasp'), sort_keys=True) for r in records]
        comparisons.append(dict(case_id=case['case_id'], identical_proposals=all(r.get('selected_grasp') is not None for r in records) and len(set(proposals)) == 1,
            outcomes={r['branch']: r['success'] for r in records}))
    by_cap = {}
    for cap in CAPS:
        rows = [r for r in results if r['near_contact_cap'] == cap]
        successful = [r for r in rows if r['success']]
        speeds = [r['near_contact_measurements']['mean_actual_ee_speed_m_per_s'] for r in rows
                  if r.get('near_contact_measurements', {}).get('mean_actual_ee_speed_m_per_s') is not None]
        by_cap[f'{cap:.2f}'] = dict(completed=len(rows), successes=len(successful),
            success_fraction_of_declared=sum(r['success'] for r in rows)/len(cases),
            total_physics_ticks=sum(r.get('total_physics_ticks_including_initialization', 0) for r in rows),
            mean_successful_ticks=float(np.mean([r['total_physics_ticks_including_initialization'] for r in successful])) if successful else None,
            mean_near_contact_ee_speed_m_per_s=float(np.mean(speeds)) if speeds else None)
    return dict(partition='train', scope='teacher_approach_speed_and_complete_transport',
        full_task_evaluation=False, completed_branches=len(results), planned_branches=len(CAPS)*len(cases),
        all_failures_retained=True, common_setting_results=by_cap, proposal_comparisons=comparisons,
        cost_accounting_complete=all(r['cost_accounting_complete'] for r in results),
        **{k: sum(r.get(k, 0) for r in results) for k in ('source_action_calls', 'physics_ticks',
            'initialization_physics_ticks', 'total_physics_ticks_including_initialization')},
        results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    cases = select_cases(bank, 'train', [14, 5, 2, 7, 10], layout_limit=4)
    sources = freeze_sources(out/'source')
    for path in [Path(__file__).resolve(), ROOT/'scripts/phase2_apple_retention_branches.py']:
        destination = out/'source'/'scripts'/path.name
        destination.write_bytes(path.read_bytes())
        sources['scripts/'+path.name] = sha256_file(path)
    plan = dict(protocol='train-teacher-near-contact-speed-full-transport-v1',
        cases=cases, caps=CAPS, source_hashes=sources,
        bank_sha256=bank['manifest_sha256'], selection='first four TRAIN layouts cross all five frozen objects',
        original_start=True, independent_branch_replay=True, inter_skill_resets=False,
        snapshot_restoration=False, near_contact_radius_m=.03, close_distance_m=.018,
        ordinary_teacher_cap=.65, teacher_approach_control='ray-tilted', grasp_repeat=3,
        grasp_lift_m=.24, grasp_proposal_mode='clearance-v2',
        teacher_carry_delta_scale=.075, teacher_carry_repeat=5, teacher_carry_horizon=1600,
        final_hold_native_calls=50, full_task_evaluation=False, physics_unchanged=True,
        production_policy_changed=False, observation_contract_changed=False, workers=args.workers)
    plan['plan_sha256'] = plan_digest(plan)
    atomic_json(out/'diagnostic_plan.json', plan)
    logger = RunLogger(args.run_id, kind='diagnostic', budget=60, skill='grasp',
        name='TRAIN teacher approach speed and full transport',
        summary='Sixty original-start branches; three common near-contact caps and measured downstream retention.')
    logger.artifact('artifacts/diagnostic_plan.json', 'Frozen TRAIN-only transport diagnostic')
    results = []
    tasks = [(case, cap, str(out)) for case in cases for cap in CAPS]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            case, cap, _ = futures[future]
            branch = f'cap_{cap:.2f}'
            try:
                result = future.result()
            except Exception as error:
                result = dict(case_id=case['case_id'], branch=branch, near_contact_cap=cap, success=False,
                    error=repr(error), cost_accounting_complete=False, failed_stage='worker_exception')
                atomic_json(out/(case['case_id']+'_'+branch+'_worker_exception.json'), result)
            results.append(result)
            summary = summarize(results, cases)
            atomic_json(out/'summary.json', summary)
            logger.metric(len(results), completed_branches=len(results), planned_branches=60,
                          common_setting_results=summary['common_setting_results'],
                          physics_ticks=summary['total_physics_ticks_including_initialization'])
    logger.artifact('artifacts/summary.json', 'Uniform teacher settings: transport, speeds, contact geometry, and costs')
    logger.finish(summary='TRAIN speed grid complete; no data qualification, recollection, or production control changes.')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
