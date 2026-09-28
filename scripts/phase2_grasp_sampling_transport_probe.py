#!/usr/bin/env python3
"""TRAIN-only frozen SAC sampling versus deterministic grasp transport.

Each of thirty branches starts independently from the original native state.
Only native policy sampling changes; Torch state is isolated and every other
RNG, controller, proposal, action interface, and physical parameter is retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import random
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

BRANCHES = ('deterministic', 'stochastic_1', 'stochastic_2')
_GRASP = None


def serializable(value):
    return json.loads(json.dumps(value, default=plain, allow_nan=False))


class UnusedPolicy:
    def predict(self, *args, **kwargs):
        raise AssertionError('Transport is an explicit TRAIN teacher, not a learned policy')


def actor_digest(model):
    result = hashlib.sha256()
    for key, value in sorted(model.actor.state_dict().items()):
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def sampling_seed(case_id, branch):
    text = 'native-sac-grasp-sampling-v1:'+case_id+':'+branch
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2**31)


class IsolatedSACPolicy:
    def __init__(self, model, seed, stochastic, stop_path=None):
        import torch
        if str(model.device) != 'cpu' or model.use_sde:
            raise ValueError('This diagnostic requires the saved ordinary CPU SAC distribution')
        self.model, self.seed, self.stochastic = model, int(seed), bool(stochastic)
        self.stop_path = stop_path
        self.model.policy.set_training_mode(False)
        self.rng_state = torch.Generator(device='cpu').manual_seed(self.seed).get_state()
        self.decisions = []
        self.weights_before = actor_digest(model)

    def predict(self, observation, deterministic=True):
        import torch
        if self.stop_path is not None and self.stop_path.exists():
            raise RuntimeError('STOP: peer sampling isolation failed')
        before_torch = torch.get_rng_state().clone()
        before_numpy = pickle.dumps(np.random.get_state())
        before_python = random.getstate()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng_state)
            action, state = self.model.predict(observation, deterministic=not self.stochastic)
            self.rng_state = torch.get_rng_state().clone()
            with torch.no_grad():
                tensors, _ = self.model.policy.obs_to_tensor(observation)
                mean, log_std, _ = self.model.actor.get_action_dist_params(tensors)
        if (not torch.equal(before_torch, torch.get_rng_state())
                or before_numpy != pickle.dumps(np.random.get_state())
                or before_python != random.getstate()):
            raise RuntimeError('STOP: native SAC sampling changed unrelated RNG state')
        self.decisions.append(dict(action=np.asarray(action).copy(),
            saved_actor_pre_tanh_mean=mean.cpu().numpy().tolist(),
            saved_actor_log_std=log_std.cpu().numpy().tolist(),
            deterministic=not self.stochastic, global_rngs_preserved=True))
        return action, state


class GraspProbe:
    def __init__(self, wrapped, system):
        self.wrapped, self.system = wrapped, system
        self.action_space = wrapped.action_space
        self.first_contact = None
        self.pregrasp = contact_record(system.native, system.controller,
            system.native.observe(), system.native.evaluate(), system)

    def _observation(self):
        return self.wrapped._observation()

    def step(self, action):
        result = self.wrapped.step(action)
        observation, reward, terminated, truncated, info = result
        if self.first_contact is None and info['bilateral_contact']:
            self.first_contact = dict(source_action_call=self.system.native.step_count,
                measured=contact_record(self.system.native, self.system.controller,
                    self.system.native.observe(), info, self.system))
        return result


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
    native = system = probe = sampled_policy = None
    rows = []
    started = time.monotonic()
    with (out/(key+'.log')).open('w') as log, (out/(key+'_contacts.jsonl')).open('w') as trace:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            if (out/'stop_requested').exists():
                raise RuntimeError('STOP: peer sampling isolation failed before preparation')
            native = PickNavReachEnv(**case['env_kwargs'])
            if native.scene_manifest()['maze_hash'] != case['maze_hash']:
                raise ValueError('Frozen TRAIN layout mismatch')

            def on_step(current, observation, info):
                row = contact_record(native, current.controller, observation, info, current)
                rows.append(row)
                trace.write(json.dumps(row, default=plain, allow_nan=False)+'\n')

            sampled_policy = IsolatedSACPolicy(_GRASP, sampling_seed(case['case_id'], branch),
                                              stochastic=branch != 'deterministic', stop_path=out/'stop_requested')
            system = LearnedSystem(native,
                {'grasp': sampled_policy, 'carry': UnusedPolicy(), 'reach': UnusedPolicy()},
                grasp_horizon=600, on_step=on_step, grasp_proposal_mode='clearance-v2')
            system.calls_by_source['teacher'] = 0

            def bind_grasp(n, controller, horizon):
                nonlocal probe
                probe = GraspProbe(GeneralGraspEnv.bind(n, controller,
                    horizon=horizon, goal_lift_m=.24), system)
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
            if 'STOP:' in str(error):
                (out/'stop_requested').write_text(str(error))
            result.update(error=repr(error), failed_stage=system.stage if system else 'environment_setup')
            if system is not None and system._active_stage_start is not None:
                system._finish(system.stage, system._active_stage_start, False, error=repr(error))
        finally:
            if probe is not None:
                result['first_contact'] = probe.first_contact
                result['pregrasp_measured'] = probe.pregrasp
            if sampled_policy is not None:
                result['sampling'] = dict(torch_seed=sampled_policy.seed,
                    deterministic=not sampled_policy.stochastic, decision_count=len(sampled_policy.decisions),
                    rng_scope='isolated CPU Torch state; caller NumPy/Python/Torch states checked after every prediction',
                    actor_sha256_before=sampled_policy.weights_before, actor_sha256_after=actor_digest(_GRASP),
                    no_added_action_noise=True, saved_log_std_unchanged=True)
                if result['sampling']['actor_sha256_after'] != result['sampling']['actor_sha256_before']:
                    result.update(success=False, error='STOP: frozen actor parameters changed')
                    (out/'stop_requested').write_text(result['error'])
                with (out/(key+'_grasp_actions.pkl')).open('wb') as stream:
                    pickle.dump(sampled_policy.decisions, stream, protocol=pickle.HIGHEST_PROTOCOL)
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
        rows = [r for r in results if r['case_id'] == case['case_id']]
        if len(rows) != len(BRANCHES):
            continue
        poses = [r.get('pregrasp_measured') for r in rows]
        match = all(p is not None for p in poses)
        differences = {}
        if match:
            for field in ('qpos', 'qvel', 'object_world', 'object_quaternion', 'ee_world', 'ee_quaternion'):
                differences[field] = float(max(np.max(np.abs(np.asarray(p[field])-poses[0][field])) for p in poses))
            match = max(differences.values()) < 1e-9
        handovers = [np.asarray(r['after_carry_posture']['object_in_hand']) for r in rows
                    if r.get('stages', {}).get('planned_carry_posture', {}).get('success')]
        spread = max((float(np.linalg.norm(a-b)) for a in handovers for b in handovers), default=None)
        comparisons.append(dict(case_id=case['case_id'], identical_pregrasp_states=bool(match),
            pregrasp_max_absolute_differences=differences,
            identical_proposals=len({json.dumps(r.get('selected_grasp'), sort_keys=True) for r in rows}) == 1,
            prepared_handover_payload_position_max_pairwise_difference_m=spread,
            outcomes={r['branch']: r['success'] for r in rows}))
    return dict(partition='train', scope='native_stochastic_grasp_preparation_and_full_teacher_transport',
        full_task_evaluation=False, completed_branches=len(results), planned_branches=len(BRANCHES)*len(cases),
        all_failures_retained=True, comparisons=comparisons,
        branch_successes={b: sum(r['success'] for r in results if r['branch'] == b) for b in BRANCHES},
        branch_prepared_handovers={b: sum(r.get('stages', {}).get('planned_carry_posture', {}).get('success', False)
                                        for r in results if r['branch'] == b) for b in BRANCHES},
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
    plan = dict(protocol='train-native-sac-sampling-full-transport-v1',
        cases=cases, branches=BRANCHES, checkpoint=checkpoint, source_hashes=sources,
        bank_sha256=bank['manifest_sha256'], selection='first two TRAIN layouts cross all five frozen objects',
        original_start=True, independent_branch_replay=True, inter_skill_resets=False,
        snapshot_restoration=False, native_saved_sac_sampling=True, added_action_noise=False,
        sampling_seeds={case['case_id']:{branch:sampling_seed(case['case_id'], branch) for branch in BRANCHES} for case in cases},
        rng_isolation='fork CPU Torch RNG per prediction; independent recorded persistent state per branch; all other RNGs preserved',
        grasp_repeat=3,
        grasp_lift_m=.24, grasp_proposal_mode='clearance-v2',
        teacher_carry_delta_scale=.075, teacher_carry_repeat=5, teacher_carry_horizon=1600,
        final_hold_native_calls=50, full_task_evaluation=False, physics_unchanged=True,
        production_policy_changed=False, observation_contract_changed=False, workers=args.workers)
    plan['plan_sha256'] = plan_digest(plan)
    atomic_json(out/'diagnostic_plan.json', plan)
    logger = RunLogger(args.run_id, kind='diagnostic', budget=30, skill='grasp',
        name='TRAIN native SAC sampling and full transport',
        summary='Thirty independent original starts: deterministic and two native stochastic grasps; unchanged teacher transport.')
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
            logger.metric(len(results), completed_branches=len(results), planned_branches=30,
                          branch_successes=summary['branch_successes'],
                          physics_ticks=summary['total_physics_ticks_including_initialization'])
    logger.artifact('artifacts/summary.json', 'Native stochastic preparation, measured seating, transport outcomes and costs')
    if (out/'stop_requested').exists():
        logger.finish('failed', (out/'stop_requested').read_text())
    else:
        logger.finish(summary='TRAIN native SAC sampling diagnostic complete; no production or training change.')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
