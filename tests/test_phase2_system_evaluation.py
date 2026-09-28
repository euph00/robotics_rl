"""Protocol checks only: no trained-policy evaluations or held-out physics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from pnr.scenes import build_scene_bank
from pnr.system import LearnedSystem
from scripts.phase2_evaluate_system import (
    DEFAULT_OBJECTS, PLAN_VERSION, deadline_metrics, failure_result, plan_digest,
    select_cases, sha256_file, summarize_results, validate_policy_interfaces, verify_frozen_plan,
)


@pytest.fixture(scope='module')
def bank():
    return build_scene_bank(counts={'train': 2, 'validation': 2, 'milestone_test': 2, 'final_reserve': 2})


def frozen_plan(bank, tmp_path, partition='milestone_test'):
    records = {}
    for name in ('grasp', 'carry', 'reach'):
        path = tmp_path / (name + '.zip')
        path.write_bytes(('CHECKPOINT HASH TEST ONLY ' + name).encode())
        records[name] = {'frozen_path': str(path), 'sha256': sha256_file(path)}
    plan = {'version': PLAN_VERSION, 'status': 'frozen', 'partition': partition,
            'object_indices': DEFAULT_OBJECTS, 'scene_bank_sha256': bank['manifest_sha256'],
            'cases': select_cases(bank, partition, DEFAULT_OBJECTS, allow_heldout=True),
            'checkpoints': records, 'original_start': True, 'inter_skill_resets': False,
            'teacher_at_inference': False, 'original_gripper': True, 'retain_all_failures': True,
            'execution': {'delta_scale': .075, 'planned_speed_scale': 1., 'deadline_physics_ticks': 15000,
                          'grasp_horizon': 600, 'carry_horizon': 1200, 'reach_horizon': 150}}
    plan['plan_sha256'] = plan_digest(plan)
    return plan


def test_heldout_requires_full_partition_and_explicit_access(bank, tmp_path):
    with pytest.raises(ValueError, match='allow-heldout'):
        select_cases(bank, 'milestone_test', DEFAULT_OBJECTS)
    with pytest.raises(ValueError, match='complete frozen partition'):
        select_cases(bank, 'milestone_test', DEFAULT_OBJECTS, allow_heldout=True, case_limit=1)
    with pytest.raises(ValueError, match='five predeclared'):
        select_cases(bank, 'final_reserve', [14], allow_heldout=True)
    plan = frozen_plan(bank, tmp_path)
    with pytest.raises(ValueError, match='allow-heldout'):
        verify_frozen_plan(plan, bank)
    assert verify_frozen_plan(plan, bank, allow_heldout=True) is plan
    changed = deepcopy(plan)
    changed['cases'].pop()
    changed['plan_sha256'] = plan_digest(changed)
    with pytest.raises(ValueError, match='complete frozen partition'):
        verify_frozen_plan(changed, bank, allow_heldout=True)


def test_frozen_plan_checks_actual_checkpoint_bytes_and_execution_digest(bank, tmp_path):
    plan = frozen_plan(bank, tmp_path)
    changed = deepcopy(plan)
    changed['execution']['delta_scale'] = .2
    with pytest.raises(ValueError, match='digest'):
        verify_frozen_plan(changed, bank, allow_heldout=True)
    Path(plan['checkpoints']['carry']['frozen_path']).write_bytes(b'CHANGED TEST BYTES')
    with pytest.raises(ValueError, match='carry frozen checkpoint hash'):
        verify_frozen_plan(plan, bank, allow_heldout=True)


def test_development_subsets_and_course_legacy_are_separate(bank):
    selected = select_cases(bank, 'validation', [14, 5], layout_limit=1)
    assert len(selected) == 2 and len({case['maze_hash'] for case in selected}) == 1
    legacy = select_cases(bank, 'course_legacy', [14, 5], legacy_seeds=[42, 97])
    assert len(legacy) == 4
    assert all(case['maze_hash'] is None and 'maze_seed' not in case['env_kwargs'] for case in legacy)
    assert all(case['env_kwargs']['seed_mode'] == 'course-legacy-v1' for case in legacy)
    assert all(case['partition'] == 'course_legacy' for case in legacy)


def test_deadline_separates_reaching_final_hold_and_settling():
    outcome = {'success': True, 'physics_ticks': 15120, 'first_complete_reach_physics_tick': 14990,
               'first_lift_physics_tick': 2300, 'first_grasped_lift_action': 465}
    result = deadline_metrics(outcome, 480)
    assert result['task_success'] and result['reached_within_deadline']
    assert not result['success_within_deadline']
    assert not result['reached_within_deadline_including_initialization']
    assert result['first_grasped_lift_physics_tick'] == 2325
    assert result['first_grasped_lift_physics_tick_including_initialization'] == 2805
    assert result['total_physics_ticks_including_initialization'] == 15600
    outcome.update(physics_ticks=14500, first_complete_reach_physics_tick=14400)
    result = deadline_metrics(outcome, 480)
    assert result['success_within_deadline_including_initialization']
    outcome['success'] = False
    result = deadline_metrics(outcome, 480)
    assert not any(result[key] for key in ('task_success', 'success_within_deadline', 'reached_within_deadline'))


def test_setup_failures_and_pending_cases_stay_in_full_denominator(bank):
    cases = select_cases(bank, 'train', [14, 5], layout_limit=1)
    failed = failure_result(cases[0], RuntimeError('planner failed'))
    failed.update(deadline_metrics(failed, 0))
    summary = summarize_results([failed], cases)
    assert summary['planned_cases'] == 2 and summary['completed_cases'] == 1
    assert summary['success_rate'] == 0. and not summary['cost_accounting_complete']
    assert summary['missing_case_ids'] == [cases[1]['case_id']]
    passed = {'case_id': cases[1]['case_id'], 'success': True, 'success_within_deadline': False}
    summary = summarize_results([failed, passed], cases)
    assert summary['success_rate'] == .5 and summary['success_rate_within_deadline'] == 0.
    assert len(summary['failed_cases']) == 1


def test_declared_checkpoint_observation_interfaces_are_enforced():
    def policy(obs, goal, action):
        return SimpleNamespace(action_space=spaces.Box(-1., 1., (action,), dtype=np.float32),
                               observation_space=spaces.Dict({
                                   'observation': spaces.Box(-np.inf, np.inf, (obs,), dtype=np.float32),
                                   'achieved_goal': spaces.Box(-np.inf, np.inf, (goal,), dtype=np.float32),
                                   'desired_goal': spaces.Box(-np.inf, np.inf, (goal,), dtype=np.float32)}))
    policies = {'grasp': policy(54, 3, 4), 'carry': policy(580, 2, 2), 'reach': policy(50, 3, 3)}
    validate_policy_interfaces(policies)
    policies['grasp'] = policy(50, 3, 4)
    with pytest.raises(ValueError, match='grasp checkpoint'):
        validate_policy_interfaces(policies)


def test_system_optional_factory_binds_live_state_without_reset(monkeypatch):
    native = SimpleNamespace(seed=42, step_count=0, substeps=5, drop_events=0)
    calls = []
    monkeypatch.setattr('pnr.system.ReferenceController',
                        lambda env, seed, speed_scale: SimpleNamespace(env=env, seed=seed, speed_scale=speed_scale))
    class Wrapper:
        action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        def _observation(self):
            return {'achieved_goal': np.zeros(3, dtype=np.float32), 'desired_goal': np.ones(3, dtype=np.float32)}
        def step(self, action):
            native.step_count += 3
            return self._observation(), 0., False, False, {'is_success': True, 'bilateral_contact': True}
    def factory(env, controller, horizon):
        calls.append((env, controller, horizon))
        return Wrapper()
    policies = {name: SimpleNamespace(predict=lambda observation, deterministic: (np.zeros(4), None))
                for name in ('grasp', 'carry', 'reach')}
    system = LearnedSystem(native, policies, grasp_horizon=4,
                           skill_factories={'grasp': factory}, controller_speed_scale=1.2)
    system._skill('grasp')
    assert calls == [(native, system.controller, 4)]
    assert system.controller.speed_scale == 1.2
    assert system.stage_results['grasp']['source_action_calls'] == 3
    assert system.stage_results['grasp']['success']
