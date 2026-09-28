"""Reach acceptance regression tests using mocked physics only."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from pnr.system import LearnedSystem, StageFailure


def make_system(monkeypatch, *, tolerance=.10, info=None, terminated=False, drops=0):
    report = dict(is_success=False, complete_task_success=True, dist_to_goal=.05,
                  goal_distance=.05, bilateral_contact=True)
    report.update(info or {})
    native = SimpleNamespace(seed=42, step_count=0, substeps=5, drop_events=drops,
                             physics_steps=0, object_idx=2, object_name='can',
                             initial_object_pos=np.zeros(3), init_qpos=np.zeros(4),
                             last_target=np.zeros(4), ever_lifted=False)
    native.observe = lambda: dict(object_pos=np.zeros(3), qpos=np.zeros(4))
    native.contacts = lambda: dict(bilateral_contact=True)
    native.scene_manifest = lambda: {}
    native.evaluate = lambda: dict(report)
    grasp = SimpleNamespace(position=np.zeros(3), quaternion=np.array([0., 0., 0., 1.]),
                            as_dict=lambda: {})
    controller = SimpleNamespace(prepare_grasp=lambda: grasp, prepare_carry=lambda: None,
                                 planner=SimpleNamespace(ee_pose=lambda: (
                                     np.array([0., 0., .20]), grasp.quaternion)))
    monkeypatch.setattr('pnr.system.ReferenceController', lambda *args, **kwargs: controller)

    class Wrapper:
        action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        def _observation(self):
            return dict(achieved_goal=np.array([0., 0., .05]), desired_goal=np.zeros(3))
        def step(self, action):
            native.step_count += 3
            native.physics_steps += 15
            return self._observation(), 0., terminated, False, dict(report)

    policy = SimpleNamespace(predict=lambda observation, deterministic: (np.zeros(3), None))
    system = LearnedSystem(native, dict.fromkeys(('grasp', 'carry', 'reach'), policy),
                           reach_horizon=2, reach_tolerance_m=tolerance,
                           skill_factories={'reach': lambda *args, **kwargs: Wrapper()})
    return system, report


def test_native_task_acceptance_uses_10cm_despite_stricter_training_goal(monkeypatch):
    system, _ = make_system(monkeypatch)
    system._skill('reach')
    stage = system.stage_results['reach']
    assert stage['success'] and stage['policy_decisions'] == 1
    assert stage['reach_tolerance_m'] == .10
    assert stage['reach_acceptance'] == 'native_complete_task_success'
    assert stage['native_goal_distance'] == .05


def test_legacy_acceptance_retains_skill_predicate_exactly(monkeypatch):
    system, _ = make_system(monkeypatch, tolerance=.04)
    with pytest.raises(StageFailure, match='decision_horizon'):
        system._skill('reach')
    assert system.policy_decisions['reach'] == 2
    # Do not add native-history requirements to historical execution semantics.
    system, _ = make_system(monkeypatch, tolerance=.04,
                           info=dict(is_success=True, complete_task_success=False))
    system._skill('reach')
    assert system.stage_results['reach']['reach_acceptance'] == 'legacy_skill_is_success'


@pytest.mark.parametrize('tolerance', [.08, .10])
@pytest.mark.parametrize('info,terminated,drops', [
    ({'dist_to_goal': .10}, False, 0),
    ({'dist_to_goal': float('nan')}, False, 0),
    ({'dist_to_goal': .11, 'is_success': True}, False, 0),
    ({'complete_task_success': False, 'is_success': True}, False, 0),
    ({'bilateral_contact': False}, False, 0),
    ({}, True, 0),
    ({}, False, 1),
])
def test_native_handover_rejects_missing_current_goal_or_retention(monkeypatch, tolerance, info, terminated, drops):
    system, _ = make_system(monkeypatch, tolerance=tolerance, info=info, terminated=terminated, drops=drops)
    with pytest.raises(StageFailure):
        system._skill('reach')
    assert not system.stage_results['reach']['success']


@pytest.mark.parametrize('tolerance', [.08, .10])
@pytest.mark.parametrize('hold_outcome', ['retained', 'unstable', 'outside_goal'])
def test_final_hold_and_final_native_recheck_remain_mandatory(monkeypatch, tolerance, hold_outcome):
    system, report = make_system(monkeypatch, tolerance=tolerance)
    system._observe_native_steps = nullcontext
    real_skill = system._skill
    calls = []
    def skill(name):
        calls.append(name)
        if name == 'reach':
            real_skill(name)
    def hold(stage, steps):
        calls.append((stage, steps))
        system.stage = stage
        if stage == 'final_hold':
            if hold_outcome == 'unstable':
                raise StageFailure('final_hold: retention failed')
            if hold_outcome == 'outside_goal':
                report['complete_task_success'] = False
    system._skill, system._held_pause = skill, hold
    result = system.run()
    assert calls == ['grasp', ('pick_hold', 50), 'carry', 'reach', ('final_hold', 60)]
    assert result['success'] == (hold_outcome == 'retained')
    assert result['reach_tolerance_m'] == tolerance and result['final_hold_source_calls'] == 60
    if hold_outcome == 'unstable':
        assert result['failed_stage'] == 'final_hold'


@pytest.mark.parametrize('tolerance', [.03, .05, .11, float('nan'), float('inf')])
def test_only_versioned_tolerances_are_allowed(monkeypatch, tolerance):
    with pytest.raises(ValueError, match='Reach tolerance'):
        make_system(monkeypatch, tolerance=tolerance)


def test_8cm_margin_keeps_acting_inside_native10cm_until_below8cm(monkeypatch):
    system, report = make_system(monkeypatch, tolerance=.08)
    distances=iter([.09,.079])
    def predict(observation, deterministic):
        report['dist_to_goal']=next(distances)
        return np.zeros(3),None
    system.policies['reach']=SimpleNamespace(predict=predict)
    system._skill('reach')
    stage=system.stage_results['reach']
    assert stage['success'] and stage['policy_decisions']==2
    assert stage['reach_acceptance']=='native_complete_task_success_with_8cm_margin'
    assert stage['native_goal_distance']==.079
    assert stage['source_action_calls']==6 and stage['physics_ticks']==30


def test_8cm_margin_does_not_tighten_final_native10cm_goal(monkeypatch):
    system, report = make_system(monkeypatch, tolerance=.08)
    system._observe_native_steps=nullcontext
    real_skill=system._skill
    system._skill=lambda name: real_skill(name) if name=='reach' else None
    calls=[]
    def hold(stage,steps):
        calls.append((stage,steps))
        if stage=='final_hold':
            report['dist_to_goal']=.09
            report['complete_task_success']=True
    system._held_pause=hold
    result=system.run()
    assert result['success'] and result['metrics']['dist_to_goal']==.09
    assert calls==[('pick_hold',50),('final_hold',60)]


@pytest.mark.parametrize('distance',[.08,.081,.099])
def test_8cm_handover_uses_strict_internal_margin(monkeypatch,distance):
    system,_=make_system(monkeypatch,tolerance=.08,info={'dist_to_goal':distance,'is_success':True})
    with pytest.raises(StageFailure,match='decision_horizon'):
        system._skill('reach')
