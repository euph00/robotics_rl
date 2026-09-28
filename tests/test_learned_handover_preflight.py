"""The diagnostic stops before reach and keeps failed preparation denominators."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import phase2_learned_handover_preflight as probe


def test_fixed_denominator_retains_failed_preparations_and_costs():
    cases=[{'case_id':'a'},{'case_id':'b'}]
    failed={'case_id':'a','prepared_handover':False,'retained_short_transport':False,
            'source_action_calls':20,'physics_ticks':100,'initialization_physics_ticks':480,
            'total_physics_ticks_including_initialization':580,'cost_accounting_complete':True}
    result=probe.aggregate_results([failed],cases)
    assert result['planned_attempts']==2 and result['completed_attempts']==1
    assert result['retained_fraction_of_planned']==0 and result['missing_case_ids']==['b']
    assert result['total_physics_ticks_including_initialization']==580
    assert result['full_task_evaluation'] is False
    with pytest.raises(ValueError):probe.aggregate_results([failed,failed],cases)


def fake_system(monkeypatch,fail=False):
    s=probe.TeacherCarryPreflight.__new__(probe.TeacherCarryPreflight)
    s.teacher_budget=3;s.delta_scale=.075;s.teacher_transitions=[];s.retained_short_transport=False
    s.native=SimpleNamespace(step_count=100,physics_steps=500,drop_events=0,
        cube_positions=[],goal_pos=np.array([5.,0.,.9]),table_id=0,
        p=SimpleNamespace(getAABB=lambda _:None))
    s.controller=SimpleNamespace(base_position=lambda:np.array([0.,0.]))
    goal=np.array([.25,0.],dtype=np.float32)
    obs=lambda:{'observation':np.zeros(580,dtype=np.float32),'achieved_goal':np.zeros(2,dtype=np.float32),'desired_goal':goal.copy()}
    n=[0]
    def step(action):
        n[0]+=1;s.native.step_count+=5;s.native.physics_steps+=25
        return obs(),-1.,fail,n[0]==3,{'bilateral_contact':not fail,'failure_reason':'lost_grasp' if fail else None,'is_success':False,'goal_distance':5.}
    carry=SimpleNamespace(_observation=obs,step=step,goal_scale=20.,delta_scale=.075)
    monkeypatch.setattr(probe.GeneralCarryEnv,'bind',lambda *args,**kwargs:carry)
    monkeypatch.setattr(probe,'maze_astar',lambda *args,**kwargs:np.array([[0.,0.],[5.,0.]]))
    s._begin=lambda *args:(100,0.)
    s.stage_results={}
    def finish(stage,start,success,**kwargs):s.stage_results[stage]={'success':success,'calls':s.native.step_count-start[0],**kwargs}
    s._finish=finish
    return s


def test_teacher_uses_continuous_state_and_stops_after_exact_budget(monkeypatch):
    s=fake_system(monkeypatch)
    with pytest.raises(probe._DiagnosticComplete):s._skill('carry')
    assert s.preparation_calls==100 and s.native.step_count==115
    assert s.retained_short_transport and len(s.teacher_transitions)==3
    assert s.stage_results['teacher_carry']['calls']==15
    assert all(np.array_equal(t['observation']['desired_goal'],[.25,0.]) for t in s.teacher_transitions)
    with pytest.raises(AssertionError):s._skill('reach')


def test_teacher_failure_keeps_real_terminal_transition(monkeypatch):
    s=fake_system(monkeypatch,fail=True)
    with pytest.raises(probe.StageFailure,match='lost_grasp'):s._skill('carry')
    assert not s.retained_short_transport and s.native.step_count==105
    assert len(s.teacher_transitions)==1 and s.teacher_transitions[0]['terminated']
    assert not s.stage_results['teacher_carry']['success']


def test_diagnostic_success_is_never_full_task_success(monkeypatch):
    s=fake_system(monkeypatch)
    s.retained_short_transport=True;s.policy_decisions={'grasp':10,'carry':0,'reach':0}
    s.calls_by_source={'planned':100,'learned':30,'hold':50,'teacher':15}
    s.wall_contact_calls=s.robot_table_contact_calls=0
    s.first_lift_action=50;s.first_grasped_lift_action=51;s.navigation_start_action=100
    s.preparation_calls=100
    s.stage_results={'planned_carry_posture':{'success':True}}
    def run(_):raise probe._DiagnosticComplete()
    monkeypatch.setattr(probe.LearnedSystem,'run',run)
    result=s.run_preflight()
    assert result['success'] and result['prepared_handover']
    assert not result['full_task_evaluation'] and result['teacher_in_carry']
    assert not result['teacher_in_grasp'] and result['policy_decisions']['reach']==0
