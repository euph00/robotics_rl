"""Learner/expert labels and replay boundaries, using mocked physics only."""
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.phase2_carry_demos as collector


class FakeEnv:
    goal_scale=20.;delta_scale=.1;_episode_horizon=100
    def __init__(self,xy=(4.5,0),goal=(3.5,0),failure_step=None):
        self.xy=np.array(xy,dtype=float);self.goal=np.array(goal,dtype=np.float32)/20
        self.steps=0;self.failure_step=failure_step;self.executed=[]
        self.native=SimpleNamespace(step_count=0,physics_steps=0,cube_positions=[],table_id=1,
                                    p=SimpleNamespace(getAABB=lambda _:None))
    def obs(self):
        return {'achieved_goal':self.xy.astype(np.float32)/20,'desired_goal':self.goal.copy()}
    def reset(self,*args,**kwargs):
        raise AssertionError('No reset inside a learner/expert episode')
    def step(self,action):
        self.executed.append(action.copy());self.xy+=np.asarray(action)*self.delta_scale
        self.steps+=1;self.native.step_count+=5;self.native.physics_steps+=25
        success=np.linalg.norm(self.xy-self.goal*20)<.03
        terminated=self.steps==self.failure_step
        return self.obs(),-1.,terminated,success,dict(is_success=success,bilateral_contact=not terminated,
            action_calls_delta=5,physics_ticks_delta=25,physical_failure=terminated)


def collect(monkeypatch,tmp_path,env,action=(1.,0),**kwargs):
    monkeypatch.setattr(collector,'maze_astar',lambda walls,start,goal,**kw:[np.array(start),np.array(goal)])
    policy=SimpleNamespace(predict=lambda obs,deterministic:(np.array(action,dtype=np.float32),None))
    cost={key:0 for key in collector.COST_FIELDS}
    result=collector.collect_on_policy_episode(env,env.obs(),policy,cost,tmp_path/'STOP',**kwargs)
    return result,cost


def test_true_actions_expert_labels_and_live_takeover_preserve_replay(monkeypatch,tmp_path):
    env=FakeEnv();(rows,success,info,stopped,report),cost=collect(monkeypatch,tmp_path,env,learner_budget=2)
    assert success and not stopped and report['takeover']['decision']==3
    assert report['takeover']['reason']=='learner_budget'
    assert report['learner_decisions']==2 and report['expert_recovery_decisions']>0
    assert report['interior_teacher_west_rows']>0 and report['distinct_action_label_rows']==2
    assert cost['source_action_calls']==len(rows)*5==env.native.step_count
    for index,row in enumerate(rows):
        np.testing.assert_array_equal(row['action'],env.executed[index])
        np.testing.assert_array_equal(row['observation']['desired_goal'],env.goal)
        if index<len(rows)-1:
            for key in row['next_observation']:
                np.testing.assert_array_equal(row['next_observation'][key],rows[index+1]['observation'][key])
            assert not row['terminated'] and not row['truncated']
    assert rows[0]['action'][0]==1 and rows[0]['teacher_action'][0]==-1
    assert rows[2]['action'][0]==rows[2]['teacher_action'][0]==-1
    assert rows[-1]['truncated'] and not rows[-1]['terminated']


def test_stall_uses_full_declared_window(monkeypatch,tmp_path):
    env=FakeEnv();(rows,success,_,_,report),_=collect(monkeypatch,tmp_path,env,action=(0,0),stall_window=6)
    assert success and report['takeover']['decision']==7
    assert report['takeover']['reason']=='geometric_progress_stalled'
    assert [row['control_source'] for row in rows[:6]]==['learner']*6


def test_geometric_progress_allows_euclidean_detour(monkeypatch,tmp_path):
    env=FakeEnv(xy=(0,0),goal=(0,10));env.delta_scale=1;env._episode_horizon=5
    monkeypatch.setattr(collector,'maze_astar',lambda walls,start,goal,**kw:
                        [np.array(start),np.array([5.,0]),np.array([5.,10]),np.array(goal)])
    policy=SimpleNamespace(predict=lambda obs,deterministic:(np.array([1.,0]),None))
    rows,_,_,_,report=collector.collect_on_policy_episode(env,env.obs(),policy,
        {key:0 for key in collector.COST_FIELDS},tmp_path/'STOP',learner_budget=10,stall_window=3)
    assert report['takeover'] is None
    assert rows[-1]['expert_remaining_path_m']<rows[0]['expert_remaining_path_m']
    assert np.linalg.norm(env.xy-env.goal*20)>10  # Euclidean distance legitimately increased.


def test_physical_failure_and_costs_are_retained(monkeypatch,tmp_path):
    env=FakeEnv(failure_step=3)
    (rows,success,_,_,report),cost=collect(monkeypatch,tmp_path,env,learner_budget=2)
    assert not success and len(rows)==3 and rows[-1]['terminated']
    assert not rows[-1]['truncated'] and cost['source_action_calls']==15
    assert report['takeover'] is not None


def test_unavailable_teacher_closes_previous_boundary_without_fabricated_label(monkeypatch,tmp_path):
    env=FakeEnv();queries=[]
    def route(walls,start,goal,**kwargs):
        queries.append(start)
        if len(queries)==3:raise ValueError('No clear endpoint')
        return [np.array(start),np.array(goal)]
    monkeypatch.setattr(collector,'maze_astar',route)
    policy=SimpleNamespace(predict=lambda obs,deterministic:(np.array([1.,0]),None))
    cost={key:0 for key in collector.COST_FIELDS}
    rows,success,_,_,report=collector.collect_on_policy_episode(env,env.obs(),policy,cost,tmp_path/'STOP')
    assert not success and len(rows)==2 and rows[-1]['truncated'] and not rows[-1]['terminated']
    assert report['collector_error'].startswith('teacher_route_unavailable')
    assert cost['source_action_calls']==10


def test_seed_fixed_correlated_noise_preserves_mean_and_actual_action(monkeypatch,tmp_path):
    collected=[]
    for seed in (123,123,124):
        env=FakeEnv();env._episode_horizon=12
        result,_=collect(monkeypatch,tmp_path,env,action=(.2,.0),learner_budget=10,
                         noise_std=.35,noise_rho=.9,rng=np.random.default_rng(seed))
        collected.append(result)
    first,collected_again,other=collected
    for a,b in zip(first[0],collected_again[0]):
        np.testing.assert_array_equal(a['action'],b['action'])
    assert not np.array_equal(first[0][0]['action'],other[0][0]['action'])
    for row in first[0][:10]:
        np.testing.assert_allclose(row['action'],row['learner_mean_action']+row['applied_action_noise'],atol=1e-7)
        assert np.max(np.abs(row['applied_action_noise']))<=.500001
        assert row['teacher_action'][0]<-.5
    assert not first[-1]['learner_deterministic'] and first[-1]['base_policy_predict_deterministic']
    assert first[-1]['takeover']['decision']==11
