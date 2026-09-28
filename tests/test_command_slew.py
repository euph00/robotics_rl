"""No robot: exact baseline identity, native command bound and diagnostic gates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pnr.command_slew import CommandSlewPolicy,native_control,slew_control
from scripts import phase2_carry_slew_diagnostic as runner


class Policy:
    action_space=None;observation_space=None
    def __init__(self,action):self.action=action;self.state=object();self.calls=0;self.result=(action,self.state)
    def predict(self,obs,deterministic):assert deterministic is True;self.calls+=1;return self.result


def test_raw_branch_returns_exact_prediction_objects_and_does_not_mutate_inputs():
    raw=np.array([1.,1.],np.float32);policy=Policy(raw);obs={'observation':np.arange(4,dtype=np.float32)}
    wrapper=CommandSlewPolicy(policy);before=obs['observation'].copy();result=wrapper.predict(obs)
    assert result is policy.result and result[0] is raw and result[1] is policy.state and policy.calls==1
    np.testing.assert_array_equal(raw,[1.,1.]);np.testing.assert_array_equal(obs['observation'],before)
    np.testing.assert_allclose(wrapper.records[0]['native_raw_control'],[2**-.5]*2)


def test_radial_bound_uses_normalized_native_control_and_resets_per_case():
    model=Policy(np.array([1.,1.],np.float32));obs={'observation':np.zeros(2)}
    wrapper=CommandSlewPolicy(model,filtered=True);action,_=wrapper.predict(obs)
    np.testing.assert_allclose(action,[.18/np.sqrt(2)]*2,atol=3e-7,rtol=0)
    previous=native_control(action)
    for i in range(50):
        model.action[:]=[(-1.)**i,(-1.)**(i//3)]
        output,_=wrapper.predict(obs);effective=native_control(output)
        assert output.dtype==np.float32
        assert np.linalg.norm(effective-previous)<=.18+1e-12
        assert np.linalg.norm(effective)<=1.+1e-12
        previous=effective
    fresh=CommandSlewPolicy(model,filtered=True);action,_=fresh.predict(obs)
    assert np.linalg.norm(native_control(action))<=.18+1e-12
    np.testing.assert_array_equal(fresh.records[0]['previous_effective_requested_control'],[0.,0.])


def test_no_change_inside_limit_and_no_hidden_vector_clipping():
    output,effective=slew_control([.2,-.1],[.1,-.1])
    np.testing.assert_allclose(output,[.2,-.1]);np.testing.assert_array_equal(native_control(output),effective)
    for bad in ([np.nan,0.],[0.,0.,0.]):
        with pytest.raises(ValueError):native_control(bad)


def test_predictor_or_persistence_errors_latch_even_before_any_native_action():
    wrapper=CommandSlewPolicy(Policy(np.array([np.nan,0.])),filtered=True)
    with pytest.raises(ValueError):wrapper.predict({'observation':np.zeros(2)})
    assert wrapper.integrity_error and wrapper.records==[]
    def broken(row):raise OSError('disk failure')
    wrapper=CommandSlewPolicy(Policy(np.zeros(2)),persist=broken)
    with pytest.raises(OSError):wrapper.predict({'observation':np.zeros(2)})
    assert wrapper.integrity_error and len(wrapper.records)==1


def rows(control,candidate,raw_timed=20,slew_timed=20):
    return [dict(branch=b,case_id=str(i),success=i<n,success_within_deadline_including_initialization=i<t,
        diagnostic_outcome_known=True,cost_accounting_complete=True) for b,n,t in [('raw',control,raw_timed),('slew',candidate,slew_timed)] for i in range(40)]


@pytest.mark.parametrize('control,candidate,passed',[(31,35,False),(31,36,True),(35,37,False),(37,40,True),(38,39,False),(38,40,True),(39,40,True),(40,40,False)])
def test_fixed_full_task_gate_ceiling_and_absolute_quality(control,candidate,passed):
    assert runner.gate(rows(control,candidate),[{'case_id':str(i)} for i in range(40)])['passed'] is passed


def test_gate_rejects_time_regression_unknown_missing_and_duplicate():
    cases=[{'case_id':str(i)} for i in range(40)];data=rows(30,40,21,20)
    assert not runner.gate(data,cases)['passed']
    data=rows(30,40);data[0]['diagnostic_outcome_known']=False
    assert not runner.gate(data,cases)['complete']
    assert not runner.gate(rows(30,40)[:-1],cases)['complete']
    assert not runner.gate(rows(30,40)+[rows(30,40)[0]],cases)['complete']


@pytest.mark.parametrize('mode',['clean','integrity','deadline','crossed_deadline','wrong_scene'])
def test_reused_evaluator_wrapper_restores_global_state_and_marks_unknown(monkeypatch,tmp_path,mode):
    now=[0.];monkeypatch.setattr(runner.time,'time',lambda:now[0])
    policies={k:Policy(np.zeros(2,np.float32)) for k in ('grasp','carry','reach')};carry=policies['carry']
    monkeypatch.setattr(runner.evaluator,'_WORKER_POLICIES',policies)
    monkeypatch.setattr(runner,'actor_parameter_digest',lambda model:'fixed')
    class Native:
        def __init__(self):self.step_count=0
        def step(self,action):
            self.step_count+=1
            if mode=='crossed_deadline':now[0]=101.
            return {}
    monkeypatch.setattr(runner.evaluator,'PickNavReachEnv',Native);original=Native.step
    def evaluate(case,execution,directory,video):
        adapter=runner.evaluator._WORKER_POLICIES['carry'];native=Native()
        if mode=='integrity':
            carry.action[:]=np.nan
            try:adapter.predict({'observation':np.zeros(2)})
            except ValueError:pass # emulate evaluator's broad catch
        else:adapter.predict({'observation':np.zeros(2)})
        if mode=='deadline':
            (tmp_path/'STOP').touch()
            try:native.step(np.zeros(2))
            except runner.DiagnosticInterrupted:pass
        if mode=='crossed_deadline':native.step(np.zeros(2))
        return {'case_id':case['case_id'],'scene_manifest':{'maze_hash':'wrong' if mode=='wrong_scene' else 'expected'},'success':False,'source_action_calls':native.step_count,'physics_ticks':native.step_count*5,
                'cost_accounting_complete':True,'total_physics_ticks_including_initialization':480,
                'policy_decisions':{'carry':len(adapter.records) if mode in ('clean','crossed_deadline','wrong_scene') else 0}}
    monkeypatch.setattr(runner.evaluator,'evaluate_case',evaluate)
    case={'case_id':'train_016_object_14','maze_hash':'expected'}
    result=runner.evaluate_branch((case,'slew',{},str(tmp_path/'slew'),100.,str(tmp_path/'STOP')))
    assert result['diagnostic_outcome_known'] is (mode=='clean')
    assert runner.evaluator._WORKER_POLICIES['carry'] is carry and Native.step is original
    assert (tmp_path/'STOP').exists() is (mode!='clean')
    if mode=='crossed_deadline':assert result['source_action_calls']==1 and result['physics_ticks']==5
    assert (tmp_path/'slew/train_016_object_14_carry_commands.jsonl').exists()


def test_post_float32_native_controls_obey_bound_over_random_sequences():
    rng=np.random.default_rng(218);previous=np.zeros(2)
    for desired in rng.uniform(-1.5,1.5,(1000,2)):
        action,effective=slew_control(desired.astype(np.float32),previous)
        actual=native_control(np.asarray(action,dtype=np.float32))
        np.testing.assert_array_equal(actual,effective)
        assert np.linalg.norm(actual-previous)<=.18+1e-12
        previous=actual
