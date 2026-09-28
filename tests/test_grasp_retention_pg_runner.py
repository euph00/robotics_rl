"""No-physics checks for fresh-score orchestration and accounting boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
from scripts import phase2_train_grasp_retention_pg as runner


def obs(v=0.):
    return {'observation':np.full(54,v,np.float32),'achieved_goal':np.array([0,0,v],np.float32),
            'desired_goal':np.array([0,0,.24],np.float32)}


def results(control,candidate,prep_loss=False):
    rows=[]
    for branch,total in [('control',control),('retention',candidate)]:
        for i in range(40):rows.append({'branch':branch,'case_id':str(i),'Y':int(i<total),'outcome_known':True,
            'cost_accounting_complete':True,'prepared_handover':not(prep_loss and branch=='retention' and i==0)})
    return rows


@pytest.mark.parametrize('control,candidate,passed',[(30,33,False),(33,36,True),(35,37,False),(35,38,True),
    (37,40,True),(38,39,False),(38,40,True),(39,40,True),(40,40,False)])
def test_prospective_gate_absolute_and_feasible_ceiling(control,candidate,passed):
    gate=runner.endpoint_gate(results(control,candidate),list(map(str,range(40))))
    assert gate['passed'] is passed and gate['complete']
    assert gate['net_gain']==candidate-control


def test_gate_never_drops_missing_unknown_duplicate_or_preparation_losses():
    ids=list(map(str,range(40)));rows=results(33,40)
    assert not runner.endpoint_gate(rows[:-1],ids)['complete']
    assert not runner.endpoint_gate(rows+[rows[0]],ids)['complete']
    bad=deepcopy(rows);bad[0]['outcome_known']=False
    assert not runner.endpoint_gate(bad,ids)['complete']
    bad=deepcopy(rows);bad[0]['cost_accounting_complete']=False
    assert not runner.endpoint_gate(bad,ids)['complete']
    assert not runner.endpoint_gate(results(33,40,True),ids)['passed']


def test_extra_bc_sampling_is_fixed_and_independent_of_global_rng():
    rows=[{'observation':obs(float(i)),'action':np.full(4,i,np.float32)} for i in range(11)]
    groups={'a':np.arange(3),'b':np.arange(3,11)}
    np.random.seed(33);before=pickle.dumps(np.random.get_state())
    a=runner.fixed_bc_batch(rows,groups,2)
    assert pickle.dumps(np.random.get_state())==before
    np.random.seed(900);b=runner.fixed_bc_batch(rows,groups,2)
    np.testing.assert_array_equal(a[2],b[2]);assert len(a[2])==400
    assert set(a[2])==set(range(11))


def test_partial_carry_action_keeps_actual_costs_without_fake_tuple(tmp_path):
    native=SimpleNamespace(step_count=7,physics_steps=35,observe=lambda:{'actual':'after partial'})
    class Env:
        def __init__(self):self.native=native
        def _observation(self):return obs()
        def step(self,action):
            native.step_count+=2;native.physics_steps+=10
            raise runner.DeadlineReached('bounded stop')
    wrapped=runner.NativeTupleRecorder(Env(),tmp_path/'carry.pkl');wrapped._observation()
    with pytest.raises(runner.DeadlineReached):wrapped.step(np.ones(2))
    assert wrapped.rows==[]
    row=pickle.loads((tmp_path/'carry.partial.pkl').read_bytes())
    assert row['before']==(7,35) and row['after']==(9,45) and row['not_a_complete_transition']
    assert row['actual_native_observation']=={'actual':'after partial'}


@pytest.mark.parametrize('integrity_error',[False,True])
def test_zero_action_physical_failure_known_but_swallowed_sampling_failure_unknown(monkeypatch,tmp_path,integrity_error):
    out=tmp_path/'artifacts';out.mkdir()
    monkeypatch.setattr(runner.os,'dup2',lambda *a:None)
    monkeypatch.setattr(runner,'actor_digest',lambda actor:'actor')
    monkeypatch.setattr(runner,'_MODEL',SimpleNamespace(actor=object()))
    class Predictor:
        def __init__(self,*a,**kw):self.samples=[];self.integrity_error=None
        def sampling_report(self):
            if self.integrity_error:raise ValueError(self.integrity_error)
            return {'intact':True}
    monkeypatch.setattr(runner,'RecordedGaussianPredictor',Predictor)
    class Native:
        settle_steps=480;drop_events=0
        def __init__(self,**kw):self.step_count=0;self.physics_steps=0
        def scene_manifest(self):return {'maze_hash':'maze'}
        def observe(self):return {'actual':'original start'}
        def evaluate(self):return {'drop_events':0}
        def close(self):pass
    class System:
        stage='pregrasp';stage_results={};wall_contact_calls=robot_table_contact_calls=0
        def __init__(self,native,predictor,on_step):
            self.predictor=predictor;self.skill_factories={};self.calls_by_source={'planned':0}
        def prepare(self):
            # Mirrors LearnedSystem broad catch after instrumentation raised.
            if integrity_error:self.predictor.integrity_error='Gaussian mismatch before sample append'
            return {'prepared_handover':False,'failed_stage':'pregrasp','error':'failed preparation','stages':{}}
    monkeypatch.setattr(runner,'PickNavReachEnv',Native);monkeypatch.setattr(runner,'LearnedCarryPreparation',System)
    case={'case_id':'train_000_object_14','maze_hash':'maze','env_kwargs':{}}
    slot={'episode_id':'example','replicate_index':0,'torch_seed':117001}
    result=runner.suffix_worker((case,slot,str(out),{},float('inf'),'actor'))
    assert result['outcome_known'] is not integrity_error
    assert result['Y']==(None if integrity_error else 0)
    assert result['total_physics_ticks_including_initialization']==480
    assert result['cost_accounting_complete']
    assert result['actual_suffix_end_observation']=={'actual':'original start'}
    assert (tmp_path/'STOP').exists() is integrity_error
    episode=pickle.loads((out/'example_episode.pkl').read_bytes())
    assert episode['samples']==episode['transitions']==[]


@pytest.mark.parametrize('during_setup',[True,False])
def test_deadline_after_native_call_counts_that_call_in_setup_or_online(monkeypatch,tmp_path,during_setup):
    stop=tmp_path/'STOP';directory=tmp_path/'artifacts';directory.mkdir()
    class Native:
        def __init__(self):self.step_count=0;self.physics_steps=0;self.stop_on=2 if during_setup else None
        def step(self,action):
            self.step_count+=1;self.physics_steps+=5
            if self.stop_on==self.step_count:stop.touch()
            return {},0.,False,False,{}
        def observe(self):return {'native_calls':self.step_count}
    original=Native.step
    class Multi(gym.Env):
        def __init__(self,*a,**kw):
            self.observation_space=gym.spaces.Box(-1,1,(1,));self.action_space=gym.spaces.Box(-1,1,(1,))
            self.setup_action_calls=self.setup_physics_ticks=self.initialization_physics_ticks=0
            self.failed_setup_attempts=[];self.active=None
        def reset(self,**kw):
            native=Native();self.active=SimpleNamespace(native=native)
            try:
                for _ in range(3):native.step(np.zeros(1))
            except Exception as error:
                self.setup_action_calls+=native.step_count;self.setup_physics_ticks+=native.physics_steps
                self.failed_setup_attempts.append({'costs':{'source_action_calls':native.step_count},'error':repr(error)})
                raise
            self.setup_action_calls+=native.step_count;self.setup_physics_ticks+=native.physics_steps
            return np.zeros(1),{}
        def step(self,action):
            for _ in range(3):self.active.native.step(action)
            return np.zeros(1),0.,False,False,{}
        def close(self):pass
    monkeypatch.setattr(runner,'PickNavReachEnv',Native);monkeypatch.setattr(runner,'MultiObjectGrasp',Multi)
    env=runner.CountedPickup([],0,1,directory,float('inf'))
    try:
        if during_setup:
            with pytest.raises(runner.DeadlineReached):env.reset()
            assert env.calls==0 and env.env.setup_action_calls==2
        else:
            env.reset();env.env.active.native.stop_on=5
            with pytest.raises(runner.DeadlineReached):env.step(np.zeros(1))
            assert env.calls==2 and env.env.setup_action_calls==3 and env.decisions==0
            assert env.partial[0]['native_calls_completed']==2
        ledger=json.loads(env.path.read_text())
        assert ledger['total_physics_ticks_including_initialization']==(10 if during_setup else 25)
        assert ledger['cost_accounting_complete']
    finally:env.close()
    assert Native.step is original


def test_pre_treatment_guard_checks_rng_and_optimizers_not_wall_time(tmp_path):
    a,b=tmp_path/'control',tmp_path/'retention'
    for d in (a,b):
        (d/'artifacts').mkdir(parents=True)
        (d/'artifacts/initializer.json').write_text(json.dumps({'actor_sha256':'a','protected_state':{'optimizer':'x'},
            'numpy_rng':'n','torch_rng':'t','python_rng':'p','demo_rng':'d','critic_warmup':{'wall_seconds':str(d)}}))
        (d/'artifacts/pre_first_auxiliary.json').write_text(json.dumps({'actor_sha256':'a','protected_state':{'optimizer':'x'},'rng':{'torch':'t'}}))
    runner.matched_initial_states(a,b)
    path=b/'artifacts/pre_first_auxiliary.json';changed=json.loads(path.read_text());changed['rng']['torch']='other';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Unmatched'):runner.matched_initial_states(a,b)


def test_invalid_pre_action_observation_latches_recorder_integrity(tmp_path):
    env=SimpleNamespace(action_space=None,observation_space=None,_observation=lambda:{'wrong':'schema'})
    recorder=runner.GraspRecorder(env,tmp_path/'grasp.pkl')
    with pytest.raises(ValueError,match='keys'):recorder._observation()
    assert recorder.integrity_error and recorder.rows==[]
