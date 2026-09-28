"""The collector preserves fixed cases, real corrective actions, and costs."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import phase2_learned_carry_demos as collector
from scripts import phase2_carry_demos as teacher


def test_two_attempts_are_declared_before_transport_outcomes():
    attempts=collector.expected_attempts(4,8401)
    assert [a['noise_std'] for a in attempts]==[0.,.12]
    assert [a['rng_seed'] for a in attempts]==[8409,8410]
    assert attempts[1]['state_origin']=='restored_physically_prepared_learned_handover'
    assert collector.expected_attempts(4,8401)==attempts


def test_preparation_failure_remains_in_case_and_cost_denominators():
    cases=[{'case_id':'good'},{'case_id':'failed_setup'}]
    results=[{'case_id':'failed_setup','prepared_handover':False,'attempts':[],
              'preparation_source_action_calls':60,'preparation_physics_ticks':300,
              'initialization_physics_ticks':480,'cost_accounting_complete':True},
             {'case_id':'good','prepared_handover':True,'attempts':[
                 {'attempt_index':0,'success':False},{'attempt_index':1,'success':True}],
              'preparation_source_action_calls':100,'preparation_physics_ticks':500,
              'source_action_calls':20,'physics_ticks':100,'initialization_physics_ticks':480,
              'cost_accounting_complete':True}]
    s=collector.case_summary(results,cases)
    assert s['planned_cases']==2 and s['prepared_handovers']==1
    assert s['planned_transport_opportunities']==4 and s['attempted_transports']==2
    assert s['successful_transports']==1 and s['successful_by_attempt']=={'0':0,'1':1}
    assert s['all_source_action_calls']==180 and s['all_physics_ticks']==1860
    assert not s['full_task_evaluation']
    with pytest.raises(ValueError):collector.case_summary(results+[results[0]],cases)


def test_shared_teacher_records_executed_noise_and_true_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(teacher,'maze_astar',lambda *a,**kw:np.array([[0.,0.],[3.,0.]]))
    native=SimpleNamespace(step_count=80,physics_steps=400,cube_positions=[],table_id=0,
                           p=SimpleNamespace(getAABB=lambda _:None))
    env=SimpleNamespace(native=native,controller=SimpleNamespace(base_position=lambda:np.zeros(2)),
                        _episode_horizon=3,goal_scale=20.,delta_scale=.075)
    observation={'observation':np.zeros(580,dtype=np.float32),'achieved_goal':np.zeros(2,dtype=np.float32),
                 'desired_goal':np.array([.15,0.],dtype=np.float32)}
    executed=[]
    def step(action):
        executed.append(action.copy());native.step_count+=2;native.physics_steps+=10
        following={k:v.copy() for k,v in observation.items()};following['achieved_goal'][0]=.001
        return following,-200.,True,False,{'is_success':False,'failure_reason':'lost_grasp',
            'action_calls_delta':2,'physics_ticks_delta':10}
    env.step=step;cost={'source_action_calls':0,'physics_ticks':0}
    rows,success,info,stopped=teacher.collect_episode(env,observation,np.random.default_rng(50),.12,cost,tmp_path/'stop')
    assert not success and not stopped and info['failure_reason']=='lost_grasp'
    assert len(rows)==1 and rows[-1]['terminated'] and not rows[-1]['truncated']
    assert np.array_equal(rows[0]['action'],executed[0])
    assert not np.array_equal(rows[0]['action'],rows[0]['teacher_action'])
    assert cost=={'source_action_calls':2,'physics_ticks':10}


def test_atomic_case_episode_write_preserves_failure_rows(tmp_path):
    import pickle
    path=tmp_path/'case.pkl'
    value=[{'success':False,'transitions':[{'terminated':True,'truncated':False}]}]
    collector.atomic_pickle(path,value)
    assert pickle.load(path.open('rb'))==value
    assert not path.with_suffix('.pkl.tmp').exists()


def test_diverse_schedule_keeps_canonical_identity_and_one_clean_attempt():
    cases=[{'case_id':f'case_{i}'} for i in range(80)]
    schedule=collector.preparation_schedule(cases,3,9401)
    assert len(schedule)==240 and len({collector.preparation_key(c) for c in schedule})==240
    assert len({c['case_id'] for c in schedule})==80
    assert schedule[-1]['preparation_variant']['torch_seed']==9640
    for i,case in enumerate(schedule):
        attempts=collector.planned_attempts(case,i,9401)
        assert len(attempts)==1 and attempts[0]['noise_std']==0.
        assert attempts[0]['attempt_index']==0 and attempts[0]['state_origin']=='continuous_original_start'
    failed=dict(schedule[0],prepared_handover=False,attempts=[],preparation_source_action_calls=50,
                preparation_physics_ticks=250,initialization_physics_ticks=480,cost_accounting_complete=True)
    summary=collector.case_summary([failed],schedule)
    assert summary['planned_cases']==80 and summary['completed_cases']==0
    assert summary['planned_preparations']==240 and summary['completed_preparations']==1
    assert summary['planned_transport_opportunities']==240 and summary['all_physics_ticks']==730
    assert summary['by_variant']['deterministic']['completed']==1
    with pytest.raises(ValueError):collector.preparation_schedule(schedule,3,9401)


def test_diverse_transition_provenance_and_schema_are_independent_of_physics():
    import copy
    from test_carry_dataset_merge import episode
    e=episode(success=False)
    case=collector.preparation_schedule([dict(case_id='case',maze_hash=e['maze_hash'])],3,9401)[1]
    before=copy.deepcopy(e)
    collector.attach_preparation_metadata(e['transitions'],case)
    costs=collector.validate_recorded_episode(e,case)
    assert costs=={'source_action_calls':15,'physics_ticks':75}
    for old,new in zip(before['transitions'],e['transitions']):
        assert new['info']['case_id']=='case' and new['info']['preparation_id']=='case__stochastic_1'
        assert np.array_equal(old['action'],new['action']) and old['reward']==new['reward']
        assert old['terminated']==new['terminated'] and old['truncated']==new['truncated']
    e['transitions'][0]['info']['preparation_variant']['torch_seed']=0
    assert e['transitions'][1]['info']['preparation_variant']['torch_seed']==9402
    with pytest.raises(ValueError):collector.validate_recorded_episode(e,case)
    collector.attach_preparation_metadata(e['transitions'],case)
    e['transitions'][0]['reward']=500.
    with pytest.raises(ValueError,match='rewards'):collector.validate_recorded_episode(e,case)


@pytest.mark.parametrize('schema_fault',[False,True])
def test_diverse_worker_executes_one_fresh_preparation_without_restore(monkeypatch,tmp_path,schema_fault):
    import copy
    from contextlib import nullcontext
    from test_carry_dataset_merge import episode
    sample=episode(success=False)
    case=collector.preparation_schedule([dict(case_id='case',layout_id=sample['layout_id'],
        maze_hash=sample['maze_hash'],object_idx=14,env_kwargs={'seed':42})],3,9401)[1]
    observed=copy.deepcopy(sample['transitions'][0]['observation'])
    instances=[]
    class Native:
        settle_steps=480;substeps=5;step_count=0;physics_steps=0
        def __init__(self,**kwargs):instances.append(self)
        def scene_manifest(self):return {'maze_hash':sample['maze_hash']}
        def evaluate(self):return {'drop_events':0}
        def close(self):pass
        def restore_state(self,*args,**kwargs):raise AssertionError('New mode must never restore')
    class Preparation:
        stage='test';display_stage='test';_active_stage_start=None
        def __init__(self,native,grasp,on_step):
            self.native=native;self.on_step=on_step;self.calls_by_source={'policy':0,'teacher':0}
            self.controller=SimpleNamespace(stage_results={},planner=SimpleNamespace(ee_pose=lambda:(np.zeros(3),np.zeros(4))))
            self.handover_observation=observed;self.handover_state=object()
        def tick(self,source):
            self.native.step_count+=1;self.native.physics_steps+=5;self.calls_by_source[source]+=1
            obs={'qpos':np.zeros(15),'qvel':np.zeros(15),'object_pos':np.zeros(3),'object_xyzw':np.zeros(4)}
            self.on_step(self,obs,{'bilateral_contact':True,'drop_events':0})
        def prepare(self):self.tick('policy');return {'prepared_handover':True}
        def _begin(self,*args):pass
        def _observe_native_steps(self):return nullcontext()
    active=[]
    def preparation(*args,**kwargs):
        result=Preparation(*args,**kwargs);active.append(result);return result
    def collect(carry,obs,rng,noise,costs,stop):
        assert noise==0. and len(instances)==1
        for _ in range(15):active[0].tick('teacher')
        costs.update(source_action_calls=15,physics_ticks=75)
        rows=copy.deepcopy(sample['transitions'])
        if schema_fault:rows[0]['reward']=500.
        return rows,False,{'failure_reason':'lost_grasp'},False
    monkeypatch.setattr(collector.os,'dup2',lambda *a:None)
    monkeypatch.setattr(collector,'PickNavReachEnv',Native)
    monkeypatch.setattr(collector,'LearnedCarryPreparation',preparation)
    monkeypatch.setattr(collector,'predictor_for_variant',lambda *a:SimpleNamespace(sampling_report=lambda:{'predictions':1}))
    monkeypatch.setattr(collector,'collect_episode',collect)
    monkeypatch.setattr(collector.GeneralCarryEnv,'bind',lambda *a,**kw:SimpleNamespace(_observation=lambda:observed))
    out=tmp_path/'artifacts';out.mkdir()
    result=collector.worker((case,1,str(out),dict(collection_seed=9401,horizon=900,
        protocol=collector.DIVERSE_PROTOCOL,grasp_sha256='test')))
    assert len(instances)==1 and len(result['attempts'])==1
    assert result['all_source_action_calls']==16 and result['all_physics_ticks']==560
    assert result['cost_accounting_complete'] and not result['attempts'][0]['success']
    assert (tmp_path/'stop_requested').exists()==schema_fault
    assert bool(result.get('fatal_collection_error'))==schema_fault
    import pickle
    stored=pickle.loads((out/'case__stochastic_1_episodes.pkl').read_bytes())
    assert len(stored)==1 and stored[0]['collection_mode']=='learned-handover'
    assert stored[0]['transitions'][0]['info']['preparation_id']=='case__stochastic_1'
