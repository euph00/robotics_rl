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
