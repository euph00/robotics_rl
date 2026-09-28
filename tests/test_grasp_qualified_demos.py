"""No-physics tests for actual-boundary recording and downstream BC qualification."""
from contextlib import nullcontext
from copy import deepcopy
import json
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import phase2_grasp_qualified_demos as collector


def observation(value=0.):
    return {'observation': np.full(54, value, dtype=np.float32),
            'achieved_goal': np.array([0., 0., value], dtype=np.float32),
            'desired_goal': np.array([0., 0., .24], dtype=np.float32)}


class MeasuredGrasp:
    action_space = SimpleNamespace(shape=(4,))
    observation_space = {'observation': SimpleNamespace(shape=(54,))}
    def __init__(self, native=None, failure=False):
        self.native = native or SimpleNamespace(step_count=100, physics_steps=500)
        self.state = observation(); self.failure = failure; self.actions = []; self.returns = []
    def _observation(self):
        return self.state
    def step(self, action):
        self.actions.append(action)
        self.native.step_count += 3; self.native.physics_steps += 15
        self.state = observation(.24)
        info = {'bilateral_contact': not self.failure, 'physical_failure': self.failure,
                'is_success': not self.failure, 'action_calls_delta': 3, 'physics_ticks_delta': 15}
        result = (self.state, -50. if self.failure else 0., self.failure, False, info)
        self.returns.append(result)
        return result


def test_exact_tuple_action_identity_and_persistence_before_any_caller_motion():
    physical = MeasuredGrasp(); saved = []
    def persist(rows):
        # This executes inside step, at the true grasp boundary.
        assert physical.native.step_count == 103
        saved.append(deepcopy(rows))
    wrapped = collector.RecordingGrasp(physical, persist)
    obs = wrapped._observation(); prior = deepcopy(obs)
    action = np.array([.1, .2, -.3, -1.], np.float32)
    result = wrapped.step(action)
    assert result is physical.returns[0] and physical.actions == [action]
    assert physical.actions[0] is action and len(saved) == 1
    row = saved[0][0]
    assert collector.validate_prefix(saved[0]) == {'decisions': 1, 'source_action_calls': 3, 'physics_ticks': 15}
    for key in prior:
        np.testing.assert_array_equal(row['observation'][key], prior[key])
    assert row['info'] == result[-1] and row['reward'] == 0.
    # Caller/native mutation after the returned step cannot alter saved data.
    obs['observation'][:] = 9.; action[:] = 0.; result[0]['observation'][:] = 8.
    result[-1]['physical_failure'] = True; physical.native.step_count += 77
    assert np.all(row['observation']['observation'] == 0.)
    assert np.all(row['next_observation']['observation'] == np.float32(.24))
    assert row['action'][3] == -1. and not row['info']['physical_failure']
    boundary = wrapped.seal('caller_handover_prefix_end')
    assert not boundary['directly_replay_seedable'] and not boundary['flags_rewritten']
    assert not row['terminated'] and not row['truncated']
    with pytest.raises(ValueError, match='sealed'):
        wrapped.step(np.zeros(4, np.float32))


def test_native_failure_and_her_reward_are_preserved_not_replaced_by_qualification():
    physical = MeasuredGrasp(failure=True); wrapped = collector.RecordingGrasp(physical, lambda rows: None)
    wrapped._observation(); returned = wrapped.step(np.zeros(4, np.float32))
    assert returned[1:4] == (-50., True, False)
    row = wrapped.rows[0]
    assert collector.GraspSchema().compute_reward(row['next_observation']['achieved_goal'],
        row['next_observation']['achieved_goal'], row['info']) == -50.
    assert wrapped.seal('caller_boundary')['kind'] == 'native_termination'
    good = MeasuredGrasp(); recorded = collector.RecordingGrasp(good, lambda rows: None)
    recorded._observation(); recorded.step(np.zeros(4, np.float32)); before = pickle.dumps(recorded.rows)
    assert not collector.qualification(True, False, False, wall_contacts=0, table_contacts=0)
    assert pickle.dumps(recorded.rows) == before
    assert recorded.rows[0]['reward'] == 0. and not recorded.rows[0]['terminated']


def test_malformed_actual_data_is_persisted_then_rejected():
    physical = MeasuredGrasp(); saved = []
    original = physical.step
    def wrong(action):
        obs, reward, term, trunc, info = original(action)
        return obs, 37., term, trunc, info
    physical.step = wrong
    wrapped = collector.RecordingGrasp(physical, lambda rows: saved.append(deepcopy(rows)))
    wrapped._observation()
    with pytest.raises(ValueError, match='reward'):
        wrapped.step(np.zeros(4, np.float32))
    assert len(saved) == 1 and saved[0][0]['reward'] == 37.
    assert physical.native.step_count == 103


def test_prefix_continuity_and_no_counter_or_goal_fabrication():
    physical = MeasuredGrasp(); wrapped = collector.RecordingGrasp(physical, lambda rows: None)
    wrapped._observation()
    for _ in range(2):wrapped.step(np.zeros(4, np.float32))
    assert collector.validate_prefix(wrapped.rows)['source_action_calls'] == 6
    corrupt = deepcopy(wrapped.rows); corrupt[1]['native_call_before'] += 1
    with pytest.raises(ValueError, match='Unrecorded'):
        collector.validate_prefix(corrupt)
    corrupt = deepcopy(wrapped.rows); corrupt[0]['next_observation']['desired_goal'][0] += .1
    with pytest.raises(ValueError, match='goal changed'):
        collector.validate_prefix(corrupt)


def test_plan_declares120_independent_slots_and20_development_comparisons(monkeypatch,tmp_path):
    cases = [dict(case_id=f'train_{layout:03d}_object_{obj:02d}', layout_id=f'train_{layout:03d}',
                  partition='train', object_idx=obj) for layout in range(12) for obj in collector.OBJECTS]
    monkeypatch.setattr(collector, 'select_cases', lambda *a: deepcopy(cases))
    monkeypatch.setattr(collector, 'PickNavReachEnv', lambda **kw: pytest.fail('Planning must not construct physics'))
    plan = collector.make_plan({}, {'sha256':'f'*64}, {'sha256':'d'*64}, root=tmp_path)
    schedule = plan['schedule']; comparison = plan['development_comparison']
    assert len(schedule) == len({r['preparation_id'] for r in schedule}) == 120
    assert len({r['case_id'] for r in schedule}) == 40
    assert [r['preparation_variant']['torch_seed'] for r in schedule] == list(range(9401, 9521))
    assert len(comparison['cases']) == 20
    assert not ({r['case_id'] for r in schedule} & {r['case_id'] for r in comparison['cases']})
    assert 'ancestor' in comparison['ancestry'] and not comparison['full_task_evaluation']
    bc = plan['bc_refinement']
    assert bc['updates'] == 2000 and bc['supervised_examples'] == 800000
    assert bc['old_examples_per_update'] == bc['qualified_examples_per_update'] == 200
    assert bc['online_sac_updates'] == 0


def test_conditional_launch_rejects_missing_review_incomplete_or_improved_core():
    plan = {'plan_sha256':'p'}; review = {'approved':True,'scope':'collection_only','plan_sha256':'p'}
    control = dict(partition='validation',planned_cases=60,completed_cases=60,cost_accounting_complete=True,
                   successes=51,successes_within_deadline_including_initialization=47,missing_case_ids=[])
    weak = dict(control, successes=49, successes_within_deadline_including_initialization=44)
    collector.verify_gate(plan, review, weak, control)
    collector.verify_gate(plan, review, control, control)
    for bad_review in ({},dict(review,plan_sha256='other'),dict(review,scope='training')):
        with pytest.raises(ValueError, match='review'):collector.verify_gate(plan,bad_review,weak,control)
    with pytest.raises(ValueError, match='complete'):collector.verify_gate(plan,review,dict(weak,completed_cases=59),control)
    for improved in [dict(control,successes=52),dict(control,successes_within_deadline_including_initialization=48)]:
        with pytest.raises(ValueError, match='beats'):collector.verify_gate(plan,review,improved,control)


@pytest.mark.parametrize('route_success,hold_failure', [(False,False),(True,False),(True,True)])
def test_worker_records_grasp_before_posture_and_preserves_later_failures(monkeypatch,tmp_path,route_success,hold_failure):
    out=tmp_path/'artifacts';out.mkdir()
    case=collector.preparation_schedule([dict(case_id='train_000_object_14',layout_id='train_000',
        partition='train',object_idx=14,maze_hash='m',env_kwargs={})],3,9401)[0]
    made=[];active=[];prefix_before=[]
    class Native:
        settle_steps=480;substeps=5;drop_events=0
        def __init__(self,**kwargs):self.step_count=0;self.physics_steps=0;made.append(self)
        def scene_manifest(self):return {'maze_hash':'m'}
        def observe(self):return {}
        def evaluate(self):return {'bilateral_contact':True,'drop_events':0}
        def close(self):pass
        def restore_state(self,*a,**k):pytest.fail('No snapshot restoration allowed')
    class System:
        stage='grasp';stage_results={};wall_contact_calls=0;robot_table_contact_calls=0
        def __init__(self,native,predictor,on_step):
            self.native=native;self.on_step=on_step;self.calls_by_source={'learned':0,'planned':0,'teacher':0,'hold':0}
            self.skill_factories={};self.controller=SimpleNamespace(stage_results={});self.stage_results={};active.append(self)
        def tick(self,source,calls):
            for _ in range(calls):
                self.native.step_count+=1;self.native.physics_steps+=5;self.calls_by_source[source]+=1
                self.on_step(self,{}, {})
        def prepare(self):
            env=self.skill_factories['grasp'](self.native,self.controller,600)
            obs=env._observation();env.step(np.zeros(4,np.float32))
            p=out/(case['preparation_id']+'_grasp.pkl');prefix_before.append(p.read_bytes())
            saved=pickle.loads(prefix_before[-1]);assert len(saved['transitions'])==1
            assert saved['transitions'][0]['native_call_after']==3
            self.tick('planned',7)
            return {'prepared_handover':True,'stages':{}}
        def _observe_native_steps(self):return nullcontext()
        def _begin(self,stage,source):self.stage=stage;return self.native.step_count
        def _finish(self,stage,start,success,**kwargs):self.stage_results[stage]={'success':success}
        def _held_pause(self,stage,calls):
            self.stage=stage;self.tick('hold',calls)
            if hold_failure:raise collector.StageFailure('hold failed')
    def measured(n,controller,horizon,goal_lift_m):
        env=MeasuredGrasp(n)
        def step(action):
            active[0].tick('learned',3);env.state=observation(.24)
            return env.state,0.,False,False,{'is_success':True,'bilateral_contact':True,
                'physical_failure':False,'action_calls_delta':3,'physics_ticks_delta':15}
        env.step=step;return env
    def route(env,obs,rng,noise,costs,stop):
        assert noise==0.;active[0].tick('teacher',5);costs.update(source_action_calls=5,physics_ticks=25)
        return [{'actual':'teacher carry only'}],route_success,{'is_success':route_success,
            'bilateral_contact':route_success,'failure_reason':None if route_success else 'lost_grasp'},False
    monkeypatch.setattr(collector.os,'dup2',lambda *a:None)
    monkeypatch.setattr(collector,'PickNavReachEnv',Native)
    monkeypatch.setattr(collector,'LearnedCarryPreparation',System)
    monkeypatch.setattr(collector.GeneralGraspEnv,'bind',measured)
    monkeypatch.setattr(collector.GeneralCarryEnv,'bind',lambda *a,**k:SimpleNamespace(_observation=lambda:{}))
    monkeypatch.setattr(collector,'predictor_for_variant',lambda *a:SimpleNamespace(sampling_report=lambda:{'predictions':1}))
    monkeypatch.setattr(collector,'contact_record',lambda *a:{'measured':True})
    monkeypatch.setattr(collector,'collect_episode',route)
    result=collector.worker((case,str(out),dict(grasp_lift_m=.24,carry_horizon=1600,delta_scale=.075,final_hold_calls=50)))
    assert len(made)==1 and result['cost_accounting_complete']
    assert result['bc_eligible']==(route_success and not hold_failure)
    assert result['source_action_calls']==15+(50 if route_success else 0)
    assert result['total_physics_ticks_including_initialization']==480+result['source_action_calls']*5
    assert (out/(case['preparation_id']+'_grasp.pkl')).read_bytes()==prefix_before[0]
    stored=pickle.loads(prefix_before[0])['transitions'][0]
    assert stored['reward']==0. and not stored['terminated'] and not stored['truncated']
    assert not stored['info']['physical_failure']
    assert json.loads((out/(case['preparation_id']+'.json')).read_text())['bc_eligible']==result['bc_eligible']
    assert not (tmp_path/'STOP').exists()


def test_failed_preparations_stay_in_summary_and_cost_denominator():
    schedule=collector.preparation_schedule([dict(case_id='train_000_object_14')],3,9401)
    results=[dict(preparation_id=c['preparation_id'],case=c,prepared_handover=False,bc_eligible=False,
        cost_accounting_complete=True,source_action_calls=7,physics_ticks=35,
        initialization_physics_ticks=480,total_physics_ticks_including_initialization=515) for c in schedule]
    s=collector.summarize(results,schedule)
    assert s['planned_preparations']==s['completed_preparations']==3
    assert s['qualified_prefixes']==0 and s['total_physics_ticks_including_initialization']==1545
    assert s['all_failures_retained'] and not s['full_task_evaluation']
    with pytest.raises(ValueError,match='Duplicate'):collector.summarize(results+[results[0]],schedule)


def test_checkpoint_contract_separates_raw_spaces_features_and_rl_ancestry():
    feature = type('GeneralGraspFeatures', (), {'features_dim': 70})
    model = SimpleNamespace(observation_space=SimpleNamespace(spaces={k: SimpleNamespace(shape=(v,))
        for k,v in [('observation',54),('achieved_goal',3),('desired_goal',3)]}),
        action_space=SimpleNamespace(shape=(4,)), actor=SimpleNamespace(features_extractor=feature()),
        critic=SimpleNamespace(features_extractor=feature()), num_timesteps=12000, _n_updates=2000)
    result = collector.checkpoint_contract(model)
    assert result['raw_observation_shapes']['observation']==[54]
    assert result['actor_engineered_feature_width']==result['critic_engineered_feature_width']==70
    assert result['historical_online_decisions']==12000 and result['historical_online_sac_updates']==2000
    assert result['model_predictions_during_inspection']==result['physical_interactions_during_inspection']==0
    model.observation_space.spaces['observation'].shape=(70,)
    with pytest.raises(ValueError,match='raw54'):collector.checkpoint_contract(model)
