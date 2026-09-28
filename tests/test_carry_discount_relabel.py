"""Discount ablation relabels real rewards only; native target HER stays intact."""
import copy
import hashlib
import pickle

import numpy as np
import pytest
from stable_baselines3 import SAC,HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.demonstrations import seed_replay
from pnr.multi_carry import CarrySchema


def dataset():
    source=CarrySchema(gamma=.995,shaping_weight=2.)
    episodes=[]
    for ep,length in enumerate((3,5)):
        rows=[]
        for step in range(length):
            obs={'observation':np.full(580,ep+step/10,np.float32),
                 'achieved_goal':np.array([ep+step*.05,0.],np.float32),
                 'desired_goal':np.array([ep+.25,0.],np.float32)}
            nxt={k:v.copy() for k,v in obs.items()};nxt['achieved_goal'][0]+=np.float32(.05)
            nxt['observation'][:]=ep+(step+1)/10
            failure=ep==0 and step==length-1
            info={'previous_achieved_goal':obs['achieved_goal'].copy(),'bilateral_contact':not failure,
                  'retention_ok':not failure,'physical_failure':failure,
                  'action_calls_delta':5,'physics_ticks_delta':25,'case_id':f'train_{ep:03d}',
                  'preparation_id':f'train_{ep:03d}__deterministic','preserved':{'value':ep}}
            rows.append({'observation':obs,'next_observation':nxt,'action':np.array([.5,-.1],np.float32),
                         'reward':float(source.compute_reward(nxt['achieved_goal'],obs['desired_goal'],info)),
                         'terminated':failure,'truncated':ep==1 and step==length-1,'info':info})
        episodes.append({'object_idx':14 if ep==0 else 2,'collection_mode':'learned-handover','transitions':rows})
    return {'schema_version':1,'metadata':{'partition':'train','gamma':.995,'shaping_weight':2.,'delta_scale':.075,'repeat':5},'episodes':episodes}


def model(gamma=.999,environment_gamma=None,weight=2.,failure_cost=None):
    def factory():
        result=CarrySchema(gamma=gamma if environment_gamma is None else environment_gamma,shaping_weight=weight)
        if failure_cost is not None:result.reward_proxy.failure_cost=failure_cost
        return result
    vector=DummyVecEnv([factory,factory])
    result=SAC('MultiInputPolicy',vector,gamma=gamma,buffer_size=100,
        replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs={'copy_info_dict':True,'n_sampled_goal':1},
        policy_kwargs={'net_arch':[8]},device='cpu',seed=1)
    return result


def test_source_verified_target_real_and_native_her_rewards_preserve_boundaries(tmp_path):
    data=dataset();path=tmp_path/'demos.pkl';original=pickle.dumps(data);path.write_bytes(original)
    control=model(gamma=.995);target=model()
    old=seed_replay(control,path,object_labels=True)
    report=seed_replay(target,path,object_labels=True,reward_relabel_gamma=.999)
    assert path.read_bytes()==original
    assert 'reward_relabeling' not in old
    assert {k:v for k,v in report.items() if k!='reward_relabeling'}==old
    proof=report['reward_relabeling']
    assert proof['source_gamma']==.995 and proof['target_gamma']==.999
    assert proof['source_failure_cost']==pytest.approx(200.) and proof['target_failure_cost']==pytest.approx(1000.)
    assert proof['source_transitions_verified']==8 and proof['real_rewards_recomputed']==6
    assert proof['changed_real_rewards']==6 and proof['new_physical_interactions']==0
    assert proof['source_dataset_sha256']==hashlib.sha256(original).hexdigest()
    before,after=control.replay_buffer,target.replay_buffer
    for key in before.observations:
        np.testing.assert_array_equal(before.observations[key][:3],after.observations[key][:3])
        np.testing.assert_array_equal(before.next_observations[key][:3],after.next_observations[key][:3])
    for key in ('actions','dones','timeouts','ep_start','ep_length'):
        np.testing.assert_array_equal(getattr(before,key)[:3],getattr(after,key)[:3])
    for i in range(3):
        for env in range(2):
            assert pickle.dumps(before.infos[i,env])==pickle.dumps(after.infos[i,env])
    assert report['collector_timeouts']==1 and after.dones[2].tolist()==[1.,1.]
    assert after.timeouts[2].tolist()==[0.,1.]  # Physical failure vs cropped ongoing episode.
    np.testing.assert_allclose(after.rewards[2,0]-before.rewards[2,0],-800.,atol=1e-5)
    delta=after.rewards[:3].astype(float)-before.rewards[:3].astype(float)
    assert proof['reward_delta_sum']==pytest.approx(delta.sum())
    assert proof['reward_delta_mean_abs']==pytest.approx(np.abs(delta).mean())
    # Stock HER recomputes changed future goals from the active .999 environment;
    # failure stays absorbing and collector timeouts still bootstrap.
    indices=np.array([0,2,0,2]);envs=np.array([0,0,1,1])
    sample=after._get_virtual_samples(indices,envs)
    infos=[after.infos[i,e] for i,e in zip(indices,envs)]
    achieved=sample.next_observations['achieved_goal'].numpy()
    goals=sample.observations['desired_goal'].numpy()
    expected=target.env.env_method('compute_reward',achieved,goals,infos,indices=[0])[0]
    source=CarrySchema(gamma=.995).compute_reward(achieved,goals,infos)
    np.testing.assert_array_equal(sample.rewards.numpy().ravel(),expected)
    assert sample.dones.numpy().ravel().tolist()==[0.,1.,0.,0.]
    assert float(expected[1]-source[1])==pytest.approx(-800.)
    assert type(after)._get_virtual_samples is HerReplayBuffer._get_virtual_samples
    control.env.close();target.env.close()


@pytest.mark.parametrize('mutation,match',[
    (lambda d:d['metadata'].update(gamma=.999),'original TRAIN'),
    (lambda d:d['metadata'].update(shaping_weight=3.),'original TRAIN'),
    (lambda d:d['episodes'][1]['transitions'][-1].update(reward=123.),'Stored source'),
    (lambda d:d['episodes'][1]['transitions'][-1]['info'].pop('previous_achieved_goal'),'previous physical'),
])
def test_bad_source_even_in_omitted_tail_rejected_before_any_insertion(tmp_path,mutation,match):
    data=dataset();mutation(data);path=tmp_path/'bad.pkl';path.write_bytes(pickle.dumps(data));target=model()
    with pytest.raises(ValueError,match=match):seed_replay(target,path,reward_relabel_gamma=.999)
    assert target.replay_buffer.pos==0
    target.env.close()


@pytest.mark.parametrize('kwargs',[{'environment_gamma':.995},{'weight':3.},{'failure_cost':200.}])
def test_target_contract_mismatch_rejected_before_insertion(tmp_path,kwargs):
    path=tmp_path/'demos.pkl';path.write_bytes(pickle.dumps(dataset()));target=model(**kwargs)
    with pytest.raises(ValueError,match='Active target reward'):seed_replay(target,path,reward_relabel_gamma=.999)
    assert target.replay_buffer.pos==0
    target.env.close()


def test_gamma_change_requires_explicit_option_and_matching_learner(tmp_path):
    path=tmp_path/'demos.pkl';path.write_bytes(pickle.dumps(dataset()))
    changed=model()
    with pytest.raises(ValueError,match='active environment'):seed_replay(changed,path)
    assert changed.replay_buffer.pos==0
    original=model(gamma=.995)
    with pytest.raises(ValueError,match='matching the learner'):seed_replay(original,path,reward_relabel_gamma=.999)
    assert original.replay_buffer.pos==0
    changed.env.close();original.env.close()
