"""Object balance changes only eligible original-transition sampling in HER."""
import copy
import pickle
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from pnr import object_balanced_her as module


class RewardEnv:
    def __init__(self):self.calls=[]
    def env_method(self,name,achieved,desired,infos,indices):
        assert name=='compute_reward' and indices==[0]
        self.calls.append((achieved.copy(),desired.copy(),copy.deepcopy(infos)))
        failure=np.array([info.get('physical_failure',False) for info in infos])
        result=np.where(failure,-100.,-np.abs(achieved[:,0]-desired[:,0]))
        return [result.astype(np.float32)]


def buffer(objects=(14,5),size=256,n_envs=1,kind=module.ObjectBalancedHerReplayBuffer):
    observation_space=spaces.Dict({'observation':spaces.Box(-np.inf,np.inf,(2,),np.float32),
        'achieved_goal':spaces.Box(-np.inf,np.inf,(1,),np.float32),
        'desired_goal':spaces.Box(-np.inf,np.inf,(1,),np.float32)})
    kwargs=dict(buffer_size=size,observation_space=observation_space,
        action_space=spaces.Box(-1.,1.,(1,),np.float32),env=RewardEnv(),device='cpu',
        n_envs=n_envs,n_sampled_goal=1,goal_selection_strategy='future',copy_info_dict=True)
    if kind is module.ObjectBalancedHerReplayBuffer:kwargs['object_indices']=objects
    return kind(**kwargs)


def add(buf,obj,step,*,done=False,timeout=False,failure=False):
    objs=np.broadcast_to(obj,(buf.n_envs,));steps=np.broadcast_to(step,(buf.n_envs,))
    achieved=np.array([int(o)*100+int(s) for o,s in zip(objs,steps)],dtype=np.float32)[:,None]
    obs={'observation':np.c_[objs,steps].astype(np.float32),'achieved_goal':achieved,
         'desired_goal':np.full((buf.n_envs,1),9999.,np.float32)}
    following={key:value.copy() for key,value in obs.items()};following['achieved_goal']+=1;following['observation'][:,1]+=1
    infos=[{'object_idx':int(o),'physical_failure':bool(failure),'TimeLimit.truncated':bool(timeout),
            'replay_source':'actual-test-transition'} for o in objs]
    buf.add(obs,following,np.zeros((buf.n_envs,1),np.float32),np.full(buf.n_envs,-5.,np.float32),
            np.full(buf.n_envs,done),infos)


def test_equal_objects_despite_transition_imbalance_and_actual_draw_report():
    b=buffer(objects=(14,5,2,7,10))
    for obj,length in [(14,100),(5,3),(2,4),(7,2),(10,5)]:
        for step in range(length):add(b,obj,step,done=step==length-1,timeout=step==length-1)
    np.random.seed(6)
    indices,envs=b._sample_transition_indices(20000)
    labels=b.object_labels[indices,envs]
    counts=[int(np.sum(labels==obj)) for obj in b.object_indices]
    assert max(abs(n-4000) for n in counts)<250
    report=b.sampling_report()
    assert report['eligible_counts_by_object']=={'14':100,'5':3,'2':4,'7':2,'10':5}
    assert list(report['object_sampling_probabilities'])==['14','5','2','7','10']
    assert all(p==.2 for p in report['object_sampling_probabilities'].values())
    assert report['cumulative_draws']==20000
    assert list(report['cumulative_draw_counts_by_object'].values())==counts
    rng=np.random.get_state();b.sampling_report();after=np.random.get_state()
    assert np.array_equal(rng[1],after[1]) and rng[2:]==after[2:]


def test_missing_object_and_incomplete_episode_are_not_silently_dropped():
    b=buffer()
    add(b,14,0,done=True,timeout=True)
    add(b,5,0)
    assert b.sampling_report()['missing_objects']==[5]
    with pytest.raises(RuntimeError,match='no completed eligible'):b.sample(8)
    add(b,5,1,done=True,timeout=True)
    assert b.sampling_report()['sampling_ready']


def test_invalid_labels_fail_before_replay_mutation_and_episode_objects_cannot_mix():
    b=buffer()
    obs={'observation':np.zeros((1,2),np.float32),'achieved_goal':np.zeros((1,1),np.float32),'desired_goal':np.zeros((1,1),np.float32)}
    for info in [{},{'object_idx':True},{'object_idx':3},{'object_idx':14.}]:
        with pytest.raises(ValueError,match='declared integer'):
            b.add(obs,obs,np.zeros((1,1)),np.zeros(1),np.zeros(1),[info])
        assert b.pos==0 and (b.object_labels==-1).all()
    add(b,14,0)
    with pytest.raises(ValueError,match='inside one physical'):add(b,5,1,done=True)
    assert b.pos==1
    add(b,14,1,done=True,timeout=True)
    add(b,5,0,done=True,timeout=True)
    assert b.pos==3


def test_same_indices_match_stock_her_future_goals_rewards_and_timeouts(monkeypatch):
    balanced=buffer();stock=buffer(kind=HerReplayBuffer)
    assert type(balanced)._sample_goals is HerReplayBuffer._sample_goals
    assert type(balanced)._get_virtual_samples is HerReplayBuffer._get_virtual_samples
    assert type(balanced)._get_real_samples is HerReplayBuffer._get_real_samples
    for b in [balanced,stock]:
        add(b,14,0);add(b,14,1,done=True,failure=True)
        add(b,5,0);add(b,5,1,done=True,timeout=True)
    chosen=np.array([1,3,0,2])
    monkeypatch.setattr(balanced,'_sample_transition_indices',lambda n:(chosen.copy(),np.zeros(n,dtype=int)))
    np.random.seed(20);actual=balanced.sample(4)
    monkeypatch.setattr(np.random,'choice',lambda values,size,replace:chosen.copy())
    np.random.seed(20);expected=stock.sample(4)
    for field in ['actions','dones','rewards']:
        assert torch.equal(getattr(actual,field),getattr(expected,field))
    for field in ['observations','next_observations']:
        for key in getattr(actual,field):assert torch.equal(getattr(actual,field)[key],getattr(expected,field)[key])
    assert actual.dones[:,0].tolist()==[0.,0.,1.,0.]
    assert actual.rewards[2,0].item()==-100.
    assert balanced.env.calls[-1][2][0]['physical_failure']
    assert len(balanced.env.calls[-1][0])==2  # Original0.5 HER ratio.
    assert np.array_equal(balanced.env.calls[-1][1][:,0],[1402.,502.])


def test_future_goals_stay_within_the_selected_object_episode():
    b=buffer()
    for obj in (14,5):
        for step in range(4):add(b,obj,step,done=step==3,timeout=step==3)
    np.random.seed(42);sample=b.sample(256)
    # Native HER returns original-goal samples first, virtual samples second.
    real_goals=sample.observations['desired_goal'][:128].numpy()
    assert (real_goals==9999.).all()
    virtual_obs=sample.observations['observation'][128:].numpy()
    virtual_goals=sample.observations['desired_goal'][128:,0].numpy()
    obj=virtual_obs[:,0];step=virtual_obs[:,1]
    assert (virtual_goals>=obj*100+step+1).all()
    assert (virtual_goals<=obj*100+4).all()


def test_ring_overwrite_excludes_partially_replaced_and_open_episodes():
    b=buffer(size=12)
    for obj in (14,5,14):
        for step in range(4):add(b,obj,step,done=step==3,timeout=step==3)
    assert b.full and b.pos==0
    add(b,5,0);add(b,5,1)
    inds,_=b._sample_transition_indices(500)
    assert not np.isin(inds,[0,1,2,3]).any()
    assert b.sampling_report()['eligible_counts_by_object']=={'14':4,'5':4}
    add(b,5,2,done=True,timeout=True)
    assert b.sampling_report()['eligible_counts_by_object']=={'14':4,'5':7}


def test_vector_workers_have_independent_episode_labels():
    b=buffer(n_envs=2)
    add(b,[14,5],[0,0]);add(b,[14,5],[1,1],done=True,timeout=True)
    add(b,[5,14],[0,0],done=True,timeout=True)
    inds,envs=b._sample_transition_indices(100)
    assert set(b.object_labels[inds,envs])=={14,5}
    assert b.sampling_report()['eligible_counts_by_object']=={'14':3,'5':3}


def test_pickle_preserves_labels_sampling_counters_and_native_env_detachment():
    b=buffer()
    add(b,14,0,done=True,timeout=True);add(b,5,0,done=True,timeout=True)
    b.sample(20);before=b.sampling_report()
    restored=pickle.loads(pickle.dumps(b))
    assert restored.env is None and restored.sampling_report()==before
    restored.set_env(RewardEnv());restored.sample(10)
    assert restored.sampling_report()['cumulative_draws']==30


def test_version_and_info_copy_contract_are_explicit(monkeypatch):
    monkeypatch.setattr(module.stable_baselines3,'__version__','2.8.0')
    with pytest.raises(RuntimeError,match='2.7.0'):buffer()


def test_physical_info_copy_and_declared_object_contract():
    stock=buffer(kind=HerReplayBuffer)
    with pytest.raises(ValueError,match='copy_info_dict=True'):
        module.ObjectBalancedHerReplayBuffer(buffer_size=32,observation_space=stock.observation_space,
            action_space=stock.action_space,env=RewardEnv(),object_indices=[14,5],copy_info_dict=False)
    for objects in [(),(14,14),(True,5),(14,99),(14.,5)]:
        with pytest.raises(ValueError,match='distinct integer'):buffer(objects=objects)
