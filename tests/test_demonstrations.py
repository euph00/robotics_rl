"""Replay seeding preserves real observations, episode boundaries and HER rewards."""
import pickle
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.profile_learner import SyntheticProfileEnv
from pnr.demonstrations import seed_replay


def test_real_prefixes_are_imported_without_cross_reset_hindsight(tmp_path):
    vector=DummyVecEnv([SyntheticProfileEnv,SyntheticProfileEnv])
    model=SAC('MultiInputPolicy',vector,buffer_size=100,replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'copy_info_dict':True},policy_kwargs={'net_arch':[8,8]},device='cpu')
    episodes=[]
    for episode,length in enumerate((4,6)):
        transitions=[]
        for step in range(length):
            obs={key:np.zeros(space.shape,dtype=np.float32) for key,space in model.observation_space.spaces.items()}
            obs['achieved_goal'][:]=episode*10+step
            following={key:value.copy() for key,value in obs.items()}
            following['achieved_goal']+=1
            transitions.append({'observation':obs,'next_observation':following,'action':np.zeros(4,dtype=np.float32),
                'reward':-1.,'terminated':False,'truncated':step==length-1,
                'info':{'bilateral_contact':True,'action_calls_delta':3,'physics_ticks_delta':15}})
        episodes.append({'transitions':transitions})
    path=tmp_path/'physical.pkl';path.write_bytes(pickle.dumps({'schema_version':1,'metadata':{},'episodes':episodes}))
    report=seed_replay(model,path)
    assert report['transitions_imported']==8
    assert report['transitions_omitted_from_replay']==2
    assert report['collector_timeouts']==1
    buffer=model.replay_buffer
    np.testing.assert_array_equal(buffer.ep_length[:4],4)
    assert buffer.dones[3].all() and buffer.timeouts[3].all()
    np.testing.assert_array_equal(buffer.next_observations['achieved_goal'][3,:,0],[4,14])
    sample=buffer._get_virtual_samples(np.array([0,1,2,3]),np.array([0,1,0,1]))
    assert not sample.dones.numpy().any()
    expected=vector.env_method('compute_reward',sample.next_observations['achieved_goal'].numpy(),
        sample.observations['desired_goal'].numpy(),[{'bilateral_contact':True}]*4,indices=[0])[0]
    np.testing.assert_array_equal(sample.rewards.numpy().ravel(),expected)
    vector.close()
