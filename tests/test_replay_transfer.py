"""Appending physical data cannot join unrelated episodes or substitute labels."""
import pickle
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.profile_learner import SyntheticProfileEnv
from pnr.replay_transfer import append_demonstrations


def test_append_closes_old_tails_and_uses_executed_actions(tmp_path):
    vec = DummyVecEnv([SyntheticProfileEnv,SyntheticProfileEnv])
    model = SAC('MultiInputPolicy',vec,buffer_size=32,replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'copy_info_dict':True},policy_kwargs={'net_arch':[8,8]},device='cpu')
    def obs(value):
        result = {key:np.zeros(space.shape,dtype=np.float32) for key,space in model.observation_space.spaces.items()}
        result['achieved_goal'][:] = value
        return result
    # Actual prior online tails have not ended at the transfer boundary.
    batch = {key:np.stack([value,value]) for key,value in obs(50).items()}
    for _ in range(2):
        model.replay_buffer.add(batch,batch,np.zeros((2,4)),np.full(2,-1.),np.zeros(2,dtype=bool),[{},{}])
    episodes=[]
    for number,length in enumerate((3,4,6)):
        records=[]
        for step in range(length):
            records.append({'observation':obs(number*10+step),'next_observation':obs(number*10+step+1),
                'action':np.full(4,.2,dtype=np.float32),'teacher_action':np.full(4,-.9,dtype=np.float32),
                'reward':-1.,'terminated':False,'truncated':step==length-1,
                'info':{'bilateral_contact':True,'action_calls_delta':3,'physics_ticks_delta':15}})
        episodes.append({'transitions':records})
    path=tmp_path/'physical.pkl';path.write_bytes(pickle.dumps({'schema_version':1,'episodes':episodes}))
    report=append_demonstrations(model,path)
    replay=model.replay_buffer
    assert report['transitions_imported']==12 and report['transitions_omitted_from_replay']==1
    np.testing.assert_array_equal(replay.ep_length[:2],2)
    assert replay.dones[1].all() and replay.timeouts[1].all()
    np.testing.assert_allclose(replay.actions[2:8],.2)
    assert np.all(replay._current_ep_start==replay.pos)
    assert replay.dones[7].all() and replay.timeouts[7].all()
    # Future HER goals must stay within their original physical episode.
    for _ in range(10):
        goals=replay._sample_goals(np.arange(2,8),np.zeros(6,dtype=int))
        assert np.all((goals>=21)&(goals<=26))
    assert report['teacher_labels_used_as_executed_actions'] is False
    vec.close()
