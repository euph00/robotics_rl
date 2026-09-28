"""Source annotation preserves real transitions, resets and accounting APIs."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from pnr.preparation_metadata import LearnedPreparationMetadata


class MeasuredHandover(gym.Env):
    observation_space=spaces.Box(-10,10,(1,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(1,),dtype=np.float32)
    setup_action_calls=17
    def __init__(self):self.resets=0;self.last_info=None
    def reset(self,*,seed=None,options=None):
        self.resets+=1
        return np.array([self.resets],np.float32),{}
    def step(self,action):
        self.last_info={'conditional_on_prepared_handover':True,'object_idx':14,'physics_ticks_delta':25}
        return np.array([7.],np.float32),-1.,False,True,self.last_info
    def prime_cache(self):return {'actual_setup_calls':self.setup_action_calls}


def test_metadata_wrapper_preserves_autoreset_transition_and_accounting():
    native=MeasuredHandover();vector=DummyVecEnv([lambda:LearnedPreparationMetadata(native)])
    try:
        vector.reset();obs,reward,done,infos=vector.step(np.zeros((1,1),np.float32))
        assert obs[0,0]==2 and infos[0]['terminal_observation'][0]==7
        assert reward[0]==-1 and done[0] and infos[0]['TimeLimit.truncated']
        assert infos[0]['preparation_source']=='learned' and infos[0]['physics_ticks_delta']==25
        assert 'preparation_source' not in native.last_info
        assert vector.get_attr('setup_action_calls')==[17]
        assert vector.env_method('prime_cache')==[{'actual_setup_calls':17}]
    finally:vector.close()


def test_metadata_wrapper_rejects_undeclared_preparation(monkeypatch):
    native=MeasuredHandover();wrapper=LearnedPreparationMetadata(native)
    monkeypatch.setattr(native,'step',lambda action:(np.zeros(1,np.float32),0.,False,False,{}))
    with pytest.raises(ValueError,match='declared learned handover'):wrapper.step(np.zeros(1))
    wrapper.close()
