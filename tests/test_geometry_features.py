"""Functional invariance of actor/critic transfer, including HER relabeling."""
import unittest
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC
from pnr.features import RelativeGoalExtractor
from pnr.geometry_features import MeasuredGeometryExtractor,transfer_expanded_geometry_policy


class SpacesOnly(gym.Env):
    observation_space=spaces.Dict({
        'observation':spaces.Box(-np.inf,np.inf,(50,),dtype=np.float32),
        'achieved_goal':spaces.Box(-np.inf,np.inf,(3,),dtype=np.float32),
        'desired_goal':spaces.Box(-np.inf,np.inf,(3,),dtype=np.float32)})
    action_space=spaces.Box(-1.,1.,(4,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        return {key:np.zeros(space.shape,dtype=np.float32) for key,space in self.observation_space.spaces.items()},{}


class GeometryFeatureTests(unittest.TestCase):
    def test_zero_feature_expansion_preserves_actor_and_both_critics_after_her(self):
        torch.set_num_threads(1)
        def model(cls):
            return SAC('MultiInputPolicy',SpacesOnly(),buffer_size=32,seed=7,device='cpu',
                       policy_kwargs={'net_arch':[32,32],'features_extractor_class':cls,'features_extractor_kwargs':{'goal_scale':5.}})
        source=model(RelativeGoalExtractor);target=model(MeasuredGeometryExtractor)
        report=transfer_expanded_geometry_policy(source.policy,target.policy)
        self.assertEqual(len(report['expanded_tensors']),5)
        obs={key:torch.randn((8,*space.shape)) for key,space in source.observation_space.spaces.items()}
        action=torch.randn((8,4)).tanh()
        for relabel in (False,True):
            if relabel:obs['desired_goal']=torch.randn((8,3))
            with torch.no_grad():
                torch.testing.assert_close(source.actor(obs,deterministic=True),target.actor(obs,deterministic=True),atol=1e-6,rtol=1e-6)
                for left,right in zip(source.critic(obs,action),target.critic(obs,action)):
                    torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
                for left,right in zip(source.critic_target(obs,action),target.critic_target(obs,action)):
                    torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
        initial=target.actor.features_extractor(obs)
        obs['desired_goal']=obs['desired_goal']+1.
        following=target.actor.features_extractor(obs)
        torch.testing.assert_close(initial[:,-3:],following[:,-3:])
        torch.testing.assert_close(following[:,-3:],20*(obs['observation'][:,28:31]-obs['observation'][:,35:38]))
        torch.testing.assert_close(following[:,-6:-3]-initial[:,-6:-3],torch.ones((8,3))*5.,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
