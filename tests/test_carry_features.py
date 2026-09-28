"""Compute-only carry representation checks; no robotics performance evidence."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import SAC,HerReplayBuffer
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.features import BoundedGoalExtractor
from pnr.carry_features import CarryWorldGeometryExtractor,transfer_carry_geometry_policy
from pnr.policies import load_policy_checkpoint


class SpaceOnlyEnv(gym.Env):
    observation_space=spaces.Dict({'observation':spaces.Box(-np.inf,np.inf,(580,),np.float32),
        'achieved_goal':spaces.Box(-np.inf,np.inf,(2,),np.float32),
        'desired_goal':spaces.Box(-np.inf,np.inf,(2,),np.float32)})
    action_space=spaces.Box(-1.,1.,(2,),np.float32)
    def reset(self,*,seed=None,options=None):
        return {key:np.zeros(space.shape,np.float32) for key,space in self.observation_space.spaces.items()},{}
    def step(self,action):
        raise AssertionError('Representation checks must not execute transitions')


def batch():
    generator=torch.Generator().manual_seed(171)
    return {key:torch.randn(16,*space.shape,generator=generator)
            for key,space in SpaceOnlyEnv.observation_space.spaces.items()}


def test_world_geometry_is_derived_after_goal_relabeling_without_mutating_raw_state():
    observed=batch();raw=observed['observation'].clone()
    extractor=CarryWorldGeometryExtractor(SpaceOnlyEnv.observation_space)
    extractor.set_all_velocity_mask()
    baseline=BoundedGoalExtractor(SpaceOnlyEnv.observation_space,goal_scale=20.)
    for relabel in (False,True):
        if relabel:observed['desired_goal']=observed['desired_goal'].roll(3,0)+.07
        actual=extractor(observed)
        expected_state=raw.clone();expected_state[:,543:558]=0.;expected_state[:,572:578]=0.
        expected=dict(observed,observation=expected_state)
        torch.testing.assert_close(actual[:,:-4],baseline(expected))
        torch.testing.assert_close(actual[:,-4:-2],observed['achieved_goal']*20.)
        torch.testing.assert_close(actual[:,-2:],observed['desired_goal']*20.)
        torch.testing.assert_close(observed['observation'],raw)


def test_zero_padding_preserves_critic_action_columns_and_saved_actor_only_mask(tmp_path):
    torch.set_num_threads(1)
    vector=DummyVecEnv([SpaceOnlyEnv])
    try:
        def make(extractor):
            return SAC('MultiInputPolicy',vector,buffer_size=32,learning_starts=10,batch_size=8,
                policy_kwargs={'net_arch':[8,8],'features_extractor_class':extractor,
                               'features_extractor_kwargs':{'goal_scale':20.}},
                replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs={'copy_info_dict':True},device='cpu')
        source=make(BoundedGoalExtractor);destination=make(CarryWorldGeometryExtractor)
        transfer_carry_geometry_policy(source.policy,destination.policy)
        observed=batch();actions=torch.linspace(-1.,1.,32).reshape(16,2)
        for relabel in (False,True):
            if relabel:observed['desired_goal']=observed['desired_goal'].roll(2,0)-.03
            with torch.no_grad():
                torch.testing.assert_close(source.actor(observed,deterministic=True),destination.actor(observed,deterministic=True))
                for old,new in zip(source.critic(observed,actions),destination.critic(observed,actions)):
                    torch.testing.assert_close(old,new)
                for old,new in zip(source.critic_target(observed,actions),destination.critic_target(observed,actions)):
                    torch.testing.assert_close(old,new)
        destination.actor.features_extractor.set_all_velocity_mask()
        assert torch.all(destination.critic.features_extractor.state_mask==1.)
        assert torch.all(destination.critic_target.features_extractor.state_mask==1.)
        destination.save(tmp_path/'normal_sac_her.zip')
        metadata,_,_=load_from_zip_file(tmp_path/'normal_sac_her.zip',device='cpu')
        assert metadata['replay_buffer_class'] is HerReplayBuffer
        restored=load_policy_checkpoint(tmp_path/'normal_sac_her.zip')
        mask=restored.actor.features_extractor.state_mask
        assert torch.all(mask[543:558]==0.) and torch.all(mask[572:578]==0.)
        assert int((mask==0.).sum())==21
        assert torch.all(restored.critic.features_extractor.state_mask==1.)
        sample={key:value[0].numpy() for key,value in observed.items()}
        np.testing.assert_allclose(restored.predict(sample,deterministic=True)[0],
                                   destination.predict(sample,deterministic=True)[0],atol=1e-7)
        with torch.no_grad():
            for old,new in zip(source.critic(observed,actions),restored.critic(observed,actions)):
                torch.testing.assert_close(old,new)
    finally:
        vector.close()
