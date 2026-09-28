"""Initialization invariants on actual SB3 SAC+HER, without robot simulation."""
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.sac_initialization import calibrate_actor_std, critic_only_warmup


class GoalEnv(gym.Env):
    observation_space = spaces.Dict({
        'observation': spaces.Box(-10,10,(3,),np.float32),
        'achieved_goal': spaces.Box(-10,10,(2,),np.float32),
        'desired_goal': spaces.Box(-10,10,(2,),np.float32)})
    action_space = spaces.Box(-1,1,(2,),np.float32)
    def reset(self, *, seed=None, options=None):
        return {key:np.zeros(space.shape,np.float32) for key,space in self.observation_space.spaces.items()},{}
    def step(self, action):
        raise AssertionError('This initialization test cannot generate physics')
    def compute_reward(self, achieved_goal, desired_goal, info):
        return -(np.linalg.norm(achieved_goal-desired_goal,axis=-1)>.1).astype(np.float32)


class LearnedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space,features_dim=12)
        self.layers=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.BatchNorm1d(8),torch.nn.ReLU())
    def forward(self, observation):
        return torch.cat([self.layers(observation['observation']),observation['achieved_goal'],observation['desired_goal']],dim=1)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    env=DummyVecEnv([GoalEnv])
    model=SAC('MultiInputPolicy',env,buffer_size=128,learning_starts=0,batch_size=16,
        tau=.2,gamma=.91,ent_coef='auto_0.007',target_update_interval=2,
        replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs={'copy_info_dict':True},
        policy_kwargs={'net_arch':[16,16],'features_extractor_class':LearnedExtractor},
        seed=141,device='cpu')
    rng=np.random.default_rng(22)
    for i in range(64):
        obs={key:rng.normal(size=(1,*space.shape)).astype(np.float32) for key,space in env.observation_space.spaces.items()}
        after={key:value.copy() for key,value in obs.items()}
        action=rng.uniform(-1,1,(1,2)).astype(np.float32)
        after['achieved_goal']+=action*.1
        reward=env.env_method('compute_reward',after['achieved_goal'],obs['desired_goal'],[{}])[0]
        model.replay_buffer.add(obs,after,action,reward,np.array([(i+1)%8==0]),[{'TimeLimit.truncated':True}])
    yield model
    env.close()


def states_equal(left,right):
    assert left.keys()==right.keys()
    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)


def observed(model,n=16):
    return {key:torch.randn(n,*space.shape) for key,space in model.observation_space.spaces.items()}


def test_calibration_preserves_mean_and_critics_despite_changed_bc_backbone(model):
    model.policy.set_training_mode(False)
    data=observed(model)
    # Populate optimizer moments, then mimic a BC backbone change that alters
    # stochastic outputs despite leaving the original std head weights alone.
    loss=model.actor(data,deterministic=True).square().mean()+model.actor.get_action_dist_params(data)[1].mean()
    model.actor.optimizer.zero_grad();loss.backward();model.actor.optimizer.step()
    with torch.no_grad():
        for parameter in model.actor.latent_pi.parameters():parameter.add_(torch.randn_like(parameter)*2.)
    before=model.actor(data,deterministic=True).detach().clone()
    actor_state=copy.deepcopy(model.actor.state_dict())
    critic_state=copy.deepcopy(model.critic.state_dict())
    target_state=copy.deepcopy(model.critic_target.state_dict())
    latent_parameter=next(model.actor.latent_pi.parameters())
    latent_moments=copy.deepcopy(model.actor.optimizer.state[latent_parameter])
    report=calibrate_actor_std(model,-3.)
    assert report['optimizer_parameter_states_cleared']==2
    torch.testing.assert_close(before,model.actor(data,deterministic=True),rtol=0,atol=0)
    for name,value in model.actor.state_dict().items():
        if not name.startswith('log_std.'):
            torch.testing.assert_close(value,actor_state[name],rtol=0,atol=0)
    for sample in [data,observed(model,31)]:
        torch.testing.assert_close(model.actor.get_action_dist_params(sample)[1],torch.full((len(sample['observation']),2),-3.))
    assert all(parameter not in model.actor.optimizer.state for parameter in model.actor.log_std.parameters())
    states_equal(latent_moments,model.actor.optimizer.state[latent_parameter])
    states_equal(critic_state,model.critic.state_dict());states_equal(target_state,model.critic_target.state_dict())


@pytest.mark.parametrize('fixed_alpha',[False,True])
def test_warmup_changes_critics_and_targets_but_no_actor_or_temperature_state(model,fixed_alpha):
    calibrate_actor_std(model,-3.)
    if fixed_alpha:
        model.ent_coef_optimizer=None;model.ent_coef_tensor=torch.tensor(.002)
    actor=copy.deepcopy(model.actor.state_dict())
    critic=copy.deepcopy(model.critic.state_dict())
    target=copy.deepcopy(model.critic_target.state_dict())
    alpha=model.log_ent_coef.detach().clone()
    counters=(model.num_timesteps,model._n_updates)
    modes={module:module.training for network in (model.actor,model.critic,model.critic_target) for module in network.modules()}
    old_lr=model.critic.optimizer.param_groups[0]['lr']
    records=[]
    report=critic_only_warmup(model,12,learning_rate=1e-3,log_every=5,on_log=records.append)
    assert report['critic_updates']==12 and report['actor_updates']==0 and report['temperature_updates']==0
    assert report['target_updates']==6 and report['examples_sampled']==192
    assert report['alpha']==pytest.approx(.002 if fixed_alpha else .007)
    assert report['gamma']==pytest.approx(.91)
    assert len(records)==4
    states_equal(actor,model.actor.state_dict())
    torch.testing.assert_close(alpha,model.log_ent_coef,rtol=0,atol=0)
    assert any(not torch.equal(value,model.critic.state_dict()[name]) for name,value in critic.items())
    assert any(not torch.equal(value,model.critic_target.state_dict()[name]) for name,value in target.items())
    assert counters==(model.num_timesteps,model._n_updates)
    assert model._pnr_critic_warmup_updates==12
    assert old_lr==model.critic.optimizer.param_groups[0]['lr']
    assert all(module.training==mode for module,mode in modes.items())
    # Actor's learned BatchNorm buffers are included in exact state comparison.
    assert model.actor.features_extractor.layers[1].num_batches_tracked==actor['features_extractor.layers.1.num_batches_tracked']


def test_warmup_restores_modes_and_learning_rate_when_observer_interrupts(model):
    modes={module:module.training for network in (model.actor,model.critic,model.critic_target) for module in network.modules()}
    actor=copy.deepcopy(model.actor.state_dict());rate=model.critic.optimizer.param_groups[0]['lr']
    def interrupted(record):raise RuntimeError('bounded interruption')
    with pytest.raises(RuntimeError,match='bounded interruption'):
        critic_only_warmup(model,5,learning_rate=1e-2,on_log=interrupted)
    assert model._pnr_critic_warmup_updates==1
    assert model.critic.optimizer.param_groups[0]['lr']==rate
    assert all(module.training==mode for module,mode in modes.items())
    states_equal(actor,model.actor.state_dict())


def test_warmup_rejects_shared_actor_feature_tensors(model):
    model.critic.features_extractor=model.actor.features_extractor
    with pytest.raises(ValueError,match='separate actor and critic'):
        critic_only_warmup(model,5)


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),-21.,3.])
def test_calibration_rejects_invalid_std_without_mutation(model,invalid):
    actor=copy.deepcopy(model.actor.state_dict())
    with pytest.raises(ValueError,match='log_std'):
        calibrate_actor_std(model,invalid)
    states_equal(actor,model.actor.state_dict())


def test_warmup_tau_override_changes_only_initialization_target_update(model):
    actor=copy.deepcopy(model.actor.state_dict())
    alpha=model.log_ent_coef.detach().clone()
    target_before={name:value.detach().clone() for name,value in model.critic_target.named_parameters()}
    online_tau=model.tau
    records=[]
    report=critic_only_warmup(model,1,warmup_tau=.8,on_log=records.append)
    assert model.tau==online_tau==.2
    assert report['warmup_tau']==.8 and report['online_tau']==.2
    assert records[0]['warmup_tau']==.8 and records[0]['online_tau']==.2
    for (name,target),(critic_name,critic) in zip(model.critic_target.named_parameters(),model.critic.named_parameters()):
        assert name==critic_name
        torch.testing.assert_close(target,.2*target_before[name]+.8*critic,rtol=1e-6,atol=1e-7)
    states_equal(actor,model.actor.state_dict())
    torch.testing.assert_close(alpha,model.log_ent_coef,rtol=0,atol=0)
    assert not model.ent_coef_optimizer.state


def test_warmup_tau_override_leaves_online_tau_and_actor_unchanged_on_failure(model):
    actor=copy.deepcopy(model.actor.state_dict());alpha=model.log_ent_coef.detach().clone()
    online_tau=model.tau;rate=model.critic.optimizer.param_groups[0]['lr']
    def interrupted(record):raise RuntimeError('bounded override interruption')
    with pytest.raises(RuntimeError,match='bounded override interruption'):
        critic_only_warmup(model,5,learning_rate=1e-2,warmup_tau=.8,on_log=interrupted)
    assert model.tau==online_tau==.2
    assert model.critic.optimizer.param_groups[0]['lr']==rate
    states_equal(actor,model.actor.state_dict())
    torch.testing.assert_close(alpha,model.log_ent_coef,rtol=0,atol=0)
    assert not model.ent_coef_optimizer.state


@pytest.mark.parametrize('invalid',[0.,-.1,1.01,float('nan'),float('inf'),True])
def test_warmup_rejects_invalid_target_rate_before_any_update(model,invalid):
    critic=copy.deepcopy(model.critic.state_dict());online_tau=model.tau
    with pytest.raises(ValueError,match='warmup_tau'):
        critic_only_warmup(model,5,warmup_tau=invalid)
    states_equal(critic,model.critic.state_dict())
    assert model.tau==online_tau and not hasattr(model,'_pnr_critic_warmup_updates')
