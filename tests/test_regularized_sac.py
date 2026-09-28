"""Behavioral invariants for the explicit SAC+HER+demo-regularization branch."""
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.regularized_sac import RegularizedSAC


class Goals(gym.Env):
    observation_space = spaces.Dict({
        'observation': spaces.Box(-10, 10, (3,), np.float32),
        'achieved_goal': spaces.Box(-10, 10, (2,), np.float32),
        'desired_goal': spaces.Box(-10, 10, (2,), np.float32)})
    action_space = spaces.Box(-1, 1, (2,), np.float32)
    def reset(self, *, seed=None, options=None):
        return {key: np.zeros(space.shape, np.float32) for key, space in self.observation_space.spaces.items()}, {}
    def step(self, action):
        raise AssertionError('This test must not generate physical interactions')
    def compute_reward(self, achieved_goal, desired_goal, info):
        return -(np.linalg.norm(achieved_goal-desired_goal, axis=-1) > .1).astype(np.float32)


def make(cls, **extra):
    torch.set_num_threads(1)
    model = cls('MultiInputPolicy', DummyVecEnv([Goals]), seed=841, device='cpu', learning_starts=0,
                learning_rate=1e-3, buffer_size=256, batch_size=16, tau=.1,
                ent_coef='auto_0.005', target_entropy=-4., gamma=.97,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs={'n_sampled_goal': 1, 'copy_info_dict': True},
                policy_kwargs={'net_arch': [16, 16]}, **extra)
    model.set_logger(configure(folder=None, format_strings=[]))
    rng = np.random.default_rng(22)
    for index in range(64):
        obs = {key: rng.normal(size=(1, *space.shape)).astype(np.float32)
               for key, space in model.observation_space.spaces.items()}
        after = {key: value+.01 for key, value in obs.items()}
        model.replay_buffer.add(obs, after, rng.uniform(-1, 1, (1, 2)).astype(np.float32),
                               np.array([-1.], np.float32), np.array([(index+1) % 8 == 0]),
                               [{'TimeLimit.truncated': True}])
    return model


def demos():
    rng = np.random.default_rng(83)
    obs = {key: rng.normal(size=(32, *space.shape)).astype(np.float32)
           for key, space in Goals.observation_space.spaces.items()}
    obs['desired_goal'][:] = [3., -4.]
    return obs, np.full((32, 2), .8, np.float32), {'rare': np.array([0]), 'common': np.arange(1, 32)}


def train_seeded(model, updates):
    np.random.seed(211); torch.manual_seed(232)
    model.train(updates, 16)


def equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_zero_weight_is_bitwise_ordinary_sac_with_real_her():
    ordinary, regular = make(SAC), make(RegularizedSAC, bc_weight=0.)
    try:
        train_seeded(ordinary, 5); train_seeded(regular, 5)
        equal(ordinary.policy.state_dict(), regular.policy.state_dict())
        torch.testing.assert_close(ordinary.log_ent_coef, regular.log_ent_coef, rtol=0, atol=0)
        assert ordinary._n_updates == regular._n_updates == 5
        assert regular._regularization_updates == 0
    finally:
        ordinary.env.close(); regular.env.close()


def test_positive_regularization_changes_actor_but_not_first_critic_temperature_or_std_update():
    ordinary, regular = make(SAC), make(RegularizedSAC, bc_weight=100.)
    obs, actions, groups = demos(); regular.set_demonstrations(obs, actions, groups, seed=19)
    try:
        train_seeded(ordinary, 1); train_seeded(regular, 1)
        equal(ordinary.critic.state_dict(), regular.critic.state_dict())
        equal(ordinary.critic_target.state_dict(), regular.critic_target.state_dict())
        equal(ordinary.actor.log_std.state_dict(), regular.actor.log_std.state_dict())
        torch.testing.assert_close(ordinary.log_ent_coef, regular.log_ent_coef, rtol=0, atol=0)
        tensors = {key: torch.tensor(value) for key, value in obs.items()}
        with torch.no_grad():
            plain_error = ((ordinary.actor(tensors, deterministic=True)-torch.tensor(actions))**2).mean()
            regular_error = ((regular.actor(tensors, deterministic=True)-torch.tensor(actions))**2).mean()
        assert regular_error < plain_error
        metrics = regular.last_regularization_metrics
        assert metrics['actor_loss'] == pytest.approx(metrics['actor_rl_loss']+100*metrics['actor_bc_loss'])
        assert metrics['weighted_actor_bc_grad_norm'] == pytest.approx(100*metrics['actor_bc_grad_norm'])
        assert np.isfinite(metrics['rl_bc_grad_cosine']) and -1.0001 <= metrics['rl_bc_grad_cosine'] <= 1.0001
        assert metrics['demo_examples_sampled'] == 16 and regular.num_timesteps == 0
        assert regular.replay_buffer.her_ratio == .5
    finally:
        ordinary.env.close(); regular.env.close()


def test_sampling_balances_rare_groups_keeps_original_goals_and_copies_inputs():
    model = make(RegularizedSAC)
    obs, actions, groups = demos()
    obs['observation'][0, 0] = 9.; obs['observation'][1:, 0] = -9.
    try:
        manifest = model.set_demonstrations(obs, actions, groups, seed=14)
        obs['desired_goal'][:] = 0; actions[:] = 0
        batch, labels = model._sample_demonstrations(10000)
        rare_fraction = float((batch['observation'][:, 0] == 9.).float().mean())
        assert .48 < rare_fraction < .52
        torch.testing.assert_close(batch['desired_goal'], torch.tensor([3., -4.]).expand(10000, 2))
        torch.testing.assert_close(labels, torch.full((10000, 2), .8))
        assert manifest['unique_group_coverage'] == 32
    finally:
        model.env.close()


def test_save_load_requires_matching_reattachment_and_preserves_sampler_rng(tmp_path):
    model = make(RegularizedSAC, bc_weight=17., demo_batch_size=11, gradient_diagnostics_interval=2)
    obs, actions, groups = demos(); model.set_demonstrations(obs, actions, groups, seed=38)
    try:
        train_seeded(model, 3)
        path = tmp_path/'regularized.zip'; model.save(path)
        expected = model._sample_demonstrations(19)
        loaded = RegularizedSAC.load(path, env=model.env, device='cpu')
        assert loaded.bc_weight == 17. and loaded._regularization_updates == 3
        assert loaded._demo_observations is None
        with pytest.raises(RuntimeError, match='set_demonstrations'):
            loaded.train(1, 16)
        bad = actions.copy(); bad[0] = 0
        with pytest.raises(ValueError, match='differ'):
            loaded.set_demonstrations(obs, bad, groups)
        loaded.set_demonstrations(obs, actions, groups)
        actual = loaded._sample_demonstrations(19)
        equal(expected[0], actual[0]); torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
        assert loaded._demo_examples_sampled == 33
    finally:
        model.env.close()


def test_train_fails_before_mutation_without_demos_and_rejects_invalid_data():
    model = make(RegularizedSAC)
    obs, actions, groups = demos(); before = copy.deepcopy(model.policy.state_dict())
    try:
        with pytest.raises(RuntimeError, match='set_demonstrations'):
            model.train(1, 16)
        equal(before, model.policy.state_dict()); assert model._n_updates == 0
        actions[0, 0] = 1.1
        with pytest.raises(ValueError, match='normalized'):
            model.set_demonstrations(obs, actions, groups)
        actions[0, 0] = .8
        with pytest.raises(ValueError, match='outside'):
            model.set_demonstrations(obs, actions, {'bad': np.array([100])})
        with pytest.raises(ValueError, match='probabilities'):
            model.set_demonstrations(obs, actions, groups, group_probabilities=[.3, .3])
    finally:
        model.env.close()
