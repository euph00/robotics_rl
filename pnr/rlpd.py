"""RLPD components for SB3 SAC (Ball et al. 2023, "Efficient Online RL with Offline Data").

- DemoMixReplayBuffer: every batch is half online transitions, half prior-data transitions
  (symmetric sampling); prior observations are normalised with the environment's VecNormalize.
- LayerNormSACPolicy: critics with LayerNorm after every hidden layer, which keeps Q estimates
  bounded when many gradient steps are taken per environment step.
The update-to-data ratio is set through SAC's `gradient_steps`.
"""
from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.sac.policies import SACPolicy


class DemoMixReplayBuffer(ReplayBuffer):
    def __init__(self, *args, demos=None, demo_ratio=.5, **kwargs):
        super().__init__(*args, **kwargs)
        data = np.load(demos)
        self.demo = {key: np.asarray(data[key], dtype=np.float32) for key in
                     ('observations', 'actions', 'rewards', 'next_observations', 'dones')}
        self.demo_ratio = float(demo_ratio)

    def sample(self, batch_size, env=None):
        n_demo = int(round(batch_size * self.demo_ratio))
        online = super().sample(batch_size - n_demo, env=env)
        index = np.random.randint(0, len(self.demo['rewards']), size=n_demo)
        demo = ReplayBufferSamples(
            self.to_torch(self._normalize_obs(self.demo['observations'][index], env)),
            self.to_torch(self.demo['actions'][index]),
            self.to_torch(self._normalize_obs(self.demo['next_observations'][index], env)),
            self.to_torch(self.demo['dones'][index].reshape(-1, 1)),
            self.to_torch(self._normalize_reward(self.demo['rewards'][index].reshape(-1, 1), env)),
            None if online.discounts is None else th.full((n_demo, 1), float(online.discounts[0]), device=self.device))
        return ReplayBufferSamples(*(None if a is None else th.cat([a, b]) for a, b in zip(online, demo)))


class LayerNormCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim,
                 activation_fn=nn.ReLU, normalize_images=True, n_critics=2, share_features_extractor=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn,
                         normalize_images, n_critics, share_features_extractor)
        action_dim = int(np.prod(action_space.shape))
        self.q_networks = []
        for idx in range(n_critics):
            layers, width_in = [], features_dim + action_dim
            for width in net_arch:
                layers += [nn.Linear(width_in, width), nn.LayerNorm(width), activation_fn()]
                width_in = width
            layers.append(nn.Linear(width_in, 1))
            q_net = nn.Sequential(*layers)
            self.add_module(f'qf{idx}', q_net)       # replaces the plain MLP built by the parent
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormCritic(**critic_kwargs).to(self.device)
