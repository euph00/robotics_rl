"""Synthetic goal-conditioned environment for learner/replay tests; never steps physics."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SyntheticProfileEnv(gym.Env):
    """Shape/reward contract only; observations do not represent robot states."""
    def __init__(self):
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, (50,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)})

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(np.asarray(achieved_goal)-np.asarray(desired_goal), axis=-1)
        if isinstance(info, dict):
            retained = bool(info.get('bilateral_contact', False))
        else:
            retained = np.array([bool(item.get('bilateral_contact', False)) for item in info])
        return -np.asarray((distance >= .04) | ~np.asarray(retained), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {key: np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()}, {}

    def step(self, action):
        raise RuntimeError('Synthetic test environment must not generate environment interactions')
