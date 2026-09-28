#!/usr/bin/env python3
"""Compute-only SAC+HER CPU/CUDA benchmark using explicitly synthetic replay.

No simulator is constructed and no task policy/checkpoint is saved. Results
measure hardware/software throughput only, never learning or task performance.
Run with project Python: .venv/bin/python scripts/profile_learner.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stable_baselines3
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

ROOT = Path(__file__).resolve().parents[1]


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
        return {key:np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()}, {}

    def step(self, action):
        raise RuntimeError('Compute benchmark must not generate actual environment interactions')


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def fill_synthetic_replay(model, seed=123, transitions=16000, horizon=100):
    rng = np.random.default_rng(seed)
    n_envs = model.n_envs
    if transitions % (n_envs*horizon):
        raise ValueError('Synthetic replay must consist of complete episodes')
    for step in range(transitions//n_envs):
        obs = {key:rng.standard_normal((n_envs, *space.shape)).astype(np.float32)
               for key, space in model.observation_space.spaces.items()}
        next_obs = {key:value.copy() for key, value in obs.items()}
        next_obs['observation'] += rng.normal(0, .01, next_obs['observation'].shape).astype(np.float32)
        next_obs['achieved_goal'] += rng.normal(0, .01, next_obs['achieved_goal'].shape).astype(np.float32)
        action = rng.uniform(-1, 1, (n_envs, 4)).astype(np.float32)
        done = np.full(n_envs, (step+1) % horizon == 0)
        infos = [{'bilateral_contact': bool(value), 'TimeLimit.truncated': bool(done[i]),
                  'action_calls_delta':3, 'physics_ticks_delta':15,
                  'goal_distance':1., 'is_success':False, 'episode_success':False,
                  'curriculum_level':0, 'action_rejected':False}
                 for i, value in enumerate(rng.integers(0, 2, n_envs))]
        reward = model.get_env().env_method('compute_reward', next_obs['achieved_goal'], obs['desired_goal'], infos, indices=[0])[0]
        model.replay_buffer.add(obs, next_obs, action, reward, done, infos)


def profile(device, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vector = DummyVecEnv([SyntheticProfileEnv for _ in range(args.num_envs)])
    model = SAC('MultiInputPolicy', vector, learning_rate=3e-4, buffer_size=100000,
        learning_starts=1000, batch_size=256, tau=.05, gamma=.98, train_freq=1,
        gradient_steps=1, policy_kwargs={'net_arch':[128,128]}, device=device,
        seed=args.seed, replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'n_sampled_goal':4, 'goal_selection_strategy':'future', 'copy_info_dict':True}, verbose=0)
    model.set_logger(configure(format_strings=[]))
    fill_synthetic_replay(model)
    model.train(gradient_steps=args.warmup_updates, batch_size=256)
    synchronize(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train(gradient_steps=args.timed_updates, batch_size=256)
    synchronize(device)
    elapsed = time.perf_counter()-start
    result = {'device':device, 'update_wall_seconds':elapsed,
        'gradient_updates_per_second':args.timed_updates/elapsed,
        'milliseconds_per_update':1000*elapsed/args.timed_updates,
        'timed_gradient_updates':args.timed_updates, 'warmup_gradient_updates':args.warmup_updates,
        'updates_include':'SAC actor, both critics, automatic entropy, target updates, HER sample/reward recomputation',
        'inference':{}}
    rng = np.random.default_rng(args.seed)
    for batch in (1, 8, 16):
        obs = {key:rng.standard_normal((batch, *space.shape)).astype(np.float32)
               for key, space in model.observation_space.spaces.items()}
        for _ in range(100):
            model.predict(obs, deterministic=True)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.inference_calls):
            model.predict(obs, deterministic=True)
        synchronize(device)
        elapsed = time.perf_counter()-start
        result['inference'][str(batch)] = {'batch_size':batch, 'calls':args.inference_calls,
            'wall_seconds':elapsed, 'milliseconds_per_call':1000*elapsed/args.inference_calls,
            'calls_per_second':args.inference_calls/elapsed,
            'actions_per_second':args.inference_calls*batch/elapsed,
            'includes':'SB3 dict preprocessing and NumPy actions returned to host'}
    if device == 'cuda':
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    vector.close()
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup-updates', type=int, default=200)
    parser.add_argument('--timed-updates', type=int, default=1000)
    parser.add_argument('--inference-calls', type=int, default=1000)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/setup/artifacts/learner-compute-profile.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {'kind':'synthetic_compute_benchmark', 'task_evidence':False,
        'created_at':datetime.now(timezone.utc).isoformat(), 'hostname':platform.node(),
        'torch':torch.__version__, 'stable_baselines3':stable_baselines3.__version__,
        'numpy':np.__version__, 'torch_threads':torch.get_num_threads(),
        'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'network':{'observation':50, 'achieved_goal':3, 'desired_goal':3,
                   'actions':4, 'hidden_layers':[128,128], 'batch_size':256},
        'synthetic_replay_transitions':16000, 'replay_vector_envs':args.num_envs,
        'caveat':'Random synthetic replay only; no simulator, policy task training, or task success evidence. Concurrent project simulations may affect wall timings.',
        'devices':{}}
    for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        print(f'Profiling {device}: {args.warmup_updates} warmup and {args.timed_updates} timed updates.', flush=True)
        report['devices'][device] = profile(device, args)
        print(json.dumps(report['devices'][device]), flush=True)
    report['fastest_measured_update_device'] = min(report['devices'], key=lambda key:report['devices'][key]['milliseconds_per_update'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
