"""Explicit initialization stages before ordinary online SAC+HER updates.

These functions do not add an imitation term to SAC or generate transitions.
Their updates and compute must be accounted for separately from online SAC.
"""
from __future__ import annotations

import math
from typing import Callable
import time

import numpy as np
import torch
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN


def calibrate_actor_std(model, log_std: float = -3.0) -> dict:
    """Make the pre-tanh Gaussian sigma input-independent at initialization.

    BC can change a shared actor backbone while leaving its log_std head weights
    unchanged, so unchanged head weights do not imply unchanged exploration.
    Zeroing this linear head and setting its bias preserves deterministic actions
    and all critic weights. Ordinary SAC may learn the head again afterward.
    Clear only this head's optimizer moments, which describe superseded weights.
    """
    value = float(log_std)
    if not math.isfinite(value) or not LOG_STD_MIN <= value <= LOG_STD_MAX:
        raise ValueError(f'log_std must be finite and within [{LOG_STD_MIN}, {LOG_STD_MAX}]')
    actor = model.actor
    head = actor.log_std
    if getattr(actor, 'use_sde', False) or not isinstance(head, torch.nn.Linear) or head.bias is None:
        raise ValueError('Std calibration requires a biased linear Gaussian log_std head; gSDE is unsupported')
    if not torch.isfinite(head.weight).all() or not torch.isfinite(head.bias).all():
        raise ValueError('Cannot calibrate a nonfinite actor head without diagnosing the model')
    optimizer = actor.optimizer
    head_parameters = list(head.parameters())
    reset_moments = sum(parameter in optimizer.state for parameter in head_parameters)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(value)
    for parameter in head_parameters:
        optimizer.state.pop(parameter, None)
        parameter.grad = None
    return {'method': 'constant_initial_gaussian_std', 'log_std': value,
            'pre_tanh_sigma': math.exp(value), 'optimizer_parameter_states_cleared': reset_moments,
            'deterministic_actor_changed': False, 'critic_changed': False,
            'physical_interactions': 0, 'ongoing_constraint': False}


def critic_only_warmup(model, gradient_steps: int, batch_size: int | None = None,
                       learning_rate: float | None = None, log_every: int = 100,
                       on_log: Callable[[dict], None] | None = None) -> dict:
    """Fit critics with ordinary SAC soft Bellman targets and a frozen actor.

    Uses the model's existing replay (including HER), current alpha and gamma,
    truncation-aware replay dones, n-step discounts when provided, target update
    interval/tau, and critic batch-norm statistics transfer. The actor runs in
    evaluation mode to freeze feature-extractor buffers as well as parameters.
    Actor/critic shared tensors are rejected rather than silently updating the
    actor. No actor or temperature optimizer step occurs.

    Returns explicit initialization counters. model._n_updates continues to mean
    complete online SAC update iterations; cumulative warmup critic steps are
    separately stored as model._pnr_critic_warmup_updates (normal SB3 saveable
    metadata). A temporary critic learning rate is restored even on failure.
    """
    if isinstance(gradient_steps, bool) or int(gradient_steps) != gradient_steps or gradient_steps < 1:
        raise ValueError('gradient_steps must be a positive integer')
    steps = int(gradient_steps)
    size = model.batch_size if batch_size is None else batch_size
    if isinstance(size, bool) or int(size) != size or size < 1:
        raise ValueError('batch_size must be a positive integer')
    size = int(size)
    if isinstance(log_every, bool) or int(log_every) != log_every or log_every < 1:
        raise ValueError('log_every must be a positive integer')
    log_every = int(log_every)
    replay = model.replay_buffer
    if replay is None:
        raise ValueError('Critic warmup requires populated replay')
    actor, critic, target = model.actor, model.critic, model.critic_target
    actor_tensors = {id(x) for x in list(actor.parameters()) + list(actor.buffers())}
    critic_tensors = {id(x) for x in list(critic.parameters()) + list(critic.buffers())}
    target_tensors = {id(x) for x in list(target.parameters()) + list(target.buffers())}
    if actor_tensors & (critic_tensors | target_tensors):
        raise ValueError('Critic warmup requires separate actor and critic tensors')
    if critic_tensors & target_tensors:
        raise ValueError('Critic and target must be separate modules')
    if getattr(model, 'use_sde', False):
        raise ValueError('Critic warmup currently supports ordinary Gaussian SAC, not gSDE')
    rate = (float(model.lr_schedule(model._current_progress_remaining))
            if learning_rate is None else float(learning_rate))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Critic learning rate must be finite and positive')
    if model.target_update_interval < 1:
        raise ValueError('Target update interval must be positive')
    if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
        alpha = model.log_ent_coef.detach().exp().clone()
    else:
        alpha = model.ent_coef_tensor.detach().clone()
    if not torch.isfinite(alpha).all() or torch.any(alpha < 0):
        raise ValueError('Entropy coefficient must be finite and nonnegative')
    optimizer = critic.optimizer
    if any(id(p) in actor_tensors for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Critic optimizer unexpectedly owns actor tensors')
    old_rates = [group['lr'] for group in optimizer.param_groups]
    modes = {module: module.training for network in (actor, critic, target) for module in network.modules()}
    started = time.monotonic()
    losses, records = [], []
    target_updates = 0
    completed = 0
    actor.eval();critic.train();target.eval()
    for group in optimizer.param_groups:
        group['lr'] = rate
    try:
        for index in range(steps):
            batch = replay.sample(size, env=getattr(model, '_vec_normalize_env', None))
            discounts = getattr(batch, 'discounts', None)
            discounts = model.gamma if discounts is None else discounts
            with torch.no_grad():
                next_action, next_log_prob = actor.action_log_prob(batch.next_observations)
                next_q = torch.cat(target(batch.next_observations, next_action), dim=1).min(dim=1, keepdim=True).values
                target_q = batch.rewards + (1 - batch.dones) * discounts * (next_q - alpha * next_log_prob.reshape(-1, 1))
            values = critic(batch.observations, batch.actions)
            loss = .5 * sum(torch.nn.functional.mse_loss(value, target_q) for value in values)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SAC critic warmup loss')
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in critic.parameters()):
                optimizer.zero_grad()
                raise FloatingPointError('Nonfinite SAC critic warmup gradient')
            optimizer.step()
            if index % model.target_update_interval == 0:
                polyak_update(critic.parameters(), target.parameters(), model.tau)
                polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.)
                target_updates += 1
            losses.append(float(loss.detach()))
            completed += 1
            model._pnr_critic_warmup_updates = int(getattr(model, '_pnr_critic_warmup_updates', 0)) + 1
            if completed == 1 or completed % log_every == 0 or completed == steps:
                window = losses[-min(log_every, len(losses)):]
                record = {'critic_updates': completed, 'critic_loss': float(np.mean(window)),
                          'target_q_mean': float(target_q.mean()), 'q_mean': float(torch.cat(values, dim=1).detach().mean()),
                          'alpha': float(alpha), 'critic_learning_rate': rate,
                          'actor_updates': 0, 'temperature_updates': 0, 'physical_interactions': 0}
                records.append(record)
                if on_log is not None:
                    on_log(dict(record))
                if hasattr(model, '_logger'):
                    model.logger.record('initialization/critic_loss', record['critic_loss'])
                    model.logger.record('initialization/critic_updates', model._pnr_critic_warmup_updates)
    finally:
        for group, old_rate in zip(optimizer.param_groups, old_rates):
            group['lr'] = old_rate
        for module, mode in modes.items():
            module.training = mode
    return {'method': 'critic_only_sac_bellman_initialization', 'critic_updates': completed,
            'cumulative_critic_warmup_updates': model._pnr_critic_warmup_updates,
            'actor_updates': 0, 'temperature_updates': 0, 'target_updates': target_updates,
            'batch_size': size, 'examples_sampled': completed * size,
            'physical_interactions': 0, 'alpha': float(alpha), 'gamma': float(model.gamma),
            'critic_learning_rate': rate, 'loss_mean': float(np.mean(losses)),
            'loss_first': losses[0], 'loss_last': losses[-1], 'records': records,
            'wall_seconds': time.monotonic() - started}
