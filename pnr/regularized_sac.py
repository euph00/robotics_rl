"""SAC + demonstration action regularization (a modified algorithm, not plain SAC).

The SAC 2.7 update is retained for critics, entropy and target networks. Only the
actor objective gains beta * MSE(tanh(mean(s_demo)), a_demo). Demo observations
keep their original goals; no HER-relabeled action is treated as an expert label.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class RegularizedSAC(SAC):
    """Ordinary SAC updates with a separately sampled, fixed-weight actor BC term.

    Demo actions are in SAC's normalized [-1, 1] action coordinates. Observations
    are raw environment observations (VecNormalize, when present, is applied at
    sampling). Groups are optional index arrays, sampled uniformly by group and
    then uniformly within group; explicit group_probabilities can override this.
    Demo arrays are intentionally excluded from model ZIPs. Reattach the same
    arrays/groups after loading; a content digest prevents accidental data drift.
    Sampler RNG and the digest ARE saved. Omitting seed on reattachment preserves
    RNG state. No demonstrator is queried at inference or during online rollouts.
    """
    algorithm_label = 'SAC+HER+demonstration_regularization'

    def __init__(self, *args, bc_weight: float = 10., demo_batch_size: int | None = None,
                 gradient_diagnostics_interval: int = 100, **kwargs):
        self.bc_weight = float(bc_weight)
        if not math.isfinite(self.bc_weight) or self.bc_weight < 0:
            raise ValueError('bc_weight must be finite and nonnegative')
        if demo_batch_size is not None and (isinstance(demo_batch_size, bool) or
                int(demo_batch_size) != demo_batch_size or demo_batch_size < 1):
            raise ValueError('demo_batch_size must be a positive integer or None')
        if (isinstance(gradient_diagnostics_interval, bool) or
                int(gradient_diagnostics_interval) != gradient_diagnostics_interval or
                gradient_diagnostics_interval < 1):
            raise ValueError('gradient_diagnostics_interval must be a positive integer')
        self.demo_batch_size = demo_batch_size
        self.gradient_diagnostics_interval = int(gradient_diagnostics_interval)
        self.demo_manifest = None
        self._demo_observations = self._demo_actions = self._demo_groups = None
        self._demo_probabilities = None
        self._demo_rng = np.random.default_rng(kwargs.get('seed'))
        self._regularization_updates = self._demo_examples_sampled = 0
        self.last_regularization_metrics = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            '_demo_observations', '_demo_actions', '_demo_groups', '_demo_probabilities']

    @staticmethod
    def _array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.array(value, dtype=np.float32, order='C', copy=True)

    def set_demonstrations(self, observations: Mapping, actions, groups: Mapping | None = None,
                           *, group_probabilities=None, seed: int | None = None) -> dict:
        """Attach original-goal expert labels; return a persisted data manifest.

        Call before train/learn when bc_weight > 0. A different dataset requires
        an explicit new model/branch, rather than silently changing a resumed run.
        """
        if not isinstance(observations, Mapping) or not hasattr(self.observation_space, 'spaces'):
            raise ValueError('Demonstration regularization requires dict observations')
        if set(observations) != set(self.observation_space.spaces):
            raise ValueError('Demonstration observation keys differ from observation space')
        labels = self._array(actions)
        if labels.ndim != 2 or labels.shape[1:] != self.action_space.shape or len(labels) == 0:
            raise ValueError('Demo actions must be nonempty [examples, action dimensions]')
        if not np.isfinite(labels).all() or np.any(np.abs(labels) > 1.):
            raise ValueError('Demo actions must be finite SAC-normalized values in [-1, 1]')
        arrays = {key: self._array(value) for key, value in observations.items()}
        for key, value in arrays.items():
            if value.shape != (len(labels), *self.observation_space.spaces[key].shape):
                raise ValueError(f'Demo observation shape mismatch for {key}')
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite demo observations in {key}')
        indices = {'all': np.arange(len(labels))} if groups is None else groups
        if not isinstance(indices, Mapping) or not indices:
            raise ValueError('Groups must be a nonempty mapping of names to index arrays')
        names, strata = [], []
        for key, raw in indices.items():
            raw = np.asarray(raw)
            if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in 'iu':
                raise ValueError('Every demo group must contain a nonempty integer index array')
            if raw.min() < 0 or raw.max() >= len(labels):
                raise ValueError('Demo group index outside dataset')
            names.append(str(key)); strata.append(raw.astype(np.int64, copy=True))
        if len(set(names)) != len(names):
            raise ValueError('Demo group names must have unique string representations')
        probabilities = (np.ones(len(strata), dtype=np.float64) / len(strata)
                         if group_probabilities is None else np.asarray(group_probabilities, dtype=np.float64))
        if (probabilities.shape != (len(strata),) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.)):
            raise ValueError('group_probabilities must be finite nonnegative probabilities summing to 1')
        probabilities = probabilities / probabilities.sum()
        hasher = hashlib.sha256()
        for key, value in sorted(arrays.items()):
            hasher.update(key.encode()); hasher.update(str(value.shape).encode()); hasher.update(value.tobytes())
        hasher.update(labels.tobytes())
        for name, values in zip(names, strata):
            hasher.update(name.encode()); hasher.update(values.tobytes())
        hasher.update(probabilities.tobytes())
        manifest = {'sha256': hasher.hexdigest(), 'examples': len(labels),
                    'groups': {name: len(values) for name, values in zip(names, strata)},
                    'group_probabilities': probabilities.tolist(),
                    'unique_group_coverage': len(np.unique(np.concatenate(strata))),
                    'action_coordinates': 'normalized_minus_one_to_one',
                    'goal_labels': 'original_unrelabeled_demonstration_goals'}
        if self.demo_manifest is not None and self.demo_manifest != manifest:
            raise ValueError('Reattached demonstrations differ from saved data/groups; start an explicit new branch')
        self._demo_observations, self._demo_actions = arrays, labels
        self._demo_groups, self._demo_probabilities = strata, probabilities
        self.demo_manifest = manifest
        if seed is not None:
            self._demo_rng = np.random.default_rng(seed)
        return dict(manifest)

    def _sample_demonstrations(self, size):
        if self._demo_observations is None:
            raise RuntimeError('Call set_demonstrations with the original data before regularized training')
        which = self._demo_rng.choice(len(self._demo_groups), size=size, p=self._demo_probabilities)
        indices = np.empty(size, dtype=np.int64)
        for group, values in enumerate(self._demo_groups):
            selected = np.flatnonzero(which == group)
            indices[selected] = values[self._demo_rng.integers(len(values), size=len(selected))]
        observations = {key: value[indices] for key, value in self._demo_observations.items()}
        if self._vec_normalize_env is not None:
            observations = self._vec_normalize_env.normalize_obs(observations)
        return ({key: torch.as_tensor(value, device=self.device) for key, value in observations.items()},
                torch.as_tensor(self._demo_actions[indices], device=self.device))

    def _gradient_metrics(self, rl_loss, bc_loss):
        parameters = [p for p in self.actor.parameters() if p.requires_grad]
        rl = torch.autograd.grad(rl_loss, parameters, retain_graph=True, allow_unused=True)
        bc = torch.autograd.grad(bc_loss, parameters, retain_graph=True, allow_unused=True)
        rl_sq = sum(g.detach().square().sum() for g in rl if g is not None)
        bc_sq = sum(g.detach().square().sum() for g in bc if g is not None)
        dot = sum((r.detach()*b.detach()).sum() for r, b in zip(rl, bc) if r is not None and b is not None)
        rn, bn = float(torch.sqrt(rl_sq)), float(torch.sqrt(bc_sq))
        return {'actor_rl_grad_norm': rn, 'actor_bc_grad_norm': bn,
                'weighted_actor_bc_grad_norm': self.bc_weight*bn,
                'bc_to_rl_grad_ratio': self.bc_weight*bn/rn if rn > 0 else None,
                'rl_bc_grad_cosine': float(dot)/(rn*bn) if rn*bn > 0 else None,
                'gradient_diagnostic_update': int(self._n_updates)+1}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Exact SB3 path (including RNG and buffers) for the zero-weight ablation.
        if self.bc_weight == 0:
            return super().train(gradient_steps, batch_size)
        if not math.isfinite(self.bc_weight) or self.bc_weight < 0:
            raise ValueError('bc_weight must remain finite and nonnegative')
        if self._demo_observations is None:
            raise RuntimeError('Call set_demonstrations before regularized training')
        if getattr(self.policy, 'share_features_extractor', False):
            raise ValueError('Separate actor/critic extractors required to preserve critic update semantics')
        if gradient_steps < 1:
            return
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        losses = {key: [] for key in ('actor_rl_loss', 'actor_bc_loss', 'actor_loss', 'critic_loss', 'ent_coef', 'ent_coef_loss')}
        diagnostics = {}
        for gradient_step in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(batch, 'discounts', None)
            discounts = self.gamma if discounts is None else discounts
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(batch.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                alpha = torch.exp(self.log_ent_coef.detach())
                ent_loss = -(self.log_ent_coef*(log_prob+self.target_entropy).detach()).mean()
                losses['ent_coef_loss'].append(float(ent_loss.detach()))
            else:
                alpha = self.ent_coef_tensor
            losses['ent_coef'].append(float(alpha))
            if ent_loss is not None:
                self.ent_coef_optimizer.zero_grad(); ent_loss.backward(); self.ent_coef_optimizer.step()
            with torch.no_grad():
                next_action, next_log_prob = self.actor.action_log_prob(batch.next_observations)
                next_q = torch.cat(self.critic_target(batch.next_observations, next_action), dim=1).min(dim=1, keepdim=True).values
                target = batch.rewards+(1-batch.dones)*discounts*(next_q-alpha*next_log_prob.reshape(-1, 1))
            values = self.critic(batch.observations, batch.actions)
            critic_loss = .5*sum(F.mse_loss(value, target) for value in values)
            self.critic.optimizer.zero_grad(); critic_loss.backward(); self.critic.optimizer.step()
            q_pi = torch.cat(self.critic(batch.observations, actions_pi), dim=1).min(dim=1, keepdim=True).values
            rl_loss = (alpha*log_prob-q_pi).mean()
            size = self.demo_batch_size or batch_size
            demo_obs, demo_actions = self._sample_demonstrations(size)
            bc_loss = F.mse_loss(self.actor(demo_obs, deterministic=True), demo_actions)
            actor_loss = rl_loss+self.bc_weight*bc_loss
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite regularized SAC loss')
            update = self._n_updates+gradient_step+1
            if update == 1 or update % self.gradient_diagnostics_interval == 0:
                diagnostics = self._gradient_metrics(rl_loss, bc_loss)
                diagnostics['gradient_diagnostic_update'] = update
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.)
            for key, value in [('actor_rl_loss', rl_loss), ('actor_bc_loss', bc_loss),
                               ('actor_loss', actor_loss), ('critic_loss', critic_loss)]:
                losses[key].append(float(value.detach()))
            self._regularization_updates += 1
            self._demo_examples_sampled += size
        self._n_updates += gradient_steps
        metrics = {key: float(np.mean(values)) for key, values in losses.items() if values}
        metrics.update(bc_weight=float(self.bc_weight), weighted_actor_bc_loss=self.bc_weight*metrics['actor_bc_loss'],
                       regularization_updates=self._regularization_updates, demo_examples_sampled=self._demo_examples_sampled)
        metrics.update(diagnostics)
        self.last_regularization_metrics = metrics
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        for key, value in metrics.items():
            if value is not None:
                self.logger.record('train/'+key, value)
