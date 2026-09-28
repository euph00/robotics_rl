"""Opt-in minimum demonstration exposure within each object's native HER pool.

Only original transition-index selection changes. Import provenance is explicit,
never inferred from policy observations, collection mode, success, or object ID.
Future goals, rewards, episode validity, and timeout semantics remain native HER.
"""
from __future__ import annotations

from contextlib import contextmanager
from numbers import Real

import numpy as np

from .object_balanced_her import ObjectBalancedHerReplayBuffer

SAMPLING_PROTOCOL = 'uniform-object-minimum-demonstration-her-v1'
ORIGINS = ('online', 'demonstration')


def validate_fraction(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError('min_demo_fraction must be finite and within (0,1]')
    return float(value)


class DemonstrationFloorHerReplayBuffer(ObjectBalancedHerReplayBuffer):
    """Uniform object; retain its natural demo share, with a declared lower bound.

    For eligible counts D and O, P(demo|object)=max(min_demo_fraction,D/(D+O)).
    If either partition is empty, use the available partition and count these
    draws explicitly. If an object has no completed episode, fail as the parent
    sampler does. Source pools contain all outcomes, including physical failures.

    All add() calls are online except during a one-time fresh-buffer import
    context. Origin labels survive serialization and are replaced on overwrite.
    Old buffers cannot acquire this protocol through loading or inferred labels.
    """

    def __init__(self, *args, min_demo_fraction=.5, **kwargs):
        self.min_demo_fraction = validate_fraction(min_demo_fraction)
        self.demo_floor_protocol = SAMPLING_PROTOCOL
        super().__init__(*args, **kwargs)
        self.origin_labels = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int8)
        self.origin_draw_counts = np.zeros((len(self.object_indices), 2), dtype=np.int64)
        self.available_partition_draw_counts = np.zeros_like(self.origin_draw_counts)
        self._demo_import_active = False
        self._demo_import_complete = False

    @contextmanager
    def demonstration_import(self):
        """Label one explicit initial import; never relabel existing replay."""
        if (self._demo_import_active or self._demo_import_complete or self.pos or self.full
                or np.any(self.origin_labels != -1)):
            raise ValueError('Demonstration import requires a fresh empty replay buffer')
        self._demo_import_active = True
        try:
            yield self
            if np.any(self._current_ep_start != self.pos):
                raise ValueError('Demonstration import must close every physical episode')
            self._demo_import_complete = True
        finally:
            self._demo_import_active = False

    def __getstate__(self):
        if self._demo_import_active:
            raise ValueError('Cannot serialize an active demonstration import')
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        if getattr(self, 'demo_floor_protocol', None) != SAMPLING_PROTOCOL:
            raise ValueError('Saved replay lacks the demonstration-floor protocol; no migration supported')
        self.min_demo_fraction = validate_fraction(getattr(self, 'min_demo_fraction', None))
        shape = (self.buffer_size, self.n_envs)
        counter_shape = (len(self.object_indices), 2)
        if (not hasattr(self, 'origin_labels') or self.origin_labels.shape != shape
                or not np.issubdtype(self.origin_labels.dtype, np.integer)
                or not hasattr(self, 'origin_draw_counts') or self.origin_draw_counts.shape != counter_shape
                or not np.issubdtype(self.origin_draw_counts.dtype, np.integer)
                or not hasattr(self, 'available_partition_draw_counts')
                or self.available_partition_draw_counts.shape != counter_shape
                or not np.issubdtype(self.available_partition_draw_counts.dtype, np.integer)
                or getattr(self, '_demo_import_active', None) is not False
                or not isinstance(getattr(self, '_demo_import_complete', None), bool)):
            raise ValueError('Saved replay has invalid demonstration origin metadata')
        if (np.any(self.origin_draw_counts < 0) or np.any(self.available_partition_draw_counts < 0)
                or np.any(self.available_partition_draw_counts > self.origin_draw_counts)
                or not np.array_equal(self.origin_draw_counts.sum(axis=1), self.object_draw_counts)):
            raise ValueError('Saved replay origin/object draw counters disagree')
        written = self.object_labels >= 0
        if (np.any(~np.isin(self.origin_labels[written], (0, 1)))
                or np.any(self.origin_labels[~written] != -1)):
            raise ValueError('Saved replay has missing or invalid origin labels')
        rows, envs = np.nonzero(self.ep_length > 0)
        starts = self.ep_start[rows, envs] % self.buffer_size
        if np.any(self.origin_labels[rows, envs] != self.origin_labels[starts, envs]):
            raise ValueError('Saved replay origin changes inside an episode')

    def add(self, obs, next_obs, action, reward, done, infos):
        origin = int(self._demo_import_active)
        for env_index in range(self.n_envs):
            if self.pos != self._current_ep_start[env_index]:
                previous = self.origin_labels[(self.pos - 1) % self.buffer_size, env_index]
                if previous != origin:
                    raise ValueError('Replay origin cannot change inside one physical episode')
        position = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.origin_labels[position] = origin

    def _origin_pools(self):
        valid, object_pools = self._eligible_object_pools()
        labels = self.origin_labels.reshape(-1)
        if np.any(~np.isin(labels[np.flatnonzero(valid)], (0, 1))):
            raise RuntimeError('Eligible HER transition lacks explicit import/online provenance')
        return valid, object_pools, [tuple(pool[labels[pool] == origin] for origin in range(2))
                                     for pool in object_pools]

    def _demo_probability(self, online_count, demo_count):
        if demo_count == 0:
            return 0.
        if online_count == 0:
            return 1.
        return max(self.min_demo_fraction, demo_count / (online_count + demo_count))

    def _sample_transition_indices(self, batch_size):
        valid, object_pools, pools = self._origin_pools()
        missing = [obj for obj, pair in zip(self.object_indices, pools) if not sum(map(len, pair))]
        if missing:
            raise RuntimeError('Cannot sample uniform declared objects: no completed eligible '
                               f'transitions for object(s) {missing}')
        chosen_objects = np.random.randint(len(self.object_indices), size=batch_size)
        sampled = np.empty(batch_size, dtype=np.int64)
        object_counts = np.bincount(chosen_objects, minlength=len(self.object_indices))
        for index, (pair, count) in enumerate(zip(pools, object_counts)):
            if not count:
                continue
            positions = np.flatnonzero(chosen_objects == index)
            online_count, demo_count = map(len, pair)
            binding = bool(online_count and demo_count
                           and demo_count / (online_count + demo_count) < self.min_demo_fraction)
            if not binding:
                # Exact parent draw/RNG path, including all-demo critic warmup.
                values = np.random.choice(object_pools[index], size=int(count), replace=True)
                sampled[positions] = values
                origins = self.origin_labels.reshape(-1)[values]
            else:
                origins = (np.random.random(int(count)) < self.min_demo_fraction).astype(np.int8)
                for origin, pool in enumerate(pair):
                    selected = positions[origins == origin]
                    if len(selected):
                        sampled[selected] = np.random.choice(pool, size=len(selected), replace=True)
            origin_counts = np.bincount(origins, minlength=2)
            self.origin_draw_counts[index] += origin_counts
            if not online_count or not demo_count:
                self.available_partition_draw_counts[index] += origin_counts
        self.object_draw_counts += object_counts
        return np.unravel_index(sampled, valid.shape)

    def sampling_report(self):
        """Eligibility, intended probabilities, and actual draws; no RNG changes."""
        report = super().sampling_report()
        _, _, pools = self._origin_pools()
        counts = {}; probabilities = {}; joint = {}; draws = {}; fallbacks = {}
        for index, (obj, pair) in enumerate(zip(self.object_indices, pools)):
            key = str(obj)
            counts[key] = {name: len(pool) for name, pool in zip(ORIGINS, pair)}
            p = self._demo_probability(len(pair[0]), len(pair[1])) if sum(map(len, pair)) else None
            probabilities[key] = None if p is None else {'online': 1 - p, 'demonstration': p}
            joint[key] = None if p is None else {name: value / len(pools)
                                                for name, value in probabilities[key].items()}
            draws[key] = {name: int(n) for name, n in zip(ORIGINS, self.origin_draw_counts[index])}
            fallbacks[key] = {name: int(n) for name, n in zip(ORIGINS, self.available_partition_draw_counts[index])}
        report.update(protocol=self.demo_floor_protocol, base_sampling_protocol=self.sampling_protocol,
                      min_demo_fraction=self.min_demo_fraction,
                      origin_assignment='explicit fresh demonstration import; all other add calls online',
                      eligible_counts_by_object_and_origin=counts,
                      conditional_origin_probabilities=probabilities, joint_sampling_probabilities=joint,
                      cumulative_draw_counts_by_object_and_origin=draws,
                      available_partition_draw_counts_by_object=fallbacks,
                      cold_start_rule='use available eligible partition; fail if neither exists',
                      nonbinding_sampling='exact parent indices and RNG consumption',
                      within_object_sampling='minimum demonstration share; uniform eligible transitions within origin',
                      source_prioritization=True, failure_prioritization=False,
                      origin_label_added_to_policy_observation=False,
                      demonstration_import_complete=self._demo_import_complete)
        return report
