"""Opt-in HER sampling over fixed declared object/preparation-source strata.

This only changes original transition-index selection. Native SB3 2.7.0 HER
future goals, physical reward info, terminals/timeouts and episode storage remain
inherited. No old replay migration or default sampler changes are performed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from .object_balanced_her import ObjectBalancedHerReplayBuffer

PREPARATION_INFO_KEY = 'preparation_source'
PREPARATION_SOURCES = ('reference', 'learned')
PREPARATION_SAMPLING_PROTOCOL = 'uniform-declared-object-preparation-her-v1'


def _declaration(objects, families):
    if (not isinstance(families, Mapping)
            or any(isinstance(obj, bool) or not isinstance(obj, Integral) for obj in families)
            or set(families) != set(objects)):
        raise ValueError('preparation_sources_by_object must declare exactly the object indices')
    result = {}
    for obj in objects:
        values = families[obj]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError('Each object requires a nonempty sequence of preparation sources')
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError('Each object requires a nonempty sequence of preparation sources') from exc
        if (not values or any(not isinstance(value, str) or value not in PREPARATION_SOURCES
                              for value in values) or len(set(values)) != len(values)):
            raise ValueError('Declare distinct canonical preparation sources: reference, learned')
        result[obj] = values
    return result


class PreparationBalancedHerReplayBuffer(ObjectBalancedHerReplayBuffer):
    """Uniform object, uniform fixed preparation family, uniform real transition.

    Required ``preparation_sources_by_object`` maps every declared object to an
    ordered sequence of canonical strings. Every required pair must have eligible
    completed-episode data; missing strata raise rather than changing weights.
    Every add() info must contain ``preparation_source``. The family is metadata
    only and must remain constant throughout one real episode.

    The parent sampling_protocol remains intact for its serialization contract;
    preparation_sampling_protocol and sampling_report()['protocol'] identify
    this explicit extension. Old object-only replay cannot load as this class.
    """
    def __init__(self, *args, preparation_sources_by_object, **kwargs):
        super().__init__(*args, **kwargs)
        self.preparation_sources_by_object = _declaration(
            self.object_indices, preparation_sources_by_object)
        self.preparation_sampling_protocol = PREPARATION_SAMPLING_PROTOCOL
        self.preparation_strata = tuple((obj, source) for obj in self.object_indices
                                       for source in self.preparation_sources_by_object[obj])
        self.preparation_labels = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int8)
        self.preparation_draw_counts = np.zeros(len(self.preparation_strata), dtype=np.int64)

    def __setstate__(self, state):
        super().__setstate__(state)
        if getattr(self, 'preparation_sampling_protocol', None) != PREPARATION_SAMPLING_PROTOCOL:
            raise ValueError('Saved replay lacks the fixed preparation-source protocol')
        declaration = _declaration(self.object_indices,
                                   getattr(self, 'preparation_sources_by_object', None))
        expected = tuple((obj, source) for obj in self.object_indices for source in declaration[obj])
        if (getattr(self, 'preparation_strata', None) != expected
                or not hasattr(self, 'preparation_labels')
                or self.preparation_labels.shape != (self.buffer_size, self.n_envs)
                or not np.issubdtype(self.preparation_labels.dtype, np.integer)
                or not hasattr(self, 'preparation_draw_counts')
                or self.preparation_draw_counts.shape != (len(expected),)
                or not np.issubdtype(self.preparation_draw_counts.dtype, np.integer)
                or np.any(self.preparation_draw_counts < 0)):
            raise ValueError('Saved replay has invalid preparation-source labels or counters')
        self._eligible_preparation_pools()
        rows, envs = np.nonzero(self.ep_length > 0)
        starts = self.ep_start[rows, envs] % self.buffer_size
        if (np.any(self.preparation_labels[rows, envs] != self.preparation_labels[starts, envs])
                or np.any(self.object_labels[rows, envs] != self.object_labels[starts, envs])):
            raise ValueError('Saved replay preparation/object labels change inside an episode')
        for i, obj in enumerate(self.object_indices):
            count = sum(int(self.preparation_draw_counts[j]) for j, (o, _) in enumerate(expected) if o == obj)
            if count != int(self.object_draw_counts[i]):
                raise ValueError('Saved replay preparation and object draw counters disagree')

    def add(self, obs, next_obs, action, reward, done, infos):
        if len(infos) != self.n_envs:
            raise ValueError('Every worker transition requires a preparation source')
        labels = []
        for env_idx, info in enumerate(infos):
            obj = info.get('object_idx')
            if isinstance(obj, bool) or not isinstance(obj, Integral) or obj not in self.object_indices:
                raise ValueError('Preparation-balanced replay requires a declared integer object_idx')
            source = info.get(PREPARATION_INFO_KEY)
            if not isinstance(source, str) or source not in self.preparation_sources_by_object[obj]:
                raise ValueError("info['preparation_source'] must be a declared canonical source for its object")
            code = PREPARATION_SOURCES.index(source)
            if self.pos != self._current_ep_start[env_idx]:
                previous = int(self.preparation_labels[(self.pos - 1) % self.buffer_size, env_idx])
                if previous != code:
                    raise ValueError('Preparation source changed inside one physical HER episode')
            labels.append(code)
        position = self.pos
        # Parent validates object consistency before native add mutates the ring.
        super().add(obs, next_obs, action, reward, done, infos)
        self.preparation_labels[position] = labels

    def _eligible_preparation_pools(self):
        valid = self.ep_length > 0
        indices = np.flatnonzero(valid)
        objects = self.object_labels.reshape(-1)[indices]
        sources = self.preparation_labels.reshape(-1)[indices]
        known = np.zeros(len(indices), dtype=bool)
        pools = []
        for obj, source in self.preparation_strata:
            match = (objects == obj) & (sources == PREPARATION_SOURCES.index(source))
            pools.append(indices[match])
            known |= match
        if not np.all(known):
            raise RuntimeError('Eligible HER transition lacks declared object/preparation provenance')
        return valid, pools

    def _sample_transition_indices(self, batch_size):
        valid, pools = self._eligible_preparation_pools()
        missing = [pair for pair, pool in zip(self.preparation_strata, pools) if not len(pool)]
        if missing:
            raise RuntimeError('No completed eligible transitions for required preparation strata: '
                               + repr(missing))
        chosen_objects = np.random.randint(len(self.object_indices), size=batch_size)
        sampled = np.empty(batch_size, dtype=np.int64)
        joint_counts = np.zeros(len(pools), dtype=np.int64)
        offset = 0
        for index, obj in enumerate(self.object_indices):
            positions = np.flatnonzero(chosen_objects == index)
            families = self.preparation_sources_by_object[obj]
            selected_sources = np.random.randint(len(families), size=len(positions))
            for family_idx in range(len(families)):
                targets = positions[selected_sources == family_idx]
                joint_counts[offset + family_idx] = len(targets)
                if len(targets):
                    sampled[targets] = np.random.choice(pools[offset + family_idx], size=len(targets), replace=True)
            offset += len(families)
        self.object_draw_counts += np.bincount(chosen_objects, minlength=len(self.object_indices))
        self.preparation_draw_counts += joint_counts
        return np.unravel_index(sampled, valid.shape)

    def sampling_report(self):
        report = super().sampling_report()
        _, pools = self._eligible_preparation_pools()
        counts, draws, probabilities = {}, {}, {}
        missing = []
        for index, ((obj, source), pool) in enumerate(zip(self.preparation_strata, pools)):
            key = str(obj)
            counts.setdefault(key, {})[source] = len(pool)
            draws.setdefault(key, {})[source] = int(self.preparation_draw_counts[index])
            probabilities.setdefault(key, {})[source] = 1. / (len(self.object_indices)
                                                             * len(self.preparation_sources_by_object[obj]))
            if not len(pool):
                missing.append({'object_idx': obj, 'preparation_source': source})
        report.update(base_sampling_protocol=report['protocol'],
                      protocol=self.preparation_sampling_protocol,
                      preparation_source_info_key=PREPARATION_INFO_KEY,
                      preparation_sources_by_object={str(obj): list(self.preparation_sources_by_object[obj])
                                                     for obj in self.object_indices},
                      eligible_counts_by_object_and_source=counts,
                      joint_sampling_probabilities=probabilities,
                      cumulative_draw_counts_by_object_and_source=draws,
                      missing_preparation_strata=missing,
                      sampling_ready=report['sampling_ready'] and not missing,
                      fixed_preparation_families_required=True,
                      source_prioritization=True, failure_prioritization=False,
                      within_object_sampling='uniform fixed declared preparation family, then eligible original transitions',
                      preparation_source_added_to_policy_observation=False)
        return report
