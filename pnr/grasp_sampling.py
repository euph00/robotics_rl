"""Frozen native-SAC preparation variants with isolated CPU sampling state.

The variant changes training preparation only. It adds no action noise and does
not alter the saved actor, its log-standard-deviation parameters, or physics.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
import pickle
import random

import numpy as np
import torch

SAMPLING_PROTOCOL = 'native-sac-grasp-preparation-variants-v1'
VARIANT_IDS = ('deterministic', 'stochastic_1', 'stochastic_2')
DEFAULT_COLLECTION_SEED = 9401


class SamplingIntegrityError(RuntimeError):
    """Preparation must stop rather than silently change unrelated state."""


def fixed_variants(case_index, seed=DEFAULT_COLLECTION_SEED):
    """Three fixed variants; seeds are seed + 3*case_index + variant_index.

    case_index is the zero-based index in the frozen canonical TRAIN case list.
    The deterministic variant records its seed but never draws an action from it.
    """
    if (isinstance(case_index, bool) or not isinstance(case_index, Integral)
            or int(case_index) < 0 or isinstance(seed, bool) or not isinstance(seed, Integral)
            or not 0 <= int(seed) <= 2**32-1 or int(seed)+3*int(case_index)+2 > 2**32-1):
        raise ValueError('Variant case index and seed must fit the declared nonnegative seed mapping')
    return [dict(protocol=SAMPLING_PROTOCOL, case_index=int(case_index), collection_seed=int(seed),
                 variant_index=i, variant_id=name, deterministic=(i == 0),
                 torch_seed=int(seed)+3*int(case_index)+i)
            for i, name in enumerate(VARIANT_IDS)]


def validate_variant(variant):
    """Return a detached canonical record; reject hidden or inconsistent settings."""
    if not isinstance(variant, dict):
        raise ValueError('Preparation variant must be an explicit dictionary')
    index = variant.get('variant_index')
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= int(index) < 3:
        raise ValueError('Unknown preparation variant index')
    expected = fixed_variants(variant.get('case_index'), variant.get('collection_seed'))[int(index)]
    if set(variant) != set(expected) or type(variant.get('deterministic')) is not bool:
        raise ValueError('Preparation variant fields differ from the declared protocol')
    if any(isinstance(variant[key], bool) or not isinstance(variant[key], Integral)
           for key in ('case_index', 'collection_seed', 'variant_index', 'torch_seed')):
        raise ValueError('Preparation variant indices and seeds must be integers')
    if any(variant[key] != value for key, value in expected.items()):
        raise ValueError('Preparation variant differs from its fixed seed/protocol mapping')
    return expected


def variant_signature(variant):
    return hashlib.sha256(json.dumps(validate_variant(variant), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def actor_parameter_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.actor.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class IsolatedGraspPredictor:
    """Predict using an advancing variant-local RNG while preserving caller RNGs.

    LearnedSystem passes deterministic=True to predictors. This explicit TRAIN
    preparation wrapper uses its frozen variant instead; deployment never wraps
    the policy with a stochastic variant implicitly.
    """
    def __init__(self, model, variant):
        self.variant = validate_variant(variant)
        if str(model.device) != 'cpu' or model.use_sde:
            raise ValueError('Native preparation sampling requires ordinary CPU SAC, without gSDE')
        self.model = model
        self.model.policy.set_training_mode(False)
        self._rng_state = torch.Generator(device='cpu').manual_seed(self.variant['torch_seed']).get_state()
        self.predictions = 0
        self._actor_digest = actor_parameter_digest(model)
        self._integrity_error = None
        self.observation_space = model.observation_space
        self.action_space = model.action_space

    def predict(self, observation, deterministic=True):
        before_torch = torch.get_rng_state().clone()
        before_numpy = pickle.dumps(np.random.get_state())
        before_python = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._rng_state)
                prediction = self.model.predict(observation, deterministic=self.variant['deterministic'])
                self._rng_state = torch.get_rng_state().clone()
        finally:
            if (not torch.equal(before_torch, torch.get_rng_state())
                    or before_numpy != pickle.dumps(np.random.get_state())
                    or before_python != random.getstate()):
                self._integrity_error = 'Native grasp sampling changed caller NumPy/Python/Torch RNG state'
                raise SamplingIntegrityError(self._integrity_error)
        self.predictions += 1
        return prediction

    def sampling_report(self):
        after = actor_parameter_digest(self.model)
        if self._integrity_error or after != self._actor_digest:
            raise SamplingIntegrityError(self._integrity_error or 'Frozen grasp actor/log_std parameters changed')
        return dict(protocol=SAMPLING_PROTOCOL, preparation_variant=dict(self.variant),
            variant_sha256=variant_signature(self.variant), predictions=self.predictions,
            native_saved_sac_distribution=True, added_action_noise=False,
            caller_numpy_python_torch_rngs_preserved=True,
            actor_parameter_sha256_before=self._actor_digest, actor_parameter_sha256_after=after,
            saved_log_std_parameters_unchanged=True,
            initial_torch_seed=self.variant['torch_seed'],
            final_local_torch_rng_sha256=hashlib.sha256(self._rng_state.numpy().tobytes()).hexdigest())


def predictor_for_variant(model, variant):
    return IsolatedGraspPredictor(model, variant)
