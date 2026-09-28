"""Native SAC sampling must advance locally without changing caller RNGs."""
import pickle
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC
import torch

from pnr.grasp_sampling import (fixed_variants,validate_variant,variant_signature,
    predictor_for_variant,SamplingIntegrityError,actor_parameter_digest)


@pytest.fixture
def model():
    env=SimpleNamespace()
    class Schema(gym.Env):
        observation_space=gym.spaces.Box(-1.,1.,(3,),dtype=np.float32)
        action_space=gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
    return SAC('MlpPolicy',Schema(),policy_kwargs={'net_arch':[8]},device='cpu',seed=91)


def rng_state():
    return torch.get_rng_state().clone(),pickle.dumps(np.random.get_state()),random.getstate()


def same_rng(a,b):
    return torch.equal(a[0],b[0]) and a[1:]==b[1:]


def test_native_distribution_advances_locally_and_is_reproducible(model):
    obs=np.array([.1,-.2,.3],np.float32)
    before=rng_state();digest=actor_parameter_digest(model)
    a=predictor_for_variant(model,fixed_variants(4)[1]);b=predictor_for_variant(model,fixed_variants(4)[1])
    c=predictor_for_variant(model,fixed_variants(4)[2])
    seq_a=np.stack([a.predict(obs,deterministic=True)[0] for _ in range(4)])
    seq_b=np.stack([b.predict(obs)[0] for _ in range(4)])
    seq_c=np.stack([c.predict(obs)[0] for _ in range(4)])
    assert same_rng(before,rng_state())
    assert np.array_equal(seq_a,seq_b) and not np.array_equal(seq_a,seq_c)
    assert not np.array_equal(seq_a[0],seq_a[1])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fixed_variants(4)[1]['torch_seed'])
        native=np.stack([model.predict(obs,deterministic=False)[0] for _ in range(4)])
    assert np.array_equal(seq_a,native)
    assert actor_parameter_digest(model)==digest
    assert a.sampling_report()['predictions']==4


def test_deterministic_variant_is_exact_native_policy(model):
    obs=np.zeros(3,np.float32);wrapped=predictor_for_variant(model,fixed_variants(0)[0])
    native=model.predict(obs,deterministic=True)[0];before=rng_state()
    assert np.array_equal(native,wrapped.predict(obs,deterministic=False)[0])
    assert same_rng(before,rng_state())
    assert wrapped.sampling_report()['saved_log_std_parameters_unchanged']


def test_variant_seed_mapping_validation_and_frozen_actor_detection(model):
    variants=fixed_variants(79)
    assert [v['torch_seed'] for v in variants]==[9638,9639,9640]
    assert len({variant_signature(v) for v in variants})==3
    for malformed in (dict(variants[1],torch_seed=0),dict(variants[1],extra=1),dict(variants[0],case_index=True)):
        with pytest.raises(ValueError):validate_variant(malformed)
    for index,seed in ((True,0),(-1,0),(0,2**32-1)):
        with pytest.raises(ValueError):fixed_variants(index,seed)
    wrapped=predictor_for_variant(model,variants[0])
    with torch.no_grad():next(model.actor.parameters()).add_(.1)
    with pytest.raises(SamplingIntegrityError,match='parameters changed'):wrapped.sampling_report()


@pytest.mark.parametrize('throws',[False,True])
def test_unrelated_rng_mutation_fails_even_when_predict_raises(model,monkeypatch,throws):
    wrapped=predictor_for_variant(model,fixed_variants(0)[1]);state=np.random.get_state()
    def corrupt(*args,**kwargs):
        np.random.random()
        if throws:raise RuntimeError('underlying failure')
        return np.zeros(2),None
    monkeypatch.setattr(model,'predict',corrupt)
    try:
        with pytest.raises(SamplingIntegrityError,match='caller'):wrapped.predict(np.zeros(3))
        with pytest.raises(SamplingIntegrityError):wrapped.sampling_report()
    finally:np.random.set_state(state)


def test_unsupported_sampling_is_rejected(model):
    model.use_sde=True
    with pytest.raises(ValueError,match='ordinary CPU'):predictor_for_variant(model,fixed_variants(0)[0])
