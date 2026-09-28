"""Fixed object/preparation balance preserves stock physical HER semantics."""
import copy
import pickle

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from pnr.object_balanced_her import ObjectBalancedHerReplayBuffer
from pnr.preparation_balanced_her import PreparationBalancedHerReplayBuffer


FAMILIES = {14: ['reference', 'learned'], 5: ['learned'], 2: ['learned'],
            7: ['learned'], 10: ['learned']}


class RewardEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, achieved, desired, infos, indices):
        assert name == 'compute_reward' and indices == [0]
        self.calls.append((achieved.copy(), desired.copy(), copy.deepcopy(infos)))
        failure = np.array([info.get('physical_failure', False) for info in infos])
        return [np.where(failure, -100., -np.abs(achieved[:, 0] - desired[:, 0])).astype(np.float32)]


def buffer(objects=(14, 5), size=256, n_envs=1, families=None,
           kind=PreparationBalancedHerReplayBuffer):
    observation_space = spaces.Dict({
        'observation': spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'achieved_goal': spaces.Box(-np.inf, np.inf, (1,), np.float32),
        'desired_goal': spaces.Box(-np.inf, np.inf, (1,), np.float32)})
    kwargs = dict(buffer_size=size, observation_space=observation_space,
                  action_space=spaces.Box(-1., 1., (1,), np.float32),
                  env=RewardEnv(), device='cpu', n_envs=n_envs, n_sampled_goal=1,
                  goal_selection_strategy='future', copy_info_dict=True)
    if kind is not HerReplayBuffer:
        kwargs['object_indices'] = objects
    if kind is PreparationBalancedHerReplayBuffer:
        kwargs['preparation_sources_by_object'] = (
            {o: FAMILIES[o] for o in objects} if families is None else families)
    return kind(**kwargs)


def add(b, obj, source, step, *, done=False, timeout=False, failure=False,
        overrides=None):
    objects = np.broadcast_to(obj, (b.n_envs,))
    sources = np.broadcast_to(source, (b.n_envs,))
    steps = np.broadcast_to(step, (b.n_envs,))
    # Distinct physical goal ranges identify real episodes for HER-boundary tests.
    achieved = np.array([int(o) * 1000 + (100 if s == 'learned' else 0) + int(t)
                         for o, s, t in zip(objects, sources, steps)], dtype=np.float32)[:, None]
    obs = {'observation': np.c_[objects, steps].astype(np.float32),
           'achieved_goal': achieved, 'desired_goal': np.full((b.n_envs, 1), 99999., np.float32)}
    following = {k: v.copy() for k, v in obs.items()}
    following['achieved_goal'] += 1
    following['observation'][:, 1] += 1
    infos = [{'object_idx': int(o), 'preparation_source': str(s),
              'physical_failure': bool(failure), 'TimeLimit.truncated': bool(timeout)}
             for o, s in zip(objects, sources)]
    if overrides is not None:
        for info, override in zip(infos, overrides):
            info.update(override)
    b.add(obs, following, np.zeros((b.n_envs, 1), np.float32),
          np.full(b.n_envs, -5., np.float32), np.full(b.n_envs, done), infos)


def populate(b, lengths=None):
    lengths = lengths or {(14, 'reference'): 100, (14, 'learned'): 3,
                          (5, 'learned'): 4, (2, 'learned'): 2,
                          (7, 'learned'): 5, (10, 'learned'): 6}
    for obj in b.object_indices:
        for source in b.preparation_sources_by_object[obj]:
            for step in range(lengths[obj, source]):
                add(b, obj, source, step, done=step == lengths[obj, source] - 1,
                    timeout=step == lengths[obj, source] - 1)


def test_fixed_joint_probabilities_despite_imbalance_and_actual_draw_telemetry():
    b = buffer(objects=(14, 5, 2, 7, 10))
    populate(b)
    np.random.seed(6)
    rows, envs = b._sample_transition_indices(60000)
    objects, sources = b.object_labels[rows, envs], b.preparation_labels[rows, envs]
    report = b.sampling_report()
    assert report['protocol'] == 'uniform-declared-object-preparation-her-v1'
    assert report['base_sampling_protocol'] == 'uniform-declared-object-her-v1'
    assert report['eligible_counts_by_object_and_source']['14'] == {'reference': 100, 'learned': 3}
    assert list(report['joint_sampling_probabilities']) == ['14', '5', '2', '7', '10']
    assert list(report['joint_sampling_probabilities']['14']) == ['reference', 'learned']
    assert report['joint_sampling_probabilities']['14'] == {'reference': .1, 'learned': .1}
    for obj, source in b.preparation_strata:
        code = 0 if source == 'reference' else 1
        count = int(np.sum((objects == obj) & (sources == code)))
        expected = 6000 if obj == 14 else 12000
        assert abs(count - expected) < 400
        assert report['cumulative_draw_counts_by_object_and_source'][str(obj)][source] == count
    assert report['cumulative_draws'] == 60000
    assert sum(b.preparation_draw_counts) == sum(b.object_draw_counts) == 60000
    rng = np.random.get_state()
    assert b.sampling_report() == report
    after = np.random.get_state()
    assert np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]


def test_missing_required_family_and_open_episode_fail_without_rng_or_counter_change():
    b = buffer()
    add(b, 14, 'reference', 0, done=True, timeout=True)
    add(b, 5, 'learned', 0, done=True, timeout=True)
    add(b, 14, 'learned', 0)
    report = b.sampling_report()
    assert report['missing_objects'] == []
    assert report['missing_preparation_strata'] == [{'object_idx': 14, 'preparation_source': 'learned'}]
    assert not report['sampling_ready']
    state = np.random.get_state()
    with pytest.raises(RuntimeError, match='required preparation strata'):
        b.sample(8)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert b.sampling_report()['cumulative_draws'] == 0
    add(b, 14, 'learned', 1, done=True, timeout=True)
    assert b.sampling_report()['sampling_ready']


def test_invalid_or_changing_source_rejects_before_mutation():
    b = buffer()
    for value in [None, '', True, 1, 'online', 'Learned']:
        with pytest.raises(ValueError, match='canonical source'):
            add(b, 14, 'reference', 0, overrides=[{'preparation_source': value}])
        assert b.pos == 0 and np.all(b.preparation_labels == -1)
    with pytest.raises(ValueError, match='canonical source'):
        add(b, 5, 'reference', 0)
    add(b, 14, 'reference', 0)
    with pytest.raises(ValueError, match='inside one physical'):
        add(b, 14, 'learned', 1, done=True)
    assert b.pos == 1
    add(b, 14, 'reference', 1, done=True)
    add(b, 14, 'learned', 0, done=True)
    assert b.pos == 3


def test_same_indices_exactly_match_native_her_and_source_is_reward_metadata(monkeypatch):
    balanced = buffer(objects=(14,))
    stock = buffer(kind=HerReplayBuffer)
    assert PreparationBalancedHerReplayBuffer.sample is ObjectBalancedHerReplayBuffer.sample
    for name in ['_sample_goals', '_get_real_samples', '_get_virtual_samples']:
        assert getattr(PreparationBalancedHerReplayBuffer, name) is getattr(HerReplayBuffer, name)
    for b in [balanced, stock]:
        add(b, 14, 'reference', 0)
        add(b, 14, 'reference', 1, done=True, failure=True)
        add(b, 14, 'learned', 0)
        add(b, 14, 'learned', 1, done=True, timeout=True)
    chosen = np.array([1, 3, 0, 2])
    monkeypatch.setattr(balanced, '_sample_transition_indices',
                        lambda n: (chosen.copy(), np.zeros(n, dtype=int)))
    np.random.seed(20)
    actual = balanced.sample(4)
    monkeypatch.setattr(np.random, 'choice', lambda values, size, replace: chosen.copy())
    np.random.seed(20)
    expected = stock.sample(4)
    for field in ['actions', 'dones', 'rewards']:
        assert torch.equal(getattr(actual, field), getattr(expected, field))
    for field in ['observations', 'next_observations']:
        for key in getattr(actual, field):
            assert torch.equal(getattr(actual, field)[key], getattr(expected, field)[key])
    assert actual.dones[:, 0].tolist() == [0., 0., 1., 0.]
    assert actual.rewards[2, 0].item() == -100.
    assert len(balanced.env.calls[-1][0]) == 2  # Unchanged0.5 HER ratio.
    assert [info['preparation_source'] for info in balanced.env.calls[-1][2]] == ['reference', 'learned']
    assert np.array_equal(balanced.env.calls[-1][1][:, 0], [14002., 14102.])


def test_future_goals_never_cross_same_object_preparation_episodes():
    b = buffer(objects=(14,))
    for source in ['reference', 'learned']:
        for step in range(4):
            add(b, 14, source, step, done=step == 3, timeout=step == 3)
    np.random.seed(42)
    samples = b.sample(512)
    assert (samples.observations['desired_goal'][:256] == 99999.).all()
    achieved = samples.observations['achieved_goal'][256:, 0].numpy()
    desired = samples.observations['desired_goal'][256:, 0].numpy()
    episode_base = np.floor(achieved / 100) * 100
    assert (desired >= achieved + 1).all()
    assert (desired <= episode_base + 4).all()


def test_ring_overwrite_cannot_silently_remove_required_reference_family():
    b = buffer(size=9)
    for obj, source in [(14, 'reference'), (14, 'learned'), (5, 'learned')]:
        for step in range(3):
            add(b, obj, source, step, done=step == 2, timeout=step == 2)
    assert b.full and b.pos == 0
    add(b, 14, 'learned', 0)
    with pytest.raises(RuntimeError, match='required preparation strata'):
        b.sample(4)
    assert b.sampling_report()['eligible_counts_by_object_and_source']['14'] == {'reference': 0, 'learned': 3}
    add(b, 14, 'learned', 1, done=True, timeout=True)
    add(b, 14, 'reference', 0, done=True, timeout=True)
    assert b.sampling_report()['eligible_counts_by_object_and_source']['14'] == {'reference': 1, 'learned': 5}
    rows, _ = b._sample_transition_indices(200)
    assert set(rows) == set(range(9))


def test_vector_workers_source_labels_are_independent_and_validation_is_atomic():
    b = buffer(n_envs=2)
    with pytest.raises(ValueError, match='canonical source'):
        add(b, [14, 5], ['reference', 'reference'], [0, 0])
    assert b.pos == 0
    add(b, [14, 14], ['reference', 'learned'], [0, 0])
    add(b, [14, 14], ['reference', 'learned'], [1, 1], done=True, timeout=True)
    add(b, [5, 14], ['learned', 'learned'], [0, 0], done=True, timeout=True)
    report = b.sampling_report()
    assert report['eligible_counts_by_object_and_source'] == {'14': {'reference': 2, 'learned': 3}, '5': {'learned': 1}}
    assert report['sampling_ready']
    b.sample(64)


def test_pickle_retains_source_labels_draw_counts_and_reproducible_indices():
    b = buffer()
    populate(b)
    b.sample(20)
    restored = pickle.loads(pickle.dumps(b))
    assert restored.env is None
    assert restored.sampling_report() == b.sampling_report()
    assert np.array_equal(restored.preparation_labels, b.preparation_labels)
    rng = np.random.get_state()
    expected = b._sample_transition_indices(40)
    np.random.set_state(rng)
    actual = restored._sample_transition_indices(40)
    assert all(np.array_equal(x, y) for x, y in zip(expected, actual))
    restored.set_env(RewardEnv())
    restored.sample(10)
    assert restored.sampling_report()['cumulative_draws'] == 70


def test_corrupt_or_old_metadata_cannot_load_as_preparation_balanced_replay():
    b = buffer()
    populate(b)
    state = b.__getstate__()
    for key in ['preparation_labels', 'preparation_sampling_protocol', 'preparation_draw_counts']:
        bad = copy.deepcopy(state)
        del bad[key]
        with pytest.raises((ValueError, RuntimeError)):
            PreparationBalancedHerReplayBuffer.__new__(PreparationBalancedHerReplayBuffer).__setstate__(bad)
    bad = copy.deepcopy(state)
    bad['preparation_labels'][1, 0] = 1  # Valid family, invalid within-episode switch.
    with pytest.raises(ValueError, match='inside an episode'):
        PreparationBalancedHerReplayBuffer.__new__(PreparationBalancedHerReplayBuffer).__setstate__(bad)
    bad = copy.deepcopy(state)
    bad['preparation_draw_counts'][0] = 1
    with pytest.raises(ValueError, match='counters disagree'):
        PreparationBalancedHerReplayBuffer.__new__(PreparationBalancedHerReplayBuffer).__setstate__(bad)
    parent = buffer(kind=ObjectBalancedHerReplayBuffer)
    with pytest.raises(ValueError, match='fixed preparation-source protocol'):
        PreparationBalancedHerReplayBuffer.__new__(PreparationBalancedHerReplayBuffer).__setstate__(parent.__getstate__())


def test_declarations_require_all_objects_and_known_nonempty_unique_families():
    for families in [{14: ['reference']}, {14: [], 5: ['learned']},
                     {14: ['reference', 'reference'], 5: ['learned']},
                     {14: 'reference', 5: ['learned']},
                     {14: {'reference', 'learned'}, 5: ['learned']},
                     {14: ['online'], 5: ['learned']},
                     {14: ['reference'], 5: ['learned'], 2: ['learned']}]:
        with pytest.raises(ValueError):
            buffer(families=families)
    reversed_order = buffer(families={5: ['learned'], 14: ['learned', 'reference']})
    assert reversed_order.preparation_strata == ((14, 'learned'), (14, 'reference'), (5, 'learned'))
