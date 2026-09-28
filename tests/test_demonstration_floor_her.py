"""Origin exposure changes while native physical reward/HER semantics stay exact."""
import copy
from contextlib import nullcontext
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from pnr.demonstration_floor_her import DemonstrationFloorHerReplayBuffer, SAMPLING_PROTOCOL
from pnr.object_balanced_her import ObjectBalancedHerReplayBuffer
from pnr.demonstrations import seed_replay
from pnr import phase2_checkpoints as checkpoints


class RewardEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, achieved, desired, infos, indices):
        assert name == 'compute_reward' and indices == [0]
        self.calls.append((achieved.copy(), desired.copy(), copy.deepcopy(infos)))
        failure = np.array([i.get('physical_failure', False) for i in infos])
        return [np.where(failure, -100., -np.abs(achieved[:, 0] - desired[:, 0])).astype(np.float32)]


def buffer(objects=(14, 5), size=256, n_envs=1, kind=DemonstrationFloorHerReplayBuffer, fraction=.5):
    observation_space = spaces.Dict({
        'observation': spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'achieved_goal': spaces.Box(-np.inf, np.inf, (1,), np.float32),
        'desired_goal': spaces.Box(-np.inf, np.inf, (1,), np.float32)})
    kwargs = dict(buffer_size=size, observation_space=observation_space,
                  action_space=spaces.Box(-1., 1., (1,), np.float32), env=RewardEnv(),
                  device='cpu', n_envs=n_envs, n_sampled_goal=1,
                  goal_selection_strategy='future', copy_info_dict=True)
    if kind is not HerReplayBuffer:
        kwargs['object_indices'] = objects
    if kind is DemonstrationFloorHerReplayBuffer:
        kwargs['min_demo_fraction'] = fraction
    return kind(**kwargs)


def add(b, obj, step, *, tag=0, done=False, timeout=False, failure=False):
    objects = np.broadcast_to(obj, (b.n_envs,))
    steps = np.broadcast_to(step, (b.n_envs,))
    achieved = (objects * 1000 + tag * 100 + steps).astype(np.float32)[:, None]
    observation = {'observation': np.c_[objects, steps].astype(np.float32),
                   'achieved_goal': achieved, 'desired_goal': np.full((b.n_envs, 1), 99999., np.float32)}
    following = {k: v.copy() for k, v in observation.items()}
    following['achieved_goal'] += 1
    following['observation'][:, 1] += 1
    infos = [{'object_idx': int(o), 'physical_failure': failure, 'TimeLimit.truncated': timeout,
              'origin_tag_for_test': tag} for o in objects]
    b.add(observation, following, np.zeros((b.n_envs, 1), np.float32),
          np.full(b.n_envs, -5., np.float32), np.full(b.n_envs, done), infos)


def episode(b, obj, length, tag=0):
    for step in range(length):
        add(b, obj, step, tag=tag, done=step == length - 1, timeout=step == length - 1)


def test_uniform_objects_exact_floor_and_prior_preservation_with_actual_draws():
    b = buffer(objects=(14, 5, 2, 7, 10))
    with b.demonstration_import():
        for obj, length in [(14, 9), (5, 2), (2, 3), (7, 4)]:
            episode(b, obj, length)
    for obj, length in [(14, 1), (5, 8), (2, 3), (10, 6)]:
        episode(b, obj, length, tag=1)
    report = b.sampling_report()
    expected = {14: .9, 5: .5, 2: .5, 7: 1., 10: 0.}
    for obj, p in expected.items():
        assert report['conditional_origin_probabilities'][str(obj)] == {'online': 1 - p, 'demonstration': p}
        assert report['joint_sampling_probabilities'][str(obj)]['demonstration'] == p / 5
    # Natural .9 demo share retains uniform probability across all10 object14 rows.
    assert .9 / 9 == pytest.approx((1 - .9) / 1)
    np.random.seed(45)
    rows, envs = b._sample_transition_indices(100000)
    objects, origins = b.object_labels[rows, envs], b.origin_labels[rows, envs]
    for obj, p in expected.items():
        chosen = objects == obj
        assert chosen.mean() == pytest.approx(.2, abs=.008)
        assert origins[chosen].mean() == pytest.approx(p, abs=.012)
        for origin in (0, 1):
            pool = np.flatnonzero((b.object_labels[:, 0] == obj) & (b.origin_labels[:, 0] == origin))
            n = np.count_nonzero(chosen & (origins == origin))
            if n:
                counts = np.bincount(rows[chosen & (origins == origin)], minlength=b.buffer_size)[pool]
                assert np.max(np.abs(counts - n / len(pool))) < .08 * n / len(pool)
    after = b.sampling_report()
    assert after['cumulative_draws'] == 100000
    assert sum(sum(v.values()) for v in after['cumulative_draw_counts_by_object_and_origin'].values()) == 100000
    assert after['available_partition_draw_counts_by_object']['7']['demonstration'] == int(np.sum(objects == 7))
    assert after['available_partition_draw_counts_by_object']['10']['online'] == int(np.sum(objects == 10))
    state = np.random.get_state(); b.sampling_report(); following = np.random.get_state()
    assert np.array_equal(state[1], following[1]) and state[2:] == following[2:]


def test_incomplete_online_episode_does_not_dilute_demo_and_missing_object_fails_without_draws():
    b = buffer()
    with b.demonstration_import():
        episode(b, 14, 2)
    add(b, 5, 0)
    state = np.random.get_state()
    with pytest.raises(RuntimeError, match='no completed eligible'):
        b.sample(8)
    assert b.sampling_report()['cumulative_draws'] == 0
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    add(b, 5, 1, done=True, timeout=True)
    add(b, 14, 0, tag=1)
    assert b.sampling_report()['conditional_origin_probabilities']['14']['demonstration'] == 1.
    add(b, 14, 1, tag=1, done=True, timeout=True)
    assert b.sampling_report()['conditional_origin_probabilities']['14']['demonstration'] == .5


def test_explicit_fresh_import_only_no_origin_inference_and_exception_restores_mode():
    b = buffer(objects=(14,))
    with b.demonstration_import():
        episode(b, 14, 2)
        with pytest.raises(ValueError, match='active demonstration import'):
            pickle.dumps(b)
    # Arbitrary existing info source tags never make an online add a demonstration.
    episode(b, 14, 2, tag=0)
    assert b.origin_labels[:4, 0].tolist() == [1, 1, 0, 0]
    with pytest.raises(ValueError, match='fresh empty'):
        with b.demonstration_import():
            pass
    failed = buffer(objects=(14,))
    with pytest.raises(RuntimeError, match='interrupted'):
        with failed.demonstration_import():
            raise RuntimeError('interrupted')
    assert not failed._demo_import_active
    add(failed, 14, 0, done=True)
    assert failed.origin_labels[0, 0] == 0
    open_episode = buffer(objects=(14,))
    with pytest.raises(ValueError, match='close every physical episode'):
        with open_episode.demonstration_import():
            add(open_episode, 14, 0)
    before = open_episode.pos
    with pytest.raises(ValueError, match='inside one physical episode'):
        add(open_episode, 14, 1, done=True)
    assert open_episode.pos == before


def test_same_indices_match_native_real_and_her_samples_rewards_and_timeouts(monkeypatch):
    actual_buffer = buffer(objects=(14,))
    stock = buffer(kind=HerReplayBuffer)
    assert DemonstrationFloorHerReplayBuffer.sample is ObjectBalancedHerReplayBuffer.sample
    for name in ['_sample_goals', '_get_real_samples', '_get_virtual_samples']:
        assert getattr(DemonstrationFloorHerReplayBuffer, name) is getattr(HerReplayBuffer, name)
    for b in (actual_buffer, stock):
        context = b.demonstration_import() if b is actual_buffer else nullcontext()
        with context:
            add(b, 14, 0, tag=0)
            add(b, 14, 1, tag=0, done=True, failure=True)
        add(b, 14, 0, tag=1)
        add(b, 14, 1, tag=1, done=True, timeout=True)
    chosen = np.array([1, 3, 0, 2])
    assert actual_buffer.origin_labels[chosen, 0].tolist() == [1, 0, 1, 0]
    monkeypatch.setattr(actual_buffer, '_sample_transition_indices', lambda n: (chosen.copy(), np.zeros(n, int)))
    np.random.seed(20); actual = actual_buffer.sample(4)
    monkeypatch.setattr(np.random, 'choice', lambda values, size, replace: chosen.copy())
    np.random.seed(20); expected = stock.sample(4)
    for field in ['actions', 'dones', 'rewards']:
        assert torch.equal(getattr(actual, field), getattr(expected, field))
    for field in ['observations', 'next_observations']:
        for key in getattr(actual, field):
            assert torch.equal(getattr(actual, field)[key], getattr(expected, field)[key])
    assert actual.dones[:, 0].tolist() == [0., 0., 1., 0.]
    assert actual.rewards[2, 0].item() == -100.
    assert [i['origin_tag_for_test'] for i in actual_buffer.env.calls[-1][2]] == [0, 1]
    assert actual_buffer.env.calls[-1][1][:, 0].tolist() == [14002., 14102.]


def test_future_goals_cannot_cross_origin_episode_and_actual_half_her_is_unchanged():
    b = buffer(objects=(14,))
    with b.demonstration_import():
        episode(b, 14, 4)
    episode(b, 14, 12, tag=1)
    np.random.seed(51); sample = b.sample(20000)
    assert (sample.observations['desired_goal'][:10000].numpy() == 99999.).all()
    achieved = sample.observations['achieved_goal'][10000:, 0].numpy()
    goals = sample.observations['desired_goal'][10000:, 0].numpy()
    base = np.floor(achieved / 100) * 100
    lengths = np.where(base == 14000, 4, 12)
    assert (goals >= achieved + 1).all() and (goals <= base + lengths).all()
    assert np.mean(base == 14000) == pytest.approx(.5, abs=.025)
    assert b.sampling_report()['her_ratio'] == .5


def test_native_ring_invalidates_whole_old_episode_and_overwrites_origin_labels():
    b = buffer(objects=(14,), size=9)
    with b.demonstration_import():
        episode(b, 14, 3)
        episode(b, 14, 3)
    episode(b, 14, 3, tag=1)
    assert b.full and b.pos == 0
    add(b, 14, 0, tag=2)
    assert b.origin_labels[:3, 0].tolist() == [0, 1, 1]
    assert b.sampling_report()['eligible_counts_by_object_and_origin']['14'] == {'online': 3, 'demonstration': 3}
    rows, _ = b._sample_transition_indices(1000)
    assert not np.isin(rows, [0, 1, 2]).any()
    add(b, 14, 1, tag=2, done=True, timeout=True)
    assert b.sampling_report()['conditional_origin_probabilities']['14']['demonstration'] == .5
    episode(b, 14, 2, tag=3)  # Writes slot3: remaining old demo episode becomes ineligible.
    report = b.sampling_report()
    assert report['eligible_counts_by_object_and_origin']['14'] == {'online': 7, 'demonstration': 0}
    assert report['conditional_origin_probabilities']['14']['demonstration'] == 0.
    b.sample(100)
    assert b.sampling_report()['available_partition_draw_counts_by_object']['14']['online'] == 100


def test_vector_origins_serialization_reproducible_draws_and_counter_validation():
    b = buffer(n_envs=2)
    with b.demonstration_import():
        add(b, [14, 5], [0, 0]); add(b, [14, 5], [1, 1], done=True, timeout=True)
    add(b, [5, 14], [0, 0], tag=1, done=True, timeout=True)
    b.sample(32)
    restored = pickle.loads(pickle.dumps(b))
    assert restored.env is None and restored.sampling_report() == b.sampling_report()
    rng = np.random.get_state(); before = b._sample_transition_indices(100)
    np.random.set_state(rng); after = restored._sample_transition_indices(100)
    assert all(np.array_equal(x, y) for x, y in zip(before, after))
    restored.set_env(RewardEnv()); restored.sample(8)
    for key in ['origin_labels', 'demo_floor_protocol', 'origin_draw_counts', 'available_partition_draw_counts']:
        state = b.__getstate__(); del state[key]
        with pytest.raises(ValueError):
            DemonstrationFloorHerReplayBuffer.__new__(DemonstrationFloorHerReplayBuffer).__setstate__(state)
    state = copy.deepcopy(b.__getstate__()); state['origin_labels'][1, 0] = 0
    with pytest.raises(ValueError, match='inside an episode'):
        DemonstrationFloorHerReplayBuffer.__new__(DemonstrationFloorHerReplayBuffer).__setstate__(state)
    state = copy.deepcopy(b.__getstate__()); state['origin_draw_counts'][0, 0] += 1
    with pytest.raises(ValueError, match='counters disagree'):
        DemonstrationFloorHerReplayBuffer.__new__(DemonstrationFloorHerReplayBuffer).__setstate__(state)
    old = buffer(kind=ObjectBalancedHerReplayBuffer)
    with pytest.raises(ValueError, match='no migration'):
        DemonstrationFloorHerReplayBuffer.__new__(DemonstrationFloorHerReplayBuffer).__setstate__(old.__getstate__())


@pytest.mark.parametrize('fraction', [0., -1., 1.01, np.nan, np.inf, True, None, '0.5'])
def test_invalid_floor_rejected(fraction):
    with pytest.raises(ValueError, match='min_demo_fraction'):
        buffer(fraction=fraction)


def test_real_seeder_context_preserves_rows_rewards_info_and_collector_timeouts(tmp_path):
    b = buffer(objects=(14, 5), n_envs=2)
    old = buffer(objects=(14, 5), n_envs=2, kind=ObjectBalancedHerReplayBuffer)
    episodes = []
    for obj, length in [(14, 3), (5, 5)]:
        rows = []
        for step in range(length):
            obs = {'observation': np.array([obj, step], np.float32),
                   'achieved_goal': np.array([step], np.float32), 'desired_goal': np.array([10], np.float32)}
            following = {k: v.copy() for k, v in obs.items()}; following['achieved_goal'] += 1
            rows.append(dict(observation=obs, next_observation=following, action=np.zeros(1, np.float32),
                             reward=float(step + 1 - 10), terminated=False, truncated=step == length - 1,
                             info={'action_calls_delta': 5, 'physics_ticks_delta': 25}))
        episodes.append(dict(object_idx=obj, transitions=rows, collection_mode='learned-handover'))
    path = tmp_path/'demo.pkl'; path.write_bytes(pickle.dumps(dict(schema_version=1, episodes=episodes)))
    original = path.read_bytes()
    reports = []
    for replay in (b, old):
        model = SimpleNamespace(replay_buffer=replay, n_envs=2, observation_space=replay.observation_space,
                                action_space=replay.action_space, env=replay.env)
        with replay.demonstration_import() if replay is b else nullcontext():
            reports.append(seed_replay(model, path, object_labels=True))
    assert reports[0] == reports[1]
    assert reports[0]['transitions_imported'] == 6 and reports[0]['collector_timeouts'] == 1
    assert path.read_bytes() == original and np.all(b.origin_labels[:3] == 1)
    for key in b.observations:
        np.testing.assert_array_equal(b.observations[key], old.observations[key])
        np.testing.assert_array_equal(b.next_observations[key], old.next_observations[key])
    for name in ['actions', 'rewards', 'dones', 'timeouts', 'ep_start', 'ep_length']:
        np.testing.assert_array_equal(getattr(b, name), getattr(old, name))
    assert b.infos.tolist() == old.infos.tolist()
    add(b, [5, 14], [0, 0], done=True, timeout=True)
    assert (b.origin_labels[3] == 0).all()


def config():
    result = {key: 0 for key in checkpoints.METHOD_FIELDS}
    result.update(workers=1, layouts=8, seed=7401, horizon=900, delta_scale=.075, batch_size=8,
                  learning_rate=3e-5, gamma=.995, shaping_weight=2., target_entropy=-4., n_sampled_goal=1,
                  demonstrations_sha256='d'*64, scene_bank_sha256='b'*64, buffer_size=256,
                  balance_replay_objects=True, preparation_mode='reference', preparation_grasp_sha256=None,
                  preparation_settings=None, training_cases_sha256=None)
    return result


def test_conditional_signature_preserves_default_and_rejects_mode_fraction_source_changes(tmp_path):
    base = config(); enabled = dict(base, min_demo_fraction=.5, demo_floor_protocol=SAMPLING_PROTOCOL)
    sources = set(checkpoints._semantic_sources(checkpoints._method(enabled)))
    for name in sources:
        p = tmp_path/name; p.parent.mkdir(exist_ok=True, parents=True); p.write_text('# fixed synthetic source\n')
    baseline = checkpoints.signature(base, tmp_path)
    assert checkpoints.signature(dict(base, min_demo_fraction=None), tmp_path) == baseline
    assert 'min_demo_fraction' not in baseline['method']
    assert 'pnr/demonstration_floor_her.py' not in baseline['semantic_sources']
    active = checkpoints.signature(enabled, tmp_path)
    assert active['method']['demo_floor_protocol'] == SAMPLING_PROTOCOL
    with pytest.raises(ValueError, match='signature fields differ'):
        checkpoints.check_signature(baseline, active)
    with pytest.raises(ValueError, match='min_demo_fraction'):
        checkpoints.check_signature(active, checkpoints.signature(dict(enabled, min_demo_fraction=.75), tmp_path))
    for bad in [dict(enabled, demo_floor_protocol='old'), dict(enabled, balance_replay_sources=True),
                dict(enabled, balance_replay_objects=False), dict(base, demo_floor_protocol=SAMPLING_PROTOCOL)]:
        with pytest.raises(ValueError):
            checkpoints.signature(bad, tmp_path)
    (tmp_path/'pnr/demonstration_floor_her.py').write_text('# changed sampler\n')
    assert checkpoints.signature(base, tmp_path) == baseline
    with pytest.raises(ValueError, match='demonstration_floor_her'):
        checkpoints.check_signature(active, checkpoints.signature(enabled, tmp_path))


def test_loaded_model_checks_actual_sampler_and_fraction_not_just_declaration():
    b = buffer(objects=(14, 5, 2, 7, 10))
    with b.demonstration_import():
        for obj in b.object_indices:
            episode(b, obj, 1)
    model = SimpleNamespace(n_envs=1, replay_buffer=b, num_timesteps=0, _n_updates=0,
                            buffer_size=256, gamma=.995, target_entropy=-4., batch_size=8)
    method = dict(config(), min_demo_fraction=.5, demo_floor_protocol=SAMPLING_PROTOCOL)
    state = dict(online_decisions=0, online_gradient_updates=0, replay_position=b.pos,
                 training_states=[], resume_signature={'method': method})
    checkpoint = dict(state=state, legacy=False)
    checkpoints.validate_loaded_model(model, checkpoint, 1)
    state['resume_signature']['method'] = config()
    with pytest.raises(ValueError, match='demonstration-floor sampling'):
        checkpoints.validate_loaded_model(model, checkpoint, 1)
    state['resume_signature']['method'] = dict(method, min_demo_fraction=.75)
    with pytest.raises(ValueError, match='fraction/protocol'):
        checkpoints.validate_loaded_model(model, checkpoint, 1)
    state['resume_signature']['method'] = method
    model.replay_buffer = buffer(objects=(14, 5, 2, 7, 10), kind=ObjectBalancedHerReplayBuffer)
    with pytest.raises(ValueError, match='demonstration-floor sampling'):
        checkpoints.validate_loaded_model(model, checkpoint, 1)


def test_runner_rejects_legacy_replay_migration_and_conflicting_sampler_flags(tmp_path, monkeypatch, capsys):
    from scripts import phase2_train_carry as runner
    parent = tmp_path/'runs/old'
    parent.mkdir(parents=True)
    import json
    (parent/'config.json').write_text(json.dumps(dict(demonstrations='unused.pkl', balance_replay_objects=True)))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    cases = [(['--min-demo-fraction', '.5'], 'requires object-balanced'),
             (['--min-demo-fraction', '.5', '--balance-replay-objects', '--balance-replay-sources'], 'separate ablations'),
             (['--min-demo-fraction', '.5', '--resume-run', 'old'], 'old replay origins cannot be migrated')]
    for flags, message in cases:
        monkeypatch.setattr('sys.argv', ['phase2_train_carry.py', '--run-id', 'never-created',
                                         '--demonstrations', 'unused.pkl', *flags])
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path/'runs/never-created').exists()


@pytest.mark.parametrize('counts', [((4, 0), (5, 0)), ((0, 4), (0, 5)), ((9, 1), (3, 3)), ((5, 0), (0, 4))])
def test_nonbinding_and_coldstart_preserve_exact_parent_indices_her_samples_and_rng(counts):
    floor = buffer(); parent = buffer(kind=ObjectBalancedHerReplayBuffer)
    for b in (floor, parent):
        with b.demonstration_import() if b is floor else nullcontext():
            for obj, (demo, online) in zip(b.object_indices, counts):
                if demo:
                    episode(b, obj, demo)
        for obj, (demo, online) in zip(b.object_indices, counts):
            if online:
                episode(b, obj, online, tag=1)
    np.random.seed(155)
    before = np.random.get_state()
    expected = parent._sample_transition_indices(10000)
    expected_rng = np.random.get_state()
    np.random.set_state(before)
    actual = floor._sample_transition_indices(10000)
    actual_rng = np.random.get_state()
    assert all(np.array_equal(a, e) for a, e in zip(actual, expected))
    assert np.array_equal(actual_rng[1], expected_rng[1]) and actual_rng[2:] == expected_rng[2:]
    np.random.set_state(before); expected = parent.sample(64); expected_rng = np.random.get_state()
    np.random.set_state(before); actual = floor.sample(64); actual_rng = np.random.get_state()
    for field in ['actions', 'dones', 'rewards']:
        assert torch.equal(getattr(actual, field), getattr(expected, field))
    for field in ['observations', 'next_observations']:
        for key in getattr(actual, field):
            assert torch.equal(getattr(actual, field)[key], getattr(expected, field)[key])
    assert np.array_equal(actual_rng[1], expected_rng[1]) and actual_rng[2:] == expected_rng[2:]
