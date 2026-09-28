"""Cache identity, physical ledger, failure retention, and fresh resume semantics."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import pnr.learned_carry as module


def case(index, obj=14):
    return {'case_id': f'train_{index:03d}_object_{obj:02d}', 'layout_id': f'train_{index:03d}',
        'partition': 'train', 'maze_hash': f'map-{index}', 'object_idx': obj,
        'env_kwargs': {'seed': 100 + index, 'maze_seed': index, 'object_idx': obj,
                       'seed_mode': 'independent-maze-v1'}}


class Native:
    instances = []
    fail_objects = set()
    raise_objects = set()
    restore_fail_objects = set()

    def __init__(self, seed, maze_seed, object_idx, **kwargs):
        self.seed, self.maze_seed, self.object_idx = seed, maze_seed, object_idx
        self.settle_steps, self.substeps = 7, 5
        self.step_count = self.physics_steps = 0
        self.agent_qpos = np.zeros(4)
        self.finger_slots = [2, 3]
        self.goal_pos = np.array([10., 0., 1.])
        self.ever_lifted = self.ever_stably_held = self.had_grasp = True
        self.drop_events = 0
        self.closed = False
        self.restore_flags = []
        self.contact = True
        self.instances.append(self)

    def scene_manifest(self):
        return {'maze_hash': f'map-{self.maze_seed}'}

    def restore_state(self, state, reset_counters):
        self.restore_flags.append(reset_counters)
        self.step_count, self.physics_steps = state
        self.agent_qpos[:] = 0.
        self.contact = False  # PyBullet restore needs actual physics to refresh.
        if reset_counters:
            self.had_grasp = self.ever_lifted = self.ever_stably_held = False

    def step(self, action):
        assert np.array_equal(action[self.finger_slots], [0., 0.])
        self.step_count += 1
        self.physics_steps += self.substeps
        self.contact = self.object_idx not in self.restore_fail_objects
        return {}, {'bilateral_contact': self.contact, 'lifted': True,
            'drop_events': self.drop_events, 'ever_lifted': self.ever_lifted,
            'ever_stably_held': self.ever_stably_held}

    def close(self):
        self.closed = True


class Carry:
    goal_scale = 20.
    reset_count = 0

    def __init__(self, native, controller):
        self.native, self.controller = native, controller

    def _collision(self):
        return False

    def _observation(self):
        return {'observation': np.full(580, self.native.object_idx, dtype=np.float32),
            'achieved_goal': self.native.agent_qpos[:2].astype(np.float32) / 20.,
            'desired_goal': self.native.goal_pos[:2].astype(np.float32) / 20.}

    def step(self, action):
        self.native.agent_qpos[:2] += action * .075
        for _ in range(5):
            self.native.step(self.native.agent_qpos.copy())
        return self._observation(), -1., False, False, {}


class Preparation:
    def __init__(self, native, grasp):
        self.native = native
        self.controller = SimpleNamespace(planner=SimpleNamespace(set_payload=lambda: None))
        self.stage = 'grasp'

    def prepare(self):
        for _ in range(11):
            self.native.step(self.native.agent_qpos.copy())
        if self.native.object_idx in Native.raise_objects:
            raise RuntimeError('partial prefix failure')
        okay = self.native.object_idx not in Native.fail_objects
        self.handover_state = (self.native.step_count, self.native.physics_steps) if okay else None
        self.prepared_carry = Carry(self.native, self.controller) if okay else None
        self.handover_observation = self.prepared_carry._observation() if okay else None
        return {'prepared_handover': okay, 'failed_stage': None if okay else 'grasp'}


@pytest.fixture
def factory(monkeypatch, tmp_path):
    Native.instances = []
    Native.fail_objects = set()
    Native.raise_objects = set()
    Native.restore_fail_objects = set()
    path = tmp_path / 'grasp.zip'
    path.write_bytes(b'frozen grasp fixture')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, 'PickNavReachEnv', Native)
    monkeypatch.setattr(module, 'LearnedCarryPreparation', Preparation)
    monkeypatch.setattr(module, 'load_policy_checkpoint', lambda *a, **k: SimpleNamespace(
        observation_space={'observation': SimpleNamespace(shape=(54,))},
        action_space=SimpleNamespace(shape=(4,)),
        _pnr_grasp_protocol_signature={'observation_version': 'antipodal-relative-grasp-v1',
            'control_version': 'bounded_local_global_ik-v1',
            'grasp_proposal_mode': 'clearance-v2', 'eval_lift_m': .24}))
    def make(cases, **kwargs):
        return module.LearnedMultiCaseCarry(cases, path, sha, **kwargs)
    make.path, make.sha = path, sha
    return make


def test_identity_includes_object_point_seed_checkpoint_and_reward():
    original = case(0)
    key = module.learned_carry_identity(original, 'a' * 64)
    other_object = case(0, 2)
    other_point = deepcopy(original)
    other_point['env_kwargs']['seed'] += 1
    assert all(module.learned_carry_identity(c, 'a' * 64) != key
               for c in (other_object, other_point))
    assert module.learned_carry_identity(original, 'b' * 64) != key
    assert module.learned_carry_identity(original, 'a' * 64, gamma=.99) != key


def test_prime_is_idempotent_rotation_preserves_history_and_counts_actual_work(factory):
    cases = [case(0), case(1, 2)]
    env = factory(cases)
    report = env.prime_cache()
    assert report['declared_case_ids'] == report['valid_case_ids'] == [c['case_id'] for c in cases]
    assert report['valid_object_counts'] == {'2': 1, '14': 1}
    assert env.setup_action_calls == 2 * (11 + 5)
    assert env.initialization_physics_ticks == 14
    assert env.reset_count == 0
    assert env.prime_cache() == report
    visited = []
    for _ in range(5):
        _, info = env.reset()
        visited.append(info['case_id'])
        _, _, _, _, step_info = env.step(np.array([1., 0.]))
        assert step_info['action_calls_delta'] == 5
    assert visited == [cases[i % 2]['case_id'] for i in range(5)]
    assert env.setup_action_calls == 32 + 5 * 5
    assert env.source_action_calls == 25
    assert env.physics_ticks == 125
    assert all(n.ever_stably_held and n.had_grasp for n in Native.instances)
    assert all(not flag for n in Native.instances for flag in n.restore_flags)
    assert sorted(v['decisions'] for v in env.case_online_counts.values()) == [2, 3]
    env.close()
    env.close()
    assert all(n.closed for n in Native.instances)


@pytest.mark.parametrize('mode', ['reported', 'exception', 'restore'])
def test_failed_case_cost_is_preserved_without_substitution_or_retry(factory, mode):
    getattr(Native, {'reported': 'fail_objects', 'exception': 'raise_objects',
                    'restore': 'restore_fail_objects'}[mode]).add(2)
    cases = [case(0), case(1, 2)]
    env = factory(cases)
    report = env.prime_cache()
    assert report['failed_case_ids'] == [cases[1]['case_id']]
    assert report['valid_case_ids'] == [cases[0]['case_id']]
    assert len(report['preparation_attempts']) == 2
    assert env.setup_action_calls == 27 + (5 if mode == 'restore' else 0)
    assert len(env.failed_setup_attempts) == 1
    for _ in range(3):
        assert env.reset()[1]['case_id'] == cases[0]['case_id']
    assert len(Native.instances) == 2
    assert Native.instances[1].closed


def test_all_failures_reported_before_reset_raises(factory):
    Native.fail_objects.add(14)
    env = factory([case(0)])
    report = env.prime_cache()
    assert not report['ready_for_training']
    assert report['setup_source_action_calls'] == 11
    with pytest.raises(RuntimeError, match='No valid learned handover'):
        env.reset()


def test_failed_later_restore_disables_case_and_records_exact_hold_cost(factory):
    env = factory([case(0), case(1, 2)])
    env.prime_cache()
    Native.restore_fail_objects.add(14)
    _, info = env.reset()
    assert info['object_idx'] == 2
    report = env.cache_report()
    assert report['failed_case_ids'] == [case(0)['case_id']]
    assert report['setup_source_action_calls'] == 32 + 5 + 5
    assert report['restore_attempts'][-2]['success'] is False
    assert env.case_online_counts[case(0)['case_id']]['resets'] == 0


def test_resume_rebuilds_successes_counts_only_new_work_and_keeps_cursor(factory):
    Native.fail_objects.add(7)
    cases = [case(0), case(1, 2), case(2, 7)]
    first = factory(cases)
    first.prime_cache()
    first.reset()
    first.step(np.array([0., 1.]))
    state = first.get_training_state()
    first.close()
    second = factory(cases)
    second.set_training_state(state)
    assert second.setup_action_calls == second.source_action_calls == 0
    report = second.prime_cache()
    assert second.setup_action_calls == 2 * (11 + 5)
    assert len(report['preparation_attempts']) == 5  # Three inherited; two rebuilds.
    assert len(Native.instances) == 5  # Failed third case never retried.
    _, info = second.reset()
    assert info['case_id'] == cases[1]['case_id']
    assert second.case_online_counts[cases[0]['case_id']]['decisions'] == 1
    assert second.case_online_counts[cases[1]['case_id']]['resets'] == 1
    assert second.reset_count == 2


def test_resume_signature_and_capacity_and_partition_checks(factory):
    state = factory([case(0)]).get_training_state()
    changed = factory([case(0, 2)])
    with pytest.raises(ValueError, match='signature'):
        changed.set_training_state(state)
    with pytest.raises(ValueError, match='complete declared worker slice'):
        factory([case(0), case(1)], cache_size=1)
    invalid = case(0)
    invalid['partition'] = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        factory([invalid])
    factory.path.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='digest'):
        factory([case(0)])


def test_original_goal_ignores_segment_curriculum(factory):
    env = factory([case(0)])
    env.set_curriculum_level(5)
    obs, info = env.reset()
    assert not info['curriculum_applied']
    assert np.allclose(obs['desired_goal'], [.5, 0.])
    with pytest.raises(ValueError, match='original-exit'):
        env.reset(options={'full': False})


def test_partial_transport_exception_still_counts_executed_native_calls(factory):
    env = factory([case(0)])
    env.reset()
    def interrupted(action):
        env.active.native.step(env.active.native.agent_qpos.copy())
        raise RuntimeError('after physical step')
    env.active.step = interrupted
    with pytest.raises(RuntimeError, match='after physical'):
        env.step(np.zeros(2))
    assert env.source_action_calls == 1
    assert env.case_online_counts[case(0)['case_id']]['decisions'] == 0
    assert env.case_online_counts[case(0)['case_id']]['source_action_calls'] == 1


@pytest.mark.parametrize('metadata', [None, {}, {'observation_version': 'antipodal-relative-grasp-v1',
    'control_version': 'bounded_local_global_ik-v1', 'grasp_proposal_mode': 'clearance-v2', 'eval_lift_m': .2}])
def test_rejects_missing_or_incompatible_embedded_grasp_metadata(factory, monkeypatch, metadata):
    monkeypatch.setattr(module, 'load_policy_checkpoint', lambda *a, **k: SimpleNamespace(
        observation_space={'observation': SimpleNamespace(shape=(54,))},
        action_space=SimpleNamespace(shape=(4,)), _pnr_grasp_protocol_signature=metadata))
    with pytest.raises(ValueError, match='protocol metadata'):
        factory([case(0)])


def test_diverse_slots_preserve_scene_identity_rng_provenance_and_physical_ledger(factory, monkeypatch):
    import pnr.diverse_learned_carry as variants
    calls = []
    class Predictor:
        def __init__(self, variant): self.variant = variant
        def sampling_report(self): return {'seed': self.variant['torch_seed'], 'checked': True}
    def predictor(model, variant):
        calls.append(deepcopy(variant))
        return Predictor(variant)
    monkeypatch.setattr(variants, 'predictor_for_variant', predictor)
    original = [case(0), case(1, 2)]
    expanded = variants.expand_preparation_cases(original)
    env = variants.DiverseLearnedMultiCaseCarry(expanded, factory.path, factory.sha)
    original_grasp = env.grasp
    report = env.prime_cache()
    assert len(calls) == len(Native.instances) == report['scenes'] == 6
    assert len(set(env.case_keys)) == 6
    assert env.grasp is original_grasp
    assert report['canonical_case_ids'] == [c['case_id'] for c in original]
    assert env.setup_action_calls == 6 * 16
    assert env.initialization_physics_ticks == 6 * 7
    assert all(r['post_refresh_observation']['observation'] for r in report['restore_attempts'])
    assert len({r['sampling_report']['seed'] for r in report['preparation_attempts']}) == 6
    visited = []
    for _ in range(6):
        _, info = env.reset()
        visited.append((info['case_id'], info['preparation_id']))
        _, _, _, _, step_info = env.step(np.array([0., 1.]))
        assert step_info['preparation_id'] == info['preparation_id']
        assert step_info['case_id'] == info['case_id']
    assert [v[0] for v in visited] == [original[i//3]['case_id'] for i in range(6)]
    assert len({v[1] for v in visited}) == 6
    assert env.source_action_calls == 30
    assert all('preparation_variant' not in c for c in original)
    state = env.get_training_state()
    env.close()
    resumed = variants.DiverseLearnedMultiCaseCarry(expanded, factory.path, factory.sha)
    resumed.set_training_state(state)
    assert resumed.setup_action_calls == 0
    resumed.prime_cache()
    assert resumed.setup_action_calls == 96
    assert resumed.reset_count == 6
    assert len(calls) == 12
    assert calls[:6] == calls[6:]
    changed = variants.DiverseLearnedMultiCaseCarry(variants.expand_preparation_cases(original, seed=9402), factory.path, factory.sha)
    with pytest.raises(ValueError, match='signature'):
        changed.set_training_state(state)
    with pytest.raises(ValueError, match='protocol'):
        changed.set_training_state(factory(original).get_training_state())
    resumed.close()
    changed.close()


def test_diverse_failures_are_retained_per_variant_without_retry(factory, monkeypatch):
    import pnr.diverse_learned_carry as variants
    seen = []
    def predictor(model, variant):
        seen.append(variant['torch_seed'])
        return SimpleNamespace(sampling_report=lambda: {'seed': variant['torch_seed']})
    monkeypatch.setattr(variants, 'predictor_for_variant', predictor)
    Native.fail_objects.add(2)
    env = variants.DiverseLearnedMultiCaseCarry(variants.expand_preparation_cases([case(0), case(1, 2)]), factory.path, factory.sha)
    report = env.prime_cache()
    assert len(report['failed_case_ids']) == 3
    assert len(report['valid_case_ids']) == 3
    assert env.setup_action_calls == 3 * 16 + 3 * 11
    assert all('sampling_report' in record for record in env.failed_setup_attempts)
    for _ in range(4):
        assert env.reset()[1]['case_id'] == case(0)['case_id']
    assert len(seen) == 6
    env.close()
