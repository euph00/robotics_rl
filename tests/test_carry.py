"""Carry reward, observation boundaries and real physical-state integration."""
import numpy as np
import pytest

from pnr.carry import CarrySkillEnv, table_occupancy, wall_occupancy


def test_sparse_reward_relabels_only_xy_and_preserves_retention():
    env = CarrySkillEnv.__new__(CarrySkillEnv)
    achieved = np.array([[0., 0.], [.01, 0.], [0., 0.], [0., 0.], [0., 0.]])
    goals = np.zeros((5, 2))
    info = np.array([
        {'bilateral_contact': True}, {'bilateral_contact': True},
        {'bilateral_contact': False}, {'bilateral_contact': True, 'retention_ok': False},
        {'bilateral_contact': True, 'physical_failure': True}], dtype=object)
    np.testing.assert_array_equal(env.compute_reward(achieved, goals, info), [0., -1., -1., -1., -50.])
    np.testing.assert_array_equal(env.compute_reward(achieved, achieved, info), [0., 0., -1., -1., -50.])
    assert env.compute_reward([0., 0.], [0., 0.], {'bilateral_contact': True}) == 0
    assert env.compute_reward([0., 0.], [.009, 0.], {'bilateral_contact': True}) == -1
    # Immediate collision cannot improve the return of indefinitely surviving.
    collision=env.compute_reward([0.,0.],[0.,0.],{'bilateral_contact':True,'physical_failure':True})
    assert collision <= -sum(.98**step for step in range(10000))


def test_occupancy_uses_course_world_coordinates_and_separate_table_layer():
    grid = wall_occupancy([[-3.5, 5.5, 0.], [17.5, -5.5, 0.], [0.5, -.5, 0.]])
    assert grid.shape == (12, 22)
    assert grid[0, 0] == grid[-1, -1] == grid[6, 4] == 1
    assert grid.sum() == 3
    assert wall_occupancy(np.empty((0, 3))).sum() == 0
    table = table_occupancy(([-.75, -.5, 0], [.75, .5, .625]))
    assert table.shape == grid.shape and table.sum() > 0
    with pytest.raises(ValueError):
        wall_occupancy([[40., 0.]])


@pytest.fixture(scope='module')
def carry():
    env = CarrySkillEnv(scene_seed=42, object_idx=14, seed=0, horizon=8, evaluation=True)
    yield env
    env.close()


def test_physical_bank_has_contacts_no_constraints_and_accounted_setup(carry):
    assert carry.setup_action_calls > 0
    assert carry.setup_physics_ticks == carry.setup_action_calls * carry.native.substeps
    assert len(carry.snapshots) > 10
    assert carry.native.model.neq == 0
    observation, info = carry.reset(seed=123)
    assert carry.observation_space.contains(observation)
    assert carry.native.contacts()['bilateral_contact']
    assert info['evaluation_scope'] == 'easy_carry_diagnostic'
    assert info['segment_length_m'] == pytest.approx(.3)
    assert carry.native.step_count == 0
    assert np.linalg.norm(observation['desired_goal'] - observation['achieved_goal']) * carry.goal_scale > .15


def test_fixed_goal_reward_and_native_counters(carry):
    observation, _ = carry.reset(seed=456)
    goal = observation['desired_goal'].copy()
    initial_target = carry._frozen_target.copy()
    for step in range(carry.horizon):
        observation, reward, terminated, truncated, info = carry.step(np.zeros(2))
        assert not terminated, info.get('failure_reason')
        np.testing.assert_array_equal(observation['desired_goal'], goal)
        assert info['action_calls_delta'] == carry.repeat
        assert info['physics_ticks_delta'] == carry.repeat * carry.native.substeps
        assert reward == carry.compute_reward(observation['achieved_goal'], goal, info)
        np.testing.assert_allclose(carry.native.last_target[2:], initial_target[2:])
        assert bool(truncated) == (step == carry.horizon - 1)
    assert carry.native.contacts()['bilateral_contact']
    for invalid in ([0], [0, 0, 0], [float('nan'), 0.]):
        with pytest.raises(ValueError):
            carry.step(invalid)


def test_full_carry_diagnostic_and_resume_state(carry):
    observation, info = carry.reset(seed=789, options={'evaluation_mode': 'full'})
    assert info['evaluation_scope'] == 'full_carry'
    np.testing.assert_allclose(observation['desired_goal'], carry.native.goal_pos[:2] / carry.goal_scale)
    np.testing.assert_allclose(observation['achieved_goal'], carry.snapshots[0].base_xy / carry.goal_scale)
    # Desired goal cannot leak into the non-goal features.
    features = observation['observation'].copy()
    carry.goal += [.1, -.1]
    np.testing.assert_array_equal(carry._observation()['observation'], features)
    state = carry.get_training_state()
    expected, _ = carry.reset()
    carry.set_training_state(state)
    actual, _ = carry.reset()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-7)


def test_xy_action_physically_reaches_a_nearby_goal_without_attachments(carry):
    observation, _ = carry.reset(seed=123)
    initial_xy = observation['achieved_goal'].copy() * carry.goal_scale
    success = False
    for _ in range(carry.horizon):
        direction = observation['desired_goal'] - observation['achieved_goal']
        action = direction / max(float(np.linalg.norm(direction)), 1e-9)
        observation, _, terminated, truncated, info = carry.step(action)
        assert not terminated, info.get('failure_reason')
        if info['is_success']:
            success = True
            break
        if truncated:
            break
    assert success, info['goal_distance']
    assert np.linalg.norm(observation['achieved_goal'] * carry.goal_scale - initial_xy) > .05
    assert carry.native.contacts()['bilateral_contact']
    assert carry.native.model.neq == 0


def test_live_binding_preserves_state_policy_contract_and_external_ownership(carry):
    observation, _ = carry.reset(seed=789, options={'evaluation_mode': 'full'})
    before = carry.native.observe()
    counters = (carry.native.step_count, carry.native.physics_steps)
    state_ids = set(carry.native._saved)
    bound = CarrySkillEnv.bind(carry.native, carry.controller, horizon=10)
    try:
        assert (carry.native.step_count, carry.native.physics_steps) == counters
        assert set(carry.native._saved) == state_ids
        assert bound.setup_action_calls == bound.setup_physics_ticks == 0
        assert not bound.snapshots
        assert bound.observation_space == carry.observation_space
        assert bound.action_space == carry.action_space
        for key, values in before.items():
            np.testing.assert_array_equal(carry.native.observe()[key], values)
        for key, values in observation.items():
            np.testing.assert_array_equal(bound._observation()[key], values)
        with pytest.raises(RuntimeError, match='cannot reset'):
            bound.reset()
        _, _, terminated, _, info = bound.step(np.zeros(2))
        assert not terminated
        assert info['evaluation_scope'] == 'full_carry'
        assert info['action_calls_delta'] == carry.repeat
    finally:
        bound.close()
    assert not carry.native._closed
