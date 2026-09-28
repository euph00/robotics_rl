"""Reward relabeling and teacher recovery checks for the multi-maze protocol."""
import numpy as np
from pnr.general_carry import GeneralCarryEnv, route_action


def reward_env():
    env = GeneralCarryEnv.__new__(GeneralCarryEnv)
    env.gamma = .995
    env.shaping_weight = 2.
    env.failure_cost = 200.
    return env


def test_reward_relabels_both_transition_endpoints_and_matches_vector_batch():
    env = reward_env()
    before = np.array([0., 0.])
    after = np.array([.05, 0.])
    info = {'bilateral_contact':True, 'retention_ok':True,
            'physical_failure':False, 'previous_achieved_goal':before}
    goals = np.array([[.1, 0.], [0., 0.]])
    values = env.compute_reward(np.stack([after, after]), goals, [info, info])
    expected = [-1.+2.*(2.-.995*1.), -1.+2.*(0.-.995*1.)]
    np.testing.assert_allclose(values, expected, atol=1e-6)
    for goal, value in zip(goals, values):
        assert np.isclose(env.compute_reward(after, goal, info), value)


def test_hindsight_cannot_remove_physical_failure_or_retention_obligation():
    env = reward_env()
    at = np.array([.1, .2])
    info = {'previous_achieved_goal':at, 'bilateral_contact':False, 'retention_ok':False}
    assert env.compute_reward(at, at, info) == -1.
    info.update(physical_failure=True, bilateral_contact=True, retention_ok=True)
    assert env.compute_reward(at, at, info) == -200.


def test_route_teacher_recovers_from_both_sides_and_stops_at_goal():
    path = [[0.,0.],[2.,0.],[2.,2.]]
    for sign in [-1,1]:
        action = route_action(path, [1.,sign*.05], [2.,2.], .075)
        assert action[0] > 0 and sign*action[1] < 0
        assert np.linalg.norm(action) <= 1.00001
    np.testing.assert_array_equal(route_action(path,[2.,2.],[2.,2.],.075),[0.,0.])


def test_potential_reward_is_bitwise_equal_for_scalar_and_vector_calls():
    # Near-equal float32 endpoint distances made the old scalar/batch paths differ
    # by ~4e-6, causing valid physical demonstrations to fail replay import.
    env = reward_env()
    rng = np.random.default_rng(826)
    previous = rng.uniform([-.2, -.3], [.9, .3], (2048, 2)).astype(np.float32)
    achieved = previous + rng.normal(0., .003, (2048, 2)).astype(np.float32)
    desired = rng.uniform([-.2, -.3], [.9, .3], (2048, 2)).astype(np.float32)
    infos = [{'previous_achieved_goal': value, 'bilateral_contact':True,
              'retention_ok':True, 'physical_failure':bool(i % 11 == 0)}
             for i, value in enumerate(previous)]
    vector = env.compute_reward(achieved, desired, infos)
    scalar = np.asarray([env.compute_reward(a, g, info) for a, g, info in zip(achieved, desired, infos)])
    assert vector.dtype == scalar.dtype == np.float32
    np.testing.assert_array_equal(vector, scalar)
