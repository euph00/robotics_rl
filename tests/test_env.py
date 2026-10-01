"""Integration checks on the MuJoCo environment and course scene assets."""
import random

import numpy as np
import pytest

from pnr.env import PickNavReachEnv


@pytest.fixture(scope='module')
def env():
    with PickNavReachEnv(seed=0, object_idx=5) as simulation:
        yield simulation


def test_native_contract_and_settled_reset(env):
    obs = env.reset()
    assert len(env.joint_names) == 15 and len(env.arm_slots) == 7 and len(env.finger_slots) == 2
    assert env.maze_map.shape == (12, 22)
    assert env.goal_pos[0] == 17.5 and env.goal_pos[2] == 0.9
    assert obs['object_pc'].shape == (1024, 3) and obs['object_normals'].shape == (1024, 3)
    assert all(value.dtype == np.float32 for value in obs.values())
    assert env.table_height < obs['object_pos'][2] < env.table_height + 0.15
    assert env.touching('object', 'table')
    assert env.evaluate()['physics_ticks'] == 0
    target = env.init_qpos.copy()
    target[0] += 0.05
    first_obs, info = env.step(target)
    assert info['source_action_calls'] == 1 and info['physics_ticks'] == 5
    assert info['sim_seconds'] == pytest.approx(5 / 240)
    assert not info['success'] and not info['complete_task_success']
    reset_obs = env.reset()
    for key in obs:
        np.testing.assert_array_equal(reset_obs[key], obs[key])
    second_obs, _ = env.step(target)
    for key in first_obs:
        np.testing.assert_allclose(first_obs[key], second_obs[key], atol=1e-7)


def test_course_scene_and_settled_object():
    """Course seed 1000: frozen maze and the banana resting flat where the course scene puts it."""
    with PickNavReachEnv(seed=1000, object_idx=5) as course:
        assert course.scene_manifest()['maze_hash'] == '0d9af7803736d0a846f979384de5c20ab3c4dcc9bffd630af699704848e8fbae'
        position, quaternion = course.object_pose()
        np.testing.assert_allclose(position, [0., 0., .6445], atol=2e-3)
        assert abs(quaternion[3]) > .999     # flat (regression: centre of mass 1 m above the mesh)


def test_rng_isolation_and_independent_instances(env):
    env.reset()
    before = env.observe()
    np.random.seed(713)
    random.seed(997)
    expected_numpy = np.random.RandomState(713).random_sample(4)
    expected_python = random.Random(997).random()
    with PickNavReachEnv(seed=0, object_idx=5) as other:
        np.testing.assert_array_equal(np.random.random_sample(4), expected_numpy)
        assert random.random() == expected_python
        np.testing.assert_array_equal(other.maze_map, env.maze_map)
        np.testing.assert_array_equal(other.object_canonical_pc, env.object_canonical_pc)
        np.testing.assert_array_equal(other.initial_object_pos, env.initial_object_pos)
        target = other.init_qpos.copy()
        target[0] += 0.1
        other.step(target)
        np.testing.assert_array_equal(env.observe()['qpos'], before['qpos'])
        snapshot = env.save_state()
        with pytest.raises(ValueError, match='another'):
            other.restore_state(snapshot)
        env.remove_state(snapshot)


def test_snapshot_restores_motion_and_counter_state(env):
    env.reset()
    target = env.init_qpos.copy()
    target[0] = 0.06
    env.step(target)
    state = env.save_state()
    measured, contacts = env.observe(), env.contacts()
    expected, expected_info = env.step(target)
    env.restore_state(state)
    for key, value in env.observe().items():     # bit-exact, including contact-derived quantities
        np.testing.assert_array_equal(value, measured[key])
    assert env.contacts() == contacts
    actual, actual_info = env.step(target)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-7)
    for key in expected_info:
        if isinstance(expected_info[key], float):
            assert actual_info[key] == pytest.approx(expected_info[key], abs=1e-12)
        else:
            assert actual_info[key] == expected_info[key]
    env.remove_state(state)
    with pytest.raises(ValueError):
        env.restore_state(state)


def test_motor_law_closes_thirty_percent_per_tick(env):
    env.reset()
    state = env.save_state()
    target = env.agent_qpos.astype(float)
    target[0] += .015
    env.last_target = target
    start, moves = float(env.agent_qpos[0]), []
    for _ in range(3):
        env._physics_tick()
        moves.append(float(env.agent_qpos[0]) - start - sum(moves))
    np.testing.assert_allclose(moves, [.0045, .00315, .002205], atol=1e-4)
    env.restore_state(state)
    env.remove_state(state)


def test_jacobian_matches_finite_differences(env):
    q = np.zeros(15)
    q[env.arm_slots] = [.3, -.4, .2, 1., -.3, .8, .1]
    jacobian = env.ee_jacobian(q)
    with env.query():
        env.set_joint_positions(q)
        base, _ = env.ee_pose()
        for slot in env.arm_slots:
            shifted = q.copy()
            shifted[slot] += 1e-6
            env.set_joint_positions(shifted)
            np.testing.assert_allclose((env.ee_pose()[0] - base) / 1e-6, jacobian[:3, slot], atol=1e-4)


def test_action_validation_and_render(env):
    for invalid in ([0] * 14, [float('nan')] * 15, np.zeros((3, 5))):
        with pytest.raises(ValueError):
            env.step(invalid)
    rgb = env.render_rgb('grasp', width=160, height=90)
    assert rgb.shape == (90, 160, 3) and rgb.dtype == np.uint8
    assert rgb.std() > 1
