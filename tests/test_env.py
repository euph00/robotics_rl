"""Integration checks on original Bullet physics and scene assets."""
import hashlib
import random

import numpy as np
import pytest

from pnr.env import PickNavReachEnv, UPSTREAM


@pytest.fixture(scope="module")
def env():
    with PickNavReachEnv(seed=0, object_idx=5) as simulation:
        yield simulation


def test_native_contract_and_settled_reset(env):
    obs = env.reset()
    assert env.joint_indices == [0, 1, 2, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 21, 22]
    assert env.ee_link == env.link_name_to_index["gripper_link"]
    assert len(env.arm_slots) == 7 and len(env.finger_slots) == 2
    assert env.maze_map.shape == (12, 22)
    assert env.goal_pos[0] == 17.5 and env.goal_pos[2] == 0.9
    assert obs["object_pc"].shape == (1024, 3)
    assert obs["object_normals"].shape == (1024, 3)
    assert all(value.dtype == np.float32 for value in obs.values())
    assert env.table_height < obs["object_pos"][2] < env.table_height + 0.15
    assert env.p.getContactPoints(env.object_id, env.table_id)
    assert env.evaluate()["physics_ticks"] == 0
    target = env.init_qpos.copy()
    target[0] += 0.05
    first_obs, info = env.step(target)
    assert info["source_action_calls"] == 1 and info["physics_ticks"] == 5
    assert info["sim_seconds"] == pytest.approx(5 / 240)
    assert not info["success"] and not info["complete_task_success"]
    reset_obs = env.reset()
    for key in obs:
        np.testing.assert_array_equal(reset_obs[key], obs[key])
    second_obs, _ = env.step(target)
    for key in first_obs:
        np.testing.assert_allclose(first_obs[key], second_obs[key], atol=1e-7)


def test_clients_rng_and_scene_files_are_isolated(env):
    env.reset()
    before = env.observe()
    np.random.seed(713)
    random.seed(997)
    expected_numpy = np.random.RandomState(713).random_sample(4)
    expected_python = random.Random(997).random()
    original_maze = UPSTREAM / "assets/maze.urdf"
    digest_before = hashlib.sha256(original_maze.read_bytes()).hexdigest() if original_maze.exists() else None
    with PickNavReachEnv(seed=0, object_idx=5) as other:
        np.testing.assert_array_equal(np.random.random_sample(4), expected_numpy)
        assert random.random() == expected_python
        assert other.pb_physics_client != env.pb_physics_client
        assert other.maze_urdf_path != env.maze_urdf_path
        np.testing.assert_array_equal(other.maze_map, env.maze_map)
        np.testing.assert_array_equal(other.object_canonical_pc, env.object_canonical_pc)
        np.testing.assert_array_equal(other.initial_object_pos, env.initial_object_pos)
        target = other.init_qpos.copy()
        target[0] += 0.1
        other.step(target)
        np.testing.assert_array_equal(env.observe()["qpos"], before["qpos"])
        snapshot = env.save_state()
        with pytest.raises(ValueError, match="another"):
            other.restore_state(snapshot)
        env.remove_state(snapshot)
        path = other.maze_urdf_path
    assert not path.exists()
    digest_after = hashlib.sha256(original_maze.read_bytes()).hexdigest() if original_maze.exists() else None
    assert digest_before == digest_after
    assert env.p.isConnected()


def test_snapshot_restores_motion_and_counter_state(env):
    env.reset()
    target = env.init_qpos.copy()
    target[0] = 0.06
    env.step(target)
    state = env.save_state()
    expected, expected_info = env.step(target)
    env.restore_state(state)
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


def test_action_validation_and_render(env):
    for invalid in ([0] * 14, [float("nan")] * 15, np.zeros((3, 5))):
        with pytest.raises(ValueError):
            env.step(invalid)
    rgb = env.render_rgb("grasp", width=160, height=90)
    assert rgb.shape == (90, 160, 3) and rgb.dtype == np.uint8
    assert rgb.std() > 1
