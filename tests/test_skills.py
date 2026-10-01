"""Physical skill interface, pure HER rewards, and local-IK integration checks."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.local_control import LocalCartesianIK
from pnr.skills import ManipulationSkillEnv


@pytest.fixture(scope='module')
def skill_env():
    env = ManipulationSkillEnv(skill='grasp', scene_seed=42, object_idx=14, seed=7, horizon=4)
    yield env
    env.close()


def test_reward_relabeling_preserves_recorded_retention():
    env = object.__new__(ManipulationSkillEnv)
    env.tolerance = .04
    achieved = np.array([[0, 0, .9], [0, 0, .9], [0, 0, .9]], dtype=np.float32)
    goals = np.array([[0, 0, .9], [0, 0, .9], [.2, 0, .9]], dtype=np.float32)
    info = np.array([{'bilateral_contact':True}, {'bilateral_contact':False}, {'bilateral_contact':True}], dtype=object)
    np.testing.assert_array_equal(env.compute_reward(achieved, goals, info), [0, -1, -1])
    # Relabeling changes only the goal. Contact-free transitions still fail even
    # when the hindsight goal exactly matches their recorded achieved position.
    np.testing.assert_array_equal(env.compute_reward(achieved, achieved, info), [0, -1, 0])
    for i in range(3):
        assert env.compute_reward(achieved[i], goals[i], info[i]) == env.compute_reward(achieved, goals, info)[i]


def test_physical_skill_reset_shapes_counters_and_timeout(skill_env):
    env = skill_env
    first, _ = env.reset(seed=79)
    assert env.observation_space.contains(first)
    assert first['observation'].shape == (50,)
    assert env.native.touching('object', 'table')
    np.testing.assert_allclose(env.base_xy, env.controller.base_position(), atol=1e-6)
    for i in range(env.horizon):
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, -1], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert reward == env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        assert info['action_calls_delta'] == env.repeat
        assert info['physics_ticks_delta'] == env.repeat * env.native.substeps
        assert not terminated
        assert truncated == (i+1 == env.horizon)
    repeat, _ = env.reset(seed=79)
    for key in first:
        np.testing.assert_array_equal(repeat[key], first[key])
    assert env.native.step_count == 0


def test_her_uses_terminal_observation_and_saved_transition_info(skill_env):
    env = skill_env
    vector = DummyVecEnv([lambda: env])
    replay = HerReplayBuffer(buffer_size=32, observation_space=env.observation_space,
        action_space=env.action_space, env=vector, device='cpu', n_envs=1, copy_info_dict=True)
    obs = vector.reset()
    action = np.array([[0, 0, 0, -1]], dtype=np.float32)
    for _ in range(env.horizon * 2):
        next_obs, reward, done, infos = vector.step(action)
        stored_next = {key:value.copy() for key, value in next_obs.items()}
        if done[0]:
            assert infos[0]['TimeLimit.truncated']
            for key in stored_next:
                stored_next[key][0] = infos[0]['terminal_observation'][key]
        replay.add(obs, stored_next, action, reward, done, infos)
        obs = next_obs
    indices = np.arange(env.horizon * 2)
    virtual = replay._get_virtual_samples(indices, np.zeros(len(indices), dtype=int))
    expected = env.compute_reward(virtual.next_observations['achieved_goal'].numpy(),
        virtual.observations['desired_goal'].numpy(), replay.infos[indices, 0])
    np.testing.assert_array_equal(virtual.rewards.numpy().ravel(), expected)
    # Time limits bootstrap; they are not incorrectly treated as physical failures.
    assert not virtual.dones.numpy().any()
    np.testing.assert_array_equal(replay.ep_length[:env.horizon*2, 0], env.horizon)


def test_analytic_jacobian_and_local_ik_do_not_mutate_physics(skill_env):
    env = skill_env
    env.reset(seed=79)
    native = env.native
    local = LocalCartesianIK(native)
    q = native.agent_qpos.astype(float)
    before = native.observe()
    counters = native.step_count, native.physics_steps
    state = native.save_state()
    try:
        native._set_qpos(q)
        position, orientation = env.controller.planner.ee_pose()
        analytic = local.jacobian(q)
        numeric = np.zeros_like(analytic)
        epsilon = 1e-3
        for slot in local.active:
            trial = q.copy(); trial[slot] += epsilon
            native._set_qpos(trial)
            changed_pos, changed_orn = env.controller.planner.ee_pose()
            numeric[:3, slot] = (changed_pos-position)/epsilon
            numeric[3:, slot] = (Rotation.from_quat(changed_orn)*Rotation.from_quat(orientation).inv()).as_rotvec()/epsilon
        np.testing.assert_allclose(analytic[:, local.active], numeric[:, local.active], atol=8e-4, rtol=.008)
    finally:
        native.restore_state(state)
        native.remove_state(state)
    position, orientation = env.controller.planner.ee_pose()
    fixed = np.setdiff1d(np.arange(15), local.active)
    for direction in ([.012, 0, 0], [-.012, 0, 0], [0, .012, 0], [0, -.012, 0], [0, 0, .012]):
        target = local.ik(position+direction, orientation)
        np.testing.assert_array_equal(target[fixed], q[fixed])
        assert np.max(np.abs(target[local.active]-q[local.active])) <= .35+1e-8
        with env.controller.planner.query():
            native.set_joint_positions(target)
            end_pos, end_orn = env.controller.planner.ee_pose()
            assert np.linalg.norm(end_pos-position-direction) <= .0015
            assert (Rotation.from_quat(end_orn)*Rotation.from_quat(orientation).inv()).magnitude() <= .02
    for key, value in before.items():
        np.testing.assert_array_equal(native.observe()[key], value)
    assert (native.step_count, native.physics_steps) == counters
    with pytest.raises(ValueError, match='Local IK rejected'):
        local.ik(position+[10, 0, 0], orientation)
    np.testing.assert_array_equal(native.agent_qpos, before['qpos'])
