"""MuJoCo backend: same scene as PyBullet and the course motor law."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from pnr.env import PickNavReachEnv
from pnr.mj_env import MjPickNavReachEnv


@pytest.fixture(scope='module')
def envs():
    mj, pb = MjPickNavReachEnv(seed=1000, object_idx=5), PickNavReachEnv(seed=1000, object_idx=5)
    yield mj, pb
    mj.close(); pb.close()


def test_same_scene(envs):
    mj, pb = envs
    assert mj.scene_manifest()['maze_hash'] == pb.scene_manifest()['maze_hash']
    assert mj.joint_names == pb.joint_names
    np.testing.assert_allclose(mj.goal_pos, pb.goal_pos)
    # The banana settles flat where PyBullet puts it (regression: centre of mass 1 m above the mesh).
    np.testing.assert_allclose(mj.object_pose()[0], pb.object_pose()[0], atol=2e-3)


def test_motor_law_closes_thirty_percent_per_tick(envs):
    mj, _ = envs
    state = mj.save_state()
    target = mj.agent_qpos.astype(float)
    target[0] += .015
    mj.last_target = target
    start, moves = float(mj.agent_qpos[0]), []
    for _ in range(3):
        mj._physics_tick()
        moves.append(float(mj.agent_qpos[0]) - start - sum(moves))
    np.testing.assert_allclose(moves, [.0045, .00315, .002205], atol=1e-4)
    mj.restore_state(state)
    mj.remove_state(state)


def test_kinematics_match(envs):
    mj, pb = envs
    q = np.zeros(15)
    q[mj.arm_slots] = [.3, -.4, .2, 1., -.3, .8, .1]
    with mj.query(), pb.query():
        mj.set_joint_positions(q)
        pb.set_joint_positions(q)
        np.testing.assert_allclose(mj.ee_pose()[0], pb.ee_pose()[0], atol=1e-5)
    np.testing.assert_allclose(mj.ee_jacobian(q), pb.ee_jacobian(q), atol=1e-4)
