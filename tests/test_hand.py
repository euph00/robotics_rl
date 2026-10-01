"""Allegro hand rig and grasp environment."""
import numpy as np
import pytest

from pnr.hand import HandRig, pregrasp_wrist
from pnr.hand_env import HandGraspEnv


@pytest.fixture(scope='module')
def rig():
    return HandRig(object_idx=2)


def test_rig_settles_object_and_tracks_wrist_with_course_motor_law(rig):
    rig.restore(rig.settled)
    assert rig.model.nu == 16
    assert abs(rig.object_pose()[0][2] - rig.rest_height) < 1e-6
    start = rig.wrist().copy()
    target = start + [.015, 0, 0, 0, 0, 0]
    rig.wrist_target = target
    moves = []
    for _ in range(3):
        before = rig.wrist()[0]
        rig.tick()
        moves.append(rig.wrist()[0] - before)
    np.testing.assert_allclose(moves, [.0045, .00315, .002205], atol=2e-4)
    rig.restore(rig.settled)


def test_pregrasp_puts_grasp_point_above_object_without_contact(rig):
    rig.restore(rig.settled)
    rig.set_wrist(pregrasp_wrist(rig, .7, height=.12))
    import mujoco
    mujoco.mj_forward(rig.model, rig.data)
    centre = rig.object_pose()[0]
    np.testing.assert_allclose(rig.grasp_point(), centre + [0, 0, .12], atol=1e-6)
    fingers, palm, table, depth = rig.contacts()
    assert not fingers.any() and not palm and not table and depth == 0
    rig.restore(rig.settled)


def test_environment_contract_and_hand_friction_override():
    env = HandGraspEnv(objects=[14], seed=3, friction=.6)
    obs, info = env.reset(seed=4)
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
    rig = env.rig
    hand_geoms = [g for g in range(rig.model.ngeom) if rig.model.geom_bodyid[g] in rig._hand_bodies and rig.model.geom_contype[g]]
    assert hand_geoms and np.allclose(rig.model.geom_friction[hand_geoms, 0], .6)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.zeros(22, dtype=np.float32))
    assert np.isfinite(reward) and not terminated and {'success', 'lift', 'contacts', 'penetration_mm'} <= set(info)
