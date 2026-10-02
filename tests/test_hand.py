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


def test_side_grasp_objects_start_beside_the_object_with_the_palm_vertical():
    env = HandGraspEnv(objects=[0], seed=0, physics='realistic', side_objects=(0,))
    env.reset(seed=3)
    rig = env.rig
    assert abs(rig.wrist()[5] - np.pi / 2) < 1e-6          # palm vertical
    fingers, palm, table, depth = rig.contacts()
    assert not fingers.any() and not palm and not table
    centre = rig.object_pose()[0]
    assert np.linalg.norm(rig.grasp_point()[:2] - centre[:2]) > .05   # out beside it, not over it


def test_allegro_replaces_the_fetch_gripper():
    from pnr.env import PickNavReachEnv
    env = PickNavReachEnv(seed=0, object_idx=2, physics='realistic', effector='allegro')
    assert len(env.joint_names) == 13 and 'r_gripper_finger_joint' not in env.joint_names
    assert env.model.nu == 16 and env.action_size == 29 and len(env.finger_slots) == 0
    np.testing.assert_allclose(env.model.actuator_forcerange, [[-.7, .7]] * 16)
    target = np.r_[env.agent_qpos, env.hand_open + .3 * (env.hand_high - env.hand_open)]
    env.step(target)
    np.testing.assert_allclose(env.data.ctrl, target[13:])
    assert not env.contacts()['bilateral_contact']


def test_fetch_rig_places_and_moves_the_palm_like_the_hand_rig():
    from pnr.fetch_hand import FetchHandRig
    rig = FetchHandRig(object_idx=2, physics='realistic')
    rig.restore(rig.settled)
    wrist = pregrasp_wrist(rig, .7, height=.12)
    rig.set_wrist(wrist)
    import mujoco
    mujoco.mj_forward(rig.model, rig.data)
    np.testing.assert_allclose(rig.wrist(), wrist, atol=.01)          # planner IK through the flange mount
    for _ in range(4):
        rig.step(wrist, rig.finger_open)
    np.testing.assert_allclose(rig.wrist(), wrist, atol=1e-3)         # holds the pose
    target = wrist + [.01, 0, 0, 0, 0, 0]
    for _ in range(24):      # per-joint rate limits: a 1 cm move settles in ~0.4 s (rig wrist: ~3 ticks)
        rig.step(target, rig.finger_open)
    np.testing.assert_allclose(rig.wrist(), target, atol=2e-3)


def test_hand_planner_system_picks_the_mug_with_the_scripted_hand():
    from pnr.env import PickNavReachEnv
    from pnr.hand_system import HandPlannerSystem, TeacherPolicy
    env = PickNavReachEnv(seed=0, object_idx=10, physics='realistic', effector='allegro')
    result = HandPlannerSystem(env, TeacherPolicy(), seed=0, speed_scale=3., holding_speed_scale=2.).run(stop_after='pick')
    assert result['status'] == 'pick_complete', result.get('error')
    assert result['stages']['hand_grasp']['held'] and env.ever_lifted and env.contacts()['bilateral_contact']
    assert env.drop_events == 0
