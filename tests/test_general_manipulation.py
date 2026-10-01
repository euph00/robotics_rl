"""Multi-object observation/HER invariants and zero-step physical binding."""
import numpy as np
import pytest
import torch
from pnr.general_manipulation import GeneralGraspEnv, GeneralGraspFeatures, GraspSchema
from pnr.env import PickNavReachEnv
from pnr.reference import ReferenceController


def test_proposal_features_do_not_cache_relabelable_goal_arithmetic():
    schema = GraspSchema()
    extractor = GeneralGraspFeatures(schema.observation_space)
    observation = {'observation': torch.randn(4, 54),
                   'achieved_goal': torch.randn(4, 3), 'desired_goal': torch.randn(4, 3)}
    before = extractor(observation)
    changed = {key: value.clone() for key, value in observation.items()}
    changed['desired_goal'] += .2
    after = extractor(changed)
    assert before.shape == (4, 70)
    torch.testing.assert_close(after[:, 60:63]-before[:, 60:63], torch.ones(4, 3))
    torch.testing.assert_close(after[:, 63:], before[:, 63:])
    achieved = np.zeros((2, 3), dtype=np.float32)
    infos = [{'bilateral_contact': True}, {'bilateral_contact': False}]
    np.testing.assert_array_equal(schema.compute_reward(achieved, achieved, infos), [0., -1.])
    with pytest.raises(RuntimeError, match='cannot generate'):
        schema.reset()


def test_bound_general_grasp_reads_only_and_exposes_measured_target():
    with PickNavReachEnv(seed=42, object_idx=14) as native:
        controller = ReferenceController(native, seed=42)
        controller.prepare_grasp()
        before = native.observe()
        target_before = native.last_target.copy()
        counters = (native.step_count, native.physics_steps)
        wrapper = GeneralGraspEnv.bind(native, controller, horizon=100, goal_lift_m=.24)
        observed = wrapper._observation()
        assert wrapper.observation_space.contains(observed)
        assert np.isclose(observed['desired_goal'][2]-observed['achieved_goal'][2], .24)
        np.testing.assert_allclose(observed['observation'][50:53],
                                   controller.grasp.position-controller.planner.ee_pose()[0], atol=1e-7)
        assert np.isclose(observed['observation'][53], controller.grasp.width)
        assert counters == (native.step_count, native.physics_steps)
        np.testing.assert_array_equal(target_before, native.last_target)
        for key, value in before.items():
            np.testing.assert_array_equal(value, native.observe()[key])
        wrapper.close()
        assert not native._closed


def test_bounded_fallback_preserves_physics_and_rejects_branch_jump(monkeypatch):
    from pnr.general_manipulation import BoundedGraspIK
    from pnr.local_control import LocalCartesianIK
    with PickNavReachEnv(seed=42, object_idx=14) as native:
        controller = ReferenceController(native, seed=42)
        controller.prepare_grasp()
        solver = BoundedGraspIK(native, controller.planner)
        before, motors = native.observe(), native.last_target.copy()
        counters = (native.step_count, native.physics_steps)
        contacts = native.contacts()
        position, orientation = controller.planner.ee_pose()
        initial = native.agent_qpos.copy()
        def local_rejection(*args, **kwargs):
            raise ValueError('forced local diagnostic rejection')
        monkeypatch.setattr(LocalCartesianIK, 'ik', local_rejection)
        target = solver.ik(position, orientation, seed=initial)
        np.testing.assert_allclose(target, initial, atol=1e-5)
        assert solver.fallback_successes == 1
        assert counters == (native.step_count, native.physics_steps)
        np.testing.assert_array_equal(motors, native.last_target)
        assert native.contacts() == contacts
        for key, value in before.items():
            np.testing.assert_array_equal(value, native.observe()[key])
        def distant_answer(*args, **kwargs):
            result = initial.copy(); result[solver.active[1]] += .36
            return result
        monkeypatch.setattr(controller.planner, 'ik', distant_answer)
        with pytest.raises(ValueError, match='forced local'):
            solver.ik(position, orientation, seed=initial)
        assert solver.fallback_attempts == 2 and solver.fallback_successes == 1


def test_multiobject_goal_sampling_changes_recorded_goal_and_checks_resume_contract():
    from pnr.general_manipulation import MultiObjectGrasp
    wrapper = MultiObjectGrasp([{'object_idx':14, 'scene_seed':42}], seed=821,
                               lift_range=(.24,.24), horizon=3)
    try:
        observation, info = wrapper.reset(seed=123)
        assert info['goal_lift_m'] == .24
        assert np.isclose(observation['desired_goal'][2]-wrapper.active.base_goal[2], .04)
        np.testing.assert_array_equal(observation['desired_goal'], wrapper.active.goal.astype(np.float32))
        state = wrapper.get_training_state()
        wrapper.set_training_state(state)
        with pytest.raises(ValueError, match='proposal mode'):
            wrapper.set_training_state(dict(state, grasp_proposal_mode='clearance-v2'))
        changed = dict(state, lift_range=(.20,.26))
        with pytest.raises(ValueError, match='goal distribution'):
            wrapper.set_training_state(changed)
    finally:
        wrapper.close()


def test_clearance_curriculum_reaches_tilted_starts_and_physically_picks():
    from pnr.general_manipulation import make_grasp_case, ContactGraspTeacher
    env = make_grasp_case(2, 42, seed=61000, horizon=100, evaluation=True,
                          goal_lift_m=.24, grasp_proposal_mode='clearance-v2')
    try:
        assert env.controller.stage_results['grasp_generation']['selected_variant'] == 'tilted_clearance'
        for distance, snapshot in env.snapshots:
            env.native.restore_state(snapshot, reset_counters=True)
            actual = env.controller.planner.ee_pose()[0]
            expected = env.controller.pregrasp_position(distance)
            assert np.linalg.norm(actual-expected) < .02
            if distance == .17:
                assert np.linalg.norm((actual-env.controller.grasp.position)[:2]) > .12
        observation, _ = env.reset(seed=42, options={'full_pregrasp':True})
        env.goal = env.base_goal.copy(); observation = env._observation()
        teacher = ContactGraspTeacher(env)
        for _ in range(100):
            action, _ = teacher.action(observation)
            observation, _, terminated, truncated, info = env.step(action)
            assert not terminated
            if truncated: break
        metrics = env.native.evaluate()
        assert metrics['lifted'] and metrics['bilateral_contact']
        assert metrics['stable_hold_steps'] >= 50 and metrics['drop_events'] == 0
        assert info['goal_distance'] < env.tolerance
    finally:
        env.close()


def test_training_ray_control_preserves_topdown_labels_and_tilt_direction():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from pnr.general_manipulation import ContactGraspTeacher
    delta = np.array([.14,-.018,-.08])
    environment = SimpleNamespace(
        controller=SimpleNamespace(planner=SimpleNamespace(ee_pose=lambda: (np.zeros(3),None))),
        native=SimpleNamespace(contacts=lambda: {'bilateral_contact':False}),
        measured_grasp_target=lambda: delta, delta_scale=.012,
        orientation=Rotation.from_euler('y',90,degrees=True).as_quat())
    old = ContactGraspTeacher(environment)
    ray = ContactGraspTeacher(environment, approach_control='ray-tilted')
    np.testing.assert_array_equal(old.action({})[0],ray.action({})[0])
    environment.orientation=Rotation.from_euler('y',30,degrees=True).as_quat()
    action,_=ray.action({})
    np.testing.assert_allclose(np.cross(action[:3],delta),0.,atol=1e-8)
    assert np.isclose(np.linalg.norm(action[:3]),.65)
    assert action[3] == 1.
    environment.native.contacts=lambda: {'bilateral_contact':True}
    observation={'desired_goal':delta,'achieved_goal':np.zeros(3)}
    np.testing.assert_array_equal(old.action(observation)[0],ray.action(observation)[0])
