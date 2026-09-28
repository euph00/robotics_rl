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
        wrapper = GeneralGraspEnv.bind(native, controller, horizon=100)
        observed = wrapper._observation()
        assert wrapper.observation_space.contains(observed)
        np.testing.assert_allclose(observed['observation'][50:53],
                                   controller.grasp.position-controller.planner.ee_pose()[0], atol=1e-7)
        assert np.isclose(observed['observation'][53], controller.grasp.width)
        assert counters == (native.step_count, native.physics_steps)
        np.testing.assert_array_equal(target_before, native.last_target)
        for key, value in before.items():
            np.testing.assert_array_equal(value, native.observe()[key])
        wrapper.close()
        assert native.p.isConnected()
