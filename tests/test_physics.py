"""Realistic physics settings: rate-limited joints, rated grip force, rubber grip friction."""
import numpy as np
import pytest

from pnr.env import GRIP_FRICTION, REALISTIC, PickNavReachEnv, limited_velocity
from pnr.reference import ReferenceController


def test_velocity_reference_respects_stopping_distance_speed_and_acceleration():
    dt = 1 / 240
    v = limited_velocity(np.array([100.]), np.array([1.]), np.array([0.]), np.array([1.]), np.array([.75]), dt)
    assert v[0] == pytest.approx(.75 * dt)                 # acceleration limited from rest
    v = limited_velocity(np.array([100.]), np.array([1.]), np.array([.9]), np.array([1.]), np.array([.75]), dt)
    assert v[0] == pytest.approx(min(1., .9 + .75 * dt))   # velocity limit
    v = limited_velocity(np.array([100.]), np.array([.01]), np.array([.46]), np.array([1.]), np.array([10.]), dt)
    assert v[0] == pytest.approx(np.sqrt(2 * 10. * .01))   # brakes so it can stop within the remaining error
    v = limited_velocity(np.array([100.]), np.array([.01]), np.array([.9]), np.array([1.]), np.array([10.]), dt)
    assert v[0] == pytest.approx(.9 - 10. * dt)            # ... but never faster than the acceleration limit


@pytest.fixture(scope='module')
def env():
    with PickNavReachEnv(seed=1000, object_idx=2, physics='realistic') as simulation:
        yield simulation


def test_base_accelerates_at_the_realistic_limit(env):
    state = env.save_state()
    target = env.agent_qpos.astype(float)
    target[0] += 1.
    positions = []
    for _ in range(24):          # 0.5 s
        env.step(target)
        positions.append(float(env.agent_qpos[0]))
    assert positions[-1] == pytest.approx(.5 * .75 * .5 ** 2, rel=.15)
    assert np.max(np.diff(positions)) / (5 / 240) <= REALISTIC['root_x_axis_joint'][0] + 1e-6
    env.restore_state(state)
    env.remove_state(state)


def test_planned_grasp_squeezes_at_rated_force_with_rubber_fingers(env):
    rig_state = env.save_state()
    assert GRIP_FRICTION['realistic'] == 1.
    controller = ReferenceController(env, seed=1000, speed_scale=3., grasp_proposal_mode='clearance-v2', holding_speed_scale=2.)
    assert controller.execute_grasp(controller.prepare_grasp())
    force = env.contacts()['normal_force']          # summed over both jaws
    assert 2 * 200 < force < 2 * 260
    env.restore_state(rig_state)
    env.remove_state(rig_state)
