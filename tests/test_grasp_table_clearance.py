"""Original-pad geometry and a physical low-object proposal regression."""
import numpy as np
from pnr.planning import antipodal_grasps
from pnr.env import PickNavReachEnv
from pnr.reference import ReferenceController


def pair(z):
    return np.array([[-.025, 0., z], [.025, 0., z]]), np.array([[-1., 0., 0.], [1., 0., 0.]])


def test_low_contact_points_remain_on_original_pads_and_clear_table():
    points, normals = pair(.645)
    grasps = antipodal_grasps(points, normals, table_height=.626)
    assert len(grasps) == 1
    grasp = grasps[0]
    np.testing.assert_array_equal(grasp.contacts, points)
    assert np.isclose(grasp.position[2], .663)
    assert grasp.position[2] - .0315 > .626
    assert np.max(grasp.position[2] - grasp.contacts[:, 2]) <= .025
    # Contacts too close to the table cannot fit the retained pad margin.
    points, normals = pair(.632)
    assert antipodal_grasps(points, normals, table_height=.626) == []


def test_existing_high_proposal_is_unchanged():
    points, normals = pair(.68)
    unrestricted = antipodal_grasps(points, normals)
    filtered = antipodal_grasps(points, normals, table_height=.626)
    assert [g.as_dict() for g in filtered] == [g.as_dict() for g in unrestricted]


def test_original_banana_physically_lifts_and_holds_without_table_contact():
    with PickNavReachEnv(seed=42, object_idx=5) as env:
        controller = ReferenceController(env, seed=42)
        grasp = controller.prepare_grasp()
        assert grasp.position[2] > grasp.contacts[:, 2].mean()
        assert controller.execute_grasp(grasp)
        assert env.ever_stably_held
        assert env.drop_events == 0
        assert controller.table_contact_steps == 0
        assert controller.wall_contact_steps == 0
