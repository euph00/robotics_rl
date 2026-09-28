"""Geometric contracts independent of successful learned outcomes."""
import numpy as np
from scipy.spatial.transform import Rotation
from pnr.clearance_grasps import tilted_candidate, approach_position, shape_eligibility
from pnr.planning import Grasp
from pnr.reference import ReferenceController


def sample_grasp():
    y=np.array([-.3,np.sqrt(.91),0.]);x=np.array([0.,0.,-1.])
    quaternion=Rotation.from_matrix(np.column_stack((x,y,np.cross(x,y)))).as_quat()
    midpoint=np.array([.1,.2,.68])
    return Grasp(midpoint,quaternion,.068,1.8,np.array([midpoint-.034*y,midpoint+.034*y]))


def test_tilt_preserves_measured_contacts_and_pinch_axis():
    original=sample_grasp();before=original.as_dict();candidate=tilted_candidate(original)
    orientation=Rotation.from_quat(candidate.quaternion)
    old=Rotation.from_quat(original.quaternion)
    assert np.allclose(orientation.apply([0,1,0]),old.apply([0,1,0]))
    assert np.isclose(orientation.apply([1,0,0])[2],-.5)
    assert candidate.width==original.width
    assert np.array_equal(candidate.contacts,original.contacts)
    assert original.as_dict()==before
    assert np.allclose(orientation.inv().apply(original.position-candidate.position),[.024,0,0])
    for distance in [.008,.04,.09,.17]:
        assert np.allclose(orientation.inv().apply(approach_position(candidate,distance)-candidate.position),[-distance,0,0])


def test_legacy_curriculum_offsets_remain_exact():
    controller=ReferenceController.__new__(ReferenceController)
    controller.grasp=sample_grasp();controller.grasp_proposal_mode='legacy'
    for distance in [.008,.04,.09,.17]:
        assert np.array_equal(controller.pregrasp_position(distance),controller.grasp.position+[0,0,distance])
    controller.grasp_proposal_mode='clearance-v2'
    controller.grasp=tilted_candidate(controller.grasp)
    offset=controller.pregrasp_position(.17)-controller.grasp.position
    assert np.linalg.norm(offset[:2])>.14 and np.isclose(offset[2],.085)


def test_shape_gate_uses_geometry_and_is_scale_translation_invariant():
    import itertools
    tall=np.array(list(itertools.product([-.03,.03],[-.03,.03],[-.05,.05])))
    flat=tall[:,[0,2,1]]
    assert shape_eligibility(tall)['eligible']
    assert not shape_eligibility(flat)['eligible']
    assert shape_eligibility(tall*.5+[3,5,7])['eligible']
