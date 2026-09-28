"""Geometric invariants, independent of any successful rollout."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pnr.planning import antipodal_grasps, maze_astar


def test_antipodal_contacts_use_opposed_normals_and_clear_table():
    # Explicit opposing surfaces: enough vertical points to reject unsafe pairs.
    points=[]; normals=[]
    for x in [-.025,.025]:
        for y in [-.008,0,.008]:
            for z in [.65,.665,.675]:
                points.append([x,y,z]); normals.append([np.sign(x),0,0])
    candidates=antipodal_grasps(np.array(points),np.array(normals),table_height=.626)
    assert candidates
    for grasp in candidates:
        assert grasp.position[2] >= .663
        assert .012 < grasp.width < .098
        assert grasp.contacts[0,0]*grasp.contacts[1,0] < 0
        assert np.allclose(Rotation.from_quat(grasp.quaternion).apply([1,0,0]),[0,0,-1])
    # Parallel outward normals cannot define an antipodal pinch.
    assert antipodal_grasps(np.array(points),np.tile([1,0,0],(len(points),1))) == []


def test_astar_inflates_wall_and_avoids_diagonal_corner_cutting():
    walls=np.array([[0.,0.]])
    path=maze_astar(walls,[-1.5,0],[1.5,0],footprint_radius=.35,bounds=(-2,2,-2,2))
    assert len(path)>2
    for a,b in zip(path[:-1],path[1:]):
        for t in np.linspace(0,1,100):
            point=a+(b-a)*t
            gap=np.maximum(np.abs(point)-.5,0)
            assert np.linalg.norm(gap)>=.35-1e-8


def test_astar_rejects_endpoint_without_robot_clearance():
    with pytest.raises(ValueError,match='endpoint'):
        maze_astar([[0,0]],[.7,0],[1.5,0],footprint_radius=.35,bounds=(-2,2,-2,2))
