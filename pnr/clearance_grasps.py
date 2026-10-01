"""Opt-in grasp orientation selection from measured pairs and physical clearance.

No object identity, stored outcomes, attachments or dynamics edits are used.
The existing antipodal score orders pairs; clearance decides whether that pair
can use its original top approach or a declared tilted approach.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .planning import Grasp

PROTOCOL = 'clearance-v2'
TILT_DEGREES = 60.
TCP_SETBACK = .024
APPROACH_DISTANCE = .17
CARRY_HALF_EXTENT = .36
TALL_SHAPE_RATIO = 1.15


def tilted_candidate(grasp):
    y = Rotation.from_quat(grasp.quaternion).apply([0.,1.,0.])
    horizontal = np.array([y[1], -y[0], 0.])
    angle = np.deg2rad(TILT_DEGREES)
    x = np.sin(angle)*horizontal + np.cos(angle)*np.array([0.,0.,-1.])
    quaternion = Rotation.from_matrix(np.column_stack((x,y,np.cross(x,y)))).as_quat()
    return Grasp(grasp.position-TCP_SETBACK*x, quaternion, grasp.width,
                 grasp.score, grasp.contacts.copy())


def approach_position(grasp, distance=APPROACH_DISTANCE):
    axis = Rotation.from_quat(grasp.quaternion).apply([1.,0.,0.])
    return grasp.position-float(distance)*axis


def checked_pregrasp(controller, grasp):
    """Read-only IK/collision queries for the open endpoint and approach."""
    planner = controller.planner
    pre = approach_position(grasp)
    q_pre = planner.ik(pre, grasp.quaternion)
    q_pre[controller.finger_slots] = .05
    if not planner.collision_free(q_pre, include_object=True):
        raise ValueError('Open pregrasp collides')
    # Verify the endpoint first, avoiding unnecessary intermediate IK queries
    # when the palm or wrist cannot fit around the measured object.
    endpoint = planner.ik(grasp.position, grasp.quaternion, seed=q_pre)
    endpoint[controller.finger_slots] = .05
    if not planner.collision_free(endpoint, include_object=True):
        raise ValueError('Open grasp endpoint collides')
    previous = q_pre
    for fraction in np.linspace(0.,1.,10)[1:]:
        q = planner.ik(pre+fraction*(grasp.position-pre), grasp.quaternion, seed=previous)
        q[controller.finger_slots] = .05
        if not planner.collision_free(q, include_object=True):
            raise ValueError(f'Open Cartesian approach collides at{fraction:.3f}')
        previous = q
    return q_pre


def shape_eligibility(points):
    points = np.asarray(points, dtype=float)
    lower, upper = np.quantile(points, [.02,.98], axis=0)
    extents = upper-lower
    ratio = float(extents[2]/max(float(max(extents[:2])),1e-9))
    return {'center_world': np.median(points,axis=0).tolist(),
            'robust_extents_m': extents.tolist(), 'height_to_horizontal_extent': ratio,
            'threshold': TALL_SHAPE_RATIO, 'eligible': ratio >= TALL_SHAPE_RATIO}


def checked_carry_envelope(controller, grasp, q_pre):
    """Predict a reachable folded carry pose without advancing physics.

    The bounded rectangular envelope rejects arm protrusions beyond the existing
    navigation clearance scale. Actual transport remains a required test.
    """
    planner, native = controller.planner, controller.env
    endpoint = planner.ik(grasp.position, grasp.quaternion, seed=q_pre)
    lifted = planner.ik(grasp.position+[0,0,.20], grasp.quaternion, seed=endpoint)
    carry_base = np.array([-1.25,0.])
    lifted[:2] += carry_base-controller.base_position()
    target = np.r_[carry_base+[.24,0.],1.05]
    carry = planner.ik(target, grasp.quaternion, seed=lifted)
    carry[controller.finger_slots] = .05  # conservatively bound the open fingers
    if not planner.collision_free(carry):
        raise ValueError('Predicted carry posture collides with table or maze')
    envelope = []
    with planner.query():
        planner._put(carry)
        for link, (lower, upper) in enumerate(native.robot_link_aabbs()):
            lo=np.asarray(lower[:2])-carry_base;hi=np.asarray(upper[:2])-carry_base
            extent=float(max(np.max(np.abs(lo)),np.max(np.abs(hi))))
            envelope.append({'body':'robot','link':link,'max_axis_extent_m':extent})
    # A physically held object retains the measured grasp-relative geometry.
    # Bound its measured surface and allow20mm of compliant displacement.
    points=native.observe()['object_pc']+target-grasp.position
    object_extent=float(np.max(np.abs(points[:,:2]-carry_base)))+.020
    envelope.append({'body':'object','link':-1,'max_axis_extent_m':object_extent})
    largest=max(envelope,key=lambda row:row['max_axis_extent_m'])
    if largest['max_axis_extent_m'] > CARRY_HALF_EXTENT:
        raise ValueError(f"Predicted carry envelope {largest['body']} link{largest['link']} extends"
                         f" {largest['max_axis_extent_m']:.3f}m beyond{CARRY_HALF_EXTENT:.2f}m bound")
    return {'half_extent_limit_m':CARRY_HALF_EXTENT,'maximum_axis_extent_m':largest['max_axis_extent_m'],
            'predicted_ee_world':target.tolist(),'envelope':envelope,
            'joint_target':carry.tolist()}


def prepare_clearance_grasp(controller, candidates):
    """Try both declared approaches per ranked pair; preserve legacy fallback.

    The caller has physically folded the arm and approached the original table.
    Every rejected candidate is logged; all execution uses native controls.
    """
    attempts = []
    shape = shape_eligibility(controller.env.observe()['object_pc'])
    record = {'protocol': PROTOCOL, 'candidates': len(candidates),
              'tilt_degrees': TILT_DEGREES, 'tcp_setback_m': TCP_SETBACK,
              'candidate_attempts': attempts, 'fallback_used': False,
              'shape_eligibility': shape, 'carry_envelope_limit_m':CARRY_HALF_EXTENT}
    controller.stage_results['grasp_generation'] = record
    selected_pairs = candidates if shape['eligible'] else []
    for pair_index, original in enumerate(selected_pairs):
        for variant, grasp in [('topdown_clearance', original),
                               ('tilted_clearance', tilted_candidate(original))]:
            attempt = {'pair_index': pair_index, 'variant': variant,
                       'source_calls_before': controller.env.step_count}
            attempts.append(attempt)
            try:
                if variant == 'tilted_clearance':
                    center=np.asarray(shape['center_world'])
                    contact_center=np.mean(original.contacts,axis=0)
                    if abs(contact_center[2]-center[2]) > .20*shape['robust_extents_m'][2]:
                        raise ValueError('Contact pair lies outside central40percent of measured body height')
                q_pre = checked_pregrasp(controller, grasp)
                attempt['predicted_carry_envelope'] = checked_carry_envelope(controller,grasp,q_pre)
                controller.move_joint(q_pre, stage='pregrasp', include_object=True)
                controller.grasp = grasp
                controller._clearance_carry_target = np.asarray(attempt['predicted_carry_envelope']['joint_target']).copy()
                attempt['status'] = 'selected'
                record.update(selected=grasp.as_dict(), selected_variant=variant,
                              selected_pair_index=pair_index,
                              pregrasp_position=approach_position(grasp).tolist())
                return grasp
            except ValueError as error:
                attempt.update(status='rejected', error=str(error))
            finally:
                attempt['source_calls_after'] = controller.env.step_count
    # Keep the established original topdown option if every fully open
    # clearance candidate fails. The original skill's early closure gate is
    # unchanged, and its physical outcomes must still be counted normally.
    record['fallback_used'] = True
    for pair_index, grasp in enumerate(candidates):
        attempt = {'pair_index': pair_index, 'variant': 'legacy_topdown_fallback',
                   'source_calls_before': controller.env.step_count}
        attempts.append(attempt)
        try:
            pre = grasp.position + [0,0,.17]
            q = controller.planner.ik(pre, grasp.quaternion)
            q[controller.finger_slots] = .05
            if not controller.planner.collision_free(q, include_object=True):
                raise ValueError('Legacy pregrasp collides')
            q_grasp = controller.planner.ik(grasp.position, grasp.quaternion, seed=q)
            q_grasp[controller.finger_slots] = .05
            if not controller.planner.collision_free(q_grasp):
                raise ValueError('Legacy endpoint collides with table or maze')
            controller.move_joint(q, stage='pregrasp', include_object=True)
            controller.grasp = grasp
            attempt['status'] = 'selected'
            record.update(selected=grasp.as_dict(), selected_variant='legacy_topdown_fallback',
                          selected_pair_index=pair_index,pregrasp_position=pre.tolist())
            return grasp
        except ValueError as error:
            attempt.update(status='rejected', error=str(error))
        finally:
            attempt['source_calls_after'] = controller.env.step_count
    raise RuntimeError(f'No feasible grasp under{PROTOCOL} ({len(candidates)} measured pairs)')
