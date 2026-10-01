"""Bounded Jacobian IK for small policy-controlled Cartesian displacements.

The global planner still chooses collision-free approaches and grasp branches.
This solver changes only torso/arm joints around the current configuration. It
never steps physics and always restores queried state, including contacts and
motor state. Collision checking belongs to the caller's existing JointPlanner.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class LocalCartesianIK:
    def __init__(self, env, max_iterations=16, damping=.015, max_joint_change=.35):
        self.env = env
        self.active = np.array([env.joint_name_to_slot['torso_lift_joint'], *env.arm_slots])
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        self.max_joint_change = float(max_joint_change)
        self.last_iterations = 0
        self.last_position_error = float('inf')
        self.last_orientation_error = float('inf')
        self.last_ik_error = float('inf')

    def _put(self, q):
        self.env.set_joint_positions(q)

    def _pose(self):
        position, quaternion = self.env.ee_pose()
        return position, Rotation.from_quat(quaternion)

    def jacobian(self, q):
        """Spatial Jacobian of the gripper frame origin in world coordinates (6x15)."""
        return self.env.ee_jacobian(q)

    def ik(self, position, quaternion, seed=None, tolerance=.0015, orientation_tolerance=.02):
        """Return 15 targets near `seed`, or reject an unreachable local target.

        Base/head/fingers remain exactly fixed. A .35-rad neighborhood prevents
        branch switches, with a .04-m bound for the torso. This is intended for
        approximately 1-cm deltas, not initial approach or global arm planning.
        """
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all():
            raise ValueError('Expected finite Cartesian position and quaternion')
        desired_rotation = Rotation.from_quat(quaternion)
        initial = np.asarray(self.env.agent_qpos if seed is None else seed, dtype=float).copy()
        if initial.shape != (self.env.action_size,) or not np.isfinite(initial).all():
            raise ValueError('Expected complete finite seed configuration')
        neighborhood = np.full(len(self.active), self.max_joint_change)
        neighborhood[0] = .04
        lower = np.maximum(self.env.joint_lower[self.active], initial[self.active]-neighborhood)
        upper = np.minimum(self.env.joint_upper[self.active], initial[self.active]+neighborhood)
        q = initial.copy()
        q[self.active] = np.clip(q[self.active], lower, upper)
        with self.env.query():
            for iteration in range(self.max_iterations + 1):
                self._put(q)
                current_position, current_rotation = self._pose()
                translation_error = position-current_position
                rotation_error = (desired_rotation*current_rotation.inv()).as_rotvec()
                self.last_position_error = float(np.linalg.norm(translation_error))
                self.last_orientation_error = float(np.linalg.norm(rotation_error))
                self.last_ik_error = float(np.linalg.norm(np.r_[translation_error, .2*rotation_error]))
                self.last_iterations = iteration
                if self.last_position_error <= tolerance and self.last_orientation_error <= orientation_tolerance:
                    return q
                if iteration == self.max_iterations:
                    break
                error = np.r_[translation_error, .2*rotation_error]
                jacobian = self.jacobian(q)[:, self.active]
                jacobian[3:] *= .2
                step = jacobian.T @ np.linalg.solve(jacobian@jacobian.T + self.damping**2*np.eye(6), error)
                step *= min(1., .12/max(np.max(np.abs(step)), 1e-12))
                q[self.active] = np.clip(q[self.active]+step, lower, upper)
            raise ValueError(f'Local IK rejected: position error {self.last_position_error:.4f} m, '
                             f'orientation error {self.last_orientation_error:.4f} rad')
