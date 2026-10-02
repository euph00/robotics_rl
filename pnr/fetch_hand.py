"""The Allegro hand on the Fetch at the course table, behind HandRig's interface (pnr/hand.py).

HandGraspEnv, its scripted teachers and trained hand policies run unchanged on the real robot:
the rig's six wrist coordinates (palm origin x, y, z; palm yaw, pitch, roll) become a gripper_link
pose target (the palm is bolted to the flange, pnr/env.py's 'allegro' effector), and a damped
least-squares step turns it into arm joint targets every decision. The arm then moves under the
Fetch's joint limits instead of the rig's idealised wrist. During the carry the arm holds still
and the base drives the carry path, so the grip is tested against real base accelerations.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .env import ALLEGRO_MOUNT, PickNavReachEnv
from .hand import FINGERS, GRASP_POINT, MOUNT, TABLE_TOP, object_dirs, palm_rotation
from .planning import JointPlanner

BASE_XY = np.array([-.78, 0.])          # the planner's table approach position (world)
READY = np.array([-.15, 0., TABLE_TOP + .3, 0., 0., 0.])   # parked wrist: palm down over the table
IK_DAMPING = .05


def wrist_angles(rotation, near):
    """Rig wrist yaw, pitch, roll of a palm rotation (palm_rotation's convention), nearest to `near`.

    Damped least squares from `near` on the rotation error: continuous through pitch = +-90 degrees,
    where yaw and roll share an axis and the closed-form extraction jumps between equivalent pairs."""
    angles = np.asarray(near, dtype=float).copy()
    for _ in range(12):
        current = palm_rotation(*angles)
        error = Rotation.from_matrix(rotation @ current.T).as_rotvec()           # world frame
        if np.linalg.norm(error) < 1e-7:
            break
        yaw, pitch = angles[0], angles[1]
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        axes = np.column_stack([[0, 0, 1], rz @ [0, 1, 0], rz @ ry @ [1, 0, 0]])   # world rate axes
        angles += axes.T @ np.linalg.solve(axes @ axes.T + 1e-6 * np.eye(3), error)
    return angles


class FetchHandRig:
    """The Fetch with the Allegro at the course table. By default it builds its own scene (training);
    given `env` (a PickNavReachEnv with effector='allegro', e.g. mid-task) it wraps that one as it is,
    and steps through env.step so the task's lift, hold and drop bookkeeping stays exact."""

    def __init__(self, object_idx=14, physics=None, seed=0, settle_ticks=240, env=None, stepper=None):
        self.bookkeeping = env is not None
        self.stepper = stepper or (env.step if env is not None else None)   # e.g. a controller's tick
        if env is None:
            env = PickNavReachEnv(seed=seed, object_idx=object_idx, physics=physics, effector='allegro')
        self.env = env
        self.physics, self.object_idx, self.object_name = env.physics, int(env.object_idx), env.object_name
        self.model, self.data = env.model, env.data
        self.planner = JointPlanner(env, seed=seed)
        mesh = trimesh.load(str(object_dirs()[self.object_idx] / 'textured.obj'), force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, 64, seed=np.random.default_rng(0))   # as HandRig
        self.object_points = np.asarray(points, dtype=np.float64)
        self.object_extents = np.asarray(mesh.extents, dtype=np.float64)
        self._index()
        self.carrying = False
        if self.bookkeeping:
            self.rest_height = float(env.initial_object_pos[2])
            self.wrist_target = np.zeros(6)
            self.wrist_target = self.wrist()
            return

        q = env.agent_qpos.astype(np.float64)
        q[:2] = BASE_XY - env.root_position[:2]
        env._set_qpos(q)
        env.last_target = q.copy()
        self.set_wrist(READY)
        self.data.ctrl[:] = self.finger_open
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_ticks):
            self.tick()
        self.data.qvel[self._obj_dof:self._obj_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.rest_height = self.object_pose()[0][2]
        self.settled = self.snapshot()

    def wrist_to_joints(self, wrist):
        """Arm joint targets (full Fetch vector) that put the palm at rig wrist pose `wrist`: planner IK
        (torso included), refined by damped least squares. Raises ValueError if unreachable."""
        position, rotation = self._flange_target(np.asarray(wrist, dtype=float))
        q = self.planner.ik(position, Rotation.from_matrix(rotation).as_quat(), tolerance=.03)
        with self.env.query():
            self.env.set_joint_positions(q)
            q[self._arm] = self._arm_step(np.asarray(wrist, dtype=float), iterations=30, start=q)
        return q

    def _index(self):
        m, env = self.model, self.env
        self._obj_q, self._obj_dof, self._obj_body = env._obj_qpos, env._obj_dof, env._obj_body
        joints = [m.actuator_trnid[a, 0] for a in range(m.nu)]
        self._fq = np.array([m.jnt_qposadr[j] for j in joints])
        self._fd = np.array([m.jnt_dofadr[j] for j in joints])
        self.finger_low, self.finger_high, self.finger_open = env.hand_low, env.hand_high, env.hand_open
        self._palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand/palm')
        self._tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'hand/{f}_tip') for f in FINGERS]
        self._finger_of_body = env._hand_finger_of_body
        self._hand_bodies = set(self._finger_of_body)
        self._object_geoms = set(env._object_geoms)
        self._table_geom = next(g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('table'))
        self._arm = env.arm_slots

    # ---------------- wrist <-> arm ----------------
    def _flange_target(self, wrist):
        rotation = palm_rotation(*wrist[3:])
        return wrist[:3] - rotation @ np.asarray(ALLEGRO_MOUNT), rotation

    def set_wrist(self, wrist):
        """Place the arm (planner IK, torso included) so the palm is at the rig wrist pose `wrist`."""
        position, rotation = self._flange_target(np.asarray(wrist, dtype=float))
        try:
            q = self.planner.ik(position, Rotation.from_matrix(rotation).as_quat(), tolerance=.03)
        except ValueError:
            q = self.env.agent_qpos.astype(np.float64)
            q[self.planner.active] = np.clip(q[self.planner.active], self.planner.lower, self.planner.upper)
        self.env._set_qpos(q)
        q[self._arm] = self._arm_step(np.asarray(wrist, dtype=float), iterations=30)   # refine the planner's solution
        self.env._set_qpos(q)
        self.env.last_target = q.copy()
        self.wrist_target = np.asarray(wrist, dtype=float).copy()
        self.carrying = False

    def _arm_step(self, wrist, iterations=3, start=None):
        """Arm joint targets that put the palm at `wrist`: damped least squares from the measured arm
        (or from `start`)."""
        position, rotation = self._flange_target(wrist)
        env = self.env
        with env.query():
            q = (env.agent_qpos if start is None else start).astype(np.float64).copy()
            for _ in range(iterations):
                env.set_joint_positions(q)
                pos, quat = env.ee_pose()
                error = np.r_[position - pos, Rotation.from_matrix(rotation @ Rotation.from_quat(quat).as_matrix().T).as_rotvec()]
                jac = env.ee_jacobian(q)[:, self._arm]
                q[self._arm] += jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING ** 2 * np.eye(6), error)
                q[self._arm] = np.clip(q[self._arm], env.joint_lower[self._arm], env.joint_upper[self._arm])
        return q[self._arm]

    def begin_carry(self, path):
        """Hold the arm and drive the base along `path` (base xy offsets), turned in 90 degree steps so
        it heads away from the table into the start room."""
        turns = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in np.arange(4) * np.pi / 2]
        path = min((np.asarray(path) @ turn.T for turn in turns), key=lambda p: p[:, 0].max())
        self.carrying, self._carry_q, self._carry_wrist = True, self.env.last_target.copy(), self.wrist_target.copy()
        return path

    # ---------------- control ----------------
    def tick(self):
        self.env._physics_tick()

    def step(self, wrist_target, finger_target):
        self.wrist_target = np.asarray(wrist_target, dtype=float)
        target = self.env.last_target.copy()
        if self.carrying:
            target = self._carry_q.copy()
            target[:2] += self.wrist_target[:2] - self._carry_wrist[:2]
        else:
            target[self._arm] = self._arm_step(self.wrist_target)
        action = np.r_[target, np.clip(finger_target, self.finger_low, self.finger_high)]
        if self.bookkeeping:          # inside the task: the env's own step keeps its counters
            self.stepper(action)
            return
        # Training: targets and ticks only; HandGraspEnv measures contacts, lift and reward itself.
        self.env.command(action)
        for _ in range(self.env.substeps):
            self.env._physics_tick()

    # ---------------- state ----------------
    def snapshot(self):
        copy = mujoco.MjData(self.model)
        mujoco.mj_copyData(copy, self.model, self.data)
        return copy, self.env.last_target.copy(), self.env._reference.copy(), self.wrist_target.copy()

    def restore(self, snapshot):
        data, target, reference, wrist = snapshot
        mujoco.mj_copyData(self.data, self.model, data)
        self.env.last_target, self.env._reference, self.wrist_target = target.copy(), reference.copy(), wrist.copy()
        self.carrying = False

    # ---------------- measurements (HandRig's) ----------------
    def wrist(self):
        position, rotation = self.palm_pose()
        return np.r_[position, wrist_angles(rotation, self.wrist_target[3:])]

    def palm_pose(self):
        return self.data.xpos[self._palm].copy(), self.data.xmat[self._palm].reshape(3, 3).copy()

    def palm_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._palm, velocity, 0)
        return velocity[3:]

    def grasp_point(self):
        position, rotation = self.palm_pose()
        return position + rotation @ GRASP_POINT

    def fingertips(self):
        return np.array([self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ (0, 0, .025 if f != 'th' else .04)
                         for b, f in zip(self._tips, FINGERS)])

    def fingers(self):
        return self.data.qpos[self._fq].copy(), self.data.qvel[self._fd].copy()

    def contacts(self):
        """Per finger (ff, mf, rf, th) and palm touching the object with positive normal force; hand-table
        contact; deepest hand-object penetration (m)."""
        m, d = self.model, self.data
        fingers, palm, table, depth = np.zeros(4, bool), False, False, 0.
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            other = b2 if c.geom1 in self._object_geoms else b1 if c.geom2 in self._object_geoms else None
            if other is not None and other in self._hand_bodies:
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] <= 0:
                    continue
                depth = max(depth, -c.dist)
                finger = self._finger_of_body.get(other)
                if finger is None:
                    palm = True
                else:
                    fingers[finger] = True
            elif self._table_geom in (c.geom1, c.geom2) and (b1 in self._hand_bodies or b2 in self._hand_bodies):
                table = True
        return fingers, palm, table, depth

    def object_pose(self):
        q = self.data.qpos[self._obj_q:self._obj_q + 7]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, q[3:7])
        return q[:3].copy(), rotation.reshape(3, 3)

    def object_velocity(self):
        return self.data.qvel[self._obj_dof:self._obj_dof + 6].copy()

    def object_points_world(self):
        position, rotation = self.object_pose()
        return self.object_points @ rotation.T + position

    def render(self, width=640, height=480, distance=1.2, azimuth=150, elevation=-25):
        if getattr(self, '_renderer', None) is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.object_pose()[0]
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()
