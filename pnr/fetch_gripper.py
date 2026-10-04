"""The Fetch's parallel gripper at the course table, behind the same rig interface as FetchHandRig.

HandGraspEnv, its trainer (scripts/train_hand.py) and its teachers then run both end effectors through
one learning recipe (docs/project_spec.md, fairness controls). The gripper's 'palm' frame is its
gripper_link: the rig's six wrist coordinates are the grasp centre between the pads (the gripper_link
origin) and the palm angles of pnr/hand.py's convention, in which zero angles point the approach axis
straight down with the jaws closing along world y. The two finger joints are the 'fingers' (0 closed,
0.05 m open each); they are Fetch joint targets, not separate actuators. Grasped: both pads touch the
object with positive normal force.
"""
from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from .env import PickNavReachEnv, object_source
from .fetch_hand import BASE_XY, READY, FetchHandRig, closed_form_angles
from .hand import TABLE_TOP, object_dirs, palm_rotation
from .planning import JointPlanner

PAD_OFFSET = .008          # pad box centre from the finger link origin, towards the other pad (m)
PAD_HALF_LENGTH = .0315    # pads reach this far below the grasp centre along the approach axis


class FetchGripperRig(FetchHandRig):
    """The Fetch with its parallel gripper; see the module docstring. `env` wraps a task environment
    (effector='gripper') mid-task, as FetchHandRig does."""

    n_tips = 2

    def __init__(self, object_idx=14, physics=None, seed=0, settle_ticks=240, env=None, stepper=None, **_):
        self.bookkeeping = env is not None
        self.stepper = stepper or (env.step if env is not None else None)
        if env is None:
            env = PickNavReachEnv(seed=seed, object_idx=object_idx, physics=physics, effector='gripper')
        self.env = env
        self.physics, self.object_idx, self.object_name = env.physics, int(env.object_idx), env.object_name
        self.model, self.data = env.model, env.data
        self.planner = JointPlanner(env, seed=seed)
        object_dir, scale, _ = object_source(self.object_idx, object_dirs())
        mesh = trimesh.load(str(object_dir / 'textured.obj'), force='mesh')
        mesh.apply_scale(scale)
        points, _ = trimesh.sample.sample_surface(mesh, 64, seed=np.random.default_rng(0))
        self.object_points = np.asarray(points, dtype=np.float64)
        self.object_extents = np.asarray(mesh.extents, dtype=np.float64)
        self._index()
        self.carrying = False
        if self.bookkeeping:
            self.rest_height = float(env.initial_object_pos[2])
            position, rotation = self.palm_pose()
            self.wrist_target = np.r_[position, closed_form_angles(rotation)]
            return
        q = env.agent_qpos.astype(np.float64)
        q[:2] = BASE_XY - env.root_position[:2]
        q[self._fingers] = self.finger_open
        env._set_qpos(q)
        env.last_target = q.copy()
        self.set_wrist(READY)
        for _ in range(settle_ticks):
            self.tick()
        self.data.qvel[self._obj_dof:self._obj_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.rest_height = self.object_pose()[0][2]
        self.settled = self.snapshot()

    def _index(self):
        m, env = self.model, self.env
        self._obj_q, self._obj_dof, self._obj_body = env._obj_qpos, env._obj_dof, env._obj_body
        self._fingers = np.asarray(env.finger_slots)
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'fetch/{n}')
                  for n in ('r_gripper_finger_joint', 'l_gripper_finger_joint')]
        self._fq = np.array([m.jnt_qposadr[j] for j in joints])
        self._fd = np.array([m.jnt_dofadr[j] for j in joints])
        self.finger_low = env.joint_lower[self._fingers].astype(np.float64)
        self.finger_high = env.joint_upper[self._fingers].astype(np.float64)
        self.finger_open = self.finger_high.copy()
        # Finger action step: the hand's 0.08 rad is ~5% of a finger joint's range; the same share of the
        # jaw range (2.5 mm per decision), so both close in about 20 decisions.
        self.finger_step = .05 * (self.finger_high - self.finger_low)
        self._palm = env._ee_body
        self._pads = list(env._finger_bodies)                  # right, left
        self._finger_of_body = {b: k for k, b in enumerate(self._pads)}
        gripper = {self._palm, *self._pads}
        self._hand_bodies = gripper
        self._object_geoms = set(env._object_geoms)
        self._table_geom = next(g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('table'))
        self._arm = env.arm_slots

    # ---------------- wrist <-> arm ----------------
    def _flange_target(self, wrist):
        return np.asarray(wrist[:3], dtype=float), palm_rotation(*wrist[3:])

    def pregrasp(self, yaw, height, offset=(0., 0.)):
        """Wrist pose with the grasp centre `height` above the object's centre, approach straight down."""
        centre = self.object_pose()[0]
        return np.array([centre[0] + offset[0], centre[1] + offset[1], centre[2] + height, yaw, 0., 0.])

    def grasped(self, fingers):
        return bool(fingers.all())

    # ---------------- control ----------------
    def step(self, wrist_target, finger_target):
        self.wrist_target = np.asarray(wrist_target, dtype=float)
        target = self.env.last_target.copy()
        if self.carrying:
            if self._carry_k < len(self._carry_plan):
                target = self._carry_plan[self._carry_k].copy()
            else:
                target = self._carry_q.copy()
                target[:2] += self.wrist_target[:2] - self._carry_wrist[:2]
            self._carry_k += 1
        else:
            target[self._arm] = self._arm_step(self.wrist_target)
        target[self._fingers] = np.clip(finger_target, self.finger_low, self.finger_high)
        if self.bookkeeping:
            self.stepper(target)
            return
        self.env.command(target)
        for _ in range(self.env.substeps):
            self.env._physics_tick()

    # ---------------- measurements ----------------
    def grasp_point(self):
        return self.palm_pose()[0]

    def fingertips(self):
        """Pad centres (right, left)."""
        d = self.data
        return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ (0., sign * PAD_OFFSET, 0.)
                         for b, sign in zip(self._pads, (-1., 1.))])

    def contacts(self):
        """Pads (right, left) and gripper_link touching the object with positive normal force; gripper-table
        contact; deepest gripper-object penetration (m)."""
        m, d = self.model, self.data
        fingers, palm, table, depth = np.zeros(2, bool), False, False, 0.
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
