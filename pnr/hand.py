"""Dexterous-hand test rig: an Allegro hand on a six-axis wrist above the course table.

Same physics as the task environment (pnr/env.py): 1/240 s ticks, five ticks per
decision, elliptic cones with impratio 10 and noslip, 2 dt contact time constant, the
course table and YCB collision parts. The hand's geoms get the Fetch fingers' friction
(0.25 against objects), so neither effector has a friction advantage.

The wrist (x, y, z slides; yaw, pitch, roll hinges) follows the course motor law used for
the Fetch arm: computed torque that closes 30% of the target error per tick. The fingers
keep Menagerie's position actuators (kp = 1 N m/rad), i.e. a realistically weak hand.
At zero wrist angles the palm faces down and the fingers point along +x.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh

from .env import (BITS, CONTACT_SOLIMP, CONTACT_SOLREF, COLORS, FRICTION, FRICTION_CONE, IMPRATIO,
                  NOSLIP_ITERATIONS, ROOT, UPSTREAM, _object_parts)

ALLEGRO = ROOT / 'assets/hands/wonik_allegro/right_hand.xml'
WRIST_GAIN = 0.3
TICKS = 5
TABLE_TOP = .625
FINGERS = ('ff', 'mf', 'rf', 'th')
# Palm-frame point between the fingers where a power grasp closes (grasp side is palm +x).
GRASP_POINT = np.array([.05, 0., .03])
HAND_BITS = (2 | 8, 1 | 8)    # collides with objects, table and itself (fingers); not the floor


def object_dirs():
    return sorted(path for path in (UPSTREAM / 'assets/ycb_objects').iterdir() if path.is_dir())


class HandRig:
    """One compiled scene: table, one object, wrist-mounted hand."""

    def __init__(self, object_idx=14, settle_ticks=480):
        self.object_idx = int(object_idx)
        object_dir = object_dirs()[self.object_idx]
        self.object_name = object_dir.name
        mesh = trimesh.load(str(object_dir / 'textured.obj'), force='mesh')
        rng = np.random.default_rng(0)
        points, faces = trimesh.sample.sample_surface(mesh, 64, seed=rng)
        self.object_points = np.asarray(points, dtype=np.float64)    # canonical frame (mesh origin = COM)
        self.object_extents = np.asarray(mesh.extents, dtype=np.float64)

        spec = mujoco.MjSpec()
        spec.option.timestep = 1 / 240
        spec.option.gravity = [0, 0, -9.8]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = getattr(mujoco.mjtCone, 'mjCONE_' + FRICTION_CONE.upper())
        spec.option.impratio = IMPRATIO
        spec.option.noslip_iterations = NOSLIP_ITERATIONS
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
        world = spec.worldbody
        world.add_light(pos=[0, 0, 3], dir=[.2, .1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        diffuse=[.5, .5, .5], castshadow=True)

        def geom(parent, group, **kwargs):
            contype, conaffinity = BITS['static' if group == 'table' else group]
            g = parent.add_geom(contype=contype, conaffinity=conaffinity, condim=3,
                                friction=[FRICTION[group], .005, .0001], **kwargs)
            g.solref, g.solimp = list(CONTACT_SOLREF), list(CONTACT_SOLIMP)
            if group in COLORS:
                g.rgba = COLORS[group]
            g.name = f'{group}_{len(spec.geoms)}'
            return g

        geom(world, 'floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, .1])
        geom(world, 'table', type=mujoco.mjtGeom.mjGEOM_BOX, size=[.5, .75, .025], pos=[0, 0, .6])
        parts, mass, inertia = _object_parts(object_dir)
        body = world.add_body(name='object', pos=[0, 0, 1.0])
        body.add_freejoint(name='object_free')
        body.mass, body.fullinertia, body.explicitinertial = mass, inertia, True
        body.ipos = [0., 0., 0.]
        for i, part in enumerate(parts):
            spec.add_mesh(name=f'object_part_{i}', file=str(part))
            geom(body, 'object', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'object_part_{i}', density=0)

        wrist = world.add_body(name='wrist')
        for name, kind, axis in (('wx', 'slide', [1, 0, 0]), ('wy', 'slide', [0, 1, 0]), ('wz', 'slide', [0, 0, 1]),
                                 ('wyaw', 'hinge', [0, 0, 1]), ('wpitch', 'hinge', [0, 1, 0]), ('wroll', 'hinge', [1, 0, 0])):
            wrist.add_joint(name=name, type=getattr(mujoco.mjtJoint, f'mjJNT_{kind.upper()}'), axis=axis)
        # Wrist carrier mass so the six joints have well-defined inertia (a Fetch forearm stand-in).
        wrist.mass, wrist.inertia, wrist.explicitinertial = .5, [1e-3, 1e-3, 1e-3], True
        wrist.ipos = [0., 0., 0.]
        hand = mujoco.MjSpec.from_file(str(ALLEGRO))
        palm = hand.worldbody.first_body()
        palm.quat = [1, 0, 0, 0]
        mount = wrist.add_frame(quat=[np.sqrt(.5), 0, np.sqrt(.5), 0])   # palm +x (grasp side) -> world -z
        mount.attach_body(palm, 'hand/', '')
        for g in spec.geoms:
            if g.name.startswith(('floor', 'table', 'object')):
                continue
            if g.contype or g.conaffinity:          # hand collision geoms; visual meshes stay non-colliding
                g.contype, g.conaffinity, g.condim = HAND_BITS[0], HAND_BITS[1], 3
                g.friction = [FRICTION['robot'], .005, .0001]
                g.solref, g.solimp = list(CONTACT_SOLREF), list(CONTACT_SOLIMP)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index()
        self._mass = np.zeros((self.model.nv, self.model.nv))

        # Settle the object on the table with the hand parked high and open.
        self.set_wrist(np.array([-.3, 0., 1.2, 0., 0., 0.]))
        self.data.ctrl[:] = self.finger_open
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_ticks):
            self.tick()
        self.data.qvel[self._obj_dof:self._obj_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.rest_height = self.object_pose()[0][2]
        self.settled = self.snapshot()

    def _index(self):
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        wrist = [jid(n) for n in ('wx', 'wy', 'wz', 'wyaw', 'wpitch', 'wroll')]
        self._wq = np.array([m.jnt_qposadr[j] for j in wrist])
        self._wd = np.array([m.jnt_dofadr[j] for j in wrist])
        self.finger_actuators = np.arange(m.nu)
        fj = [m.actuator_trnid[a, 0] for a in self.finger_actuators]
        self._fq = np.array([m.jnt_qposadr[j] for j in fj])
        self._fd = np.array([m.jnt_dofadr[j] for j in fj])
        self.finger_low, self.finger_high = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.finger_open = np.clip(np.zeros(m.nu), self.finger_low, self.finger_high)
        free = jid('object_free')
        self._obj_q, self._obj_dof = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self._obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'object')
        self._palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand/palm')
        self._tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'hand/{f}_tip') for f in FINGERS]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(m.nbody)]
        self._finger_of_body = {b: next((i for i, f in enumerate(FINGERS) if n.startswith(f'hand/{f}_')), None)
                                for b, n in enumerate(names)}
        self._object_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self._obj_body}
        self._table_geom = next(g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('table'))
        self._hand_bodies = {b for b, n in enumerate(names) if n.startswith('hand/')}

    # ---------------- control ----------------
    def set_wrist(self, q):
        self.data.qpos[self._wq] = q
        self.data.qvel[self._wd] = 0
        self.wrist_target = np.asarray(q, dtype=float).copy()
        mujoco.mj_kinematics(self.model, self.data)

    def tick(self):
        m, d, dt = self.model, self.data, self.model.opt.timestep
        mujoco.mj_step1(m, d)
        command = WRIST_GAIN * (self.wrist_target - d.qpos[self._wq]) / dt
        mujoco.mj_fullM(m, d, self._mass)
        mass = self._mass[np.ix_(self._wd, self._wd)]
        d.qfrc_applied[self._wd] = mass @ (command - d.qvel[self._wd]) / dt + d.qfrc_bias[self._wd]
        mujoco.mj_step2(m, d)

    def step(self, wrist_target, finger_target):
        self.wrist_target = np.asarray(wrist_target, dtype=float)
        self.data.ctrl[:] = np.clip(finger_target, self.finger_low, self.finger_high)
        for _ in range(TICKS):
            self.tick()

    # ---------------- state ----------------
    def snapshot(self):
        copy = mujoco.MjData(self.model)
        mujoco.mj_copyData(copy, self.model, self.data)
        return copy, self.wrist_target.copy()

    def restore(self, snapshot):
        data, target = snapshot
        mujoco.mj_copyData(self.data, self.model, data)
        self.wrist_target = target.copy()

    def object_pose(self):
        q = self.data.qpos[self._obj_q:self._obj_q + 7]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, q[3:7])
        return q[:3].copy(), rotation.reshape(3, 3)

    def object_velocity(self):
        return self.data.qvel[self._obj_dof:self._obj_dof + 6].copy()

    def palm_pose(self):
        return self.data.xpos[self._palm].copy(), self.data.xmat[self._palm].reshape(3, 3).copy()

    def grasp_point(self):
        position, rotation = self.palm_pose()
        return position + rotation @ GRASP_POINT

    def fingertips(self):
        # Tip bodies sit at the distal joints; the fingertip pads are ~2.5 cm further along the finger.
        return np.array([self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ (0, 0, .025 if f != 'th' else .04)
                         for b, f in zip(self._tips, FINGERS)])

    def wrist(self):
        return self.data.qpos[self._wq].copy()

    def fingers(self):
        return self.data.qpos[self._fq].copy(), self.data.qvel[self._fd].copy()

    def contacts(self):
        """Per finger (ff, mf, rf, th) and palm: touching the object with positive normal force;
        plus hand-table contact and the deepest hand-object penetration (m)."""
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

    def object_points_world(self):
        position, rotation = self.object_pose()
        return self.object_points @ rotation.T + position

    def render(self, width=640, height=480, distance=.75, azimuth=150, elevation=-25):
        if getattr(self, '_renderer', None) is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.object_pose()[0]
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()


def pregrasp_wrist(rig, yaw, height=.12, offset=(0., 0.)):
    """Wrist pose with the grasp point `height` above the object's centre, palm down, given yaw."""
    centre = rig.object_pose()[0]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    mount = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])   # palm frame in wrist frame (see HandRig mount)
    palm_offset = rotation @ mount @ GRASP_POINT
    target = centre + [offset[0], offset[1], height]
    return np.array([*(target - palm_offset), yaw, 0., 0.])
