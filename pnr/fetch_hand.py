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

from .env import ALLEGRO_MOUNTS, PickNavReachEnv, object_source
from .hand import FINGERS, GRASP_POINT, MOUNT, TABLE_TOP, object_dirs, palm_rotation
from .planning import JointPlanner

BASE_XY = np.array([-.78, 0.])          # the planner's table approach position (world)
READY = np.array([-.15, 0., TABLE_TOP + .3, 0., 0., 0.])   # parked wrist: palm down over the table
IK_DAMPING = .05
RETREAT_X = -1.25          # world x the full task's base backs off to before tucking the arm
TUCK_HEIGHTS = (.90, .85)              # held-object heights to try (the goal's height, then lower)
TUCK_AHEAD = (.24, .20, .28)            # held object's distance ahead of the base centre
TUCK_TURNS = (-90, 90, -135, 135, 180, 0, -45, 45)   # hand turns about the vertical, degrees
TUCK_COMPACT = .36     # stop searching at a posture this compact: every arm link, hand link and the object within
                       # this distance of the base centre along x and along y (the base translates without turning
                       # through axis-aligned corridors)
TUCK_BUDGET = 24       # IK solves before settling for the most compact posture found (four times that: give up)
GOAL_HEIGHT = .90
ARM_LINKS = ('upperarm_roll_link', 'elbow_flex_link', 'forearm_roll_link', 'wrist_flex_link', 'wrist_roll_link')


def plan_tuck(planner, q, base_xy, flange, orientation, held, first=False):
    """Carry-posture joint targets for a held object, shared by training (FetchHandRig) and the hand system.
    Candidates: the held object TUCK_AHEAD of the base centre at TUCK_HEIGHTS, the hand turned about the
    vertical by one of TUCK_TURNS (palm tilt kept, so gravity pulls on the grasp as before), the torso at its
    lowest so the reach can raise the object; IK from the current configuration and the initial arm pose. A
    candidate counts if it keeps the robot and the held object (planner payload) off the table, maze and
    robot body and the straight joint-space move to it from `q` is clear too. Returns the most compact one
    (largest distance along x or y of arm links, hand links and object from the base centre: the maze carry
    is planned for the base's footprint), stopping at the first within TUCK_COMPACT; with no candidate, the search is
    repeated with the torso free (if its remaining range still brings the object to the goal height). At most
    TUCK_BUDGET IK solves once a candidate is found, four times that otherwise; the chosen candidate is in `plan_tuck.last`. `first`: the first candidate that counts, compact
    or not, and give up after TUCK_BUDGET solves (training: its carry runs in open space, and its vectorised
    environments all wait for the slowest one's search). None if nothing is found. The planner's fixed tuck (flange 24 cm ahead at 1.05 m, orientation kept) carried some
    grasps 23 cm above the goal or pressed the object into the robot's head; with the wrist-end mount the
    unturned hand points away from the robot, so few such poses exist close to the base."""
    q = np.asarray(q, dtype=float)
    env = planner.env
    seeds = [q, np.r_[q[:3], env.init_qpos[3:]]]
    low, high = planner.lower[0], planner.upper[0]
    offset = np.asarray(held) - np.asarray(flange)
    grasp = Rotation.from_quat(orientation)
    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(env.model.nbody)]
    bodies = [b for b, n in enumerate(names) if n.startswith('hand/') or n in [f'fetch/{a}' for a in ARM_LINKS]]
    bodies.append(env._obj_body)

    def extent(tuck):                           # the object placed by the planner's payload
        with planner.query():
            planner._put(tuck)
            return np.abs(env._query.xpos[bodies, :2] - np.asarray(base_xy)[:2]).max()

    best, solves, plan_tuck.last = (np.inf, None), 0, None
    give_up = TUCK_BUDGET * (1 if first else 4)
    for torso, heights in ((low, TUCK_HEIGHTS), (None, TUCK_HEIGHTS[:1])):
        for height in heights:
            for k, seed in enumerate(seeds):              # every turn from the current arm first: cheapest
                for ahead in TUCK_AHEAD:
                    for turn in TUCK_TURNS:
                        if solves >= (TUCK_BUDGET if best[1] is not None else give_up):
                            return best[1]
                        solves += 1
                        rotation = Rotation.from_euler('z', turn, degrees=True)
                        turned, quaternion = rotation.apply(offset), (rotation * grasp).as_quat()
                        target = np.r_[base_xy[0] + ahead, base_xy[1], height] - turned
                        try:
                            tuck = planner.ik(target, quaternion, seed=seed, torso=torso)
                        except ValueError:
                            continue
                        lift = tuck[planner.active[0]]
                        if not height - (lift - low) - .03 <= GOAL_HEIGHT <= height + (high - lift) + .03:
                            continue
                        size = extent(tuck)
                        if size >= best[0] or not planner.collision_free(tuck):
                            continue
                        with planner.query():
                            if not planner._edge_free(q, tuck):
                                continue
                        best = (size, tuck)
                        plan_tuck.last = dict(torso_free=torso is None, height=height, turn=turn, ahead=ahead, seed=k,
                                              extent=round(float(size), 3), solves=solves)
                        if first or size <= TUCK_COMPACT:
                            return tuck
        if best[1] is not None:
            return best[1]
    return None


# Side grasps: the forearm runs level with the palm, so grip at least this high above the table (it
# clears the table edge down to ~9 cm; the rig's palm alone clears at 6.6 cm).
SIDE_CLEARANCE = .10


def closed_form_angles(rotation):
    """Rig wrist yaw, pitch, roll of a palm rotation with pitch in [-90, 90] degrees."""
    r = rotation @ MOUNT.T
    return np.array([np.arctan2(r[1, 0], r[0, 0]), -np.arcsin(np.clip(r[2, 0], -1, 1)), np.arctan2(r[2, 1], r[2, 2])])


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

    def __init__(self, object_idx=14, physics=None, seed=0, settle_ticks=240, env=None, stepper=None, hand_mount='wrist'):
        self.bookkeeping = env is not None
        self.stepper = stepper or (env.step if env is not None else None)   # e.g. a controller's tick
        if env is None:
            env = PickNavReachEnv(seed=seed, object_idx=object_idx, physics=physics, effector='allegro', hand_mount=hand_mount)
        self.env = env
        self.physics, self.object_idx, self.object_name = env.physics, int(env.object_idx), env.object_name
        self.model, self.data = env.model, env.data
        self.planner = JointPlanner(env, seed=seed)
        object_dir, scale, _ = object_source(self.object_idx, object_dirs())
        mesh = trimesh.load(str(object_dir / 'textured.obj'), force='mesh')
        mesh.apply_scale(scale)
        points, _ = trimesh.sample.sample_surface(mesh, 64, seed=np.random.default_rng(0))   # as HandRig
        self.object_points = np.asarray(points, dtype=np.float64)
        self.object_extents = np.asarray(mesh.extents, dtype=np.float64)
        self._index()
        self.carrying = False
        self.side_clearance = SIDE_CLEARANCE
        if self.bookkeeping:
            self.rest_height = float(env.initial_object_pos[2])
            position, rotation = self.palm_pose()
            self.wrist_target = np.r_[position, closed_form_angles(rotation)]   # no target yet: the pitch <= 90 deg branch
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

    def wrist_to_joints(self, wrist, collision_free=False, seed=None):
        """Arm joint targets (full Fetch vector) that put the palm at rig wrist pose `wrist`: planner IK
        (torso included) refined by damped least squares. With `collision_free`, IK also starts from the
        initial arm pose and random arm poses until the arm clears the table, maze and object (the seed
        picks the IK branch). Raises ValueError if unreachable."""
        position, rotation = self._flange_target(np.asarray(wrist, dtype=float))
        quaternion = Rotation.from_matrix(rotation).as_quat()
        current = self.env.agent_qpos.astype(np.float64) if seed is None else np.asarray(seed, dtype=np.float64)
        seeds = [current, np.r_[current[:3], self.env.init_qpos[3:]]]
        rng = np.random.default_rng(0)
        if collision_free:
            for _ in range(8):
                seed = current.copy()
                seed[self.planner.active] = rng.uniform(self.planner.lower, self.planner.upper)
                seeds.append(seed)
        error = None
        for seed in seeds:
            try:
                q = self.planner.ik(position, quaternion, seed=seed, tolerance=.03)
            except ValueError as exc:
                error = exc
                continue
            with self.env.query():
                self.env.set_joint_positions(q)
                q[self._arm] = self._arm_step(np.asarray(wrist, dtype=float), iterations=30, start=q)
            if not collision_free or self.planner.collision_free(q, include_object=True):
                return q
        raise error or ValueError('No collision-free IK solution for the hand pose')

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
        position, axes = ALLEGRO_MOUNTS[self.env.hand_mount]
        palm = palm_rotation(*wrist[3:])
        flange = palm @ axes.T
        return wrist[:3] - flange @ np.asarray(position), flange

    def set_wrist(self, wrist):
        """Place the arm so the palm is at the rig wrist pose `wrist`: a collision-free IK solution when
        there is one (as the task's planned pregrasp uses), else the planner's nearest."""
        wrist = np.asarray(wrist, dtype=float)
        try:
            q = self.wrist_to_joints(wrist, collision_free=True)
        except ValueError:
            q = self.env.agent_qpos.astype(np.float64)
            q[self.planner.active] = np.clip(q[self.planner.active], self.planner.lower, self.planner.upper)
        self.env._set_qpos(q)
        self.env.last_target = q.copy()
        self.wrist_target = wrist.copy()
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
        """Drive the base along `path` (base xy offsets), turned in 90 degree steps so it heads away from
        the table into the start room. With `task_carry`, first the full task's post-grasp sequence
        (pnr/hand_system.py): retreat from the table and tuck the arm into the carry posture, one planned
        joint target per decision; the returned path then begins with that many zero offsets."""
        turns = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in np.arange(4) * np.pi / 2]
        path = min((np.asarray(path) @ turn.T for turn in turns), key=lambda p: p[:, 0].max())
        start = self.env.last_target.copy()
        self._carry_plan = self._retreat_and_tuck(start) if getattr(self, 'task_carry', False) else []
        self.carrying, self._carry_k = True, 0
        self._carry_q = (self._carry_plan[-1] if self._carry_plan else start).copy()
        self._carry_wrist = self.wrist_target.copy()
        return np.vstack([np.zeros((len(self._carry_plan), 2)), path]) if self._carry_plan else path

    def _retreat_and_tuck(self, q):
        """Joint targets, one per decision, for the hand system's post-grasp sequence: the base backs off to
        x = -1.25 at 1.8 cm per decision, then the arm moves (0.012 rad per decision) to the carry posture
        (`plan_tuck`: object at or below the goal's height close to the base, hand turned about the vertical), each
        followed by 10 settling decisions."""
        plan, q = [], np.asarray(q, dtype=float).copy()
        goal_x = RETREAT_X - self.env.root_position[0]
        for t in np.linspace(0, 1, int(np.ceil(abs(goal_x - q[0]) / .018)) + 1)[1:]:
            target = q.copy()
            target[0] = q[0] + (goal_x - q[0]) * t
            plan.append(target)
        if plan:
            q = plan[-1].copy()
        plan += [q.copy()] * 10
        tuck = self._tuck_joints(q)
        if tuck is not None:
            steps = int(np.ceil(np.abs(tuck[self._arm] - q[self._arm]).max() / .012))
            plan += [q + (tuck - q) * t for t in np.linspace(0, 1, steps + 1)[1:]]
            plan += [tuck.copy()] * 10
        return plan

    def _tuck_joints(self, q):
        """The carry posture's joint targets (`plan_tuck`, as HandPlannerSystem.prepare_carry), or None."""
        env = self.env
        held = env.observe()['object_pos']          # the retreat moves the base in x only: same height
        with env.query():
            env.set_joint_positions(q)
            flange, orientation = env.ee_pose()
        self.planner.set_payload()                  # the grasp transform, for checking the held object
        try:
            return plan_tuck(self.planner, q, env.root_position[:2] + q[:2], flange, orientation, held, first=True)
        finally:
            self.planner.payload = None

    # ---------------- control ----------------
    def tick(self):
        self.env._physics_tick()

    def step(self, wrist_target, finger_target):
        self.wrist_target = np.asarray(wrist_target, dtype=float)
        target = self.env.last_target.copy()
        if self.carrying:
            if self._carry_k < len(self._carry_plan):           # retreat and tuck first
                target = self._carry_plan[self._carry_k].copy()
            else:
                target = self._carry_q.copy()
                target[:2] += self.wrist_target[:2] - self._carry_wrist[:2]
            self._carry_k += 1
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
