"""Small headless adaptation of the pinned course environment.

Source: NUS-LinS-Lab/Pick-Nav-Reach-Default-Project, commit 54ab2d1.
The robot, objects, maze generator, motor gains/forces, gravity, five-tick
absolute-position actions, and proximity criterion retain their course values.
Repairs are client ownership, paths, seeding, reset/settling, and headless IO.
No weld, object attachment, friction override, or hidden controller is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import random
import tempfile
import threading
import time

import numpy as np
import pybullet as pb
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
import trimesh

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "Pick-Nav-Reach-Default-Project"
UPSTREAM_COMMIT = "54ab2d1faa54d1aa7e2738f114531a76c6514cd3"
_GLOBAL_RNG_LOCK = threading.Lock()


def _load_maze_helpers():
    spec = importlib.util.spec_from_file_location("pnr_upstream_maze", UPSTREAM / "maze_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _preserve_global_rng():
    # The course helper seeds process-global random and numpy. Restrict that
    # behavior to a short serialized operation; do not disturb learner RNGs.
    with _GLOBAL_RNG_LOCK:
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


@dataclass(frozen=True)
class SavedState:
    """Simulator snapshot; valid only in its originating live environment."""
    client_id: int
    state_id: int
    target: tuple
    step_count: int
    physics_steps: int
    stable_hold_steps: int
    ever_lifted: bool
    ever_stably_held: bool
    had_grasp: bool
    drop_events: int


class PickNavReachEnv:
    """Native course interface, with deterministic table resets and RGB render.

    `step(action)` accepts 15 absolute position targets and returns `(obs, info)`.
    This intentionally leaves Gymnasium/rewards to a separate learning adapter.
    With no maze_seed, course-legacy-v1 preserves the original seeded scene.
    Explicit maze_seed selects independent-maze-v1; seed then controls object
    point sampling independently of layout. reset(seed=..., maze_seed=...)
    rebuilds only if a seed changes, otherwise restores the settled snapshot.
    """

    root_position = np.array([-2.0, 0.0, 0.0])
    substeps = 5
    max_force = 2_000_000
    seed_mode = "course-legacy-v1"
    position_gains = [0.3] * 13 + [0.03, 0.03]

    def __init__(self, seed=0, object_idx=5, gui=False, settle_steps=480, real_time=False,
                 maze_seed=None, seed_mode=None):
        self.seed = int(seed)
        self.seed_mode = seed_mode or ("course-legacy-v1" if maze_seed is None else "independent-maze-v1")
        if self.seed_mode not in ("course-legacy-v1", "independent-maze-v1"):
            raise ValueError(f"Unknown scene seed mode: {self.seed_mode}")
        if (maze_seed is None) != (self.seed_mode == "course-legacy-v1"):
            raise ValueError("independent-maze-v1 requires maze_seed; course-legacy-v1 forbids it")
        self.maze_seed = None if maze_seed is None else int(maze_seed)
        if self.maze_seed is not None and not 0 <= self.maze_seed < 2**32:
            raise ValueError("maze_seed must fit an unsigned 32-bit seed")
        self.object_idx = int(object_idx)
        self.gui = bool(gui)
        self.real_time = bool(real_time)
        self.settle_steps = int(settle_steps)
        self._closed = False
        self._saved_states = {}
        cache_dir = ROOT / ".cache" / "pnr"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="scene-", dir=cache_dir)
        self.maze_urdf_path = Path(self._temporary.name) / "maze.urdf"
        self.p = BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        self.client = self.p
        self.pb_physics_client = self.p._client
        try:
            self._build_scene()
        except BaseException:
            self.close()
            raise

    def _build_scene(self):
        self.p.resetSimulation()
        self._saved_states.clear()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.8)
        self.p.configureDebugVisualizer(lightPosition=[0, 0, 10], rgbBackground=[1, 1, 1])
        self.plane_id = self.p.loadURDF("plane.urdf")
        self.robot_id = self.p.loadURDF(
            str(UPSTREAM / "assets/fetch/fetch.urdf"), self.root_position, [0, 0, 0, 1], useFixedBase=True
        )
        joint_info = [self.p.getJointInfo(self.robot_id, j) for j in range(self.p.getNumJoints(self.robot_id))]
        controlled = [j for j in joint_info if j[2] in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC)]
        self.joint_indices = [j[0] for j in controlled]
        self.joint_names = [j[1].decode() for j in controlled]
        self.joint_name_to_index = {j[1].decode(): j[0] for j in joint_info}
        self.joint_name_to_slot = {name: slot for slot, name in enumerate(self.joint_names)}
        self.link_name_to_index = {j[12].decode(): j[0] for j in joint_info}
        self.link_name_to_index["root"] = -1
        self.joint_lower = np.array([j[8] for j in controlled], dtype=np.float32)
        self.joint_upper = np.array([j[9] for j in controlled], dtype=np.float32)
        self.joint_ranges = self.joint_upper - self.joint_lower
        self.init_qpos = np.clip(np.zeros(len(controlled)), self.joint_lower, self.joint_upper)
        self.rest_poses = self.init_qpos.copy()
        self.arm_slots = np.array([self.joint_name_to_slot[name] for name in (
            "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
            "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")], dtype=int)
        self.finger_slots = np.array([self.joint_name_to_slot[name] for name in
            ("r_gripper_finger_joint", "l_gripper_finger_joint")], dtype=int)
        self.finger_links = tuple(self.joint_indices[i] for i in self.finger_slots)
        self.ee_link = self.link_name_to_index["gripper_link"]
        if len(controlled) != 15:
            raise RuntimeError(f"Pinned Fetch must have 15 controlled joints, got {len(controlled)}")
        for joint in self.joint_indices:
            self.p.setJointMotorControl2(self.robot_id, joint, pb.VELOCITY_CONTROL, force=0.0)
        self._set_qpos(self.init_qpos)
        self.table_id = self.p.loadURDF("table/table.urdf", baseOrientation=self.p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
        self.table_height = float(max(self.p.getAABB(self.table_id, link)[1][2]
            for link in range(-1, self.p.getNumJoints(self.table_id))))
        objects = sorted(path for path in (UPSTREAM / "assets/ycb_objects").iterdir() if path.is_dir())
        if not 0 <= self.object_idx < len(objects):
            raise ValueError(f"object_idx must be in [0, {len(objects) - 1}]")
        self.object_name = objects[self.object_idx].name
        self.object_id = self.p.loadURDF(str(objects[self.object_idx] / "coacd_decomposed_object_one_link.urdf"), [0, 0, 1.0], useFixedBase=False)
        self.object_canonical_mesh = trimesh.load(str(objects[self.object_idx] / "textured.obj"), force="mesh")
        helpers = _load_maze_helpers()
        with _preserve_global_rng():
            np.random.seed(self.seed)
            random.seed(self.seed)
            points, faces = trimesh.sample.sample_surface(self.object_canonical_mesh, 1024)
            self.object_canonical_pc = points.astype(np.float32)
            self.object_canonical_normals = self.object_canonical_mesh.face_normals[faces].astype(np.float32)
            if self.seed_mode == "course-legacy-v1":
                # Preserve point-sampling/entrance RNG coupling and internal seed=5.
                in_row, out_row = int(np.random.randint(1, 11)), int(np.random.randint(1, 11))
                maze = helpers.generate_maze_map(12, 16, (in_row, 0), (out_row, 15), grid_density=0.8,
                    add_x_min_wall=True, add_x_max_wall=True, add_y_min_wall=True, add_y_max_wall=True,
                    p_greedy=0.6, seed=5)
                self.maze_map = helpers.add_left_room_to_maze(maze, room_width=6)
        if self.seed_mode == "independent-maze-v1":
            # Object and point-cloud seeds cannot change the explicit layout.
            # Geometry, generator, physics, and wall generation stay unchanged.
            from .scenes import generate_layout
            layout = generate_layout(self.maze_seed)
            self.maze_map = layout.maze_map
            in_row, out_row = layout.entrance_cell[0], layout.exit_cell[0]
        self.maze_entrance_cell = (in_row, 6)
        self.maze_exit_cell = (out_row, 21)
        self.cube_positions = np.asarray(helpers.create_maze_urdf(self.maze_map, urdf_path=str(self.maze_urdf_path),
            grid_xy=1.0, grid_z=2.0, maze_center_x=7.0, maze_center_y=0.0, maze_center_z=0.0,
            box_color=(0.95, 0.95, 0.98, 1.0)), dtype=np.float32)
        self.maze_id = self.p.loadURDF(str(self.maze_urdf_path), useFixedBase=True, flags=pb.URDF_MERGE_FIXED_LINKS)
        self.goal_pos = np.array([17.5, 5.5 - out_row, 0.9], dtype=np.float64)
        visual = self.p.createVisualShape(pb.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 0.9])
        self.goal_marker_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
            basePosition=self.goal_pos, baseOrientation=[0, 0, 0, 1], useMaximalCoordinates=True)
        self.p.setCollisionFilterGroupMask(self.goal_marker_id, -1, 0, 0)
        self.physics_timestep = float(self.p.getPhysicsEngineParameters()["fixedTimeStep"])
        self._set_motor_targets(self.init_qpos)
        for _ in range(self.settle_steps):
            self.p.stepSimulation()
        self.initial_object_pos = np.array(self.p.getBasePositionAndOrientation(self.object_id)[0])
        self.initial_object_xyzw = np.array(self.p.getBasePositionAndOrientation(self.object_id)[1])
        # Settling is setup time, not task interaction. Clear the negligible
        # resting velocity and snapshot the actual contact-supported pose.
        self.p.resetBaseVelocity(self.object_id, [0, 0, 0], [0, 0, 0])
        self._reset_counters()
        self._initial_state = self.save_state()
        self.obs = self.observe()

    def _reset_counters(self):
        self.step_count = 0
        self.physics_steps = 0
        self.stable_hold_steps = 0
        self.ever_lifted = False
        self.ever_stably_held = False
        self.had_grasp = False
        self.drop_events = 0

    def _set_motor_targets(self, targets):
        self.last_target = np.array(targets, dtype=np.float64)
        self.p.setJointMotorControlArray(self.robot_id, self.joint_indices, pb.POSITION_CONTROL,
            targetPositions=self.last_target.tolist(), forces=[self.max_force] * self.action_size,
            positionGains=self.position_gains)

    def step(self, action):
        targets = np.asarray(action, dtype=np.float64)
        if targets.shape != (self.action_size,) or not np.isfinite(targets).all():
            raise ValueError(f"Expected {self.action_size} finite absolute joint targets")
        targets = targets.copy()
        current = self.agent_qpos
        wrap = self.joint_upper == 314
        targets[wrap] = current[wrap] + (targets[wrap] - current[wrap] + np.pi) % (2 * np.pi) - np.pi
        self._set_motor_targets(np.clip(targets, self.joint_lower, self.joint_upper))
        for _ in range(self.substeps):
            self.p.stepSimulation()
            if self.real_time:
                time.sleep(self.physics_timestep)
        self.step_count += 1
        self.physics_steps += self.substeps
        contacts = self.contacts()
        clearance = self.p.getAABB(self.object_id)[0][2] - self.table_height
        lifted = clearance >= 0.1
        self.ever_lifted |= lifted
        retained = contacts["bilateral_contact"] and lifted
        self.stable_hold_steps = self.stable_hold_steps + 1 if retained else 0
        self.ever_stably_held |= self.stable_hold_steps >= 50
        if self.had_grasp and self.ever_lifted and not contacts["bilateral_contact"] and clearance < 0.02:
            self.drop_events += 1
            self.had_grasp = False
        self.had_grasp |= contacts["bilateral_contact"]
        self.obs = self.observe()
        return self.obs, self.evaluate()

    def reset(self, seed=None, maze_seed=None):
        if maze_seed is not None and self.seed_mode != "independent-maze-v1":
            raise ValueError("reset(maze_seed=...) requires independent-maze-v1 at construction")
        if maze_seed is not None and not 0 <= int(maze_seed) < 2**32:
            raise ValueError("maze_seed must fit an unsigned 32-bit seed")
        scene_changed = seed is not None and int(seed) != self.seed
        maze_changed = maze_seed is not None and int(maze_seed) != self.maze_seed
        if scene_changed or maze_changed:
            if seed is not None:
                self.seed = int(seed)
            if maze_seed is not None:
                self.maze_seed = int(maze_seed)
            self._build_scene()
        else:
            self.restore_state(self._initial_state, reset_counters=True)
        self.obs = self.observe()
        return self.obs

    def contacts(self):
        points = self.p.getContactPoints(self.robot_id, self.object_id)
        finger_contact = [any(c[3] == link and c[9] > 0 for c in points) for link in self.finger_links]
        return {"bilateral_contact": bool(all(finger_contact)), "finger_contacts": finger_contact,
            "contact_count": len(points), "normal_force": float(sum(c[9] for c in points))}

    def evaluate(self):
        pos = np.array(self.p.getBasePositionAndOrientation(self.object_id)[0])
        distance = float(np.linalg.norm(pos - self.goal_pos))
        clearance = float(self.p.getAABB(self.object_id)[0][2] - self.table_height)
        contact = self.contacts()
        return {"dist_to_goal": distance, "success": distance < 0.1, "object_height": float(pos[2]),
            "table_clearance": clearance, "lifted": clearance >= 0.1, "ever_lifted": self.ever_lifted,
            "ever_stably_held": self.ever_stably_held,
            "stable_hold_steps": self.stable_hold_steps, "stable_hold_physics_steps": self.stable_hold_steps * self.substeps,
            "drop_events": self.drop_events, "source_action_calls": self.step_count, "physics_ticks": self.physics_steps,
            "sim_seconds": self.physics_steps * self.physics_timestep,
            "complete_task_success": bool(distance < 0.1 and self.ever_lifted and self.ever_stably_held and contact["bilateral_contact"] and self.drop_events == 0),
            **contact}

    def _get_state(self):
        states = self.p.getJointStates(self.robot_id, self.joint_indices)
        pos, quat = self.p.getBasePositionAndOrientation(self.object_id)
        return (np.array([s[0] for s in states], dtype=np.float32), np.array([s[1] for s in states], dtype=np.float32),
            np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32))

    def observe(self):
        qpos, qvel, pos, quat = self._get_state()
        rotation = np.array(self.p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return {"qpos": qpos, "qvel": qvel, "object_pos": pos, "object_xyzw": quat,
            "goal_pos": self.goal_pos.astype(np.float32),
            "object_pc": (self.object_canonical_pc @ rotation.T + pos).astype(np.float32),
            "object_normals": (self.object_canonical_normals @ rotation.T).astype(np.float32),
            "cube_positions": self.cube_positions[:, :2].copy()}

    _get_obs = observe

    def _set_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.action_size,) or not np.isfinite(qpos).all():
            raise ValueError("Expected finite complete qpos array")
        for joint, position in zip(self.joint_indices, qpos):
            self.p.resetJointState(self.robot_id, joint, float(position), targetVelocity=0.0)

    @property
    def agent_qpos(self):
        return self._get_state()[0]

    @property
    def action_size(self):
        return len(self.joint_indices)

    def save_state(self):
        state = SavedState(self.pb_physics_client, self.p.saveState(), tuple(self.last_target), self.step_count,
            self.physics_steps, self.stable_hold_steps, self.ever_lifted, self.ever_stably_held, self.had_grasp, self.drop_events)
        self._saved_states[state.state_id] = state
        return state

    def restore_state(self, state, reset_counters=False):
        if state.client_id != self.pb_physics_client or self._saved_states.get(state.state_id) is not state:
            raise ValueError("Snapshot belongs to another or reset/closed environment")
        self.p.restoreState(stateId=state.state_id)
        self._set_motor_targets(state.target)
        for field in ("step_count", "physics_steps", "stable_hold_steps", "ever_lifted", "ever_stably_held", "had_grasp", "drop_events"):
            setattr(self, field, getattr(state, field))
        if reset_counters:
            self._reset_counters()
        self.obs = self.observe()
        return self.obs

    def remove_state(self, state):
        if self._saved_states.get(state.state_id) is not state:
            raise ValueError("Unknown state")
        if state is self._initial_state:
            raise ValueError("Cannot remove the environment reset snapshot")
        self.p.removeState(state.state_id)
        del self._saved_states[state.state_id]

    # ---- Simulator-neutral query interface (also implemented by the MuJoCo backend) ----
    # Planner, IK and skill code use only these methods, never the PyBullet client.
    # Kinematic setters are meant for use inside `query()`, which restores the scene.

    @contextmanager
    def query(self):
        state = self.p.saveState()
        try:
            yield
        finally:
            self.p.restoreState(state)
            self.p.removeState(state)

    def set_joint_positions(self, q):
        for joint, value in zip(self.joint_indices, q):
            self.p.resetJointState(self.robot_id, joint, float(value), targetVelocity=0.)

    def ee_pose(self):
        state = self.p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def ee_jacobian(self, q):
        """6x15 world-frame Jacobian of the gripper frame origin (linear; angular)."""
        zeros = [0.] * self.action_size
        linear, angular = self.p.calculateJacobian(self.robot_id, self.ee_link,
            [0., 0., 0.], np.asarray(q, dtype=float).tolist(), zeros, zeros)
        return np.vstack((linear, angular))

    def object_pose(self):
        pos, quat = self.p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos), np.array(quat)

    def set_object_pose(self, position, quaternion):
        self.p.resetBasePositionAndOrientation(self.object_id, position, quaternion)

    def object_velocity(self):
        return np.concatenate(self.p.getBaseVelocity(self.object_id))

    def table_aabb(self):
        return self.p.getAABB(self.table_id)

    def robot_link_aabbs(self):
        return [self.p.getAABB(self.robot_id, link) for link in range(-1, self.p.getNumJoints(self.robot_id))
                if self.p.getCollisionShapeData(self.robot_id, link)]

    def _body(self, name):
        return {"robot": self.robot_id, "object": self.object_id, "table": self.table_id, "maze": self.maze_id}[name]

    def touching(self, a, b):
        """True if bodies a and b have a contact with positive normal force (last physics step)."""
        return any(c[9] > 0 for c in self.p.getContactPoints(self._body(a), self._body(b)))

    def any_within(self, pairs, margin):
        """True if any (a, b) pair is closer than `margin` (negative = penetrating) at current poses."""
        self.p.performCollisionDetection()
        return any(self.p.getClosestPoints(self._body(a), self._body(b), distance=margin) for a, b in pairs)

    def scene_manifest(self):
        return {"upstream_commit": UPSTREAM_COMMIT, "environment_version": "course-adapter-v1",
            "seed": self.seed, "seed_mode": self.seed_mode, "maze_seed": self.maze_seed,
            "object_point_cloud_seed": self.seed, "object_idx": self.object_idx,
            "object_name": self.object_name, "maze_hash": hashlib.sha256(self.maze_map.tobytes()).hexdigest(),
            "maze_map": self.maze_map.tolist(), "maze_entrance_cell": self.maze_entrance_cell,
            "maze_exit_cell": self.maze_exit_cell, "goal_pos": self.goal_pos.tolist(),
            "initial_object_pos": self.initial_object_pos.tolist(), "table_height": self.table_height,
            "joint_names": self.joint_names, "joint_indices": self.joint_indices, "physics_timestep": self.physics_timestep,
            "ticks_per_action": self.substeps, "settling_physics_ticks": self.settle_steps,
            "position_gains": self.position_gains, "motor_max_force": self.max_force,
            "object_attachment": False, "self_collision": False}

    def render_rgb(self, camera="overview", width=960, height=540, target=None, distance=None, yaw=None, pitch=None, shadow=True):
        if camera == "overview":
            defaults = ([7, 0, 0.6], 18, 0, -75)
        elif camera == "follow":
            qpos = self.agent_qpos
            defaults = ([qpos[0] - 2, qpos[1], 0.8], 3.4, 45, -25)
        elif camera == "grasp":
            defaults = (self.p.getBasePositionAndOrientation(self.object_id)[0], 1.25, 130, -25)
        else:
            raise ValueError(f"Unknown camera: {camera}")
        target = defaults[0] if target is None else target
        distance = defaults[1] if distance is None else distance
        yaw = defaults[2] if yaw is None else yaw
        pitch = defaults[3] if pitch is None else pitch
        view = self.p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2)
        projection = self.p.computeProjectionMatrixFOV(55, width / height, 0.02, 80)
        pixels = self.p.getCameraImage(width, height, view, projection, shadow=int(shadow), renderer=pb.ER_TINY_RENDERER)[2]
        return np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()

    def close(self):
        if not self._closed:
            self._closed = True
            if self.p.isConnected():
                self.p.disconnect()
            self._temporary.cleanup()
            self._saved_states.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
