"""Pick-Nav-Reach environment on MuJoCo.

Source task: NUS-LinS-Lab/Pick-Nav-Reach-Default-Project, commit 54ab2d1.
Same scene and contract as the course: its Fetch URDF, YCB objects (CoACD convex parts
as separate collision geoms), the table's collision box, the maze generator and seed
mapping, a 0.9 m goal, and 15 absolute joint-position targets over five 1/240 s ticks.

Motor law. The course's position motors (gain k, 2e6 N cap) command the joint velocity
k * (target - q) / dt every tick, closing a fraction k of the error per tick independently
of mass (k = 0.3 base/torso/head/arm, 0.03 fingers). It is applied as explicit computed
torque, tau = M (v_cmd - v) / dt + bias: exact in free motion, while contacts and joint
limits see the true masses (stiff implicit velocity servos let the fingers pass through
objects, because the constraint solver ignores implicit damping). A blocked joint keeps
pushing with M * k * error / dt^2; FINGER_ARMATURE sets the blocked grip to ~215 N per
finger on the can, the course grip.

Friction. Per-pair coefficients follow the course assets (product of body values):
finger/robot/maze on object 0.25, table or floor on object 0.5. MuJoCo combines geoms by
maximum, so geoms carry robot/object/maze 0.25 and table/floor 0.5. Elliptic cones,
impratio 10 and noslip iterations give static friction without soft-contact creep; the
contact time constant is 2 dt.

Physics settings (`physics=`). 'realistic' (default since 2026-10-02) keeps the same
position-target interface and law but limits every joint like the real Fetch: the
commanded velocity is capped by the stopping distance sqrt(2 a |err|), the joint's
velocity limit and its acceleration limit, and the torque by its effort limit (REALISTIC
in this module, with sources). Gripping surfaces (finger links and palm) are rubber, mu 1.0;
the jaws close at <= 5 cm/s and squeeze with up to 245 N (Fetch spec). 'course' reproduces
the course's unlimited motors and mu 0.25 grip, as used for all results before 2026-10-02.

End effector (`effector=`). 'gripper' is the course's parallel gripper. 'allegro' removes the
jaws and the gripper housing and bolts the Menagerie Allegro hand (pnr/hand.py's model, same
grip friction and joint torque cap) to the wrist flange (`hand_mount=`, ALLEGRO_MOUNTS): 'wrist'
(default) by the hand's wrist end, fingers continuing the forearm, as Allegro hands are mounted on
arms; 'dorsal' (2026-10-02 results) by the back of the palm, palm facing along the approach axis.
The hand keeps its own position servos; step() takes the 13 Fetch joint targets followed by the
16 hand joint targets (or only the 13, keeping the hand's).

Kinematic queries (planner, IK, collision checks) run on a separate scratch MjData, so
they never touch the live simulation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import threading
import xml.etree.ElementTree as ET

import mujoco   # rendering defaults (headless EGL) are set in pnr/__init__.py
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream' / 'Pick-Nav-Reach-Default-Project'
UPSTREAM_COMMIT = '54ab2d1faa54d1aa7e2738f114531a76c6514cd3'
_GLOBAL_RNG_LOCK = threading.Lock()

CACHE = ROOT / '.cache' / 'pnr_mujoco'
ALLEGRO_PATH = ROOT / 'assets/hands/wonik_allegro/right_hand.xml'
COLORS = {'robot': (.78, .79, .82, 1), 'table': (.62, .47, .33, 1), 'maze': (.88, .90, .95, 1),
          'object': (.93, .45, .16, 1)}
CONTACT_SOLREF = (2 / 240, 1.)  # contact time constant 2 dt (stiffest stable), critically damped
CONTACT_SOLIMP = (.9, .95, .001, .5, 2.)
# Coulomb friction without creep: MuJoCo's defaults (pyramidal, impratio 1, no noslip) let held
# objects slide out of the grip under sustained load (4/20 planner-only course cases vs 19/20).
FRICTION_CONE = 'elliptic'
IMPRATIO = 10.
NOSLIP_ITERATIONS = 10
PHYSICS = os.environ.get('PNR_PHYSICS', 'realistic')   # default for every environment built in this process
# Course: blocked grip = (0.08 + A) * 0.03 * error / dt^2 ~ 216 N at 3 cm. Realistic: the 5 cm/s
# velocity cap bounds the computed-torque push at (0.08 + A) * 0.05 / dt, = 245 N for A = 20.3.
FINGER_ARMATURE = {'course': 4.09, 'realistic': 20.3}
GRIP_FRICTION = {'course': .25, 'realistic': 1.}
GRIP_BODIES = ('fetch/r_gripper_finger_link', 'fetch/l_gripper_finger_link', 'fetch/gripper_link')
ALLEGRO_TORQUE = .7      # N m per joint (Allegro hand v4 specification)
# Allegro mounts in the gripper_link frame (the wrist flange face is at x = -0.147): palm origin, and
# the palm's axes as columns. Palm frame: x out of the palm (grasp side, surface at +0.011, back at
# -0.03), z along the fingers (palm from z = -0.095 at the wrist end to 0 at the knuckles).
ALLEGRO_MOUNTS = {
    # Wrist end on the flange through a 1 cm adapter, fingers along the approach axis, palm facing -z.
    'wrist': ((-.042, 0., 0.), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)),
    # Back of the palm on the flange, palm length centred on the arm axis, palm frame = gripper_link.
    'dorsal': ((-.117, 0., .0475), np.eye(3)),
}
ALLEGRO_BITS = (2 | 8, 1 | 8)    # hand: objects, table, maze and its own fingers; not the arm or floor
_INF = float('inf')
# Realistic joint limits: velocity, acceleration, effort (SI units). Base: Fetch spec (1.0 m/s) and
# Fetch base-controller defaults (0.75 m/s^2, 3 rad/s^2); torso and arm: fetch_moveit_config
# joint_limits.yaml (velocity, acceleration) and the course URDF (effort); jaws: URDF 0.05 m/s and the
# Fetch spec 245 N grasp force. The head never moves in this task and keeps uncapped effort.
REALISTIC = {
    'root_x_axis_joint': (1.0, .75, 500.), 'root_y_axis_joint': (1.0, .75, 500.),
    'root_z_rotation_joint': (1.5, 3.0, 500.), 'torso_lift_joint': (.1, .5, 450.),
    'head_pan_joint': (1.57, 3.0, _INF), 'head_tilt_joint': (1.57, 3.0, _INF),
    'shoulder_pan_joint': (1.256, 1.5, 33.82), 'shoulder_lift_joint': (1.0, 1.0, 131.76),
    'upperarm_roll_joint': (1.57, 1.5, 76.94), 'elbow_flex_joint': (1.5, 1.5, 66.18),
    'forearm_roll_joint': (1.57, 1.5, 29.35), 'wrist_flex_joint': (2.26, 2.5, 25.7),
    'wrist_roll_joint': (2.26, 2.5, 7.36),
    'r_gripper_finger_joint': (.05, 1.0, 245.), 'l_gripper_finger_joint': (.05, 1.0, 245.)}
GAINS = [0.3] * 13 + [0.03, 0.03]
FRICTION = {'robot': .25, 'object': .25, 'maze': .25, 'table': .5, 'floor': .5}
# Collision bitmasks: robot never collides with itself or the floor.
BITS = {'static': (1, 3), 'robot': (2, 1), 'object': (1, 7), 'floor': (4, 4)}


def limited_velocity(command, error, previous, vmax, amax, dt):
    """Velocity reference limited by the stopping distance, the velocity limit and the acceleration limit.

    The acceleration limit applies to the reference (`previous` command), as in a real servo: a blocked
    joint keeps its reference, so it pushes up to its effort limit instead of stalling at zero.
    """
    command = np.clip(command, -np.sqrt(2 * amax * np.abs(error)), np.sqrt(2 * amax * np.abs(error)))
    command = np.clip(command, -vmax, vmax)
    return np.clip(command, previous - amax * dt, previous + amax * dt)


def _load_maze_helpers():
    spec = importlib.util.spec_from_file_location('pnr_upstream_maze', UPSTREAM / 'maze_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _preserve_global_rng():
    # The course helper seeds process-global random and numpy. Restrict that
    # behaviour to a short serialized operation; do not disturb learner RNGs.
    with _GLOBAL_RNG_LOCK:
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def _course_fetch_urdf():
    """Course Fetch URDF prepared for MuJoCo: meshes resolved, default inertials.

    Links without <inertial> (the five head-camera frames) get the course simulator's
    default of mass 1 and unit inertia at the link origin.
    """
    out = CACHE / 'fetch_v2.urdf'
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        root = ET.parse(UPSTREAM / 'assets/fetch/fetch.urdf').getroot()
        for link in root.findall('link'):
            if link.find('inertial') is None:
                inertial = ET.SubElement(link, 'inertial')
                ET.SubElement(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
                ET.SubElement(inertial, 'mass', value='1')
                ET.SubElement(inertial, 'inertia', ixx='1', iyy='1', izz='1', ixy='0', ixz='0', iyz='0')
        meshdir = UPSTREAM / 'assets/fetch'
        root.insert(0, ET.fromstring(
            f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" boundmass="0.001" '
            f'boundinertia="0.000001" discardvisual="true" fusestatic="false" strippath="false"/></mujoco>'))
        out.write_text(ET.tostring(root, encoding='unicode'))
    return out


# Enlarged variants of course objects for the hand-suited task: id -> (course object, scale), same
# density (mass x scale^3, inertia x scale^5). Each is about 11 cm across at its narrowest, wider than
# the Fetch gripper's 9.8 cm opening.
OBJECT_VARIANTS = {102: (2, 1.6), 107: (7, 1.45), 110: (10, 1.35)}


def object_source(idx, objects):
    """(object directory, scale, name) for a course object index or an enlarged variant."""
    base, scale = OBJECT_VARIANTS.get(int(idx), (int(idx), 1.))
    if not 0 <= base < len(objects):
        raise ValueError(f'object_idx must be in [0, {len(objects) - 1}] or one of {sorted(OBJECT_VARIANTS)}')
    directory = objects[base]
    return directory, scale, directory.name if scale == 1 else f'{directory.name}_x{scale:g}'


def _object_parts(object_dir):
    """Split coacd_allinone.obj into one OBJ per convex part; return (part paths, mass, inertia)."""
    target = CACHE / 'ycb' / object_dir.name
    if not (target / 'done').exists():
        target.mkdir(parents=True, exist_ok=True)
        vertices, parts, current = [], [], None
        for line in (object_dir / 'coacd_allinone.obj').read_text().splitlines():
            if line.startswith('v '):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith('o '):
                current = []
                parts.append(current)
            elif line.startswith('f '):
                if current is None:
                    current = []; parts.append(current)
                current.append([int(tok.split('/')[0]) - 1 for tok in line.split()[1:4]])
        vertices = np.asarray(vertices)
        for i, faces in enumerate(parts):
            faces = np.asarray(faces)
            used, inverse = np.unique(faces, return_inverse=True)
            trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), process=False).export(target / f'part_{i:03d}.obj')
        (target / 'done').write_text(str(len(parts)))
    root = ET.parse(object_dir / 'coacd_decomposed_object_one_link.urdf').getroot()
    inertial = root.find('.//inertial')
    mass = float(inertial.find('mass').get('value'))
    i = inertial.find('inertia').attrib
    full = [float(i[k]) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]
    return sorted(target.glob('part_*.obj')), mass, full


@dataclass(frozen=True)
class SavedState:
    env_id: int
    state: object      # full MjData copy, so contacts and constraint forces restore bit-exactly
    target: tuple
    step_count: int
    physics_steps: int
    stable_hold_steps: int
    ever_lifted: bool
    ever_stably_held: bool
    had_grasp: bool
    drop_events: int
    reference: tuple = ()


class PickNavReachEnv:
    """Native course interface: 15 absolute joint targets per step, five physics ticks.

    `step(action)` returns `(obs, info)`; Gymnasium wrappers and rewards live elsewhere.
    With no maze_seed, course-legacy-v1 reproduces the course's seeded scene. An explicit
    maze_seed selects independent-maze-v1, where seed only controls object point sampling.
    """

    root_position = np.array([-2.0, 0.0, 0.0])
    substeps = 5
    max_force = 2_000_000          # course motor cap, reported for reference; see module docstring
    position_gains = GAINS

    def __init__(self, seed=0, object_idx=5, gui=False, settle_steps=480, real_time=False,
                 maze_seed=None, seed_mode=None, physics=None, effector='gripper', object_pose_seed=None,
                 hand_mount='wrist'):
        if gui or real_time:
            raise ValueError('The environment is headless; use render_rgb')
        if effector not in ('gripper', 'allegro'):
            raise ValueError(f'Unknown end effector: {effector}')
        self.effector = effector
        if hand_mount not in ALLEGRO_MOUNTS:
            raise ValueError(f'Unknown hand mount: {hand_mount}')
        self.hand_mount = hand_mount
        # None: the course's object start (dropped at the table centre, canonical yaw). An integer: the
        # object drops at a seeded offset (x +-8 cm, y +-12 cm) and yaw, then settles the same way.
        self.object_pose_seed = None if object_pose_seed is None else int(object_pose_seed)
        self.physics = physics or PHYSICS
        if self.physics not in ('course', 'realistic'):
            raise ValueError(f'Unknown physics settings: {self.physics}')
        self.seed = int(seed)
        self.seed_mode = seed_mode or ('course-legacy-v1' if maze_seed is None else 'independent-maze-v1')
        if self.seed_mode not in ('course-legacy-v1', 'independent-maze-v1'):
            raise ValueError(f'Unknown scene seed mode: {self.seed_mode}')
        if (maze_seed is None) != (self.seed_mode == 'course-legacy-v1'):
            raise ValueError('independent-maze-v1 requires maze_seed; course-legacy-v1 forbids it')
        self.maze_seed = None if maze_seed is None else int(maze_seed)
        self.object_idx = int(object_idx)
        self.settle_steps = int(settle_steps)
        self._closed = False
        self._saved = {}
        self._build_scene()

    # ---------------- scene construction ----------------
    def _build_scene(self):
        objects = sorted(path for path in (UPSTREAM / 'assets/ycb_objects').iterdir() if path.is_dir())
        object_dir, self.object_scale, self.object_name = object_source(self.object_idx, objects)
        self.object_canonical_mesh = trimesh.load(str(object_dir / 'textured.obj'), force='mesh')
        self.object_canonical_mesh.apply_scale(self.object_scale)
        helpers = _load_maze_helpers()
        with _preserve_global_rng():   # identical RNG use to the course: same point clouds and mazes
            np.random.seed(self.seed)
            random.seed(self.seed)
            points, faces = trimesh.sample.sample_surface(self.object_canonical_mesh, 1024)
            self.object_canonical_pc = points.astype(np.float32)
            self.object_canonical_normals = self.object_canonical_mesh.face_normals[faces].astype(np.float32)
            if self.seed_mode == 'course-legacy-v1':
                in_row, out_row = int(np.random.randint(1, 11)), int(np.random.randint(1, 11))
                maze = helpers.generate_maze_map(12, 16, (in_row, 0), (out_row, 15), grid_density=0.8,
                    add_x_min_wall=True, add_x_max_wall=True, add_y_min_wall=True, add_y_max_wall=True,
                    p_greedy=0.6, seed=5)
                self.maze_map = helpers.add_left_room_to_maze(maze, room_width=6)
        if self.seed_mode == 'independent-maze-v1':
            from .scenes import generate_layout
            layout = generate_layout(self.maze_seed)
            self.maze_map = layout.maze_map
            in_row, out_row = layout.entrance_cell[0], layout.exit_cell[0]
        self.maze_entrance_cell, self.maze_exit_cell = (in_row, 6), (out_row, 21)
        scratch = CACHE / 'maze_scratch.urdf'
        CACHE.mkdir(parents=True, exist_ok=True)
        self.cube_positions = np.asarray(helpers.create_maze_urdf(self.maze_map, urdf_path=str(scratch),
            grid_xy=1.0, grid_z=2.0, maze_center_x=7.0, maze_center_y=0.0, maze_center_z=0.0,
            box_color=(0.95, 0.95, 0.98, 1.0)), dtype=np.float32)
        self.goal_pos = np.array([17.5, 5.5 - out_row, 0.9], dtype=np.float64)

        spec = mujoco.MjSpec()
        spec.option.timestep = 1 / 240
        spec.option.gravity = [0, 0, -9.8]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = getattr(mujoco.mjtCone, 'mjCONE_' + FRICTION_CONE.upper())
        spec.option.impratio = IMPRATIO
        spec.option.noslip_iterations = NOSLIP_ITERATIONS
        world = spec.worldbody

        def geom(parent, group, **kwargs):
            contype, conaffinity = BITS['static' if group in ('table', 'maze') else group]
            kind = group
            g = parent.add_geom(contype=contype, conaffinity=conaffinity, condim=3,
                                friction=[FRICTION[kind], .005, .0001], **kwargs)
            if group in COLORS:
                g.rgba = COLORS[group]
            g.name = f'{group}_{len(self._geom_groups)}'
            self._geom_groups.append(group)
            return g

        self._geom_groups = []
        # Rendering only: offscreen buffer, course camera field of view, light, floor grid.
        spec.visual.global_.offwidth, spec.visual.global_.offheight, spec.visual.global_.fovy = 1920, 1080, 55
        spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[.60, .62, .66], rgb2=[.54, .56, .60], width=300, height=300)
        spec.add_material(name='grid', texrepeat=[60, 60], reflectance=.05).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
        world.add_light(pos=[7, 0, 12], dir=[.3, .2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        diffuse=[.35, .35, .35], castshadow=True)
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [.32, .32, .32], [.32, .32, .32]
        floor = geom(world, 'floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[60, 60, .1])
        floor.material = 'grid'
        # The course table's collision box (1.5 x 1 x 0.05 at z 0.6), rotated pi/2 about z.
        geom(world, 'table', type=mujoco.mjtGeom.mjGEOM_BOX, size=[.5, .75, .025], pos=[0, 0, .6])
        for x, y, z in self.cube_positions:
            geom(world, 'maze', type=mujoco.mjtGeom.mjGEOM_BOX, size=[.5, .5, 1.], pos=[x, y, z])
        world.add_site(name='goal', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.05, 0, 0], pos=self.goal_pos.tolist(),
                       rgba=[0, 1, 0, .9])

        parts, mass, inertia = _object_parts(object_dir)
        body = world.add_body(name='object', pos=[0, 0, 1.0])
        if self.object_pose_seed is not None:
            rng = np.random.default_rng(self.object_pose_seed)
            yaw = rng.uniform(-np.pi, np.pi)
            body.pos = [rng.uniform(-.08, .08), rng.uniform(-.12, .12), 1.0]
            body.quat = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        body.add_freejoint(name='object_free')
        # URDF inertial origin is the mesh origin; MjSpec's unset ipos (NaN) must be given explicitly.
        s = self.object_scale
        body.mass, body.fullinertia, body.explicitinertial = mass * s ** 3, [v * s ** 5 for v in inertia], True
        body.ipos = [0., 0., 0.]
        for i, part in enumerate(parts):
            spec.add_mesh(name=f'object_part_{i}', file=str(part), scale=[s, s, s])
            geom(body, 'object', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'object_part_{i}', density=0)

        robot = mujoco.MjSpec.from_file(str(_course_fetch_urdf()))
        if self.effector == 'allegro':
            # Jaws and gripper housing off; gripper_link stays as the flange frame (an adapter plate).
            for name in ('r_gripper_finger_link', 'l_gripper_finger_link'):
                robot.delete(robot.body(name))
            flange = robot.body('gripper_link')
            for g in list(flange.geoms):
                robot.delete(g)
            flange.mass, flange.explicitinertial = .1, True
            flange.fullinertia, flange.ipos = [1e-4, 1e-4, 1e-4, 0., 0., 0.], [-.14, 0., 0.]
        frame = world.add_frame(pos=self.root_position.tolist())
        frame.attach_body(robot.worldbody.first_body(), 'fetch/', '')
        for g in spec.geoms:
            g.solref, g.solimp = list(CONTACT_SOLREF), list(CONTACT_SOLIMP)
        for g in spec.geoms:   # attached robot geoms: no self-collision, course friction
            if not g.name.startswith(('floor', 'table', 'maze', 'object')):
                contype, conaffinity = BITS['robot']
                g.contype, g.conaffinity, g.condim = contype, conaffinity, 3
                g.friction = [FRICTION['robot'], .005, .0001]
                g.rgba = COLORS['robot']
        self.joint_names = [j.name.split('/', 1)[1] for j in spec.joints if j.name.startswith('fetch/')]
        for joint in spec.joints:
            if joint.name.startswith('fetch/'):
                # The course motors override URDF effort limits (2e6 N cap)
                # and the environment disables the default joint-friction motors. Mirror both.
                # The velocity motors override URDF joint damping as well.
                joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
                joint.frictionloss, joint.damping = 0., [0., 0., 0.]
                if 'finger' in joint.name:
                    joint.armature = FINGER_ARMATURE[self.physics]
        if self.effector == 'allegro':
            hand = mujoco.MjSpec.from_file(str(ALLEGRO_PATH))
            for g in hand.geoms:
                if g.contype or g.conaffinity:          # collision geoms; visual meshes stay non-colliding
                    g.contype, g.conaffinity, g.condim = ALLEGRO_BITS[0], ALLEGRO_BITS[1], 3
                    g.friction = [GRIP_FRICTION[self.physics], .005, .0001]
                    g.solref, g.solimp = list(CONTACT_SOLREF), list(CONTACT_SOLIMP)
            palm = hand.worldbody.first_body()
            palm.pos, palm.quat = [0., 0., 0.], [1., 0., 0., 0.]
            position, axes = ALLEGRO_MOUNTS[self.hand_mount]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.ravel())
            spec.body('fetch/gripper_link').add_frame(pos=list(position), quat=quat.tolist()).attach_body(palm, 'hand/', '')
        self.model = spec.compile()
        if self.effector == 'allegro' and self.physics == 'realistic':
            self.model.actuator_forcelimited[:] = 1
            self.model.actuator_forcerange[:] = [-ALLEGRO_TORQUE, ALLEGRO_TORQUE]
        grip = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in GRIP_BODIES}
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] in grip:
                self.model.geom_friction[g, 0] = GRIP_FRICTION[self.physics]
        self.data = mujoco.MjData(self.model)
        self._query = mujoco.MjData(self.model)
        self._jac = mujoco.MjData(self.model)
        self._index_model()

        self.physics_timestep = float(self.model.opt.timestep)
        self.table_height = .625 + .001   # box top plus the course's 1 mm AABB margin
        self.init_qpos = np.clip(np.zeros(len(self.joint_names)), self.joint_lower, self.joint_upper)
        self.rest_poses = self.init_qpos.copy()
        self._set_qpos(self.init_qpos)
        self.data.ctrl[:] = self.hand_open
        self.last_target = self.init_qpos.astype(np.float64).copy()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.settle_steps):
            self._physics_tick()
        self.initial_object_pos, self.initial_object_xyzw = self.object_pose()
        self.data.qvel[self._obj_dof:self._obj_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self._reset_counters()
        self._initial_state = self.save_state()
        self.obs = self.observe()

    def _index_model(self):
        m = self.model
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'fetch/{n}') for n in self.joint_names]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self._dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.joint_lower = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.inf for j in jids], dtype=np.float32)
        self.joint_upper = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.inf for j in jids], dtype=np.float32)
        self.joint_ranges = self.joint_upper - self.joint_lower
        self.joint_indices = list(range(len(jids)))
        self.joint_name_to_slot = {name: slot for slot, name in enumerate(self.joint_names)}
        self.joint_name_to_index = dict(self.joint_name_to_slot)
        self.arm_slots = np.array([self.joint_name_to_slot[n] for n in (
            'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint',
            'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint')], dtype=int)
        self.finger_slots = np.array([self.joint_name_to_slot[n] for n in ('r_gripper_finger_joint', 'l_gripper_finger_joint')
                                      if n in self.joint_name_to_slot], dtype=int)
        self._ee_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'fetch/gripper_link')
        self.ee_link = self._ee_body
        self._obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'object')
        free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'object_free')
        self._obj_qpos, self._obj_dof = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self._finger_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'fetch/{n}')
                               for n in ('r_gripper_finger_link', 'l_gripper_finger_link')]
        # Allegro: hand actuators (position servos on the 16 hand joints) and finger membership.
        self.hand_low, self.hand_high = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.hand_open = np.clip(np.zeros(m.nu), self.hand_low, self.hand_high)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(m.nbody)]
        self._hand_finger_of_body = {b: next((i for i, f in enumerate(('ff', 'mf', 'rf', 'th')) if n.startswith(f'hand/{f}_')), None)
                                     for b, n in enumerate(names) if n.startswith('hand/')}
        # Geom groups by name prefix (robot geoms carry the attach prefix or are unnamed).
        self._group = np.empty(m.ngeom, dtype=object)
        for g in range(m.ngeom):
            name = m.geom(g).name
            self._group[g] = next((k for k in ('floor', 'table', 'maze', 'object') if name.startswith(k)), 'robot')
        self._robot_bodies = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom) if self._group[g] == 'robot'})
        # MuJoCo stores mesh vertices in each geom's frame; world vertices use geom poses.
        self._object_geoms = [g for g in range(m.ngeom) if self._group[g] == 'object']
        self._gain = np.asarray(GAINS)[:len(self.joint_names)]   # the two jaw joints come last
        limits = np.array([REALISTIC[n] for n in self.joint_names]).T if self.physics == 'realistic' else None
        self._vmax, self._amax, self._effort = limits if limits is not None else (None, None, None)
        self._reference = np.zeros(len(self.joint_names))   # rate-limited velocity reference (realistic)
        self._mass = np.zeros((m.nv, m.nv))

    # ---------------- physics ----------------
    def _physics_tick(self):
        m, d, dt = self.model, self.data, self.physics_timestep
        mujoco.mj_step1(m, d)          # positions/velocities -> mass matrix and bias forces
        error = self.last_target - d.qpos[self._qadr]
        velocity = d.qvel[self._dadr]
        command = self._gain * error / dt
        if self._vmax is not None:
            command = limited_velocity(command, error, self._reference, self._vmax, self._amax, dt)
            self._reference = command
        mujoco.mj_fullM(m, d, self._mass)
        robot_mass = self._mass[np.ix_(self._dadr, self._dadr)]
        torque = robot_mass @ (command - velocity) / dt + d.qfrc_bias[self._dadr]
        if self._effort is not None:
            torque = np.clip(torque, -self._effort, self._effort)
        d.qfrc_applied[self._dadr] = torque
        mujoco.mj_step2(m, d)

    def _reset_counters(self):
        self.step_count = 0
        self.physics_steps = 0
        self.stable_hold_steps = 0
        self.ever_lifted = False
        self.ever_stably_held = False
        self.had_grasp = False
        self.drop_events = 0

    def _set_qpos(self, qpos, data=None):
        data = self.data if data is None else data
        data.qpos[self._qadr] = qpos
        data.qvel[self._dadr] = 0
        mujoco.mj_kinematics(self.model, data)
        if data is self.data:
            self._reference = np.zeros(len(self.joint_names))

    @property
    def action_size(self):
        return len(self.joint_names) + self.model.nu

    @property
    def agent_qpos(self):
        return self.data.qpos[self._qadr].astype(np.float32)

    def command(self, action):
        """Set the joint (and hand servo) targets that the following physics ticks track."""
        targets = np.asarray(action, dtype=np.float64)
        sizes = (self.action_size, len(self.joint_names))     # hand targets optional: omitted keeps the hand's
        if targets.shape not in [(n,) for n in sizes] or not np.isfinite(targets).all():
            raise ValueError(f'Expected {self.action_size} finite absolute joint targets')
        targets, hand = targets[:len(self.joint_names)].copy(), targets[len(self.joint_names):]
        if len(hand):
            self.data.ctrl[:] = np.clip(hand, self.hand_low, self.hand_high)
        current = self.agent_qpos
        wrap = self.joint_upper == 314
        targets[wrap] = current[wrap] + (targets[wrap] - current[wrap] + np.pi) % (2 * np.pi) - np.pi
        self.last_target = np.clip(targets, self.joint_lower, self.joint_upper)

    def step(self, action):
        self.command(action)
        for _ in range(self.substeps):
            self._physics_tick()
        self.step_count += 1
        self.physics_steps += self.substeps
        contacts = self.contacts()
        clearance = self._object_min_z() - self.table_height
        lifted = clearance >= 0.1
        self.ever_lifted |= lifted
        retained = contacts['bilateral_contact'] and lifted
        self.stable_hold_steps = self.stable_hold_steps + 1 if retained else 0
        self.ever_stably_held |= self.stable_hold_steps >= 50
        if self.had_grasp and self.ever_lifted and not contacts['bilateral_contact'] and clearance < 0.02:
            self.drop_events += 1
            self.had_grasp = False
        self.had_grasp |= contacts['bilateral_contact']
        self.obs = self.observe()
        return self.obs, self.evaluate()

    def reset(self, seed=None, maze_seed=None):
        if maze_seed is not None and self.seed_mode != 'independent-maze-v1':
            raise ValueError('maze_seed requires independent-maze-v1')
        changed = (seed is not None and int(seed) != self.seed) or (maze_seed is not None and int(maze_seed) != self.maze_seed)
        if changed:
            self.seed = self.seed if seed is None else int(seed)
            self.maze_seed = self.maze_seed if maze_seed is None else int(maze_seed)
            self._saved.clear()
            self._build_scene()
        else:
            self.restore_state(self._initial_state, reset_counters=True)
        self.obs = self.observe()
        return self.obs

    # ---------------- contacts and evaluation ----------------
    def _contacts(self, data, a=None, b=None):
        """Yield (contact index, group1, group2, body1, body2) from a data's contact list."""
        m = self.model
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = self._group[c.geom1], self._group[c.geom2]
            if a is None or {g1, g2} == {a, b} or (a == b == g1 == g2):
                yield i, g1, g2, int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])

    def _normal_force(self, i):
        force = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, i, force)
        return float(force[0])

    def contacts(self):
        """Robot-object contacts. Bilateral: both jaws, or for the Allegro the thumb and a finger."""
        finger = [False, False] if self.effector == 'gripper' else [False] * 4
        count, total = 0, 0.
        for i, g1, g2, b1, b2 in self._contacts(self.data, 'robot', 'object'):
            f = self._normal_force(i)
            count += 1
            total += f
            if f <= 0:
                continue
            if self.effector == 'gripper':
                for k, body in enumerate(self._finger_bodies):
                    if body in (b1, b2):
                        finger[k] = True
            else:
                for body in (b1, b2):
                    k = self._hand_finger_of_body.get(body)
                    if k is not None:
                        finger[k] = True
        bilateral = all(finger) if self.effector == 'gripper' else finger[3] and any(finger[:3])
        return {'bilateral_contact': bool(bilateral), 'finger_contacts': finger,
                'contact_count': count, 'normal_force': total}

    def _object_min_z(self, data=None):
        data = self.data if data is None else data
        low = np.inf
        for g in self._object_geoms:
            mesh = self.model.geom_dataid[g]
            verts = self.model.mesh_vert[self.model.mesh_vertadr[mesh]:self.model.mesh_vertadr[mesh] + self.model.mesh_vertnum[mesh]]
            world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            low = min(low, float(world[:, 2].min()))
        return low - .001   # the course's AABBs include a 1 mm collision margin

    def evaluate(self):
        pos, _ = self.object_pose()
        distance = float(np.linalg.norm(pos - self.goal_pos))
        clearance = float(self._object_min_z() - self.table_height)
        contact = self.contacts()
        return {'dist_to_goal': distance, 'success': distance < 0.1, 'object_height': float(pos[2]),
                'table_clearance': clearance, 'lifted': clearance >= 0.1, 'ever_lifted': self.ever_lifted,
                'ever_stably_held': self.ever_stably_held, 'stable_hold_steps': self.stable_hold_steps,
                'stable_hold_physics_steps': self.stable_hold_steps * self.substeps, 'drop_events': self.drop_events,
                'source_action_calls': self.step_count, 'physics_ticks': self.physics_steps,
                'sim_seconds': self.physics_steps * self.physics_timestep,
                'complete_task_success': bool(distance < 0.1 and self.ever_lifted and self.ever_stably_held
                                              and contact['bilateral_contact'] and self.drop_events == 0),
                **contact}

    def observe(self):
        pos, quat = self.object_pose()
        rotation = self._rotation(quat)
        return {'qpos': self.data.qpos[self._qadr].astype(np.float32), 'qvel': self.data.qvel[self._dadr].astype(np.float32),
                'object_pos': pos.astype(np.float32), 'object_xyzw': quat.astype(np.float32),
                'goal_pos': self.goal_pos.astype(np.float32),
                'object_pc': (self.object_canonical_pc @ rotation.T + pos).astype(np.float32),
                'object_normals': (self.object_canonical_normals @ rotation.T).astype(np.float32),
                'cube_positions': self.cube_positions[:, :2].copy()}

    _get_obs = observe

    @staticmethod
    def _rotation(xyzw):
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]))
        return matrix.reshape(3, 3)

    # ---------------- snapshots ----------------
    def save_state(self):
        # A full copy rather than mj_getState + mj_forward: after mj_step the contact forces in
        # data belong to the pre-integration state, and recomputing them changes observations.
        state = mujoco.MjData(self.model)
        mujoco.mj_copyData(state, self.model, self.data)
        saved = SavedState(id(self), state, tuple(self.last_target), self.step_count, self.physics_steps,
                           self.stable_hold_steps, self.ever_lifted, self.ever_stably_held, self.had_grasp, self.drop_events,
                           tuple(self._reference))
        self._saved[id(saved)] = saved
        return saved

    def restore_state(self, state, reset_counters=False):
        if state.env_id != id(self) or self._saved.get(id(state)) is not state:
            raise ValueError('Snapshot belongs to another or reset/closed environment')
        mujoco.mj_copyData(self.data, self.model, state.state)
        self.last_target = np.asarray(state.target, dtype=np.float64)
        self._reference = np.asarray(state.reference, dtype=np.float64)
        for field in ('step_count', 'physics_steps', 'stable_hold_steps', 'ever_lifted', 'ever_stably_held', 'had_grasp', 'drop_events'):
            setattr(self, field, getattr(state, field))
        if reset_counters:
            self._reset_counters()
        self.obs = self.observe()
        return self.obs

    def remove_state(self, state):
        if self._saved.get(id(state)) is not state:
            raise ValueError('Unknown state')
        if state is self._initial_state:
            raise ValueError('Cannot remove the environment reset snapshot')
        del self._saved[id(state)]

    # ---------------- simulator-neutral query interface ----------------
    @contextmanager
    def query(self):
        outer = getattr(self, '_query_depth', 0)
        if outer == 0:
            self._query.qpos[:] = self.data.qpos
            self._query.qvel[:] = 0
            mujoco.mj_kinematics(self.model, self._query)
        saved = self._query.qpos.copy()
        self._query_depth = outer + 1
        try:
            yield
        finally:
            self._query_depth = outer
            self._query.qpos[:] = saved
            mujoco.mj_kinematics(self.model, self._query)

    def _active(self):
        return self._query if getattr(self, '_query_depth', 0) else self.data

    def set_joint_positions(self, q):
        if not getattr(self, '_query_depth', 0):
            raise RuntimeError('Kinematic edits are only allowed inside env.query()')
        self._query.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, self._query)

    def ee_pose(self):
        data = self._active()
        w = data.xquat[self._ee_body]
        return data.xpos[self._ee_body].copy(), np.array([w[1], w[2], w[3], w[0]])

    def ee_jacobian(self, q):
        self._jac.qpos[:] = self._active().qpos
        self._jac.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, self._jac)
        mujoco.mj_comPos(self.model, self._jac)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self._jac, jacp, jacr, self._jac.xpos[self._ee_body], self._ee_body)
        return np.vstack((jacp[:, self._dadr], jacr[:, self._dadr]))

    def object_pose(self):
        data = self._active()
        q = data.qpos[self._obj_qpos:self._obj_qpos + 7]
        return q[:3].copy(), np.array([q[4], q[5], q[6], q[3]])

    def set_object_pose(self, position, quaternion):
        if not getattr(self, '_query_depth', 0):
            raise RuntimeError('Kinematic edits are only allowed inside env.query()')
        x, y, z, w = quaternion
        self._query.qpos[self._obj_qpos:self._obj_qpos + 7] = [*position, w, x, y, z]
        mujoco.mj_kinematics(self.model, self._query)

    def object_velocity(self):
        v = self.data.qvel[self._obj_dof:self._obj_dof + 6]
        return np.concatenate((v[:3], v[3:]))

    def table_aabb(self):
        return ((-.5 - .001, -.75 - .001, .575 - .001), (.5 + .001, .75 + .001, .625 + .001))

    def robot_link_aabbs(self):
        data, m = self._active(), self.model
        boxes = []
        for body in self._robot_bodies:
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != body or self._group[g] != 'robot':
                    continue
                centre, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                corners = centre + half * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
                world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
            boxes.append((tuple(lo), tuple(hi)))
        return boxes

    def touching(self, a, b):
        return any(self._normal_force(i) > 0 for i, *_ in self._contacts(self.data, a, b))

    def any_within(self, pairs, margin):
        """Closest-distance test via MuJoCo's collision pipeline, always on the scratch query data."""
        data = self._query
        if not getattr(self, '_query_depth', 0):
            data.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, data)
        wanted = {frozenset(p) for p in pairs}
        m = self.model
        saved_margin = m.geom_margin.copy()
        try:
            m.geom_margin[:] = max(margin, 0.) + 1e-4
            mujoco.mj_collision(m, data)
            for i in range(data.ncon):
                c = data.contact[i]
                pair = frozenset((self._group[c.geom1], self._group[c.geom2]))
                if pair in wanted and c.dist < margin:
                    return True
            return False
        finally:
            m.geom_margin[:] = saved_margin

    # ---------------- bookkeeping ----------------
    def scene_manifest(self):
        return {'upstream_commit': UPSTREAM_COMMIT, 'environment_version': 'mujoco-v1', 'simulator': 'mujoco',
                'seed': self.seed, 'seed_mode': self.seed_mode, 'maze_seed': self.maze_seed,
                'object_point_cloud_seed': self.seed, 'object_idx': self.object_idx, 'object_name': self.object_name,
                'maze_hash': hashlib.sha256(self.maze_map.tobytes()).hexdigest(), 'maze_map': self.maze_map.tolist(),
                'maze_entrance_cell': self.maze_entrance_cell, 'maze_exit_cell': self.maze_exit_cell,
                'goal_pos': self.goal_pos.tolist(), 'initial_object_pos': self.initial_object_pos.tolist(),
                'table_height': self.table_height, 'joint_names': self.joint_names,
                'physics_timestep': self.physics_timestep, 'ticks_per_action': self.substeps,
                'settling_physics_ticks': self.settle_steps, 'position_gains': self.position_gains,
                'physics': self.physics, 'finger_armature': FINGER_ARMATURE[self.physics],
                'grip_friction': GRIP_FRICTION[self.physics], 'mujoco_version': mujoco.__version__,
                'object_attachment': False, 'self_collision': False}

    def render_rgb(self, camera='overview', width=960, height=540, target=None, distance=None, yaw=None,
                   pitch=None, shadow=True):
        """RGB frame from the course cameras (orbit target, distance, yaw and pitch in degrees)."""
        if camera == 'overview':
            defaults = ([7, 0, .6], 18, 0, -75)
        elif camera == 'follow':
            q = self.agent_qpos
            defaults = ([self.root_position[0] + q[0], self.root_position[1] + q[1], .8], 3.4, 45, -25)
        elif camera == 'grasp':
            defaults = (self.object_pose()[0], 1.25, 130, -25)
        else:
            raise ValueError(f'Unknown camera: {camera}')
        target, distance, yaw, pitch = (d if v is None else v for v, d in zip((target, distance, yaw, pitch), defaults))
        if getattr(self, '_renderer', None) is None or self._renderer_size != (width, height):
            self._close_renderer()
            self._renderer, self._renderer_size = mujoco.Renderer(self.model, height, width), (width, height)
        view = mujoco.MjvCamera()
        view.type = mujoco.mjtCamera.mjCAMERA_FREE
        view.lookat[:], view.distance = target, distance
        view.azimuth, view.elevation = yaw + 90, pitch   # course yaw 0 looks along +y
        self._renderer.update_scene(self.data, camera=view)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(shadow)
        return self._renderer.render().copy()

    def _close_renderer(self):
        if getattr(self, '_renderer', None) is not None:
            self._renderer.close()
            self._renderer = None

    def close(self):
        self._closed = True
        self._saved.clear()
        self._close_renderer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
