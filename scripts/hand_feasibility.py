#!/usr/bin/env python3
"""Feasibility spike: does a multi-finger hand hold objects better than Fetch's gripper?

Each hand is mounted on a 6-DoF "gantry" (x, y, z slides; yaw, pitch, roll) driven
exactly like the Fetch base and arm: position control, 2e6 N force cap, gain 0.3,
five physics ticks per command. At that force cap the controller moves 30% of the
remaining error per tick independently of mass, so the gantry reproduces the base's
motion pulses during carry. Fingers use the environment's finger gain (0.03).

Per trial: settle the object on the table, run a scripted grasp, lift 20 cm, hold,
then carry along a route recorded from a real learned carry (15 mm per command, the
carry adapter's speed). Retention, slip and costs are recorded. Same table, objects,
gravity and timestep as pnr/env.py; no task code is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation
import trimesh

from pnr.planning import antipodal_grasps

UPSTREAM = ROOT / 'upstream/Pick-Nav-Reach-Default-Project'
FETCH_URDF = UPSTREAM / 'assets/fetch/fetch.urdf'
ALLEGRO_URDF = ROOT / '.tools/robot_descriptions_cache/drake/manipulation/models/allegro_hand_description/urdf/allegro_hand_description_left.urdf'
TRACES = ROOT / 'runs/phase2_system_confirmation_test01/artifacts'
OUT = ROOT / 'runs/hand_feasibility01'
OBJECTS = {14: 'foam', 5: 'banana', 2: 'can', 7: 'apple', 10: 'mug'}
MAX_FORCE, GANTRY_GAIN, FINGER_GAIN, SUBSTEPS = 2_000_000, 0.3, 0.03, 5
# Both hands' nominal top-down frame: hand x -> world -z, hand y -> world +y, hand z -> world +x.
R_NOMINAL = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
GANTRY_JOINTS = [('gx', 'prismatic', '1 0 0'), ('gy', 'prismatic', '0 1 0'), ('gz', 'prismatic', '0 0 1'),
                 ('gyaw', 'revolute', '0 0 1'), ('gpitch', 'revolute', '0 1 0'), ('groll', 'revolute', '1 0 0')]


def _absolute_meshes(element, base_dir):
    for mesh in element.iter('mesh'):
        name = mesh.get('filename', '')
        if name.startswith('package://'):
            name = name.split('/', 3)[-1]
        if not os.path.isabs(name):
            mesh.set('filename', str((base_dir / name).resolve()))


def gantry_urdf(hand, out_dir):
    """Write gantry + hand URDF; returns (path, finger joint names)."""
    robot = ET.Element('robot', name=f'gantry_{hand}')
    def link(name, mass=1.):
        el = ET.SubElement(robot, 'link', name=name)
        inertial = ET.SubElement(el, 'inertial')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', ixx='0.01', iyy='0.01', izz='0.01', ixy='0', ixz='0', iyz='0')
    link('gantry_base')
    parent = 'gantry_base'
    for name, kind, axis in GANTRY_JOINTS:
        link(name + '_link', 0.5)
        joint = ET.SubElement(robot, 'joint', name=name, type=kind)
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=name + '_link')
        ET.SubElement(joint, 'axis', xyz=axis)
        limit = (-20, 20) if kind == 'prismatic' else (-6.3, 6.3)
        ET.SubElement(joint, 'limit', lower=str(limit[0]), upper=str(limit[1]), effort='1e9', velocity='100')
        parent = name + '_link'
    if hand == 'fetch':
        source = ET.parse(FETCH_URDF).getroot()
        keep_links = {'gripper_link', 'r_gripper_finger_link', 'l_gripper_finger_link'}
        keep_joints = {'r_gripper_finger_joint', 'l_gripper_finger_joint'}
        root_link, base_dir = 'gripper_link', FETCH_URDF.parent
    else:
        source = ET.parse(ALLEGRO_URDF).getroot()
        keep_links = {el.get('name') for el in source.findall('link')}
        keep_joints = {el.get('name') for el in source.findall('joint')}
        root_link, base_dir = 'hand_root', ALLEGRO_URDF.parent
    for el in source:
        if el.tag == 'link' and el.get('name') in keep_links or el.tag == 'joint' and el.get('name') in keep_joints:
            for visual in el.findall('visual'):  # headless physics only; collision geometry is kept
                el.remove(visual)
            _absolute_meshes(el, base_dir)
            if el.tag == 'link' and el.find('inertial') is None:
                inertial = ET.SubElement(el, 'inertial')
                ET.SubElement(inertial, 'mass', value='0.01')
                ET.SubElement(inertial, 'inertia', ixx='1e-5', iyy='1e-5', izz='1e-5', ixy='0', ixz='0', iyz='0')
            robot.append(el)
        elif el.tag == 'material':
            robot.append(el)
    mount = ET.SubElement(robot, 'joint', name='mount', type='fixed')
    ET.SubElement(mount, 'parent', link=parent)
    ET.SubElement(mount, 'child', link=root_link)
    rpy = Rotation.from_matrix(R_NOMINAL).as_euler('xyz')
    ET.SubElement(mount, 'origin', xyz='0 0 0', rpy=' '.join(f'{a:.6f}' for a in rpy))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'gantry_{hand}.urdf'
    ET.ElementTree(robot).write(path)
    return path


class Scene:
    def __init__(self, hand, object_idx, urdf):
        self.p = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.p)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.p)
        pb.loadURDF('plane.urdf', physicsClientId=self.p)
        self.table = pb.loadURDF('table/table.urdf', baseOrientation=pb.getQuaternionFromEuler([0, 0, np.pi / 2]),
                                 useFixedBase=True, physicsClientId=self.p)
        self.table_height = max(pb.getAABB(self.table, l, physicsClientId=self.p)[1][2]
                                for l in range(-1, pb.getNumJoints(self.table, physicsClientId=self.p)))
        objects = sorted(path for path in (UPSTREAM / 'assets/ycb_objects').iterdir() if path.is_dir())
        self.object_dir = objects[object_idx]
        self.obj = pb.loadURDF(str(self.object_dir / 'coacd_decomposed_object_one_link.urdf'), [0, 0, 1.0],
                               physicsClientId=self.p)
        self.hand = pb.loadURDF(str(urdf), useFixedBase=True, flags=pb.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.p)
        info = [pb.getJointInfo(self.hand, j, physicsClientId=self.p) for j in range(pb.getNumJoints(self.hand, physicsClientId=self.p))]
        self.joints = [j[0] for j in info if j[2] != pb.JOINT_FIXED]
        self.names = [j[1].decode() for j in info if j[2] != pb.JOINT_FIXED]
        self.gantry = [self.names.index(n) for n, _, _ in GANTRY_JOINTS]
        self.fingers = [i for i, n in enumerate(self.names) if not n.startswith('g')]
        self.hand_links = {j[0] for j in info if not j[12].decode().startswith('g')}
        self.gains = [GANTRY_GAIN if i in self.gantry else FINGER_GAIN for i in range(len(self.joints))]
        self.target = np.zeros(len(self.joints))
        self.target[self.gantry[2]] = 2.0  # park the hand high above the table while the object settles
        for j, value in zip(self.joints, self.target):
            pb.resetJointState(self.hand, j, value, physicsClientId=self.p)
        for _ in range(480):  # same settling as the environment
            self._command(); pb.stepSimulation(physicsClientId=self.p)
        self.ticks = 0

    def _command(self):
        pb.setJointMotorControlArray(self.hand, self.joints, pb.POSITION_CONTROL, targetPositions=self.target.tolist(),
                                     forces=[MAX_FORCE] * len(self.joints), positionGains=self.gains, physicsClientId=self.p)

    def call(self, n=1):
        for _ in range(n):
            self._command()
            for _ in range(SUBSTEPS):
                pb.stepSimulation(physicsClientId=self.p)
            self.ticks += SUBSTEPS

    def set_pose(self, position, rotation):
        """Gantry targets placing the hand frame at a world pose (rotation as matrix)."""
        deviation = Rotation.from_matrix(rotation @ R_NOMINAL.T).as_euler('ZYX')
        for index, value in zip(self.gantry, list(position) + list(deviation)):
            self.target[index] = value

    def move(self, position, rotation, calls):
        start = np.array([self.target[i] for i in self.gantry[:3]])
        for k in range(1, calls + 1):
            self.set_pose(start + (np.asarray(position) - start) * k / calls, rotation)
            self.call()

    def measured_xy(self):
        return np.array([pb.getJointState(self.hand, self.joints[i], physicsClientId=self.p)[0] for i in self.gantry[:2]])

    def object_state(self):
        position, orientation = pb.getBasePositionAndOrientation(self.obj, physicsClientId=self.p)
        return np.array(position), np.array(orientation)

    def hand_frame(self):
        state = pb.getLinkState(self.hand, self.joints[self.gantry[-1]], physicsClientId=self.p)  # roll link
        return np.array(state[4]), Rotation.from_quat(state[5]).as_matrix() @ R_NOMINAL

    def contact_links(self):
        return {c[3] for c in pb.getContactPoints(self.hand, self.obj, physicsClientId=self.p) if c[9] > 0}

    def point_cloud(self, seed):
        mesh = trimesh.load(str(self.object_dir / 'textured.obj'), force='mesh')
        points, faces = trimesh.sample.sample_surface(mesh, 1024, seed=seed)
        position, orientation = self.object_state()
        rotation = Rotation.from_quat(orientation).as_matrix()
        return points @ rotation.T + position, mesh.face_normals[faces] @ rotation.T

    def close(self):
        pb.disconnect(self.p)


def grasp_fetch(scene, seed):
    """Reference controller's grasp: top antipodal pair, 17 cm pregrasp, descend, close, lift."""
    points, normals = scene.point_cloud(seed)
    grasp = antipodal_grasps(points, normals, table_height=scene.table_height)[0]
    rotation = Rotation.from_quat(grasp.quaternion).as_matrix()
    fingers = [scene.names.index('r_gripper_finger_joint'), scene.names.index('l_gripper_finger_joint')]
    for i in fingers: scene.target[i] = .05
    pre = grasp.position + [0, 0, .17]
    scene.set_pose(pre, rotation)
    for i in scene.gantry:
        pb.resetJointState(scene.hand, scene.joints[i], scene.target[i], physicsClientId=scene.p)
    for i in fingers: pb.resetJointState(scene.hand, scene.joints[i], .05, physicsClientId=scene.p)
    scene.call(20)
    scene.move(grasp.position, rotation, 60)
    for i in fingers: scene.target[i] = 0.
    scene.call(80)
    scene.move(grasp.position + [0, 0, .20], rotation, 60)
    scene.call(60)
    return {'grasp_width_m': float(grasp.width)}, rotation


ALLEGRO_FINGERS = {(1, 5, 9): 'mcp', (2, 6, 10): 'pip', (3, 7, 11): 'dip'}
# Scripted-teacher settings found by grid search on one recorded route (.tools/allegro_grid.py):
# one setting holds foam, banana, apple and mug; the tall can needs its own.
ALLEGRO_PER_OBJECT = {
    'can': {'reach': .14, 'palm_gap': .005, 'curl': 1.2, 'thumb': 1.3, 'along': True, 'spread': 0.},
    'default': {'reach': .14, 'palm_gap': .015, 'curl': 1.6, 'thumb': 1.6, 'along': False, 'spread': .3},
}
ALLEGRO_DEFAULT = {'reach': .13, 'palm_gap': .01, 'curl': 1.0, 'thumb': 1.0, 'along': False, 'spread': 0.}


def allegro_targets(curl, thumb, spread=0.):
    """Closing synergy: one curl level for the three fingers, one for the thumb."""
    targets = {'joint_0': spread, 'joint_4': 0., 'joint_8': -spread}
    for joints, kind in ALLEGRO_FINGERS.items():
        for j in joints:
            targets[f'joint_{j}'] = {'mcp': 1.2, 'pip': 1.4, 'dip': 1.2}[kind] * curl
    targets.update(joint_12=1.35, joint_13=.9 * thumb, joint_14=.9 * thumb, joint_15=.9 * thumb)
    return targets


def grasp_allegro(scene, seed, params=None):
    """Scripted top-down power grasp: palm down, fingers across the object's long axis.

    The thumb stays folded flat beside the palm during the descent (joint_12 at its lower
    limit) and swings into opposition while the fingers curl. ``reach`` places the object
    centre that far along the fingers from the hand root; ``palm_gap`` is the palm's
    clearance above the object top.
    """
    params = dict(ALLEGRO_DEFAULT, **(params or {}))
    points, _ = scene.point_cloud(seed)
    centre = np.median(points, axis=0)
    top = np.percentile(points[:, 2], 98)
    horizontal = points[:, :2] - centre[:2]
    _, _, vt = np.linalg.svd(horizontal - horizontal.mean(0), full_matrices=False)
    long_axis = vt[0]
    fingers_dir = (np.array([long_axis[0], long_axis[1], 0.]) if params['along']
                   else np.array([-long_axis[1], long_axis[0], 0.]))
    if fingers_dir[0] < 0: fingers_dir = -fingers_dir
    z_h = fingers_dir / np.linalg.norm(fingers_dir)
    x_h = np.array([0., 0., -1.])
    rotation = np.column_stack((x_h, np.cross(z_h, x_h), z_h))
    palm = np.array([centre[0], centre[1], top + params['palm_gap']]) - params['reach'] * z_h
    scene.target[scene.names.index('joint_12')] = .26
    pre = palm + [0, 0, .12]
    scene.set_pose(pre, rotation)
    for i, j in enumerate(scene.joints):
        pb.resetJointState(scene.hand, j, scene.target[i], physicsClientId=scene.p)
    scene.call(20)
    scene.move(palm, rotation, 60)
    closed = allegro_targets(params['curl'], params['thumb'], params['spread'])
    start = {n: scene.target[scene.names.index(n)] for n in closed}
    for k in range(1, 41):                         # ramp the closing synergy over 40 commands
        for n in closed:
            scene.target[scene.names.index(n)] = start[n] + (closed[n] - start[n]) * k / 40
        scene.call()
    scene.call(40)
    scene.move(palm + [0, 0, .20], rotation, 60)
    scene.call(60)
    return {'allegro_params': params}, rotation


def route(trace_case):
    rows = [json.loads(line) for line in (TRACES / f'{trace_case}_trajectory.jsonl').open()]
    xy = np.array([r['qpos'][:2] for r in rows if r['stage'] == 'carry'])
    return xy - xy[0]


def carry(scene, path, rotation, lookahead=.15, step=.075):
    """Carry-adapter motion: each decision heads 0.15 m ahead on the route, 0.075 m over 5 commands."""
    start = np.array([scene.target[i] for i in scene.gantry[:3]])
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    hand_pos, hand_rot = scene.hand_frame()
    obj0 = hand_rot.T @ (scene.object_state()[0] - hand_pos)
    lost_calls, decisions = 0, 0
    while True:
        here = np.array([scene.target[i] for i in scene.gantry[:2]]) - start[:2]
        nearest = int(np.argmin(np.linalg.norm(path - here, axis=1)))
        if lengths[-1] - lengths[nearest] < .05 or decisions > 2000:
            break
        goal = np.array([np.interp(lengths[nearest] + lookahead, lengths, path[:, k]) for k in range(2)])
        delta = goal - here
        delta = delta / max(1e-9, np.linalg.norm(delta)) * min(step, np.linalg.norm(delta))
        base = scene.measured_xy()  # like the carry adapter: start each decision from the measured base
        for k in range(1, 6):
            xy = base + delta * k / 5
            scene.set_pose([xy[0], xy[1], start[2]], rotation)
            scene.call()
            held = len(scene.contact_links()) >= 2 and scene.object_state()[0][2] > scene.table_height + .05
            lost_calls = 0 if held else lost_calls + 1
            if lost_calls >= 5 or scene.object_state()[0][2] < scene.table_height + .05:
                hand_pos, hand_rot = scene.hand_frame()
                return {'retained': False, 'decisions': decisions, 'route_fraction': float(lengths[nearest] / lengths[-1])}
        decisions += 1
    scene.call(60)
    hand_pos, hand_rot = scene.hand_frame()
    obj1 = hand_rot.T @ (scene.object_state()[0] - hand_pos)
    held = len(scene.contact_links()) >= 2 and scene.object_state()[0][2] > scene.table_height + .05
    return {'retained': bool(held), 'decisions': decisions, 'route_fraction': 1.0,
            'slip_in_hand_mm': float(np.linalg.norm(obj1 - obj0) * 1000)}


def trial(task):
    hand, object_idx, trace_case, seed, urdf = task
    scene = Scene(hand, object_idx, urdf)
    try:
        z0 = scene.object_state()[0][2]
        if hand == 'fetch':
            details, rotation = grasp_fetch(scene, seed)
        else:
            name = OBJECTS[object_idx]
            details, rotation = grasp_allegro(scene, seed, ALLEGRO_PER_OBJECT.get(name, ALLEGRO_PER_OBJECT['default']))
        lifted = scene.object_state()[0][2] - z0
        contacts = len(scene.contact_links())
        result = {'hand': hand, 'object': OBJECTS[object_idx], 'route': trace_case, 'lift_m': float(lifted),
                  'contact_links': contacts, 'grasped': bool(lifted > .10 and contacts >= 2), **details}
        if result['grasped']:
            result.update(carry(scene, route(trace_case), rotation))
        else:
            result.update(retained=False, decisions=0, route_fraction=0.)
        result['physics_ticks'] = scene.ticks
        return result
    except Exception as exc:
        return {'hand': hand, 'object': OBJECTS[object_idx], 'route': trace_case, 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', nargs='+', default=['fetch', 'allegro'])
    parser.add_argument('--routes', type=int, default=10)
    parser.add_argument('--workers', type=int, default=44)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    summary = json.loads((TRACES / 'summary.json').read_text())
    failed = {f['case_id'] for f in summary['failed_cases']}
    successful = sorted({p.name[:-len('_trajectory.jsonl')] for p in TRACES.glob('*_trajectory.jsonl')} - failed)
    layouts, routes = set(), []
    for case in successful:           # one recorded successful carry per maze
        layout = case.rsplit('_object_', 1)[0]
        if layout not in layouts:
            layouts.add(layout); routes.append(case)
    routes = routes[:args.routes]
    urdfs = {hand: gantry_urdf(hand, args.output / 'urdf') for hand in args.hands}
    tasks = [(hand, obj, case, 1000 + i, str(urdfs[hand])) for hand in args.hands for obj in OBJECTS for i, case in enumerate(routes)]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool, (args.output / 'results.jsonl').open('w') as out:
        results = []
        for result in pool.map(trial, tasks):
            results.append(result)
            out.write(json.dumps(result) + '\n')
    for hand in args.hands:
        rows = [r for r in results if r['hand'] == hand]
        print(f'== {hand}: grasped {sum(r.get("grasped", False) for r in rows)}/{len(rows)}, '
              f'retained through carry {sum(r.get("retained", False) for r in rows)}/{len(rows)}, '
              f'errors {sum("error" in r for r in rows)}')
        for name in OBJECTS.values():
            sub = [r for r in rows if r['object'] == name]
            slips = [r['slip_in_hand_mm'] for r in sub if 'slip_in_hand_mm' in r]
            print(f'   {name:7s} grasped {sum(r.get("grasped", False) for r in sub)}/{len(sub)}  retained '
                  f'{sum(r.get("retained", False) for r in sub)}/{len(sub)}  median slip '
                  f'{np.median(slips) if slips else float("nan"):.1f} mm')


if __name__ == '__main__':
    main()
