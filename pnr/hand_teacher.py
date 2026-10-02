"""Scripted power-grasp teacher for the Allegro rig, acting in HandGraspEnv's action space.

Closed loop on the measured state: turn the wrist so the thumb and fingers close across the
object's narrow horizontal axis (point-cloud principal axes), centre the grasp point over
the object, lower the palm to just above the object's top, close the fingers past contact
(the joint torque limit then sets the squeeze), lift, and hold still. Training data only:
its actions are wrist and finger deltas, the same interface the learned policy uses.
"""
from __future__ import annotations

import numpy as np

from .hand import GRASP_POINT, TABLE_TOP, pregrasp_wrist
from .hand_env import FINGER_STEP, WRIST_STEP

# Closed power grasp: fingers flexed past typical contact, thumb rotated in and flexed.
CLOSED = np.array([0, 1.5, 1.4, 1.2] * 3 + [1.396, 1.1, 1.3, 1.3])


def narrow_axis_yaw(points, current):
    """Wrist yaw that points the fingers along the object's narrow horizontal axis (closest to current)."""
    xy = points[:, :2] - points[:, :2].mean(0)
    _, vectors = np.linalg.eigh(np.cov(xy.T))
    minor = vectors[:, 0]
    yaw = np.arctan2(minor[1], minor[0])
    options = yaw + np.pi * np.arange(-2, 3)
    return float(options[np.argmin(np.abs(options - current))])


class PowerGraspTeacher:
    """`along`: palm-frame position of the object centre along the fingers (GRASP_POINT has .03);
    `scale`: how far the fingers close toward CLOSED (thumb rotation always full)."""

    def __init__(self, env, clearance=.008, lift=.2, noise=0., rng=None, along=.03, scale=1.):
        self.env, self.clearance, self.lift, self.noise = env, clearance, lift, noise
        self.along, self.scale = along, scale
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.phase, self.ticks = 'align', 0
        rig = self.env.rig
        points = rig.object_points_world()
        self.yaw = narrow_axis_yaw(points, rig.wrist()[3])
        self.top = points[:, 2].max()

    def _wrist_goal(self, height):
        rig = self.env.rig
        goal = pregrasp_wrist(rig, self.yaw, height=height)
        shift = GRASP_POINT[2] - self.along          # move the wrist so the object sits at `along`
        goal[:2] += shift * np.array([np.cos(self.yaw), np.sin(self.yaw)])
        goal[4:] = 0.
        return goal

    def _closed(self):
        rig = self.env.rig
        closed = np.clip(CLOSED, rig.finger_low, rig.finger_high)
        target = rig.finger_open + self.scale * (closed - rig.finger_open)
        target[12] = closed[12]
        return target

    def act(self):
        rig, env = self.env.rig, self.env
        centre = rig.object_pose()[0]
        above = (self.top + .011 + self.clearance - .05) - centre[2]     # palm surface just above the top
        wrist = rig.wrist_target
        fingers = env.finger_target
        finger_goal = rig.finger_open.copy()
        finger_goal[12] = 1.0                                            # thumb pre-rotated toward opposition
        if self.phase == 'align':
            goal = self._wrist_goal(above + .06)
            if np.abs(goal - rig.wrist()).max() < .01:
                self.phase = 'descend'
        elif self.phase == 'descend':
            goal = self._wrist_goal(above)
            if abs(goal[2] - rig.wrist()[2]) < .004:
                self.phase = 'close'
        elif self.phase == 'close':
            goal = self._wrist_goal(above)
            finger_goal = self._closed()
            self.ticks += 1
            if self.ticks > 25:
                self.phase, self.lift_start = 'lift', rig.wrist()[2]
        else:
            goal = wrist.copy()
            finger_goal = self._closed()
            if self.phase == 'lift':
                goal[2] = min(wrist[2] + WRIST_STEP[2], self.lift_start + self.lift)
                if rig.wrist()[2] >= self.lift_start + self.lift - .005:
                    self.phase = 'hold'
        action = np.zeros(22)
        action[:6] = np.clip((goal - wrist) / WRIST_STEP, -1, 1)
        action[6:] = np.clip((finger_goal - fingers) / FINGER_STEP, -1, 1)
        if self.noise:
            action = np.clip(action + self.rng.normal(0, self.noise, 22), -1, 1)
        return action.astype(np.float32)


def run_teacher(env, object_idx, seed, noise=0., horizon=None, kind='power', **params):
    """One teacher episode from a normal start; returns transitions and the final info."""
    obs, _ = env.reset(seed=seed, options={'object_idx': object_idx})
    teacher = TEACHERS[kind](env, noise=noise, rng=np.random.default_rng(seed), **params)
    rows, info = [], {}
    for _ in range(horizon or env.horizon):
        action = teacher.act()
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append((obs, action, reward, next_obs, terminated))
        obs = next_obs
        if terminated or truncated:
            break
    return rows, info


SIDE_CLEARANCE = .066   # m: palm half-width 5.65 cm plus margin
MOUNT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)   # palm frame in the wrist frame (pnr/hand.py)


def palm_rotation(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx @ MOUNT


def wrist_for(point, palm_offset, yaw, pitch=0., roll=0.):
    """Wrist pose that puts `point` (world) at `palm_offset` in the palm frame."""
    return np.array([*(np.asarray(point) - palm_rotation(yaw, pitch, roll) @ palm_offset), yaw, pitch, roll])


class SideGraspTeacher(PowerGraspTeacher):
    """Side power grasp for upright objects: palm vertical beside the object (wrist roll 90 deg),
    fingers wrapping horizontally around it, thumb opposite; approach from the side at mid-height."""

    def __init__(self, env, lift=.2, noise=0., rng=None, along=.03, gap=.006, height=.5):
        self.gap, self.height_fraction = gap, height
        super().__init__(env, lift=lift, noise=noise, rng=rng, along=along)

    def reset(self):
        self.phase, self.ticks = 'raise', 0
        rig = self.env.rig
        points = rig.object_points_world()
        centre = points.mean(0)
        self.top, self.bottom = points[:, 2].max(), points[:, 2].min()
        # Approach from the side the hand already faces; the grasp side points at the object.
        self.yaw = float(rig.wrist()[3])
        side = np.array([-np.sin(self.yaw), np.cos(self.yaw)])        # palm +x after a +90 deg roll
        radius = float(np.max(np.abs((points[:, :2] - centre[:2]) @ side)))
        # A vertical palm reaches ~5.7 cm below its centre: grip high enough to clear the table.
        height = max(self.bottom + self.height_fraction * (self.top - self.bottom), TABLE_TOP + SIDE_CLEARANCE)
        self.grip_point = np.array([centre[0], centre[1], height])
        self.contact_depth = radius + .011 + self.gap                  # palm surface 1.1 cm in front of the palm origin
        self.clear_depth = self.contact_depth + .06

    def _goal(self, depth, z_shift=0.):
        point = self.grip_point + [0., 0., z_shift]
        return wrist_for(point, np.array([depth, 0., self.along]), self.yaw, 0., np.pi / 2)

    def act(self):
        rig, env = self.env.rig, self.env
        wrist, fingers = rig.wrist_target, env.finger_target
        open_hand = rig.finger_open.copy()
        open_hand[12] = 1.0
        finger_goal = open_hand
        if self.phase == 'raise':          # roll the palm vertical high above the object, then go round it
            goal = rig.wrist().copy()
            goal[2] = max(goal[2], self.top + .16)
            goal[5] = np.pi / 2
            if abs(rig.wrist()[5] - np.pi / 2) < .05 and rig.wrist()[2] >= self.top + .15:
                self.phase = 'around'
        elif self.phase == 'around':
            goal = self._goal(self.clear_depth)
            goal[2] = max(goal[2], self.top + .16)
            if np.abs(goal[:2] - rig.wrist()[:2]).max() < .01:
                self.phase = 'lower'
        elif self.phase == 'lower':
            goal = self._goal(self.clear_depth)
            if np.abs(goal - rig.wrist()).max() < .008:
                self.phase = 'approach'
        elif self.phase == 'approach':
            goal = self._goal(self.contact_depth)
            if np.abs(goal - rig.wrist()).max() < .006:
                self.phase = 'close'
        elif self.phase == 'close':
            goal = self._goal(self.contact_depth)
            finger_goal = self._closed()
            self.ticks += 1
            if self.ticks > 25:
                self.phase, self.lift_start = 'lift', rig.wrist()[2]
        else:
            goal = wrist.copy()
            finger_goal = self._closed()
            if self.phase == 'lift':
                goal[2] = min(wrist[2] + WRIST_STEP[2], self.lift_start + self.lift)
                if rig.wrist()[2] >= self.lift_start + self.lift - .005:
                    self.phase = 'hold'
        action = np.zeros(22)
        delta = goal - wrist
        action[:6] = np.clip(delta / WRIST_STEP, -1, 1)
        action[6:] = np.clip((finger_goal - fingers) / FINGER_STEP, -1, 1)
        if self.noise:
            action = np.clip(action + self.rng.normal(0, self.noise, 22), -1, 1)
        return action.astype(np.float32)


TEACHERS = {'power': PowerGraspTeacher, 'side': SideGraspTeacher}
