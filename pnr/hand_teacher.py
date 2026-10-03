"""Scripted power-grasp teacher for the Allegro rig, acting in HandGraspEnv's action space.

Closed loop on the measured state: turn the wrist so the thumb and fingers close across the
object's narrow horizontal axis (point-cloud principal axes), centre the grasp point over
the object, lower the palm to just above the object's top, close the fingers past contact
(the joint torque limit then sets the squeeze), lift, and hold still. Training data only:
its actions are wrist and finger deltas, the same interface the learned policy uses.
"""
from __future__ import annotations

import numpy as np

from .hand import GRASP_POINT, TABLE_TOP, palm_rotation, pregrasp_wrist, side_grasp_geometry, wrist_for  # noqa: F401
from .hand_env import FINGER_STEP, WRIST_STEP

# Closed power grasp: fingers flexed past typical contact, thumb rotated in and flexed.
CLOSED = np.array([0, 1.5, 1.4, 1.2] * 3 + [1.396, 1.1, 1.3, 1.3])


def narrow_axis_yaw(points, current, round_ratio=.75):
    """Wrist yaw that points the fingers along the object's narrow horizontal axis (closest to current).
    A nearly round footprint (minor/major spread above `round_ratio`) has no meaningful narrow axis:
    keep the current yaw."""
    xy = points[:, :2] - points[:, :2].mean(0)
    values, vectors = np.linalg.eigh(np.cov(xy.T))
    if values[0] > round_ratio * values[1]:
        return float(current)
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
        if getattr(getattr(rig, 'env', None), 'hand_mount', None) == 'wrist':
            # On the Fetch the fingers must point away from the robot: of the two equivalent narrow-axis
            # directions (thumb and fingers swapped), take the one within +-90 deg of +x.
            self.yaw = (self.yaw + np.pi / 2) % np.pi - np.pi / 2
        # The Fetch arm settles more slowly than the rig's wrist: looser hand-over tolerances there.
        on_fetch = hasattr(rig, 'env')
        self.align_tolerance = np.array([.02] * 3 + [.05] * 3) if on_fetch else .01
        self.descend_tolerance = .008 if on_fetch else .004
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
            if (np.abs(goal - rig.wrist()) < self.align_tolerance).all():
                self.phase = 'descend'
        elif self.phase == 'descend':
            goal = self._wrist_goal(above)
            touching, palm, table, _ = rig.contacts()
            # Guarded descent: close on reaching the height, or as soon as the hand touches the object or table.
            if abs(goal[2] - rig.wrist()[2]) < self.descend_tolerance or table or palm or touching.any():
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


class SideGraspTeacher(PowerGraspTeacher):
    """Side power grasp for upright objects: palm vertical beside the object (wrist roll 90 deg),
    fingers wrapping horizontally around it, thumb opposite; approach from the side at mid-height."""

    def __init__(self, env, lift=.26, noise=0., rng=None, along=.03, gap=-.005, height=.5):
        self.gap, self.height_fraction = gap, height
        super().__init__(env, lift=lift, noise=noise, rng=rng, along=along)

    def reset(self):
        rig = self.env.rig
        # Approach from the side the hand already faces; the grasp side points at the object.
        self.yaw = float(rig.wrist()[3])
        self.grip_point, self.contact_depth, self.top = side_grasp_geometry(rig, self.yaw, self.gap, self.height_fraction)
        self.clear_depth = self.contact_depth + .06
        # From a side pregrasp (palm already vertical beside the object) go straight to lowering.
        self.phase, self.ticks = ('lower' if abs(rig.wrist()[5] - np.pi / 2) < .05 else 'raise'), 0

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


def teacher_label(env, clearance=.008, lift=.2):
    """Power-grasp teacher action from the current state alone (for DAgger relabelling).

    Fingers open: align above the object, descend, then start closing; once closing has begun it
    commits (the closing fingers nudge the object, which must not trigger a realignment).
    Fingers closed: lift to `lift` above the resting height, then hold. Reopen (and realign) when the
    wrist has risen 4 cm without the object or a lifted object has lost its grasp. Carry: keep the grasp.
    """
    rig = env.rig
    closed = np.clip(CLOSED, rig.finger_low, rig.finger_high)
    open_hand = rig.finger_open.copy()
    open_hand[12] = 1.0
    wrist, fingers = rig.wrist_target, env.finger_target
    contact, palm, table, _ = rig.contacts()
    grasped = bool(contact[3] and contact[:3].any())
    closure = float(np.mean((fingers - open_hand)[[1, 2, 3, 5, 6, 7, 9, 10, 11]] /
                            (closed - open_hand)[[1, 2, 3, 5, 6, 7, 9, 10, 11]]))
    centre = rig.object_pose()[0]
    lifted = centre[2] - rig.rest_height
    goal, finger_goal = wrist.copy(), closed
    points = rig.object_points_world()
    yaw = narrow_axis_yaw(points, rig.wrist()[3])
    above = (points[:, 2].max() + .011 + clearance - .05) - centre[2]
    descend = pregrasp_wrist(rig, yaw, height=above)
    missed = closure > .9 and lifted < .01 and rig.wrist()[2] > descend[2] + .04 and getattr(env, 'phase', 'grasp') != 'carry'
    if getattr(env, 'phase', 'grasp') == 'carry' or (closure > .9 and not missed):
        if lifted < lift:                                # closed: lift (the grasp is judged by the object rising)
            goal[2] = wrist[2] + WRIST_STEP[2]
    elif missed or (lifted > .03 and not grasped):
        finger_goal = open_hand                          # wrist rose without the object, or lost it: reopen
    elif closure > .15 and rig.wrist()[2] < descend[2] + .03:
        pass                                             # closing at the grasp height: commit, hold the wrist
    elif closure > .15:
        finger_goal = open_hand                          # partly closed but high (a miss): open before realigning
    else:
        align = pregrasp_wrist(rig, yaw, height=above + .06)
        align[4:] = descend[4:] = 0.
        # Position and yaw only: contact with the table can tilt the capped wrist without misaligning the grasp.
        aligned = np.abs(np.r_[align[:2] - rig.wrist()[:2], (align[3] - rig.wrist()[3]) / 10]).max() < .01
        touching = bool(contact.any() or palm)                     # on the object (table contact is not a cue)
        blocked = wrist[2] - rig.wrist()[2] < -.015                 # target well below the wrist: it cannot descend
        low = rig.wrist()[2] < align[2] - .03
        if not aligned and not low:
            goal, finger_goal = align, open_hand
        else:
            goal = descend
            finger_goal = closed if abs(descend[2] - rig.wrist()[2]) < .004 or touching or blocked else open_hand
    action = np.zeros(22)
    action[:6] = np.clip((goal - wrist) / WRIST_STEP, -1, 1)
    action[6:] = np.clip((finger_goal - fingers) / FINGER_STEP, -1, 1)
    return action.astype(np.float32)
