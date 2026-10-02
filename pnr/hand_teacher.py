"""Scripted power-grasp teacher for the Allegro rig, acting in HandGraspEnv's action space.

Closed loop on the measured state: turn the wrist so the thumb and fingers close across the
object's narrow horizontal axis (point-cloud principal axes), centre the grasp point over
the object, lower the palm to just above the object's top, close the fingers past contact
(the joint torque limit then sets the squeeze), lift, and hold still. Training data only:
its actions are wrist and finger deltas, the same interface the learned policy uses.
"""
from __future__ import annotations

import numpy as np

from .hand import GRASP_POINT, pregrasp_wrist
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


def run_teacher(env, object_idx, seed, noise=0., horizon=None, **params):
    """One teacher episode from a normal start; returns transitions and the final info."""
    obs, _ = env.reset(seed=seed, options={'object_idx': object_idx})
    teacher = PowerGraspTeacher(env, noise=noise, rng=np.random.default_rng(seed), **params)
    rows, info = [], {}
    for _ in range(horizon or env.horizon):
        action = teacher.act()
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append((obs, action, reward, next_obs, terminated))
        obs = next_obs
        if terminated or truncated:
            break
    return rows, info
