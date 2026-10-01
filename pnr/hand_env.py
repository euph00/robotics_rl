"""Grasp-lift-hold with the Allegro rig, as a Gymnasium environment for RL.

One episode: an object (random among `objects`) rests on the table with a random yaw and a
small shift; the hand starts palm-down above it with a random wrist yaw. Each decision sets
wrist deltas (position ±1 cm, angles ±0.05 rad) and absolute finger joint targets, and runs
five physics ticks. Success: object lifted at least 15 cm above its resting height while
the thumb and another finger touch it.

Observations are in the palm frame (object pose, velocity, 32 surface points, fingertips),
plus finger joints, per-finger contacts, gravity direction, lift and the previous action.
Rewards are dense: approach (grasp point and fingertips to the object), contacts, a grasp
bonus (thumb + another finger), lift while grasped, and a success bonus.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .hand import GRASP_POINT, TABLE_TOP, HandRig, pregrasp_wrist

POINTS = 32
WRIST_STEP = np.array([.01, .01, .01, .05, .05, .05])
WRIST_ANGLE_LIMIT = 1.6
SUCCESS_LIFT = .15


class HandGraspEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, objects=(14, 5, 2, 7, 10), horizon=120, seed=0, friction=None, finger_kp=None,
                 render_mode=None):
        self.objects = list(objects)
        self.horizon = int(horizon)
        self.render_mode = render_mode
        self.rigs = {idx: HandRig(idx) for idx in self.objects}
        for rig in self.rigs.values():
            m = rig.model
            if finger_kp is not None:
                m.actuator_gainprm[:, 0] = finger_kp
                m.actuator_biasprm[:, 1] = -finger_kp
            if friction is not None:   # hand geoms only; MuJoCo uses the larger of the pair
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] in rig._hand_bodies and m.geom_contype[g]:
                        m.geom_friction[g, 0] = friction
        self.np_random = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1., 1., shape=(22,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._observation_size(),), dtype=np.float32)
        self.rig = None

    @staticmethod
    def _observation_size():
        return 3 + 6 + 6 + 3 * POINTS + 16 + 16 + 3 + 12 + 5 + 1 + 1 + 22

    # ---------------- episode ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        rng = self.np_random
        idx = (options or {}).get('object_idx', self.objects[rng.integers(len(self.objects))])
        self.rig = rig = self.rigs[idx]
        rig.restore(rig.settled)
        # Random yaw and a small shift of the resting object, then let it settle with the hand parked.
        d, q0 = rig.data, rig._obj_q
        yaw = rng.uniform(-np.pi, np.pi)
        turn = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, d.qpos[q0 + 3:q0 + 7].copy())
        d.qpos[q0 + 3:q0 + 7] = quat
        d.qpos[q0:q0 + 2] += rng.uniform(-.03, .03, 2)
        d.qvel[rig._obj_dof:rig._obj_dof + 6] = 0
        mujoco.mj_forward(rig.model, d)
        for _ in range(30):
            rig.tick()
        self.start_xy = rig.object_pose()[0][:2].copy()
        top = rig.object_points_world()[:, 2].max()
        centre = rig.object_pose()[0]
        clearance = rng.uniform(.03, .06)    # palm surface above the object's top
        height = (top + .011 + clearance - .05) - centre[2]
        wrist = pregrasp_wrist(rig, rng.uniform(-np.pi, np.pi), height=height, offset=rng.uniform(-.02, .02, 2))
        rig.set_wrist(wrist)
        self.wrist_home = wrist.copy()
        self.finger_target = rig.finger_open.copy()
        rig.data.ctrl[:] = self.finger_target
        mujoco.mj_forward(rig.model, d)
        self.steps, self.previous_action = 0, np.zeros(22)
        self.ever_success = False
        return self._observation(), {'object_idx': idx}

    def step(self, action):
        rig = self.rig
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        target = rig.wrist_target + action[:6] * WRIST_STEP
        target[:2] = np.clip(target[:2], self.wrist_home[:2] - .25, self.wrist_home[:2] + .25)
        target[2] = np.clip(target[2], TABLE_TOP + .02, TABLE_TOP + .6)
        target[4:] = np.clip(target[4:], -WRIST_ANGLE_LIMIT, WRIST_ANGLE_LIMIT)
        fingers = rig.finger_low + (action[6:] + 1) / 2 * (rig.finger_high - rig.finger_low)
        rig.step(target, fingers)
        self.steps += 1
        self.previous_action = action
        reward, info = self._reward()
        terminated = info['fell']
        truncated = self.steps >= self.horizon
        return self._observation(), reward, terminated, truncated, info

    # ---------------- reward ----------------
    def _reward(self):
        rig = self.rig
        centre, _ = rig.object_pose()
        points = rig.object_points_world()
        grasp_gap = np.linalg.norm(rig.grasp_point() - centre)
        tips = rig.fingertips()
        tip_gaps = np.min(np.linalg.norm(tips[:, None, :] - points[None], axis=2), axis=1)
        fingers, palm, table, depth = rig.contacts()
        grasped = bool(fingers[3] and fingers[:3].any())
        lift = float(np.clip(centre[2] - rig.rest_height, 0., .2))
        success = grasped and lift >= SUCCESS_LIFT
        fell = centre[2] < TABLE_TOP - .05
        reward = (.5 * (1 - np.tanh(10 * grasp_gap)) + .5 * (1 - np.tanh(10 * tip_gaps.mean()))
                  + .25 * fingers.sum() / 4 + .5 * grasped + 20 * lift * grasped + 2 * success - 5 * fell)
        self.ever_success |= success
        info = {'success': success, 'ever_success': self.ever_success, 'lift': lift, 'grasped': grasped,
                'contacts': int(fingers.sum()), 'palm_contact': bool(palm), 'table_contact': bool(table),
                'penetration_mm': 1000 * depth, 'fell': bool(fell), 'object_idx': rig.object_idx}
        return float(reward), info

    # ---------------- observation ----------------
    def _observation(self):
        rig = self.rig
        palm, rotation = rig.palm_pose()
        anchor = palm + rotation @ GRASP_POINT
        centre, object_rotation = rig.object_pose()
        local = lambda v: v @ rotation        # world vectors -> palm frame
        velocity = rig.object_velocity()
        points = local(rig.object_points_world()[:POINTS] - anchor)
        q, qd = rig.fingers()
        fingers, palm_contact, table, _ = rig.contacts()
        observation = np.concatenate([
            local(centre - anchor),
            (rotation.T @ object_rotation)[:, :2].ravel(),
            local(velocity[:3]), local(velocity[3:]) * .1,
            points.ravel(),
            2 * (q - rig.finger_low) / (rig.finger_high - rig.finger_low) - 1, np.clip(qd * .1, -5, 5),
            local(np.array([0., 0., -1.])),
            local(rig.fingertips() - centre).ravel(),
            fingers.astype(float), [float(palm_contact)],
            [np.clip(centre[2] - rig.rest_height, -.1, .3)],
            [anchor[2] - TABLE_TOP],
            self.previous_action])
        return observation.astype(np.float32)

    def render(self):
        return self.rig.render()
