"""Grasp-lift-hold with the Allegro rig, as a Gymnasium environment for RL.

One episode: an object (random among `objects`) rests on the table with a random yaw and a
small shift; the hand starts palm-down above it with a random wrist yaw. Each decision moves
the wrist target (±1 cm, ±0.05 rad) and the finger joint targets (±0.08 rad; 'absolute' mode
sets the finger targets directly), and runs five physics ticks. Delta targets hold the hand
still under zero action, so exploration noise does not reopen a grasp. Success: object lifted at least 15 cm above its resting height while
the thumb and another finger touch it.

With `carry=True`, once the object has been held (success) for 10 decisions the wrist follows
a synthetic base carry for `carry_calls` decisions while the policy keeps controlling the
fingers (its wrist actions are ignored): +1 per decision while the object stays >= 10 cm up and
touches the hand twice, on top of the grasp-phase reward (so the carry is always worth
entering), -20 and termination if it drops, +10 for finishing the carry. Carry
motions reproduce the course base controller: commanded targets advancing 18 mm per call with
pauses at waypoints (A*), or jumping 7.5 cm every five calls (learned carry), along
axis-aligned corridors, executed by the course motor law on the wrist.

Observations are in the palm frame (object pose, velocity, 32 surface points, fingertips),
plus finger joints, per-finger contacts, gravity direction, lift and the previous action.
Rewards are dense but dominated by lifting: approach (grasp point and fingertips to the
object, at most 1 per step), small contact and grasp bonuses, 50 x lift height while grasped
(two or more hand contacts, palm included; up to 10) and +5 per step at success. Smaller lift weights leave a plateau where the hand
holds the object on the table forever. The wrist target stays within `wrist_lag` (default
2 cm) / 0.1 rad of the measured wrist, so the stiff course motor law cannot build up large
pressing forces.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .hand import GRASP_POINT, TABLE_TOP, HandRig, pregrasp_wrist, side_grasp_geometry, wrist_for

POINTS = 32
WRIST_STEP = np.array([.01, .01, .01, .05, .05, .05])
WRIST_ANGLE_LIMIT = 1.6
FINGER_STEP = .08
TICKS_PER_CALL = 5
HOLD_TO_CARRY = 10
CARRY_LIFT = .10


def synthetic_carry_realistic(rng, calls, acceleration=.75, tick=1 / 240):
    """Base xy positions (relative to the start) per native call under realistic base limits.

    Axis-aligned corridor segments; on each, the velocity reference rises to a cruise speed
    (0.5-0.86 m/s) and brakes to a corner speed (0-0.15 m/s) by the segment end, then turns
    90 degrees. The velocity vector is rate-limited per axis, so turns are short curves.
    """
    cruise = rng.uniform(.5, .86)
    heading = rng.integers(4) * np.pi / 2
    position, velocity, path = np.zeros(2), np.zeros(2), []
    while len(path) < calls:
        direction = np.array([np.cos(heading), np.sin(heading)])
        start, length, corner = position.copy(), rng.uniform(.5, 2.5), rng.uniform(0., .15)
        while np.dot(position - start, direction) < length and len(path) < calls:
            for _ in range(TICKS_PER_CALL):
                remaining = max(length - np.dot(position - start, direction), 0.)
                speed = min(cruise, np.sqrt(corner ** 2 + 2 * acceleration * remaining))
                velocity = velocity + np.clip(direction * speed - velocity, -acceleration * tick, acceleration * tick)
                position = position + velocity * tick
            path.append(position.copy())
        heading += rng.choice((-1, 1)) * np.pi / 2
    return np.array(path[:calls])


def synthetic_carry(rng, calls):
    """Commanded base xy targets (relative to the start) for `calls` native calls."""
    style = rng.choice(('astar', 'learned'))
    heading = rng.integers(4) * np.pi / 2
    targets, position, k = [], np.zeros(2), 0
    while len(targets) < calls:
        direction = np.array([np.cos(heading), np.sin(heading)])
        goal = position + direction * rng.uniform(.5, 2.5)
        if style == 'astar':
            while np.linalg.norm(goal - position) > 1e-9 and len(targets) < calls:
                step = goal - position
                position = position + step * min(1., .018 / np.linalg.norm(step))
                targets.append(position.copy())
            targets += [position.copy()] * 8
        else:
            while np.dot(goal - position, direction) > 0 and len(targets) < calls:
                k += 1
                if k % 5 == 0:
                    position = position + direction * .075
                targets.append(position.copy())
        heading += rng.choice((-1, 1)) * np.pi / 2
    return np.array(targets[:calls])
# Part-closed power grasp used for curriculum starts (fingers flexed, thumb rotated in).
CLOSED = np.array([0, 1.4, 1.2, 1.0] * 3 + [1.396, 1.0, 1.0, 1.0])
SUCCESS_LIFT = .15
# Upright objects too wide for a top-down claw start from a side pregrasp (palm vertical beside them).
SIDE_GRASP_OBJECTS = (0,)


class HandGraspEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, objects=(14, 5, 2, 7, 10), horizon=120, seed=0, friction=None, finger_kp=None,
                 finger_actions='delta', near_start=0., wrist_lag=.02, carry=False, carry_calls=200,
                 physics=None, render_mode=None, side_objects=SIDE_GRASP_OBJECTS, robot='rig'):
        if finger_actions not in ('delta', 'absolute'):
            raise ValueError('finger_actions must be delta or absolute')
        self.finger_actions = finger_actions
        # Training curriculum: this fraction of episodes starts with the palm just above the object
        # and the fingers part-closed, so holding and lifting are experienced early. Evaluate with 0.
        self.near_start = float(near_start)
        self.side_objects = set(side_objects)
        self.carry, self.carry_calls = bool(carry), int(carry_calls)
        self.wrist_lag = np.array([wrist_lag] * 3 + [.1] * 3)
        self.objects = list(objects)
        self.horizon = int(horizon)
        self.render_mode = render_mode
        if robot == 'fetch':        # the hand on the Fetch's arm at the course table (pnr/fetch_hand.py)
            from .fetch_hand import FetchHandRig
            self.rigs = {idx: FetchHandRig(idx, physics=physics) for idx in self.objects}
        else:
            self.rigs = {idx: HandRig(idx, physics=physics) for idx in self.objects}
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

    def set_curriculum(self, objects=None, near_start=None):
        """Change the training distribution between episodes (active objects, near-grasp start share)."""
        if objects is not None:
            missing = set(objects) - set(self.rigs)
            if missing:
                raise ValueError(f'Objects {sorted(missing)} were not built into this environment')
            self.objects = list(objects)
        if near_start is not None:
            self.near_start = float(near_start)
        return self.objects, self.near_start

    @staticmethod
    def _observation_size():
        return 3 + 6 + 6 + 3 * POINTS + 16 + 16 + 16 + 3 + 12 + 5 + 1 + 1 + 22 + 4

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
        if idx in self.side_objects:
            # Palm vertical, facing the object from a random side, 4-8 cm out at about mid-height.
            yaw = rng.uniform(-np.pi, np.pi)
            grip, depth, _ = side_grasp_geometry(rig, yaw)
            offset = [depth + rng.uniform(.04, .08), rng.uniform(-.01, .01), .03 + rng.uniform(-.01, .01)]
            wrist = wrist_for(grip + [0., 0., rng.uniform(-.01, .01)], np.array(offset), yaw, 0., np.pi / 2)
        else:
            wrist = pregrasp_wrist(rig, rng.uniform(-np.pi, np.pi), height=height, offset=rng.uniform(-.02, .02, 2))
        rig.set_wrist(wrist)
        self.wrist_home = wrist.copy()
        self.finger_target = rig.finger_open.copy()
        rig.data.ctrl[:] = self.finger_target
        mujoco.mj_forward(rig.model, d)
        if rng.random() < self.near_start and idx not in self.side_objects:
            low = (top + .011 + rng.uniform(.0, .015) - .05) - centre[2]
            wrist = pregrasp_wrist(rig, wrist[3], height=low, offset=rng.uniform(-.015, .015, 2))
            rig.set_wrist(wrist)
            self.wrist_home = wrist.copy()
            closed = np.clip(CLOSED, rig.finger_low, rig.finger_high)
            self.finger_target = rig.finger_open + rng.uniform(.2, .8) * (closed - rig.finger_open)
            mujoco.mj_forward(rig.model, d)
            for _ in range(8):
                rig.step(wrist, self.finger_target)
        self.steps, self.previous_action = 0, np.zeros(22)
        self.ever_success = False
        self.phase, self.held, self.carry_step, self.carried = 'grasp', 0, 0, False
        return self._observation(), {'object_idx': idx}

    def attach(self, rig):
        """Begin an episode on `rig` as it is (no reset randomisation): e.g. a FetchHandRig wrapping a
        task environment whose arm a planner has brought to the pregrasp."""
        self.rig = rig
        rig.wrist_target = rig.wrist()
        self.wrist_home = rig.wrist_target.copy()
        self.finger_target = np.clip(rig.data.ctrl.copy(), rig.finger_low, rig.finger_high)
        self.steps, self.previous_action = 0, np.zeros(22)
        self.ever_success = False
        self.phase, self.held, self.carry_step, self.carried = 'grasp', 0, 0, False
        return self._observation()

    def step(self, action):
        rig = self.rig
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        current = rig.wrist()
        target = np.clip(rig.wrist_target + action[:6] * WRIST_STEP, current - self.wrist_lag, current + self.wrist_lag)
        target[:2] = np.clip(target[:2], self.wrist_home[:2] - .25, self.wrist_home[:2] + .25)
        target[2] = np.clip(target[2], TABLE_TOP + .02, TABLE_TOP + .6)
        target[4:] = np.clip(target[4:], -WRIST_ANGLE_LIMIT, WRIST_ANGLE_LIMIT)
        if self.phase == 'carry':
            dx, dy = self.carry_path[min(self.carry_step, len(self.carry_path) - 1)]
            target = self.carry_origin + [dx, dy, 0., 0., 0., 0.]
            self.carry_step += 1
        if self.finger_actions == 'delta':
            self.finger_target = np.clip(self.finger_target + action[6:] * FINGER_STEP, rig.finger_low, rig.finger_high)
        else:
            self.finger_target = rig.finger_low + (action[6:] + 1) / 2 * (rig.finger_high - rig.finger_low)
        unstable = rig.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        rig.step(target, self.finger_target)
        # MuJoCo resets the state after a numerical blow-up; never learn from that transition.
        unstable = rig.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > unstable
        self.steps += 1
        self.previous_action = action
        reward, info = self._reward()
        terminated = info['fell']
        if unstable:
            reward, terminated, info['unstable'] = 0., True, True
        if self.phase == 'carry':
            lift, hold = info['lift'], info['contacts'] + info['palm_contact'] >= 2
            kept = lift >= CARRY_LIFT and hold
            # Keep paying the grasp-phase reward (lift, grasp, success) and add the carry bonus: a lower
            # carry reward made hovering just under the success height (never triggering the carry) optimal.
            reward = reward + 1. * kept
            if lift < .05:
                reward, terminated, info['dropped_in_carry'] = -20., True, True
            elif self.carry_step >= self.carry_calls:
                self.carried = kept
                reward += 10. * kept
        elif self.carry:
            self.held = self.held + 1 if info['success'] else 0
            if self.held >= HOLD_TO_CARRY:
                self.phase, self.carry_step = 'carry', 0
                generate = synthetic_carry_realistic if rig.physics == 'realistic' else synthetic_carry
                self.carry_path = generate(self.np_random, self.carry_calls)
                if hasattr(rig, 'begin_carry'):     # the Fetch drives the path with its base
                    self.carry_path = rig.begin_carry(self.carry_path)
                self.carry_origin = rig.wrist_target.copy()
        info.update(phase=self.phase, carried=self.carried)
        done_carry = self.phase == 'carry' and self.carry_step >= self.carry_calls
        truncated = done_carry or self.steps >= self.horizon + (self.carry_calls if self.phase == 'carry' else 0)
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
        holding = int(fingers.sum()) + int(palm) >= 2
        lift = float(np.clip(centre[2] - rig.rest_height, 0., .2))
        success = grasped and lift >= SUCCESS_LIFT
        fell = centre[2] < TABLE_TOP - .05
        reward = (.5 * (1 - np.tanh(10 * grasp_gap)) + .5 * (1 - np.tanh(10 * tip_gaps.mean()))
                  + .1 * fingers.sum() / 4 + .25 * grasped + 50 * lift * holding + 5 * success - 5 * fell)
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
            2 * (self.finger_target - rig.finger_low) / (rig.finger_high - rig.finger_low) - 1,
            local(np.array([0., 0., -1.])),
            local(rig.fingertips() - centre).ravel(),
            fingers.astype(float), [float(palm_contact)],
            [np.clip(centre[2] - rig.rest_height, -.1, .3)],
            [anchor[2] - TABLE_TOP],
            self.previous_action,
            [float(getattr(self, 'phase', 'grasp') == 'carry')],
            local(rig.palm_velocity()) * .5])
        return observation.astype(np.float32)

    def render(self):
        return self.rig.render()
