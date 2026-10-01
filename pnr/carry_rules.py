"""Hand-written rules layered on a carry policy's commanded base motion.

Used to test whether how the robot moves (rather than the grasp itself) can
reduce objects slipping out of the grip during carry. Rules act only on the
2-D carry action; the policy, physics and adapter are unchanged.
"""
from __future__ import annotations

import numpy as np

# Indices into the 580-value carry observation (see CarrySkillEnv._observation).
LOCAL_EE = slice(558, 561)
LOCAL_OBJECT = slice(565, 568)


class CarryRules:
    """Wrap a policy with ``predict``; create one instance per episode.

    max_turn_deg: cap on the change of commanded heading between decisions.
    slip_drift_mm: once the object has slid this far down in the hand since the
        carry started, scale the commanded speed by ``slip_scale`` for the rest
        of the carry.
    """

    def __init__(self, policy, max_turn_deg=0., slip_drift_mm=0., slip_scale=.5):
        self.policy = policy
        self.max_turn = np.deg2rad(float(max_turn_deg))
        self.slip_drift = float(slip_drift_mm) / 1000.
        self.slip_scale = float(slip_scale)
        self.action_space, self.observation_space = policy.action_space, policy.observation_space
        self.previous = self.reference = None
        self.slowed = False
        self.turns_limited = 0

    def predict(self, observation, deterministic=True):
        action, state = self.policy.predict(observation, deterministic=deterministic)
        command = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1., 1.)
        command /= max(1., float(np.linalg.norm(command)))  # same normalisation as the carry adapter
        raw = np.asarray(observation['observation'], dtype=np.float64).reshape(-1)
        in_hand = raw[LOCAL_OBJECT] - raw[LOCAL_EE]
        if self.reference is None:
            self.reference = in_hand.copy()
        if self.slip_drift > 0 and self.reference[2] - in_hand[2] > self.slip_drift:
            self.slowed = True
        size = float(np.linalg.norm(command))
        if self.max_turn > 0 and self.previous is not None and size > 1e-6 and np.linalg.norm(self.previous) > 1e-6:
            heading = np.arctan2(command[1], command[0])
            previous = np.arctan2(self.previous[1], self.previous[0])
            turn = (heading - previous + np.pi) % (2 * np.pi) - np.pi
            if abs(turn) > self.max_turn:
                heading = previous + np.sign(turn) * self.max_turn
                command = size * np.array([np.cos(heading), np.sin(heading)])
                self.turns_limited += 1
        if self.slowed:
            command = command * self.slip_scale
        self.previous = command.copy()
        return command.astype(np.float32), state
