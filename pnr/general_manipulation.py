"""Multi-object grasp observations using supplied antipodal proposal geometry.

The shared planner already selects the pregrasp orientation/contact pair. This
adapter exposes its measured object-relative grasp-frame position and width to
learning. It adds no teacher actions, route, object label, or goal-dependent
cached feature. Dynamics, sparse HER rewards and action meanings are unchanged.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .features import RelativeGoalExtractor
from .skills import ManipulationSkillEnv
from .local_control import LocalCartesianIK


class BoundedGraspIK(LocalCartesianIK):
    """Fast local IK with a same-neighborhood bounded least-squares fallback.

    Clipping a damped Jacobian step at the torso limit can stall even when a
    nearby exact solution exists. The already verified global least-squares
    solver handles those active bounds. Reject its answer unless it remains
    within the original local torso/arm neighborhood; no branch jump is allowed.
    Normal endpoint collision validation remains the skill adapter's job.
    """
    def __init__(self, env, planner, **kwargs):
        super().__init__(env, **kwargs)
        self.planner = planner
        self.fallback_attempts = self.fallback_successes = 0
        self.last_used_fallback = False

    def ik(self, position, quaternion, seed=None, **kwargs):
        initial = np.asarray(self.env.agent_qpos if seed is None else seed, dtype=float).copy()
        self.last_used_fallback = False
        try:
            return super().ik(position, quaternion, seed=initial, **kwargs)
        except ValueError as original:
            self.fallback_attempts += 1
            try:
                target = self.planner.ik(position, quaternion, seed=initial,
                                         tolerance=kwargs.get('tolerance', .0015))
                limits = np.full(len(self.active), self.max_joint_change)
                limits[0] = .04
                if np.any(np.abs(target[self.active]-initial[self.active]) > limits+1e-7):
                    raise ValueError('Global IK answer exceeds the retained local neighborhood')
                self.fallback_successes += 1
                self.last_used_fallback = True
                self.last_ik_error = self.planner.last_ik_error
                with self.planner.query():
                    self._put(target)
                    achieved_position, achieved_rotation = self._pose()
                    self.last_position_error = float(np.linalg.norm(np.asarray(position)-achieved_position))
                    self.last_orientation_error = float((Rotation.from_quat(quaternion)*achieved_rotation.inv()).magnitude())
                return target
            except ValueError as fallback_error:
                raise original from fallback_error


class GeneralGraspEnv(ManipulationSkillEnv):
    observation_version = 'antipodal-relative-grasp-v1'

    def __init__(self, skill='grasp', bounded_ik_fallback=True, goal_lift_m=.20, **kwargs):
        if skill != 'grasp':
            raise ValueError('GeneralGraspEnv currently supports grasp only')
        super().__init__(skill=skill, **kwargs)
        self.grasp_proposal_mode = self.controller.grasp_proposal_mode
        self.goal_lift_m = float(goal_lift_m)
        self.base_goal[2] += self.goal_lift_m-.20
        self.goal = self.base_goal.copy()
        if bounded_ik_fallback:
            self.local_ik = BoundedGraspIK(self.native, self.controller.planner)
        self.control_version = 'bounded_local_global_ik-v1' if bounded_ik_fallback else 'original_local_ik-v1'
        measured = self.native.observe()
        self.grasp_local_position = Rotation.from_quat(measured['object_xyzw']).inv().apply(
            self.controller.grasp.position-measured['object_pos'])
        self.proposal_width = float(self.controller.grasp.width)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)})

    @classmethod
    def bind(cls, native, controller, skill='grasp', horizon=200, *, goal_lift_m=.20,
             bounded_ik_fallback=True):
        """Read-only binding of an existing physical pregrasp; no setup or reset."""
        return cls(skill=skill, horizon=horizon, evaluation=True, native=native,
                   controller=controller, build_starts=False, goal_lift_m=goal_lift_m,
                   bounded_ik_fallback=bounded_ik_fallback)

    def measured_grasp_target(self):
        measured = self.native.observe()
        return measured['object_pos'] + Rotation.from_quat(measured['object_xyzw']).apply(
            self.grasp_local_position)

    def _observation(self):
        observation = super()._observation()
        ee, _ = self.controller.planner.ee_pose()
        target_delta = self.measured_grasp_target()-ee
        observation['observation'] = np.r_[observation['observation'], target_delta,
                                          self.proposal_width].astype(np.float32)
        return observation

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if not self.evaluation and self.goal_lift_m != .20:
            # Maintain the requested final lift while retaining the inherited
            # small-lift curriculum only until its final level.
            self.goal[2] -= self.goal_lift_m-.20
            if self.curriculum_level >= 3:
                self.goal[2] += self.goal_lift_m-.20
            observation = self._observation()
        return observation, info

    @property
    def curriculum_protocol(self):
        return 'multi-object-antipodal-lift-then-pregrasp-v1'


class GeneralGraspFeatures(RelativeGoalExtractor):
    """All state plus useful-scale live proposal and goal differences.

    HER can change achieved/desired goals in replay. Goal subtraction therefore
    runs here after relabeling, while recorded proposal/contact geometry stays
    independent of the desired lift goal.
    """
    def __init__(self, observation_space, goal_scale=5., geometry_scale=20.):
        if observation_space['observation'].shape != (54,):
            raise ValueError('General grasp features require54 recorded state values')
        super().__init__(observation_space, goal_scale=goal_scale)
        self.geometry_scale = float(geometry_scale)
        self._features_dim += 7

    def forward(self, observations):
        raw = observations['observation']
        return torch.cat([super().forward(observations),
                          (raw[:, 28:31]-raw[:, 35:38])*self.geometry_scale,
                          raw[:, 50:54]*self.geometry_scale], dim=1)


class ContactGraspTeacher:
    """Training-only current-state approach, closure and held-goal controller."""
    def __init__(self, env, max_action=.65, close_distance=.018, approach_control='component'):
        self.env = env
        self.max_action = float(max_action)
        self.close_distance = float(close_distance)
        if approach_control not in ('component', 'ray-tilted'):
            raise ValueError('Unsupported demonstration approach control')
        self.approach_control = approach_control

    def action(self, observation):
        env = self.env
        ee, _ = env.controller.planner.ee_pose()
        if env.native.contacts()['bilateral_contact']:
            delta = observation['desired_goal']-observation['achieved_goal']
            finger = -1.
            phase = 'held_goal_correction'
        else:
            delta = env.measured_grasp_target()-ee
            close = np.linalg.norm(delta) <= self.close_distance
            finger = -1. if close else 1.
            phase = 'close' if close else 'approach'
        translation = np.clip(delta/env.delta_scale, -self.max_action, self.max_action)
        if (self.approach_control == 'ray-tilted' and phase != 'held_goal_correction'
                and abs(Rotation.from_quat(env.orientation).apply([1.,0.,0.])[2]) < .999):
            # Componentwise saturation can leave a checked tilted approach ray
            # and request an unreachable sideways/downward move. This changes
            # training-teacher labels only; learned inference is never projected.
            translation = delta/max(env.delta_scale, np.linalg.norm(delta)/self.max_action)
        return np.r_[translation, finger].astype(np.float32), phase


class GraspSchema(gym.Env):
    """Spaces and replay-safe reward only; cannot fabricate transitions."""
    def __init__(self):
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)})
        self.tolerance = .04

    compute_reward = ManipulationSkillEnv.compute_reward

    def reset(self, **kwargs):
        raise RuntimeError('Schema-only environment cannot generate experience')

    def step(self, action):
        raise RuntimeError('Schema-only environment cannot generate experience')


def make_grasp_case(object_idx, scene_seed, seed=0, horizon=100, evaluation=False, *,
                    goal_lift_m=.20, bounded_ik_fallback=True, grasp_proposal_mode='legacy'):
    """Construct a physical case, preserving costs even on setup exceptions."""
    from .env import PickNavReachEnv
    from .reference import ReferenceController
    native = PickNavReachEnv(seed=scene_seed, object_idx=object_idx)
    try:
        controller = ReferenceController(native, seed=seed, grasp_proposal_mode=grasp_proposal_mode)
        env = GeneralGraspEnv(native=native, controller=controller, seed=seed,
                              horizon=horizon, evaluation=evaluation, goal_lift_m=goal_lift_m,
                              bounded_ik_fallback=bounded_ik_fallback)
        env._owns_native = True
        return env
    except BaseException as error:
        error.setup_accounting = {'source_action_calls': native.step_count,
                                  'physics_ticks': native.physics_steps,
                                  'initialization_physics_ticks': native.settle_steps}
        native.close()
        raise


class MultiObjectGrasp(GraspSchema):
    """Cached physically prepared object/point-sampling cases; no hidden mazes."""
    def __init__(self, cases, seed=0, horizon=100, cache_size=4, episodes_per_case=4,
                 lift_range=(.20, .20), grasp_proposal_mode='legacy'):
        super().__init__()
        from collections import OrderedDict
        if not cases:
            raise ValueError('At least one training case is required')
        self.cases = list(cases)
        if grasp_proposal_mode not in ('legacy', 'clearance-v2'):
            raise ValueError('Unsupported grasp proposal mode')
        self.grasp_proposal_mode = grasp_proposal_mode
        self.lift_range = tuple(float(value) for value in lift_range)
        if len(self.lift_range) != 2 or not .15 <= self.lift_range[0] <= self.lift_range[1] <= .30:
            raise ValueError('Grasp lift range must be ordered and within15–30cm')
        self.rng = np.random.default_rng(seed)
        self.seed, self.horizon = int(seed), int(horizon)
        self.cache, self.active = OrderedDict(), None
        self.cache_size, self.episodes_per_case = int(cache_size), int(episodes_per_case)
        self.reset_count = 0
        self.setup_action_calls = self.setup_physics_ticks = self.initialization_physics_ticks = 0
        self.failed_setup_attempts = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        index = (self.reset_count//self.episodes_per_case) % len(self.cases)
        case = self.cases[index]
        key = (int(case['object_idx']), int(case['scene_seed']))
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                _, old = self.cache.popitem(last=False)
                old.close()
            try:
                env = make_grasp_case(*key, seed=self.seed+index, horizon=self.horizon,
                                      grasp_proposal_mode=self.grasp_proposal_mode)
            except Exception as error:
                costs = getattr(error, 'setup_accounting', {})
                self.setup_action_calls += costs.get('source_action_calls', 0)
                self.setup_physics_ticks += costs.get('physics_ticks', 0)
                self.initialization_physics_ticks += costs.get('initialization_physics_ticks', 0)
                self.failed_setup_attempts.append({'case': case, 'error': repr(error), 'costs': costs})
                raise
            self.setup_action_calls += env.setup_action_calls
            self.setup_physics_ticks += env.setup_action_calls*env.native.substeps
            self.initialization_physics_ticks += env.native.settle_steps
            self.cache[key] = env
        self.active = self.cache[key]
        self.cache.move_to_end(key)
        # BC covers the declared lift range and all four physical pregrasp heights.
        self.active.curriculum_level = 6
        self.reset_count += 1
        observation, info = self.active.reset(seed=int(self.rng.integers(0, 2**31)), options=options)
        lift = float(self.rng.uniform(*self.lift_range))
        self.active.goal[2] += lift-self.active.goal_lift_m
        info['goal_lift_m'] = lift
        return self.active._observation(), info

    def step(self, action):
        if self.active is None:
            raise RuntimeError('Reset before stepping')
        return self.active.step(action)

    def get_training_state(self):
        return {'protocol': 'multi-object-grasp-v1', 'reset_count': self.reset_count,
                'rng_state': self.rng.bit_generator.state, 'lift_range': self.lift_range,
                'grasp_proposal_mode': self.grasp_proposal_mode,
                'failed_setup_attempts': self.failed_setup_attempts}

    def set_training_state(self, state):
        if state['protocol'] != 'multi-object-grasp-v1':
            raise ValueError('Wrong manipulation training protocol')
        if state.get('grasp_proposal_mode', 'legacy') != self.grasp_proposal_mode:
            raise ValueError('Cannot silently change resumed grasp proposal mode')
        if tuple(state.get('lift_range', (.20, .20))) != self.lift_range:
            raise ValueError('Cannot silently change resumed grasp goal distribution')
        self.reset_count = int(state['reset_count'])
        self.rng.bit_generator.state = state['rng_state']
        self.failed_setup_attempts = list(state['failed_setup_attempts'])

    def close(self):
        for env in self.cache.values():
            env.close()
        self.cache.clear()
