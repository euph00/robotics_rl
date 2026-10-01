"""Goal-conditioned holonomic carry with physically collected training starts.

The route planner is used only while building the curriculum's start/goal bank.
Policy execution receives wall occupancy and one fixed XY goal, never waypoints.
Every retained object is held by ordinary contacts; snapshots restore physical
states collected by the original position-control interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .env import PickNavReachEnv
from .reference import ReferenceController


@dataclass
class CarryStart:
    arc_length: float
    base_xy: np.ndarray
    state: object


def wall_occupancy(wall_centers):
    """Course cell centers: x=column-3.5, y=5.5-row; 12 rows × 22 columns."""
    walls = np.asarray(wall_centers, dtype=float)
    result = np.zeros((12, 22), dtype=np.float32)
    if walls.size:
        if walls.ndim != 2 or walls.shape[1] < 2 or not np.isfinite(walls).all():
            raise ValueError('Expected finite wall-center coordinates')
        columns = np.rint(walls[:, 0] + 3.5).astype(int)
        rows = np.rint(5.5 - walls[:, 1]).astype(int)
        if np.any((columns < 0) | (columns >= 22) | (rows < 0) | (rows >= 12)):
            raise ValueError('Wall centers lie outside the preserved course grid')
        result[rows, columns] = 1.
    return result


def table_occupancy(bounds):
    """Separate table layer keeps the supplied wall map and task geometry clear."""
    low, high = np.asarray(bounds[0])[:2], np.asarray(bounds[1])[:2]
    x = np.arange(22, dtype=float) - 3.5
    y = 5.5 - np.arange(12, dtype=float)
    return ((x[None, :] + .5 > low[0]) & (x[None, :] - .5 < high[0])
            & (y[:, None] + .5 > low[1]) & (y[:, None] - .5 < high[1])).astype(np.float32)


class CarrySkillEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 24}
    goal_scale = 20.
    tolerance = .15  # metres; goals themselves are normalized world XY.
    delta_scale = .045
    repeat = 5
    failure_cost = 50.
    segment_lengths = (.3, .6, 1.2, 2.5, 5., 10.)

    def __init__(self, scene_seed=42, object_idx=14, seed=0, horizon=100,
                 evaluation=False, snapshot_spacing=.5, native=None, controller=None, build_starts=True,
                 failure_cost=50., curriculum_window=20, adaptive_horizon=False):
        if evaluation not in (False, True, 'easy', 'full'):
            raise ValueError("evaluation must be False, True/'easy', or 'full'")
        if int(horizon) < 1 or snapshot_spacing <= 0:
            raise ValueError('Positive horizon and snapshot spacing required')
        if native is not None and build_starts:
            raise ValueError('Existing simulators require build_starts=False')
        if not build_starts and (native is None or controller is None):
            raise ValueError('Binding requires an existing native environment and controller')
        if controller is not None and (native is None or controller.env is not native):
            raise ValueError('Controller must belong to the supplied native environment')
        self._owns_native = native is None
        self._has_start_bank = bool(build_starts)
        self.skill = 'carry'
        self.horizon = int(horizon)
        self.failure_cost=float(failure_cost)
        self.curriculum_window=int(curriculum_window)
        self.adaptive_horizon=bool(adaptive_horizon)
        self._episode_horizon=self.horizon
        if self.failure_cost<1 or self.curriculum_window<1:
            raise ValueError('Positive failure cost and curriculum window required')
        self.evaluation = evaluation
        self.evaluation_mode = ('easy' if evaluation is True else evaluation) or None
        self.rng = np.random.default_rng(seed)
        self.snapshot_spacing = float(snapshot_spacing)
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        # Two map layers (528) and actual robot/payload state (52).
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(580,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        })
        self.curriculum_level = 0
        self.reset_count = 0
        self._recent_success = []
        self.snapshots = []
        self.native = PickNavReachEnv(seed=scene_seed, object_idx=object_idx) if native is None else native
        self.controller = ReferenceController(self.native, seed=seed) if controller is None else controller
        try:
            self.map_features = np.concatenate([
                wall_occupancy(self.native.cube_positions).ravel(),
                table_occupancy(self.native.table_aabb()).ravel()])
            if build_starts:
                self._build_starts()
                self.setup_action_calls = self.native.step_count
                self.setup_physics_ticks = self.native.physics_steps
            else:
                # Read the existing physical state only: no reset, saveState,
                # control command, planner query, or simulator step is needed.
                self.setup_action_calls = self.setup_physics_ticks = 0
                self.route_length = self.full_carry_min_decisions = None
            self.setup_manifest = {
                'method': 'physical_reference_route_snapshots' if build_starts else 'live_inference_binding',
                'source_action_calls': self.setup_action_calls,
                'physics_ticks': self.setup_physics_ticks,
                'snapshots': len(self.snapshots),
                'route_length_m': self.route_length,
                'object_attachment': False,
                'planner_waypoints_during_policy': False,
            }
            self.goal = self.native.goal_pos[:2].copy() / self.goal_scale
            self._episode_steps = 0
            self._episode_success = False
            self._retention_failed = False
            self._missing_contact_calls = 0
            self._frozen_target = self.native.agent_qpos.copy()
            self._frozen_target[self.native.finger_slots] = 0.
            self.episode_scope = 'full_carry' if not build_starts else 'carry_segment_curriculum'
        except BaseException:
            if self._owns_native:
                self.native.close()
            raise

    @classmethod
    def bind(cls, native, controller, horizon=1000):
        """Attach the trained interface to an already-held uninterrupted episode.

        The caller owns the simulator. Binding does not advance or reset physics;
        obtain the initial policy input with ``bound._observation()`` and call
        ``step`` normally. ``reset`` is unavailable and ``close`` is non-owning.
        """
        return cls(native=native, controller=controller, build_starts=False,
                   horizon=horizon, evaluation='full')

    def _build_starts(self):
        grasp = self.controller.prepare_grasp()
        if not self.controller.execute_grasp(grasp):
            raise RuntimeError('Cannot train carry: physical reference grasp failed')
        self.controller.prepare_carry()
        if not self.native.contacts()['bilateral_contact']:
            raise RuntimeError('Cannot train carry: folded carry posture lost contact')
        initial = self.controller.base_position().copy()
        self.snapshots.append(CarryStart(0., initial, self.native.save_state()))
        points, arcs = [initial.copy()], [0.]
        previous = initial.copy()

        def collect(controller, observation, info):
            nonlocal previous
            position = controller.base_position().copy()
            distance = float(np.linalg.norm(position - previous))
            previous = position
            if distance > 1e-6:
                arcs.append(arcs[-1] + distance)
                points.append(position)
            if (arcs[-1] - self.snapshots[-1].arc_length >= self.snapshot_spacing
                    and info['bilateral_contact']):
                self.snapshots.append(CarryStart(arcs[-1], position, self.native.save_state()))

        self.controller.on_step = collect
        try:
            self.controller.carry()
        finally:
            self.controller.on_step = None
        if not self.native.contacts()['bilateral_contact'] or self.native.drop_events:
            raise RuntimeError('Cannot train carry: reference bank collection lost the payload')
        self.route_points = np.asarray(points, dtype=np.float64)
        self.route_arcs = np.asarray(arcs, dtype=np.float64)
        self.route_length = float(arcs[-1])
        if self.route_length < self.segment_lengths[0] + self.tolerance:
            raise RuntimeError('Carry reference route is too short for the curriculum')
        self.full_carry_min_decisions = int(np.ceil(self.route_length / self.delta_scale))

    def _point_at(self, arc_length):
        return np.array([np.interp(arc_length, self.route_arcs, self.route_points[:, axis]) for axis in range(2)])

    def _max_curriculum_level(self):
        # A fixed horizon must admit the whole requested segment with a margin.
        max_length = self.horizon * self.delta_scale * .75
        levels = [i for i, length in enumerate(self.segment_lengths) if length <= max_length]
        if self.route_length <= max_length:
            levels.append(len(self.segment_lengths))
        return max(levels, default=0)

    def _observation(self):
        obs = self.native.observe()
        base = self.controller.base_position()
        ee, orientation = self.controller.planner.ee_pose()
        local_ee = ee.copy()
        local_ee[:2] -= base
        local_object = np.asarray(obs['object_pos'], dtype=float).copy()
        local_object[:2] -= base
        velocity = self.native.object_velocity()
        contacts = np.asarray(self.native.contacts()['finger_contacts'], dtype=float)
        features = np.concatenate([self.map_features, base / self.goal_scale,
            obs['qpos'][2:], obs['qvel'], local_ee, orientation, local_object,
            obs['object_xyzw'], velocity, contacts]).astype(np.float32)
        return {'observation': features,
                'achieved_goal': (base / self.goal_scale).astype(np.float32),
                'desired_goal': self.goal.astype(np.float32)}

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Pure replay reward: no simulator, mutable goal, or route lookup."""
        distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1) * self.goal_scale

        def retained(item):
            return (bool(item.get('bilateral_contact', False))
                    and bool(item.get('retention_ok', True))
                    and not bool(item.get('physical_failure', False)))

        obligations = retained(info) if isinstance(info, dict) else np.asarray([retained(item) for item in info])
        failure = bool(info.get('physical_failure',False)) if isinstance(info,dict) else np.asarray([bool(item.get('physical_failure',False)) for item in info])
        ordinary = -np.asarray((distance >= self.tolerance) | ~np.asarray(obligations), dtype=np.float32)
        # A terminal collision must not be cheaper than continuing at -1/step.
        # Match 1/(1-gamma) when changing the learner's discount.
        return np.asarray(np.where(failure,-self.failure_cost,ordinary),dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if not self._has_start_bank:
            raise RuntimeError('Bound carry environment cannot reset an externally owned episode')
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        mode = options.get('evaluation_mode', self.evaluation_mode)
        if mode not in (None, 'easy', 'full'):
            raise ValueError('Unknown carry evaluation mode')
        if mode is None and len(self._recent_success) >= self.curriculum_window:
            if np.mean(self._recent_success[-self.curriculum_window:]) >= .65 and self.curriculum_level < self._max_curriculum_level():
                self.curriculum_level += 1
                self._recent_success = []
        level = 0 if mode == 'easy' else self.curriculum_level
        full = mode == 'full' or level >= len(self.segment_lengths)
        if full:
            start = self.snapshots[0]
            goal = self.native.goal_pos[:2].copy()
            segment_length = self.route_length
            self.episode_scope = 'full_carry'
        else:
            segment_length = min(self.segment_lengths[level], self.route_length - self.tolerance)
            candidates = [item for item in self.snapshots if item.arc_length + segment_length <= self.route_length]
            start = candidates[int(self.rng.integers(len(candidates)))]
            goal = self._point_at(start.arc_length + segment_length)
            self.episode_scope = 'easy_carry_diagnostic' if mode == 'easy' else 'carry_segment_curriculum'
        self._episode_horizon=(min(self.horizon,max(100,int(np.ceil(3*segment_length/self.delta_scale))+40))
                               if self.adaptive_horizon else self.horizon)
        self.native.restore_state(start.state, reset_counters=True)
        self._frozen_target = self.native.agent_qpos.copy()
        self._frozen_target[self.native.finger_slots] = 0.
        self.goal = goal / self.goal_scale
        self._episode_steps = 0
        self._episode_success = False
        self._retention_failed = False
        self._missing_contact_calls = 0
        self.reset_count += 1
        return self._observation(), {'curriculum_level': self.curriculum_level,
            'evaluation_scope': self.episode_scope, 'segment_length_m': segment_length,
            'start_arc_length_m': start.arc_length, 'route_length_m': self.route_length,
            'full_carry_min_decisions': self.full_carry_min_decisions,'episode_horizon':self._episode_horizon}

    def _collision(self):
        return any(self.native.touching(a, b) for a, b in
                   (('robot', 'maze'), ('object', 'maze'), ('robot', 'table'), ('object', 'table')))

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Carry action must contain two finite XY controls')
        action = np.clip(action, -1., 1.)
        action /= max(1., float(np.linalg.norm(action)))
        delta = self.delta_scale * action
        base_q = self.native.agent_qpos[:2].copy()
        before = self.native.step_count
        collision = False
        failure_reason = None
        for i in range(self.repeat):
            target = self._frozen_target.copy()
            target[:2] = base_q + delta * ((i + 1) / self.repeat)
            target[self.native.finger_slots] = 0.
            obs, info = self.native.step(target)
            self._missing_contact_calls = 0 if info['bilateral_contact'] else self._missing_contact_calls + 1
            collision |= self._collision()
            if float(obs['object_pos'][2]) < self.native.table_height + .05:
                failure_reason = 'dropped_object'
            elif self._missing_contact_calls >= 5:
                failure_reason = 'lost_grasp'
            elif collision:
                failure_reason = 'collision'
            base = self.controller.base_position()
            if not (-4. <= base[0] <= 18.5 and -6.5 <= base[1] <= 6.5):
                failure_reason = 'outside_scene'
            if failure_reason:
                self._retention_failed = True
                break
        self._episode_steps += 1
        observation = self._observation()
        distance = float(np.linalg.norm(observation['achieved_goal'] - observation['desired_goal']) * self.goal_scale)
        terminated = bool(failure_reason)
        truncated = self._episode_steps >= self._episode_horizon
        success = bool(distance < self.tolerance and info['bilateral_contact'] and not self._retention_failed)
        self._episode_success = bool((self._episode_success or success) and not terminated)
        info.update(is_success=success, episode_success=self._episode_success, goal_distance=distance,
            action_calls_delta=self.native.step_count - before,
            physics_ticks_delta=(self.native.step_count - before) * self.native.substeps,
            curriculum_level=self.curriculum_level, evaluation_scope=self.episode_scope,
            retention_ok=not self._retention_failed, physical_failure=terminated,
            failure_reason=failure_reason, collision=collision)
        reward = float(self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info))
        if terminated or truncated:
            self._recent_success.append(float(self._episode_success))
            self._recent_success = self._recent_success[-100:]
        return observation, reward, terminated, truncated, info

    def get_training_state(self):
        return {'curriculum_level': self.curriculum_level, 'recent_success': self._recent_success,
                'reset_count': self.reset_count, 'rng_state': self.rng.bit_generator.state}

    def set_training_state(self, state):
        level = int(state['curriculum_level'])
        if not 0 <= level <= len(self.segment_lengths):
            raise ValueError('Invalid carry curriculum level')
        self.curriculum_level = level
        self._recent_success = list(state['recent_success'])
        self.reset_count = int(state['reset_count'])
        self.rng.bit_generator.state = state['rng_state']

    def render(self):
        return self.native.render_rgb(camera='follow')

    def close(self):
        if self._owns_native:
            self.native.close()
