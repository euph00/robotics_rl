"""Multi-layout physical carry training, separate from the frozen pilot protocol.

Only construction of physical curriculum states and demonstration actions uses a
route planner. Observations contain measured map/state and a single fixed goal.
The policy never receives route waypoints or shortest-path distances.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pnr.env import PickNavReachEnv
from pnr.carry import CarrySkillEnv
from pnr.reference import ReferenceController
from pnr.planning import maze_astar


@dataclass
class PhysicalStart:
    arc: float
    xy: np.ndarray
    state: object


def clean_path(path):
    path = np.asarray(path, dtype=float)
    return path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8]]


CORNER_SPEED, TEACHER_DECELERATION, DECISION_SECONDS = .15, .5, 25 / 240


def route_action(path, position, goal, scale, lookahead=.15, brake=False):
    """Training-only teacher, recomputing recovery from the measured position.

    With `brake` (realistic, rate-limited base) the action shrinks so the commanded speed
    (scale per decision at full action) stays below sqrt(v_c^2 + 2 a d), d being the path
    distance to the next turn of more than 30 degrees or to the goal: the base slows for corners.
    """
    path = clean_path(path)
    position = np.asarray(position)
    speed = None
    if len(path) < 2:
        target = np.asarray(goal)
    else:
        vectors = np.diff(path, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        t = np.clip(np.sum((position-path[:-1])*vectors, axis=1)/lengths**2, 0., 1.)
        projections = path[:-1]+t[:, None]*vectors
        index = int(np.argmin(np.linalg.norm(position-projections, axis=1)))
        arcs = np.r_[0., np.cumsum(lengths)]
        target_arc = min(arcs[index]+t[index]*lengths[index]+lookahead, arcs[-1])
        j = min(int(np.searchsorted(arcs, target_arc, side='right')-1), len(lengths)-1)
        target = path[j]+(target_arc-arcs[j])*vectors[j]/lengths[j]
        if np.linalg.norm(position-goal) < lookahead:
            target = np.asarray(goal)
        if brake:
            here = arcs[index]+t[index]*lengths[index]
            directions = vectors/lengths[:, None]
            turns = [arcs[k+1] for k in range(len(directions)-1)
                     if np.dot(directions[k], directions[k+1]) < np.cos(np.radians(30)) and arcs[k+1] > here]
            distance = min(turns + [arcs[-1]]) - here
            speed = np.sqrt(CORNER_SPEED**2 + 2*TEACHER_DECELERATION*max(distance, 0.))
    action = (target-position)/scale
    action = action/max(1., np.linalg.norm(action))
    if speed is not None:
        action = action*min(1., speed*DECISION_SECONDS/scale)
    return action.astype(np.float32)


class GeneralCarryEnv(CarrySkillEnv):
    """One maze's physically prepared state bank with explicit training controls."""
    protocol = 'physical-multimaze-carry-v1'

    def __init__(self, *, maze_seed=None, scene_seed=42, object_idx=14, seed=0,
                 horizon=900, delta_scale=.075, gamma=.995, shaping_weight=2.,
                 evaluation=False, build_bank=True, native=None, controller=None):
        owned = native is None
        native = PickNavReachEnv(seed=scene_seed, object_idx=object_idx, maze_seed=maze_seed) if owned else native
        controller = ReferenceController(native, seed=seed) if controller is None else controller
        try:
            if owned:
                grasp = controller.prepare_grasp()
                if not controller.execute_grasp(grasp):
                    raise RuntimeError('Physical carry setup could not grasp/lift the object')
                controller.prepare_carry()
            if not native.contacts()['bilateral_contact']:
                raise RuntimeError('Physical carry setup has no retained grasp')
            super().__init__(native=native, controller=controller, build_starts=False,
                             seed=seed, horizon=horizon, evaluation='full' if evaluation else False,
                             failure_cost=1./(1.-gamma))
            self._owns_native = owned
            self.delta_scale = float(delta_scale)
            self.gamma, self.shaping_weight = float(gamma), float(shaping_weight)
            self.full_evaluation = bool(evaluation)
            self.starts = [PhysicalStart(0., controller.base_position().copy(), native.save_state())]
            self.route_points = np.array([self.starts[0].xy])
            self.route_arcs = np.array([0.])
            self.setup_action_calls = native.step_count
            self.setup_physics_ticks = native.physics_steps
            self.initialization_physics_ticks = native.settle_steps if owned else 0
            self.route_setup_success = None
            self._goal_hold = 0
            self.total_episode_decisions = 0
            if build_bank:
                self._build_physical_bank()
            self.setup_action_calls = native.step_count
            self.setup_physics_ticks = native.physics_steps
            self.route_length = float(self.route_arcs[-1])
            self.full_carry_min_decisions = int(np.ceil(self.route_length/self.delta_scale))
            self.curriculum_level = 0
        except BaseException as exc:
            exc.setup_accounting = {'source_action_calls':native.step_count,
                                    'physics_ticks':native.physics_steps,
                                    'initialization_physics_ticks':native.settle_steps if owned else 0}
            if owned:
                native.close()
            raise

    @classmethod
    def bind(cls, native, controller, horizon=900, **kwargs):
        return cls(native=native, controller=controller, horizon=horizon,
                   evaluation=True, build_bank=False, **kwargs)

    def _build_physical_bank(self):
        path = maze_astar(self.native.cube_positions, self.controller.base_position(),
                          self.native.goal_pos[:2], table_bounds=self.native.table_aabb())
        points = [self.controller.base_position().copy()]
        arcs = [0.]
        obs = self._observation()
        # Every snapshot is reached by actual control; no joint/object teleportation.
        for i in range(self.horizon):
            action = route_action(path, obs['achieved_goal']*self.goal_scale,
                                  self.native.goal_pos[:2], self.delta_scale,
                                  brake=getattr(self.native, 'physics', 'course') == 'realistic')
            obs, _, terminated, truncated, info = self.step(action)
            xy = self.controller.base_position().copy()
            arc = arcs[-1]+float(np.linalg.norm(xy-points[-1]))
            points.append(xy); arcs.append(arc)
            if arc-self.starts[-1].arc >= 1. and info['bilateral_contact']:
                self.starts.append(PhysicalStart(arc, xy.copy(), self.native.save_state()))
            if terminated:
                raise RuntimeError('Physical route-bank failure: '+str(info.get('failure_reason')))
            if info['is_success']:
                self.route_setup_success = True
                break
            if truncated:
                raise RuntimeError('Physical route-bank teacher exceeded its horizon')
        else:
            raise RuntimeError('Physical route bank did not reach its goal')
        self.route_points, self.route_arcs = np.asarray(points), np.asarray(arcs)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Recompute both potentials against the supplied goal, including HER goals.
        # Scalar NumPy float32 arithmetic can promote to float64 while batch
        # arithmetic stays float32. Potentials subtract nearby distances, so
        # force one precision before recomputation and cast only the reward.
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        goal = np.asarray(desired_goal, dtype=np.float64)
        distance = np.linalg.norm(achieved-goal, axis=-1)*self.goal_scale
        single = isinstance(info, dict)
        infos = [info] if single else list(info)
        retained = np.asarray([bool(item.get('bilateral_contact', False)) and
                               bool(item.get('retention_ok', True)) for item in infos])
        failures = np.asarray([bool(item.get('physical_failure', False)) for item in infos])
        previous = np.asarray([item.get('previous_achieved_goal', achieved if single else achieved[i])
                               for i, item in enumerate(infos)], dtype=np.float64)
        previous_distance = np.linalg.norm(previous-goal, axis=-1)*self.goal_scale
        ordinary = -((distance >= self.tolerance) | ~retained).astype(np.float32)
        reward = np.where(failures, -self.failure_cost, ordinary)
        # Phi = -distance. True failure is absorbing with Phi=0; timeouts bootstrap.
        reward += self.shaping_weight*(previous_distance-self.gamma*np.where(failures, 0., distance))
        result = np.asarray(reward, dtype=np.float32)
        return result.reshape(-1)[0] if single else result

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        full = options.get('full', self.full_evaluation)
        lengths = (.6, 1.2, 3., 6., 12., float('inf'))
        length = lengths[min(self.curriculum_level, len(lengths)-1)]
        if full or self.route_length <= length or self.rng.random() < .2:
            start, goal = self.starts[0], self.native.goal_pos[:2].copy()
            segment_length = max(self.route_length, np.linalg.norm(start.xy-goal))
            scope = 'full_carry'
        else:
            possible = [s for s in self.starts if s.arc+length < self.route_length]
            start = possible[int(self.rng.integers(len(possible)))] if possible else self.starts[0]
            goal = np.array([np.interp(start.arc+length, self.route_arcs, self.route_points[:, j]) for j in range(2)])
            segment_length, scope = length, 'carry_segment_curriculum'
        self.native.restore_state(start.state, reset_counters=True)
        self.controller.planner.set_payload()
        self._frozen_target = self.native.agent_qpos.copy()
        self._frozen_target[self.native.finger_slots] = 0.
        self.goal = goal/self.goal_scale
        self._episode_steps = self._goal_hold = 0
        self._episode_success = self._retention_failed = False
        self._missing_contact_calls = 0
        self._episode_horizon = self.horizon if full else min(self.horizon, max(80, int(3*segment_length/self.delta_scale)+30))
        self.episode_scope = scope
        self.reset_count += 1
        return self._observation(), {'evaluation_scope':scope, 'maze_seed':self.native.maze_seed,
            'curriculum_level':self.curriculum_level, 'segment_length_m':float(segment_length)}

    def step(self, action):
        previous = (self.controller.base_position()/self.goal_scale).astype(np.float32)
        obs, _, terminated, truncated, info = super().step(action)
        info['previous_achieved_goal'] = previous
        self._goal_hold = self._goal_hold+1 if info['is_success'] else 0
        # End successful collection as a timeout, preserving goal-independent HER terminals.
        if self._goal_hold >= 5:
            truncated = True
        self.total_episode_decisions += 1
        return obs, float(self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)), terminated, truncated, info

    def set_curriculum_level(self, level):
        self.curriculum_level = int(np.clip(level, 0, 5))

    def get_training_state(self):
        return {'protocol':self.protocol, 'curriculum_level':self.curriculum_level,
                'reset_count':self.reset_count, 'rng_state':self.rng.bit_generator.state}

    def set_training_state(self, state):
        if state['protocol'] != self.protocol:
            raise ValueError('Mismatched physical carry training protocol')
        self.curriculum_level = int(state['curriculum_level'])
        self.reset_count = int(state['reset_count'])
        self.rng.bit_generator.state = state['rng_state']
