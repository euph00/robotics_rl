"""The full task with the Allegro on the Fetch: planned approach, learned hand grasp, planned rest.

The planner (ReferenceController) folds the arm, drives to the table and moves the hand to a
pregrasp; the hand policy (scripts/train_hand.py, HandGraspEnv's observation and actions) then
grasps and lifts through FetchHandRig on this task environment, until it has held a 15 cm lift
with thumb and finger for HOLD_TO_CARRY decisions. From there the hand's servo targets stay
fixed while the planner's carry posture, A* maze carry, reach and final hold run unchanged.
Every hand decision goes through the controller's tick, so contact counters, lift timing and
recordings see it. After the grasp the hand's servo targets stay fixed (frozen grip) through the
carry posture, maze carry and reach; fingers_after_grasp='policy' lets the hand policy keep setting
the finger targets instead (wrist actions ignored), the sequence train_hand.py --task-carry trains.
Development comparison 2026-10-04 (validation mazes, random starts): imitation + RL seed 2 delivered
43/50 with the frozen grip, 37/50 with the policy holding; imitation 24 vs 25; wide objects 1 vs 1.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .fetch_gripper import PAD_HALF_LENGTH, FetchGripperRig
from .fetch_hand import READY, FetchHandRig, plan_tuck
from .hand import pregrasp_wrist, side_grasp_geometry, wrist_for
from .hand_env import FINGER_STEP, SIDE_GRASP_OBJECTS, HandGraspEnv
from .hand_teacher import narrow_axis_yaw
from .reference import ReferenceController

TOPDOWN = Rotation.from_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])).as_quat()


class HandPolicy:
    """A train_hand.py PPO checkpoint with its observation normalisation."""

    def __init__(self, run_dir, step):
        import json
        from stable_baselines3 import PPO, SAC
        config = json.loads((Path(run_dir) / 'config.json').read_text()) if (Path(run_dir) / 'config.json').is_file() else {}
        algorithm = SAC if config.get('algo') in ('sac', 'rlpd') else PPO
        base = Path(run_dir) / 'checkpoints' / f'step_{int(step):08d}'
        self.model = algorithm.load(str(base) + '.zip', device='cpu')
        stats = pickle.loads(Path(str(base) + '_vecnormalize.pkl').read_bytes())
        self.mean, self.std, self.clip = stats.obs_rms.mean, np.sqrt(stats.obs_rms.var + stats.epsilon), stats.clip_obs
        self.source = f'{Path(run_dir).name}/{int(step)}'

    def act(self, observation):
        normal = np.clip((observation[:len(self.mean)] - self.mean) / self.std, -self.clip, self.clip)
        return self.model.predict(normal.astype(np.float32), deterministic=True)[0]


class TeacherPolicy:
    """The scripted grasp teacher (pnr/hand_teacher.py) as a hand policy: side grasp for side-grasp
    objects, else top-down. It reads the state through the task, not the observation."""

    source = 'scripted teacher'

    def bind(self, task):
        from .hand_teacher import PowerGraspTeacher, SideGraspTeacher
        teacher = SideGraspTeacher if task.rig.object_idx in SIDE_GRASP_OBJECTS else PowerGraspTeacher
        self.teacher = teacher(task)

    def act(self, observation):
        return self.teacher.act()


class HandPlannerSystem(ReferenceController):
    """The full task with a learned (or scripted) grasp: planned approach to the training pregrasp, the
    grasp policy through the end effector's rig, then the planned carry posture, maze carry and reach.
    Either end effector (env.effector 'allegro' or 'gripper', the shared trainer's two rigs); the gripper
    carries in the planner's own tuck pose."""

    def __init__(self, env, policy, horizon=300, clearance=.045, wrist_box=(.08, .5), fingers_after_grasp='frozen', cradle=False,
                 side_grasp_objects=SIDE_GRASP_OBJECTS, **kwargs):
        if getattr(env, 'effector', 'gripper') not in ('allegro', 'gripper'):
            raise ValueError('HandPlannerSystem needs effector "allegro" or "gripper"')
        super().__init__(env, **kwargs)
        self.gripper = env.effector == 'gripper'
        self.policy, self.horizon, self.clearance = policy, int(horizon), float(clearance)
        self._holding = None          # the grasp's HandGraspEnv once it holds the object
        self.wrist_box = wrist_box
        self.fingers_after_grasp = fingers_after_grasp     # 'policy' or 'frozen' (servo targets fixed)
        self.cradle = cradle                               # carry palm-up under the object (plan_tuck)
        self.side_grasp_objects = tuple(side_grasp_objects)   # hand pregrasp beside these (as the policy trained)
        self.rig = None

    def hand_pregrasp(self):
        """Rig wrist pose for the pregrasp: palm down `clearance` above the top, fingers across the object's
        narrow horizontal axis (the one nearest pointing away from the robot); side-grasp objects: palm
        vertical on the robot's side, 6 cm out."""
        rig = self.rig
        if self.gripper:                            # jaws from above, closing across the narrow axis
            points = rig.object_points_world()
            spread = points[:, :2] - points[:, :2].mean(0)
            narrow = np.linalg.eigh(spread.T @ spread)[1][:, 0]
            yaw = float(np.arctan2(-narrow[0], narrow[1]))
            yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
            top, centre = points[:, 2].max(), rig.object_pose()[0]
            return rig.pregrasp(yaw, top + self.clearance + PAD_HALF_LENGTH - centre[2])
        if self.env.object_idx in self.side_grasp_objects:
            yaw = -np.pi / 2                       # palm +x (towards the object) along world +x
            grip, depth, _ = side_grasp_geometry(rig, yaw)
            return wrist_for(grip, np.array([depth + .06, 0., .03]), yaw, 0., np.pi / 2)
        top = rig.object_points_world()[:, 2].max()
        centre = rig.object_pose()[0]
        yaw = narrow_axis_yaw(rig.object_points_world(), 0.)
        return pregrasp_wrist(rig, yaw, height=(top + .011 + self.clearance - .05) - centre[2])

    def prepare_hand_grasp(self):
        """Fold the arm with the hand already in its pregrasp orientation, drive to the table, move the hand
        to 20 cm above the pregrasp and come straight down. With the wrist-end mount the hand reaches
        ~25 cm past the flange: turning it near the table, or sweeping between planner waypoints, pushes
        the object away."""
        Rig = FetchGripperRig if self.gripper else FetchHandRig
        self.rig = Rig(env=self.env, stepper=lambda action: self.tick(action, 'grasp'))
        wrist = self.pregrasp = self.hand_pregrasp()
        _, rotation = self.rig._flange_target(wrist)
        orientation = Rotation.from_matrix(rotation).as_quat()
        base, error = self.base_position(), None
        for dx, dy, z in ((.3, 0., 1.15), (.35, 0., 1.), (.3, .15, 1.1), (.3, -.15, 1.1), (.4, 0., .95)):
            try:                                    # fold poses near the base, first reachable one
                self.move_ee([base[0] + dx, base[1] + dy, z], orientation, 'fold', closed=False if self.gripper else None)
                break
            except ValueError as exc:
                error = exc
        else:
            raise error
        self.move_base([-.78, 0.], stage='approach')
        # Plan backwards from the pregrasp: its arm configuration as the training resets choose it
        # (collision-free IK seeded from the initial arm pose), then 20 cm straight up by IK continuation;
        # move to the top of that path and come down along it. A descent planned from above can end on a
        # branch that cannot reach the pregrasp (the hand then starts too high for the policy).
        # Exactly as a training reset: the arm parked at READY (solved from the initial pose), then the
        # pregrasp solved from there (pnr/fetch_hand.py FetchHandRig.__init__ and set_wrist).
        initial = np.r_[self.env.agent_qpos[:3], self.env.init_qpos[3:]]
        ready = self.rig.wrist_to_joints(READY, collision_free=True, seed=initial)
        path = [self.rig.wrist_to_joints(wrist, collision_free=True, seed=ready)]
        for lift in np.linspace(0, .2, 9)[1:]:
            q = path[-1].copy()
            q[self.rig._arm] = self.rig._arm_step(wrist + [0., 0., lift, 0., 0., 0.], iterations=10, start=q)
            with self.env.query():
                self.env.set_joint_positions(q)
                position, _ = self.rig._flange_target(wrist + [0., 0., lift, 0., 0., 0.])
                if np.linalg.norm(self.env.ee_pose()[0] - position) > .01:
                    break
            path.append(q)
        if self.gripper:                            # jaws open, as every training episode starts
            for q in path:
                q[self.finger_slots] = self.rig.finger_open
        self.move_joint(path[-1], stage='pregrasp', include_object=True, step_size=.012)
        for q in reversed(path[:-1]):
            self.move_joint(q, stage='pregrasp', include_object=True, step_size=.006)
        self.rig.wrist_target = wrist.copy()        # read the reached pose's angles near the planned ones
        self.stage_results['hand_pregrasp'] = {'wrist': wrist.tolist(), 'reached': self.rig.wrist().tolist()}
        return wrist

    def execute_hand_grasp(self):
        task = HandGraspEnv(objects=[], physics=self.env.physics, horizon=self.horizon, carry=True,
                            wrist_box=self.wrist_box, robot='fetch-gripper' if self.gripper else 'rig')
        observation = task.attach(self.rig)
        if hasattr(self.policy, 'bind'):
            self.policy.bind(task)
        info, decisions = {}, 0
        for decisions in range(1, self.horizon + 1):
            observation, _, terminated, _, info = task.step(self.policy.act(observation))
            if task.phase == 'carry' or terminated:
                break
        held = task.phase == 'carry'
        if held and self.fingers_after_grasp == 'policy':
            self._holding = task
        self.stage_results['hand_grasp'] = {'policy': self.policy.source, 'decisions': decisions, 'held': held,
                                            'lift_m': info.get('lift'), 'contacts': info.get('contacts'),
                                            'palm_contact': info.get('palm_contact')}
        return held

    def tick(self, target, stage=None):
        """Planned ticks after the grasp: the hand policy sets the finger targets from its observation."""
        task = self._holding
        if task is not None and len(target) == len(self.env.joint_names):
            action = np.clip(self.policy.act(task._observation()), -1, 1)
            rig = task.rig
            task.finger_target = np.clip(task.finger_target + action[6:] * getattr(rig, 'finger_step', FINGER_STEP),
                                         rig.finger_low, rig.finger_high)
            task.previous_action = action
            if self.gripper:                        # the jaws are Fetch joints
                target = np.asarray(target, dtype=float).copy()
                target[self.finger_slots] = task.finger_target
            else:
                target = np.r_[target, task.finger_target]
        return super().tick(target, stage)

    def prepare_carry(self):
        """Retreat from the table, then tuck the held object close to the base for the maze (`plan_tuck`:
        object at or below the goal's height, hand turned about the vertical, held object kept off the robot's
        body), then a planned joint-space move. If no tuck is found, a straight-line path as far as it reaches."""
        if self.gripper:
            return super().prepare_carry()          # the gripper planner's tuck pose
        self.planner.set_payload()
        self.move_base([-1.25, 0], stage='retreat', step_size=.006, check=False)   # straight back from the table
        position, orientation = self.planner.ee_pose()
        q = plan_tuck(self.planner, self.env.agent_qpos.astype(np.float64), self.base_position(), position, orientation,
                      self.env.observe()['object_pos'], cradle=self.cradle)
        try:
            if q is None:
                raise RuntimeError('no tuck')
            self.move_joint(q, 'carry_posture', step_size=.006)
        except (ValueError, RuntimeError):             # no tuck, or no planned path to it
            return self._continuation_carry_posture(retreat=False)
        self.hold(50, 'hold')
        self.planner.set_payload()
        self.stage_results['carry_posture'] = {'tucked': True, 'tuck': plan_tuck.last,
                                               'object_position': self.env.observe()['object_pos'].tolist()}
        return self.env.observe()

    def _cartesian_path(self, start, goal, stage, steps=8, include_object=False, step_size=.006):
        """Move the palm from rig wrist pose `start` towards `goal` in `steps` straight-line waypoints, each
        solved by damped least squares from the current arm configuration and collision-checked (with the
        held object as payload). Stops at the last waypoint that is reachable in this branch and clear;
        returns the fraction of the path completed."""
        done = 0.
        for t in np.linspace(0, 1, steps + 1)[1:]:
            wrist = start + (goal - start) * t
            target = self.env.last_target.copy()
            target[self.rig._arm] = self.rig._arm_step(wrist, iterations=10)
            with self.env.query():
                self.env.set_joint_positions(target)
                position, _ = self.rig._flange_target(wrist)
                reachable = np.linalg.norm(self.env.ee_pose()[0] - position) < .01
            if not reachable or not self.planner.collision_free(target, include_object=include_object):
                break
            self.move_joint(target, stage, step_size=step_size, check=False)
            done = float(t)
        return done

    def _continuation_carry_posture(self, steps=12, reach=.30, height=1., retreat=True):
        """Retreat, then move the held object in a straight line, upright and without turning, to `reach`
        in front of the base at `height`. Each waypoint's arm targets continue the current arm
        configuration (damped least squares), so the arm never switches IK branch with the object in
        hand; each step is collision-checked with the object attached."""
        self.planner.set_payload()
        if retreat:
            self.move_base([-1.25, 0], stage='retreat', step_size=.006, check=False)   # straight back from the table
        start = self.rig.wrist()
        base = self.base_position()
        shift = np.array([base[0] + reach, base[1], height]) - self.env.observe()['object_pos']
        reached = self._cartesian_path(start, start + np.r_[shift, 0., 0., 0.], 'carry_posture', steps)
        self.hold(50, 'hold')
        self.planner.set_payload()
        self.stage_results['carry_posture'] = {'base_position': self.base_position().tolist(),
                                               'fraction_of_path': reached,
                                               'ee_position': self.planner.ee_pose()[0].tolist(),
                                               'object_position': self.env.observe()['object_pos'].tolist()}
        return self.env.observe()

    def run(self, stop_after='reach'):
        result = {'status': 'running', 'success': False, 'effector': 'allegro'}
        try:
            self.prepare_hand_grasp()
            if not self.execute_hand_grasp():
                raise RuntimeError('Hand grasp: no held 15 cm lift')
            self.hold(50, 'hold')
            if stop_after == 'pick':
                result['status'] = 'pick_complete'
            else:
                self.prepare_carry()
                self.carry()
                info = self.reach()
                result.update(status='complete', success=bool(info['complete_task_success']),
                              upstream_success=bool(info['success']))
                if not result['success']:
                    result.update(failed_stage='reach', error='Object outside the goal tolerance after the reach')
        except (ValueError, RuntimeError) as exc:
            result.update(status='failed', error=str(exc), failed_stage=self.stage)
        result.update(stages=self.stage_results, metrics=self.env.evaluate(),
                      wall_contact_steps=self.wall_contact_steps, robot_table_contact_steps=self.table_contact_steps,
                      first_lift_action=self.first_lift_action, maximum_object_height=self.max_object_height,
                      first_lift_physics_tick=None if self.first_lift_action is None else self.first_lift_action * self.env.substeps)
        return result
