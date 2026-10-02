"""The full task with the Allegro on the Fetch: planned approach, learned hand grasp, planned rest.

The planner (ReferenceController) folds the arm, drives to the table and moves the hand to a
pregrasp; the hand policy (scripts/train_hand.py, HandGraspEnv's observation and actions) then
grasps and lifts through FetchHandRig on this task environment, until it has held a 15 cm lift
with thumb and finger for HOLD_TO_CARRY decisions. From there the hand's servo targets stay
fixed while the planner's carry posture, A* maze carry, reach and final hold run unchanged.
Every hand decision goes through the controller's tick, so contact counters, lift timing and
recordings see it.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .fetch_hand import FetchHandRig
from .hand import pregrasp_wrist, side_grasp_geometry, wrist_for
from .hand_env import SIDE_GRASP_OBJECTS, HandGraspEnv
from .hand_teacher import narrow_axis_yaw
from .reference import ReferenceController

TOPDOWN = Rotation.from_matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])).as_quat()


class HandPolicy:
    """A train_hand.py PPO checkpoint with its observation normalisation."""

    def __init__(self, run_dir, step):
        from stable_baselines3 import PPO
        base = Path(run_dir) / 'checkpoints' / f'step_{int(step):08d}'
        self.model = PPO.load(str(base) + '.zip', device='cpu')
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
    def __init__(self, env, policy, horizon=300, clearance=.045, wrist_box=(.08, .5), **kwargs):
        if getattr(env, 'effector', 'gripper') != 'allegro':
            raise ValueError('HandPlannerSystem needs an environment with effector="allegro"')
        super().__init__(env, **kwargs)
        self.policy, self.horizon, self.clearance = policy, int(horizon), float(clearance)
        self.wrist_box = wrist_box
        self.rig = None

    def hand_pregrasp(self):
        """Rig wrist pose for the pregrasp: palm down `clearance` above the top, fingers across the object's
        narrow horizontal axis (the one nearest pointing away from the robot); side-grasp objects: palm
        vertical on the robot's side, 6 cm out."""
        rig = self.rig
        if self.env.object_idx in SIDE_GRASP_OBJECTS:
            yaw = -np.pi / 2                       # palm +x (towards the object) along world +x
            grip, depth, _ = side_grasp_geometry(rig, yaw)
            return wrist_for(grip, np.array([depth + .06, 0., .03]), yaw, 0., np.pi / 2)
        top = rig.object_points_world()[:, 2].max()
        centre = rig.object_pose()[0]
        yaw = narrow_axis_yaw(rig.object_points_world(), 0.)
        return pregrasp_wrist(rig, yaw, height=(top + .011 + self.clearance - .05) - centre[2])

    def prepare_hand_grasp(self):
        base = self.base_position()
        self.move_ee([base[0] + .35, base[1], 1.1], TOPDOWN, 'fold')
        self.move_base([-.78, 0.], stage='approach')
        self.rig = FetchHandRig(env=self.env, stepper=lambda action: self.tick(action, 'grasp'))
        wrist = self.pregrasp = self.hand_pregrasp()
        self.move_joint(self.rig.wrist_to_joints(wrist), stage='pregrasp', include_object=True)
        self.stage_results['hand_pregrasp'] = {'wrist': wrist.tolist(), 'reached': self.rig.wrist().tolist()}
        return wrist

    def execute_hand_grasp(self):
        task = HandGraspEnv(objects=[], physics=self.env.physics, horizon=self.horizon, carry=True,
                            wrist_box=self.wrist_box)
        observation = task.attach(self.rig)
        if hasattr(self.policy, 'bind'):
            self.policy.bind(task)
        info, decisions = {}, 0
        for decisions in range(1, self.horizon + 1):
            observation, _, terminated, _, info = task.step(self.policy.act(observation))
            if task.phase == 'carry' or terminated:
                break
        held = task.phase == 'carry'
        self.stage_results['hand_grasp'] = {'policy': self.policy.source, 'decisions': decisions, 'held': held,
                                            'lift_m': info.get('lift'), 'contacts': info.get('contacts'),
                                            'palm_contact': info.get('palm_contact')}
        return held

    def prepare_carry(self):
        """The planner's retreat and carry posture; if the grasp ended at a tilt the tucked posture cannot
        take, level the palm back to the pregrasp's pitch and roll (keeping its yaw) and try again."""
        try:
            return super().prepare_carry()
        except ValueError:
            pass
        wrist = self.rig.wrist()
        wrist[4:] = self.pregrasp[4:]
        position, rotation = self.rig._flange_target(wrist)
        self.move_ee(position, Rotation.from_matrix(rotation).as_quat(), 'carry_posture', step_size=.006)
        self.stage_results['carry_posture_levelled'] = True
        return super().prepare_carry()

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
        except (ValueError, RuntimeError) as exc:
            result.update(status='failed', error=str(exc), failed_stage=self.stage)
        result.update(stages=self.stage_results, metrics=self.env.evaluate(),
                      wall_contact_steps=self.wall_contact_steps, robot_table_contact_steps=self.table_contact_steps,
                      first_lift_action=self.first_lift_action, maximum_object_height=self.max_object_height,
                      first_lift_physics_tick=None if self.first_lift_action is None else self.first_lift_action * self.env.substeps)
        return result
