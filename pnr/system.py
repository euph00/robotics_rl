"""Uninterrupted inference for the three learned SAC+HER skills.

Only pre-grasp approach and carry-posture transitions use the shared geometric
controller. The learned grasp starts at the full 17 cm pre-grasp, carrying gets
one fixed exit goal, and reaching gets the actual supplied object goal. There
are no inter-skill resets, teacher descents, waypoint goals, or object welds.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Callable, Mapping

import numpy as np

from .carry import CarrySkillEnv
from .reference import ReferenceController
from .skills import ManipulationSkillEnv


class StageFailure(RuntimeError):
    pass


class LearnedSystem:
    """Execute loaded predictors on one existing environment at its original start.

    Each policy implements SB3's ``predict(observation, deterministic=True)``.
    ``on_step(system, observation, info)`` observes *every native call*, including
    repeated skill controls and all geometric assistance. It must not step/reset
    the simulator. The caller owns the native environment and loaded policies.
    """
    def __init__(self, native, policies: Mapping, *, grasp_horizon=600,
                 carry_horizon=1500, reach_horizon=150, on_step: Callable | None=None,
                 skill_factories: Mapping | None=None, controller_speed_scale=1.,
                 reach_tolerance_m=.10, grasp_proposal_mode='legacy'):
        if set(policies) != {'grasp','carry','reach'}:
            raise ValueError('Exactly grasp, carry, and reach policies are required')
        if any(not hasattr(policy,'predict') for policy in policies.values()):
            raise TypeError('Each skill policy must provide predict()')
        self.native = native
        self.policies = dict(policies)
        self.skill_factories = dict(skill_factories or {})
        if set(self.skill_factories)-set(policies) or any(not callable(factory) for factory in self.skill_factories.values()):
            raise ValueError('Skill factories must be callable entries for grasp, carry, or reach')
        if not np.isfinite(controller_speed_scale) or controller_speed_scale <= 0:
            raise ValueError('Planned controller speed scale must be positive and finite')
        if reach_tolerance_m not in (.04, .08, .10):
            raise ValueError('Reach tolerance must be native 0.10 m, internal margin 0.08 m, or legacy 0.04 m')
        self.reach_tolerance_m = float(reach_tolerance_m)
        self.controller = ReferenceController(native,seed=native.seed,speed_scale=controller_speed_scale,
                                              grasp_proposal_mode=grasp_proposal_mode)
        self.horizons = {'grasp':int(grasp_horizon),'carry':int(carry_horizon),'reach':int(reach_horizon)}
        if any(value < 1 for value in self.horizons.values()):
            raise ValueError('Skill horizons must be positive')
        self.on_step = on_step
        self.stage = 'initial'
        self.control_source = 'planned'
        self.stage_results = {}
        self.policy_decisions = {name:0 for name in self.policies}
        self.calls_by_source = {'planned':0,'learned':0,'hold':0}
        self.wall_contact_calls = 0
        self.robot_table_contact_calls = 0
        self.first_lift_action = None
        self.first_grasped_lift_action = None
        self.first_upstream_reach_action = None
        self.first_complete_reach_action = None
        self.navigation_start_action = None
        self._active_stage_start = None
        self.max_contact_force = 0.
        self.inference_seconds = 0.
        self._running = False

    @property
    def display_stage(self):
        return self.controller.stage if self.control_source=='planned' else self.stage

    @contextmanager
    def _observe_native_steps(self):
        # An instance-local observer wraps the existing operation. No simulator
        # API, return value, action, timing/substep count, or motor target changes.
        original_step = self.native.step
        def observed(action):
            observation,info = original_step(action)
            self.calls_by_source[self.control_source]+=1
            p=self.native.p
            wall_contacts=(p.getContactPoints(self.native.robot_id,self.native.maze_id)
                           +p.getContactPoints(self.native.object_id,self.native.maze_id))
            self.wall_contact_calls+=int(any(c[9]>0 for c in wall_contacts))
            self.robot_table_contact_calls+=int(any(c[9]>0 for c in
                p.getContactPoints(self.native.robot_id,self.native.table_id)))
            self.max_contact_force=max(self.max_contact_force,float(info.get('normal_force',0.)))
            if info.get('lifted') and self.first_lift_action is None:
                self.first_lift_action=self.native.step_count
            if info.get('lifted') and info.get('bilateral_contact') and self.first_grasped_lift_action is None:
                self.first_grasped_lift_action=self.native.step_count
            if info.get('success') and self.first_upstream_reach_action is None:
                self.first_upstream_reach_action=self.native.step_count
            if info.get('complete_task_success') and self.first_complete_reach_action is None:
                self.first_complete_reach_action=self.native.step_count
            if self.on_step:
                self.on_step(self,observation,info)
            return observation,info
        self.native.step=observed
        try:
            yield
        finally:
            self.native.step=original_step

    def _begin(self,stage,source):
        self.stage,self.control_source=stage,source
        self._active_stage_start=(self.native.step_count,time.monotonic())
        if stage=='carry':
            self.navigation_start_action=self.native.step_count
        return self._active_stage_start

    def _finish(self,stage,start,success,**details):
        calls=self.native.step_count-start[0]
        self.stage_results[stage]={'success':bool(success),'source_action_calls':calls,
            'physics_ticks':calls*self.native.substeps,'wall_seconds':time.monotonic()-start[1],
            **details}
        self._active_stage_start=None

    def _skill(self,name):
        start=self._begin(name,'learned')
        if name in self.skill_factories:
            wrapper=self.skill_factories[name](self.native,self.controller,horizon=self.horizons[name])
        elif name=='carry':
            wrapper=CarrySkillEnv.bind(self.native,self.controller,horizon=self.horizons[name])
        else:
            wrapper=ManipulationSkillEnv.bind(self.native,self.controller,skill=name,horizon=self.horizons[name])
        observation=wrapper._observation()
        initial_goal=observation['desired_goal'].copy()
        initial_achieved=observation['achieved_goal'].copy()
        goal_frame='world_xy_divided_by_20' if name=='carry' else 'fixed_bind_base_xy_and_world_z'
        last_info={}
        reason='decision_horizon'
        reached=False
        rejected=0
        for decision in range(self.horizons[name]):
            before=time.monotonic()
            prediction=self.policies[name].predict(observation,deterministic=True)
            self.inference_seconds+=time.monotonic()-before
            action=prediction[0] if isinstance(prediction,tuple) else prediction
            action=np.asarray(action,dtype=np.float32)
            if action.shape != wrapper.action_space.shape or not np.isfinite(action).all():
                raise StageFailure(f'{name}: policy returned invalid action')
            observation,_,terminated,truncated,last_info=wrapper.step(action)
            self.policy_decisions[name]+=1
            rejected+=int(last_info.get('action_rejected',False))
            # A historical episode_success is insufficient for a stage handover.
            goal_reached=last_info.get('is_success',False)
            if name=='reach' and self.reach_tolerance_m in (.08, .10):
                # Training still uses its 4 cm goal. Full-task handover follows
                # the native contract, including pickup/retention history.
                # Optional 8 cm handover leaves room for drift during the final
                # hold; run() still requires the unchanged native 10 cm result.
                goal_reached=bool(last_info.get('complete_task_success',False)
                                  and last_info.get('dist_to_goal',np.inf)<self.reach_tolerance_m)
            reached=bool(goal_reached and last_info.get('bilateral_contact',False)
                         and not terminated and self.native.drop_events==0)
            if reached:
                reason='current_goal_and_retention'
                break
            if terminated:
                reason=last_info.get('failure_reason') or 'physical_failure'
                break
            if self.native.drop_events:
                reason='dropped_object'
                break
            if truncated:
                break
        self._finish(name,start,reached,policy_decisions=self.policy_decisions[name],
                     action_rejections=rejected,reason=reason,goal_frame=goal_frame,
                     initial_achieved_goal=initial_achieved.tolist(),desired_goal=initial_goal.tolist(),
                     final_achieved_goal=observation['achieved_goal'].tolist(),
                     goal_distance=last_info.get('goal_distance'),
                     bilateral_contact=last_info.get('bilateral_contact',False),
                     **({'reach_tolerance_m':self.reach_tolerance_m,
                         'reach_acceptance':('native_complete_task_success' if self.reach_tolerance_m==.10
                                             else 'native_complete_task_success_with_8cm_margin'
                                             if self.reach_tolerance_m==.08 else 'legacy_skill_is_success'),
                         'native_goal_distance':last_info.get('dist_to_goal')} if name=='reach' else {}))
        if not reached:
            raise StageFailure(f'{name}: {reason}')

    def _held_pause(self,stage,steps):
        start=self._begin(stage,'hold')
        target=self.native.agent_qpos.copy()
        target[self.native.finger_slots]=0.
        positions=[]
        for _ in range(steps):
            observation,_=self.controller.tick(target,'hold')
            positions.append(observation['object_pos'].copy())
        positions=np.asarray(positions)
        height_range=float(np.ptp(positions[:,2]))
        motion_extent=float(np.linalg.norm(np.ptp(positions,axis=0)))
        info=self.native.evaluate()
        retained=bool(info['bilateral_contact'] and info['stable_hold_steps']>=steps
                      and info['drop_events']==0 and info['lifted'] and height_range<.02)
        self._finish(stage,start,retained,stable_hold_steps=info['stable_hold_steps'],
            height_range_m=height_range,object_motion_extent_m=motion_extent,
            operational_stability='bilateral contact and >=10cm lift throughout; height range <2cm')
        if not retained:
            raise StageFailure(f'{stage}: stable physical hold was not sustained')

    def run(self):
        if self._running or self.native.step_count != 0:
            raise ValueError('Full-system evaluation requires a fresh original-start environment')
        initial=self.native.observe()
        if (np.linalg.norm(initial['object_pos']-self.native.initial_object_pos)>.005
                or np.max(np.abs(initial['qpos'][:3]-self.native.init_qpos[:3]))>.005
                or np.max(np.abs(initial['qpos']-self.native.init_qpos))>.05
                or np.max(np.abs(self.native.last_target-self.native.init_qpos))>1e-5
                or self.native.ever_lifted or self.native.drop_events):
            raise ValueError('Full-system evaluation requires the original physical start, not a reset curriculum snapshot')
        self._running=True
        started=time.monotonic()
        result={'kind':'learned_system','scope':'full_task','success':False,'status':'running',
                'scene_seed':self.native.seed,'object_idx':self.native.object_idx,
                'object_name':self.native.object_name,'scene_manifest':self.native.scene_manifest(),
                'reach_tolerance_m':self.reach_tolerance_m,'final_hold_source_calls':60,
                'assistance':{'planned_pregrasp':True,'planned_carry_posture':True,
                              'carry_waypoints_during_policy':False,'inter_skill_resets':False,
                              'object_attachment':False}}
        try:
            with self._observe_native_steps():
                start=self._begin('planned_approach','planned')
                grasp=self.controller.prepare_grasp()
                self._finish('planned_approach',start,True,grasp=grasp.as_dict())
                self._skill('grasp')
                self._held_pause('pick_hold',50)
                ee,orientation=self.controller.planner.ee_pose()
                result['pick_pose_error']={
                    'position_m':float(np.linalg.norm(ee-(grasp.position+[0,0,.20]))),
                    'orientation_rad':float(2*np.arccos(np.clip(abs(np.dot(orientation,grasp.quaternion)),0,1)))}
                start=self._begin('planned_carry_posture','planned')
                self.controller.prepare_carry()
                contact=self.native.contacts()['bilateral_contact']
                self._finish('planned_carry_posture',start,contact)
                if not contact:
                    raise StageFailure('planned_carry_posture: lost grasp')
                self._skill('carry')
                self._skill('reach')
                self._held_pause('final_hold',60)
                final=self.native.evaluate()
                result.update(success=bool(final['complete_task_success']),status='complete')
                if not result['success']:
                    result.update(status='failed',error='Final strict physical success check failed')
        except (StageFailure,ValueError,RuntimeError) as exc:
            if self._active_stage_start is not None:
                self._finish(self.stage,self._active_stage_start,False,error=str(exc))
            result.update(status='failed',error=str(exc),failed_stage=self.stage)
        finally:
            self._running=False
        metrics=self.native.evaluate()
        result.update(metrics=metrics,stages=self.stage_results,
            policy_decisions=dict(self.policy_decisions),source_calls_by_controller=dict(self.calls_by_source),
            wall_contact_calls=self.wall_contact_calls,robot_table_contact_calls=self.robot_table_contact_calls,
            first_lift_action=self.first_lift_action,
            first_grasped_lift_action=self.first_grasped_lift_action,
            first_upstream_reach_action=self.first_upstream_reach_action,
            first_complete_reach_action=self.first_complete_reach_action,
            first_complete_reach_physics_tick=None if self.first_complete_reach_action is None else self.first_complete_reach_action*self.native.substeps,
            navigation_start_action=self.navigation_start_action,
            navigation_reach_source_calls=(None if self.first_complete_reach_action is None or self.navigation_start_action is None
                                           else self.first_complete_reach_action-self.navigation_start_action),
            first_lift_physics_tick=None if self.first_lift_action is None else self.first_lift_action*self.native.substeps,
            maximum_contact_force=self.max_contact_force,inference_seconds=self.inference_seconds,
            wall_seconds=time.monotonic()-started,source_action_calls=self.native.step_count,
            physics_ticks=self.native.physics_steps,
            lift_by_3000_source_calls=bool(self.first_lift_action is not None and self.first_lift_action<=3000),
            lift_by_3000_physics_ticks=bool(self.first_lift_action is not None and self.first_lift_action*self.native.substeps<=3000),
            reached_by_15000_source_calls=bool(result['success'] and self.first_complete_reach_action is not None
                                              and self.first_complete_reach_action<=15000),
            reached_by_15000_physics_ticks=bool(result['success'] and self.first_complete_reach_action is not None
                                               and self.first_complete_reach_action*self.native.substeps<=15000))
        return result
