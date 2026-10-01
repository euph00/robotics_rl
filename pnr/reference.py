"""Physical geometric reference and shared, explicitly declared assistance.

No constraints attach the object to the robot. Every execution motion uses the
native 15-position-target interface, including the original table approach.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial.transform import Rotation

from .planning import JointPlanner, antipodal_grasps, maze_astar


class ReferenceController:
    HOLDING_STAGES = ('lift', 'retreat', 'carry_posture', 'carry', 'reach')

    def __init__(self, env, on_step=None, seed=0, speed_scale=1., grasp_proposal_mode='legacy',
                 holding_speed_scale=None):
        if grasp_proposal_mode not in ('legacy', 'clearance-v2'):
            raise ValueError('Unknown grasp proposal mode')
        self.grasp_proposal_mode = grasp_proposal_mode
        self._clearance_carry_target = None
        self.env = env
        self.planner = JointPlanner(env,seed)
        self.on_step = on_step
        self.stage = 'initial'
        self.speed_scale = float(speed_scale)
        # Planned motions while holding the object can run slower than empty-hand motions.
        self.holding_speed_scale = float(speed_scale if holding_speed_scale is None else holding_speed_scale)
        self.wall_contact_steps = 0
        self.table_contact_steps = 0
        self.first_lift_action = None
        self.trace = []
        self.grasp = None
        self.path = None
        self.stage_results = {}
        self.finger_slots = list(env.finger_slots)
        self.started = time.monotonic()
        self.max_object_height = float(env.observe()['object_pos'][2])

    def tick(self, target, stage=None):
        if stage is not None:
            self.stage = stage
        obs, info = self.env.step(target)
        self.max_object_height = max(self.max_object_height,float(obs['object_pos'][2]))
        if info.get('lifted') and self.first_lift_action is None:
            self.first_lift_action = self.env.step_count
        p=self.env.p
        wall_points=(p.getContactPoints(self.env.robot_id,self.env.maze_id)
                     +p.getContactPoints(self.env.object_id,self.env.maze_id))
        self.wall_contact_steps += int(any(c[9]>0 for c in wall_points))
        self.table_contact_steps += int(any(c[9]>0 for c in p.getContactPoints(self.env.robot_id,self.env.table_id)))
        if self.on_step:
            self.on_step(self,obs,info)
        return obs,info

    def hold(self, steps=50, stage=None):
        target = self.env.agent_qpos.copy()
        if (stage or self.stage) in ('lift','hold','carry','reach','retreat','carry_posture'):
            target[self.finger_slots] = 0
        result = None
        for _ in range(steps):
            result = self.tick(target,stage)
        return result

    def move_joint(self,target,stage=None,step_size=.018,check=True,include_object=False):
        target = np.asarray(target,dtype=float)
        path = self.planner.plan(target,include_object) if check else [self.env.agent_qpos.copy(),target]
        for a,b in zip(path[:-1],path[1:]):
            scale = self.holding_speed_scale if stage in self.HOLDING_STAGES else self.speed_scale
            n = max(2,int(np.ceil(np.max(np.abs(b-a))/(step_size*scale))))
            for t in np.linspace(0,1,n+1)[1:]:
                command = a+(b-a)*t
                if stage in ('lift','hold','carry','reach','retreat','carry_posture'):
                    command[self.finger_slots] = 0.
                self.tick(command,stage)
        for _ in range(8):
            self.tick(target,stage)
        return self.env.observe()

    def move_ee(self,position,quaternion,stage=None,closed=None,check=True,step_size=.018):
        q = self.planner.ik(position,quaternion)
        if closed is not None:
            q[self.finger_slots] = 0. if closed else .05
        return self.move_joint(q,stage=stage,step_size=step_size,check=check)

    def base_position(self):
        return self.env.root_position[:2]+self.env.agent_qpos[:2]

    def move_base(self,xy,stage='approach',step_size=.009,check=True):
        target = self.env.agent_qpos.copy()
        target[:2] = np.asarray(xy)-self.env.root_position[:2]
        if stage in ('carry','reach','retreat'):
            target[self.finger_slots] = 0.
        return self.move_joint(target,stage=stage,step_size=step_size,check=check)

    def pregrasp_position(self, distance, grasp=None):
        grasp = self.grasp if grasp is None else grasp
        if grasp is None:
            raise ValueError('Select a grasp before computing its approach position')
        if self.grasp_proposal_mode == 'legacy':
            return grasp.position + [0,0,float(distance)]
        from .clearance_grasps import approach_position
        return approach_position(grasp, distance)

    def prepare_grasp(self):
        """Original start -> folded arm -> table approach -> selected pregrasp."""
        self._clearance_carry_target = None
        topdown = Rotation.from_matrix(np.array([[0,0,1],[0,1,0],[-1,0,0]])).as_quat()
        base = self.base_position()
        self.move_ee([base[0]+.35,base[1],1.1],topdown,'fold',closed=False)
        self.move_base([-.78,0.],stage='approach')
        obs = self.env.observe()
        candidates = antipodal_grasps(obs['object_pc'],obs['object_normals'],table_height=self.env.table_height)
        if self.grasp_proposal_mode == 'clearance-v2':
            from .clearance_grasps import prepare_clearance_grasp
            return prepare_clearance_grasp(self, candidates)
        errors=[]
        for grasp in candidates:
            try:
                pre = grasp.position + [0,0,.17]
                q = self.planner.ik(pre,grasp.quaternion)
                q[self.finger_slots] = .05
                if not self.planner.collision_free(q,include_object=True):
                    continue
                q_grasp = self.planner.ik(grasp.position,grasp.quaternion,seed=q)
                q_grasp[self.finger_slots] = .05
                if not self.planner.collision_free(q_grasp):
                    continue
                self.move_joint(q,stage='pregrasp',include_object=True)
                self.grasp = grasp
                self.stage_results['grasp_generation'] = {'candidates':len(candidates),'selected':grasp.as_dict()}
                return grasp
            except ValueError as exc:
                errors.append(str(exc))
        raise RuntimeError(f'No feasible antipodal proposal ({len(candidates)} pairs): {errors[:3]}')

    def execute_grasp(self,grasp=None):
        grasp = self.grasp if grasp is None else grasp
        if grasp is None:
            grasp = self.prepare_grasp()
        # Cartesian descent avoids joint-space arcs into the table.
        start,_ = self.planner.ee_pose()
        for t in np.linspace(0,1,18)[1:]:
            self.move_ee(start+(grasp.position-start)*t,grasp.quaternion,'descend',closed=False,
                         step_size=.018)
        q = self.env.agent_qpos.copy()
        q[self.finger_slots] = 0
        for _ in range(80):
            self.tick(q,'close')
        prelift = self.env.observe()['object_pos'].copy()
        ee,orn = self.planner.ee_pose()
        for t in np.linspace(0,1,21)[1:]:
            self.move_ee(ee+[0,0,.20*t],orn,'lift',closed=True,step_size=.015)
        held = []
        for _ in range(60):
            obs,info = self.tick(self._closed_target(),'hold')
            held.append(float(obs['object_pos'][2]))
        result = {'prelift_position':prelift.tolist(), 'object_position':obs['object_pos'].tolist(),
                  'lift_delta':float(obs['object_pos'][2]-prelift[2]),
                  'height_range_last_60':float(np.ptp(held)),
                  'stable_hold_steps':self.env.stable_hold_steps, 'contacts':self.env.contacts(),
                  'success':bool(obs['object_pos'][2] > prelift[2]+.10 and np.ptp(held)<.02
                                 and self.env.contacts()['bilateral_contact']
                                 and self.env.ever_stably_held)}
        self.stage_results['pick'] = result
        return result['success']

    def _closed_target(self):
        q = self.env.agent_qpos.copy()
        q[self.finger_slots] = 0
        return q

    def prepare_carry(self):
        """Retreat from table before folding the physically held object inward."""
        self.planner.set_payload()
        self.move_base([-1.25,0],stage='retreat',step_size=.006)
        ee,orn = self.planner.ee_pose()
        base = self.base_position()
        if self.grasp_proposal_mode == 'clearance-v2' and self._clearance_carry_target is not None:
            target = self._clearance_carry_target.copy()
            target[self.finger_slots] = 0.
            self.move_joint(target, 'carry_posture', step_size=.012)
        else:
            self.move_ee([base[0]+.24,base[1],1.05],orn,'carry_posture',closed=True,step_size=.012)
        self.hold(50,'hold')
        self.planner.set_payload()
        self.stage_results['carry_posture'] = {'base_position':self.base_position().tolist(),
                                             'ee_position':self.planner.ee_pose()[0].tolist(),
                                             'object_position':self.env.observe()['object_pos'].tolist()}
        return self.env.observe()

    def carry(self,goal_xy=None):
        if goal_xy is None:
            goal_xy = self.env.goal_pos[:2].copy()
        table_bounds = self.env.p.getAABB(self.env.table_id)
        self.path = maze_astar(self.env.cube_positions,self.base_position(),goal_xy,
                               footprint_radius=.36,table_bounds=table_bounds)
        self.stage_results['navigation'] = {'path':[v.tolist() for v in self.path]}
        for point in self.path[1:]:
            self.move_base(point,'carry',step_size=.009)
            obs=self.env.observe()
            self.planner.set_payload()
            if obs['object_pos'][2] < self.env.table_height+.06:
                raise RuntimeError('Object dropped during maze transport')
        return self.env.observe()

    def reach(self):
        # Keep the verified carry orientation. The supplied robot's planar base
        # and torso provide three continuous translations without an IK branch
        # change in the tightly folded arm at the maze exit.
        torso = self.env.joint_name_to_slot['torso_lift_joint']
        for _ in range(4):
            obs = self.env.observe()
            error = self.env.goal_pos-obs['object_pos']
            if np.linalg.norm(error)<.025:
                break
            target = self._closed_target()
            target[:2] += error[:2]
            # Clamp to the torso range; any remaining height error counts against
            # the native 10 cm tolerance instead of aborting the reach.
            target[torso] = np.clip(target[torso]+error[2], self.env.joint_lower[torso], self.env.joint_upper[torso])
            self.move_joint(target,'reach',step_size=.006)
        self.hold(60,'reach')
        return self.env.evaluate()

    def run(self,stop_after='reach'):
        result={'status':'running','success':False}
        try:
            self.prepare_grasp()
            if stop_after=='pregrasp':
                result['status']='pregrasp_ready'
            else:
                if not self.execute_grasp():
                    raise RuntimeError('Physical grasp/lift/hold validation failed')
                if stop_after=='pick':
                    result['status']='pick_complete'
                else:
                    self.prepare_carry()
                    if stop_after=='carry_posture':
                        result['status']='carry_posture_ready'
                    else:
                        self.carry()
                        info=self.reach()
                        result.update(status='complete',success=bool(info['complete_task_success']),
                                      upstream_success=bool(info['success']))
        except (ValueError,RuntimeError) as exc:
            result.update(status='failed',error=str(exc),failed_stage=self.stage)
        result.update(stages=self.stage_results,metrics=self.env.evaluate(),
                      wall_seconds=time.monotonic()-self.started,
                      collision_checks=self.planner.collision_checks,rrt_calls=self.planner.rrt_calls,
                      maximum_object_height=self.max_object_height,
                      wall_contact_steps=self.wall_contact_steps,robot_table_contact_steps=self.table_contact_steps,
                      first_lift_action=self.first_lift_action,
                      first_lift_physics_tick=None if self.first_lift_action is None else self.first_lift_action*self.env.substeps,
                      speed_scale=self.speed_scale)
        return result
