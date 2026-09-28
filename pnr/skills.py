"""Goal-conditioned manipulation skills on physically constructed course states.

The geometric controller supplies grasp proposal, planned approach and fixed
orientation. SAC controls Cartesian refinement and coupled finger closure.
Snapshots are built by physical reference motions, never by object attachments.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .env import PickNavReachEnv
from .reference import ReferenceController
from .local_control import LocalCartesianIK


class ManipulationSkillEnv(gym.Env):
    metadata = {'render_modes':['rgb_array'],'render_fps':24}

    def __init__(self, skill='grasp', scene_seed=42, object_idx=14, seed=0,
                 horizon=100, evaluation=False, native=None, controller=None, build_starts=True):
        if skill not in ('grasp','reach'):
            raise ValueError(skill)
        self.skill, self.horizon, self.evaluation = skill, int(horizon), bool(evaluation)
        self._owns_native=native is None
        self.native = PickNavReachEnv(seed=scene_seed,object_idx=object_idx) if native is None else native
        self.controller = ReferenceController(self.native,seed=seed) if controller is None else controller
        self.local_ik=LocalCartesianIK(self.native)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.,1.,shape=(4 if skill=='grasp' else 3,),dtype=np.float32)
        self.observation_space = spaces.Dict({
            'observation':spaces.Box(-np.inf,np.inf,shape=(50,),dtype=np.float32),
            'achieved_goal':spaces.Box(-np.inf,np.inf,shape=(3,),dtype=np.float32),
            'desired_goal':spaces.Box(-np.inf,np.inf,shape=(3,),dtype=np.float32)})
        self.tolerance = .04
        self.repeat = 3
        self.delta_scale = .012
        self.snapshots = []
        self.reset_count = 0
        self.curriculum_level = 0
        self._recent_success = []
        self.setup_action_calls = 0
        if build_starts:
            self._build_starts()
            self.setup_action_calls = self.native.step_count
        else:
            self.base_xy=self.controller.base_position().copy()
            self.orientation=(self.controller.grasp.quaternion.copy() if skill=='grasp' and self.controller.grasp is not None
                              else self.controller.planner.ee_pose()[1])
            self.base_goal=(self.native.observe()['object_pos'].copy()+[0,0,.20] if skill=='grasp'
                            else self.native.goal_pos.copy())
            self.base_goal[:2]-=self.base_xy
        self._episode_steps = 0
        self._episode_success = False
        self.goal = self.base_goal.copy()

    @classmethod
    def bind(cls,native,controller,skill='grasp',horizon=200):
        return cls(skill=skill,horizon=horizon,evaluation=True,native=native,
                   controller=controller,build_starts=False)

    def _build_starts(self):
        grasp=self.controller.prepare_grasp()
        self.orientation=grasp.quaternion.copy()
        self.base_xy=self.controller.base_position().copy()
        if self.skill=='grasp':
            self.base_goal=self.native.observe()['object_pos'].copy()+[0,0,.20]
            self.base_goal[:2]-=self.base_xy
            # Difficult starts remain available for later curriculum levels.
            self.snapshots.append((.17,self.native.save_state()))
            for height in [.09,.04,.008]:
                self.controller.move_ee(grasp.position+[0,0,height],self.orientation,
                    stage='curriculum_setup',closed=False)
                self.snapshots.append((height,self.native.save_state()))
            self.snapshots.sort(key=lambda item:item[0])
        else:
            if not self.controller.execute_grasp(grasp):
                raise RuntimeError('Cannot train reach: reference grasp failed')
            self.controller.prepare_carry()
            self.controller.carry()
            self.base_xy=self.controller.base_position().copy()
            # Physically collect the actual carry-to-reach handoff; the object
            # projects ahead of the base and must move back toward the sphere.
            self.orientation=self.controller.planner.ee_pose()[1]
            obj=self.native.observe()['object_pos']
            self.base_goal=self.native.goal_pos.copy()
            self.base_goal[:2]-=self.base_xy
            self.snapshots=[(0.,self.native.save_state())]

    def _observation(self):
        obs=self.native.observe()
        ee,orn=self.controller.planner.ee_pose()
        obj=np.array(obs['object_pos']); local_ee=ee.copy()
        obj[:2]-=self.base_xy; local_ee[:2]-=self.base_xy
        velocity=np.concatenate(self.native.p.getBaseVelocity(self.native.object_id))
        contacts=np.asarray(self.native.contacts()['finger_contacts'],dtype=float)
        features=np.concatenate([obs['qpos'][2:],obs['qvel'],local_ee,orn,
            obj,obs['object_xyzw'],velocity,contacts]).astype(np.float32)
        return {'observation':features,'achieved_goal':obj.astype(np.float32),
                'desired_goal':self.goal.astype(np.float32)}

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal=np.asarray(achieved_goal)
        distance=np.linalg.norm(achieved_goal-np.asarray(desired_goal),axis=-1)
        if isinstance(info,dict):
            retained=bool(info.get('bilateral_contact',False))
        else:
            retained=np.array([bool(item.get('bilateral_contact',False)) for item in info])
        failure = bool(info.get('physical_failure',False)) if isinstance(info,dict) else np.asarray([bool(item.get('physical_failure',False)) for item in info])
        ordinary = -np.asarray((distance>=self.tolerance)|~np.asarray(retained),dtype=np.float32)
        return np.asarray(np.where(failure,-50.,ordinary),dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if not self.snapshots:
            raise RuntimeError('A bound skill cannot reset the uninterrupted task')
        super().reset(seed=seed)
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        max_level=6 if self.skill=='grasp' else 0
        if not self.evaluation and len(self._recent_success)>=10:
            if np.mean(self._recent_success[-10:])>=.7 and self.curriculum_level<max_level:
                self.curriculum_level+=1
                self._recent_success=[]
        # Evaluation tracks the fixed easy diagnostic bank in the first pilot;
        # full-pregrasp success is separately checked before complete-system claims.
        # First learn contact and increasingly high lifts; only then broaden
        # the pre-grasp distance. Genuine evaluation always requests a 20cm lift.
        level=0 if self.evaluation else max(0,self.curriculum_level-3)
        index=int(self.rng.integers(0,level+1)) if level else 0
        if self.skill=='grasp' and self.evaluation and (options or {}).get('full_pregrasp'):
            index=len(self.snapshots)-1
        self.evaluation_scope=('full_pregrasp_diagnostic' if self.skill=='grasp' and index==3
                               else 'easy_manipulation_diagnostic')
        self.native.restore_state(self.snapshots[index][1],reset_counters=True)
        self._episode_steps=0; self._episode_success=False
        self.goal=self.base_goal.copy()
        if self.skill=='grasp' and not self.evaluation:
            lift_heights=(.06,.10,.15,.20)
            self.goal[2]=self.base_goal[2]-.20+lift_heights[min(self.curriculum_level,3)]
        if self.skill=='grasp':
            self.goal[:2]+=self.rng.uniform(-.015,.015,2)
        elif not self.evaluation:
            self.goal[:2]+=self.rng.uniform(-.15,.15,2)
        self.reset_count+=1
        return self._observation(), {'curriculum_level':self.curriculum_level,
                                    'start_height':self.snapshots[index][0]}

    def step(self, action):
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        rejected=False
        if self.skill=='reach':
            target=self.native.agent_qpos.copy()
            target[:2]+=self.delta_scale*action[:2]
            torso=self.native.joint_name_to_slot['torso_lift_joint']
            target[torso]=np.clip(target[torso]+self.delta_scale*action[2],
                                  self.native.joint_lower[torso],self.native.joint_upper[torso])
            target[self.native.finger_slots]=0.
            if not self.controller.planner.collision_free(target,include_object=False,margin=-.0005):
                target=self.native.agent_qpos.copy()
                target[self.native.finger_slots]=0.
                rejected=True
        else:
            ee,_=self.controller.planner.ee_pose()
            requested=ee+self.delta_scale*action[:3]
            requested[:2]=np.clip(requested[:2],self.base_xy+[.10,-.28],self.base_xy+[.95,.28])
            requested[2]=np.clip(requested[2],self.native.table_height+.037,1.35)
            try:
                target=self.local_ik.ik(requested,self.orientation)
                if not self.controller.planner.collision_free(target,include_object=False,margin=-.0005):
                    raise ValueError('Arm target collision')
            except ValueError:
                target=self.native.agent_qpos.copy(); rejected=True
            target[self.native.finger_slots]=(action[3]+1)*.025
        before=self.native.step_count
        for _ in range(self.repeat):
            _,info=self.native.step(target)
        self._episode_steps+=1
        observation=self._observation()
        distance=float(np.linalg.norm(observation['achieved_goal']-observation['desired_goal']))
        success=bool(distance<self.tolerance and info['bilateral_contact'])
        self._episode_success |= success
        terminated=bool(self.native.observe()['object_pos'][2]<self.native.table_height-.08)
        truncated=self._episode_steps>=self.horizon
        info.update(is_success=success,episode_success=self._episode_success,goal_distance=distance,
                    action_calls_delta=self.native.step_count-before,
                    physics_ticks_delta=(self.native.step_count-before)*self.native.substeps,
                    curriculum_level=self.curriculum_level,action_rejected=rejected,physical_failure=terminated,
                    evaluation_scope='reference_carry_handoff' if self.skill=='reach' else getattr(self,'evaluation_scope','full_pregrasp_diagnostic'))
        reward=float(self.compute_reward(observation['achieved_goal'],observation['desired_goal'],info))
        if terminated or truncated:
            self._recent_success.append(float(self._episode_success))
        return observation,reward,terminated,truncated,info

    def get_training_state(self):
        return {'curriculum_protocol':self.curriculum_protocol,'curriculum_level':self.curriculum_level,'recent_success':self._recent_success,
                'reset_count':self.reset_count,'rng_state':self.rng.bit_generator.state}

    @property
    def curriculum_protocol(self):
        return 'reach-handoff-v2' if self.skill=='reach' else 'lift-then-pregrasp-v2'

    def set_training_state(self,state):
        if state.get('curriculum_protocol')!=self.curriculum_protocol:
            raise ValueError('Resume requires matching manipulation curriculum protocol')
        self.curriculum_level=int(state['curriculum_level'])
        self._recent_success=list(state['recent_success'])
        self.reset_count=int(state['reset_count'])
        self.rng.bit_generator.state=state['rng_state']

    def render(self):
        return self.native.render_rgb(camera='grasp')

    def close(self):
        if self._owns_native:
            self.native.close()


def make_skill_env(skill,**kwargs):
    if skill=='carry':
        from .carry import CarrySkillEnv
        return CarrySkillEnv(**kwargs)
    return ManipulationSkillEnv(skill=skill,**kwargs)
