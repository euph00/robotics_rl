"""Lazy physical scene rotation and a schema-only environment for initialization."""
from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pnr.general_carry import GeneralCarryEnv


class CarrySchema(gym.Env):
    """Defines spaces/reward only; never supplies transitions to training."""
    def __init__(self, gamma=.995, shaping_weight=2.):
        self.action_space=spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.observation_space=spaces.Dict({'observation':spaces.Box(-np.inf,np.inf,shape=(580,),dtype=np.float32),
            'achieved_goal':spaces.Box(-np.inf,np.inf,shape=(2,),dtype=np.float32),
            'desired_goal':spaces.Box(-np.inf,np.inf,shape=(2,),dtype=np.float32)})
        self.reward_proxy=GeneralCarryEnv.__new__(GeneralCarryEnv)
        self.reward_proxy.gamma=gamma;self.reward_proxy.shaping_weight=shaping_weight
        self.reward_proxy.failure_cost=1./(1.-gamma)
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_proxy.compute_reward(achieved_goal,desired_goal,info)
    def reset(self,**kwargs):
        raise RuntimeError('Schema-only environment must not generate learning experience')
    def step(self,action):
        raise RuntimeError('Schema-only environment must not generate learning experience')


class MultiSceneCarry(CarrySchema):
    def __init__(self,layouts,seed=0,cache_size=4,episodes_per_scene=4,horizon=900,
                 delta_scale=.075,gamma=.995,shaping_weight=2.,evaluation=False):
        super().__init__(gamma,shaping_weight)
        if not layouts:raise ValueError('At least one declared scene is required')
        self.layouts=list(layouts);self.seed=seed;self.rng=np.random.default_rng(seed)
        self.cache=OrderedDict();self.cache_size=cache_size;self.episodes_per_scene=episodes_per_scene
        self.horizon=horizon;self.delta_scale=delta_scale;self.gamma=gamma;self.shaping_weight=shaping_weight
        self.evaluation=evaluation;self.reset_count=0;self.curriculum_level=0;self.active=None
        self.setup_action_calls=self.setup_physics_ticks=self.initialization_physics_ticks=0
        self.failed_setup_attempts=[]
    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        index=(self.reset_count//self.episodes_per_scene)%len(self.layouts)
        layout=self.layouts[index];key=int(layout['maze_seed'])
        if key not in self.cache:
            if len(self.cache)>=self.cache_size:
                _,old=self.cache.popitem(last=False);old.close()
            try:
                env=GeneralCarryEnv(maze_seed=key,scene_seed=42,object_idx=14,seed=self.seed+index,
                    horizon=self.horizon,delta_scale=self.delta_scale,gamma=self.gamma,
                    shaping_weight=self.shaping_weight,evaluation=self.evaluation,build_bank=not self.evaluation)
            except Exception as exc:
                costs=getattr(exc,'setup_accounting',{})
                self.setup_action_calls+=costs.get('source_action_calls',0)
                self.setup_physics_ticks+=costs.get('physics_ticks',0)
                self.initialization_physics_ticks+=costs.get('initialization_physics_ticks',0)
                self.failed_setup_attempts.append({'maze_seed':key,'error':repr(exc),'accounting':costs})
                raise
            self.setup_action_calls+=env.setup_action_calls
            self.setup_physics_ticks+=env.setup_physics_ticks
            self.initialization_physics_ticks+=env.initialization_physics_ticks
            self.cache[key]=env
        self.active=self.cache[key];self.cache.move_to_end(key)
        self.active.set_curriculum_level(self.curriculum_level)
        self.reset_count+=1
        return self.active.reset(options=options)
    def step(self,action):
        if self.active is None:raise RuntimeError('Reset before stepping')
        return self.active.step(action)
    def set_curriculum_level(self,level):self.curriculum_level=int(level)
    def get_training_state(self):
        return {'protocol':'scene-rotation-v1','reset_count':self.reset_count,'curriculum_level':self.curriculum_level,
                'rng_state':self.rng.bit_generator.state,'failed_setup_attempts':self.failed_setup_attempts}
    def set_training_state(self,state):
        if state['protocol']!='scene-rotation-v1':raise ValueError('Wrong scene rotation protocol')
        self.reset_count=state['reset_count'];self.curriculum_level=state['curriculum_level']
        self.rng.bit_generator.state=state['rng_state']
    def prime_cache(self):
        """Prepare physical scenes concurrently across workers without episode resets.

        Only setup timing/cache capacity changes. Episode-selection counters and
        scene RNGs remain untouched; each setup cost is still recorded once.
        """
        self.cache_size=max(self.cache_size,len(self.layouts))
        for index,layout in enumerate(self.layouts):
            key=int(layout['maze_seed'])
            if key in self.cache:
                continue
            try:
                env=GeneralCarryEnv(maze_seed=key,scene_seed=42,object_idx=14,seed=self.seed+index,
                    horizon=self.horizon,delta_scale=self.delta_scale,gamma=self.gamma,
                    shaping_weight=self.shaping_weight,evaluation=self.evaluation,build_bank=not self.evaluation)
            except Exception as exc:
                costs=getattr(exc,'setup_accounting',{})
                self.setup_action_calls+=costs.get('source_action_calls',0)
                self.setup_physics_ticks+=costs.get('physics_ticks',0)
                self.initialization_physics_ticks+=costs.get('initialization_physics_ticks',0)
                self.failed_setup_attempts.append({'maze_seed':key,'error':repr(exc),'accounting':costs})
                raise
            self.setup_action_calls+=env.setup_action_calls
            self.setup_physics_ticks+=env.setup_physics_ticks
            self.initialization_physics_ticks+=env.initialization_physics_ticks
            self.cache[key]=env
        return {'scenes':len(self.cache),'setup_source_action_calls':self.setup_action_calls,
                'setup_physics_ticks':self.setup_physics_ticks}
    def close(self):
        for env in self.cache.values():env.close()
        self.cache.clear()
