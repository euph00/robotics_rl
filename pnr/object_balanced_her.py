"""Opt-in object-balanced original-transition sampling for SB3 HER2.7.0.

Only the distribution of valid original transition indices changes. Native HER
still chooses future goals from the same real episode, recomputes rewards with
copied info, and handles true terminals/timeouts. No episode/transition copies,
new observations, fabricated failures, or physical cost changes are introduced.
The ordinary HerReplayBuffer remains the project's default elsewhere.
"""
from __future__ import annotations

from numbers import Integral
from typing import Optional, Sequence

import numpy as np
import stable_baselines3
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import torch as th

SUPPORTED_SB3_VERSION='2.7.0'
SAMPLING_PROTOCOL='uniform-declared-object-her-v1'
OBJECT_INFO_KEY='object_idx'


def _check_sb3_version():
    if stable_baselines3.__version__!=SUPPORTED_SB3_VERSION:
        raise RuntimeError('ObjectBalancedHerReplayBuffer supports stable-baselines3==2.7.0; '
                           'audit the installed HER sample contract before upgrading')


class ObjectBalancedHerReplayBuffer(HerReplayBuffer):
    """Sample an object uniformly, then a completed transition of that object.

    ``object_indices`` is a required, fixed TRAIN declaration. Every real add()
    info must supply integer ``object_idx`` from the source episode/online case.
    Object labels are replay metadata only; policy observations are unchanged.
    Missing objects raise an explicit error, never silently change the declared
    distribution. Both SAC critic and actor replay states use this distribution.

    Compatibility is deliberately pinned to installed SB3 2.7.0. sample() mirrors
    that version's split/concatenation while delegating _get_real_samples(),
    _get_virtual_samples(), _sample_goals(), and episode/ring handling unchanged.
    """
    def __init__(self,*args,object_indices: Sequence[int],**kwargs):
        _check_sb3_version()
        objects=tuple(object_indices)
        if (not objects or any(isinstance(value,bool) or not isinstance(value,Integral)
                               or not 0<=int(value)<15 for value in objects)
                or len(set(objects))!=len(objects)):
            raise ValueError('Declare distinct integer original-object indices in [0,14]')
        if kwargs.get('copy_info_dict',False) is not True:
            raise ValueError('Object-balanced physical HER requires copy_info_dict=True')
        self.object_indices=tuple(int(value) for value in objects)
        self.sampling_protocol=SAMPLING_PROTOCOL
        self.compatible_sb3_version=SUPPORTED_SB3_VERSION
        super().__init__(*args,**kwargs)
        self.object_labels=np.full((self.buffer_size,self.n_envs),-1,dtype=np.int16)
        self.object_draw_counts=np.zeros(len(self.object_indices),dtype=np.int64)

    def __setstate__(self,state):
        _check_sb3_version()
        super().__setstate__(state)
        if (getattr(self,'sampling_protocol',None)!=SAMPLING_PROTOCOL
                or getattr(self,'compatible_sb3_version',None)!=SUPPORTED_SB3_VERSION
                or not hasattr(self,'object_labels')
                or self.object_labels.shape!=(self.buffer_size,self.n_envs)
                or not hasattr(self,'object_draw_counts')
                or self.object_draw_counts.shape!=(len(self.object_indices),)):
            raise ValueError('Saved replay lacks the declared object-balanced metadata')
        valid=self.ep_length>0
        if np.any(valid & ~np.isin(self.object_labels,self.object_indices)):
            raise ValueError('Saved replay has an unlabelled or undeclared eligible transition')

    def add(self,obs,next_obs,action,reward,done,infos):
        # Validate before native add mutates any episode/ring indices.
        if len(infos)!=self.n_envs:
            raise ValueError('Every real worker transition requires an object label')
        labels=[]
        for env_idx,info in enumerate(infos):
            label=info.get(OBJECT_INFO_KEY)
            if isinstance(label,bool) or not isinstance(label,Integral) or int(label) not in self.object_indices:
                raise ValueError(f'info[{OBJECT_INFO_KEY!r}] must be a declared integer object index')
            label=int(label)
            start=self._current_ep_start[env_idx]
            if self.pos!=start:
                previous=int(self.object_labels[(self.pos-1)%self.buffer_size,env_idx])
                if previous!=label:
                    raise ValueError('Object label changed inside one physical HER episode')
            labels.append(label)
        position=self.pos
        super().add(obs,next_obs,action,reward,done,infos)
        self.object_labels[position]=labels

    def _eligible_object_pools(self):
        is_valid=self.ep_length>0
        valid_indices=np.flatnonzero(is_valid)
        labels=self.object_labels.reshape(-1)[valid_indices]
        if np.any(~np.isin(labels,self.object_indices)):
            raise RuntimeError('Eligible HER transition lacks declared object provenance')
        return is_valid,[valid_indices[labels==obj] for obj in self.object_indices]

    def _sample_transition_indices(self,batch_size):
        is_valid,pools=self._eligible_object_pools()
        missing=[obj for obj,pool in zip(self.object_indices,pools) if not len(pool)]
        if missing:
            raise RuntimeError('Cannot sample uniform declared objects: no completed eligible '
                               f'transitions for object(s) {missing}')
        # Each draw has probability1/K for the object, then1/N_object within it.
        chosen_objects=np.random.randint(len(self.object_indices),size=batch_size)
        sampled=np.empty(batch_size,dtype=np.int64)
        counts=np.bincount(chosen_objects,minlength=len(self.object_indices))
        for index,(pool,count) in enumerate(zip(pools,counts)):
            if count:
                sampled[chosen_objects==index]=np.random.choice(pool,size=int(count),replace=True)
        self.object_draw_counts+=counts
        return np.unravel_index(sampled,is_valid.shape)

    def sample(self,batch_size: int,env: Optional[VecNormalize]=None)->DictReplayBufferSamples:
        if isinstance(batch_size,bool) or not isinstance(batch_size,Integral) or batch_size<1:
            raise ValueError('Positive integer batch size required')
        batch_indices,env_indices=self._sample_transition_indices(int(batch_size))
        # Unchanged SB3 2.7.0 HER split and data construction from here onward.
        nb_virtual=int(self.her_ratio*batch_size)
        virtual_batch_indices,real_batch_indices=np.split(batch_indices,[nb_virtual])
        virtual_env_indices,real_env_indices=np.split(env_indices,[nb_virtual])
        real_data=self._get_real_samples(real_batch_indices,real_env_indices,env)
        virtual_data=self._get_virtual_samples(virtual_batch_indices,virtual_env_indices,env)
        observations={key:th.cat((real_data.observations[key],virtual_data.observations[key]))
                      for key in virtual_data.observations.keys()}
        actions=th.cat((real_data.actions,virtual_data.actions))
        next_observations={key:th.cat((real_data.next_observations[key],virtual_data.next_observations[key]))
                           for key in virtual_data.next_observations.keys()}
        dones=th.cat((real_data.dones,virtual_data.dones))
        rewards=th.cat((real_data.rewards,virtual_data.rewards))
        return DictReplayBufferSamples(observations=observations,actions=actions,
            next_observations=next_observations,dones=dones,rewards=rewards)

    def sampling_report(self):
        """Current eligibility and actual lifetime draws, without modifying RNG."""
        _,pools=self._eligible_object_pools()
        counts=[len(pool) for pool in pools]
        return {'protocol':self.sampling_protocol,'stable_baselines3_version':self.compatible_sb3_version,
            'object_indices':list(self.object_indices),'object_info_key':OBJECT_INFO_KEY,
            'eligible_transitions':sum(counts),
            'eligible_counts_by_object':{str(obj):count for obj,count in zip(self.object_indices,counts)},
            'object_sampling_probabilities':{str(obj):1./len(self.object_indices) for obj in self.object_indices},
            'sampling_ready':all(counts),'missing_objects':[obj for obj,count in zip(self.object_indices,counts) if not count],
            'cumulative_draw_counts_by_object':{str(obj):int(count) for obj,count in zip(self.object_indices,self.object_draw_counts)},
            'cumulative_draws':int(self.object_draw_counts.sum()),'her_ratio':self.her_ratio,
            'within_object_sampling':'uniform eligible original transitions',
            'source_prioritization':False,'failure_prioritization':False,
            'object_label_added_to_policy_observation':False}
