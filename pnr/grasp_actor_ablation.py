"""Opt-in compact grasp actor; original full-state critic and actions retained."""
from __future__ import annotations
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor,MultiInputPolicy
from .general_manipulation import GeneralGraspFeatures


class CompactGraspFeatures(BaseFeaturesExtractor):
    """28 measured relational features, with fresh goal arithmetic after HER.

    Coordinates/actions remain world-axis translations. This removes absolute
    pose inputs; it is not a claim of complete rotation-equivariant inference.
    The object/hand quaternion is relative and sign-canonicalized.
    """
    def __init__(self,observation_space):
        if observation_space['observation'].shape!=(54,):
            raise ValueError('Compact grasp actor expects the unchanged54-value state')
        super().__init__(observation_space,features_dim=28)

    def forward(self,observations):
        raw=observations['observation']
        tcp=raw[:,50:53]
        obj=raw[:,35:38]-raw[:,28:31]
        goal=observations['desired_goal']-observations['achieved_goal']
        hand=torch.nn.functional.normalize(raw[:,31:35],dim=1)
        item=torch.nn.functional.normalize(raw[:,38:42],dim=1)
        inverse_hand=torch.cat([-hand[:,:3],hand[:,3:]],dim=1)
        relative_xyz=(inverse_hand[:,3:]*item[:,:3]+item[:,3:]*inverse_hand[:,:3]
                      +torch.cross(inverse_hand[:,:3],item[:,:3],dim=1))
        relative_w=inverse_hand[:,3:]*item[:,3:]-(inverse_hand[:,:3]*item[:,:3]).sum(dim=1,keepdim=True)
        relative=torch.cat([relative_xyz,relative_w],dim=1)
        relative=relative*torch.where(relative_w<0,-torch.ones_like(relative_w),torch.ones_like(relative_w))
        return torch.cat([tcp*20.,obj*20.,goal*5.,relative,
            raw[:,11:13]*20.,raw[:,26:28]*50.,raw[:,48:50],
            raw[:,42:45]*5.,raw[:,45:48]*.5,raw[:,53:54]*20.,
            torch.linalg.vector_norm(tcp,dim=1,keepdim=True)*20.,
            torch.linalg.vector_norm(goal,dim=1,keepdim=True)*5.],dim=1)


class CompactGraspActorPolicy(MultiInputPolicy):
    """Independent compact actor extractor and unchanged70-feature critics."""
    def __init__(self,*args,**kwargs):
        if kwargs.get('share_features_extractor',False):
            raise ValueError('Compact actor must not replace the critic feature extractor')
        kwargs.setdefault('features_extractor_class',GeneralGraspFeatures)
        super().__init__(*args,**kwargs)

    def make_actor(self,features_extractor=None):
        if features_extractor is not None:
            raise ValueError('Actor feature sharing is not supported')
        extractor=CompactGraspFeatures(self.observation_space)
        kwargs=self._update_features_extractor(self.actor_kwargs,extractor)
        return Actor(**kwargs).to(self.device)
