"""Replay-safe measured manipulation geometry; no planner or teacher inputs."""
from __future__ import annotations

import torch
from .features import RelativeGoalExtractor


class MeasuredGeometryExtractor(RelativeGoalExtractor):
    """Append measured EE-minus-object XYZ in useful near-contact units.

    The original 50-state order places local EE XYZ at28:31 and local object
    XYZ at35:38. Both share the same frame. HER changes canonical goal fields,
    while this measured state difference stays physically unchanged.
    """
    def __init__(self, observation_space, goal_scale=1., geometry_scale=20.):
        if observation_space['observation'].shape != (50,):
            raise ValueError('Measured manipulation geometry requires the50-state adapter')
        super().__init__(observation_space, goal_scale=goal_scale)
        self.geometry_scale=float(geometry_scale)
        self._features_dim += 3

    def forward(self, observations):
        raw=observations['observation']
        relative=(raw[:,28:31]-raw[:,35:38])*self.geometry_scale
        return torch.cat([super().forward(observations),relative],dim=1)


def transfer_expanded_geometry_policy(source, destination):
    """Copy a policy while introducing zero-influence geometry inputs.

    Actor input is [old features,new geometry]. Critic input is [old features,
    new geometry,action], so existing action columns must move to the right.
    Optimizer states are deliberately not copied across this shape change.
    """
    old=source.state_dict(); new=destination.state_dict()
    if old.keys()!=new.keys():
        raise ValueError('Policy state keys changed beyond the feature expansion')
    old_features=source.actor.features_extractor.features_dim
    new_features=destination.actor.features_extractor.features_dim
    extra=new_features-old_features
    if extra!=3:
        raise ValueError('Expected precisely three new measured-geometry features')
    expanded=[]
    for name,value in old.items():
        if value.shape==new[name].shape:
            new[name]=value.clone();continue
        if name=='actor.latent_pi.0.weight':
            if value.shape[1]!=old_features or new[name].shape!=(value.shape[0],new_features):
                raise ValueError('Unexpected actor input shape')
            target=torch.zeros_like(new[name]);target[:,:old_features]=value
        elif name.startswith(('critic.qf','critic_target.qf')) and name.endswith('.0.weight'):
            if new[name].shape!=(value.shape[0],value.shape[1]+extra):
                raise ValueError('Unexpected critic input shape')
            target=torch.zeros_like(new[name]);target[:,:old_features]=value[:,:old_features]
            target[:,new_features:]=value[:,old_features:]
        else:
            raise ValueError('Unexpected policy tensor shape change: '+name)
        new[name]=target;expanded.append(name)
    destination.load_state_dict(new,strict=True)
    return {'old_features':old_features,'new_features':new_features,'expanded_tensors':expanded,
            'actor_new_columns':'zero at end of feature input','critic_new_columns':'zero before existing action columns',
            'optimizer_states':'reset after input-dimension expansion'}
