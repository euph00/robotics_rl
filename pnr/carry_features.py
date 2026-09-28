"""Replay-safe measured carry geometry and an explicit actor velocity ablation."""
from __future__ import annotations

import torch
from .features import BoundedGoalExtractor


class CarryWorldGeometryExtractor(BoundedGoalExtractor):
    """Append measured base/goal world XY metres after any HER relabeling.

    The optional mask is a registered buffer, so an actor-only ablation survives
    normal SAC save/load while independent critic extractors retain velocities.
    It removes only root XY joint velocity features, never map or goal inputs.
    """
    def __init__(self,observation_space,goal_scale=20.):
        if observation_space['observation'].shape!=(580,) or observation_space['achieved_goal'].shape!=(2,):
            raise ValueError('Carry world geometry requires the580-state two-goal adapter')
        super().__init__(observation_space,goal_scale=goal_scale)
        self._features_dim+=4
        self.register_buffer('state_mask',torch.ones(580))

    def set_base_velocity_mask(self,enabled=True):
        # 528 map +2 normalized base +13 qpos precede the15 native qvel values.
        self.state_mask[543:545]=0. if enabled else 1.

    def set_all_velocity_mask(self,enabled=True):
        self.state_mask[543:558]=0. if enabled else 1.
        self.state_mask[572:578]=0. if enabled else 1.

    def forward(self,observations):
        masked=dict(observations)
        masked['observation']=observations['observation']*self.state_mask
        return torch.cat([super().forward(masked),
            observations['achieved_goal']*self.goal_scale,
            observations['desired_goal']*self.goal_scale],dim=1)


def transfer_carry_geometry_policy(source,destination):
    """Zero-pad geometry inputs; move each critic's action columns intact."""
    old=source.state_dict();new=destination.state_dict()
    new_keys=set(new)-set(old)
    if set(old)-set(new) or not new_keys or any(not key.endswith('features_extractor.state_mask') for key in new_keys):
        raise ValueError('Unexpected policy state changes beyond carry geometry/mask buffers')
    if any(not torch.all(new[key]==1.) for key in new_keys):
        raise ValueError('Transfer must start with unmasked actor and critic inputs')
    old_features=source.actor.features_extractor.features_dim
    new_features=destination.actor.features_extractor.features_dim
    if new_features-old_features!=4:
        raise ValueError('Expected exactly four additional world-coordinate inputs')
    expanded=[]
    for name,value in old.items():
        if value.shape==new[name].shape:
            new[name]=value.clone();continue
        if name=='actor.latent_pi.0.weight':
            if value.shape[1]!=old_features or new[name].shape!=(value.shape[0],new_features):
                raise ValueError('Unexpected actor input shape')
            target=torch.zeros_like(new[name]);target[:,:old_features]=value
        elif name.startswith(('critic.qf','critic_target.qf')) and name.endswith('.0.weight'):
            if new[name].shape!=(value.shape[0],value.shape[1]+4):
                raise ValueError('Unexpected critic input shape')
            target=torch.zeros_like(new[name]);target[:,:old_features]=value[:,:old_features]
            target[:,new_features:]=value[:,old_features:]
        else:
            raise ValueError('Unexpected policy tensor expansion: '+name)
        new[name]=target;expanded.append(name)
    destination.load_state_dict(new,strict=True)
    return {'old_features':old_features,'new_features':new_features,'expanded_tensors':expanded,
        'new_state_buffers':sorted(new_keys),'actor_new_columns':'zero appended geometry inputs',
        'critic_new_columns':'zero geometry inputs before unchanged action columns',
        'optimizer_states':'reset after shape expansion'}
