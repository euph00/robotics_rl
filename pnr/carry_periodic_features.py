"""Replay-safe periodic carry coordinates for a bounded offline diagnostic."""
from __future__ import annotations

import math
import torch
from .carry_features import CarryWorldGeometryExtractor


class CarryPeriodicGeometryExtractor(CarryWorldGeometryExtractor):
    """Append sine/cosine of measured base and desired world XY.

    Frequencies are 1, 2, and 4 cycles/metre. All 24 new values lie in [-1, 1]
    and are recomputed from canonical achieved/desired goals after HER. They
    contain no route, teacher-action, phase, or future-state inputs.
    """
    def __init__(self, observation_space, goal_scale=20.):
        super().__init__(observation_space, goal_scale=goal_scale)
        self._features_dim += 24

    def set_carry_pose_mask(self, enabled=True):
        """Remove retained arm/payload pose and velocity only from this extractor.

        The carry adapter commands its frozen carry posture; this declared actor
        variant uses map, measured base, canonical goals, and contact flags.
        Independent critic extractors retain the complete physical state.
        """
        self.state_mask[530:578] = 0. if enabled else 1.

    def forward(self, observations):
        world = torch.cat((observations['achieved_goal'], observations['desired_goal']), dim=1) * self.goal_scale
        frequencies = world.new_tensor((1., 2., 4.))
        angles = world.unsqueeze(-1) * frequencies * (2. * math.pi)
        periodic = torch.cat((angles.sin().flatten(1), angles.cos().flatten(1)), dim=1)
        return torch.cat((super().forward(observations), periodic), dim=1)


def transfer_carry_periodic_policy(source, destination):
    """Keep policy outputs while inserting zero periodic feature weights.

    Actor inputs append the new columns; critic inputs insert them before the
    existing action columns. Mask buffers and all remaining tensors are copied.
    Optimizers must be rebuilt for the expanded input tensors.
    """
    old, new = source.state_dict(), destination.state_dict()
    if set(old) != set(new):
        raise ValueError('Periodic transfer requires matching policy state keys')
    old_features = source.actor.features_extractor.features_dim
    new_features = destination.actor.features_extractor.features_dim
    if not isinstance(source.actor.features_extractor, CarryWorldGeometryExtractor) or isinstance(source.actor.features_extractor, CarryPeriodicGeometryExtractor):
        raise ValueError('Source must use the original carry world geometry')
    if not isinstance(destination.actor.features_extractor, CarryPeriodicGeometryExtractor) or new_features - old_features != 24:
        raise ValueError('Expected exactly 24 additional periodic inputs')
    if source.actor.features_extractor.goal_scale != destination.actor.features_extractor.goal_scale:
        raise ValueError('Canonical world coordinate scaling must stay unchanged')
    expanded = []
    for name, value in old.items():
        if value.shape == new[name].shape:
            new[name] = value.clone()
            continue
        if name == 'actor.latent_pi.0.weight':
            if value.shape[1] != old_features or new[name].shape != (value.shape[0], new_features):
                raise ValueError('Unexpected actor input shape')
            target = torch.zeros_like(new[name])
            target[:, :old_features] = value
        elif name.startswith(('critic.qf', 'critic_target.qf')) and name.endswith('.0.weight'):
            if new[name].shape != (value.shape[0], value.shape[1] + 24):
                raise ValueError('Unexpected critic input shape')
            target = torch.zeros_like(new[name])
            target[:, :old_features] = value[:, :old_features]
            target[:, new_features:] = value[:, old_features:]
        else:
            raise ValueError('Unexpected policy tensor expansion: ' + name)
        new[name] = target
        expanded.append(name)
    destination.load_state_dict(new, strict=True)
    return {'old_features': old_features, 'new_features': new_features,
            'expanded_tensors': expanded, 'actor_new_columns': 'zero appended periodic inputs',
            'critic_new_columns': 'zero periodic inputs before unchanged action columns',
            'optimizer_states': 'reset after shape expansion'}
