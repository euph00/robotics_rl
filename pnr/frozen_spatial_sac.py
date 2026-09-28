"""Opt-in SAC ablation: preserve the BC actor's gated map core exactly.

Only actor.features_extractor.spatial is frozen. Fusion, action distribution,
independent critics and temperature retain RegularizedSAC's update equations.
No recurrent state is cached, so each HER goal still rebuilds spatial features.
Optimizer groups remain intact; Adam skips frozen parameters with grad=None.
"""
from __future__ import annotations

import copy
import hashlib
import json

from pnr.gated_spatial_features import GatedCarrySpatialExtractor, GatedMapCore
from pnr.regularized_sac import RegularizedSAC


FREEZE_PROTOCOL = 'actor-gated-spatial-freeze-v1'
FROZEN_MODULE = 'actor.features_extractor.spatial'


def spatial_core_digest(core):
    """Canonical names, tensor metadata and contiguous CPU bytes, never a ZIP hash."""
    result = hashlib.sha256()
    for name, tensor in sorted(core.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)],
                            separators=(',', ':')).encode()
        result.update(len(header).to_bytes(8, 'big')); result.update(header)
        result.update(value.numpy().tobytes())
    return result.hexdigest()


class FrozenSpatialRegularizedSAC(RegularizedSAC):
    """Activate after BC, then enforce the same freeze on every update/save/load.

SB3 serializes the manifest but parameter state_dicts omit requires_grad. The
load override therefore checks stored core bytes before restoring the flags.
Ordinary SAC.load remains usable for inference; this subclass is required for
training resume. A fresh subclass instance cannot train/save before activation.
"""
    def __init__(self, *args, **kwargs):
        self.actor_spatial_freeze = None
        super().__init__(*args, **kwargs)

    def _spatial_core(self):
        feature = self.actor.features_extractor
        if type(feature) is not GatedCarrySpatialExtractor or type(feature.spatial) is not GatedMapCore:
            raise ValueError('Actor spatial freeze requires the declared gated encoder')
        core = feature.spatial
        if (core.channels, core.iterations, core.kernel_size) != (48, 20, 5):
            raise ValueError('Actor spatial freeze requires gated48-k20-f5')
        return core

    def _freeze_description(self):
        core = self._spatial_core()
        frozen = sum(parameter.numel() for parameter in core.parameters())
        total = sum(parameter.numel() for parameter in self.actor.parameters())
        return {'protocol': FREEZE_PROTOCOL, 'module': FROZEN_MODULE,
                'core_sha256': spatial_core_digest(core),
                'frozen_actor_parameters': frozen,
                'trainable_actor_parameters': total-frozen,
                'total_actor_parameters': total,
                'critic_parameters': sum(parameter.numel() for parameter in self.critic.parameters()),
                'activation_online_decisions': 0, 'activation_online_updates': 0}

    def freeze_actor_spatial(self):
        """Seal the zero-online actor core after BC, preserving optimizer groups."""
        if self.actor_spatial_freeze is not None:
            return self.assert_actor_spatial_frozen()
        if self.num_timesteps != 0 or self._n_updates != 0:
            raise ValueError('Actor spatial freeze must start from a zero-online BC initializer')
        if self.bc_weight <= 0:
            raise ValueError('Actor spatial freeze requires positive demonstration regularization')
        self.actor_spatial_freeze = self._freeze_description()
        return self._restore_actor_spatial_freeze()

    def _restore_actor_spatial_freeze(self):
        if self.actor_spatial_freeze is None:
            raise ValueError('Missing actor spatial freeze manifest; legacy checkpoints cannot be resumed as frozen')
        if self.actor_spatial_freeze != self._freeze_description():
            raise ValueError('Actor spatial freeze manifest/core hash mismatch')
        for parameter in self._spatial_core().parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        return self.assert_actor_spatial_frozen()

    def assert_actor_spatial_frozen(self):
        """Fail closed on changed bytes, trainability, sharing, or optimizer membership."""
        if self.actor_spatial_freeze is None:
            raise ValueError('Activate actor spatial freeze after BC before training or saving')
        if self.actor_spatial_freeze != self._freeze_description():
            raise ValueError('Actor spatial freeze manifest/core hash mismatch')
        frozen = {id(parameter) for parameter in self._spatial_core().parameters()}
        actor = {id(parameter) for parameter in self.actor.parameters()}
        for parameter in self.actor.parameters():
            if parameter.requires_grad != (id(parameter) not in frozen):
                raise ValueError('Actor spatial freeze trainability differs from declared module')
            if id(parameter) in frozen and parameter.grad is not None:
                raise ValueError('Frozen actor spatial parameter has a gradient')
        optimizer = [parameter for group in self.actor.optimizer.param_groups for parameter in group['params']]
        if len(optimizer) != len(actor) or {id(parameter) for parameter in optimizer} != actor:
            raise ValueError('Actor spatial freeze must preserve the complete actor optimizer groups')
        for network in (self.critic, self.critic_target):
            if actor & {id(parameter) for parameter in network.parameters()}:
                raise ValueError('Actor spatial freeze requires independent critic encoders')
            if not all(parameter.requires_grad for parameter in network.parameters()):
                raise ValueError('Actor spatial freeze must leave all critic parameters trainable')
        if self.log_ent_coef is None or not self.log_ent_coef.requires_grad or self.ent_coef_optimizer is None:
            raise ValueError('Actor spatial freeze must retain trainable SAC temperature')
        return copy.deepcopy(self.actor_spatial_freeze)

    def train(self, gradient_steps, batch_size=64):
        self.assert_actor_spatial_frozen()
        try:
            return super().train(gradient_steps, batch_size)
        finally:
            self.assert_actor_spatial_frozen()

    def save(self, *args, **kwargs):
        self.assert_actor_spatial_frozen()
        return super().save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        model = super().load(*args, **kwargs)
        model._restore_actor_spatial_freeze()
        return model
