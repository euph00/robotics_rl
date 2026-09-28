"""Compute-only real SAC/HER updates, freeze persistence, and default compatibility."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.logger import configure

from pnr.frozen_spatial_sac import (FREEZE_PROTOCOL, FrozenSpatialRegularizedSAC,
                                  spatial_core_digest)
from pnr.phase2_checkpoints import (check_signature, signature, publish_checkpoint,
    load_checkpoint, validate_loaded_model, validate_spatial_freeze_model)
from pnr.regularized_sac import RegularizedSAC
from pnr.sac_initialization import calibrate_actor_std, critic_only_warmup
from scripts.phase2_train_carry import encoder_policy_kwargs
from test_spatial_features import SpaceOnlyEnv, batch

ROOT = Path(__file__).resolve().parents[1]


class RewardOnly(SpaceOnlyEnv):
    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.linalg.norm(achieved_goal-desired_goal, axis=-1).astype(np.float32)


def make(cls=FrozenSpatialRegularizedSAC, encoder='gated48-k20-f5'):
    torch.set_num_threads(1)
    model=cls('MultiInputPolicy', RewardOnly(), buffer_size=64, batch_size=4,
        learning_rate=3e-4, seed=7401, device='cpu', gamma=.995,
        ent_coef='auto_0.005', target_entropy=-4., bc_weight=100., demo_batch_size=4,
        policy_kwargs=encoder_policy_kwargs(encoder), replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'copy_info_dict':True,'n_sampled_goal':1})
    model.actor.features_extractor.set_carry_pose_mask(True)
    model.set_logger(configure(folder=None,format_strings=[]))
    observed={k:v.numpy().copy() for k,v in batch(16).items()}
    actions=np.full((16,2),.4,np.float32)
    model.set_demonstrations(observed,actions,seed=81)
    for index in range(16):
        current={k:v[index:index+1].copy() for k,v in observed.items()}
        following={k:v.copy() for k,v in current.items()}
        following['achieved_goal']+=.002
        model.replay_buffer.add(current,following,np.full((1,2),.1,np.float32),
            np.array([-.5],np.float32),np.array([(index+1)%4==0]),
            [{'TimeLimit.truncated':(index+1)%4==0}])
    calibrate_actor_std(model,-3.)
    return model,observed,actions


def method(frozen=False):
    result=dict(workers=1,layouts=192,seed=7401,horizon=900,delta_scale=.075,
        batch_size=4,learning_rate=3e-4,gamma=.995,shaping_weight=2.,
        target_entropy=-4.,n_sampled_goal=1,demonstrations_sha256='d'*64,
        scene_bank_sha256='b'*64,buffer_size=64,bc_weight=100.,demo_batch_size=4,
        encoder='gated48-k20-f5')
    if frozen:result.update(freeze_actor_spatial=True,spatial_freeze_protocol=FREEZE_PROTOCOL)
    return result


def changed(before,after):
    return any(not torch.equal(before[key],value) for key,value in after.items())


def test_activation_counts_clears_stale_grads_and_keeps_optimizer_groups(tmp_path):
    model,_,_=make()
    try:
        with pytest.raises(ValueError,match='Activate'):model.train(1,4)
        with pytest.raises(ValueError,match='Activate'):model.save(tmp_path/'unfrozen.zip')
        core=model.actor.features_extractor.spatial
        original=spatial_core_digest(core)
        groups=[[id(p) for p in group['params']] for group in model.actor.optimizer.param_groups]
        for p in core.parameters():p.grad=torch.ones_like(p)
        manifest=model.freeze_actor_spatial()
        assert manifest['core_sha256']==original
        assert manifest['frozen_actor_parameters']==54769
        assert manifest['trainable_actor_parameters']==72836
        assert manifest['total_actor_parameters']==127605
        assert manifest['critic_parameters']==160883
        assert groups==[[id(p) for p in group['params']] for group in model.actor.optimizer.param_groups]
        assert all(p.grad is None and not p.requires_grad for p in core.parameters())
        assert manifest==model.freeze_actor_spatial()
    finally:model.env.close()


def test_actual_sac_her_updates_only_unfrozen_actor_critics_and_temperature():
    model,_,_=make()
    try:
        manifest=model.freeze_actor_spatial()
        modules={'fusion':model.actor.features_extractor.fusion,'latent':model.actor.latent_pi,
            'mean':model.actor.mu,'std':model.actor.log_std,'critic':model.critic,
            'critic_target':model.critic_target}
        before={name:copy.deepcopy(module.state_dict()) for name,module in modules.items()}
        alpha=model.log_ent_coef.detach().clone()
        model.train(2,4)
        assert model.assert_actor_spatial_frozen()==manifest
        assert all(changed(before[name],module.state_dict()) for name,module in modules.items())
        assert not torch.equal(alpha,model.log_ent_coef)
        assert model._n_updates==2 and model._regularization_updates==2
        assert model.replay_buffer.her_ratio==.5
        assert np.isfinite(model.last_regularization_metrics['bc_to_rl_grad_ratio'])
    finally:model.env.close()


def test_critic_warmup_keeps_exact_frozen_actor_and_temperature():
    model,_,_=make()
    try:
        manifest=model.freeze_actor_spatial()
        actor=copy.deepcopy(model.actor.state_dict());critic=copy.deepcopy(model.critic.state_dict())
        alpha=model.log_ent_coef.detach().clone()
        result=critic_only_warmup(model,2,4,warmup_tau=.1)
        assert result['critic_updates']==2
        assert not changed(actor,model.actor.state_dict()) and changed(critic,model.critic.state_dict())
        assert torch.equal(alpha,model.log_ent_coef) and model._n_updates==0
        assert model.assert_actor_spatial_frozen()==manifest
    finally:model.env.close()


def test_save_resume_restores_flags_manifest_real_optimizer_and_training(tmp_path):
    model,observed,actions=make()
    try:
        model.freeze_actor_spatial();model.train(1,4)
        sig=signature(method(True),ROOT)
        publish_checkpoint(model,tmp_path,{'training_states':[]},sig,'synthetic_update')
        checkpoint=load_checkpoint(tmp_path,sig)
        loaded=FrozenSpatialRegularizedSAC.load(checkpoint['model'],env=model.env,device='cpu')
        loaded.load_replay_buffer(checkpoint['replay'],truncate_last_traj=True)
        validate_loaded_model(loaded,checkpoint,1)
        assert loaded.actor_spatial_freeze==checkpoint['state']['actor_spatial_freeze']
        loaded.set_demonstrations(observed,actions)
        loaded.set_logger(configure(folder=None,format_strings=[]))
        loaded.train(1,4)
        assert loaded._n_updates==2 and loaded.assert_actor_spatial_frozen()==model.actor_spatial_freeze
        # Ordinary loading remains inference-compatible but must not pass frozen resume validation.
        inference=SAC.load(checkpoint['model'],env=model.env,device='cpu')
        with torch.no_grad():
            torch.testing.assert_close(inference.actor(batch(2),deterministic=True),
                                       model.actor(batch(2),deterministic=True),rtol=0,atol=0)
        with pytest.raises(ValueError,match='resume-safe'):validate_spatial_freeze_model(inference,method(True))
        checkpoint['state']['actor_spatial_freeze']['core_sha256']='0'*64
        with pytest.raises(ValueError,match='checkpoint state'):validate_loaded_model(model,checkpoint,1)
    finally:model.env.close()


def test_frozen_core_still_rebuilds_her_goals_and_has_no_cross_call_state():
    model,_,_=make()
    try:
        manifest=model.freeze_actor_spatial();feature=model.actor.features_extractor
        obs=batch(2);other={k:v.clone() for k,v in obs.items()}
        other['desired_goal'][:]=torch.tensor([3.5,-2.5])/20.
        with torch.no_grad():
            original=feature.spatial(feature.spatial_inputs(obs))
            relabeled=feature.spatial(feature.spatial_inputs(other))
            again=feature.spatial(feature.spatial_inputs(obs))
        assert not torch.equal(original,relabeled)
        torch.testing.assert_close(original,again,rtol=0,atol=0)
        assert model.assert_actor_spatial_frozen()==manifest
    finally:model.env.close()


@pytest.mark.parametrize('mutation',['bytes','unfreeze_core','freeze_fusion','freeze_critic','optimizer'])
def test_detects_undeclared_mutations_before_update_or_save(tmp_path,mutation):
    model,_,_=make()
    try:
        model.freeze_actor_spatial()
        if mutation=='bytes':
            with torch.no_grad():next(model.actor.features_extractor.spatial.parameters()).add_(.001)
        elif mutation=='unfreeze_core':next(model.actor.features_extractor.spatial.parameters()).requires_grad_(True)
        elif mutation=='freeze_fusion':next(model.actor.features_extractor.fusion.parameters()).requires_grad_(False)
        elif mutation=='freeze_critic':next(model.critic.parameters()).requires_grad_(False)
        else:model.actor.optimizer.param_groups[0]['params'].pop()
        with pytest.raises(ValueError):model.train(1,4)
        with pytest.raises(ValueError):model.save(tmp_path/'invalid.zip')
        assert model._n_updates==0
    finally:model.env.close()


def test_legacy_load_and_signature_vocabulary_unchanged_and_no_silent_migration(tmp_path):
    ordinary,_,_=make(RegularizedSAC)
    try:
        path=tmp_path/'legacy.zip';ordinary.save(path)
        restored=RegularizedSAC.load(path,env=ordinary.env,device='cpu')
        assert all(p.requires_grad for p in restored.actor.parameters())
        with pytest.raises(ValueError,match='legacy checkpoints'):FrozenSpatialRegularizedSAC.load(path,env=ordinary.env,device='cpu')
        old=signature(method(),ROOT)
        disabled=signature(dict(method(),freeze_actor_spatial=False),ROOT)
        assert old==disabled
        assert 'freeze_actor_spatial' not in old['method']
        assert 'pnr/frozen_spatial_sac.py' not in old['semantic_sources']
        frozen=signature(method(True),ROOT)
        assert set(frozen['semantic_sources'])-set(old['semantic_sources'])=={'pnr/frozen_spatial_sac.py'}
        with pytest.raises(ValueError,match='fields differ'):check_signature(old,frozen)
        with pytest.raises(ValueError,match='fields differ'):check_signature(frozen,old)
        for bad in (dict(method(),spatial_freeze_protocol=FREEZE_PROTOCOL),
                    dict(method(True),spatial_freeze_protocol='other'),
                    dict(method(True),encoder='cnn32'),dict(method(True),bc_weight=0.),
                    dict(method(True),freeze_actor_spatial=1)):
            with pytest.raises(ValueError):signature(bad,ROOT)
    finally:ordinary.env.close()


def test_activation_rejects_online_actors_and_other_architectures():
    model,_,_=make()
    try:
        model.num_timesteps=8
        with pytest.raises(ValueError,match='zero-online'):model.freeze_actor_spatial()
    finally:model.env.close()
    model,_,_=make(encoder='cnn32')
    try:
        with pytest.raises(ValueError,match='gated encoder'):model.freeze_actor_spatial()
    finally:model.env.close()
