"""Optional map architecture, actual saved-model checks, and old resume identity."""
import copy
from pathlib import Path

import pytest
import torch
from stable_baselines3 import SAC

from pnr.gated_spatial_features import GatedCarrySpatialExtractor
from pnr.multi_carry import CarrySchema
from pnr.navigation_features import NavigationSpatialExtractor
from pnr.phase2_checkpoints import (SEMANTIC_SOURCES, check_signature, signature,
                                   validate_encoder_model)
from pnr.spatial_features import CarrySpatialExtractor
from scripts.phase2_train_carry import encoder_policy_kwargs


def method(**overrides):
    value = dict(workers=1, layouts=192, seed=7401, horizon=900, delta_scale=.075,
        batch_size=256, learning_rate=3e-5, gamma=.995, shaping_weight=2.,
        target_entropy=-4., n_sampled_goal=1, demonstrations_sha256='d'*64,
        scene_bank_sha256='b'*64)
    value.update(overrides)
    return value


@pytest.fixture
def source(tmp_path):
    for name in SEMANTIC_SOURCES + ('pnr/gated_spatial_features.py', 'pnr/navigation_features.py'):
        path=tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen test source\n')
    return tmp_path


def make_model(encoder, navigation_only=False):
    torch.set_num_threads(1)
    model=SAC('MultiInputPolicy', CarrySchema(), buffer_size=8, batch_size=4,
              policy_kwargs=encoder_policy_kwargs(encoder,navigation_only),seed=7401,device='cpu')
    model.actor.features_extractor.set_carry_pose_mask(True)
    if navigation_only:model.actor.features_extractor.set_navigation_only(True)
    return model


def test_factory_preserves_exact_default_and_fixed_gated_settings():
    assert encoder_policy_kwargs()=={'net_arch':[128,128],
        'features_extractor_class':CarrySpatialExtractor,
        'features_extractor_kwargs':{'features_dim':128,'channels':32}}
    assert encoder_policy_kwargs(navigation_only=True)['features_extractor_class'] is NavigationSpatialExtractor
    gated=encoder_policy_kwargs('gated48-k20-f5')
    assert gated['features_extractor_class'] is GatedCarrySpatialExtractor
    assert gated['features_extractor_kwargs']=={'features_dim':128,'channels':48,'iterations':20,'kernel_size':5}
    with pytest.raises(ValueError,match='Unknown carry encoder'):encoder_policy_kwargs('gated48-k10-f5')


def test_legacy_default_signatures_keep_their_original_vocabulary_and_sources(source):
    old=signature(method(),source); preserved=copy.deepcopy(old)
    new=signature(method(encoder='cnn32'),source)
    assert 'encoder' not in old['method']
    assert old['semantic_sources']==new['semantic_sources']
    check_signature(old,new);check_signature(new,old)
    assert old==preserved
    assert 'pnr/gated_spatial_features.py' not in old['semantic_sources']
    assert 'pnr/navigation_features.py' not in old['semantic_sources']


def test_gated_signature_adds_only_relevant_source_dependencies(source):
    old=signature(method(),source);gated=signature(method(encoder='gated48-k20-f5'),source)
    assert set(gated['semantic_sources'])-set(old['semantic_sources'])=={
        'pnr/gated_spatial_features.py','pnr/navigation_features.py'}
    for first,second in ((old,gated),(gated,old)):
        with pytest.raises(ValueError,match='encoder'):check_signature(first,second)
    (source/'pnr/gated_spatial_features.py').write_text('# changed recurrent computation\n')
    check_signature(old,signature(method(),source))
    with pytest.raises(ValueError,match='gated_spatial_features'):
        check_signature(gated,signature(method(encoder='gated48-k20-f5'),source))
    with pytest.raises(ValueError,match='Unknown checkpoint encoder'):
        signature(method(encoder='gated48-k19-f5'),source)


@pytest.mark.parametrize('encoder,navigation_only',[
    ('cnn32',False),('cnn32',True),('gated48-k20-f5',False),('gated48-k20-f5',True)])
def test_real_sac_archive_preserves_declared_actor_critics_and_masks(tmp_path,encoder,navigation_only):
    model=make_model(encoder,navigation_only)
    try:
        validate_encoder_model(model,encoder,navigation_only)
        destination=tmp_path/'model.zip';model.save(destination)
        loaded=SAC.load(destination,device='cpu')
        validate_encoder_model(loaded,encoder,navigation_only)
        assert loaded.num_timesteps==loaded._n_updates==0
        assert loaded.actor.features_extractor is not loaded.critic.features_extractor
        assert loaded.critic.features_extractor is not loaded.critic_target.features_extractor
        wrong='cnn32' if encoder!='cnn32' else 'gated48-k20-f5'
        with pytest.raises(ValueError,match='encoder architecture'):
            validate_encoder_model(loaded,wrong,navigation_only)
    finally:model.env.close()


@pytest.mark.parametrize('network', ['actor','critic','critic_target'])
def test_actual_gated_iteration_mutation_is_rejected(network):
    model=make_model('gated48-k20-f5')
    try:
        getattr(model,network).features_extractor.spatial.iterations=19
        with pytest.raises(ValueError,match='propagation'):validate_encoder_model(model,'gated48-k20-f5')
    finally:model.env.close()


@pytest.mark.parametrize('mutation,match',[
    ('raw_actor','raw-pose mask'),('raw_critic','raw-pose mask'),
    ('grip_actor','compact-grip mask'),('geometry','geometry'),('width','architecture')])
def test_declared_label_cannot_hide_changed_inputs_or_capacity(mutation,match):
    model=make_model('gated48-k20-f5')
    try:
        if mutation=='raw_actor':model.actor.features_extractor.state_mask.fill_(1.)
        elif mutation=='raw_critic':model.critic.features_extractor.state_mask.fill_(0.)
        elif mutation=='grip_actor':model.actor.features_extractor.navigation_grip_mask[0]=0.
        elif mutation=='geometry':model.critic_target.features_extractor.goal_scale=10.
        elif mutation=='width':model.critic.features_extractor.channels=32
        with pytest.raises(ValueError,match=match):validate_encoder_model(model,'gated48-k20-f5')
    finally:model.env.close()
