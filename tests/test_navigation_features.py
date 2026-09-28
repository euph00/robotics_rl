"""Actor route choices must ignore payload identity while critics retain it."""
import numpy as np
import torch
from stable_baselines3 import SAC
from pnr.navigation_features import NavigationSpatialExtractor
from test_spatial_features import SpaceOnlyEnv, batch


def test_navigation_mask_removes_payload_but_preserves_velocity_map_and_goal():
    torch.set_num_threads(1)
    actor=NavigationSpatialExtractor(SpaceOnlyEnv.observation_space,channels=8)
    actor.set_navigation_only(True)
    critic=NavigationSpatialExtractor(SpaceOnlyEnv.observation_space,channels=8)
    observed=batch()
    changed={k:v.clone() for k,v in observed.items()}
    changed['observation'][:,530:543]+=3.
    changed['observation'][:,545:580]-=2.
    torch.testing.assert_close(actor(observed),actor(changed),atol=0,rtol=0)
    assert not torch.equal(critic(observed),critic(changed))
    for key,index in [('observation',543),('observation',14),('desired_goal',0)]:
        changed={k:v.clone() for k,v in observed.items()}
        changed[key][:,index]+=.1
        assert not torch.equal(actor(observed),actor(changed)),(key,index)


def test_navigation_actor_transfer_and_roundtrip_preserve_critic_state(tmp_path):
    torch.set_num_threads(1)
    from pnr.spatial_features import CarrySpatialExtractor
    common=dict(buffer_size=32,policy_kwargs={'net_arch':[16,16],
        'features_extractor_class':CarrySpatialExtractor,
        'features_extractor_kwargs':{'channels':8,'features_dim':32}},device='cpu')
    source=SAC('MultiInputPolicy',SpaceOnlyEnv(),**common)
    common['policy_kwargs']['features_extractor_class']=NavigationSpatialExtractor
    model=SAC('MultiInputPolicy',SpaceOnlyEnv(),**common)
    mismatch=model.actor.load_state_dict(source.actor.state_dict(),strict=False)
    assert mismatch.missing_keys==['features_extractor.navigation_grip_mask']
    assert not mismatch.unexpected_keys
    model.actor.features_extractor.set_navigation_only(True)
    path=tmp_path/'navigation.zip';model.save(path)
    restored=SAC.load(path,env=SpaceOnlyEnv(),device='cpu')
    observed=batch()
    with torch.no_grad():
        torch.testing.assert_close(model.actor(observed,deterministic=True),restored.actor(observed,deterministic=True),atol=0,rtol=0)
    for network in [restored.critic,restored.critic_target]:
        assert torch.all(network.features_extractor.state_mask==1.)
        assert torch.all(network.features_extractor.navigation_grip_mask==1.)
