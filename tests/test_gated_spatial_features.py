"""No-physics checks for the opt-in GPPN-inspired continuous-action encoder."""
import copy

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC, HerReplayBuffer

from pnr.gated_spatial_features import GatedMapCore,GatedCarrySpatialExtractor
from pnr.spatial_features import CarrySpatialExtractor
from test_spatial_features import SpaceOnlyEnv,batch


@pytest.fixture(autouse=True)
def one_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_her_goal_recomputed_and_recurrence_resets_between_calls_and_batch_rows():
    torch.manual_seed(37)
    encoder=GatedCarrySpatialExtractor(SpaceOnlyEnv.observation_space,channels=8,iterations=3)
    observed=batch(3);saved={k:v.clone() for k,v in observed.items()}
    state=copy.deepcopy(encoder.state_dict())
    alternate={k:v.clone() for k,v in observed.items()}
    alternate['desired_goal'][:]=torch.tensor([3.5,-2.5])/20.
    first=encoder(observed);changed=encoder(alternate);again=encoder(observed)
    torch.testing.assert_close(first,again,rtol=0,atol=0)
    assert not torch.equal(first,changed)
    before_planes=encoder.spatial_inputs(observed);after_planes=encoder.spatial_inputs(alternate)
    torch.testing.assert_close(before_planes[:,:2],after_planes[:,:2],rtol=0,atol=0)
    assert torch.argmax(before_planes[0,2])!=torch.argmax(after_planes[0,2])
    # Relabeled goal must reach the recurrent core itself, not merely fusion.
    assert not torch.equal(encoder.spatial(before_planes),encoder.spatial(after_planes))
    single=encoder({k:v[:1] for k,v in observed.items()})
    torch.testing.assert_close(single,first[:1],rtol=1e-5,atol=1e-6)
    permuted=encoder({k:v[[2,0,1]] for k,v in observed.items()})
    torch.testing.assert_close(permuted,first[[2,0,1]],rtol=1e-5,atol=1e-6)
    for key in saved:torch.testing.assert_close(saved[key],observed[key],rtol=0,atol=0)
    for key in state:torch.testing.assert_close(state[key],encoder.state_dict()[key],rtol=0,atol=0)
    assert first.shape==(3,128) and torch.isfinite(first).all()


def test_existing_geometry_compact_grip_and_both_masks_are_preserved():
    torch.manual_seed(40)
    encoder=GatedCarrySpatialExtractor(SpaceOnlyEnv.observation_space,channels=8,iterations=2)
    ordinary=CarrySpatialExtractor(SpaceOnlyEnv.observation_space,channels=8)
    observed=batch(2)
    torch.testing.assert_close(encoder.spatial_inputs(observed),ordinary.spatial_inputs(observed),rtol=0,atol=0)
    torch.testing.assert_close(encoder.grip_summary(observed['observation']),ordinary.grip_summary(observed['observation']),rtol=0,atol=0)
    encoder.set_carry_pose_mask(True)
    changed={k:v.clone() for k,v in observed.items()}
    changed['observation'][:,530:541]+=4.;changed['observation'][:,545:558]-=3.
    changed['observation'][:,561:565]+=2.;changed['observation'][:,568:578]-=2.
    torch.testing.assert_close(encoder(observed),encoder(changed),rtol=0,atol=0)
    for index,amount in ((541,.007),(565,.015),(543,.2),(578,-1.)):
        changed={k:v.clone() for k,v in observed.items()};changed['observation'][:,index]+=amount
        assert not torch.equal(encoder(observed),encoder(changed)),index
    encoder.set_navigation_only(True)
    changed={k:v.clone() for k,v in observed.items()}
    changed['observation'][:,530:543]+=3.;changed['observation'][:,545:580]-=2.
    torch.testing.assert_close(encoder(observed),encoder(changed),rtol=0,atol=0)
    changed['observation'][:,543]+=.2
    assert not torch.equal(encoder(observed),encoder(changed))


def test_rectangular_map_axis_order_and_finite_propagation_radius():
    torch.manual_seed(41)
    core=GatedMapCore(channels=4,iterations=2,kernel_size=3)
    assert core.receptive_field_cells==9
    maps=torch.zeros(1,5,12,22)
    baseline=core(maps)
    # Query at row6,col12 has radius4 after initialization and two updates.
    far=maps.clone();far[0,0,0,0]=1.
    torch.testing.assert_close(core(far)[0,:,6,12],baseline[0,:,6,12],rtol=0,atol=0)
    nearby=maps.clone();nearby[0,0,6,12]=1.
    assert not torch.equal(core(nearby)[0,:,6,12],baseline[0,:,6,12])
    assert baseline.shape==(1,4,12,22)
    # Perturbation affects its own spatial neighborhood, not a transposed cell.
    probe=maps.clone();probe[0,0,1,18]=1.
    delta=(core(probe)-baseline).abs().sum(1)[0]
    assert delta[1,18]>0 and delta[10,1]==0


def test_all_recurrent_and_fusion_parameters_receive_finite_training_gradients():
    torch.manual_seed(42)
    encoder=GatedCarrySpatialExtractor(SpaceOnlyEnv.observation_space,channels=8,iterations=4)
    values=encoder(batch(3));loss=(values-torch.linspace(-.5,.5,128)).square().mean()
    loss.backward()
    for name,parameter in encoder.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
        assert parameter.grad.abs().sum()>0,name
    # Iterations reuse the same parameters; adding updates changes no capacity.
    deeper=GatedMapCore(channels=8,iterations=9)
    assert sum(p.numel() for p in deeper.parameters())==sum(p.numel() for p in encoder.spatial.parameters())


def test_sac_her_roundtrip_keeps_continuous_actions_independent_critics_and_masks(tmp_path):
    torch.manual_seed(43)
    env=SpaceOnlyEnv()
    model=SAC('MultiInputPolicy',env,buffer_size=32,batch_size=8,
        replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs={'copy_info_dict':True},
        policy_kwargs={'net_arch':[16,16],'features_extractor_class':GatedCarrySpatialExtractor,
                       'features_extractor_kwargs':{'channels':8,'features_dim':32,'iterations':3,'kernel_size':3}},device='cpu')
    model.actor.features_extractor.set_carry_pose_mask(True)
    actor_ids={id(p) for p in model.actor.parameters()}
    assert not actor_ids & {id(p) for p in model.critic.parameters()}
    observed=batch(2)
    action,logp=model.actor.action_log_prob(observed)
    assert action.shape==(2,2) and logp.shape==(2,)
    assert torch.isfinite(action).all() and torch.all(action.abs()<=1)
    (action.square().mean()+.001*logp.mean()).backward()
    assert model.actor.features_extractor.spatial.cell.weight_hh.grad is not None
    path=tmp_path/'gated.zip';model.save(path)
    restored=SAC.load(path,env=SpaceOnlyEnv(),device='cpu')
    assert isinstance(restored.replay_buffer,HerReplayBuffer)
    assert restored.actor.features_extractor.spatial.iterations==3
    assert restored.actor.features_extractor.spatial.kernel_size==3
    assert torch.count_nonzero(restored.actor.features_extractor.state_mask)==0
    for network in (restored.critic,restored.critic_target):
        assert torch.all(network.features_extractor.state_mask==1)
        assert torch.all(network.features_extractor.navigation_grip_mask==1)
    with torch.no_grad():
        torch.testing.assert_close(model.actor(observed,deterministic=True),restored.actor(observed,deterministic=True),rtol=0,atol=0)
        for before,after in zip(model.critic(observed,action),restored.critic(observed,action)):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
    model.env.close();restored.env.close()


@pytest.mark.parametrize('kwargs',[{'iterations':0},{'iterations':True},{'iterations':1.5},
    {'kernel_size':0},{'kernel_size':2},{'kernel_size':3.5},{'channels':0}])
def test_invalid_core_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):GatedMapCore(**kwargs)
