"""Compute-only spatial geometry, HER, physical-input, and serialization checks."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import SAC, HerReplayBuffer

from pnr.spatial_features import CarrySpatialExtractor


class SpaceOnlyEnv(gym.Env):
    observation_space = spaces.Dict({
        'observation': spaces.Box(-np.inf, np.inf, (580,), np.float32),
        'achieved_goal': spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'desired_goal': spaces.Box(-np.inf, np.inf, (2,), np.float32),
    })
    action_space = spaces.Box(-1., 1., (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        return {key: value[0].numpy() for key, value in batch(1).items()}, {}

    def step(self, action):
        raise AssertionError('Encoder checks must not execute physical transitions')


def batch(n=4):
    generator = torch.Generator().manual_seed(72)
    raw = torch.randn(n, 580, generator=generator) * .02
    raw[:, :528] = 0.
    maps = raw[:, :528].reshape(n, 2, 12, 22)
    maps[:, 0, 0, :] = maps[:, 0, -1, :] = 1.
    maps[:, 0, :, 0] = maps[:, 0, :, -1] = 1.
    maps[:, 0, 5, 10] = 1.
    maps[:, 1, 5:7, 3:5] = 1.
    raw[:, 541:543] = .017
    raw[:, 558:561] = torch.tensor([.24, 0., 1.05])
    raw[:, 565:568] = torch.tensor([.24, 0., 1.02])
    raw[:, 578:580] = 1.
    base = torch.tensor([[4.5, -1.5]]).repeat(n, 1) / 20.
    raw[:, 528:530] = base
    return {'observation': raw, 'achieved_goal': base,
            'desired_goal': torch.tensor([[11.5, 2.5]]).repeat(n, 1) / 20.}


def test_world_grid_transform_and_bilinear_sampling_orientation():
    xy = torch.tensor([[-3.5, 5.5], [17.5, -5.5], [7., 0.]])
    expected = torch.tensor([[-1., -1.], [1., 1.], [0., 0.]])
    torch.testing.assert_close(CarrySpatialExtractor.world_to_grid(xy), expected)
    rows, cols = torch.meshgrid(torch.arange(12.), torch.arange(22.), indexing='ij')
    world_x, world_y = cols - 3.5, 5.5 - rows
    field = (world_x + 2. * world_y)[None, None].repeat(3, 1, 1, 1)
    xy = torch.tensor([[1.25, 2.75], [4.8, -.6], [12., -3.3]])
    sampled = CarrySpatialExtractor.sample_world(field, xy)
    torch.testing.assert_close(sampled[:, 0], xy[:, 0] + 2. * xy[:, 1], atol=3e-6, rtol=1e-6)
    # Increasing world y must move UP the array, not down.
    single_wall = torch.zeros(1, 1, 12, 22)
    single_wall[0, 0, 3, 15] = 1.
    torch.testing.assert_close(CarrySpatialExtractor.sample_world(single_wall,
                               torch.tensor([[11.5, 2.5]])), torch.ones(1, 1))


def test_her_relabel_changes_goal_planes_and_latent_without_mutating_state():
    torch.set_num_threads(1)
    observed = batch()
    saved = {k: v.clone() for k, v in observed.items()}
    extractor = CarrySpatialExtractor(SpaceOnlyEnv.observation_space, channels=8)
    first_planes, first_features = extractor.spatial_inputs(observed), extractor(observed)
    assert first_planes.shape == (4, 5, 12, 22)
    assert torch.argmax(first_planes[0, 2]).item() == 3 * 22 + 15
    relabeled = dict(observed, desired_goal=torch.tensor([[3.5, -2.5]]).repeat(4, 1) / 20.)
    next_planes, next_features = extractor.spatial_inputs(relabeled), extractor(relabeled)
    torch.testing.assert_close(first_planes[:, :2], next_planes[:, :2], atol=0, rtol=0)
    assert torch.argmax(next_planes[0, 2]).item() == 8 * 22 + 7
    assert not torch.equal(first_features, next_features)
    for key in observed:
        torch.testing.assert_close(saved[key], observed[key], atol=0, rtol=0)
    assert next_features.shape == (4, 128) and torch.isfinite(next_features).all()


def test_actor_masks_extra_pose_but_keeps_measured_grip_payload_velocity_and_contacts():
    torch.set_num_threads(1)
    observed = batch()
    actor = CarrySpatialExtractor(SpaceOnlyEnv.observation_space, channels=8)
    critic = CarrySpatialExtractor(SpaceOnlyEnv.observation_space, channels=8)
    actor.set_carry_pose_mask(True)
    changed = {k: v.clone() for k, v in observed.items()}
    changed['observation'][:, 530:541] += 3.
    changed['observation'][:, 545:558] += 2.
    changed['observation'][:, 561:565] += 1.
    changed['observation'][:, 568:578] += 4.
    torch.testing.assert_close(actor(observed), actor(changed), atol=0, rtol=0)
    assert not torch.equal(critic(observed), critic(changed))
    for index, amount in ((541, .007), (565, .015), (543, .2), (578, -1.)):
        changed = {k: v.clone() for k, v in observed.items()}
        changed['observation'][:, index] += amount
        assert not torch.equal(actor(observed), actor(changed)), index
    assert torch.count_nonzero(actor.state_mask) == 0
    assert torch.all(critic.state_mask == 1.)


def test_encoder_backpropagation_and_batch_independence():
    torch.set_num_threads(1)
    torch.manual_seed(99)
    extractor = CarrySpatialExtractor(SpaceOnlyEnv.observation_space, channels=8)
    observed = batch()
    values = extractor(observed)
    single = extractor({key: value[:1] for key, value in observed.items()})
    torch.testing.assert_close(single, values[:1], atol=1e-6, rtol=1e-5)
    loss = (values - torch.linspace(-.5, .5, 128)).square().mean()
    loss.backward()
    for name, parameter in extractor.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert extractor.spatial[0].weight.grad.abs().sum() > 0


def test_normal_sac_her_checkpoint_roundtrip_keeps_independent_actor_critic_masks(tmp_path):
    torch.set_num_threads(1)
    model = SAC('MultiInputPolicy', SpaceOnlyEnv(), buffer_size=32, learning_starts=10,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs={'copy_info_dict': True},
                policy_kwargs={'net_arch': [16, 16], 'features_extractor_class': CarrySpatialExtractor,
                               'features_extractor_kwargs': {'channels': 8, 'features_dim': 32}},
                device='cpu')
    model.actor.features_extractor.set_carry_pose_mask(True)
    assert torch.all(model.critic.features_extractor.state_mask == 1.)
    assert torch.all(model.critic_target.features_extractor.state_mask == 1.)
    checkpoint = tmp_path / 'spatial_sac_her.zip'
    model.save(checkpoint)
    restored = SAC.load(checkpoint, env=SpaceOnlyEnv(), device='cpu')
    assert isinstance(restored.replay_buffer, HerReplayBuffer)
    assert torch.count_nonzero(restored.actor.features_extractor.state_mask) == 0
    assert torch.all(restored.critic.features_extractor.state_mask == 1.)
    observed, actions = batch(), torch.zeros(4, 2)
    with torch.no_grad():
        torch.testing.assert_close(model.actor(observed, deterministic=True),
                                   restored.actor(observed, deterministic=True), atol=0, rtol=0)
        for before, after in zip(model.critic(observed, actions), restored.critic(observed, actions)):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
