"""Compute-only periodic carry transfer checks; no navigation claims."""
import math
import torch
from stable_baselines3 import SAC
from test_carry_features import SpaceOnlyEnv, batch
from pnr.carry_features import CarryWorldGeometryExtractor
from pnr.carry_periodic_features import CarryPeriodicGeometryExtractor, transfer_carry_periodic_policy


def test_periodic_coordinates_follow_her_and_preserve_raw_observations():
    observed = batch()
    raw = observed['observation'].clone()
    extractor = CarryPeriodicGeometryExtractor(SpaceOnlyEnv.observation_space)
    extractor.set_all_velocity_mask()
    base = CarryWorldGeometryExtractor(SpaceOnlyEnv.observation_space)
    base.set_all_velocity_mask()
    first = extractor(observed)
    assert first.shape == (16, 614)
    for relabel in (False, True):
        if relabel:
            observed['desired_goal'] = observed['desired_goal'].roll(3, 0) + .007
        actual = extractor(observed)
        world = torch.cat((observed['achieved_goal'], observed['desired_goal']), 1) * 20.
        angle = world.unsqueeze(-1) * torch.tensor([1., 2., 4.]) * (2. * math.pi)
        expected = torch.cat((angle.sin().flatten(1), angle.cos().flatten(1)), 1)
        torch.testing.assert_close(actual[:, :-24], base(observed), atol=0, rtol=0)
        torch.testing.assert_close(actual[:, -24:], expected, atol=0, rtol=0)
        assert torch.all(actual[:, -24:].abs() <= 1.)
        torch.testing.assert_close(observed['observation'], raw, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, -24:-18], first[:, -24:-18], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, -12:-6], first[:, -12:-6], atol=0, rtol=0)
    assert not torch.equal(actual[:, -18:-12], first[:, -18:-12])


def test_periodic_transfer_preserves_original_and_her_actor_q_and_action_columns():
    torch.set_num_threads(1)
    def make(extractor):
        return SAC('MultiInputPolicy', SpaceOnlyEnv(), buffer_size=32, learning_starts=10,
                   policy_kwargs={'net_arch': [8, 8], 'features_extractor_class': extractor,
                                  'features_extractor_kwargs': {'goal_scale': 20.}}, device='cpu')
    source = make(CarryWorldGeometryExtractor)
    source.actor.features_extractor.set_all_velocity_mask()
    destination = make(CarryPeriodicGeometryExtractor)
    transfer_carry_periodic_policy(source.policy, destination.policy)
    observed, actions = batch(), torch.linspace(-1., 1., 32).reshape(16, 2)
    old, new = source.policy.state_dict(), destination.policy.state_dict()
    for name in old:
        if name.startswith(('critic.qf', 'critic_target.qf')) and name.endswith('.0.weight'):
            torch.testing.assert_close(new[name][:, -2:], old[name][:, -2:], atol=0, rtol=0)
            assert torch.count_nonzero(new[name][:, 590:614]) == 0
    assert torch.count_nonzero(new['actor.latent_pi.0.weight'][:, 590:]) == 0
    torch.testing.assert_close(destination.actor.features_extractor.state_mask,
                               source.actor.features_extractor.state_mask, atol=0, rtol=0)
    assert torch.all(destination.critic.features_extractor.state_mask == 1.)
    for relabel in (False, True):
        if relabel:
            observed['desired_goal'] = observed['desired_goal'].roll(2, 0) - .03
        with torch.no_grad():
            torch.testing.assert_close(source.actor(observed, deterministic=True),
                                       destination.actor(observed, deterministic=True), atol=1e-6, rtol=1e-6)
            for key in ('critic', 'critic_target'):
                for before, after in zip(getattr(source, key)(observed, actions), getattr(destination, key)(observed, actions)):
                    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)








def test_actor_carry_pose_invariance_and_normal_her_mask_roundtrip(tmp_path):
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    vector = DummyVecEnv([SpaceOnlyEnv])
    try:
        model = SAC('MultiInputPolicy', vector, buffer_size=32, learning_starts=10,
                    replay_buffer_class=HerReplayBuffer,
                    policy_kwargs={'net_arch': [16, 16], 'features_extractor_class': CarryPeriodicGeometryExtractor,
                                   'features_extractor_kwargs': {'goal_scale': 20.}}, device='cpu')
        actor_features = model.actor.features_extractor
        actor_features.set_all_velocity_mask()
        actor_features.set_carry_pose_mask()
        observed = batch()
        changed = {key: value.clone() for key, value in observed.items()}
        changed['observation'][:, 530:578] = torch.linspace(-100., 100., 48)
        before, after = actor_features(observed), actor_features(changed)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(model.actor(observed, deterministic=True),
                                       model.actor(changed, deterministic=True), atol=0, rtol=0)
        expected = torch.ones(580);expected[530:578] = 0.
        torch.testing.assert_close(actor_features.state_mask, expected, atol=0, rtol=0)
        for critic in (model.critic, model.critic_target):
            assert torch.all(critic.features_extractor.state_mask == 1.)
            assert not torch.equal(critic.features_extractor(observed), critic.features_extractor(changed))
        # Base, goal, and contacts remain measured inputs, independent of masked pose.
        for key, index in [('observation', 528), ('observation', 578), ('achieved_goal', 0), ('desired_goal', 1)]:
            changed_kept = {name: value.clone() for name, value in observed.items()}
            changed_kept[key][:, index] += .01
            assert not torch.equal(actor_features(changed_kept), before)
        path = tmp_path / 'pose_mask_normal_her.zip'
        model.save(path)
        restored = SAC.load(path, env=vector, device='cpu')
        assert isinstance(restored.replay_buffer, HerReplayBuffer)
        torch.testing.assert_close(restored.actor.features_extractor.state_mask, expected, atol=0, rtol=0)
        assert torch.all(restored.critic.features_extractor.state_mask == 1.)
        with torch.no_grad():
            torch.testing.assert_close(restored.actor(observed, deterministic=True),
                                       restored.actor(changed, deterministic=True), atol=0, rtol=0)
    finally:
        vector.close()
