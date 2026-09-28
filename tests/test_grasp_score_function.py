"""Synthetic policies and recorded tuples only; never construct robot physics."""
from copy import deepcopy
import pickle
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC, HerReplayBuffer

from pnr import grasp_score_function as score


class SchemaOnly(gym.Env):
    observation_space = spaces.Dict({
        'observation': spaces.Box(-np.inf, np.inf, (54,), np.float32),
        'achieved_goal': spaces.Box(-np.inf, np.inf, (3,), np.float32),
        'desired_goal': spaces.Box(-np.inf, np.inf, (3,), np.float32),
    })
    action_space = spaces.Box(-1., 1., (4,), np.float32)

    def reset(self, **kwargs):
        raise AssertionError('A score-function unit test must not reset an environment')

    def step(self, action):
        raise AssertionError('A score-function unit test must not generate experience')


def model(seed=811, mean=None, log_std=None, her=False):
    torch.set_num_threads(1)
    replay = {'replay_buffer_class': HerReplayBuffer,
              'replay_buffer_kwargs': {'n_sampled_goal': 1, 'copy_info_dict': True}} if her else {}
    result = SAC('MultiInputPolicy', SchemaOnly(), device='cpu', seed=seed,
                 buffer_size=8, batch_size=2, ent_coef='auto_0.005',
                 policy_kwargs={'net_arch': []}, **replay)
    # A constant Gaussian gives analytically known mean/log-std score gradients.
    with torch.no_grad():
        result.actor.mu.weight.zero_()
        result.actor.mu.bias.copy_(torch.tensor(mean or [.1, -.2, .05, -.1]))
        result.actor.log_std.weight.zero_()
        result.actor.log_std.bias.copy_(torch.tensor(log_std or [-.8, -.7, -.6, -.5]))
    result.num_timesteps, result._n_updates = 37, 11
    return result


def observation():
    return {key: np.zeros(space.shape, np.float32)
            for key, space in SchemaOnly.observation_space.spaces.items()}


def unchanged(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(unchanged(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(unchanged(x, y) for x, y in zip(a, b))
    return a == b


def episodes(policy, outcomes=((1, 0, 0), (0, 1, 1)), lengths=(1, 2, 4)):
    result = []
    parent = score.actor_digest(policy.actor)
    for case_index, case_outcomes in enumerate(outcomes):
        for replicate, outcome in enumerate(case_outcomes):
            predictor = score.RecordedGaussianPredictor(
                policy, 7000 + case_index * 3 + replicate,
                expected_actor_sha256=parent)
            rows = []
            for step in range(lengths[replicate]):
                obs = observation()
                action, _ = predictor.predict(obs, deterministic=True)
                rows.append({'observation': deepcopy(obs), 'action': action.copy(),
                    'next_observation': deepcopy(obs), 'reward': -1.,
                    'terminated': False, 'truncated': False,
                    'info': {'bilateral_contact': False, 'physical_failure': False,
                             'action_calls_delta': 3, 'physics_ticks_delta': 15},
                    'native_call_before': 10 + step * 3,
                    'native_call_after': 13 + step * 3,
                    'physics_tick_before': 50 + step * 15,
                    'physics_tick_after': 65 + step * 15})
            result.append({'case_id': f'train_{case_index:03d}_object_14',
                'replicate_index': replicate, 'actor_sha256': parent,
                'outcome_known': True, 'Y': outcome,
                'samples': deepcopy(predictor.samples), 'transitions': rows})
    return result


def test_predictor_matches_one_native_sac_draw_and_preserves_caller_rngs():
    policy = model()
    obs = observation()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(981)
        expected = []
        for _ in range(3):
            action, _ = policy.predict(obs, deterministic=False)
            expected.append((action.copy(), policy.actor.action_dist.gaussian_actions.detach().cpu().numpy().reshape(4).copy()))
    before = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    parent = score.actor_digest(policy.actor)
    predictor = score.RecordedGaussianPredictor(policy, 981, expected_actor_sha256=parent)
    for index, (action, latent) in enumerate(expected):
        got, _ = predictor.predict(obs, deterministic=True)
        np.testing.assert_array_equal(got, action)
        np.testing.assert_array_equal(predictor.samples[index]['z'], latent)
    assert torch.equal(before[0], torch.get_rng_state())
    assert before[1] == pickle.dumps(np.random.get_state())
    assert before[2] == random.getstate()
    assert score.actor_digest(policy.actor) == parent
    predictor.sampling_report()
    obs['observation'][:] = 10.
    assert not np.any(predictor.samples[0]['observation']['observation'])


def test_saturated_actions_retain_finite_true_latents_without_inverse_tanh():
    policy = model(mean=[30., -30., 20., -20.])
    predictor = score.RecordedGaussianPredictor(policy, 92)
    action, _ = predictor.predict(observation())
    np.testing.assert_array_equal(action, [1., -1., 1., -1.])
    sample = predictor.samples[0]
    assert np.isfinite(sample['z']).all() and np.min(np.abs(sample['z'])) > 15.
    assert np.isfinite(sample['latent_log_prob']).all()
    assert not np.array_equal(sample['z'], sample['mean'])


def test_actor_change_cannot_be_silently_scored_as_on_policy():
    policy = model()
    parent = score.actor_digest(policy.actor)
    predictor = score.RecordedGaussianPredictor(policy, 92, expected_actor_sha256=parent)
    predictor.predict(observation())
    with torch.no_grad():
        policy.actor.mu.bias[0].add_(.01)
    with pytest.raises((ValueError, RuntimeError)):
        predictor.sampling_report()
    with pytest.raises((ValueError, RuntimeError)):
        score.RecordedGaussianPredictor(policy, 92, expected_actor_sha256=parent)


def test_gaussian_score_gradient_matches_analytic_loo_trajectory_sum():
    policy = model()
    recorded = episodes(policy)
    batch = score.prepare_score_batch(recorded, score.actor_digest(policy.actor),
        expected_case_ids=sorted({e['case_id'] for e in recorded}), replicates=3)
    loss = score.score_loss(policy.actor, batch)
    loss.backward()
    mean_gradient = np.zeros(4, dtype=np.float64)
    std_gradient = np.zeros(4, dtype=np.float64)
    expected_loss = 0.
    for episode in recorded:
        others = [e['Y'] for e in recorded if e['case_id'] == episode['case_id']
                  and e['replicate_index'] != episode['replicate_index']]
        advantage = episode['Y'] - np.mean(others)
        for item in episode['samples']:
            z = np.asarray(item['z'], dtype=np.float64)
            mean = np.asarray(item['mean'], dtype=np.float64)
            sigma = np.asarray(item['std'], dtype=np.float64)
            mean_gradient -= advantage * (z - mean) / sigma**2 / len(recorded)
            std_gradient -= advantage * ((z - mean)**2 / sigma**2 - 1.) / len(recorded)
            log_pdf = -.5 * ((z - mean) / sigma)**2 - np.log(sigma) - .5 * np.log(2. * np.pi)
            expected_loss -= advantage * log_pdf.sum() / len(recorded)
    # A pathwise rsample gradient would cancel the mean score. Unequal lengths
    # and mixed outcomes also expose averaging steps or including one's own Y.
    assert np.linalg.norm(mean_gradient) > .1
    np.testing.assert_allclose(policy.actor.mu.bias.grad.numpy(), mean_gradient, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(policy.actor.log_std.bias.grad.numpy(), std_gradient, rtol=2e-5, atol=2e-5)
    assert float(loss.detach()) == pytest.approx(expected_loss, rel=2e-5, abs=2e-5)
    np.testing.assert_allclose(batch['episode_advantages'], [1., -.5, -.5, -1., .5, .5])
    # Trust-region weighting deliberately differs: each trajectory gets1/6
    # regardless of its1/2/4 decisions, and total mean-KL weight stays1.
    offset = 0
    for length in batch['episode_lengths']:
        assert float(batch['kl_weights'][offset:offset+length].sum()) == pytest.approx(1./6)
        offset += length
    assert float(batch['kl_weights'].sum()) == pytest.approx(1.)


def test_same_case_constant_outcomes_have_zero_score_even_with_different_lengths():
    policy = model()
    recorded = episodes(policy, outcomes=((1, 1, 1), (0, 0, 0)))
    batch = score.prepare_score_batch(recorded, score.actor_digest(policy.actor))
    loss = score.score_loss(policy.actor, batch)
    loss.backward()
    assert float(loss.detach()) == 0.
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in policy.actor.parameters())
    assert batch['mixed_outcome_cases'] == 0


@pytest.mark.parametrize('corruption', [
    'missing_replicate', 'duplicate_replicate', 'unknown_outcome', 'invalid_outcome',
    'wrong_parent', 'unexecuted_sample', 'changed_action', 'changed_observation',
    'changed_latent', 'changed_density',
])
def test_incomplete_or_mismatched_real_evidence_cannot_enter_score_update(corruption):
    policy = model()
    recorded = episodes(policy)
    parent = score.actor_digest(policy.actor)
    if corruption == 'missing_replicate':
        recorded.pop()
    elif corruption == 'duplicate_replicate':
        recorded.append(deepcopy(recorded[0]))
    elif corruption == 'unknown_outcome':
        recorded[0]['outcome_known'] = False
    elif corruption == 'invalid_outcome':
        recorded[0]['Y'] = .5
    elif corruption == 'wrong_parent':
        recorded[0]['actor_sha256'] = '0' * 64
    elif corruption == 'unexecuted_sample':
        # A stop between sampling and completion of its native calls must not
        # be silently shortened into an apparently complete training prefix.
        recorded[0]['transitions'].pop()
    elif corruption == 'changed_action':
        recorded[0]['transitions'][0]['action'][0] += .01
    elif corruption == 'changed_observation':
        recorded[0]['transitions'][0]['observation']['observation'][0] += .01
    elif corruption == 'changed_latent':
        recorded[0]['samples'][0]['z'][0] += .01
    else:
        recorded[0]['samples'][0]['latent_log_prob'] += .1
    before = deepcopy(recorded)
    with pytest.raises((ValueError, RuntimeError)):
        score.prepare_score_batch(recorded, parent,
            expected_case_ids=['train_000_object_14', 'train_001_object_14'])
    assert score.actor_digest(policy.actor) == parent
    assert unchanged(before, recorded)


def test_pregrasp_failure_with_no_actions_remains_in_loo_and_episode_denominator():
    policy = model()
    recorded = episodes(policy, outcomes=((0, 1, 0), (0, 1, 1)))
    recorded[0].update(Y=0, samples=[], transitions=[])
    before = deepcopy(recorded)
    batch = score.prepare_score_batch(recorded, score.actor_digest(policy.actor))
    assert batch['episodes'] == 6 and batch['episode_lengths'][0] == 0
    assert len(batch['z']) == 13
    assert float(batch['kl_weights'].sum()) == pytest.approx(5./6)
    assert batch['episode_advantages'][:3] == [-.5, 1., -.5]
    assert unchanged(recorded, before)


def demo_batch():
    return ({k: np.stack([v]*4) for k, v in observation().items()},
            np.tile(np.array([-.1, .1, .1, -.2], np.float32), (4, 1)))


def test_all_preparations_failing_before_actions_never_fabricates_an_update():
    policy = model()
    recorded = episodes(policy)
    for ep in recorded:
        ep.update(Y=0, samples=[], transitions=[])
    parent = score.actor_digest(policy.actor)
    before = score.protected_state(policy)
    batch = score.prepare_score_batch(recorded, parent)
    assert batch['episodes'] == 6 and len(batch['z']) == 0
    report = score.guarded_score_update(policy, recorded, *demo_batch(), retention_weight=1.)
    assert not report['accepted'] and report['reason'] == 'no_recorded_policy_actions'
    assert score.actor_digest(policy.actor) == parent
    assert score.protected_state(policy) == before


def test_one_accepted_proposal_only_changes_actor_and_leaves_native_her_and_adam_intact():
    policy = model(her=True)
    recorded = episodes(policy)
    for i in range(4):
        current = {k: v[None].copy() for k, v in observation().items()}
        following = deepcopy(current)
        following['achieved_goal'][:] = (i + 1) * .01
        policy.replay_buffer.add(current, following, np.zeros((1, 4), np.float32),
            np.array([-1.], np.float32), np.array([i == 3]),
            [{'TimeLimit.truncated': i == 3, 'bilateral_contact': bool(i % 2)}])
    parameter = next(policy.actor.parameters())
    policy.actor.optimizer.state[parameter] = {
        'step': torch.tensor(9.), 'exp_avg': torch.ones_like(parameter) * .2,
        'exp_avg_sq': torch.ones_like(parameter) * .3}
    replay = policy.replay_buffer
    replay_before = deepcopy({name: getattr(replay, name) for name in
        ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts',
         'infos', 'ep_start', 'ep_length', 'pos', 'full', 'her_ratio')})
    protected = score.protected_state(policy)
    actor_before = score.actor_digest(policy.actor)
    rows_before = deepcopy(recorded)
    report = score.guarded_score_update(policy, recorded, *demo_batch(), retention_weight=1.)
    assert report['accepted'] and report['proposals'] == 1 and not report['backtracking']
    assert score.actor_digest(policy.actor) != actor_before
    assert score.protected_state(policy) == protected
    assert report['native_sac_updates_added'] == report['physical_interactions'] == 0
    assert report['actual_kl_mean'] <= .005 and report['actual_kl_max_perdecision'] <= .05
    assert report['actual_log_std_shift_max'] <= .1
    assert unchanged(replay_before, {name: getattr(replay, name) for name in replay_before})
    assert unchanged(recorded, rows_before)
    assert (policy.num_timesteps, policy._n_updates) == (37, 11)


def test_zero_weight_is_matched_bc_step_independent_of_retention_labels():
    first, second = model(), model()
    left = episodes(first)
    right = deepcopy(left)
    for episode in right:
        episode['Y'] = 1 - episode['Y']
    parent = score.actor_digest(first.actor)
    a = score.guarded_score_update(first, left, *demo_batch(), retention_weight=0.)
    b = score.guarded_score_update(second, right, *demo_batch(), retention_weight=0.)
    assert a['accepted'] and b['accepted']
    assert a['actor_after'] == b['actor_after'] != parent
    assert unchanged(first.actor.state_dict(), second.actor.state_dict())
    assert a['score_loss'] == pytest.approx(-b['score_loss'])
    assert a['combined_gradient_norm'] == pytest.approx(200. * a['bc_gradient_norm'])


@pytest.mark.parametrize('guard', [
    {'kl_mean_limit': 0.}, {'kl_max_limit': 0.}, {'log_std_shift_max': 0.},
])
def test_any_failed_fixed_guard_restores_all_actor_bytes_without_retry(guard):
    policy = model()
    recorded = episodes(policy)
    actor_before = deepcopy(policy.actor.state_dict())
    native_before = score.protected_state(policy)
    policy.actor.train(True)
    report = score.guarded_score_update(policy, recorded, *demo_batch(),
        retention_weight=1., **guard)
    assert not report['accepted'] and report['reason'] == 'fixed_kl_or_std_guard'
    assert report['proposals'] == 1 and report['backtracking'] is False
    assert report['actor_before'] == report['actor_after']
    assert unchanged(actor_before, policy.actor.state_dict())
    assert score.protected_state(policy) == native_before
    assert policy.actor.training is True


def test_gaussian_agreement_guard_catches_forged_density_even_with_valid_parent_hash():
    policy = model()
    recorded = episodes(policy)
    for ep in recorded:
        for sample in ep['samples']:
            sample['mean'] = sample['mean'] + .1
            sample['latent_log_prob'] = float(torch.distributions.Normal(
                torch.as_tensor(sample['mean']), torch.as_tensor(sample['std'])
            ).log_prob(torch.as_tensor(sample['z'])).sum())
    parent = score.actor_digest(policy.actor)
    with pytest.raises(ValueError, match='Gaussian differs'):
        score.guarded_score_update(policy, recorded, *demo_batch(), retention_weight=1.)
    assert score.actor_digest(policy.actor) == parent


def test_incomplete_unknown_batch_does_not_modify_actor_or_erase_partial_evidence():
    policy = model()
    recorded = episodes(policy)
    recorded[-1]['outcome_known'] = False
    recorded[-1]['partial_costs'] = {'source_action_calls': 23, 'physics_ticks': 115,
                                    'initialization_physics_ticks': 480}
    saved = deepcopy(recorded)
    parent = score.actor_digest(policy.actor)
    with pytest.raises(ValueError, match='Unknown outcome'):
        score.guarded_score_update(policy, recorded, *demo_batch(), retention_weight=1.)
    assert score.actor_digest(policy.actor) == parent and unchanged(recorded, saved)


def test_synthetic_endpoint_save_reload_preserves_actor_and_native_ancestry(tmp_path):
    policy = model()
    report = score.guarded_score_update(policy, episodes(policy), *demo_batch(), retention_weight=1.)
    assert report['accepted']
    before = score.protected_state(policy)
    target = tmp_path/'synthetic_score_endpoint.zip'
    policy.save(target)
    reloaded = SAC.load(target, device='cpu')
    assert score.actor_digest(reloaded.actor) == report['actor_after']
    assert score.protected_state(reloaded) == before
    assert reloaded.observation_space == policy.observation_space
    assert reloaded.action_space == policy.action_space
    assert (reloaded.num_timesteps, reloaded._n_updates) == (37, 11)


def test_directional_kl_curvature_matches_closed_form_gaussian_fisher():
    policy = model()
    recorded = episodes(policy)
    batch = score.prepare_score_batch(recorded, score.actor_digest(policy.actor))
    demo_obs, targets = demo_batch()
    tensors = {k: torch.as_tensor(v) for k, v in demo_obs.items()}
    bc = (policy.actor(tensors, deterministic=True) - torch.as_tensor(targets)).square().mean()
    loss = score.score_loss(policy.actor, batch) + 200. * bc
    # Zero inputs make every weight gradient zero, leaving only the Gaussian
    # mean/log-std biases. Its exact Fisher quadratic is dmu²/sigma²+2*dlogσ².
    mu_gradient, sigma_gradient = torch.autograd.grad(
        loss, [policy.actor.mu.bias, policy.actor.log_std.bias])
    norm = torch.sqrt(mu_gradient.square().sum() + sigma_gradient.square().sum())
    clipping = min(1., 10. / float(norm))
    dmu, dlogstd = -clipping * mu_gradient, -clipping * sigma_gradient
    old_sigma = policy.actor.log_std.bias.detach().exp()
    curvature = float((dmu.square() / old_sigma.square() + 2. * dlogstd.square()).sum())
    expected_step = np.sqrt(2. * .0025 / curvature)
    assert 0. < expected_step < .1
    result = score.guarded_score_update(policy, recorded, demo_obs, targets,
        retention_weight=1., step_cap=.1)
    assert result['directional_kl_curvature'] == pytest.approx(curvature, rel=3e-5)
    assert result['step_size'] == pytest.approx(expected_step, rel=3e-5)
    assert result['accepted'] and result['actual_kl_mean'] <= .005


@pytest.mark.parametrize('failure', ['prediction', 'persistence', 'rng', 'action_linkage'])
def test_swallowed_sampling_errors_remain_unknown_even_with_unchanged_actor(monkeypatch, failure):
    policy = model()
    parent = score.actor_digest(policy.actor)
    numpy_before = np.random.get_state()

    def fail(*args, **kwargs):
        raise ValueError('Synthetic recording/sampling integrity failure')

    predictor = score.RecordedGaussianPredictor(policy, 612,
        persist=fail if failure == 'persistence' else None)
    if failure == 'prediction':
        monkeypatch.setattr(policy, 'predict', fail)
    elif failure in ('rng', 'action_linkage'):
        native_predict = policy.predict

        def contaminated(*args, **kwargs):
            value = native_predict(*args, **kwargs)
            if failure == 'rng':
                np.random.random()
            else:
                action, state = value
                action = action.copy()
                action[0] += .01
                value = action, state
            return value

        monkeypatch.setattr(policy, 'predict', contaminated)
    try:
        # The unchanged system may catch this broad exception. A subsequent
        # report must still fail so it cannot turn an empty prefix into knownY0.
        with pytest.raises((ValueError, RuntimeError)):
            predictor.predict(observation())
        with pytest.raises((ValueError, RuntimeError)):
            predictor.sampling_report()
        assert score.actor_digest(policy.actor) == parent
    finally:
        np.random.set_state(numpy_before)
