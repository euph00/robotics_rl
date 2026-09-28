#!/usr/bin/env python3
"""Bounded offline diagnosis of the milestone-1 BC-to-SAC carry transfer.

No simulator calls, environment interactions, or published policy updates occur.
The reward-only stub is used solely to recompute saved HER replay rewards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.carry import CarrySkillEnv


def stats(value):
    a = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                minimum=float(a.min()), maximum=float(a.max()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return SAC.load(path, device='cpu', replay_buffer_class=DictReplayBuffer,
                    replay_buffer_kwargs={}, buffer_size=1)


class RewardOnlyEnv(gym.Env):
    """Pure saved-reward adapter; cannot execute or invent physical transitions."""
    goal_scale, tolerance, failure_cost = 20., .15, 200.
    compute_reward = CarrySkillEnv.compute_reward

    def __init__(self, observation_space, action_space):
        self.observation_space, self.action_space = observation_space, action_space

    def reset(self, *, seed=None, options=None):
        raise RuntimeError('Offline audit cannot generate transitions')

    def step(self, action):
        raise RuntimeError('Offline audit cannot generate transitions')


def dataset_groups(path, max_rows=2048):
    dataset = pickle.loads(path.read_bytes())
    full, segment = [], []
    for episode in dataset['episodes']:
        rows = episode['transitions']
        # Recorded full route episodes have the simulator's exit goal (16,1).
        # Use declared stratum metadata, rather than assume those coordinates.
        summary = episode['summary']
        is_full = ('full' in str(summary.get('stratum', '')) or
                   'full' in str(summary.get('scope', '')) or
                   'full' in str(summary.get('evaluation_scope', '')))
        (full if is_full else segment).extend(rows)
    if not full:
        # The longest episodes are the full route in this specific immutable
        # dataset; validate the exact count against its recorded BC protocol.
        full_episodes = [e for e in dataset['episodes'] if len(e['transitions']) > 500]
        if len(full_episodes) != 8:
            raise ValueError('Cannot identify the eight recorded full-route episodes')
        full = [r for e in full_episodes for r in e['transitions']]
        segment = [r for e in dataset['episodes'] if len(e['transitions']) <= 500 for r in e['transitions']]
    result = {}
    for name, rows in [('full_route', full), ('other_goals', segment)]:
        selected = np.linspace(0, len(rows)-1, min(max_rows, len(rows))).astype(int)
        records = [rows[i] for i in selected]
        observations = {key: torch.as_tensor(np.stack([r['observation'][key] for r in records]), dtype=torch.float32)
                        for key in records[0]['observation']}
        actions = torch.as_tensor(np.stack([r.get('teacher_action',r['action']) for r in records]), dtype=torch.float32)
        result[name] = (observations, actions, len(rows))
    return result


def policy_metrics(model, observation, teacher=None, reference=None):
    with torch.no_grad():
        mean, log_std, _ = model.actor.get_action_dist_params(observation)
        action = torch.tanh(mean)
        q = torch.cat(model.critic(observation, action), dim=1).min(dim=1).values
        torch.manual_seed(19)
        sample, logp = model.actor.action_log_prob(observation)
        result = {'sigma': stats(log_std.exp()), 'deterministic_action': stats(action),
                  'sigma_at_ceiling_fraction':float((log_std>=2.-1e-6).float().mean()),
                  'critic_q_at_mean': stats(q), 'sample_log_probability': stats(logp),
                  'stochastic_action_deviation_norm': stats((sample-action).norm(dim=1))}
        executed=lambda a:a/torch.clamp(a.norm(dim=1,keepdim=True),min=1.)
        result['stochastic_native_target_deviation_m']=stats(.045*(executed(sample)-executed(action)).norm(dim=1))
        if teacher is not None:
            result['teacher_action_mse'] = float(((action-teacher)**2).mean())
        if reference is not None:
            result['mean_action_drift'] = stats((action-reference).norm(dim=1))
        return result


def gradient_metrics(model, observation, teacher, rl_observation=None):
    params = list(model.actor.parameters())
    rl_observation=observation if rl_observation is None else rl_observation
    torch.manual_seed(491)
    sampled, logp = model.actor.action_log_prob(rl_observation)
    q = torch.cat(model.critic(rl_observation, sampled), dim=1).min(dim=1).values
    alpha = model.log_ent_coef.detach().exp()
    prediction = model.actor(observation, deterministic=True)
    losses = {'q': -q.mean(), 'entropy': alpha * logp.mean(),
              'teacher': ((prediction-teacher)**2).mean()}
    gradients = {}
    for name, loss in losses.items():
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        gradients[name] = torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,g)])
    cosine = lambda a,b: float(torch.nn.functional.cosine_similarity(a,b,dim=0))
    result = {'losses': {k: float(v.detach()) for k,v in losses.items()},
              'actor_gradient_norms': {k: float(v.norm()) for k,v in gradients.items()},
              'q_teacher_gradient_cosine': cosine(gradients['q'],gradients['teacher']),
              'entropy_teacher_gradient_cosine': cosine(gradients['entropy'],gradients['teacher']),
              'sac_teacher_gradient_cosine': cosine(gradients['q']+gradients['entropy'],gradients['teacher'])}
    action = prediction.detach().requires_grad_(True)
    q = torch.cat(model.critic(observation, action), dim=1).min(dim=1).values
    dq = torch.autograd.grad(q.sum(), action)[0]
    correction = teacher-action.detach()
    dot = (dq*correction).sum(dim=1)
    result['q_action_gradient_norm'] = stats(dq.norm(dim=1))
    result['q_gradient_toward_teacher_fraction'] = float((dot>0).float().mean())
    result['q_gradient_teacher_dot'] = stats(dot)
    with torch.no_grad():
        result['q_action_rankings'] = {}
        for name, value in [('actor',action), ('teacher',teacher),('zero',torch.zeros_like(action)),('opposite_teacher',-teacher)]:
            result['q_action_rankings'][name] = stats(torch.cat(model.critic(observation,value),dim=1).min(dim=1).values)
    return result


def critic_warmup(model, replay, steps):
    """Diagnostic only: policy fixed, original soft Bellman critic objective."""
    for _ in range(steps):
        batch = replay.sample(model.batch_size)
        with torch.no_grad():
            action, logp = model.actor.action_log_prob(batch.next_observations)
            q = torch.cat(model.critic_target(batch.next_observations,action),dim=1).min(dim=1,keepdim=True).values
            target = batch.rewards + (1-batch.dones)*model.gamma*(q-model.ent_coef_tensor*logp.reshape(-1,1))
        loss = .5*sum(torch.nn.functional.mse_loss(value,target) for value in model.critic(batch.observations,batch.actions))
        model.critic.optimizer.zero_grad();loss.backward();model.critic.optimizer.step()
        polyak_update(model.critic.parameters(),model.critic_target.parameters(),model.tau)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/phase2_sac_audit01/artifacts/result.json')
    parser.add_argument('--offline-updates', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    np.random.seed(932);torch.manual_seed(932)
    started = time.monotonic()
    run = ROOT/'runs/sac_carry_bc_finetune02'
    paths = {'bc':run/'artifacts/inputs/bc_model.zip',
             'after_128_decisions':run/'checkpoints/best/model.zip',
             'after_2000_decisions':run/'checkpoints/step_000092000_1790563229405075654/model.zip'}
    initial_path = run/'checkpoints/step_000090000_1790563118291090216/model.zip'
    replay_path = initial_path.parent/'replay.pkl'
    dataset_path = ROOT/'runs/carry_bc_fullgoal02/artifacts/inputs/demonstrations.pkl'
    groups = dataset_groups(dataset_path)
    initial = load(paths['bc'])
    ref = {name: initial.actor(obs,deterministic=True).detach() for name,(obs,_,_) in groups.items()}
    result = {'schema_version':1,'method':'saved-data offline audit, no physical interactions',
              'inputs':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for k,p in paths.items()},
              'dataset_sha256':sha(dataset_path), 'physical_interactions':0,
              'audit_script_sha256':sha(Path(__file__)), 'torch_threads':torch.get_num_threads(),
              'device':'cpu', 'sample_seed':932, 'offline_probe_seed':144,
              'policies':{},'limits':['No physical success is inferred from action MSE or Q diagnostics.',
                  'Offline replay perturbations do not recreate online rollout data or exact training RNG.']}
    for name,path in paths.items():
        model = initial if name == 'bc' else load(path)
        data,parameters,variables = load_from_zip_file(path,device='cpu')
        entry={'timesteps':model.num_timesteps,'updates':model._n_updates,
               'learning_rate':model.learning_rate,'schedule':model.lr_schedule(1.),
               'alpha':float(model.log_ent_coef.exp()),
               'optimizer_lrs':{k:p['param_groups'][0]['lr'] for k,p in parameters.items() if 'optimizer' in k},
               'groups':{}}
        for group,(obs,teacher,count) in groups.items():
            entry['groups'][group]={'available_rows':count,'sampled_rows':len(teacher),
                **policy_metrics(model,obs,teacher,ref[group])}
        result['policies'][name]=entry
        if name != 'bc': del model
    obs,teacher,_ = groups['full_route']
    result['initial_full_route_gradient_diagnosis']=gradient_metrics(initial,obs,teacher)
    print('checkpoint and actor-gradient comparisons complete',flush=True)
    replay = pickle.loads(replay_path.read_bytes())
    replay.device=torch.device('cpu')
    reward_env = DummyVecEnv([lambda:RewardOnlyEnv(initial.observation_space,initial.action_space)])
    replay.set_env(reward_env)
    valid=np.flatnonzero((replay.ep_length>0).ravel())
    selected=np.random.default_rng(23).choice(valid,size=min(20000,len(valid)),replace=False)
    times,workers=np.unravel_index(selected,replay.ep_length.shape)
    stored_obs={key:torch.as_tensor(value[times,workers]) for key,value in replay.observations.items()}
    expected=reward_env.env_method('compute_reward',replay.next_observations['achieved_goal'][times,workers],
        replay.observations['desired_goal'][times,workers],replay.infos[times,workers])[0]
    actual=replay.rewards[times,workers]
    full_goal=groups['full_route'][0]['desired_goal'][0].numpy()
    result['replay']={'buffer_size_per_worker':replay.buffer_size,'workers':replay.n_envs,
        'valid_transition_slots':len(valid),'copy_info_dict':replay.copy_info_dict,
        'her_ratio':replay.her_ratio,'reward_recompute_mismatch_count':int(np.sum(expected!=actual)),
        'recorded_rewards':{str(v):int(np.sum(actual==v)) for v in np.unique(actual)},
        'full_route_goal_fraction':float(np.mean(np.linalg.norm(stored_obs['desired_goal'].numpy()-full_goal,axis=1)<1e-5)),
        'raw_done_fraction':float(replay.dones[times,workers].mean()),
        'timeout_fraction':float(replay.timeouts[times,workers].mean()),
        'physical_terminal_fraction':float((replay.dones[times,workers]*(1-replay.timeouts[times,workers])).mean()),
        'original_goal_distance_m':stats((stored_obs['desired_goal']-stored_obs['achieved_goal']).norm(dim=1)*20.),
        'actor_at_recorded_goals':policy_metrics(initial,{k:v[:2048] for k,v in stored_obs.items()})}
    sample=replay.sample(2048)
    result['replay']['actor_at_mixed_her_goals']=policy_metrics(initial,sample.observations)
    result['replay']['mixed_her_goal_distance_m']=stats((sample.observations['desired_goal']-sample.observations['achieved_goal']).norm(dim=1)*20.)
    result['replay']['mixed_her_full_route_goal_fraction']=float(((sample.observations['desired_goal']-torch.as_tensor(full_goal)).norm(dim=1)<1e-5).float().mean())
    result['replay']['gradient_on_mixed_her_states']=gradient_metrics(initial,obs,teacher,rl_observation=sample.observations)
    source_counts=json.loads((run/'artifacts/inputs/source_progress.json').read_text())
    result['replay']['inherited_worker_curriculum_levels']=[s['curriculum_level'] for s in source_counts['training_states']]
    result['replay']['sampled_failure_flag_terminal_mismatch_count']=int(sum(bool(i.get('physical_failure',False))!=bool(d*(1-t)) for i,d,t in zip(replay.infos[times,workers],replay.dones[times,workers],replay.timeouts[times,workers])))
    # LR update verification uses the installed SAC function's own scheduling.
    probe=load(initial_path);probe.set_logger(configure(folder=None,format_strings=[]))
    before={key:optimizer.param_groups[0]['lr'] for key,optimizer in [('actor',probe.actor.optimizer),('critic',probe.critic.optimizer),('temperature',probe.ent_coef_optimizer)]}
    probe._update_learning_rate([probe.actor.optimizer,probe.critic.optimizer,probe.ent_coef_optimizer])
    after={key:optimizer.param_groups[0]['lr'] for key,optimizer in [('actor',probe.actor.optimizer),('critic',probe.critic.optimizer),('temperature',probe.ent_coef_optimizer)]}
    result['learning_rate_probe']={'before_first_train_schedule':before,'after_schedule':after,'requested':1e-6,'effective_override_correct':all(v==1e-6 for v in after.values())}
    result['offline_update_probes']={}
    for name in ['ordinary','small_fixed_alpha','small_fixed_alpha_critic_warmup',
                 'small_fixed_alpha_reset_std','small_fixed_alpha_reset_std_critic_warmup',
                 'small_fixed_alpha_reset_std_fresh_critic_warmup'] if args.offline_updates else []:
        model=load(initial_path);model.replay_buffer=replay
        model.set_logger(configure(folder=None,format_strings=[]))
        if name != 'ordinary':
            model.ent_coef_optimizer=None;model.ent_coef_tensor=torch.tensor(.001);model.ent_coef=.001
        np.random.seed(144);torch.manual_seed(144)
        if 'reset_std' in name:
            with torch.no_grad():
                model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
        if 'fresh' in name:
            for module in model.critic.modules():
                if isinstance(module,torch.nn.Linear):module.reset_parameters()
            model.critic_target.load_state_dict(model.critic.state_dict())
            model.critic.optimizer=torch.optim.Adam(model.critic.parameters(),lr=3e-4)
        warmup=512 if name.endswith('warmup') else 0
        if warmup:
            # A discriminating probe, not the original online protocol: permit
            # faster critic alignment with the policy fixed before joint updates.
            for group in model.critic.optimizer.param_groups:group['lr']=3e-4
            critic_warmup(model,replay,warmup)
        model.train(gradient_steps=args.offline_updates,batch_size=model.batch_size)
        result['offline_update_probes'][name]={'actor_updates':args.offline_updates,
            'additional_critic_only_updates':warmup,
            'critic_reinitialized':'fresh' in name,
            'initial_sigma_reset':.049787068367863944 if 'reset_std' in name else None,
            'full_route':policy_metrics(model,obs,teacher,ref['full_route']),
            'interpretation':'Saved-replay sensitivity only; no new physics or success evaluation.'}
        print('offline probe complete:',name,flush=True)
        del model
    result['wall_seconds']=time.monotonic()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    reward_env.close()
    print(json.dumps({'result':str(args.output),'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
