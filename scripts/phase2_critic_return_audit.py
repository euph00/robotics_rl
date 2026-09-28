#!/usr/bin/env python3
"""Read-only Q/return and action-ranking audit on physical TRAIN demonstrations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pnr.policies import load_policy_checkpoint


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9)),
            'min':float(x.min()),'max':float(x.max())}


def dataset():
    bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layouts={r['layout_id']:r for r in bank['partitions']['train']}
    episodes=[];sources=[]
    for run in ('phase2_carry_demos01','phase2_carry_demos02'):
        for path in sorted((ROOT/'runs'/run/'artifacts').glob('layout_*.pkl')):
            data=pickle.loads(path.read_bytes())
            for ep in data['episodes']:
                rows=ep['transitions'];goal=rows[0]['observation']['desired_goal']
                if (not ep['success'] or ep['corrective_noise_std']!=0
                        or not np.allclose(goal*20,layouts[ep['layout_id']]['goal_pos'][:2],atol=1e-5,rtol=0.)):continue
                assert not rows[-1]['terminated'] and rows[-1]['truncated']
                assert all(np.array_equal(row['action'],row['teacher_action']) for row in rows)
                episodes.append(ep);sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                break
    assert len(episodes)==64 and len({ep['layout_id'] for ep in episodes})==64
    return episodes,sources


def prepare(episodes,gamma=.995,weight=2.,samples=17):
    selected=[];finals=[]
    for ep in episodes:
        rows=ep['transitions'];n=len(rows);returns=np.zeros(n);following=0.
        for i in range(n-1,-1,-1):following=float(rows[i]['reward'])+gamma*following;returns[i]=following
        final=rows[-1]['next_observation'];dlast=float(np.linalg.norm(final['achieved_goal']-final['desired_goal'])*20)
        finals.append(final)
        for index in np.unique(np.r_[np.linspace(0,n-1,samples).astype(int),n-5]):
            row=rows[index];distance=float(np.linalg.norm(row['observation']['achieved_goal']-row['observation']['desired_goal'])*20)
            tail=gamma**(n-index)*weight*dlast
            selected.append(dict(layout_id=ep['layout_id'],decision=int(index),remaining_decisions=n-int(index),
                observed_return=float(returns[index]),stationary_goal_tail=float(tail),
                teacher_return_with_stationary_goal=float(returns[index]+tail),distance_m=distance,
                terminal_distance_m=dlast,observation=row['observation'],next_observation=row['next_observation'],
                expert_action=row['action'],reward=float(row['reward'])))
    return selected,finals


def tensors(records,field=None):
    source=[record[field] for record in records] if field else records
    return {key:torch.as_tensor(np.stack([obs[key] for obs in source]),dtype=torch.float32)
            for key in source[0]}


def entropy(mu,log_std,draws=64):
    z=mu.unsqueeze(0)+log_std.exp().unsqueeze(0)*torch.randn(draws,*mu.shape)
    a=torch.tanh(z)
    logp=torch.distributions.Normal(mu,log_std.exp()).log_prob(z).sum(-1)
    sb3=(-logp+torch.log(1.-a.square()+1e-6).sum(-1)).mean(0)
    exact=(-logp+(2.*(math.log(2.)-z-torch.nn.functional.softplus(-2.*z))).sum(-1)).mean(0)
    return sb3,exact


def q_actions(model,obs,actions,target=False):
    critic=model.critic_target if target else model.critic
    features=critic.extract_features(obs,critic.features_extractor)
    # The critic receives SAC-normalized raw actions. Carry physics additionally
    # projects controls onto the unit L2 ball; report that caveat explicitly.
    n,k,_=actions.shape
    joined=torch.cat((features[:,None,:].expand(-1,k,-1),actions),dim=2).reshape(n*k,-1)
    q=torch.cat([net(joined) for net in critic.q_networks],dim=1).reshape(n,k,-1)
    return q.min(dim=2).values,q.max(dim=2).values


def evaluate(name,path,states,finals):
    model=load_policy_checkpoint(path,device='cpu');model.policy.set_training_mode(False)
    torch.manual_seed(9137)
    alpha=float(model.log_ent_coef.detach().exp()) if model.log_ent_coef is not None else float(model.ent_coef_tensor)
    records=[];terminal=[]
    with torch.no_grad():
        for start in range(0,len(states),128):
            batch=states[start:start+128];obs=tensors(batch,'observation');nxt=tensors(batch,'next_observation')
            expert=torch.as_tensor(np.stack([r['expert_action'] for r in batch]),dtype=torch.float32)
            mu,log_std,_=model.actor.get_action_dist_params(obs);actor=torch.tanh(mu)
            offsets=torch.tensor([[.1,0],[-.1,0],[0,.1],[0,-.1]],dtype=torch.float32)
            perturb=(expert[:,None,:]+offsets).clamp(-1.,1.)
            alias=(expert*1.1).clamp(-1.,1.)
            actions=torch.cat((expert[:,None,:],actor[:,None,:],perturb,torch.zeros_like(expert[:,None,:]),alias[:,None,:]),dim=1)
            q,qhi=q_actions(model,obs,actions);h,hexact=entropy(mu,log_std)
            nmu,nstd,_=model.actor.get_action_dist_params(nxt);nactor=torch.tanh(nmu)
            nh,_=entropy(nmu,nstd)
            tq,_=q_actions(model,nxt,nactor[:,None,:],target=True)
            bellman=np.asarray([r['reward'] for r in batch])+model.gamma*(tq[:,0].numpy()+alpha*nh.numpy())
            normalized=lambda action:action/torch.linalg.vector_norm(action,dim=1,keepdim=True).clamp_min(1.)
            actor_delta=torch.linalg.vector_norm(normalized(actor)-normalized(expert),dim=1)
            perturb_best=q[:,2:6].max(dim=1).values
            alias_equiv=torch.linalg.vector_norm(normalized(alias)-normalized(expert),dim=1)<1e-5
            alias_changed=torch.linalg.vector_norm(alias-expert,dim=1)>1e-3
            for i,state in enumerate(batch):
                record={k:v for k,v in state.items() if k not in ('observation','next_observation','expert_action')}
                record.update(q_expert=float(q[i,0]),q_actor=float(q[i,1]),q_stop=float(q[i,6]),
                    q_perturbed=[float(v) for v in q[i,2:6]],
                    q_expert_twin_spread=float(qhi[i,0]-q[i,0]),
                    actor_minus_expert_q=float(q[i,1]-q[i,0]),best_perturb_minus_expert_q=float(perturb_best[i]-q[i,0]),
                    actor_physical_action_error=float(actor_delta[i]),
                    expert_action=expert[i].tolist(),actor_action=actor[i].tolist(),
                    actor_entropy_sb3_estimate=float(h[i]),actor_entropy_exact_jacobian_estimate=float(hexact[i]),
                    pre_tanh_std=log_std[i].exp().tolist(),
                    target_bellman_mean_action_approx=float(bellman[i]),
                    equivalent_alias_changed=bool(alias_equiv[i] and alias_changed[i]),
                    equivalent_alias_q_change=float(q[i,7]-q[i,0]))
                records.append(record)
        for start in range(0,len(finals),128):
            obs=tensors(finals[start:start+128]);mu,std,_=model.actor.get_action_dist_params(obs)
            h,_=entropy(mu,std);actions=torch.cat((torch.zeros_like(mu)[:,None,:],torch.tanh(mu)[:,None,:]),dim=1)
            q,_=q_actions(model,obs,actions)
            for i in range(len(mu)):
                d=float(torch.linalg.vector_norm(obs['achieved_goal'][i]-obs['desired_goal'][i])*20)
                terminal.append(dict(distance_m=d,q_stop=float(q[i,0]),q_actor=float(q[i,1]),
                                     estimated_entropy=float(h[i]),stationary_hard_value=2*d))
    soft_upper=alpha*model.gamma*math.log(4)/(1-model.gamma)
    def summarize(rows):
        return dict(states=len(rows),q_expert=stats([r['q_expert'] for r in rows]),
            teacher_return_stationary_goal=stats([r['teacher_return_with_stationary_goal'] for r in rows]),
            q_minus_teacher_return=stats([r['q_expert']-r['teacher_return_with_stationary_goal'] for r in rows]),
            fraction_exceeding_teacher_plus_max_positive_entropy=float(np.mean([
                r['q_expert']>r['teacher_return_with_stationary_goal']+soft_upper for r in rows])),
            fraction_actor_ranked_above_expert=float(np.mean([r['actor_minus_expert_q']>0 for r in rows])),
            actor_minus_expert_q=stats([r['actor_minus_expert_q'] for r in rows]),
            fraction_stop_ranked_above_expert=float(np.mean([r['q_stop']>r['q_expert'] for r in rows])),
            best_local_perturb_q_gain=stats([r['best_perturb_minus_expert_q'] for r in rows]),
            actor_physical_action_error=stats([r['actor_physical_action_error'] for r in rows]))
    bins={}
    for low,high in ((0,20),(20,100),(100,200),(200,10000)):
        rows=[r for r in records if low<r['remaining_decisions']<=high]
        if rows:bins[f'{low+1}_to_{high}_remaining_decisions']=summarize(rows)
    aliases=[r['equivalent_alias_q_change'] for r in records if r['equivalent_alias_changed']]
    result=dict(name=name,checkpoint=str(path),sha256=sha(path),online_decisions=model.num_timesteps,
        sac_updates=model._n_updates,critic_warmup_updates=getattr(model,'_pnr_critic_warmup_updates',None),
        gamma=model.gamma,tau=model.tau,alpha=alpha,target_entropy=model.target_entropy,
        positive_entropy_upper_allowance=soft_upper,summary=summarize(records),remaining_decision_bins=bins,
        local_entropy=stats([r['actor_entropy_sb3_estimate'] for r in records]),
        entropy_jacobian_numeric_difference=stats([r['actor_entropy_sb3_estimate']-r['actor_entropy_exact_jacobian_estimate'] for r in records]),
        twin_disagreement=stats([r['q_expert_twin_spread'] for r in records]),
        identical_physics_raw_action_alias_q_differences=stats(aliases) if aliases else None,
        terminal_stop_q=stats([r['q_stop'] for r in terminal]),terminal_actor_q=stats([r['q_actor'] for r in terminal]),
        terminal_stationary_hard_value=stats([r['stationary_hard_value'] for r in terminal]),
        records=records,terminal_records=terminal)
    del model
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh audit output')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    episodes,sources=dataset();states,finals=prepare(episodes)
    init=next((ROOT/'runs/phase2_carry_recovery01/checkpoints').glob('step_000000000*/model.zip'))
    online=ROOT/'runs/phase2_carry_recovery01/checkpoints/best_online.zip'
    report=dict(kind='read_only_critic_audit',physics_ticks=0,gradient_updates=0,
        training_episodes=len(episodes),states=len(states),source_datasets=sources,
        assumptions=[
            'Only successful deterministic original-goal physical TRAIN demonstrations; expert label equals executed action.',
            'Observed return ends at the recorded success timeout; terminal goal state is not treated as absorbing.',
            'Stationary retained-goal hard tail is weight*terminal_distance, discounted to each state; this is conditional, not a new rollout.',
            'Critics estimate the checkpoint policy soft Q; demonstrated future actions/states belong to a different behavior policy, so hard teacher returns are a feasible reference, not exact soft-Q ground truth.',
            'A fixed-alpha positive entropy allowance gamma*alpha*log(4)/(1-gamma) bounds differential-entropy benefit over action support[-1,1]^2; there is no global finite negative entropy bound.',
            'Entropy is sampled at recorded states using64 independent Gaussian samples; no simulated transitions. SB3 epsilon and stable true tanh Jacobian are both reported.',
            'Bellman diagnostics use next deterministic mean-action Q plus estimated entropy, not exact expectation of Q under stochastic next actions.',
            'Local action ranking and physical-action aliases are offline Q probes; alternative physical outcomes are unobserved.'],
        checkpoints=[evaluate('initialized',init,states,finals),evaluate('online_4000',online,states,finals)])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for r in report['checkpoints']:
        print(json.dumps({k:v for k,v in r.items() if k not in ('records','terminal_records','remaining_decision_bins')},indent=2),flush=True)


if __name__=='__main__':main()
