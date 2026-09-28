"""Fresh original-goal latent-score updates; no robot, HER or SAC implementation.

An episode is consumed once at its generating actor. Native tuples are immutable;
retention outcomes are separate labels. Trust scaling/BC/acceptance are an explicit
heuristic around the raw on-policy score, not an improvement guarantee.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import pickle
import random
import numpy as np
import torch

PROTOCOL = 'fresh-original-goal-latent-score-retention-v1'


def tree_digest(value):
    h = hashlib.sha256()
    def put(x):
        if isinstance(x, torch.Tensor):
            a=x.detach().cpu().contiguous().numpy();put(a)
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str,x.shape)).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x,key=str):put(k);put(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(str(type(x).__name__).encode())
            for v in x:put(v)
        else:h.update(repr(x).encode())
    put(value)
    return h.hexdigest()


def actor_digest(actor):
    return tree_digest(actor.state_dict())


class RecordedGaussianPredictor:
    """One native SAC draw with an isolated advancing CPU RNG and actual z."""
    def __init__(self, model, seed, expected_actor_sha256=None, persist=None):
        if str(model.device) != 'cpu' or model.use_sde:
            raise ValueError('Ordinary CPU SAC is required')
        self.model=model;self.model.policy.set_training_mode(False)
        self.actor_sha256=actor_digest(model.actor)
        if expected_actor_sha256 is not None and self.actor_sha256!=expected_actor_sha256:
            raise ValueError('Generating actor digest differs')
        self.seed=int(seed);self.samples=[];self.persist=persist;self.integrity_error=None
        self._rng=torch.Generator(device='cpu').manual_seed(self.seed).get_state()
        self.observation_space=model.observation_space;self.action_space=model.action_space
        if (self.action_space.shape!=(4,) or not np.all(self.action_space.low==-1)
                or not np.all(self.action_space.high==1)):
            raise ValueError('Expected native four-action normalized grasp interface')

    def predict(self, observation, deterministic=True):
        try:
            return self._predict(observation)
        except BaseException as error:
            self.integrity_error=repr(error)
            raise

    def _predict(self, observation):
        before=(torch.get_rng_state().clone(),pickle.dumps(np.random.get_state()),random.getstate())
        observed=deepcopy(observation)
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.set_rng_state(self._rng)
            prediction=self.model.predict(observation,deterministic=False)
            self._rng=torch.get_rng_state().clone()
            dist=self.model.actor.action_dist
            z=dist.gaussian_actions.detach().clone()
            mean=dist.distribution.loc.detach().clone();std=dist.distribution.scale.detach().clone()
            latent_log_prob=dist.distribution.log_prob(z).sum(dim=-1)
            action=np.asarray(prediction[0],dtype=np.float32)
            expected=self.model.policy.unscale_action(torch.tanh(z).cpu().numpy()).reshape(action.shape)
            if not np.array_equal(action,expected):
                raise ValueError('Actual single latent sample does not reproduce executed action')
            # Same single-observation forward, no second sample or simulator query.
            tensors,_=self.model.policy.obs_to_tensor(observed)
            checked_mean,checked_log_std,_=self.model.actor.get_action_dist_params(tensors)
            if not torch.equal(checked_mean,mean) or not torch.equal(checked_log_std.exp(),std):
                raise ValueError('Actual generating Gaussian differs from frozen actor')
        if (not torch.equal(before[0],torch.get_rng_state()) or before[1]!=pickle.dumps(np.random.get_state())
                or before[2]!=random.getstate()):
            raise RuntimeError('Isolated sampling changed caller RNG state')
        sample={'observation':observed,'z':z.cpu().numpy().reshape(4).copy(),
                'mean':mean.cpu().numpy().reshape(4).copy(),'std':std.cpu().numpy().reshape(4).copy(),
                'latent_log_prob':float(latent_log_prob.item()),'action':action.copy(),
                'sample_index':len(self.samples),'actor_sha256':self.actor_sha256}
        self.samples.append(sample)
        if self.persist:self.persist(self.samples)
        return prediction

    def sampling_report(self):
        if self.integrity_error:raise ValueError('Sampling integrity failure: '+self.integrity_error)
        if actor_digest(self.model.actor)!=self.actor_sha256:
            raise ValueError('Generating actor changed during collection')
        return {'protocol':PROTOCOL,'actor_sha256':self.actor_sha256,'predictions':len(self.samples),
                'torch_seed':self.seed,'final_rng_sha256':tree_digest(self._rng),'deterministic':False,
                'actual_single_sample_recorded':True,'caller_rngs_preserved':True,'added_action_noise':False}


def prepare_score_batch(episodes, expected_actor_sha256, expected_case_ids=None, replicates=3):
    """Validate complete independent episodes and form exact episodic weights."""
    if not episodes or replicates<2:raise ValueError('At least two independent replicates per case are required')
    cases={}
    for e in episodes:
        if e.get('actor_sha256')!=expected_actor_sha256:raise ValueError('Stale/off-policy actor digest')
        if not e.get('outcome_known') or e.get('Y') not in (0,1,False,True):
            raise ValueError('Unknown outcome: abort whole batch')
        key=e['case_id'];rep=e['replicate_index']
        if rep in cases.setdefault(key,{}):raise ValueError('Duplicate case replicate')
        cases[key][rep]=e
        if len(e['samples'])!=len(e['transitions']):raise ValueError('Sample/native tuple mismatch or partial action')
        for sample,row in zip(e['samples'],e['transitions']):
            if sample.get('actor_sha256',expected_actor_sha256)!=expected_actor_sha256:
                raise ValueError('Mixed generating actor')
            if set(sample['observation'])!=set(row['observation']):raise ValueError('Observation keys differ')
            if any(not np.array_equal(v,row['observation'][k]) for k,v in sample['observation'].items()):
                raise ValueError('Policy input differs from actual native tuple')
            if not np.array_equal(sample['action'],row['action']):raise ValueError('Executed action differs from native tuple')
            for key2 in ('z','mean','std','action'):
                a=np.asarray(sample[key2])
                if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Malformed Gaussian/action record')
            if np.any(np.asarray(sample['std'])<=0):raise ValueError('Nonpositive Gaussian scale')
            if not math.isfinite(sample['latent_log_prob']):raise ValueError('Nonfinite generating log probability')
            # Genuine action mapping may contain float32 cancellation in SB3 unscale.
            z=torch.as_tensor(sample['z']);expected=(-1.+.5*(torch.tanh(z).numpy()+1.)*2.).astype(np.float32)
            if not np.array_equal(expected,sample['action']):raise ValueError('Latent/executed action linkage is invalid')
            lp=torch.distributions.Normal(torch.as_tensor(sample['mean']),torch.as_tensor(sample['std'])).log_prob(z).sum()
            if not np.isclose(float(lp),sample['latent_log_prob'],rtol=1e-6,atol=2e-6):
                raise ValueError('Stored latent probability does not match actual Gaussian')
    expected=set(cases) if expected_case_ids is None else set(expected_case_ids)
    if set(cases)!=expected or any(set(v)!=set(range(replicates)) for v in cases.values()):
        raise ValueError('Incomplete or undeclared case/replicate schedule')
    ordered=[cases[c][i] for c in sorted(cases) for i in range(replicates)]
    samples=[];weights=[];kl_weights=[];advantages=[];lengths=[];ids=[]
    mixed=0
    for c,group in sorted(cases.items()):
        ys=[float(group[i]['Y']) for i in range(replicates)]
        mixed+=int(min(ys)!=max(ys))
        for i,e in sorted(group.items()):
            advantage=ys[i]-(sum(ys)-ys[i])/(replicates-1)
            n=len(e['samples']);advantages.append(advantage);lengths.append(n);ids.append((c,i))
            samples.extend(e['samples']);weights.extend([advantage/len(ordered)]*n)
            if n:kl_weights.extend([1./(len(ordered)*n)]*n)
    def stacked(key):
        return torch.as_tensor(np.stack([s[key] for s in samples]),dtype=torch.float32) if samples else torch.empty((0,4))
    obs={k:torch.as_tensor(np.stack([s['observation'][k] for s in samples]),dtype=torch.float32)
         for k in samples[0]['observation']} if samples else {}
    return {'observations':obs,'z':stacked('z').detach(),'mean':stacked('mean').detach(),'std':stacked('std').detach(),
            'actions':stacked('action').detach(),'sample_weights':torch.tensor(weights,dtype=torch.float32),
            'kl_weights':torch.tensor(kl_weights,dtype=torch.float32),'episode_advantages':advantages,
            'episode_lengths':lengths,'episode_ids':ids,'mixed_outcome_cases':mixed,'episodes':len(ordered),
            'actor_sha256':expected_actor_sha256}


def gaussian_parameters(actor,batch):
    mean,log_std,_=actor.get_action_dist_params(batch['observations'])
    return mean,log_std.exp()


def score_loss(actor,batch):
    if not len(batch['z']):return sum(p.sum()*0 for p in actor.parameters())
    mean,std=gaussian_parameters(actor,batch)
    return -(torch.distributions.Normal(mean,std).log_prob(batch['z'].detach()).sum(-1)*batch['sample_weights']).sum()


def gaussian_kl(old_mean,old_std,new_mean,new_std):
    return (torch.log(new_std/old_std)+(old_std.square()+(old_mean-new_mean).square())/(2*new_std.square())-.5).sum(-1)


def protected_state(model):
    result={'num_timesteps':int(model.num_timesteps),'n_updates':int(model._n_updates)}
    for name in ('critic','critic_target'):
        module=getattr(model,name,None)
        if module is not None:result[name]=tree_digest(module.state_dict())
    for name in ('actor','critic'):
        module=getattr(model,name,None)
        if module is not None and hasattr(module,'optimizer'):result[name+'_optimizer']=tree_digest(module.optimizer.state_dict())
    for name in ('log_ent_coef','ent_coef_tensor'):
        value=getattr(model,name,None)
        if value is not None:result[name]=tree_digest(value)
    if getattr(model,'ent_coef_optimizer',None) is not None:
        result['temperature_optimizer']=tree_digest(model.ent_coef_optimizer.state_dict())
    return result


def _grads(loss,parameters,retain_graph=False,create_graph=False):
    values=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,create_graph=create_graph,allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p,g in zip(parameters,values)]


def _norm(values):return torch.sqrt(sum(v.square().sum() for v in values))


def guarded_score_update(model, episodes, demo_observations, demo_actions, retention_weight,
                         bc_weight=200., gradient_l2_clip=10., step_cap=.0003,
                         target_quadratic_kl=.0025, kl_mean_limit=.005, kl_max_limit=.05,
                         log_std_shift_max=.1, expected_case_ids=None, replicates=3):
    """One curvature-scaled plain SGD proposal; accept or exactly restore, never retry."""
    if str(model.device)!='cpu':raise ValueError('Frozen pilot uses CPU actor arithmetic')
    if retention_weight not in (0.,1.):raise ValueError('Only prospectively declared control/retention weights')
    actor=model.actor;parent=actor_digest(actor);saved=deepcopy(actor.state_dict());fixed=protected_state(model)
    batch=prepare_score_batch(episodes,parent,expected_case_ids,replicates)
    if not len(batch['z']):return {'accepted':False,'reason':'no_recorded_policy_actions','actor_before':parent,'actor_after':parent,
                                  'mixed_outcome_cases':batch['mixed_outcome_cases'],'physical_interactions':0}
    mode=actor.training;actor.eval();params=list(actor.parameters())
    accepted=False
    try:
        mu,std=gaussian_parameters(actor,batch)
        # Single-row versus batched BLAS may differ at float32 roundoff. Actor
        # identity is exact; additionally bound agreement in physical noise units.
        mean_error=float(((mu.detach()-batch['mean'])/batch['std']).abs().max())
        std_error=float((std.detach().log()-batch['std'].log()).abs().max())
        if mean_error>.01 or std_error>2e-5:
            raise ValueError('Current Gaussian differs from recorded generating distribution')
        old_mu,old_std=mu.detach().clone(),std.detach().clone()
        pg=-(torch.distributions.Normal(mu,std).log_prob(batch['z'].detach()).sum(-1)*batch['sample_weights']).sum()
        demo={k:torch.as_tensor(v,dtype=torch.float32) for k,v in demo_observations.items()}
        targets=torch.as_tensor(demo_actions,dtype=torch.float32)
        bc=(actor(demo,deterministic=True)-targets).square().mean()
        gpg=_grads(pg,params,retain_graph=True);gbc=_grads(bc,params)
        combined=[retention_weight*p+bc_weight*b for p,b in zip(gpg,gbc)]
        norm=_norm(combined)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite combined gradient')
        scale=min(1.,gradient_l2_clip/max(float(norm),1e-30))
        direction=[-scale*g.detach() for g in combined]
        kl0=(gaussian_kl(old_mu,old_std,mu,std)*batch['kl_weights']).sum()
        first=_grads(kl0,params,create_graph=True,retain_graph=True)
        dot=sum((k*d).sum() for k,d in zip(first,direction))
        hvp=_grads(dot,params)
        curvature=float(sum((h*d).sum() for h,d in zip(hvp,direction)).detach())
        if not math.isfinite(curvature) or curvature < -1e-5:raise FloatingPointError('Invalid Gaussian KL curvature')
        step=min(step_cap,math.sqrt(2*target_quadratic_kl/(max(curvature,0.)+1e-12)))
        with torch.no_grad():
            for p,d in zip(params,direction):p.add_(d,alpha=step)
            new_mu,new_std=gaussian_parameters(actor,batch)
            kl=gaussian_kl(old_mu,old_std,new_mu,new_std)
            kl_mean=float((kl*batch['kl_weights']).sum());kl_max=float(kl.max())
            std_shift=float((new_std.log()-old_std.log()).abs().max())
            mean_shift=float(((new_mu-old_mu).square().sum(-1)*batch['kl_weights']).sum().sqrt())
            action_shift=torch.tanh(new_mu)-torch.tanh(old_mu)
            action_rms=(action_shift.square()*batch['kl_weights'][:,None]).sum(0).sqrt().tolist()
        accepted=bool(all(math.isfinite(x) for x in (kl_mean,kl_max,std_shift,mean_shift))
                      and kl_mean<=kl_mean_limit and kl_max<=kl_max_limit and std_shift<=log_std_shift_max)
        if not accepted:actor.load_state_dict(saved)
        if protected_state(model)!=fixed:raise AssertionError('Auxiliary proposal changed native SAC state/Adam/temperature/counters')
        pn,bn=float(_norm(gpg)),float(_norm(gbc))
        result={'accepted':accepted,'reason':'accepted' if accepted else 'fixed_kl_or_std_guard',
                'actor_before':parent,'actor_after':actor_digest(actor),'retention_weight':retention_weight,'bc_weight':bc_weight,
                'score_loss':float(pg.detach()),'bc_loss':float(bc.detach()),'score_gradient_norm':pn,'bc_gradient_norm':bn,
                'weighted_bc_gradient_norm':bc_weight*bn,'combined_gradient_norm':float(norm),
                'score_bc_gradient_cosine':float(sum((p*b).sum() for p,b in zip(gpg,gbc)))/(pn*bn) if pn*bn else None,
                'gradient_clip_scale':scale,'directional_kl_curvature':curvature,'step_size':step,'actual_kl_mean':kl_mean,
                'actual_kl_max_perdecision':kl_max,'actual_log_std_shift_max':std_shift,'proposal_mean_shift_rms':mean_shift,
                'proposal_deterministic_action_shift_rms':action_rms,'old_std_min':float(old_std.min()),'old_std_max':float(old_std.max()),
                'executed_action_exact_saturation_fraction':float((batch['actions'].abs()==1.).float().mean()),
                'generating_mean_roundoff_in_std_units_max':mean_error,'generating_log_std_roundoff_max':std_error,
                'mixed_outcome_cases':batch['mixed_outcome_cases'],'episodes':batch['episodes'],'actual_grasp_decisions':len(batch['z']),
                'protected_state':fixed,'native_sac_updates_added':0,'physical_interactions':0,'proposals':1,'backtracking':False}
        return result
    except BaseException:
        actor.load_state_dict(saved)
        raise
    finally:actor.train(mode)
