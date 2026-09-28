#!/usr/bin/env python3
"""Paired, physics-free critic initialization test; never resumes online learning."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from pnr.multi_carry import CarrySchema
from pnr.policies import load_policy_checkpoint
from pnr.sac_initialization import critic_only_warmup
from scripts.phase2_critic_return_audit import dataset,prepare,evaluate


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def module_hash(module):
    digest=hashlib.sha256()
    for key,value in module.state_dict().items():
        tensor=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def temperature_hash(model):
    digest=hashlib.sha256()
    for name in ('log_ent_coef','ent_coef_tensor'):
        value=getattr(model,name,None)
        if value is not None:digest.update(name.encode()+value.detach().cpu().numpy().tobytes())
    # A fresh source has no temperature optimizer steps; ensure it stays so.
    digest.update(str(len(model.ent_coef_optimizer.state)).encode())
    return digest.hexdigest()


@contextmanager
def preserve_rng():
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:yield
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
        if states[3] is not None:torch.cuda.set_rng_state_all(states[3])


class SampleAudit:
    """Read-only wrapper verifies identical real/HER sampled tensors across arms."""
    def __init__(self,replay):
        self.replay=replay;self.digest=hashlib.sha256();self.calls=0
    def sample(self,*args,**kwargs):
        batch=self.replay.sample(*args,**kwargs)
        for group in (batch.observations,batch.next_observations):
            for key in sorted(group):self.digest.update(group[key].detach().cpu().numpy().tobytes())
        for value in (batch.actions,batch.rewards,batch.dones):
            self.digest.update(value.detach().cpu().numpy().tobytes())
        self.calls+=1
        return batch


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=9147)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--taus',type=float,nargs='+',default=[.01,.1,1.])
    parser.add_argument('--checkpoints',type=int,nargs='+',default=[500,1000,2000])
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory')
    if sorted(set(args.checkpoints))!=args.checkpoints or args.checkpoints[0]<1:parser.error('Checkpoints must increase')
    if any(tau<=0 or tau>1 for tau in args.taus):parser.error('Invalid target tau')
    args.output.mkdir(parents=True);(args.output/'checkpoints').mkdir()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started=time.monotonic()
    source=ROOT/'runs/phase2_carry_recovery01/checkpoints/bc_initialized.zip'
    replay_path=next((ROOT/'runs/phase2_carry_recovery01/checkpoints').glob('step_000000000*/replay.pkl'))
    episodes,sources=dataset();states,finals=prepare(episodes)
    report=dict(protocol='paired_fixed_actor_critic_warmup_v1',status='running',device=args.device,cpu_threads=4,
        gpu_numerics='cuDNN benchmark enabled, deterministic disabled, float32 matmul high; not bitwise reproducible',
        physics_ticks=0,source_action_calls=0,actor_updates=0,temperature_updates=0,
        source_checkpoint=str(source.relative_to(ROOT)),source_checkpoint_sha256=file_hash(source),
        source_replay=str(replay_path.relative_to(ROOT)),source_replay_sha256=file_hash(replay_path),
        source_demonstrations=sources,training_probe_episodes=len(episodes),training_probe_states=len(states),
        random_seed=args.seed,taus=args.taus,checkpoints=args.checkpoints,
        settings=dict(batch_size=256,critic_learning_rate=.0003,gamma=.995,online_tau_unchanged=.01),
        caveats=['Uses a frozen actor, fixed temperature, existing physical TRAIN replay and ordinary SAC/HER critic targets.',
            'Observed demonstration returns are a feasible hard-return reference, not exact soft Q for the frozen stochastic actor.',
            'No online learning, no physics, no new validation, no production checkpoint changes.',
            'Each checkpoint evaluation restores all CPU/CUDA RNG states before the next update; sample tensor hashes must match between arms.',
            'Only temporary warmup target tau differs; saved models retain the original online tau=.01.'],arms=[])
    write_json(args.output/'result.json',report)
    print('Loading frozen replay',flush=True)
    with replay_path.open('rb') as stream:replay=pickle.load(stream)
    schema=DummyVecEnv([lambda:CarrySchema() for _ in range(replay.n_envs)])
    assert replay.env is None
    replay.set_env(schema);replay.device=torch.device(args.device)
    replay_counters=(replay.pos,replay.full,int(np.sum(replay.ep_length)),int(np.sum(replay.dones)))
    report['replay']=dict(n_envs=replay.n_envs,buffer_size=replay.buffer_size,pos=replay.pos,full=replay.full,
        valid_transitions=int(np.count_nonzero(replay.ep_length)),her_ratio=replay.her_ratio)
    report['fresh_critic_probe']=evaluate('fresh_critics',source,states,finals)
    initial_hashes=None;sample_hashes={}
    for tau in args.taus:
        arm=dict(warmup_tau=tau,status='running',probes=[],update_logs=[],chunks=[])
        report['arms'].append(arm)
        # Tiny inference allocation is immediately replaced by the saved HER replay;
        # this isolated model is used only by critic_only_warmup, never by learn().
        model=load_policy_checkpoint(source,device=args.device)
        torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision('high')
        assert model.num_timesteps==0 and model._n_updates==0
        assert getattr(model,'_pnr_critic_warmup_updates',0)==0
        assert model.gamma==.995 and model.tau==.01 and model.target_update_interval==1
        assert not model.critic.optimizer.state and not model.ent_coef_optimizer.state
        sampled=SampleAudit(replay);model.replay_buffer=sampled
        model._vec_normalize_env=None
        hashes=dict(actor=module_hash(model.actor),critic=module_hash(model.critic),
                    target=module_hash(model.critic_target),temperature=temperature_hash(model))
        if initial_hashes is None:initial_hashes=hashes
        assert hashes==initial_hashes
        arm['initial_hashes']=hashes
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
        completed=0
        try:
            for checkpoint in args.checkpoints:
                model.tau=tau
                def log(row):
                    row=dict(row,cumulative_updates=completed+row['critic_updates'],warmup_tau=tau)
                    arm['update_logs'].append(row)
                    print(json.dumps(row),flush=True)
                chunk=critic_only_warmup(model,checkpoint-completed,batch_size=256,
                                         learning_rate=.0003,log_every=250,on_log=log)
                model.tau=.01
                arm['chunks'].append(chunk)
                completed=checkpoint
                after=dict(actor=module_hash(model.actor),critic=module_hash(model.critic),
                    target=module_hash(model.critic_target),temperature=temperature_hash(model))
                assert after['actor']==hashes['actor'] and after['temperature']==hashes['temperature']
                assert after['critic']!=hashes['critic'] and after['target']!=hashes['target']
                assert model.num_timesteps==0 and model._n_updates==0
                assert (replay.pos,replay.full,int(np.sum(replay.ep_length)),int(np.sum(replay.dones)))==replay_counters
                sample_hash=sampled.digest.hexdigest()
                if checkpoint in sample_hashes:assert sample_hashes[checkpoint]==sample_hash
                else:sample_hashes[checkpoint]=sample_hash
                path=args.output/'checkpoints'/f'tau_{tau:g}_updates_{checkpoint:04d}.zip'
                with preserve_rng():
                    model.save(path)
                    probe=evaluate(f'tau_{tau:g}_updates_{checkpoint}',path,states,finals)
                probe.update(warmup_tau=tau,parameter_hashes=after,sampled_tensor_sha256=sample_hash,
                             sample_calls=sampled.calls)
                arm['probes'].append(probe)
                print(json.dumps({'event':'probe','warmup_tau':tau,'updates':checkpoint,
                    'long_horizon':probe['remaining_decision_bins'].get('201_to_10000_remaining_decisions'),
                    'actor_ranked_above_expert':probe['summary']['fraction_actor_ranked_above_expert']}),flush=True)
                write_json(args.output/'result.json',report)
                if max(abs(probe['summary']['q_expert']['min']),abs(probe['summary']['q_expert']['max']))>10000:
                    raise FloatingPointError('Critic magnitude exceeded diagnostic safety threshold10000')
            arm['status']='completed'
        except Exception as exc:
            arm['status']='failed';arm['error']=repr(exc)
            print(json.dumps({'event':'arm_failed','tau':tau,'error':repr(exc)}),flush=True)
        finally:
            model.tau=.01
            arm['critic_updates_completed']=int(getattr(model,'_pnr_critic_warmup_updates',0))
            arm['actor_unchanged']=module_hash(model.actor)==hashes['actor']
            arm['temperature_unchanged']=temperature_hash(model)==hashes['temperature']
            del model,sampled;gc.collect()
        write_json(args.output/'result.json',report)
    report['total_critic_updates']=sum(arm['critic_updates_completed'] for arm in report['arms'])
    report['status']='completed' if all(arm['status']=='completed' for arm in report['arms']) else 'completed_with_failures'
    report['wall_seconds']=time.monotonic()-started
    report['source_checkpoint_unchanged']=file_hash(source)==report['source_checkpoint_sha256']
    report['source_replay_unchanged']=file_hash(replay_path)==report['source_replay_sha256']
    assert report['source_checkpoint_unchanged'] and report['source_replay_unchanged']
    write_json(args.output/'result.json',report);schema.close()
    print(json.dumps({k:report[k] for k in ('status','total_critic_updates','wall_seconds','source_checkpoint_unchanged','source_replay_unchanged')}),flush=True)


if __name__=='__main__':main()
