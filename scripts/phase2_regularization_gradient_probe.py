#!/usr/bin/env python3
"""Read-only gradient balance on saved carry initialization and physical demos."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from pnr.demonstrations import seed_replay
from pnr.multi_carry import CarrySchema
from pnr.regularized_sac import RegularizedSAC


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill',choices=['carry','grasp'],default='carry')
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/phase2_carry_spatial02/checkpoints/initialized.zip')
    parser.add_argument('--dataset',type=Path,default=ROOT/'runs/phase2_carry_dataset64/artifacts/demonstrations.pkl')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/phase2_carry_trace_audit01/artifacts/gradient_balance.json')
    args=parser.parse_args();started=time.monotonic()
    torch.set_num_threads(2);np.random.seed(42081);torch.manual_seed(42081)
    if args.skill=='grasp':
        from pnr.general_manipulation import GraspSchema
        from scripts.phase2_train_grasp import dataset_rows
        schema=DummyVecEnv([GraspSchema for _ in range(8)])
    else:
        schema=DummyVecEnv([CarrySchema for _ in range(8)])
    model=RegularizedSAC.load(args.checkpoint,env=schema,device='cpu',buffer_size=100000)
    model.bc_weight=10.;model.demo_batch_size=args.batch_size
    seeded=seed_replay(model,args.dataset)
    dataset=pickle.loads(args.dataset.read_bytes())
    if args.skill=='grasp':
        rows,groups,_=dataset_rows(dataset)
        probabilities=None
    else:
        rows=[row for episode in dataset['episodes'] for row in episode['transitions']]
        labels=np.stack([row['teacher_action'] for row in rows]);norm=np.linalg.norm(labels,axis=1)
        groups={'turn_or_vertical':np.flatnonzero((norm>.5)&(np.abs(labels[:,1])>.15)),
                'horizontal':np.flatnonzero((norm>.5)&(np.abs(labels[:,1])<=.15)),
                'hold':np.flatnonzero(norm<=.5)}
        probabilities=[.5,.35,.15]
    observations={key:np.stack([row['observation'][key] for row in rows]) for key in model.observation_space.spaces}
    action_key='action' if args.skill=='grasp' else 'teacher_action'
    labels=np.stack([row[action_key] for row in rows])
    manifest=model.set_demonstrations(observations,labels,groups,group_probabilities=probabilities,seed=42082)
    model.policy.set_training_mode(True)
    records=[]
    for index in range(8):
        batch=model.replay_buffer.sample(args.batch_size)
        action,logp=model.actor.action_log_prob(batch.observations)
        q=torch.cat(model.critic(batch.observations,action),dim=1).min(dim=1).values
        alpha=model.log_ent_coef.detach().exp()
        rl=(alpha*logp-q).mean()
        obs,target=model._sample_demonstrations(args.batch_size)
        bc=(model.actor(obs,deterministic=True)-target).square().mean()
        record=model._gradient_metrics(rl,bc)
        record.update(batch=index,rl_loss=float(rl.detach()),bc_loss=float(bc.detach()),
                      bc_to_rl_grad_ratio_weight100=10*record['bc_to_rl_grad_ratio'])
        records.append(record)
    summary={key:{'mean':float(np.mean([r[key] for r in records])),
                  'min':float(min(r[key] for r in records)),'max':float(max(r[key] for r in records))}
             for key in ('actor_rl_grad_norm','actor_bc_grad_norm','bc_to_rl_grad_ratio',
                         'bc_to_rl_grad_ratio_weight100','rl_bc_grad_cosine','bc_loss')}
    result={'method':'offline saved-initializer gradients; no updates or physics',
            'physical_interactions':0,'gradient_updates':0,'skill':args.skill,'demo_batches':8,'batch_size':args.batch_size,'seed':42081,
            'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'checkpoint_online_decisions':model.num_timesteps,'demo_manifest':manifest,
            'seeded_replay':seeded,'summary':summary,'records':records,'wall_seconds':time.monotonic()-started,
            'limits':['Raw parameter gradients precede Adam and do not predict success.',
                      'These gradients use the saved warmed critic; future fresh critics may differ.',
                      'BC has near-zero restoring gradient at exact agreement, so fixed regularization permits some drift.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'summary':summary,'wall_seconds':result['wall_seconds']},indent=2))
    schema.close()


if __name__=='__main__':main()
