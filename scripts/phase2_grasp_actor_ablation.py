#!/usr/bin/env python3
"""Offline TRAIN-only full70 versus compact28 grasp actor comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def worker(task):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from pnr.general_manipulation import GraspSchema,GeneralGraspFeatures
    from pnr.grasp_actor_ablation import CompactGraspActorPolicy
    from pnr.telemetry import atomic_json
    from scripts.phase2_train_grasp import dataset_rows,action_fit
    variant,seed,config=task;torch.set_num_threads(1)
    np.random.seed(seed);torch.manual_seed(seed)
    dataset=pickle.loads(Path(config['dataset']).read_bytes())
    held={tuple(key) for key in config['holdout_cases']}
    partitions={}
    for name,is_holdout in [('train',False),('train_pointseed_holdout',True)]:
        episodes=[e for e in dataset['episodes'] if ((e['summary']['object_idx'],e['summary']['scene_seed']) in held)==is_holdout]
        rows,groups,meta=dataset_rows({'episodes':episodes})
        obs={key:torch.as_tensor(np.stack([r['observation'][key] for r in rows])) for key in GraspSchema().observation_space.spaces}
        actions=torch.as_tensor(np.stack([r['action'] for r in rows]))
        partitions[name]=(obs,actions,groups,meta)
    schema=DummyVecEnv([GraspSchema])
    model=SAC(CompactGraspActorPolicy if variant=='compact28' else 'MultiInputPolicy',schema,
        policy_kwargs={'net_arch':[128,128],'features_extractor_class':GeneralGraspFeatures},
        learning_rate=3e-4,buffer_size=1,device='cpu',seed=seed)
    assert model.critic.features_extractor.features_dim==70
    assert model.actor.features_extractor.features_dim==(28 if variant=='compact28' else 70)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=3e-4);rng=np.random.default_rng(seed)
    obs,actions,groups,_=partitions['train'];strata=list(groups.values());evaluations=[]
    out=Path(config['output']);name=f'{variant}_seed{seed}'
    for update in range(1,config['updates']+1):
        group_ids=rng.integers(0,len(strata),size=config['batch_size'])
        indices=np.array([rng.choice(strata[group]) for group in group_ids])
        batch={key:value[indices] for key,value in obs.items()}
        predicted=model.actor(batch,deterministic=True);loss=torch.mean((predicted-actions[indices])**2)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),10.);optimizer.step()
        if update in (1000,2000,config['updates']):
            fits={}
            for partition,(values,labels,cells,meta) in partitions.items():
                fit=action_fit(model,values,labels,cells)
                fit['balanced_group_mse']=float(np.mean([r['mean_squared_action_error'] for r in fit['groups'].values()]))
                with torch.no_grad():predictions=torch.cat([model.actor({k:v[start:start+1024] for k,v in values.items()},deterministic=True) for start in range(0,len(labels),1024)])
                raw=values['observation'];near=(torch.linalg.vector_norm(raw[:,50:53],dim=1)<.04)&(raw[:,48:50].sum(dim=1)<2.)
                open_near=near&(labels[:,3]>0)
                fit['near_contact_examples']=int(near.sum())
                fit['near_contact_sign_error']=float(((predictions[near,3]>0)!=(labels[near,3]>0)).float().mean()) if near.any() else None
                fit['premature_close_fraction']=float((predictions[open_near,3]<0).float().mean()) if open_near.any() else None
                fits[partition]=fit
            evaluations.append({'update':update,'partitions':fits})
            atomic_json(out/(name+'.json'),{'variant':variant,'seed':seed,'evaluations':evaluations,'physical_interactions':0,
                'validation_bank_accessed':False,'critic_features':70,'actor_features':model.actor.features_extractor.features_dim})
    model.save(out/name);schema.close()
    return {'variant':variant,'seed':seed,'evaluations':evaluations,'physical_interactions':0,
            'supervised_examples':config['updates']*config['batch_size']}


def main():
    from pnr.telemetry import RunLogger,atomic_json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--updates',type=int,default=4000)
    args=parser.parse_args()
    if not 1<=args.updates<=8000:raise ValueError('Bounded offline budget is at most8k updates per actor')
    dataset=pickle.loads(args.dataset.read_bytes());cases=sorted({(e['summary']['object_idx'],e['summary']['scene_seed']) for e in dataset['episodes']})
    if any(seed in (43001,43002) for _,seed in cases):raise ValueError('Development validation states must not enter this study')
    held=[]
    for obj in sorted({obj for obj,_ in cases}):
        selected=[key for key in cases if key[0]==obj]
        held.extend(key for index,key in enumerate(selected) if index%3==2)
    logger=RunLogger(args.run_id,kind='train',skill='grasp',budget=0,name='TRAIN-only grasp actor feature ablation',
        algorithm='Offline balanced behavior cloning',summary='Full70 versus compact28 actor, two fresh seeds, held-out TRAIN point seeds; zero simulator interactions.')
    out=logger.directory/'artifacts';out.mkdir()
    for path in [Path(__file__),ROOT/'pnr/grasp_actor_ablation.py',ROOT/'pnr/general_manipulation.py',ROOT/'pnr/features.py']:
        (out/path.name).write_bytes(path.read_bytes())
    config={'dataset':str(args.dataset),'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'output':str(out),'updates':args.updates,'batch_size':400,'seeds':[8101,8102],'variants':['full70','compact28'],
        'holdout_cases':held,'training_cases':[key for key in cases if key not in held],
        'split_rule':'Every third sorted point seed within each object is an offline TRAIN holdout.',
        'physical_interactions':0,'original_actions_unchanged':True,'critic_features_unchanged':70,
        'no_pretrained_actor_transfer':True,'validation_bank_accessed':False}
    atomic_json(logger.directory/'config.json',config);results=[]
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(worker,(variant,seed,config)) for variant in config['variants'] for seed in config['seeds']]):
                results.append(future.result());atomic_json(out/'summary.json',{'configuration':config,'results':results})
                logger.metric(0,scope='offline_training_feature_ablation',models_completed=len(results),
                    supervised_examples=sum(r['supervised_examples'] for r in results),physical_interactions=0)
        logger.artifact('artifacts/summary.json','Offline TRAIN point-seed holdout action fit; no physical success claim')
        logger.finish(summary='Completed four matched-budget offline actor fits; review TRAIN holdout results before any physical candidate evaluation.')
    except BaseException as error:logger.finish('failed',repr(error));raise

if __name__=='__main__':main()
