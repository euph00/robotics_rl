#!/usr/bin/env python3
"""Offline matched BC: inherited90k carry actor versus fresh mean-network.

Same merged original/correction data, original goals, saved episode split,
all-velocity mask, and identical minibatches. Critics and log_std head remain
unchanged. No physics or trainable SAC/actor output checkpoint is produced.
"""
from __future__ import annotations
import argparse
import copy
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch
from pnr.policies import load_policy_checkpoint
from pnr.telemetry import RunLogger,atomic_json

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_sha(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='carry_actor_prior_diagnosis01')
    parser.add_argument('--source-run',default='carry_bc_corrections01')
    parser.add_argument('--updates',type=int,default=2000)
    args=parser.parse_args()
    if args.updates!=2000:raise ValueError('This comparison is bounded to2000 updates per actor')
    torch.set_num_threads(1);torch.manual_seed(7410)
    source=ROOT/'runs'/args.source_run
    config=json.loads((source/'config.json').read_text())
    origin=source/'artifacts/inputs/source_model.zip'
    demo=Path(config['demonstrations']);label_path=Path(config['sampling_labels'])
    original_hashes={'model':sha(origin),'demonstrations':sha(demo),'sampling_labels':sha(label_path)}
    if original_hashes['demonstrations']!=config['demonstrations_sha256']:
        raise ValueError('Original demonstration data changed')
    logger=RunLogger(args.run_id,kind='diagnostic',skill='carry',name='Carry · offline inherited versus fresh actor',
        algorithm='Matched offline BC initialization',summary='Two frozen actor copies; identical batches; no physics or online SAC.')
    artifacts=logger.directory/'artifacts';inputs=artifacts/'inputs';inputs.mkdir(parents=True)
    for src,name in [(origin,'source_model.zip'),(demo,'demonstrations.pkl'),(label_path,'sampling_labels.npz')]:
        shutil.copy2(src,inputs/name)
    shutil.copy2(__file__,artifacts/'diagnose_carry_actor_prior.py')
    shutil.copy2(source/'config.json',inputs/'source_config.json')
    dataset=pickle.loads((inputs/'demonstrations.pkl').read_bytes())
    labels=np.load(inputs/'sampling_labels.npz',allow_pickle=False)
    if str(labels['dataset_sha256'])!=original_hashes['demonstrations']:
        raise ValueError('Sampling labels do not describe the frozen dataset')
    split=config['split']
    records=[record for episode in dataset['episodes'] for record in episode['transitions']]
    episode_ids=np.asarray(labels['episode_id']);phases=np.asarray(labels['phase']);strata_labels=np.asarray(labels['stratum'])
    if not np.array_equal(episode_ids,[record['episode'] for record in records]):raise ValueError('Sampling labels no longer align with records')
    training=np.isin(episode_ids,split['train_episode_ids']);heldout=np.isin(episode_ids,split['validation_episode_ids'])
    if np.any(training&heldout) or not np.all(training|heldout):raise ValueError('Invalid frozen original episode split')
    if set(labels['group_id'][training])&set(labels['group_id'][heldout]):raise ValueError('Frozen split contains group leakage')
    groups={};validation_groups={}
    for key,count in split['stratum_train_counts'].items():
        stratum,phase=key.split(':');mask=(strata_labels==stratum)&(phases==phase)
        groups[key]=np.flatnonzero(mask&training)
        if len(groups[key])!=count:raise ValueError('Original training stratum no longer matches:'+key)
    for key,count in split['stratum_validation_counts'].items():
        stratum,phase=key.split(':');mask=(strata_labels==stratum)&(phases==phase)
        validation_groups[key]=np.flatnonzero(mask&heldout)
        if len(validation_groups[key])!=count:raise ValueError('Original validation stratum no longer matches:'+key)
    observations={key:np.stack([record['observation'][key] for record in records]).astype(np.float32) for key in records[0]['observation']}
    actions=np.stack([record.get('teacher_action',record['action']) for record in records]).astype(np.float32)
    strata=tuple(dict.fromkeys(key.split(':')[0] for key in groups))
    phase_counts=config['sampling_per_stratum']
    switches=np.zeros(len(records),dtype=bool);previous=np.full_like(actions,np.nan)
    norms=np.linalg.norm(actions,axis=1)
    for i in range(1,len(records)):
        if episode_ids[i]!=episode_ids[i-1] or min(norms[i],norms[i-1])<.1:continue
        previous[i]=actions[i-1]
        angle=np.degrees(np.arccos(np.clip(np.dot(actions[i],actions[i-1])/(norms[i]*norms[i-1]),-1.,1.)))
        switches[i]=angle>=45.
    slices={'train_all':training,'heldout_all':heldout,
        'train_exact_switch':training&switches,'heldout_exact_switch':heldout&switches,
        'train_straight':training&(phases=='straight')&~switches,
        'heldout_straight':heldout&(phases=='straight')&~switches,
        'train_turn_window':training&(phases=='turn_approach'),'heldout_turn_window':heldout&(phases=='turn_approach')}
    tensors={key:torch.as_tensor(value) for key,value in observations.items()};targets=torch.as_tensor(actions)
    from pnr.carry_features import CarryWorldGeometryExtractor,transfer_carry_geometry_policy
    original=load_policy_checkpoint(inputs/'source_model.zip',device='cpu')
    if (original.num_timesteps,original._n_updates)!=(config['source_online_transitions'],config['source_sac_updates']):
        raise ValueError('Frozen parent counters differ from source configuration')
    arguments=original.policy._get_constructor_parameters()
    arguments.update(features_extractor_class=CarryWorldGeometryExtractor,
        features_extractor_kwargs=dict(original.policy_kwargs['features_extractor_kwargs']))
    expanded=original.policy.__class__(**arguments)
    transfer=transfer_carry_geometry_policy(original.policy,expanded)
    check={key:value[:128] for key,value in tensors.items()};check_actions=targets[:128]
    errors=[]
    for relabel in (False,True):
        if relabel:check['desired_goal']=check['desired_goal'].roll(19,dims=0)+.03
        with torch.no_grad():
            pairs=[(original.actor(check,deterministic=True),expanded.actor(check,deterministic=True))]
            pairs+=list(zip(original.critic(check,check_actions),expanded.critic(check,check_actions)))
            pairs+=list(zip(original.critic_target(check,check_actions),expanded.critic_target(check,check_actions)))
        for before,after in pairs:
            torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-5)
            errors.append(float((before-after).abs().max()))
    transfer['pre_mask_original_and_HER_prediction_max_error']=max(errors)
    expanded.actor.features_extractor.set_all_velocity_mask(True)
    models={}
    for name in ('A_inherited_90k','B_fresh_mean'):
        policy=copy.deepcopy(expanded)
        models[name]=SimpleNamespace(actor=policy.actor,critic=policy.critic,critic_target=policy.critic_target,
            num_timesteps=original.num_timesteps,_n_updates=original._n_updates)
    torch.manual_seed(7410)
    for layer in list(models['B_fresh_mean'].actor.latent_pi.modules())+[models['B_fresh_mean'].actor.mu]:
        if isinstance(layer,torch.nn.Linear):layer.reset_parameters()
    expected=torch.ones(580);expected[543:558]=0.;expected[572:578]=0.
    for model in models.values():
        torch.testing.assert_close(model.actor.features_extractor.state_mask,expected,atol=0,rtol=0)
    del original,expanded
    original_model_counters={name:(model.num_timesteps,model._n_updates) for name,model in models.items()}
    invariants={name:{'critic':weights_sha(model.critic),'critic_target':weights_sha(model.critic_target),
                      'log_std':weights_sha(model.actor.log_std)} for name,model in models.items()}
    optimizers={name:torch.optim.Adam(list(model.actor.latent_pi.parameters())+list(model.actor.mu.parameters()),lr=.0003) for name,model in models.items()}
    def measure(model):
        with torch.no_grad():pred=model.actor(tensors,deterministic=True).cpu().numpy()
        squared=((pred-actions)**2).mean(axis=1)
        result={}
        for name,mask in slices.items():
            total=int(mask.sum());row={'samples':total,'mse':float(squared[mask].mean()) if total else None}
            moving=mask&(norms>=.1)
            if moving.any():
                p=pred[moving];a=actions[moving]
                cosine=np.clip(np.sum(p*a,axis=1)/(np.linalg.norm(p,axis=1).clip(1e-8)*np.linalg.norm(a,axis=1)),-1.,1.)
                row['mean_heading_error_deg']=float(np.degrees(np.arccos(cosine)).mean())
            if 'exact_switch' in name and total:
                error_old=((pred[mask]-previous[mask])**2).mean(axis=1)
                row['closer_to_previous_action_fraction']=float((error_old<squared[mask]).mean())
            result[name]=row
        result['heldout_equal_stratum_mse']=float(np.mean([squared[indices].mean() for indices in validation_groups.values()]))
        result['train_equal_stratum_mse']=float(np.mean([squared[indices].mean() for indices in groups.values()]))
        return result,pred
    protocol={'source_run':args.source_run,'source_online_transitions':config['source_online_transitions'],'source_generation':config['source_generation'],'geometry_transfer':transfer,
        'source_hashes':original_hashes,'original_split_config_sha256':sha(inputs/'source_config.json'),
        'updates_per_copy':args.updates,'batch_size':400,'learning_rate':.0003,'sampling_seed':config['seed'],
        'split':split,'both_masks_zero_ranges':[[543,558],[572,578]],
        'fresh_initialization':'PyTorch Linear.reset_parameters on latent_pi and mu only; seed7410',
        'log_std_caveat':'Head parameters remain unchanged, but resetting/training its shared latent backbone changes log_std outputs. Deterministic labels ignore those outputs; no stochastic-policy readiness claim.',
        'teacher_labels':'teacher_action where present, actual action otherwise; actual replay actions are never changed',
        'exact_switch_definition':'First current-action sample whose teacher heading differs >=45deg from preceding sample in the same episode; both norms>=0.1. Diagnostic label only.',
        'original_goals_only':True,'HER_supervised_relabeling':False,'new_physics_calls':0,'new_online_SAC_transitions':0,
        'training_artifacts':'No actor/SAC output ZIPs are published; frozen input ZIP is archival only.',
        'limitations':'Same source actor and critics, one object/maze, correlated full-route repetitions, reused merged BC train/validation split. Single fresh initialization seed. No physical success or independent generalization claim.'}
    atomic_json(logger.directory/'config.json',protocol)
    initial={name:measure(model)[0] for name,model in models.items()}
    atomic_json(artifacts/'initial.json',initial)
    history=[];rng=np.random.default_rng(config['seed']);batch_digest=hashlib.sha256();started=time.perf_counter()
    for update in range(1,args.updates+1):
        batch=np.concatenate([rng.choice(groups[f'{stratum}:{phase}'],count,replace=True)
            for stratum in strata for phase,count in phase_counts.items()])
        batch_digest.update(batch.astype(np.int64).tobytes())
        obs={key:value[batch] for key,value in tensors.items()}
        for name,model in models.items():
            actor=model.actor;prediction=actor(obs,deterministic=True)
            loss=torch.nn.functional.mse_loss(prediction,targets[batch]);optimizer=optimizers[name]
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.latent_pi.parameters())+list(actor.mu.parameters()),5.)
            optimizer.step()
        if update%200==0:
            values={name:measure(model)[0] for name,model in models.items()}
            history.append({'updates_per_copy':update,'metrics':values})
            atomic_json(artifacts/'progress.json',{'history':history})
            logger.metric(0,scope='offline_imitation_diagnostic',metric_source='matched_supervised_updates',
                updates_per_copy=update,new_physics_calls=0,new_online_SAC_transitions=0,
                heldout_A_mse=values['A_inherited_90k']['heldout_equal_stratum_mse'],
                heldout_B_mse=values['B_fresh_mean']['heldout_equal_stratum_mse'])
            logger.update(summary=f'Matched offline actor initialization {update}/{args.updates} updates each; no physical evaluation.')
            print(json.dumps({'updates_per_copy':update,'heldout_A':values['A_inherited_90k']['heldout_equal_stratum_mse'],
                'heldout_B':values['B_fresh_mean']['heldout_equal_stratum_mse']}),flush=True)
    final={};predictions={}
    for name,model in models.items():
        final[name],predictions[name]=measure(model)
        actual={'critic':weights_sha(model.critic),'critic_target':weights_sha(model.critic_target),'log_std':weights_sha(model.actor.log_std)}
        if actual!=invariants[name] or (model.num_timesteps,model._n_updates)!=original_model_counters[name]:
            raise RuntimeError('Offline actor updates changed critic/log_std weights or SAC counters')
    if original_hashes!={'model':sha(origin),'demonstrations':sha(demo),'sampling_labels':sha(label_path)}:
        raise RuntimeError('Source inputs changed during the comparison')
    report={'protocol':protocol,'initial':initial,'final':final,'shared_minibatches_sha256':batch_digest.hexdigest(),
        'critics_unchanged':True,'log_std_head_weights_unchanged':True,'source_artifacts_unchanged':True,
        'new_physics_calls':0,'new_online_SAC_transitions':0,'wall_seconds':time.perf_counter()-started,'history':history}
    atomic_json(artifacts/'result.json',report)
    np.savez_compressed(artifacts/'predictions.npz',teacher_actions=actions,episode_id=episode_ids,
        is_training=training,is_exact_switch=switches,phase=phases,**predictions)
    logger.artifact('artifacts/result.json','Matched losses, split, invariants and provenance','file')
    logger.finish('complete','Matched2000-update copies completed; physical navigation remains untested by this diagnostic.')
    print(json.dumps({'final':final,'artifacts':str(artifacts)},indent=2))


if __name__=='__main__':main()
