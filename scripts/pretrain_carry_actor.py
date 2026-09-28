#!/usr/bin/env python3
"""Bounded, disclosed carry actor initialization using original demonstration goals.

This performs supervised actor-mean updates only. Physical full-route checks use
one fixed exit goal and the learned actor, with no teacher waypoints or resets
inside an episode. Critics, entropy parameters, and SAC counters are preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import SAC,HerReplayBuffer
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.carry import CarrySkillEnv
from pnr.telemetry import RunLogger,atomic_json

PHASE_COUNTS={'turn_approach':36,'straight':36,'goal_hold':8}
STRATA=('segment_1.2','segment_2.5','segment_5.0','segment_10.0','full_route')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()



def merged_training_data(original,corrections,source_records):
    """Merge real transitions without changing executed actions or fixed goals."""
    datasets=[original,corrections]
    if any(data.get('schema_version')!=1 or not data['metadata'].get('metadata_finalized') for data in datasets):
        raise ValueError('Combined BC requires two finalized physical schema1 datasets')
    for data in datasets:
        if (data['metadata'].get('skill'),data['metadata'].get('scene_seed'),data['metadata'].get('object_idx'))!=('carry',42,14):
            raise ValueError('Correction dataset does not match the declared physical carry scene')
    episodes=[]
    for source_index,data in enumerate(datasets):
        for source_episode in data['episodes']:
            episode=copy.deepcopy(source_episode)
            index=len(episodes)
            episode['summary'].update(source_dataset=source_records[source_index]['name'],
                source_episode=episode['summary']['episode'],episode=index)
            goal=episode['transitions'][0]['observation']['desired_goal']
            for record in episode['transitions']:
                record['source_episode']=record['episode'];record['episode']=index
                record['teacher_action']=np.asarray(record.get('teacher_action',record['action']),dtype=np.float32).copy()
                if not (np.array_equal(record['observation']['desired_goal'],goal)
                        and np.array_equal(record['next_observation']['desired_goal'],goal)):
                    raise ValueError('A physical episode changed its desired goal')
            episodes.append(episode)
    metadata={'metadata_finalized':True,'collection_status':'complete','training_only':True,
        'skill':'carry','scene_seed':42,'object_idx':14,'horizon':1000,'failure_cost':200.,
        'scene':original['metadata']['scene'],'source_datasets':source_records,
        'policy_decisions':sum(len(episode['transitions']) for episode in episodes),
        'fixed_goal_per_episode':True,'teacher_at_learned_inference':False,
        'action_contract':'action is the actual command passed to the unchanged carry adapter; teacher_action is a separate BC label only.',
        'label_contract':'Original desired goals only. No HER-generated supervision. All actual termination/truncation flags retained.',
        'correlation_notice':'One provided object/maze; all correlated full-route repeats across both datasets stay in the training split.'}
    for field in ('demonstration_source_action_calls','demonstration_physics_ticks',
                  'setup_source_action_calls','setup_physics_ticks','initialization_physics_ticks'):
        metadata[field]=sum(int(data['metadata'][field]) for data in datasets)
    for field in ('preflight_source_action_calls','preflight_physics_ticks'):
        metadata[field]=sum(int(data['metadata'].get(field,0)) for data in datasets)
    metadata['total_source_action_calls']=metadata['demonstration_source_action_calls']+metadata['setup_source_action_calls']+metadata['preflight_source_action_calls']
    metadata['total_physics_ticks']=metadata['demonstration_physics_ticks']+metadata['setup_physics_ticks']+metadata['initialization_physics_ticks']+metadata['preflight_physics_ticks']
    return {'schema_version':1,'kind':'physical_training_demonstrations','training_only':True,
        'teacher_version':'original-and-mixed-policy-carry-corrections-v1','metadata':metadata,'episodes':episodes}


def sampling_labels_for_dataset(dataset,digest):
    """Use measured teacher turns only for training strata, never policy inputs."""
    phases=[];strata=[];episode_ids=[];groups=[]
    for episode in dataset['episodes']:
        summary=episode['summary'];records=episode['transitions']
        actions=np.asarray([record.get('teacher_action',record['action']) for record in records])
        bases=np.asarray([record['observation']['achieved_goal'] for record in records])*20.
        arcs=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(bases,axis=0),axis=1))]
        norms=np.linalg.norm(actions,axis=1);events=[]
        explicit=all('teacher_waypoint_index' in record for record in records)
        for i in range(1,len(records)):
            if explicit:
                changed=records[i]['teacher_waypoint_index']!=records[i-1]['teacher_waypoint_index']
            else:
                changed=False
                if min(norms[i-1],norms[i])>=.1:
                    angle=np.degrees(np.arccos(np.clip(np.dot(actions[i-1],actions[i])/(norms[i-1]*norms[i]),-1,1)))
                    changed=angle>=45.
            if changed and (not events or arcs[i]-arcs[events[-1]]>.2):events.append(i)
        labels=np.full(len(records),'straight',dtype='<U16')
        for index in events:labels[(arcs>=arcs[index]-.4)&(arcs<=arcs[index]+.2)]='turn_approach'
        for index,record in enumerate(records):
            if record['teacher_phase']=='retained_goal_hold':labels[index]='goal_hold'
        stratum='full_route' if summary['kind']=='full_route' else f"segment_{summary['segment_length_m']:.1f}"
        if stratum=='full_route':group='full_route_correlated_scene42'
        else:
            xy=np.asarray(summary['goal_xy'])
            group=f"{stratum}_arc_{summary['start_arc_length_m']:.3f}_goal_{xy[0]:.3f}_{xy[1]:.3f}"
        phases.extend(labels.tolist());strata.extend([stratum]*len(records))
        episode_ids.extend([summary['episode']]*len(records));groups.extend([group]*len(records))
    return {'phase':np.asarray(phases),'stratum':np.asarray(strata),'episode_id':np.asarray(episode_ids,np.int32),
        'group_id':np.asarray(groups),'dataset_sha256':np.array(digest)}


def dataset_arrays(dataset,labels,seed,validation_episode_ids=None):
    """Hold out one whole segment per distance; keep full repeats in one group."""
    records=[record for episode in dataset['episodes'] for record in episode['transitions']]
    episode_ids=np.asarray([record['episode'] for record in records],dtype=np.int32)
    if not np.array_equal(episode_ids,labels['episode_id']):
        raise ValueError('Sampling labels do not align with flattened demonstration records')
    phases=np.asarray(labels['phase']);strata=np.asarray(labels['stratum']);groups=np.asarray(labels['group_id'])
    if set(strata)!=set(STRATA) or set(phases)!=set(PHASE_COUNTS):
        raise ValueError('Unexpected carry phase or length strata')
    rng=np.random.default_rng(seed)
    validation=set()
    for stratum in STRATA[:-1]:
        candidates=sorted(set(groups[strata==stratum]))
        if len(candidates)<3:
            raise ValueError('Need at least three distinct physical segment cases per distance')
        if validation_episode_ids is None:
            selected=str(rng.choice(candidates))
        else:
            heldout=np.isin(episode_ids,validation_episode_ids)&(strata==stratum)
            selected_groups=set(groups[heldout])
            if len(selected_groups)!=1:
                raise ValueError('Frozen split requires one original physical group per distance')
            selected=str(next(iter(selected_groups)))
        validation.update(int(value) for value in episode_ids[groups==selected])
    valid=np.isin(episode_ids,list(validation))
    if set(groups[valid]) & set(groups[~valid]):
        raise ValueError('Correlated episode group crossed the supervised split')
    if np.any(valid & (strata=='full_route')):
        raise ValueError('Correlated full-route repeats must stay together on training side')
    train_groups={};validation_groups={}
    for stratum in STRATA:
        for phase in PHASE_COUNTS:
            mask=(strata==stratum)&(phases==phase)
            key=f'{stratum}:{phase}'
            train_groups[key]=np.flatnonzero(mask&~valid)
            if not len(train_groups[key]):
                raise ValueError('Training split lacks a requested phase/length stratum')
            if stratum!='full_route':
                validation_groups[key]=np.flatnonzero(mask&valid)
                if not len(validation_groups[key]):
                    raise ValueError('Validation split lacks a segment phase')
    observations={key:np.stack([record['observation'][key] for record in records]).astype(np.float32)
                  for key in records[0]['observation']}
    actions=np.stack([record.get('teacher_action',record['action']) for record in records]).astype(np.float32)
    if actions.shape!=(len(records),2) or not np.isfinite(actions).all():
        raise ValueError('Invalid carry action labels')
    split={'train_episode_ids':sorted(set(episode_ids.tolist())-validation),'validation_episode_ids':sorted(validation),
        'train_transitions':int((~valid).sum()),'validation_transitions':int(valid.sum()),
        'stratum_train_counts':{key:len(value) for key,value in train_groups.items()},
        'stratum_validation_counts':{key:len(value) for key,value in validation_groups.items()},
        'full_route_group':'full_route_correlated_scene42','full_route_group_side':'train',
        'group_leakage':False,'validation_full_route_episodes':0,
        'limitation':'Whole segment episodes are excluded from BC updates, but all share one object/maze and the source SAC physical route bank. Full-route physical checks reuse the demonstrated start and goal; no independent-generalization claim.'}
    return observations,actions,train_groups,validation_groups,split



def sharp_turn_groups(dataset,labels,split):
    """Locate measured teacher direction changes without changing policy inputs."""
    switches=[];neighbors=[]
    for episode in dataset['episodes']:
        rows=episode['transitions']
        actions=np.asarray([row.get('teacher_action',row['action']) for row in rows])
        norms=np.linalg.norm(actions,axis=1)
        exact=np.zeros(len(rows),bool);near=np.zeros(len(rows),bool)
        moving=np.asarray([row['teacher_phase']!='retained_goal_hold' for row in rows])
        for index in range(1,len(rows)):
            if moving[index] and moving[index-1] and min(norms[index-1],norms[index])>=.1:
                cosine=np.dot(actions[index-1],actions[index])/(norms[index-1]*norms[index])
                if np.degrees(np.arccos(np.clip(cosine,-1.,1.)))>=45.:
                    exact[index]=True
                    near[max(0,index-2):min(len(rows),index+3)]=True
        switches.extend(exact);neighbors.extend(near&~exact&moving)
    switches=np.asarray(switches);neighbors=np.asarray(neighbors)
    train=np.isin(labels['episode_id'],split['train_episode_ids'])
    strata=np.asarray(labels['stratum']);phases=np.asarray(labels['phase'])
    groups={};counts={}
    for stratum in STRATA:
        for name,mask in [('exact_switch',switches),('switch_neighbors',neighbors)]:
            key=f'{stratum}:{name}'
            indices=np.flatnonzero(mask&train&(strata==stratum)&(phases=='turn_approach'))
            if not len(indices):raise ValueError(f'No training rows for sharp-turn cell {key}')
            groups[key]=indices;counts[key]=len(indices)
    return groups,counts


def training_batch(rng,train_groups,sharp_groups=None,full_route_only=False):
    if full_route_only and sharp_groups is not None:
        raise ValueError('Full-route smooth-label sampler cannot use sharp-switch groups')
    batches=[]
    for stratum in (('full_route',) if full_route_only else STRATA):
        for phase,count in PHASE_COUNTS.items():
            if full_route_only:count*=len(STRATA)
            if phase=='turn_approach' and sharp_groups is not None:
                for kind in ('exact_switch','switch_neighbors'):
                    batches.append(rng.choice(sharp_groups[f'{stratum}:{kind}'],12,replace=True))
                count=12
            batches.append(rng.choice(train_groups[f'{stratum}:{phase}'],count,replace=True))
    return np.concatenate(batches)


def physical_full_route(env,model,update):
    observation,reset_info=env.reset(seed=42000,options={'evaluation_mode':'full'})
    goal=observation['desired_goal'].copy()
    trajectory=[];success=False;reason='decision_horizon'
    for decision in range(env.horizon):
        action=model.predict(observation,deterministic=True)[0]
        observation,reward,terminated,truncated,info=env.step(action)
        if not np.array_equal(observation['desired_goal'],goal):
            raise RuntimeError('Physical carry goal changed during evaluation')
        position=env.native.observe()['object_pos']
        trajectory.append({'decision':decision+1,'action':action.tolist(),
            'base_xy':(observation['achieved_goal']*env.goal_scale).tolist(),
            'object_position':position.tolist(),'goal_distance_m':float(info['goal_distance']),
            'bilateral_contact':bool(info['bilateral_contact']),'collision':bool(info['collision']),
            'physical_failure':bool(info['physical_failure']),'reward':float(reward),
            'source_action_calls':int(info['action_calls_delta']),'physics_ticks':int(info['physics_ticks_delta'])})
        failure=bool(terminated or info['physical_failure'] or not info['retention_ok'] or env.native.drop_events)
        success=bool(info['is_success'] and info['bilateral_contact'] and not failure)
        if success:
            reason='current_goal_and_retention';break
        if failure:
            reason=info['failure_reason'] or 'physical_failure';break
        if truncated:
            break
    result={'bc_updates':update,'scope':'full_carry','evaluation_protocol':'stop_on_first_retained_goal',
        'success':success,'failure':None if success else reason,'policy_decisions':len(trajectory),
        'final_goal_distance_m':float(info['goal_distance']),
        'minimum_goal_distance_m':min(item['goal_distance_m'] for item in trajectory),
        'final_bilateral_contact':bool(info['bilateral_contact']),
        'bilateral_contact_fraction':float(np.mean([item['bilateral_contact'] for item in trajectory])),
        'collision_decisions':sum(item['collision'] for item in trajectory),'drop_events':env.native.drop_events,
        'source_action_calls':env.native.step_count,'physics_ticks':env.native.physics_steps,
        'teacher_at_inference':False,'fixed_goal':goal.tolist(),'reset_info':reset_info,'trajectory':trajectory}
    if result['source_action_calls']!=sum(item['source_action_calls'] for item in trajectory):
        raise RuntimeError('Physical evaluation action accounting mismatch')
    return result


def demonstration_generation_costs(metadata,already_counted):
    """Attribute physical collection/setup/preflight once; keep preflight visible."""
    if already_counted:return 0,0,0,0
    preflight_calls=int(metadata.get('preflight_source_action_calls',0))
    preflight_ticks=int(metadata.get('preflight_physics_ticks',0))
    calls=int(metadata['setup_source_action_calls']+metadata['demonstration_source_action_calls'])+preflight_calls
    ticks=int(metadata['setup_physics_ticks']+metadata['demonstration_physics_ticks']+metadata['initialization_physics_ticks'])+preflight_ticks
    return calls,ticks,preflight_calls,preflight_ticks


def periodic_model_copy(original,vec,*,device,seed):
    """Expand a world-geometry SAC/HER model without changing its policy outputs.

    A normal SAC constructor keeps save/load and replay semantics intact. Actor
    and critic input expansions are explicit; optimizers start with new moments.
    The caller verifies original/HER predictions before any optional actor mask.
    """
    from pnr.carry_periodic_features import CarryPeriodicGeometryExtractor,transfer_carry_periodic_policy
    kwargs=dict(original.policy_kwargs,features_extractor_class=CarryPeriodicGeometryExtractor,
        features_extractor_kwargs={'goal_scale':original.actor.features_extractor.goal_scale})
    model=SAC('MultiInputPolicy',vec,learning_rate=original.learning_rate,
        buffer_size=original.buffer_size,learning_starts=original.learning_starts,
        batch_size=original.batch_size,tau=original.tau,gamma=original.gamma,
        train_freq=(original.train_freq.frequency,original.train_freq.unit.value),
        gradient_steps=original.gradient_steps,ent_coef='auto',target_entropy=original.target_entropy,
        target_update_interval=original.target_update_interval,policy_kwargs=kwargs,
        replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=original.replay_buffer_kwargs,
        device=device,seed=seed)
    report=transfer_carry_periodic_policy(original.policy,model.policy)
    with torch.no_grad():model.log_ent_coef.copy_(original.log_ent_coef)
    model.num_timesteps=original.num_timesteps;model._n_updates=original._n_updates
    return model,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',default='sac_carry_long01')
    parser.add_argument('--source-generation',help='Freeze this explicit completed generation instead of current latest')
    parser.add_argument('--geometry-features',action='store_true',help='Append measured world base/goal XY after HER')
    parser.add_argument('--periodic-features',action='store_true',help='Append 24 sine/cosine measured base/goal features at1,2,4 cycles/metre; requires geometry features')
    parser.add_argument('--mask-base-velocity',action='store_true',help='Declared actor-only root XY qvel ablation; requires geometry features')
    parser.add_argument('--mask-all-velocity',action='store_true',help='Declared actor-only all joint/object velocity ablation; requires geometry features')
    parser.add_argument('--mask-carry-pose',action='store_true',help='Actor uses map/base/fixed goal/contact with frozen carry posture; requires all-velocity mask and periodic features')
    parser.add_argument('--demonstrations',type=Path,default=ROOT/'runs/carry_physical_demos_v1/artifacts/demonstrations.pkl')
    parser.add_argument('--corrections',type=Path,help='Finalized mixed-action dataset with separate teacher_action BC labels')
    parser.add_argument('--prepared-dataset',action='store_true',help='Finalized consistently labelled combined data with frozen split and collector provenance')
    parser.add_argument('--collector-policy-run',default='carry_bc_geometry02')
    parser.add_argument('--sampling-labels',type=Path,default=ROOT/'runs/carry_physical_demos_v1/artifacts/sampling_labels.npz')
    parser.add_argument('--run-id',default='carry_bc_init01')
    parser.add_argument('--max-updates',type=int,default=5000)
    parser.add_argument('--sharp-turn-sampling',action='store_true',help='Declared turn-bin weighting: 12 exact switches,12 nearby rows,12 broad rows per stratum')
    parser.add_argument('--full-route-sampling',action='store_true',help='BC only from existing full-route fixed-goal training episodes; segment goals remain diagnostics')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=7400)
    args=parser.parse_args()
    if args.full_route_sampling and not (args.prepared_dataset and args.periodic_features and args.mask_carry_pose):
        raise ValueError('Full-route sampling requires prepared data, periodic features, and carry pose masking')
    if args.mask_carry_pose and not (args.mask_all_velocity and args.periodic_features):
        raise ValueError('Carry pose masking requires --mask-all-velocity and --periodic-features')
    if args.prepared_dataset:
        if args.corrections or args.sharp_turn_sampling:
            raise ValueError('Prepared smooth-label data cannot be merged again or use sharp-switch sampling')
        if not (args.geometry_features and args.periodic_features and args.mask_all_velocity):
            raise ValueError('Prepared lookahead protocol requires geometry, periodic features, and all-velocity actor mask')
    if args.periodic_features and not args.geometry_features:
        raise ValueError('Periodic coordinates require --geometry-features')
    if args.mask_base_velocity and args.mask_all_velocity:
        raise ValueError('Choose one explicitly named velocity ablation')
    if (args.mask_base_velocity or args.mask_all_velocity) and not args.geometry_features:
        raise ValueError('Actor velocity masking requires the carry geometry extractor')
    if args.corrections and not (args.geometry_features and args.mask_all_velocity):
        raise ValueError('Correction protocol requires declared world geometry and all-velocity actor mask')
    if args.sharp_turn_sampling and not args.corrections:
        raise ValueError('Sharp-turn protocol requires the frozen combined physical dataset')
    limit=30000 if args.full_route_sampling else (10000 if (args.corrections or args.prepared_dataset) else 5000)
    if not 1<=args.max_updates<=limit:
        raise ValueError(f'Carry BC protocol is bounded to at most{limit} supervised updates')
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    rng=np.random.default_rng(args.seed)
    source=ROOT/'runs'/args.source_run
    source_config=json.loads((source/'config.json').read_text())
    if source_config['skill']!='carry':raise ValueError('Expected a carry source run')
    generation=args.source_generation or json.loads((source/'checkpoints/latest.json').read_text())['generation']
    origin=source/'checkpoints'/generation
    source_counts=json.loads((origin/'progress.json').read_text())
    logger=RunLogger(args.run_id,kind='initialization',skill='carry',algorithm='BC-initialized SAC + HER',
        name='Carry · demonstration actor initialization',summary='Freezing source and grouped carry demonstration split; no online SAC updates.')
    directory=logger.directory;artifacts=directory/'artifacts';inputs=artifacts/'inputs';inputs.mkdir(parents=True)
    for old,name in [(origin/'model.zip','source_model.zip'),(origin/'progress.json','source_progress.json'),
                     (args.demonstrations,'demonstrations.pkl'),(args.sampling_labels,'sampling_labels.npz')]:
        shutil.copy2(old,inputs/name)
    source_hash=sha256(inputs/'source_model.zip');demo_hash=sha256(inputs/'demonstrations.pkl')
    dataset=pickle.loads((inputs/'demonstrations.pkl').read_bytes())
    collector_preparation=None;frozen_validation=None
    if args.corrections:
        (inputs/'demonstrations.pkl').rename(inputs/'original_demonstrations.pkl')
        (inputs/'sampling_labels.npz').rename(inputs/'original_sampling_labels.npz')
        shutil.copy2(args.corrections,inputs/'corrections.pkl')
        correction=pickle.loads((inputs/'corrections.pkl').read_bytes())
        source_records=[{'name':'original','path':str(args.demonstrations.resolve()),'sha256':demo_hash,'metadata':dataset['metadata']},
            {'name':'corrections','path':str(args.corrections.resolve()),'sha256':sha256(inputs/'corrections.pkl'),'metadata':correction['metadata']}]
        dataset=merged_training_data(dataset,correction,source_records)
        with (inputs/'demonstrations.pkl').open('wb') as handle:pickle.dump(dataset,handle,protocol=pickle.HIGHEST_PROTOCOL)
        demo_hash=sha256(inputs/'demonstrations.pkl')
        np.savez_compressed(inputs/'sampling_labels.npz',**sampling_labels_for_dataset(dataset,demo_hash))
        preparation_root=ROOT/'runs'/args.collector_policy_run
        preparation=json.loads((preparation_root/'artifacts/result.json').read_text())
        costs=preparation['accounting']
        if preparation['demonstrations_sha256']!=source_records[0]['sha256']:
            raise ValueError('Correction comparison requires the same original demonstration dataset')
        frozen_validation=preparation['split']['validation_episode_ids']
        preparation_hash=sha256(preparation_root/'checkpoints/best/model.zip')
        if (correction['metadata'].get('learner_source_run')!=args.collector_policy_run
                or correction['metadata'].get('learner_sha256')!=preparation_hash):
            raise ValueError('Collector policy preparation does not match the finalized correction dataset')
        if preparation['source_run']!=args.source_run or preparation['source_generation']!=generation:
            raise ValueError('Correction protocol requires the same original SAC source generation')
        collector_preparation={'run':args.collector_policy_run,'model_sha256':preparation_hash,
            'bc_updates':preparation['bc_updates'],'setup_source_action_calls':costs['new_setup_source_action_calls'],
            'setup_physics_ticks':costs['new_setup_physics_ticks'],
            'evaluation_source_action_calls':costs['new_evaluation_source_action_calls'],
            'evaluation_physics_ticks':costs['new_evaluation_physics_ticks'],
            'initialization_physics_ticks':costs['new_initialization_physics_ticks']}
        shutil.copy2(preparation_root/'artifacts/result.json',inputs/'collector_policy_preparation_result.json')
        atomic_json(inputs/'collector_policy_preparation.json',collector_preparation)
    if args.prepared_dataset:
        meta=dataset['metadata']
        if (meta.get('source_run'),meta.get('source_generation'),meta.get('source_model_sha256'))!=(args.source_run,generation,source_hash):
            raise ValueError('Prepared dataset does not match the immutable source SAC generation')
        frozen_validation=meta.get('validation_episode_ids')
        if not isinstance(frozen_validation,list) or not frozen_validation or any(not isinstance(value,int) for value in frozen_validation):
            raise ValueError('Prepared dataset requires explicit original validation episode IDs')
        preparation_claim=meta.get('collector_policy_preparation',{})
        preparation_root=ROOT/'runs'/preparation_claim.get('run','')
        preparation=json.loads((preparation_root/'artifacts/result.json').read_text())
        costs=preparation['accounting']
        preparation_hash=sha256(preparation_root/'checkpoints/best/model.zip')
        if (preparation['source_run'],preparation['source_generation'])!=(args.source_run,generation):
            raise ValueError('Prepared collector was not initialized from the declared original SAC source')
        collector_preparation={'run':preparation_claim['run'],'model_sha256':preparation_hash,
            'bc_updates':preparation['bc_updates'],'setup_source_action_calls':costs['new_setup_source_action_calls'],
            'setup_physics_ticks':costs['new_setup_physics_ticks'],
            'evaluation_source_action_calls':costs['new_evaluation_source_action_calls'],
            'evaluation_physics_ticks':costs['new_evaluation_physics_ticks'],
            'initialization_physics_ticks':costs['new_initialization_physics_ticks']}
        if any(preparation_claim.get(key)!=value for key,value in collector_preparation.items()):
            raise ValueError('Prepared collector provenance or physical preparation counts disagree with archived result')
        shutil.copy2(preparation_root/'artifacts/result.json',inputs/'collector_policy_preparation_result.json')
        atomic_json(inputs/'collector_policy_preparation.json',collector_preparation)
    labels=np.load(inputs/'sampling_labels.npz',allow_pickle=False)
    if str(labels['dataset_sha256'])!=demo_hash or not dataset['metadata'].get('metadata_finalized'):
        raise ValueError('Sampling labels or finalized dataset provenance mismatch')
    observations,actions,train_groups,validation_groups,split=dataset_arrays(dataset,labels,args.seed,validation_episode_ids=frozen_validation)
    full_route_groups={key:value for key,value in train_groups.items() if key.startswith('full_route:')}
    full_route_indices=np.concatenate(list(full_route_groups.values()))
    if args.full_route_sampling:
        goals=observations['desired_goal'][full_route_indices]
        expected_goal=np.asarray(dataset['metadata']['scene']['goal_pos'][:2],dtype=np.float32)/20.
        if not np.all(goals==goals[0]) or not np.allclose(goals[0],expected_goal,atol=1e-7,rtol=0):
            raise ValueError('Full-route sampler must use the single unchanged original carry exit goal')
    sharp_groups=sharp_counts=None
    if args.sharp_turn_sampling:
        sharp_groups,sharp_counts=sharp_turn_groups(dataset,labels,split)
    evaluation_updates=([value for value in range(1000,args.max_updates+1,1000)] if (args.corrections or args.prepared_dataset) else [500,1000,2000,3000,4000,5000])
    config=vars(args)|{'demonstrations':str((inputs/'demonstrations.pkl').resolve()) if args.corrections else str(args.demonstrations.resolve()),
        'corrections':None if args.corrections is None else str(args.corrections.resolve()),
        'sampling_labels':str((inputs/'sampling_labels.npz').resolve()) if args.corrections else str(args.sampling_labels.resolve()),
        'collector_policy_preparation':collector_preparation,
        'source_generation':generation,'source_model_sha256':source_hash,'demonstrations_sha256':demo_hash,
        'phase':'development','independent_seed':False,'new_online_transitions':0,
        'source_online_transitions':source_counts['transitions'],'source_sac_updates':source_counts['updates'],
        'protocol':('Original fixed full-route goal actor-mean MSE; existing eight correlated full-route episodes only,45%turn/45%straight/10%hold. Other segment goals are diagnostic, outside BC target distribution; no HER for BC labels.' if args.full_route_sampling else 'Original-goal actor-mean MSE; equal five distance strata with45%turn/45%straight/10%hold per stratum; no HER for BC labels.'),
        'actor_bc_learning_rate':.0003,'batch_size':400,
        'sampling_per_stratum':({phase:count*len(STRATA) for phase,count in PHASE_COUNTS.items()} if args.full_route_sampling else PHASE_COUNTS),
        'training_strata':['full_route'] if args.full_route_sampling else list(STRATA),
        'supervised_training_rows':len(full_route_indices) if args.full_route_sampling else split['train_transitions'],
        'segment_validation_scope':'Different-goal segment diagnostic outside BC target distribution' if args.full_route_sampling else 'Held-out matched-case segment goals',
        'sharp_turn_protocol':None if sharp_groups is None else {'definition':'Teacher action direction changes at least45 degrees, both norms>=0.1; neighbors are up to2 actual rows either side, excluding exact rows and goal hold.',
            'turn_bin_per_stratum':{'exact_switch':12,'switch_neighbors':12,'broad_turn':12},'training_cell_counts':sharp_counts},
        'physical_evaluation_updates':evaluation_updates,'physical_horizon':1000,'protocol_update_limit':limit,
        'initialization_mode':'Fresh BC experiment from the explicitly frozen SAC generation; no BC optimizer or actor resume',
        'bc_labels':'teacher_action when explicitly recorded, otherwise executed demonstration action; replay always uses action',
        'actor_observation_contract':('Map, measured baseXY, canonical fixed goal, and contact flags; raw530:578 arm/payload pose and velocity masked. Adapter preserves its fixed carry-posture/closed-finger constraint; critics retain full state.' if args.mask_carry_pose else 'Declared carry geometry and velocity-mask settings; retained payload/arm pose is observed.'),
        'failure_cost':200.,'physical_gate':'Fixed full-route XY goal <0.15m with bilateral contact, no physical failure/drop; stop at first retained goal.',
        'teacher_at_learned_inference':False,'split':split,
        'save_semantics':'Normal SAC/HER ZIP with source critics/counters and modified actor only. No online replay added; reload source replay and requested worker count explicitly for subsequent SAC.',
        'optimizer_semantics':'Separate Adam updates latent_pi and mu only; its moments are not installed into SAC actor.optimizer. Reset SAC actor optimizer for online continuation.'}
    atomic_json(directory/'config.json',config);atomic_json(artifacts/'split.json',split)
    shutil.copy2(Path(__file__),artifacts/'pretrain_carry_actor.py')
    code=artifacts/'source';code.mkdir()
    critical=['pnr/carry.py','pnr/env.py','pnr/reference.py','pnr/planning.py','pnr/features.py']
    if args.geometry_features:critical.append('pnr/carry_features.py')
    if args.periodic_features:critical.append('pnr/carry_periodic_features.py')
    for name in critical:shutil.copy2(ROOT/name,code/name.replace('/','__'))
    atomic_json(artifacts/'source_hashes.json',{name:sha256(ROOT/name) for name in critical})
    env=vec=None;evaluations=[];total_eval_calls=total_eval_ticks=0;best_key=None
    inherited=source_counts.get('accounting',{})
    demo_already_counted=bool(source_config.get('demonstrations_sha256')==demo_hash
        or (source_counts.get('demonstrations') or {}).get('sha256')==demo_hash)
    demo_meta=dataset['metadata'];started=time.perf_counter()
    try:
        env=CarrySkillEnv(scene_seed=source_config['scene_seed'],object_idx=source_config['object_idx'],
            seed=source_config['seed'],horizon=1000,evaluation='full',failure_cost=200.,adaptive_horizon=False)
        vec=DummyVecEnv([lambda:env])
        model=SAC.load(inputs/'source_model.zip',env=vec,device=args.device)
        if model.replay_buffer_class is not HerReplayBuffer:
            raise ValueError('BC requires normal SAC/HER checkpoint semantics')
        if model.num_timesteps!=source_counts['transitions'] or model._n_updates!=source_counts['updates']:
            raise ValueError('Frozen source counters disagree with completed generation')
        transfer_report=None
        if args.geometry_features:
            from pnr.carry_features import CarryWorldGeometryExtractor,transfer_carry_geometry_policy
            original=model
            kwargs=dict(original.policy_kwargs,features_extractor_class=CarryWorldGeometryExtractor,
                features_extractor_kwargs={'goal_scale':20.})
            model=SAC('MultiInputPolicy',vec,learning_rate=original.learning_rate,
                buffer_size=original.buffer_size,learning_starts=original.learning_starts,
                batch_size=original.batch_size,tau=original.tau,gamma=original.gamma,
                train_freq=(original.train_freq.frequency,original.train_freq.unit.value),
                gradient_steps=original.gradient_steps,ent_coef='auto',target_entropy=original.target_entropy,
                target_update_interval=original.target_update_interval,policy_kwargs=kwargs,
                replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=original.replay_buffer_kwargs,
                device=args.device,seed=args.seed)
            transfer_report=transfer_carry_geometry_policy(original.policy,model.policy)
            if args.periodic_features:
                model,periodic_report=periodic_model_copy(model,vec,device=args.device,seed=args.seed)
                transfer_report['periodic_transfer']=periodic_report
            with torch.no_grad():model.log_ent_coef.copy_(original.log_ent_coef)
            model.num_timesteps=original.num_timesteps;model._n_updates=original._n_updates
            check={key:torch.as_tensor(value[:128],device=model.device) for key,value in observations.items()}
            check_actions=torch.as_tensor(actions[:128],device=model.device)
            errors=[]
            for relabel in (False,True):
                if relabel:check['desired_goal']=check['desired_goal'].roll(19,dims=0)+.03
                with torch.no_grad():
                    pairs=[(original.actor(check,deterministic=True),model.actor(check,deterministic=True))]
                    pairs+=list(zip(original.critic(check,check_actions),model.critic(check,check_actions)))
                    pairs+=list(zip(original.critic_target(check,check_actions),model.critic_target(check,check_actions)))
                for before,after in pairs:
                    tolerance=1e-6 if args.periodic_features else 1e-5
                    torch.testing.assert_close(before,after,atol=tolerance,rtol=tolerance)
                    errors.append(float((before-after).abs().max()))
            transfer_report['before_mask_stored_and_relabelled_max_abs_error']=max(errors)
            transfer_report['periodic_features']=bool(args.periodic_features)
            transfer_report['prediction_tolerance']=1e-6 if args.periodic_features else 1e-5
            transfer_report['actor_only_base_velocity_mask']=bool(args.mask_base_velocity)
            transfer_report['actor_only_all_velocity_mask']=bool(args.mask_all_velocity)
            if args.mask_base_velocity:model.actor.features_extractor.set_base_velocity_mask(True)
            if args.mask_all_velocity:model.actor.features_extractor.set_all_velocity_mask(True)
            if args.mask_carry_pose:model.actor.features_extractor.set_carry_pose_mask(True)
            transfer_report['actor_carry_pose_mask']=bool(args.mask_carry_pose)
            for critic in (model.critic,model.critic_target):
                if not torch.all(critic.features_extractor.state_mask==1):
                    raise RuntimeError('Actor ablation accidentally masked critic inputs')
            atomic_json(artifacts/'geometry_transfer.json',transfer_report)
            del original
        actor=model.actor
        critic_before=tensor_hash(model.critic);target_before=tensor_hash(model.critic_target);std_before=tensor_hash(actor.log_std)
        parameters=list(actor.latent_pi.parameters())+list(actor.mu.parameters())
        optimizer=torch.optim.Adam(parameters,lr=.0003)
        tensors={key:torch.as_tensor(value,device=model.device) for key,value in observations.items()}
        targets=torch.as_tensor(actions,device=model.device)
        def losses(groups):
            with torch.no_grad():squared=((actor(tensors,deterministic=True)-targets)**2).mean(dim=1).cpu().numpy()
            values={key:float(squared[indices].mean()) for key,indices in groups.items()}
            weighted=sum(PHASE_COUNTS[key.split(':')[1]]*value for key,value in values.items())
            return weighted/sum(PHASE_COUNTS[key.split(':')[1]] for key in values),values
        def accounting():
            demo_calls,demo_ticks,preflight_calls,preflight_ticks=demonstration_generation_costs(demo_meta,demo_already_counted)
            preparation_calls=0 if collector_preparation is None else collector_preparation['setup_source_action_calls']+collector_preparation['evaluation_source_action_calls']
            preparation_ticks=0 if collector_preparation is None else collector_preparation['setup_physics_ticks']+collector_preparation['evaluation_physics_ticks']+collector_preparation['initialization_physics_ticks']
            return {'accounting_version':2,'complete':bool(inherited.get('complete',False)),
                'legacy_missing':inherited.get('legacy_missing',[]),'inherited_source_run':args.source_run,'inherited_accounting':inherited,
                'collector_policy_preparation':collector_preparation,
                'new_online_transitions':0,'supervised_train_transitions':len(full_route_indices) if args.full_route_sampling else split['train_transitions'],
                'available_training_split_transitions':split['train_transitions'],
                'demonstration_dataset_transitions':len(actions),'demonstration_generation_cost_newly_attributed':not demo_already_counted,
                'demonstration_dataset_sha256':demo_hash,'demonstration_generation_source_action_calls':demo_calls,
                'demonstration_generation_physics_ticks':demo_ticks,'demonstration_collection_metadata':demo_meta,
                'demonstration_preflight_source_action_calls':preflight_calls,'demonstration_preflight_physics_ticks':preflight_ticks,
                'demonstration_preflight_is_subset_of_generation_cost':True,
                'new_setup_source_action_calls':env.setup_action_calls,'new_setup_physics_ticks':env.setup_physics_ticks,
                'new_evaluation_source_action_calls':total_eval_calls,'new_evaluation_physics_ticks':total_eval_ticks,
                'new_initialization_physics_ticks':env.native.settle_steps,
                'replay_validation_source_action_calls':0,'replay_validation_physics_ticks':0,
                'total_source_action_calls_known':int(inherited.get('total_source_action_calls_known',0))+demo_calls+env.setup_action_calls+total_eval_calls+preparation_calls,
                'total_physics_ticks_known':int(inherited.get('total_physics_ticks_known',0))+demo_ticks+env.setup_physics_ticks+env.native.settle_steps+total_eval_ticks+preparation_ticks,
                'demonstration_data_reused_without_new_collection':True}
        initial_validation,_=losses(validation_groups)
        initial_full_route_training,_=losses(full_route_groups)
        logger.update(accounting=accounting(),source_online_transitions=model.num_timesteps)
        logger.event(f'Frozen {model.num_timesteps:,}-transition source and physical carry dataset; full correlated repeats remain together in training.',source_generation=generation)
        for update in range(1,args.max_updates+1):
            batch=training_batch(rng,train_groups,sharp_groups,full_route_only=args.full_route_sampling)
            prediction=actor({key:value[batch] for key,value in tensors.items()},deterministic=True)
            loss=torch.nn.functional.mse_loss(prediction,targets[batch])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step()
            if update==1 or update%100==0:
                validation_loss,phase_losses=losses(validation_groups)
                full_route_loss,full_route_phase_losses=losses(full_route_groups)
                logger.metric(total_eval_calls,metric_source='supervised_update',scope='actor_initialization',
                    updates=update,bc_train_loss=float(loss.detach()),bc_validation_loss=validation_loss,
                    bc_validation_strata=phase_losses,bc_validation_scope=config['segment_validation_scope'],
                    bc_full_route_training_loss=full_route_loss,bc_full_route_training_strata=full_route_phase_losses,
                    bc_training_scope='full_route_only' if args.full_route_sampling else 'five_distance_strata',source_action_calls=total_eval_calls,
                    physics_ticks=total_eval_ticks,accounting=accounting())
                summary=(f'Full-goal carry BC {update}/{args.max_updates}; full-route training MSE {full_route_loss:.4f}; other-goal segment diagnostic {validation_loss:.4f}.' if args.full_route_sampling else f'Carry actor initialization {update}/{args.max_updates}; held-out segment MSE {validation_loss:.4f}.')
                logger.update(summary=summary)
                print(json.dumps({'bc_updates':update,'validation_mse':validation_loss,'full_route_training_mse':full_route_loss}),flush=True)
            if update not in set(evaluation_updates+[args.max_updates]):continue
            case=physical_full_route(env,model,update)
            total_eval_calls+=case['source_action_calls'];total_eval_ticks+=case['physics_ticks']
            evaluation={'bc_updates':update,'scope':'full_carry','evaluation_scope':'full_carry',
                'evaluation_protocol':'stop_on_first_retained_goal','episodes':1,'success_rate':float(case['success']),
                'cases':[case],'accounting':accounting(),'new_online_transitions':0}
            evaluations.append(evaluation);atomic_json(artifacts/f'evaluation_{update:05d}.json',evaluation)
            logger.metric(total_eval_calls,metric_source='physical_evaluation',scope='full_carry',evaluation_scope='full_carry',
                updates=update,success_rate=float(case['success']),eval_episodes=1,accounting=accounting(),
                source_action_calls=total_eval_calls,physics_ticks=total_eval_ticks,mean_goal_distance=case['final_goal_distance_m'])
            logger.event(f'BC update{update}: full-route carry {"passed" if case["success"] else "failed"} ({case["failure"]}); no teacher at evaluation.')
            print(json.dumps({'bc_updates':update,'success':case['success'],'failure':case['failure'],
                'decisions':case['policy_decisions'],'goal_distance':case['final_goal_distance_m']}),flush=True)
            checkpoints=artifacts/'checkpoints';checkpoints.mkdir(exist_ok=True)
            model.save(checkpoints/f'bc_update_{update:05d}.zip')
            score=(int(case['success']),-case['final_goal_distance_m'])
            if best_key is None or score>best_key:
                best_key=score;best=directory/'checkpoints/best';best.mkdir(parents=True,exist_ok=True)
                model.save(best/'model.zip');atomic_json(best/'evaluation.json',evaluation)
            if case['success']:
                logger.event('Stopped bounded BC after the first successful physical full-route development check.')
                break
        if tensor_hash(model.critic)!=critic_before or tensor_hash(model.critic_target)!=target_before or tensor_hash(actor.log_std)!=std_before:
            raise RuntimeError('Actor initialization changed a protected critic or log_std head')
        if model.num_timesteps!=source_counts['transitions'] or model._n_updates!=source_counts['updates']:
            raise RuntimeError('BC changed source SAC interaction/update counters')
        saved,saved_parameters,_=load_from_zip_file(directory/'checkpoints/best/model.zip',device='cpu')
        if saved['replay_buffer_class'] is not HerReplayBuffer or saved['num_timesteps']!=source_counts['transitions'] or saved['_n_updates']!=source_counts['updates']:
            raise RuntimeError('Saved best checkpoint lost normal SAC/HER semantics or source counters')
        if args.periodic_features:
            from pnr.carry_periodic_features import CarryPeriodicGeometryExtractor
            if saved['policy_kwargs']['features_extractor_class'] is not CarryPeriodicGeometryExtractor:
                raise RuntimeError('Saved checkpoint lost the declared periodic extractor')
        if args.geometry_features:
            saved_mask=saved_parameters['policy']['actor.features_extractor.state_mask']
            expected=0. if (args.mask_base_velocity or args.mask_all_velocity) else 1.
            if not torch.all(saved_mask[543:545]==expected):
                raise RuntimeError('Saved actor velocity mask does not match the declared protocol')
            if args.mask_all_velocity and not (torch.all(saved_mask[543:558]==0) and torch.all(saved_mask[572:578]==0)):
                raise RuntimeError('Saved actor all-velocity mask is incomplete')
            if args.mask_carry_pose:
                expected_pose_mask=torch.ones_like(saved_mask);expected_pose_mask[530:578]=0.
                if not torch.equal(saved_mask,expected_pose_mask):
                    raise RuntimeError('Saved actor carry-pose mask changed map/base/contact inputs or left pose inputs visible')
        if sha256(inputs/'source_model.zip')!=source_hash or sha256(inputs/'demonstrations.pkl')!=demo_hash:
            raise RuntimeError('Frozen input bytes changed during initialization')
        final={'kind':'BC_initialized_SAC_HER_development','skill':'carry','source_run':args.source_run,
            'source_generation':generation,'source_model_sha256':source_hash,'demonstrations_sha256':demo_hash,
            'source_online_transitions':source_counts['transitions'],'source_sac_updates':source_counts['updates'],
            'new_online_transitions':0,'bc_updates':update,'initial_validation_mse':initial_validation,
            'final_validation_mse':losses(validation_groups)[0],'segment_validation_scope':config['segment_validation_scope'],
            'initial_full_route_training_mse':initial_full_route_training,
            'final_full_route_training_mse':losses(full_route_groups)[0],'final_full_route_training_strata':losses(full_route_groups)[1],
            'full_route_only_sampling':bool(args.full_route_sampling),'split':split,'evaluations':evaluations,'accounting':accounting(),
            'critics_unchanged':True,'log_std_head_weights_unchanged':True,'source_sac_counters_unchanged':True,
            'normal_her_checkpoint_verified':True,'teacher_at_inference':False,'independent_seed':False,
            'geometry_transfer':transfer_report,'actor_base_velocity_masked':bool(args.mask_base_velocity or args.mask_all_velocity),
            'actor_all_velocities_masked':bool(args.mask_all_velocity),
            'periodic_features':bool(args.periodic_features),
            'actor_carry_pose_masked':bool(args.mask_carry_pose),
            'log_std_output_caveat':'The unchanged head reads an updated shared actor backbone; stochastic outputs can change.',
            'best_model':str(directory/'checkpoints/best/model.zip'),'best_full_carry_success':bool(best_key[0]),
            'source_num_envs':source_config['num_envs'],'checkpoint_runtime_num_envs':model.n_envs,
            'continuation_requires':'Restore source replay and desired worker count explicitly; reset SAC actor optimizer moments.',
            'wall_seconds':time.perf_counter()-started}
        atomic_json(artifacts/'result.json',final)
        logger.artifact('artifacts/result.json','Carry BC initializer: physical outcomes, split and complete costs')
        if args.corrections or args.prepared_dataset:
            logger.artifact('artifacts/inputs/demonstrations.pkl','Merged physical replay: executed actions and separate BC labels')
            logger.artifact('artifacts/inputs/sampling_labels.npz','Grouped teacher-phase sampling labels')
        for evaluation in evaluations:
            logger.artifact(f'artifacts/evaluation_{evaluation["bc_updates"]:05d}.json',f'Full-route physical check after {evaluation["bc_updates"]} BC updates')
        logger.finish('complete',f'Carry actor initialization stopped at {update} supervised updates; full-route physical success {bool(best_key[0])}. No new online SAC transitions.')
    except BaseException as error:
        logger.finish('failed',f'{type(error).__name__}: {error}');raise
    finally:
        if vec is not None:vec.close()
        elif env is not None:env.close()


if __name__=='__main__':main()
