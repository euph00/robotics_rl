#!/usr/bin/env python3
"""Compose a declared grasp-teacher refresh from counted source episodes.

Only the named object cohorts are replaced, including all their new failures.
Retained cases must be top-down, where the new ray-tilted teacher has exactly
identical actions. No simulator is created and no new transitions are invented.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scipy.spatial.transform import Rotation
from pnr.telemetry import RunLogger,atomic_json
from scripts.generate_grasp_demos import save_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--replacement',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--objects',nargs='+',type=int,required=True)
    args=parser.parse_args()
    base=pickle.loads(args.base.read_bytes());replacement=pickle.loads(args.replacement.read_bytes())
    ids=set(args.objects);bcfg=base['metadata']['configuration'];rcfg=replacement['metadata']['configuration']
    for field in ['scene_seeds','horizon','repetitions','jitter_xy','lift_range','seed','grasp_proposal_mode']:
        if bcfg[field]!=rcfg[field]:raise ValueError('Source protocol differs: '+field)
    if rcfg.get('teacher_approach_control')!='ray-tilted' or bcfg.get('teacher_approach_control','component')!='component':
        raise ValueError('This composition verifies only component→ray-tilted teacher refresh')
    keys=lambda cases:{(c['object_idx'],c['scene_seed']) for c in cases}
    replaced=[c for c in base['metadata']['cases'] if c['object_idx'] in ids]
    if keys(replaced)!=keys(replacement['metadata']['cases']):raise ValueError('Replacement must cover the entire declared object/seed cohorts')
    kept=[c for c in base['metadata']['cases'] if c['object_idx'] not in ids]
    for case in kept:
        axis=Rotation.from_quat(case['proposal']['quaternion']).apply([1.,0.,0.])
        if abs(axis[2])<.999:raise ValueError('A retained tilted case needs actual recollection under the changed teacher')
    sources=[];episodes=[];cases=[]
    for name,path,dataset,keep,reused in [('base',args.base,base,lambda obj:obj not in ids,True),
                                         ('replacement',args.replacement,replacement,lambda obj:obj in ids,False)]:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        selected_cases=[copy.deepcopy(c) for c in dataset['metadata']['cases'] if keep(c['object_idx'])]
        selected_episodes=[copy.deepcopy(e) for e in dataset['episodes'] if keep(e['summary']['object_idx'])]
        for episode in selected_episodes:
            summary=episode['summary']
            summary['provenance']={'source_dataset':str(path.resolve()),'source_sha256':digest,
                'source_run_id':dataset['metadata']['configuration']['run_id'],
                'source_dataset_episode':summary['dataset_episode'],
                'retained_exact_topdown_teacher_equivalence':reused}
        for case in selected_cases:
            case['source_run_id']=dataset['metadata']['configuration']['run_id'];case['source_dataset_sha256']=digest
        sources.append({'role':name,'dataset':str(path.resolve()),'sha256':digest,
            'source_run_id':dataset['metadata']['configuration']['run_id'],
            'episodes_used':len(selected_episodes),'reused_unchanged_topdown':reused,
            'demonstration_source_action_calls':sum(c['demonstration_source_action_calls'] for c in selected_cases),
            'setup_source_action_calls':sum(c['setup_source_action_calls'] for c in selected_cases),
            'perturbation_source_action_calls':sum(c['perturbation_source_action_calls'] for c in selected_cases),
            'initialization_physics_ticks':sum(c['initialization_physics_ticks'] for c in selected_cases)})
        episodes.extend(selected_episodes);cases.extend(selected_cases)
    for index,episode in enumerate(episodes):
        episode['summary']['dataset_episode']=index
        for transition in episode['transitions']:transition['episode']=index
    logger=RunLogger(args.run_id,kind='data',skill='grasp',algorithm='Recorded physical dataset composition',
        name='Phase2 · corrected tilt demonstrations',budget=sum(len(e['transitions']) for e in episodes),
        summary='Recollected can cohort plus exactly unchanged recorded top-down cohorts; merge executes no physics.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    configuration=dict(bcfg,run_id=args.run_id,output=str(out),teacher_approach_control='ray-tilted',
                       record_composition=True,source_datasets=sources)
    metadata={'configuration':configuration,'cases':cases,'collection_status':'complete',
        'observation_version':base['metadata']['observation_version'],
        'new_physical_interactions_in_this_merge':0,'physical_source_provenance':sources,
        'teacher_equivalence':'ray-tilted conditional is inactive for every retained top-down proposal; focused regression tested.',
        'policy_decisions':sum(len(e['transitions']) for e in episodes),'all_attempts_retained':True,
        'prior_superseded_object_attempts_preserved_in':str(args.base.resolve())}
    for field in ['demonstration_source_action_calls','demonstration_physics_ticks','setup_source_action_calls',
                  'setup_physics_ticks','initialization_physics_ticks','perturbation_source_action_calls']:
        metadata[field]=sum(case[field] for case in cases)
    digest=save_dataset(out/'demonstrations.pkl',episodes,metadata)
    atomic_json(out/'manifest.json',metadata|{'dataset_sha256':digest,'episodes':[e['summary'] for e in episodes]})
    atomic_json(logger.directory/'config.json',configuration)
    successful=sum(e['summary']['successful_physical_lift_and_hold'] for e in episodes)
    logger.metric(metadata['policy_decisions'],scope='recorded_dataset_composition',successful_episodes=successful,
        episodes=len(episodes),newly_executed_source_action_calls=0,reused_episodes=sources[0]['episodes_used'],
        newly_collected_source_episodes=sources[1]['episodes_used'])
    logger.artifact('artifacts/demonstrations.pkl','Corrected full-bank recorded dataset')
    logger.artifact('artifacts/manifest.json','Source episode provenance and physical cost reuse')
    logger.finish(summary=f'{successful}/{len(episodes)} recorded lift/hold successes; zero physics executed during composition.')
    print({'episodes':len(episodes),'successes':successful,'sha256':digest,'sources':sources})

if __name__=='__main__':main()
