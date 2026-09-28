#!/usr/bin/env python3
"""Compose a declared grasp-teacher refresh from counted source episodes.

Refresh replaces a cohort under an equivalent teacher change. Append adds
disjoint TRAIN cases under the identical teacher protocol. Both retain failures,
execute no physics, and invent no transitions.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import pickle
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scipy.spatial.transform import Rotation
from pnr.telemetry import RunLogger,atomic_json
from scripts.generate_grasp_demos import save_dataset


def exact_equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys()==right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left)==len(right) and all(exact_equal(a,b) for a,b in zip(left,right))
    return left==right


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--replacement',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--objects',nargs='+',type=int,required=True)
    parser.add_argument('--mode',choices=['refresh','append'],default='refresh')
    args=parser.parse_args()
    base=pickle.loads(args.base.read_bytes());replacement=pickle.loads(args.replacement.read_bytes())
    ids=set(args.objects);bcfg=base['metadata']['configuration'];rcfg=replacement['metadata']['configuration']
    fields=['horizon','repetitions','jitter_xy','lift_range','seed','grasp_proposal_mode']
    if args.mode=='refresh':fields.append('scene_seeds')
    for field in fields:
        if bcfg[field]!=rcfg[field]:raise ValueError('Source protocol differs: '+field)
    keys=lambda cases:{(c['object_idx'],c['scene_seed']) for c in cases}
    if {c['object_idx'] for c in replacement['metadata']['cases']}!=ids:
        raise ValueError('Incoming dataset must exactly match declared object cohort')
    if args.mode=='refresh':
        if rcfg.get('teacher_approach_control')!='ray-tilted' or bcfg.get('teacher_approach_control','component')!='component':
            raise ValueError('Refresh verifies only component-to-ray-tilted teacher change')
        replaced=[c for c in base['metadata']['cases'] if c['object_idx'] in ids]
        if keys(replaced)!=keys(replacement['metadata']['cases']):raise ValueError('Replacement must cover entire declared cohort')
        kept=[c for c in base['metadata']['cases'] if c['object_idx'] not in ids]
        for case in kept:
            axis=Rotation.from_quat(case['proposal']['quaternion']).apply([1.,0.,0.])
            if abs(axis[2])<.999:raise ValueError('Retained tilted case requires recollection under changed teacher')
    else:
        if bcfg.get('teacher_approach_control','component')!=rcfg.get('teacher_approach_control','component'):
            raise ValueError('Append requires the same physical teacher protocol')
        if keys(base['metadata']['cases']) & keys(replacement['metadata']['cases']):
            raise ValueError('Append requires disjoint object/point-seed cases')
        if not bcfg.get('training_only') or not rcfg.get('training_only'):
            raise ValueError('Append accepts declared TRAIN collections only')
    sources=[];episodes=[];cases=[];originals=[]
    for name,path,dataset,keep,reused in [('base',args.base,base,lambda obj:args.mode=='append' or obj not in ids,True),
                                         ('replacement',args.replacement,replacement,lambda obj:obj in ids,False)]:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        selected_cases=[copy.deepcopy(c) for c in dataset['metadata']['cases'] if keep(c['object_idx'])]
        source_episodes=[e for e in dataset['episodes'] if keep(e['summary']['object_idx'])]
        originals.extend(source_episodes)
        selected_episodes=copy.deepcopy(source_episodes)
        for episode in selected_episodes:
            summary=episode['summary']
            previous=summary.get('provenance')
            summary['provenance']={'source_dataset':str(path.resolve()),'source_sha256':digest,
                'source_run_id':dataset['metadata']['configuration']['run_id'],
                'source_dataset_episode':summary['dataset_episode'],
                'retained_exact_topdown_teacher_equivalence':reused and args.mode=='refresh',
                'composition_mode':args.mode}
            if previous is not None:summary['provenance']['earlier_source_provenance']=previous
        for case in selected_cases:
            case['source_run_id']=dataset['metadata']['configuration']['run_id'];case['source_dataset_sha256']=digest
        sources.append({'role':name,'dataset':str(path.resolve()),'sha256':digest,
            'source_run_id':dataset['metadata']['configuration']['run_id'],
            'episodes_used':len(selected_episodes),'reused_unchanged_topdown':reused and args.mode=='refresh',
            'reused_recorded_episodes':reused,
            'demonstration_source_action_calls':sum(c['demonstration_source_action_calls'] for c in selected_cases),
            'setup_source_action_calls':sum(c['setup_source_action_calls'] for c in selected_cases),
            'perturbation_source_action_calls':sum(c['perturbation_source_action_calls'] for c in selected_cases),
            'initialization_physics_ticks':sum(c['initialization_physics_ticks'] for c in selected_cases)})
        episodes.extend(selected_episodes);cases.extend(selected_cases)
    for index,episode in enumerate(episodes):
        episode['summary']['dataset_episode']=index
        for transition in episode['transitions']:transition['episode']=index
    logger=RunLogger(args.run_id,kind='data',skill='grasp',algorithm='Recorded physical dataset composition',
        name='Phase2 · composed grasp demonstrations',budget=sum(len(e['transitions']) for e in episodes),
        summary='Declared recorded TRAIN cohorts with source provenance; merge executes no physics.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    configuration=dict(bcfg,run_id=args.run_id,output=str(out),teacher_approach_control=rcfg['teacher_approach_control'],
                       record_composition=True,composition_mode=args.mode,source_datasets=sources)
    configuration['scene_seeds']=sorted({c['scene_seed'] for c in cases})
    configuration['objects']=sorted({c['object_idx'] for c in cases})
    configuration['case_grid']='metadata.cases is authoritative; seed/object lists do not imply a Cartesian product.'
    metadata={'configuration':configuration,'cases':cases,'collection_status':'complete',
        'observation_version':base['metadata']['observation_version'],
        'new_physical_interactions_in_this_merge':0,'physical_source_provenance':sources,
        'teacher_equivalence':('ray-tilted is inactive for every retained top-down proposal; regression tested.'
                              if args.mode=='refresh' else 'Same declared teacher for every disjoint TRAIN case.'),
        'policy_decisions':sum(len(e['transitions']) for e in episodes),'all_attempts_retained':True}
    if args.mode=='refresh':metadata['prior_superseded_object_attempts_preserved_in']=str(args.base.resolve())
    for field in ['demonstration_source_action_calls','demonstration_physics_ticks','setup_source_action_calls',
                  'setup_physics_ticks','initialization_physics_ticks','perturbation_source_action_calls']:
        metadata[field]=sum(case[field] for case in cases)
    transition_count=0
    for original,composed in zip(originals,episodes):
        assert exact_equal({k:v for k,v in original['summary'].items() if k not in ('dataset_episode','provenance')},
                           {k:v for k,v in composed['summary'].items() if k not in ('dataset_episode','provenance')})
        assert len(original['transitions'])==len(composed['transitions'])
        for old,new in zip(original['transitions'],composed['transitions']):
            assert exact_equal({k:v for k,v in old.items() if k!='episode'},
                               {k:v for k,v in new.items() if k!='episode'})
            transition_count+=1
    integrity={'all_source_values_exact':True,'episodes_checked':len(episodes),
               'transitions_checked':transition_count,'allowed_changes':['dataset episode IDs','source provenance']}
    atomic_json(out/'source_integrity_check.json',integrity)
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
