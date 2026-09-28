#!/usr/bin/env python3
"""Offline consistent lookahead labels; preserve every actual physical transition."""
from __future__ import annotations
import argparse,copy,hashlib,json,pickle,shutil,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pybullet_data
from pnr.planning import maze_astar
from pnr.telemetry import RunLogger,atomic_json
from scripts.collect_carry_lookahead import projected_target,clean_polyline,jsonable,MODEL_SHA
from scripts.pretrain_carry_actor import sampling_labels_for_dataset,dataset_arrays

INPUTS=(('original','runs/carry_physical_demos_v1/artifacts/demonstrations.pkl','e1542a969fd693dd6c7a6e0b0cd9bbd636e7e2e6ecc4b7b6bdbc10270615c557'),
        ('lookahead','runs/carry_lookahead_demos_v1/artifacts/corrections.pkl','4367f8b9779125cd374abbaf75d46a4deec9b9078df1d61a6d0e4eb71eca020c'))
VERSION='memoryless-polyline-lookahead-0.15m-supervision-v1'
PHYSICAL_FIELDS=('observation','action','next_observation','reward','terminated','truncated','info')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_digest(row):
    return hashlib.sha256(pickle.dumps({key:row[key] for key in PHYSICAL_FIELDS},protocol=5)).hexdigest()


def target_label(path,observation):
    base=observation['achieved_goal']*20.
    target,labels=projected_target(path,base)
    hold=bool(np.linalg.norm(observation['desired_goal']-observation['achieved_goal'])*20.<.15)
    if hold:target=observation['desired_goal']*20.
    action=(target-base)/.045;action/=max(1.,float(np.linalg.norm(action)))
    labels.update(teacher_waypoint_index=labels['teacher_segment_index']+1,
        teacher_waypoint=np.asarray(path[labels['teacher_segment_index']+1]).tolist(),
        teacher_local_target=target.tolist(),teacher_lookahead_m=.15,
        teacher_subphase='goal_hold' if hold else ('turn_approach'
            if labels['teacher_segment_index']<len(path)-2 and labels['teacher_next_vertex_distance_m']<.35 else 'straight'))
    return action.astype(np.float32),'retained_goal_hold' if hold else 'astar_polyline_lookahead',labels


def original_actions_check(episode,path):
    index=1
    if len(path)!=episode['summary']['teacher_waypoints']:raise ValueError('Original waypoint count mismatch')
    for row in episode['transitions']:
        base=row['observation']['achieved_goal']*20
        if row['teacher_phase']=='retained_goal_hold':expected=np.zeros(2,dtype=np.float32)
        else:
            while index<len(path)-1 and np.linalg.norm(base-path[index])<.012:index+=1
            expected=(path[index]-base)/.045;expected/=max(1.,float(np.linalg.norm(expected)))
            expected=expected.astype(np.float32)
        if not np.array_equal(expected,row['action']):raise ValueError('Reconstructed route does not reproduce original actions exactly')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='carry_lookahead_relabel_v1');args=parser.parse_args()
    logger=RunLogger(args.run_id,kind='validation',skill='carry',algorithm='Offline teacher relabeling',
        name='Consistent carry teacher labels',summary='No new physics. Original28 plus lookahead12; actual transitions unchanged.')
    p=logger.directory/'artifacts';p.mkdir();sources=p/'source';sources.mkdir();started=time.perf_counter()
    data=[];records=[]
    for name,relative,digest in INPUTS:
        path=ROOT/relative
        if sha(path)!=digest:raise ValueError('Immutable input mismatch')
        with path.open('rb') as handle:dataset=pickle.load(handle)
        if not dataset['metadata']['metadata_finalized']:raise ValueError('Unfinished physical dataset')
        data.append(dataset);records.append({'name':name,'path':str(path),'sha256':digest,'metadata':dataset['metadata']})
    table_path=Path(pybullet_data.getDataPath())/'table/table.urdf'
    size=np.fromstring(ET.parse(table_path).find('.//collision/geometry/box').attrib['size'],sep=' ')
    np.testing.assert_array_equal(size,[1.5,1.,.05])
    half=size[[1,0]]/2.+.001;bounds=(-half,half)
    maze=np.asarray(data[0]['metadata']['scene']['maze_map']);r,c=np.where(maze);walls=np.stack([c-3.5,5.5-r],axis=1)
    prep_root=ROOT/'runs/carry_bc_geometry02';prep_path=prep_root/'artifacts/result.json';prep=json.loads(prep_path.read_text())
    if sha(prep_root/'checkpoints/best/model.zip')!=MODEL_SHA or data[1]['metadata']['learner_sha256']!=MODEL_SHA:raise ValueError('Collector provenance')
    if prep['demonstrations_sha256']!=INPUTS[0][2]:raise ValueError('Preparation original-data mismatch')
    costs=prep['accounting'];collector={'run':'carry_bc_geometry02','model_sha256':MODEL_SHA,'bc_updates':prep['bc_updates'],
        'setup_source_action_calls':costs['new_setup_source_action_calls'],'setup_physics_ticks':costs['new_setup_physics_ticks'],
        'evaluation_source_action_calls':costs['new_evaluation_source_action_calls'],'evaluation_physics_ticks':costs['new_evaluation_physics_ticks'],
        'initialization_physics_ticks':costs['new_initialization_physics_ticks'],'result_sha256':sha(prep_path)}
    episodes=[];physical_checks=0;path_checks=0;new_label_checks=0;changed_original_labels=0
    for source_index,dataset in enumerate(data):
        if dataset['metadata']['scene']['maze_hash']!=data[0]['metadata']['scene']['maze_hash']:raise ValueError('Different maze')
        for original in dataset['episodes']:
            s=original['summary'];raw=maze_astar(walls,s['start_base_xy'],s['goal_xy'],table_bounds=bounds);path=clean_polyline(raw)
            if source_index==0:original_actions_check(original,raw)
            else:
                if not np.array_equal(path,np.asarray(s['teacher_path'])):raise ValueError('Reconstructed path differs from collected path')
                path_checks+=1
            episode=copy.deepcopy(original);index=len(episodes)
            episode['summary'].update(source_dataset=records[source_index]['name'],source_episode=s['episode'],episode=index,
                teacher_path=path.tolist(),supervised_teacher_version=VERSION)
            for before,row in zip(original['transitions'],episode['transitions']):
                label,phase,geometry=target_label(path,row['observation'])
                if source_index==1:
                    if not np.array_equal(label,before['teacher_action']) or phase!=before['teacher_phase']:raise ValueError('Memoryless field differs from new collected teacher')
                    new_label_checks+=1
                else:changed_original_labels+=bool(not np.array_equal(label,before['action']))
                row.update(source_episode=before['episode'],episode=index,
                    source_teacher_action=np.asarray(before.get('teacher_action',before['action'])).copy(),
                    source_teacher_phase=before['teacher_phase'],teacher_action=label,teacher_phase=phase,
                    teacher_label_version=VERSION,**geometry)
                if physical_digest(row)!=physical_digest(before):raise ValueError('An actual physical transition changed')
                physical_checks+=1
            episodes.append(episode)
    metadata={'run_id':args.run_id,'metadata_finalized':True,'collection_status':'complete','training_only':True,
        'skill':'carry','scene_seed':42,'object_idx':14,'horizon':1000,'failure_cost':200.,'scene':data[0]['metadata']['scene'],
        'teacher_version':VERSION,'source_datasets':records,'policy_decisions':physical_checks,'episodes':len(episodes),
        'fixed_goal_per_episode':True,'policy_observations_modified':False,'teacher_at_learned_inference':False,
        'new_physical_source_action_calls':0,'new_physics_ticks':0,'new_actor_or_critic_updates':0,
        'action_contract':'action is the original actual executed command; teacher_action is a separately recomputed BC label only.',
        'label_contract':'Current base projection onto the frozen episode A* polyline, 0.15m ahead. Current measured goal distance<0.15m selects direct goal correction. No previous phase/index or HER goal changes.',
        'validation_episode_ids':prep['split']['validation_episode_ids'],'collector_policy_preparation':collector,
        'collector_policy_run':'carry_bc_geometry02','learner_sha256':MODEL_SHA,
        'source_run':prep['source_run'],'source_generation':prep['source_generation'],'source_model_sha256':prep['source_model_sha256'],
        'collector_policy_preparation_result_sha256':sha(prep_path),
        'geometry_reconstruction':{'table_urdf':str(table_path),'table_urdf_sha256':sha(table_path),'table_box_xyz':size.tolist(),
            'table_yaw_radians':float(np.pi/2),'table_aabb_padding_m':.001,'table_aabb_xy':[bounds[0].tolist(),bounds[1].tolist()],
            'footprint_radius_m':.36,'planner_resolution_m':.25,'duplicate_vertices':'Only consecutive zero-length segments removed.',
            'validation':'Reconstructed original paths reproduce all5614 original actions exactly; all12 new paths match exactly.'},
        'correlation_notice':'All eight full-route repeats stay in training. Matching segment start/goal groups across both inputs remain on one side of the original geometry02 split.',
        'preflight_accounting_notice':'The lookahead collector pure preflight is not imported; its3850 calls/19250 ticks remain a separate cost included once in totals. Original28 preflight cases were retained members of that original dataset and are already counted in its demonstration counters.',
        'offline_relabel_wall_seconds':time.perf_counter()-started}
    for field in ('demonstration_source_action_calls','demonstration_physics_ticks','setup_source_action_calls','setup_physics_ticks',
                  'initialization_physics_ticks','preflight_source_action_calls','preflight_physics_ticks'):
        metadata[field]=sum(int(d['metadata'].get(field,0)) for d in data)
    metadata['total_source_action_calls']=sum(metadata[k] for k in ('demonstration_source_action_calls','setup_source_action_calls','preflight_source_action_calls'))
    metadata['total_physics_ticks']=sum(metadata[k] for k in ('demonstration_physics_ticks','setup_physics_ticks','preflight_physics_ticks','initialization_physics_ticks'))
    hashes={}
    for name in ('scripts/relabel_carry_lookahead.py','scripts/collect_carry_lookahead.py','pnr/planning.py','scripts/pretrain_carry_actor.py'):
        payload=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(payload).hexdigest();(sources/name.replace('/','__')).write_bytes(payload)
    shutil.copy2(table_path,sources/'table.urdf');shutil.copy2(prep_path,p/'collector_policy_preparation_result.json');metadata['source_hashes']=hashes
    final={'schema_version':1,'kind':'physical_training_demonstrations','training_only':True,'teacher_version':VERSION,'metadata':metadata,'episodes':episodes}
    pending=p/'demonstrations.pending.pkl'
    with pending.open('wb') as handle:pickle.dump(final,handle,protocol=pickle.HIGHEST_PROTOCOL)
    digest=sha(pending);labels=sampling_labels_for_dataset(final,digest)
    _,_,_,_,split=dataset_arrays(final,labels,7400,validation_episode_ids=metadata['validation_episode_ids'])
    pending.replace(p/'demonstrations.pkl');np.savez_compressed(p/'sampling_labels.npz',**labels)
    checks={'physical_transitions_preserved_exactly':physical_checks,'original_actions_reproduced_exactly':5614,
        'new_paths_reconstructed_exactly':path_checks,'new_teacher_labels_reproduced_exactly':new_label_checks,
        'original_labels_changed':changed_original_labels,'new_physics_ticks':0,'split':split}
    atomic_json(p/'manifest.json',jsonable({'dataset_sha256':digest,'sampling_labels_sha256':sha(p/'sampling_labels.npz'),
        'metadata':metadata,'integrity':checks}))
    for name,label in [('demonstrations.pkl','Consistent teacher labels; all original executed actions retained'),('sampling_labels.npz','Broad route-turn/straight/hold labels'),('manifest.json','Exact preservation checks, source hashes and costs')]:logger.artifact('artifacts/'+name,label,'file')
    logger.metric(physical_checks,scope='offline_labels',episodes=len(episodes),new_physics_ticks=0)
    logger.finish('complete',f'Prepared{len(episodes)} episodes/{physical_checks} unchanged physical transitions with one memoryless teacher field; no new physics.')
    print(json.dumps(jsonable({'dataset_sha256':digest,'path':str(p/'demonstrations.pkl'),'integrity':checks,'costs':{k:v for k,v in metadata.items() if k.endswith('calls') or k.endswith('ticks')}})),flush=True)


if __name__=='__main__':main()
