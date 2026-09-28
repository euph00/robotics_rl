#!/usr/bin/env python3
"""Validate then collect training-only 0.15 m polyline-lookahead carry data.

The current base is projected onto every segment of the fixed A* polyline.
The target lies 0.15 m farther along that polyline, independent of any previous
waypoint index. A teacher-only geometric target never enters actor observations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pnr.carry import CarrySkillEnv
from pnr.policies import load_policy_checkpoint
from pnr.telemetry import RunLogger,atomic_json
from pnr.train import GracefulStop
from scripts.generate_carry_demos import GeometricCarryTeacher,start_case,HOLD_DECISIONS
from scripts.collect_carry_corrections import bounded_cases

VERSION='astar-polyline-lookahead-0.15m-v1'
MODEL_SHA='ad4a10d8c02cc7f447fe6f4d0e2c191d0232ed2490dfccd83ebdc30112430d7a'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:jsonable(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(item) for item in value]
    return value


def clean_polyline(path):
    path=np.asarray(path,dtype=float)
    return path[np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-12]]


def projected_target(path,base,lookahead=.15):
    """Memoryless closest-segment projection and arc-length lookahead."""
    path=clean_polyline(path);base=np.asarray(base,dtype=float)
    vectors=np.diff(path,axis=0);lengths=np.linalg.norm(vectors,axis=1)
    if len(lengths)==0 or np.any(lengths<=1e-12):raise ValueError('Polyline requires distinct consecutive points')
    fraction=np.clip(np.sum((base-path[:-1])*vectors,axis=1)/(lengths**2),0.,1.)
    projections=path[:-1]+fraction[:,None]*vectors
    distances=np.linalg.norm(projections-base,axis=1);nearest=int(np.argmin(distances))
    arcs=np.r_[0.,np.cumsum(lengths)]
    current_arc=float(arcs[nearest]+fraction[nearest]*lengths[nearest])
    target_arc=min(current_arc+lookahead,float(arcs[-1]))
    target_segment=min(int(np.searchsorted(arcs,target_arc,side='right')-1),len(lengths)-1)
    target=path[target_segment]+(target_arc-arcs[target_segment])/lengths[target_segment]*vectors[target_segment]
    return target,{'teacher_segment_index':nearest,'teacher_target_segment_index':target_segment,
        'teacher_projection':projections[nearest].tolist(),'teacher_cross_track_error_m':float(distances[nearest]),
        'teacher_projected_arc_length_m':current_arc,'teacher_target_arc_length_m':target_arc,
        'teacher_remaining_arc_length_m':float(arcs[-1]-current_arc),
        'teacher_next_vertex_distance_m':float(arcs[nearest+1]-current_arc)}


class LookaheadCarryTeacher(GeometricCarryTeacher):
    def __init__(self,env):
        super().__init__(env)
        self.path=clean_polyline(self.path)

    def action(self,observation):
        base=observation['achieved_goal']*self.env.goal_scale
        target,labels=projected_target(self.path,base)
        if self.holding:
            target=observation['desired_goal']*self.env.goal_scale
        action=(target-base)/self.env.delta_scale
        action/=max(1.,float(np.linalg.norm(action)))
        self.labels={**labels,'teacher_waypoint_index':labels['teacher_segment_index']+1,
            'teacher_waypoint':np.asarray(self.path[labels['teacher_segment_index']+1]).tolist(),
            'teacher_local_target':target.tolist(),'teacher_lookahead_m':.15,
            'teacher_subphase':'goal_hold' if self.holding else ('turn_approach'
                if labels['teacher_segment_index']<len(self.path)-2 and labels['teacher_next_vertex_distance_m']<.35 else 'straight')}
        return action.astype(np.float32),'retained_goal_hold' if self.holding else 'astar_polyline_lookahead'


def collect_episode(env,case,index,model,teacher_weight,stop,deadline,role):
    observation=start_case(env,case);teacher=LookaheadCarryTeacher(env)
    rows=[];first_goal=None;hold_count=0;previous_retained=False
    for step in range(env.horizon):
        teacher_action,phase=teacher.action(observation)
        learner_action=(np.asarray(model.predict(observation,deterministic=True)[0],dtype=np.float32)
                        if teacher_weight<1. else np.zeros(2,dtype=np.float32))
        action=(teacher_weight*teacher_action+(1.-teacher_weight)*learner_action).astype(np.float32)
        effective=action.astype(float)/max(1.,float(np.linalg.norm(action.astype(float))))
        following,reward,terminated,environment_truncated,info=env.step(action);info=copy.deepcopy(info)
        retained=bool(info['is_success'] and info['bilateral_contact'] and info['retention_ok'] and not terminated)
        if retained:
            if first_goal is None:first_goal=step+1
            hold_count=hold_count+1 if previous_retained else 0
        else:hold_count=0
        teacher.holding=retained;previous_retained=retained
        timeout_reason=None
        if not terminated:
            if hold_count>=HOLD_DECISIONS:timeout_reason='completed_20_retained_goal_decisions'
            elif stop.requested:timeout_reason=stop.signal_name
            elif time.perf_counter()>=deadline:timeout_reason='collector_wall_budget'
        collector_timeout=timeout_reason is not None
        truncated=bool(environment_truncated or collector_timeout)
        info.update(environment_truncated=bool(environment_truncated),collector_timeout=collector_timeout,
            collector_timeout_reason=timeout_reason,**{'TimeLimit.truncated':bool(truncated and not terminated)})
        rows.append({'episode':index,'step':step,'observation':copy.deepcopy(observation),'action':action.copy(),
            'effective_action':effective.copy(),'teacher_action':teacher_action.copy(),'learner_action':learner_action.copy(),
            'teacher_weight':teacher_weight,'teacher_phase':phase,**teacher.labels,
            'next_observation':copy.deepcopy(following),'reward':float(reward),'terminated':bool(terminated),
            'truncated':truncated,'info':info})
        observation=following
        if terminated or truncated:break
    final=rows[-1]
    passed=bool(hold_count>=HOLD_DECISIONS and final['info']['is_success'] and final['info']['bilateral_contact']
        and not final['terminated'] and not any(row['info']['collision'] for row in rows))
    summary={**case,'episode':index,'collection_role':role,'teacher_weight':teacher_weight,'decisions':len(rows),
        'successful_physical_carry_and_hold':passed,'first_goal_decision':first_goal,
        'additional_retained_goal_decisions':hold_count,'desired_goal':env.goal.tolist(),
        'final_goal_distance_m':float(final['info']['goal_distance']),'final_base_xy_m':(following['achieved_goal']*20).tolist(),
        'physical_failure':final['terminated'],'failure_reason':final['info']['failure_reason'],
        'collector_timeout':final['info']['collector_timeout'],'collector_timeout_reason':final['info']['collector_timeout_reason'],
        'source_action_calls':sum(int(row['info']['action_calls_delta']) for row in rows),
        'physics_ticks':sum(int(row['info']['physics_ticks_delta']) for row in rows),
        'bilateral_contact_fraction':float(np.mean([row['info']['bilateral_contact'] for row in rows])),
        'max_cross_track_error_m':max(row['teacher_cross_track_error_m'] for row in rows),
        'teacher_path':[np.asarray(point).tolist() for point in teacher.path]}
    if summary['source_action_calls']!=env.native.step_count or summary['physics_ticks']!=env.native.physics_steps:
        raise ValueError('Physical counters disagree with actual transitions')
    return {'summary':summary,'transitions':rows}


def validate(env,episode):
    rows=episode['transitions'];goal=rows[0]['observation']['desired_goal']
    for i,row in enumerate(rows):
        for key in row['observation']:
            if i and not np.array_equal(rows[i-1]['next_observation'][key],row['observation'][key]):raise ValueError('Continuity')
        for field in ('observation','next_observation'):
            if not env.observation_space.contains(row[field]) or not np.array_equal(row[field]['desired_goal'],goal):raise ValueError('Fixed goal/observation')
        weight=row['teacher_weight'];blend=(weight*row['teacher_action']+(1.-weight)*row['learner_action']).astype(np.float32)
        if not np.array_equal(row['action'],blend) or not env.action_space.contains(blend):raise ValueError('Executed blend')
        if not np.array_equal(row['effective_action'],blend.astype(float)/max(1.,float(np.linalg.norm(blend.astype(float))))):raise ValueError('Projection')
        if (row['terminated'] or row['truncated'])!=(i==len(rows)-1):raise ValueError('Episode boundary')
        if row['info']['TimeLimit.truncated']!=(row['truncated'] and not row['terminated']):raise ValueError('Timeout boundary')
    actual=np.asarray([row['reward'] for row in rows],dtype=np.float32)
    recomputed=env.compute_reward(np.stack([row['next_observation']['achieved_goal'] for row in rows]),
        np.stack([row['observation']['desired_goal'] for row in rows]),[row['info'] for row in rows])
    if not np.array_equal(actual,recomputed):raise ValueError('HER rewards')


def save(path,episodes,metadata):
    data={'schema_version':1,'kind':'physical_lookahead_teacher_data','training_only':True,
          'teacher_version':VERSION,'metadata':metadata,'episodes':episodes}
    pending=path.with_suffix('.pending.pkl')
    with pending.open('wb') as handle:pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)
    pending.replace(path);return sha(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='carry_lookahead_demos_v1')
    parser.add_argument('--seed',type=int,default=54000)
    parser.add_argument('--timeout-seconds',type=float,default=900.)
    parser.add_argument('--resume-setup',action='store_true',help='Resume a failed setup with zero collected episodes, preserving all setup costs.')
    args=parser.parse_args()
    if not 1<=args.timeout_seconds<=1800:parser.error('Bounded wall budget required')
    torch.set_num_threads(1)
    source=ROOT/'runs/carry_bc_geometry02';source_model=source/'checkpoints/best/model.zip'
    if sha(source_model)!=MODEL_SHA or not (source/'artifacts/result.json').is_file():raise ValueError('Frozen completed geometry02 model required')
    logger=RunLogger(args.run_id,kind='validation',skill='carry',algorithm='Lookahead teacher validation and correction collection',
        resume=args.resume_setup,status='running',name='Carry lookahead data',budget=14000,summary='Training-only 0.15 m lookahead; pure teacher preflight before 85/15 mixed collection.')
    artifacts=logger.directory/'artifacts';inputs=artifacts/'inputs';inputs.mkdir(parents=True,exist_ok=args.resume_setup)
    if not args.resume_setup:
        shutil.copy2(source_model,inputs/'learner.zip')
        for original,name in ((source/'config.json','bc_config.json'),(source/'artifacts/result.json','bc_result.json'),
                              (source/'checkpoints/best/evaluation.json','bc_selected_evaluation.json')):shutil.copy2(original,inputs/name)
    if sha(inputs/'learner.zip')!=MODEL_SHA:raise ValueError('Frozen input model differs')
    sources=artifacts/(f'source_resume_{time.time_ns()}' if args.resume_setup else 'source');sources.mkdir();hashes={}
    for name in ('scripts/collect_carry_lookahead.py','scripts/collect_carry_corrections.py','scripts/generate_carry_demos.py',
                 'pnr/carry.py','pnr/carry_features.py','pnr/features.py','pnr/env.py','pnr/reference.py','pnr/planning.py','pnr/policies.py'):
        payload=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(payload).hexdigest();(sources/name.replace('/','__')).write_bytes(payload)
    metadata={'skill':'carry','scene_seed':42,'object_idx':14,'collection_seed':args.seed,'horizon':1000,
        'teacher_version':VERSION,'teacher_lookahead_m':.15,'fixed_goal_per_episode':True,'training_only':True,
        'policy_observations_modified':False,'teacher_at_learned_inference':False,'new_actor_or_critic_updates':0,
        'learner_source_run':'carry_bc_geometry02','learner_sha256':MODEL_SHA,'learner_deterministic':True,
        'planned_preflight_episodes':2,'planned_mixed_episodes':12,'teacher_weight':.85,'learner_weight':.15,
        'source_hashes':hashes,'collection_status':'running','metadata_finalized':False,
        'setup_source_action_calls':0,'setup_physics_ticks':0,'initialization_physics_ticks':0,
        'preflight_source_action_calls':0,'preflight_physics_ticks':0,'preflight_decisions':0,
        'demonstration_source_action_calls':0,'demonstration_physics_ticks':0,'policy_decisions':0,
        'action_semantics':'Actual env input is action; teacher_action is a separate BC label; effective_action records the unchanged unit-disk projection.',
        'accounting_semantics':'Demonstration counters describe mixed episodes only. Preflight counters are separate additional physical costs; total counters include both plus setup and settling.'}
    prior_wall_seconds=0.
    if args.resume_setup:
        previous=json.loads((artifacts/'manifest.json').read_text())
        if previous.get('preflight') or previous.get('episodes'):raise ValueError('This resume supports zero collected episodes only')
        archive=artifacts/f'failed_setup_{time.time_ns()}.json';shutil.copy2(artifacts/'manifest.json',archive)
        for key in ('setup_source_action_calls','setup_physics_ticks','initialization_physics_ticks'):metadata[key]=previous[key]
        prior_wall_seconds=float(previous['wall_seconds'])
        metadata['previous_failed_setups']=previous.get('previous_failed_setups',[])+[{'manifest':str(archive.relative_to(logger.directory)),
            'error':previous['error'],'source_hashes':previous['source_hashes'],'source_action_calls':previous['setup_source_action_calls'],
            'physics_ticks':previous['setup_physics_ticks'],'initialization_physics_ticks':previous['initialization_physics_ticks']}]
        atomic_json(artifacts/'resume_config.json',vars(args)|metadata)
    else:atomic_json(logger.directory/'config.json',vars(args)|metadata)
    preflight=[];episodes=[];env=None;started=time.perf_counter();deadline=started+args.timeout_seconds
    def persist(final=False):
        metadata.update(preflight_decisions=sum(len(e['transitions']) for e in preflight),
            policy_decisions=sum(len(e['transitions']) for e in episodes),wall_seconds=prior_wall_seconds+time.perf_counter()-started,
            metadata_finalized=final,preflight_episodes_collected=len(preflight),episodes_collected=len(episodes))
        for field in ('source_action_calls','physics_ticks'):
            metadata['preflight_'+field]=sum(e['summary'][field] for e in preflight)
            metadata['demonstration_'+field]=sum(e['summary'][field] for e in episodes)
            metadata['total_'+field]=metadata['preflight_'+field]+metadata['demonstration_'+field]+metadata['setup_'+field]
        metadata['total_physics_ticks']+=metadata['initialization_physics_ticks']
        files={}
        if preflight:files['preflight.pkl']=save(artifacts/'preflight.pkl',preflight,copy.deepcopy(metadata)|{'dataset_role':'pure_teacher_preflight'})
        if episodes:files['corrections.pkl']=save(artifacts/'corrections.pkl',episodes,copy.deepcopy(metadata)|{'dataset_role':'mixed_policy_training'})
        atomic_json(artifacts/'manifest.json',jsonable(metadata|{'datasets':files,'preflight':[e['summary'] for e in preflight],
            'episodes':[e['summary'] for e in episodes]}))
        logger.metric(metadata['preflight_decisions']+metadata['policy_decisions'],scope='training_data',
            preflight_episodes=len(preflight),mixed_episodes=len(episodes),
            preflight_passes=sum(e['summary']['successful_physical_carry_and_hold'] for e in preflight),
            mixed_passes=sum(e['summary']['successful_physical_carry_and_hold'] for e in episodes),
            total_source_action_calls=metadata['total_source_action_calls'],total_physics_ticks=metadata['total_physics_ticks'])
        return files
    with GracefulStop() as stop:
        try:
            model=load_policy_checkpoint(inputs/'learner.zip',device='cpu')
            env=CarrySkillEnv(scene_seed=42,object_idx=14,seed=args.seed,horizon=1000,evaluation='full',failure_cost=200.,adaptive_horizon=False)
            metadata.update(setup_source_action_calls=metadata['setup_source_action_calls']+env.setup_action_calls,
                setup_physics_ticks=metadata['setup_physics_ticks']+env.setup_physics_ticks,
                initialization_physics_ticks=metadata['initialization_physics_ticks']+env.native.settle_steps,scene=env.native.scene_manifest())
            cases,corners=bounded_cases(env)
            metadata.update(cases=cases,route_corners=corners,route_length_m=env.route_length,route_snapshots=len(env.snapshots))
            pure_cases=[cases[0],next(case for case in cases if case['kind']=='segment' and case['segment_length_m']==5.)]
            for index,case in enumerate(pure_cases):
                if stop.requested or time.perf_counter()>=deadline:break
                episode=collect_episode(env,case,index,model,1.,stop,deadline,'pure_teacher_preflight');preflight.append(episode)
                persist();validate(env,episode)
                print(json.dumps(jsonable({k:v for k,v in episode['summary'].items() if k!='teacher_path'})),flush=True)
            preflight_passed=len(preflight)==2 and all(e['summary']['successful_physical_carry_and_hold'] for e in preflight)
            metadata['preflight_passed']=preflight_passed
            if preflight_passed:
                logger.event('Both physical teacher preflight cases passed; beginning the authorized 12 mixed cases.')
                for index,case in enumerate(cases):
                    if stop.requested or time.perf_counter()>=deadline:break
                    episode=collect_episode(env,case,index,model,.85,stop,deadline,'mixed_policy_training');episodes.append(episode)
                    persist();validate(env,episode)
                    print(json.dumps(jsonable({k:v for k,v in episode['summary'].items() if k!='teacher_path'})),flush=True)
            if sha(source_model)!=MODEL_SHA or sha(inputs/'learner.zip')!=MODEL_SHA:raise RuntimeError('Frozen policy changed')
            if any(sha(ROOT/name)!=value for name,value in hashes.items()):raise RuntimeError('Source changed; inspect frozen provenance')
            metadata.update(collection_status='complete' if len(episodes)==12 else ('preflight_failed' if len(preflight)==2 and not preflight_passed else 'stopped'),
                validation={'fixed_goals':True,'actual_blended_actions':True,'separate_teacher_targets':True,
                    'reward_recomputation':True,'actual_state_continuity':True,'failed_episodes_discarded':0,'fake_transitions':0})
            files=persist(final=True)
            for name in files:logger.artifact('artifacts/'+name,'Recorded '+name.replace('.pkl','')+' physical trajectories','file')
            logger.artifact('artifacts/manifest.json','Preflight gate, all outcomes, provenance and physical costs','file')
            logger.finish('complete' if metadata['collection_status'] in ('complete','preflight_failed') else 'stopped',
                f"Preflight {sum(e['summary']['successful_physical_carry_and_hold'] for e in preflight)}/2; mixed {len(episodes)}/12 collected. All data are training-only.")
            print(json.dumps(jsonable({'status':metadata['collection_status'],'datasets':files,'metadata':{k:v for k,v in metadata.items() if k.endswith('calls') or k.endswith('ticks')}})),flush=True)
        except BaseException as error:
            metadata.update(collection_status='failed',error=f'{type(error).__name__}: {error}');persist()
            logger.finish('failed',metadata['error']);raise
        finally:
            if env is not None:env.close()


if __name__=='__main__':main()
