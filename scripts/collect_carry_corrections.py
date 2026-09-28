#!/usr/bin/env python3
"""Bounded physical carry correction data with separate executed/teacher actions.

A* routes are training-only teacher inputs. Policy-visible goals remain fixed;
all transitions, including failures and collector timeouts, are retained.
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
from scripts.generate_carry_demos import (GeometricCarryTeacher,collection_cases,start_case,
                                         SEGMENT_LENGTHS,HOLD_DECISIONS)

TEACHER_WEIGHT=.85
LEARNER_WEIGHT=.15


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CorrectingHoldTeacher(GeometricCarryTeacher):
    """Counter learner drift during hold using the unchanged requested XY goal."""
    def action(self,observation):
        if not self.holding:return super().action(observation)
        error=(observation['desired_goal']-observation['achieved_goal'])*self.env.goal_scale
        action=error/self.env.delta_scale
        action/=max(1.,float(np.linalg.norm(action)))
        return action.astype(np.float32),'retained_goal_hold'


def bounded_cases(env):
    cases,corners=collection_cases(env)
    segments=[]
    for length in SEGMENT_LENGTHS:
        candidates=sorted((case for case in cases if case['kind']=='segment' and case['segment_length_m']==length),
                          key=lambda case:case['protocol_case_index'])
        for index in (0,len(candidates)//2):
            case=copy.deepcopy(candidates[index]);case['preflight_case']=False
            case['split_group']=f"corner_segment_scene42_length{length:g}_snapshot{case['snapshot_index']}"
            segments.append(case)
    full=sorted((copy.deepcopy(case) for case in cases if case['kind']=='full_route'),key=lambda case:case['correlated_repeat'])
    for case in full:
        case['preflight_case']=False
        case['split_group']='full_route_correlated_scene42'
        case['independent_performance_trial']=False
    # Cover every segment length early while retaining all four repeated routes.
    return [full[0],*segments[:4],full[1],*segments[4:],full[2],full[3]],corners


def waypoint_metadata(teacher,observation):
    index=teacher.index
    point=teacher.path[index]
    base=observation['achieved_goal']*teacher.env.goal_scale
    distance=float(np.linalg.norm(point-base))
    corner=False
    if 0<index<len(teacher.path)-1:
        before=point-teacher.path[index-1];after=teacher.path[index+1]-point
        norm=float(np.linalg.norm(before)*np.linalg.norm(after))
        corner=norm>1e-8 and float(np.dot(before,after))/norm<.85
    return {'teacher_waypoint_index':int(index),'teacher_waypoint':np.asarray(point).tolist(),
            'teacher_waypoint_distance_m':distance,
            'teacher_subphase':'goal_hold' if teacher.holding else ('turn_approach' if corner and distance<.35 else 'straight')}


def save_dataset(path,episodes,metadata):
    data={'schema_version':1,'kind':'physical_mixed_policy_corrections','training_only':True,
          'teacher_version':'astar-fixed-goal-active-hold-v2','metadata':metadata,'episodes':episodes}
    pending=path.with_suffix('.pending.pkl')
    with pending.open('wb') as handle:pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)
    pending.replace(path)
    return sha(path)


def validate_episode(env,episode):
    rows=episode['transitions'];goal=rows[0]['observation']['desired_goal']
    for index,row in enumerate(rows):
        if index and any(not np.array_equal(rows[index-1]['next_observation'][key],row['observation'][key]) for key in row['observation']):
            raise ValueError('Actual transition continuity failed')
        for name in ('observation','next_observation'):
            if not env.observation_space.contains(row[name]) or not np.array_equal(row[name]['desired_goal'],goal):
                raise ValueError('Observation or fixed-goal contract failed')
        blend=(TEACHER_WEIGHT*row['teacher_action']+LEARNER_WEIGHT*row['learner_action']).astype(np.float32)
        if not np.array_equal(row['action'],blend) or not env.action_space.contains(row['action']):
            raise ValueError('HER action differs from the actual environment input')
        effective=blend.astype(float)/max(1.,float(np.linalg.norm(blend.astype(float))))
        if not np.array_equal(row['effective_action'],effective):raise ValueError('Native adapter projection mismatch')
        if np.linalg.norm(row['teacher_action'])>1.+1e-6:raise ValueError('Invalid teacher label')
        if (row['terminated'] or row['truncated'])!=(index==len(rows)-1):raise ValueError('Episode boundary mismatch')
        if row['info']['TimeLimit.truncated']!=(row['truncated'] and not row['terminated']):raise ValueError('Timeout mask mismatch')
    rewards=env.compute_reward(np.stack([row['next_observation']['achieved_goal'] for row in rows]),
        np.stack([row['observation']['desired_goal'] for row in rows]),[row['info'] for row in rows])
    if not np.array_equal(rewards,np.asarray([row['reward'] for row in rows],dtype=np.float32)):
        raise ValueError('Actual rewards do not match HER recomputation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='carry_correction_demos_v1')
    parser.add_argument('--bc-run',default='carry_bc_geometry02')
    parser.add_argument('--seed',type=int,default=54000)
    parser.add_argument('--timeout-seconds',type=float,default=900.)
    parser.add_argument('--resume',action='store_true',help='Continue remaining durable cases, counting another physical setup.')
    args=parser.parse_args()
    if not 1<=args.timeout_seconds<=1800:parser.error('Collector wall budget must be1..1800 seconds')
    source=ROOT/'runs'/args.bc_run
    if not (source/'artifacts/result.json').is_file():raise ValueError('Wait for finalized BC result before freezing the policy')
    source_model=source/'checkpoints/best/model.zip';model_hash=sha(source_model)
    torch.set_num_threads(1)
    logger=RunLogger(args.run_id,kind='validation',skill='carry',algorithm='Mixed-policy correction collection',
        name='Physical carry corrections',budget=12000,resume=args.resume,status='running',
        summary='12 training-only episodes;85%geometric teacher+15%frozen carry policy. No actor training.')
    artifacts=logger.directory/'artifacts';inputs=artifacts/'inputs';inputs.mkdir(parents=True,exist_ok=args.resume)
    if not args.resume:
        shutil.copy2(source_model,inputs/'learner.zip')
        for origin,name in ((source/'config.json','bc_config.json'),(source/'artifacts/result.json','bc_result.json'),
                            (source/'checkpoints/best/evaluation.json','bc_selected_evaluation.json')):
            shutil.copy2(origin,inputs/name)
    if sha(inputs/'learner.zip')!=model_hash:raise ValueError('Frozen model copy differs')
    sources=artifacts/('source_resume_%d'%time.time_ns() if args.resume else 'source');sources.mkdir()
    hashes={}
    for name in ('scripts/collect_carry_corrections.py','scripts/generate_carry_demos.py',
                 'pnr/carry.py','pnr/carry_features.py','pnr/features.py','pnr/env.py','pnr/reference.py',
                 'pnr/planning.py','pnr/policies.py'):
        content=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(content).hexdigest()
        (sources/name.replace('/','__')).write_bytes(content)
    metadata={'skill':'carry','scene_seed':42,'object_idx':14,'collection_seed':args.seed,
        'horizon':1000,'planned_episodes':12,'segments_per_length':2,'segment_lengths_m':list(SEGMENT_LENGTHS),
        'full_route_correlated_repeats':4,'additional_goal_hold_decisions':HOLD_DECISIONS,'failure_cost':200.,
        'teacher_version':'astar-fixed-goal-active-hold-v2','teacher_weight':TEACHER_WEIGHT,'learner_weight':LEARNER_WEIGHT,
        'learner_source_run':args.bc_run,'learner_sha256':model_hash,'learner_deterministic':True,
        'teacher_at_learned_inference':False,'training_only':True,'learned_policy_training':False,
        'policy_observations_modified':False,'new_actor_or_critic_updates':0,'fixed_goal_per_episode':True,
        'action_semantics':'action is literal85/15 blend passed to CarrySkillEnv, used forHER; effective_action records its existing unit-disk projection.',
        'supervised_target_semantics':'teacher_action is the separate BC target at the original fixed goal; waypoint/phase metadata never enters policy observations.',
        'correlation_notice':'Four full routes share physical start/goal and deterministic blended controller; keep split_group together. Not independent policy evaluations.',
        'source_hashes':hashes,'metadata_finalized':False,'collection_status':'running',
        'policy_decisions':0,'demonstration_source_action_calls':0,'demonstration_physics_ticks':0,
        'setup_source_action_calls':0,'setup_physics_ticks':0,'initialization_physics_ticks':0}
    episodes=[];path=artifacts/'corrections.pkl';prior_wall_seconds=0.
    if args.resume:
        previous_hash=sha(path)
        with path.open('rb') as handle:previous=pickle.load(handle)
        previous_metadata=previous['metadata']
        for key in ('collection_seed','learner_sha256','learner_source_run','teacher_weight','learner_weight','horizon'):
            if previous_metadata[key]!=metadata[key]:raise ValueError('Resume protocol differs: '+key)
        if any(value!=hashes[name] for name,value in previous_metadata['source_hashes'].items()
               if name!='scripts/collect_carry_corrections.py'):
            raise ValueError('Resume simulation or teacher sources changed')
        archive=artifacts/f'corrections_before_resume_{time.time_ns()}.pkl'
        shutil.copy2(path,archive)
        episodes=previous['episodes'];metadata=previous_metadata
        for episode in episodes:
            episode['summary']['unit_disk_projected_decisions']=int(episode['summary']['unit_disk_projected_decisions'])
        metadata.setdefault('resumes',[]).append({'previous_dataset_sha256':previous_hash,
            'previous_dataset_archive':str(archive.relative_to(logger.directory)),
            'episodes_retained':len(episodes),'prior_error':metadata.pop('error',None),
            'source_hashes':hashes,'source_directory':str(sources.relative_to(logger.directory)),
            'reason':'Resume durable episodes after collector JSON count serialization failure.'})
        metadata.setdefault('physical_setup_invocations',[{'first_remaining_episode':0,
            'source_action_calls':metadata['setup_source_action_calls'],'physics_ticks':metadata['setup_physics_ticks'],
            'initialization_physics_ticks':metadata['initialization_physics_ticks']}])
        metadata.update(metadata_finalized=False,collection_status='running')
        prior_wall_seconds=float(metadata.get('wall_seconds',0.))
        atomic_json(artifacts/'resume_config.json',vars(args)|{'source_hashes':hashes})
    else:
        atomic_json(logger.directory/'config.json',vars(args)|metadata)
        atomic_json(artifacts/'source_hashes.json',hashes)
    env=None;started=time.perf_counter()
    with GracefulStop() as stop:
        try:
            model=load_policy_checkpoint(inputs/'learner.zip',device='cpu')
            env=CarrySkillEnv(scene_seed=42,object_idx=14,seed=args.seed,horizon=1000,
                evaluation='full',failure_cost=200.,adaptive_horizon=False)
            metadata.update(setup_source_action_calls=metadata['setup_source_action_calls']+env.setup_action_calls,
                setup_physics_ticks=metadata['setup_physics_ticks']+env.setup_physics_ticks,
                initialization_physics_ticks=metadata['initialization_physics_ticks']+env.native.settle_steps,
                scene=env.native.scene_manifest())
            metadata.setdefault('physical_setup_invocations',[]).append({'first_remaining_episode':len(episodes),
                'source_action_calls':env.setup_action_calls,'physics_ticks':env.setup_physics_ticks,
                'initialization_physics_ticks':env.native.settle_steps})
            cases,corners=bounded_cases(env)
            metadata.update(cases=cases,route_corners=corners,route_length_m=env.route_length,route_snapshots=len(env.snapshots))
            logger.update(setup_action_calls=metadata['setup_source_action_calls'],setup_physics_ticks=metadata['setup_physics_ticks'])
            for episode in episodes:validate_episode(env,episode)
            for index in range(len(episodes),len(cases)):
                case=cases[index]
                if stop.requested or time.perf_counter()-started>=args.timeout_seconds:break
                observation=start_case(env,case);teacher=CorrectingHoldTeacher(env)
                rows=[];first_goal=None;hold_count=0;previous_retained=False
                for step in range(env.horizon):
                    teacher_action,phase=teacher.action(observation)
                    labels=waypoint_metadata(teacher,observation)
                    learner_action=np.asarray(model.predict(observation,deterministic=True)[0],dtype=np.float32)
                    action=(TEACHER_WEIGHT*teacher_action+LEARNER_WEIGHT*learner_action).astype(np.float32)
                    effective=action.astype(float)/max(1.,float(np.linalg.norm(action.astype(float))))
                    following,reward,terminated,environment_truncated,info=env.step(action)
                    info=copy.deepcopy(info)
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
                        elif time.perf_counter()-started>=args.timeout_seconds:timeout_reason='collector_wall_budget'
                    collector_timeout=timeout_reason is not None
                    truncated=bool(environment_truncated or collector_timeout)
                    info.update(environment_truncated=bool(environment_truncated),collector_timeout=collector_timeout,
                        collector_timeout_reason=timeout_reason,**{'TimeLimit.truncated':bool(truncated and not terminated)})
                    rows.append({'episode':index,'step':step,'observation':copy.deepcopy(observation),
                        'action':action.copy(),'effective_action':effective.copy(),'teacher_action':teacher_action.copy(),
                        'learner_action':learner_action.copy(),'teacher_phase':phase,**labels,
                        'next_observation':copy.deepcopy(following),'reward':float(reward),
                        'terminated':bool(terminated),'truncated':truncated,'info':info})
                    observation=following
                    if terminated or truncated:break
                final=rows[-1];passed=bool(hold_count>=HOLD_DECISIONS and final['info']['is_success']
                    and final['info']['bilateral_contact'] and not final['terminated']
                    and not any(row['info']['collision'] for row in rows))
                summary={**case,'episode':index,'decisions':len(rows),'successful_physical_carry_and_hold':passed,
                    'first_goal_decision':first_goal,'additional_retained_goal_decisions':hold_count,
                    'desired_goal':env.goal.tolist(),'final_goal_distance_m':float(final['info']['goal_distance']),
                    'final_bilateral_contact':bool(final['info']['bilateral_contact']),
                    'physical_failure':final['terminated'],'failure_reason':final['info']['failure_reason'],
                    'collector_timeout':final['info']['collector_timeout'],'collector_timeout_reason':final['info']['collector_timeout_reason'],
                    'collision_decisions':sum(bool(row['info']['collision']) for row in rows),
                    'source_action_calls':sum(int(row['info']['action_calls_delta']) for row in rows),
                    'physics_ticks':sum(int(row['info']['physics_ticks_delta']) for row in rows),
                    'teacher_waypoints':len(teacher.path),'teacher_path':[np.asarray(x).tolist() for x in teacher.path],
                    'bilateral_contact_fraction':float(np.mean([row['info']['bilateral_contact'] for row in rows])),
                    'learner_teacher_action_mse':float(np.mean([(row['learner_action']-row['teacher_action'])**2 for row in rows])),
                    'unit_disk_projected_decisions':sum(bool(np.linalg.norm(row['action'])>1.) for row in rows)}
                if summary['source_action_calls']!=env.native.step_count or summary['physics_ticks']!=env.native.physics_steps:
                    raise ValueError('Physical counters differ from actual transitions')
                episodes.append({'summary':summary,'transitions':rows})
                metadata.update(policy_decisions=sum(ep['summary']['decisions'] for ep in episodes),
                    demonstration_source_action_calls=sum(ep['summary']['source_action_calls'] for ep in episodes),
                    demonstration_physics_ticks=sum(ep['summary']['physics_ticks'] for ep in episodes),
                    wall_seconds=prior_wall_seconds+time.perf_counter()-started)
                metadata['collection_source_action_calls']=metadata['demonstration_source_action_calls']
                metadata['collection_physics_ticks']=metadata['demonstration_physics_ticks']
                metadata['total_source_action_calls']=metadata['setup_source_action_calls']+metadata['demonstration_source_action_calls']
                metadata['total_physics_ticks']=metadata['setup_physics_ticks']+metadata['demonstration_physics_ticks']+metadata['initialization_physics_ticks']
                save_dataset(path,episodes,metadata)
                validate_episode(env,episodes[-1])
                logger.metric(metadata['policy_decisions'],scope='training_data',metric_source='correction_collection',
                    episodes=len(episodes),successful_episodes=sum(ep['summary']['successful_physical_carry_and_hold'] for ep in episodes),
                    source_action_calls=metadata['demonstration_source_action_calls'],physics_ticks=metadata['demonstration_physics_ticks'],
                    total_source_action_calls=metadata['total_source_action_calls'],total_physics_ticks=metadata['total_physics_ticks'])
                print(json.dumps({key:value for key,value in summary.items() if key!='teacher_path'}),flush=True)
            if sha(source_model)!=model_hash or sha(inputs/'learner.zip')!=model_hash:raise RuntimeError('Policy input changed during collection')
            if any(sha(ROOT/name)!=value for name,value in hashes.items()):raise RuntimeError('Source changed during collection; inspect frozen provenance')
            metadata.update(metadata_finalized=True,collection_status='complete' if len(episodes)==12 else 'stopped',
                episodes_collected=len(episodes),wall_seconds=prior_wall_seconds+time.perf_counter()-started)
            dataset_hash=save_dataset(path,episodes,metadata)
            manifest={**metadata,'dataset_sha256':dataset_hash,'episodes':[ep['summary'] for ep in episodes],
                'validation':{'executed_blends_verified':True,'teacher_labels_separate':True,'fixed_goals_verified':True,
                    'reward_recomputation_verified':True,'contiguous_actual_states':True,'failed_episodes_discarded':0,'padding_transitions':0},
                'limitation':'Mixed-policy training data, not learned full-route performance evidence.'}
            atomic_json(artifacts/'manifest.json',manifest)
            logger.artifact('artifacts/corrections.pkl','Actual physical carry transitions and separate teacher labels','file')
            logger.artifact('artifacts/manifest.json','All outcomes, immutable sources and full physical costs','file')
            logger.finish('complete' if len(episodes)==12 else 'stopped',
                f"Saved {len(episodes)}/12 mixed-policy episodes;{sum(ep['summary']['successful_physical_carry_and_hold'] for ep in episodes)} retained-goal outcomes. No actor training.")
            print(json.dumps({'dataset_sha256':dataset_hash,'counts':{key:metadata[key] for key in ('policy_decisions','demonstration_source_action_calls','demonstration_physics_ticks','setup_source_action_calls','setup_physics_ticks','initialization_physics_ticks','total_source_action_calls','total_physics_ticks')}}),flush=True)
        except BaseException as error:
            metadata.update(collection_status='failed',error=f'{type(error).__name__}: {error}',wall_seconds=prior_wall_seconds+time.perf_counter()-started)
            if episodes:save_dataset(path,episodes,metadata)
            atomic_json(artifacts/'manifest.json',{**metadata,'episodes':[ep['summary'] for ep in episodes]})
            logger.finish('failed',metadata['error']);raise
        finally:
            if env is not None:env.close()


if __name__=='__main__':main()
