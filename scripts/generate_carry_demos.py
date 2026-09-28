#!/usr/bin/env python3
"""Collect explicitly privileged, training-only physical carry demonstrations.

A* waypoints belong only to the data-collection teacher. Recorded observations
remain the unchanged map/state and one fixed XY goal. Snapshot starts come from
the physical route bank; every recorded transition executes real native controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from pnr.carry import CarrySkillEnv
from pnr.planning import maze_astar
from pnr.telemetry import RunLogger, atomic_json

TEACHER_VERSION='astar-fixed-goal-carry-v1'
SEGMENT_LENGTHS=(1.2,2.5,5.,10.)
HOLD_DECISIONS=20


class GeometricCarryTeacher:
    """Counted physical path following; never supplied to a learned evaluator."""
    def __init__(self,env):
        self.env=env
        self.path=maze_astar(env.native.cube_positions,env.controller.base_position(),
            env.goal*env.goal_scale,footprint_radius=.36,
            table_bounds=env.native.p.getAABB(env.native.table_id))
        self.index=1
        self.holding=False

    def action(self,observation):
        if self.holding:
            return np.zeros(2,dtype=np.float32),'retained_goal_hold'
        base=observation['achieved_goal']*self.env.goal_scale
        while self.index<len(self.path)-1 and np.linalg.norm(base-self.path[self.index])<.012:
            self.index+=1
        action=(self.path[self.index]-base)/self.env.delta_scale
        action/=max(1.,float(np.linalg.norm(action)))
        return action.astype(np.float32),'astar_path_following'


def collection_cases(env):
    """Six distinct corner-crossing starts per distance, then four full repeats."""
    route=np.asarray(env.controller.path)
    corners=[]
    for index in range(1,len(route)-1):
        before,after=route[index]-route[index-1],route[index+1]-route[index]
        lengths=np.linalg.norm(before)*np.linalg.norm(after)
        if lengths>1e-8 and np.dot(before,after)/lengths<.85:
            nearest=int(np.argmin(np.linalg.norm(env.route_points-route[index],axis=1)))
            arc=float(env.route_arcs[nearest])
            if not corners or abs(arc-corners[-1]['arc_length_m'])>.1:
                corners.append({'arc_length_m':arc,'xy':route[index].tolist()})
    cases=[]
    for length in SEGMENT_LENGTHS:
        candidates=[]
        for index,start in enumerate(env.snapshots):
            if start.arc_length+length>env.route_length:
                continue
            crossed=[i for i,corner in enumerate(corners)
                     if start.arc_length+.10<corner['arc_length_m']<start.arc_length+length-.10]
            if crossed:
                candidates.append((index,crossed))
        if len(candidates)<6:
            raise RuntimeError(f'Only {len(candidates)} distinct corner-crossing starts for {length}m')
        indices=np.rint(np.linspace(0,len(candidates)-1,6)).astype(int)
        if len(set(indices.tolist()))!=6:
            raise RuntimeError('Segment sampling did not choose six distinct starts')
        for selection in indices:
            index,crossed=candidates[int(selection)]
            start=env.snapshots[index]
            cases.append({'kind':'segment','segment_length_m':length,'snapshot_index':index,
                'start_arc_length_m':start.arc_length,'start_base_xy':start.base_xy.tolist(),
                'goal_xy':env._point_at(start.arc_length+length).tolist(),'crossed_route_corners':crossed})
    for repeat in range(4):
        cases.append({'kind':'full_route','segment_length_m':env.route_length,'snapshot_index':0,
            'start_arc_length_m':0.,'start_base_xy':env.snapshots[0].base_xy.tolist(),
            'goal_xy':env.native.goal_pos[:2].tolist(),'crossed_route_corners':list(range(len(corners))),
            'correlated_repeat':repeat,'independent_performance_trial':False})
    # Validate short/medium/long corner segments and one entire route before
    # collecting the rest. These are retained members of the final 28 episodes.
    preflight_indices=[0,12,18,24]
    ordering=preflight_indices+[index for index in range(len(cases)) if index not in preflight_indices]
    ordered=[cases[index] for index in ordering]
    for index,case in enumerate(ordered):
        case['preflight_case']=index<4
        case['protocol_case_index']=ordering[index]
    return ordered,corners


def start_case(env,case):
    start=env.snapshots[case['snapshot_index']]
    env.native.restore_state(start.state,reset_counters=True)
    env._frozen_target=env.native.agent_qpos.copy()
    env._frozen_target[env.native.finger_slots]=0.
    env.goal=np.asarray(case['goal_xy'])/env.goal_scale
    env._episode_steps=0
    env._episode_horizon=env.horizon
    env._episode_success=False
    env._retention_failed=False
    env._missing_contact_calls=0
    env.episode_scope='training_teacher_'+case['kind']
    env.reset_count+=1
    return env._observation()


def collect_episode(env,case,index):
    observation=start_case(env,case)
    teacher=GeometricCarryTeacher(env)
    transitions=[]
    first_goal=None
    hold_count=0
    for step in range(env.horizon):
        action,phase=teacher.action(observation)
        before=copy.deepcopy(observation)
        following,reward,terminated,environment_truncated,info=env.step(action)
        info=copy.deepcopy(info)
        retained_goal=bool(info['is_success'] and info['bilateral_contact'] and info['retention_ok'] and not terminated)
        if retained_goal:
            if first_goal is None:
                first_goal=step+1
                teacher.holding=True
            else:
                hold_count+=1
        elif first_goal is not None:
            hold_count=0
            teacher.holding=False
        collector_timeout=bool(hold_count>=HOLD_DECISIONS and not terminated)
        truncated=bool(environment_truncated or collector_timeout)
        info.update(environment_truncated=bool(environment_truncated),collector_timeout=collector_timeout,
                    **{'TimeLimit.truncated':bool(truncated and not terminated)})
        transitions.append({'episode':index,'step':step,'observation':before,'action':action.copy(),
            'next_observation':copy.deepcopy(following),'reward':float(reward),
            'terminated':bool(terminated),'truncated':truncated,'info':info,'teacher_phase':phase})
        observation=following
        if terminated or truncated:
            break
    final=transitions[-1]
    passed=bool(hold_count>=HOLD_DECISIONS and final['info']['is_success']
        and final['info']['bilateral_contact'] and not final['terminated']
        and not any(t['info']['collision'] for t in transitions))
    summary={**case,'episode':index,'decisions':len(transitions),
        'successful_physical_carry_and_hold':passed,'first_goal_decision':first_goal,
        'additional_retained_goal_decisions':hold_count,'desired_goal':env.goal.tolist(),
        'final_goal_distance_m':float(final['info']['goal_distance']),
        'final_bilateral_contact':final['info']['bilateral_contact'],
        'physical_failure':final['terminated'],'failure_reason':final['info']['failure_reason'],
        'collector_timeout':final['info']['collector_timeout'],
        'collision_decisions':sum(bool(t['info']['collision']) for t in transitions),
        'source_action_calls':sum(int(t['info']['action_calls_delta']) for t in transitions),
        'physics_ticks':sum(int(t['info']['physics_ticks_delta']) for t in transitions),
        'teacher_waypoints':len(teacher.path),
        'bilateral_contact_fraction':float(np.mean([t['info']['bilateral_contact'] for t in transitions]))}
    if summary['source_action_calls']!=env.native.step_count or summary['physics_ticks']!=env.native.physics_steps:
        raise RuntimeError('Native interaction accounting disagrees with recorded transitions')
    return {'summary':summary,'transitions':transitions}


def validate_episode(env,episode):
    transitions=episode['transitions']
    goal=transitions[0]['observation']['desired_goal']
    for index,record in enumerate(transitions):
        for key in env.observation_space.spaces:
            if index and not np.array_equal(record['observation'][key],transitions[index-1]['next_observation'][key]):
                raise ValueError('Demonstration state continuity failed')
        if not np.array_equal(record['observation']['desired_goal'],goal) or not np.array_equal(record['next_observation']['desired_goal'],goal):
            raise ValueError('Policy-visible goal changed within a demonstration')
        reward=env.compute_reward(record['next_observation']['achieved_goal'],goal,record['info'])
        if not np.isclose(record['reward'],reward,rtol=0.,atol=1e-6):
            raise ValueError('Recorded reward violates current carry reward contract')
        if record['action'].shape!=(2,) or np.linalg.norm(record['action'])>1.+1e-6:
            raise ValueError('Invalid teacher action')
        if (record['terminated'] or record['truncated'])!=(index==len(transitions)-1):
            raise ValueError('Demonstration episode ending is inconsistent')
        if record['info']['TimeLimit.truncated']!=(record['truncated'] and not record['terminated']):
            raise ValueError('Timeout mask is inconsistent')


def save_dataset(path,episodes,metadata):
    data={'schema_version':1,'kind':'physical_training_demonstrations','training_only':True,
        'teacher_version':TEACHER_VERSION,'metadata':metadata,'episodes':episodes}
    temporary=path.with_suffix('.pending.pkl')
    with temporary.open('wb') as handle:
        pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='carry_physical_demos_v1')
    parser.add_argument('--seed',type=int,default=24000)
    args=parser.parse_args()
    logger=RunLogger(args.run_id,kind='validation',skill='carry',algorithm='Physical demonstration collection',
        name='Physical carry demonstrations',budget=28*1000,
        summary='Training-only A* teacher; validating four physical cases before expanded collection.')
    artifacts=logger.directory/'artifacts';artifacts.mkdir(exist_ok=True)
    sources=artifacts/'source';sources.mkdir(exist_ok=True)
    hashes={}
    for name in ('scripts/generate_carry_demos.py','pnr/carry.py','pnr/env.py','pnr/reference.py','pnr/planning.py'):
        content=(ROOT/name).read_bytes()
        hashes[name]=hashlib.sha256(content).hexdigest()
        (sources/name.replace('/','__')).write_bytes(content)
    atomic_json(artifacts/'source_hashes.json',hashes)
    configuration=vars(args)|{'training_only':True,'episodes':28,'segments_per_length':6,
        'segment_lengths_m':list(SEGMENT_LENGTHS),'full_route_correlated_repeats':4,'horizon':1000,
        'failure_cost':200.,'additional_goal_hold_decisions':HOLD_DECISIONS,'teacher_version':TEACHER_VERSION,
        'policy_observations_modified':False,'teacher_at_learned_inference':False,
        'privileged_teacher_inputs':['maze geometry','A* waypoints','physically collected route-bank starts'],
        'fixed_goal_per_episode':True,'learned_policy_training':False}
    atomic_json(logger.directory/'config.json',configuration)
    metadata={**configuration,'metadata_finalized':False,'collection_status':'running','policy_decisions':0,
        'demonstration_source_action_calls':0,'demonstration_physics_ticks':0,
        'setup_source_action_calls':0,'setup_physics_ticks':0,'initialization_physics_ticks':0,
        'skill':'carry','scene_seed':42,'object_idx':14,'source_hashes':hashes,
        'correlation_notice':'All data use one provided foam-brick/maze scene; the four full-route episodes repeat the same physical start and deterministic teacher and are not independent performance trials.'}
    env=None;episodes=[];started=time.perf_counter()
    try:
        env=CarrySkillEnv(scene_seed=42,object_idx=14,seed=args.seed,horizon=1000,
            evaluation='full',failure_cost=200.,adaptive_horizon=False)
        # Preserve setup costs even if route-case selection fails before collection.
        metadata.update(setup_source_action_calls=env.setup_action_calls,setup_physics_ticks=env.setup_physics_ticks,
            initialization_physics_ticks=env.native.settle_steps)
        cases,corners=collection_cases(env)
        metadata.update(setup_source_action_calls=env.setup_action_calls,setup_physics_ticks=env.setup_physics_ticks,
            initialization_physics_ticks=env.native.settle_steps,scene=env.native.scene_manifest(),
            route_length_m=env.route_length,route_corners=corners,cases=cases,
            route_snapshots=len(env.snapshots),observation_semantics='Unmodified 580-state/map + world XY achieved/desired goals divided by 20',
            policy_action_contract='Normalized XY delta <=0.045m, five native calls per decision',
            reward_semantics='Sparse -1/0 fixed-goal+retention reward; genuine physical terminal failure costs -200',
            episode_ending_semantics='Real physical failure or 1000-decision environment horizon; after 20 additional retained-goal decisions, explicit collector timeout without transition padding.')
        logger.update(setup_action_calls=env.setup_action_calls,setup_physics_ticks=env.setup_physics_ticks)
        for index,case in enumerate(cases):
            episode=collect_episode(env,case,index)
            episodes.append(episode)  # Retain genuine failure episodes as measured.
            validate_episode(env,episode)
            metadata.update(policy_decisions=sum(e['summary']['decisions'] for e in episodes),
                demonstration_source_action_calls=sum(e['summary']['source_action_calls'] for e in episodes),
                demonstration_physics_ticks=sum(e['summary']['physics_ticks'] for e in episodes),
                wall_seconds=time.perf_counter()-started)
            metadata['collection_source_action_calls']=metadata['demonstration_source_action_calls']
            metadata['collection_physics_ticks']=metadata['demonstration_physics_ticks']
            metadata['total_source_action_calls']=metadata['setup_source_action_calls']+metadata['demonstration_source_action_calls']
            metadata['total_physics_ticks']=metadata['initialization_physics_ticks']+metadata['setup_physics_ticks']+metadata['demonstration_physics_ticks']
            save_dataset(artifacts/'demonstrations.pkl',episodes,metadata)
            logger.metric(metadata['policy_decisions'],scope='training_data',metric_source='demonstration',
                episodes=len(episodes),successful_episodes=sum(e['summary']['successful_physical_carry_and_hold'] for e in episodes),
                source_action_calls=metadata['demonstration_source_action_calls'],physics_ticks=metadata['demonstration_physics_ticks'],
                total_source_action_calls=metadata['total_source_action_calls'],total_physics_ticks=metadata['total_physics_ticks'])
            print(json.dumps(episode['summary']),flush=True)
            if index==3:
                metadata['preflight_passed']=all(e['summary']['successful_physical_carry_and_hold'] for e in episodes)
                if not metadata['preflight_passed']:
                    raise RuntimeError('Four-case physical teacher preflight failed; all measured attempts retained, expanded collection stopped')
                logger.event('Four-case preflight passed, including full route; collecting remaining training-only demonstrations.')
                logger.update(summary='Physical teacher validated; collecting 24 corner segments and four correlated full-route repeats.')
        metadata.update(metadata_finalized=True,collection_status='complete',wall_seconds=time.perf_counter()-started)
        digest=save_dataset(artifacts/'demonstrations.pkl',episodes,metadata)
        atomic_json(artifacts/'manifest.json',{**metadata,'dataset_sha256':digest,'episodes':[e['summary'] for e in episodes]})
        logger.artifact('artifacts/demonstrations.pkl','Training-only physical carry replay')
        logger.artifact('artifacts/manifest.json','Measured teacher episodes, correlation and complete costs')
        logger.finish(summary=f"Saved {len(episodes)} training-only carry episodes, {metadata['policy_decisions']} actual decisions; no policy training or learned-performance claim.")
    except BaseException as error:
        metadata.update(collection_status='failed',error=f'{type(error).__name__}: {error}',wall_seconds=time.perf_counter()-started)
        if episodes:
            save_dataset(artifacts/'demonstrations.pkl',episodes,metadata)
        atomic_json(artifacts/'manifest.json',{**metadata,'episodes':[e['summary'] for e in episodes]})
        logger.artifact('artifacts/manifest.json','Retained collection attempts and failure details')
        logger.finish('failed',f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__=='__main__':
    main()
