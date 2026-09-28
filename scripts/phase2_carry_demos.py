#!/usr/bin/env python3
"""Collect counted multi-maze physical demonstrations with corrective coverage."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from pnr.general_carry import GeneralCarryEnv, route_action
from pnr.planning import maze_astar
from pnr.scenes import load_scene_bank, scene_records
from pnr.telemetry import RunLogger, atomic_json


def worker(task):
    layout, index, out, n_episodes, delta_scale = task
    out=Path(out); env=None; episodes=[]
    with (out/f'worker_{index:03d}.log').open('w') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        started=time.monotonic()
        summary={'layout_id':layout['layout_id'],'maze_seed':layout['maze_seed'],
                 'maze_hash':layout['maze_hash'],'success':False,'episodes':[],
                 'setup_source_action_calls':0,'setup_physics_ticks':0,
                 'initialization_physics_ticks':0,'source_action_calls':0,'physics_ticks':0}
        try:
            env=GeneralCarryEnv(maze_seed=layout['maze_seed'],scene_seed=42,
                object_idx=14,seed=9100+index,horizon=900,delta_scale=delta_scale)
            summary.update(setup_source_action_calls=env.setup_action_calls,
                           setup_physics_ticks=env.setup_physics_ticks,
                           initialization_physics_ticks=env.initialization_physics_ticks,
                           route_length_m=env.route_length,physical_snapshots=len(env.starts))
            rng=np.random.default_rng(202600+index)
            for e in range(n_episodes):
                full=e<2
                env.set_curriculum_level(2 if e%2 else 3)
                obs,_=env.reset(options={'full':full})
                goal=obs['desired_goal']*env.goal_scale
                path=maze_astar(env.native.cube_positions, env.controller.base_position(),goal,
                               table_bounds=env.native.p.getAABB(env.native.table_id))
                rows=[];succeeded=False;noise_scale=0. if e==0 else .18
                for step in range(env._episode_horizon):
                    teacher=route_action(path,obs['achieved_goal']*env.goal_scale,goal,env.delta_scale)
                    noise=rng.normal(0.,noise_scale,2)
                    if np.linalg.norm(obs['achieved_goal']*env.goal_scale-goal)<.25:
                        noise*=.15
                    action=np.clip(teacher+noise,-1.,1.).astype(np.float32)
                    nxt,reward,terminated,truncated,info=env.step(action)
                    if step+1==env._episode_horizon and not terminated:truncated=True
                    rows.append({'observation':obs,'action':action,'teacher_action':teacher,
                                 'next_observation':nxt,'reward':reward,'terminated':terminated,
                                 'truncated':truncated,'info':info})
                    summary['source_action_calls']+=int(info['action_calls_delta'])
                    summary['physics_ticks']+=int(info['physics_ticks_delta'])
                    succeeded |= bool(info['is_success'])
                    obs=nxt
                    if terminated or truncated:break
                episodes.append({'layout_id':layout['layout_id'],'maze_hash':layout['maze_hash'],
                                 'object_idx':14,'full_route':full,'corrective_noise_std':noise_scale,
                                 'success':succeeded and not terminated,'transitions':rows})
                summary['episodes'].append({'full_route':full,'success':succeeded and not terminated,
                    'transitions':len(rows),'failure_reason':info.get('failure_reason'),
                    'goal_distance':info['goal_distance']})
            summary['success']=all(e['success'] for e in summary['episodes'])
        except Exception as exc:
            summary['error']=repr(exc)
            if env is None:
                costs=getattr(exc,'setup_accounting',{})
                summary.update(setup_source_action_calls=costs.get('source_action_calls',0),
                    setup_physics_ticks=costs.get('physics_ticks',0),
                    initialization_physics_ticks=costs.get('initialization_physics_ticks',0))
        finally:
            if env is not None:env.close()
        summary['wall_seconds']=time.monotonic()-started
        shard=out/f'layout_{index:03d}.pkl'
        shard.write_bytes(pickle.dumps({'episodes':episodes,'summary':summary},protocol=5))
        atomic_json(out/f'layout_{index:03d}.json',summary)
        return str(shard),summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--bank',default='configs/phase2_scene_bank.json')
    p.add_argument('--start-layout',type=int,default=0);p.add_argument('--layouts',type=int,default=16);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--episodes',type=int,default=4);p.add_argument('--delta-scale',type=float,default=.075)
    args=p.parse_args();bank=load_scene_bank(args.bank)
    layouts=scene_records(bank,'train')[args.start_layout:args.start_layout+args.layouts]
    logger=RunLogger(args.run_id,kind='reference',skill='carry',algorithm='Physical training demonstrations',
        name='Multi-maze carry data',summary='Training layouts only; physical demonstrations with corrective action noise.')
    out=logger.directory/'artifacts';out.mkdir()
    atomic_json(logger.directory/'config.json',dict(vars(args),partition='train',object_idx=14,
                bank_sha256=bank['manifest_sha256'],reward='sparse plus goal-consistent Euclidean potential, gamma=.995,weight=2'))
    episodes=[];summaries=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(worker,(layout,i,out,args.episodes,args.delta_scale)) for i,layout in enumerate(layouts)]
        for future in as_completed(jobs):
            path,summary=future.result();data=pickle.loads(Path(path).read_bytes())
            episodes.extend(data['episodes']);summaries.append(summary)
            transitions=sum(len(e['transitions']) for e in episodes)
            logger.metric(transitions,scope='training_data',metric_source='demonstration',
                layouts_completed=len(summaries),layouts_total=len(layouts),
                collection_episode_success_rate=float(np.mean([e['success'] for e in episodes])) if episodes else 0.,
                source_action_calls=sum(s['source_action_calls'] for s in summaries),
                physics_ticks=sum(s['physics_ticks'] for s in summaries),wall_seconds=time.monotonic()-started)
            logger.event(f"Collected {summary['layout_id']}: {len(data['episodes'])} episodes"+(f"; {summary['error']}" if 'error' in summary else ''))
    metadata={'protocol':'multimaze-physical-corrective-v1','partition':'train','bank_sha256':bank['manifest_sha256'],
        'layouts':len(layouts),'episodes':len(episodes),'object_idx':14,'gamma':.995,'shaping_weight':2.,
        'delta_scale':args.delta_scale,'repeat':5,'source_action_calls':sum(s['source_action_calls'] for s in summaries),
        'physics_ticks':sum(s['physics_ticks'] for s in summaries),
        'setup_source_action_calls':sum(s['setup_source_action_calls'] for s in summaries),
        'setup_physics_ticks':sum(s['setup_physics_ticks'] for s in summaries),
        'initialization_physics_ticks':sum(s['initialization_physics_ticks'] for s in summaries),
        'wall_seconds':time.monotonic()-started,'summaries':summaries,
        'teacher_at_evaluation':False,'actual_actions_preserved':True,'teacher_labels_for_BC_only':True}
    dataset={'schema_version':1,'metadata':metadata,'episodes':episodes}
    target=out/'demonstrations.pkl';target.write_bytes(pickle.dumps(dataset,protocol=5))
    metadata['dataset_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(out/'collection.json',metadata)
    logger.artifact('artifacts/collection.json','Physical collection results')
    logger.finish('complete' if episodes else 'failed',f'Collected {len(episodes)} episodes across {len(layouts)} declared training layouts; all failures retained.')
    print(json.dumps({k:v for k,v in metadata.items() if k!='summaries'}))

if __name__=='__main__':main()
