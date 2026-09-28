#!/usr/bin/env python3
"""Collect counted TRAIN-only physical carry routes or off-route recoveries."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from pnr.general_carry import GeneralCarryEnv, route_action
from pnr.planning import maze_astar
from pnr.scenes import load_scene_bank, scene_records
from pnr.telemetry import RunLogger, atomic_json
from pnr.train import GracefulStop, ignore_worker_stop_signals


COST_FIELDS=('source_action_calls','physics_ticks','preparation_source_action_calls',
             'preparation_physics_ticks','setup_source_action_calls','setup_physics_ticks',
             'initialization_physics_ticks')


def counted_step(env,action,summary,preparation=False):
    """Count physical calls even if an unexpected exception interrupts a step."""
    before_calls,before_ticks=env.native.step_count,env.native.physics_steps
    try:
        return env.step(action)
    finally:
        prefix='preparation_' if preparation else ''
        summary[prefix+'source_action_calls']+=env.native.step_count-before_calls
        summary[prefix+'physics_ticks']+=env.native.physics_steps-before_ticks


def recovery_targets(layout,episodes):
    """Targets remain on the room side of the entry wall, away from borders."""
    entrance_y=5.5-layout['maze_entrance_cell'][0]
    result=[]
    offsets=[sign*distance for distance in (.75,1.5,2.5) for sign in (1.,-1.)]
    for offset in offsets[:episodes]:
        target=[1.5,entrance_y+offset]
        result.append({'requested_offset_y_m':offset,'target_xy':target,
            'geometry_skip':None if -4.35<=target[1]<=4.35 else 'outside conservative room y bounds'})
    return result


def prepare_recovery(env,target,summary,stop_path,max_decisions=250):
    """Execute a physical path; preserve the task goal and all retention history."""
    observation,reset_info=env.reset(options={'full':True})
    original_goal=observation['desired_goal'].copy()
    path=maze_astar(env.native.cube_positions,env.controller.base_position(),target,
                   table_bounds=env.native.p.getAABB(env.native.table_id))
    trace=[(observation['achieved_goal']*env.goal_scale).tolist()]
    for decision in range(max_decisions):
        if stop_path.exists():raise InterruptedError('Collection stop requested')
        position=observation['achieved_goal']*env.goal_scale
        if np.linalg.norm(position-target)<.075:
            break
        action=route_action(path,position,np.asarray(target),env.delta_scale)
        observation,_,terminated,truncated,info=counted_step(env,action,summary,True)
        trace.append((observation['achieved_goal']*env.goal_scale).tolist())
        if not np.array_equal(original_goal,observation['desired_goal']):
            raise RuntimeError('Preparation changed the fixed original task goal')
        if terminated or truncated:raise RuntimeError('Physical preparation failed: '+str(info.get('failure_reason')))
    else:
        raise RuntimeError('Physical preparation exceeded decision limit')
    if not env.native.contacts()['bilateral_contact'] or env._retention_failed:
        raise RuntimeError('Prepared start has no continuously retained grasp')
    # New collection horizon at the measured state; no simulator reset/restore,
    # native counter reset, contact-history reset, joint or object teleportation.
    env._episode_steps=0
    env._episode_horizon=env.horizon
    env.episode_scope='carry_recovery_original_exit_goal'
    return observation,{'reset_scope':reset_info['evaluation_scope'],'preparation_trace_xy':trace,
        'preparation_decisions':len(trace)-1,'actual_start_xy':trace[-1],
        'start_error_m':float(np.linalg.norm(np.asarray(trace[-1])-target)),
        'original_goal':(original_goal*env.goal_scale).tolist(),
        'state_origin':'physical carry actions from a saved physically prepared grasp'}


def collect_episode(env,obs,rng,noise_scale,summary,stop_path):
    goal=obs['desired_goal']*env.goal_scale
    path=maze_astar(env.native.cube_positions,env.controller.base_position(),goal,
                   table_bounds=env.native.p.getAABB(env.native.table_id))
    rows=[];succeeded=False;terminated=False;info={};stopped=False
    for step in range(env._episode_horizon):
        if stop_path.exists():
            stopped=True
            if rows:rows[-1]['truncated']=not rows[-1]['terminated']
            break
        teacher=route_action(path,obs['achieved_goal']*env.goal_scale,goal,env.delta_scale)
        noise=rng.normal(0.,noise_scale,2)
        if np.linalg.norm(obs['achieved_goal']*env.goal_scale-goal)<.25:noise*=.15
        action=np.clip(teacher+noise,-1.,1.).astype(np.float32)
        nxt,reward,terminated,truncated,info=counted_step(env,action,summary)
        if step+1==env._episode_horizon and not terminated:truncated=True
        if not np.array_equal(nxt['desired_goal'],obs['desired_goal']):
            raise RuntimeError('Goal changed inside recorded recovery')
        rows.append({'observation':obs,'action':action,'teacher_action':teacher,
            'next_observation':nxt,'reward':reward,'terminated':terminated,'truncated':truncated,'info':info})
        succeeded|=bool(info['is_success']);obs=nxt
        if terminated or truncated:break
    return rows,bool(succeeded and not terminated and not stopped),info,stopped


def worker(task):
    layout,index,out,config,parent_pid=task
    ignore_worker_stop_signals(parent_pid)
    out=Path(out);stop_path=out/'STOP';env=None;episodes=[]
    with (out/f'worker_{index:03d}.log').open('w') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        started=time.monotonic()
        summary={'layout_id':layout['layout_id'],'maze_seed':layout['maze_seed'],
                 'maze_hash':layout['maze_hash'],'success':False,'episodes':[],
                 'recovery_attempts':[],**{key:0 for key in COST_FIELDS}}
        try:
            env=GeneralCarryEnv(maze_seed=layout['maze_seed'],scene_seed=42,object_idx=14,
                seed=9100+index,horizon=900,delta_scale=config['delta_scale'],
                build_bank=config['mode']=='standard')
            summary.update(setup_source_action_calls=env.setup_action_calls,
                setup_physics_ticks=env.setup_physics_ticks,initialization_physics_ticks=env.initialization_physics_ticks,
                route_length_m=env.route_length,physical_snapshots=len(env.starts))
            rng=np.random.default_rng(202600+index)
            candidates=(recovery_targets(layout,config['episodes']) if config['mode']=='recovery'
                        else [None]*config['episodes'])
            for e,candidate in enumerate(candidates):
                if stop_path.exists():summary['stopped']=True;break
                preparation={};before={key:summary[key] for key in COST_FIELDS}
                attempt=None
                if candidate is not None:
                    attempt=dict(candidate,attempt=e,status='preparing')
                    summary['recovery_attempts'].append(attempt)
                    if candidate['geometry_skip']:
                        attempt['status']='skipped_geometry';continue
                    try:
                        obs,preparation=prepare_recovery(env,candidate['target_xy'],summary,stop_path)
                        attempt.update(preparation,status='prepared')
                    except Exception as error:
                        attempt.update(status='stopped' if isinstance(error,InterruptedError) else 'failed_preparation',
                            error=repr(error),preparation_source_action_calls=summary['preparation_source_action_calls']-before['preparation_source_action_calls'],
                            preparation_physics_ticks=summary['preparation_physics_ticks']-before['preparation_physics_ticks'])
                        if isinstance(error,InterruptedError):summary['stopped']=True;break
                        continue
                    full=False;noise_scale=config['recovery_noise_std']
                else:
                    requested_full=e<2;env.set_curriculum_level(2 if e%2 else 3)
                    obs,reset_info=env.reset(options={'full':requested_full})
                    full=reset_info['evaluation_scope']=='full_carry'
                    noise_scale=0. if e==0 else .18
                rows,succeeded,info,stopped=collect_episode(env,obs,rng,noise_scale,summary,stop_path)
                if rows:
                    original_goal=bool(np.linalg.norm(rows[0]['observation']['desired_goal']*env.goal_scale-env.native.goal_pos[:2])<1e-4)
                    episode={'layout_id':layout['layout_id'],'maze_hash':layout['maze_hash'],'object_idx':14,
                        'full_route':full,'original_exit_goal':original_goal,'collection_mode':config['mode'],
                        'corrective_noise_std':noise_scale,'success':succeeded,'transitions':rows}
                    if candidate is not None:episode['recovery_start']=dict(candidate,**preparation)
                    episodes.append(episode)
                    brief={'full_route':full,'original_exit_goal':original_goal,'success':succeeded,
                        'transitions':len(rows),'failure_reason':info.get('failure_reason'),
                        'goal_distance':info.get('goal_distance'),'collection_mode':config['mode']}
                    summary['episodes'].append(brief)
                if attempt is not None:
                    attempt.update(status='stopped' if stopped else ('success' if succeeded else 'failed_recovery'),
                        transitions=len(rows),failure_reason=info.get('failure_reason'),
                        **{key:summary[key]-before[key] for key in COST_FIELDS})
                # Retain durable physical evidence after every attempted episode.
                atomic_json(out/f'layout_{index:03d}_partial.json',summary)
                if stopped:summary['stopped']=True;break
            attempted=[a for a in summary['recovery_attempts'] if a['status']!='skipped_geometry']
            summary['success']=(bool(episodes) and all(e['success'] for e in summary['episodes'])
                                and all(a['status']=='success' for a in attempted))
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
    p.add_argument('--start-layout',type=int,default=0);p.add_argument('--layouts',type=int,default=16)
    p.add_argument('--layout-ids',nargs='+',default=None)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--delta-scale',type=float,default=.075)
    p.add_argument('--mode',choices=['standard','recovery'],default='standard')
    p.add_argument('--recovery-noise-std',type=float,default=.12)
    args=p.parse_args();bank=load_scene_bank(args.bank)
    available=scene_records(bank,'train')
    if args.layout_ids:
        indexed={layout['layout_id']:layout for layout in available}
        if len(set(args.layout_ids))!=len(args.layout_ids) or any(key not in indexed for key in args.layout_ids):
            p.error('Explicit layouts must be unique members of the frozen TRAIN partition')
        layouts=[indexed[key] for key in args.layout_ids]
    else:layouts=available[args.start_layout:args.start_layout+args.layouts]
    if not layouts or not 1<=args.workers<=6 or args.episodes<1:p.error('Need layouts, episodes>0 and1–6 workers')
    if args.mode=='recovery' and args.episodes>6:p.error('Recovery mode has at most six distinct offset attempts per layout')
    if args.recovery_noise_std<0 or not np.isfinite(args.recovery_noise_std):p.error('Noise std must be finite nonnegative')
    logger=RunLogger(args.run_id,kind='reference',skill='carry',algorithm='Physical training demonstrations',
        name='Physical carry recovery data' if args.mode=='recovery' else 'Multi-maze carry data',
        summary='TRAIN layouts only; physical off-route preparation and original-goal recovery.' if args.mode=='recovery'
                else 'TRAIN layouts only; physical demonstrations with corrective action noise.')
    out=logger.directory/'artifacts';out.mkdir()
    configuration=dict(vars(args),partition='train',object_idx=14,selected_layout_ids=[l['layout_id'] for l in layouts],
        bank_sha256=bank['manifest_sha256'],reward='sparse plus goal-consistent Euclidean potential, gamma=.995,weight=2')
    atomic_json(logger.directory/'config.json',configuration)
    sources={}
    for source in sorted((ROOT/'pnr').glob('*.py'))+[Path(__file__).resolve()]:
        relative=source.relative_to(ROOT);target=logger.directory/'sources'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        sources[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(logger.directory/'source_manifest.json',sources)
    episodes=[];summaries=[];started=time.monotonic();cancelled=0
    with GracefulStop() as stop, ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(worker,(layout,i,out,vars(args),os.getpid())) for i,layout in enumerate(layouts)}
        while pending:
            if stop.requested:
                (out/'STOP').touch()
                for future in list(pending):
                    if future.cancel():pending.remove(future);cancelled+=1
            ready,_=wait(pending,timeout=1.,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.remove(future)
                path,summary=future.result();data=pickle.loads(Path(path).read_bytes())
                episodes.extend(data['episodes']);summaries.append(summary)
                transitions=sum(len(e['transitions']) for e in episodes)
                costs={key:sum(s[key] for s in summaries) for key in COST_FIELDS}
                logger.metric(transitions,scope='training_data',metric_source='demonstration',
                    layouts_completed=len(summaries),layouts_total=len(layouts),
                    collection_episode_success_rate=float(np.mean([e['success'] for e in episodes])) if episodes else 0.,
                    **costs,wall_seconds=time.monotonic()-started)
                logger.event(f"Collected {summary['layout_id']}: {len(data['episodes'])} episodes"+
                             (f"; {summary['error']}" if 'error' in summary else ''))
        stopped=stop.requested
    metadata={'protocol':'multimaze-physical-originalgoal-recovery-v1' if args.mode=='recovery' else 'multimaze-physical-corrective-v2',
        'partition':'train','bank_sha256':bank['manifest_sha256'],'configuration':configuration,
        'layouts_requested':len(layouts),'layouts_completed':len(summaries),'cancelled_layouts':cancelled,
        'episodes':len(episodes),'object_idx':14,'gamma':.995,'shaping_weight':2.,'delta_scale':args.delta_scale,'repeat':5,
        **{key:sum(s[key] for s in summaries) for key in COST_FIELDS},
        'wall_seconds':time.monotonic()-started,'summaries':summaries,'source_hashes':sources,
        'teacher_at_evaluation':False,'actual_actions_preserved':True,'teacher_labels_for_BC_only':True,
        'preparation_in_replay':False,'all_failed_attempt_costs_retained':True}
    metadata['all_source_action_calls']=sum(metadata[k] for k in ('source_action_calls','preparation_source_action_calls','setup_source_action_calls'))
    metadata['all_physics_ticks']=sum(metadata[k] for k in ('physics_ticks','preparation_physics_ticks','setup_physics_ticks','initialization_physics_ticks'))
    dataset={'schema_version':1,'metadata':metadata,'episodes':episodes}
    target=out/'demonstrations.pkl';target.write_bytes(pickle.dumps(dataset,protocol=5))
    metadata['dataset_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(out/'collection.json',metadata);logger.artifact('artifacts/collection.json','Physical collection results')
    status='stopped' if stopped else ('complete' if episodes else 'failed')
    logger.finish(status,f'Collected {len(episodes)} episodes across {len(summaries)}/{len(layouts)} TRAIN layouts; failures and preparation counted.')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('summaries','source_hashes','configuration')}))


if __name__=='__main__':main()
