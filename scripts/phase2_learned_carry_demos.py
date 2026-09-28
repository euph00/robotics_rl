#!/usr/bin/env python3
"""Bounded TRAIN carry data from physically prepared frozen learned grasps.

Fixed8 TRAIN layouts x5 objects. Each original-start preparation is attempted
once. A retained handover supplies one clean full-route teacher episode and one
modest-noise episode restored from that exact physical state. A failed first
transport never removes its second declared attempt. All preparation/transport
failures and physics costs remain in the manifest; no scene replacement.

Opt-in three-variant mode instead executes independent original-start
preparations (one deterministic plus two native SAC samples), each followed
by one clean full-route teacher episode. It never restores a snapshot.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
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
from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.grasp_sampling import (SAMPLING_PROTOCOL, DEFAULT_COLLECTION_SEED, fixed_variants,
    validate_variant, variant_signature, predictor_for_variant)
from pnr.multi_carry import CarrySchema
from pnr.learned_carry import LearnedCarryPreparation
from pnr.policies import load_policy_checkpoint
from pnr.scenes import load_scene_bank, map_hash
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import freeze_checkpoints, plain
from scripts.phase2_carry_demos import collect_episode
from scripts.phase2_evaluate_system import freeze_sources, plan_digest, select_cases, sha256_file

PROTOCOL='learned-handover-multiobject-carry-v1'
DIVERSE_PROTOCOL='learned-handover-native-sac-variants-carry-v1'
OBJECTS=[14,5,2,7,10]
COST_FIELDS=('source_action_calls','physics_ticks','preparation_source_action_calls',
             'preparation_physics_ticks','setup_source_action_calls','setup_physics_ticks',
             'initialization_physics_ticks')
_GRASP=None

def initialize_worker(record):
    global _GRASP
    import torch
    torch.set_num_threads(1)
    if sha256_file(record['frozen_path'])!=record['sha256']:raise ValueError('Frozen grasp digest mismatch')
    _GRASP=load_policy_checkpoint(record['frozen_path'],device='cpu')
    if _GRASP.observation_space['observation'].shape!=(54,) or _GRASP.action_space.shape!=(4,):
        raise ValueError('Expected54-state learned grasp')


def atomic_pickle(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def expected_attempts(case_index,collection_seed=8401):
    return [{'attempt_index':i,'noise_std':noise,'rng_seed':collection_seed+2*case_index+i,
             'state_origin':'continuous_original_start' if i==0 else 'restored_physically_prepared_learned_handover'}
            for i,noise in enumerate((0.,.12))]


def preparation_schedule(cases, variants=1, collection_seed=8401):
    """Canonical case IDs remain unchanged; preparations get separate IDs."""
    if variants not in (1,3):raise ValueError('Use one or three preparation variants')
    if len({case['case_id'] for case in cases})!=len(cases):raise ValueError('Canonical cases must be unique')
    if any('preparation_variant' in case or 'preparation_id' in case for case in cases):
        raise ValueError('Expand canonical cases only once')
    if variants==1:return [dict(case) for case in cases]
    return [dict(case,preparation_id=case['case_id']+'__'+variant['variant_id'],preparation_variant=variant)
            for index,case in enumerate(cases) for variant in fixed_variants(index,collection_seed)]


def preparation_key(case):
    return case.get('preparation_id',case['case_id'])


def planned_attempts(case,index,collection_seed):
    if 'preparation_variant' not in case:return expected_attempts(index,collection_seed)
    variant=validate_variant(case['preparation_variant'])
    return [dict(attempt_index=0,noise_std=0.,rng_seed=variant['torch_seed'],state_origin='continuous_original_start')]


def attach_preparation_metadata(rows,case):
    """Replay inherits provenance via info; physical arrays/scalars remain untouched."""
    from copy import deepcopy
    variant=validate_variant(case['preparation_variant'])
    for row in rows:
        row['info']=dict(row['info'],case_id=case['case_id'],preparation_id=preparation_key(case),
                         preparation_variant=deepcopy(variant))


def validate_recorded_episode(episode,case):
    """Check actual transitions, continuity, map, reward and original physics costs."""
    rows=episode['transitions'];schema=CarrySchema()
    if not rows or not (rows[-1]['terminated'] or rows[-1]['truncated']):raise ValueError('Unclosed carry episode')
    goal=np.asarray(rows[0]['observation']['desired_goal'])
    calls=ticks=0
    for index,row in enumerate(rows):
        for label in ('action','teacher_action'):
            action=np.asarray(row[label])
            if action.shape!=(2,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.:raise ValueError('Invalid normalized carry action')
        for label in ('observation','next_observation'):
            obs=row[label]
            if set(obs)!=set(schema.observation_space.spaces):raise ValueError('Carry observation keys differ')
            for key,space in schema.observation_space.spaces.items():
                if np.asarray(obs[key]).shape!=space.shape or not np.isfinite(obs[key]).all():raise ValueError('Carry observation shape/values differ')
            if not np.array_equal(obs['desired_goal'],goal):raise ValueError('Carry goal changed')
            if map_hash(np.asarray(obs['observation'])[:264].reshape(12,22))!=case['maze_hash']:raise ValueError('Frozen TRAIN map differs')
        if index<len(rows)-1:
            if row['terminated'] or row['truncated']:raise ValueError('Internal carry terminal boundary')
            if any(not np.array_equal(row['next_observation'][key],rows[index+1]['observation'][key]) for key in row['next_observation']):raise ValueError('Discontinuous carry transitions')
        if 'preparation_variant' in case:
            info=row['info']
            if (info.get('case_id')!=case['case_id'] or info.get('preparation_id')!=preparation_key(case)
                    or validate_variant(info.get('preparation_variant'))!=case['preparation_variant']):
                raise ValueError('Transition preparation provenance differs')
        count=row['info']['action_calls_delta'];tick=row['info']['physics_ticks_delta']
        if (isinstance(count,bool) or not isinstance(count,(int,np.integer)) or not 1<=count<=5
                or isinstance(tick,bool) or not isinstance(tick,(int,np.integer)) or tick!=5*count):raise ValueError('Carry transition costs violate native five-tick contract')
        calls+=int(count);ticks+=int(tick)
    achieved=np.stack([row['next_observation']['achieved_goal'] for row in rows])
    rewards=schema.compute_reward(achieved,np.broadcast_to(goal,achieved.shape),[row['info'] for row in rows])
    if not np.allclose(rewards,[row['reward'] for row in rows],atol=1e-5,rtol=0):raise ValueError('Carry rewards differ from HER contract')
    return {'source_action_calls':calls,'physics_ticks':ticks}


def case_summary(results,cases):
    diverse=any('preparation_variant' in case for case in cases)
    ids={preparation_key(case) for case in cases};seen={preparation_key(result) for result in results}
    if len(seen)!=len(results) or seen-ids:raise ValueError('Duplicate or undeclared case/preparation')
    canonical={case['case_id'] for case in cases};missing=ids-seen
    costs={key:sum(result.get(key,0) for result in results) for key in COST_FIELDS}
    complete_cases=sum(all(preparation_key(c) in seen for c in cases if c['case_id']==key) for key in canonical)
    summary={'protocol':DIVERSE_PROTOCOL if diverse else PROTOCOL,'planned_cases':len(canonical),
        'completed_cases':complete_cases,'planned_preparations':len(cases),'completed_preparations':len(results),
        'prepared_handovers':sum(result.get('prepared_handover',False) for result in results),
        'planned_transport_opportunities':len(cases)*(1 if diverse else 2),
        'attempted_transports':sum(len(result.get('attempts',[])) for result in results),
        'successful_transports':sum(a.get('success',False) for result in results for a in result.get('attempts',[])),
        'successful_by_attempt':{str(i):sum(a.get('success',False) for result in results for a in result.get('attempts',[]) if a['attempt_index']==i) for i in ((0,) if diverse else (0,1))},
        'all_cases_retained':True,'full_task_evaluation':False,'partition':'train',
        'missing_case_ids':sorted({c['case_id'] for c in cases if preparation_key(c) in missing}),
        'missing_preparation_ids':sorted(missing),
        'fatal_collection_errors':[preparation_key(r) for r in results if r.get('fatal_collection_error')],
        'cost_accounting_complete':all(r.get('cost_accounting_complete',False) for r in results),
        **costs,'all_source_action_calls':sum(costs[k] for k in ('source_action_calls','preparation_source_action_calls','setup_source_action_calls')),
        'all_physics_ticks':sum(costs[k] for k in ('physics_ticks','preparation_physics_ticks','setup_physics_ticks','initialization_physics_ticks')),
        'transitions':sum(r.get('transitions',0) for r in results),'results':results}
    if diverse:
        summary['by_variant']={name:{'planned':sum(c['preparation_variant']['variant_id']==name for c in cases),
            'completed':sum(r['preparation_variant']['variant_id']==name for r in results),
            'prepared_handovers':sum(r.get('prepared_handover',False) for r in results if r['preparation_variant']['variant_id']==name),
            'successful_transports':sum(a.get('success',False) for r in results if r['preparation_variant']['variant_id']==name for a in r.get('attempts',[]))}
            for name in ('deterministic','stochastic_1','stochastic_2')}
    return summary


def worker(task):
    case,index,output,config=task;out=Path(output);key=preparation_key(case);native=system=None
    diverse='preparation_variant' in case;predictor=None
    started=time.monotonic();phase='preparation';active_attempt=None;physical_calls=0;episodes=[]
    summary={'case_id':case['case_id'],'case':case,'prepared_handover':False,'attempts':[],
             'planned_attempts':planned_attempts(case,index,config['collection_seed']),
             'retain_all_failures':True,'full_task_evaluation':False,'cost_accounting_complete':True,
             **{k:0 for k in COST_FIELDS},'transitions':0}
    if diverse:
        summary.update(preparation_id=key,preparation_variant=case['preparation_variant'],preparation_variant_sha256=variant_signature(case['preparation_variant']))
        if (out.parent/'stop_requested').exists():
            summary.update(skipped_before_physics=True,failure_stage='collection_stopped')
            atomic_json(out/(key+'.json'),summary);atomic_pickle(out/(key+'_episodes.pkl'),[])
            return summary
    with (out/(key+'.log')).open('w') as log,(out/(key+'_trajectory.jsonl')).open('w') as trace:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            predictor=predictor_for_variant(_GRASP,case['preparation_variant']) if diverse else _GRASP
            native=PickNavReachEnv(**case['env_kwargs'])
            summary['initialization_physics_ticks']=native.settle_steps
            if native.scene_manifest()['maze_hash']!=case['maze_hash']:raise ValueError('Frozen TRAIN map mismatch')
            def observe(current,observation,info):
                nonlocal physical_calls
                physical_calls+=1
                prefix='preparation_' if phase=='preparation' else ''
                summary[prefix+'source_action_calls']+=1;summary[prefix+'physics_ticks']+=native.substeps
                if native.step_count%5==0 or (phase=='transport' and not info['bilateral_contact']):
                    ee,quat=current.controller.planner.ee_pose()
                    trace.write(json.dumps({'physical_action_ledger':physical_calls,'native_action':native.step_count,
                        'phase':phase,'attempt_index':active_attempt,'stage':current.display_stage,
                        'qpos':observation['qpos'].tolist(),'qvel':observation['qvel'].tolist(),
                        'ee_world':ee.tolist(),'ee_quaternion':quat.tolist(),
                        'object_world':observation['object_pos'].tolist(),'object_quaternion':observation['object_xyzw'].tolist(),
                        'bilateral_contact':info['bilateral_contact'],'drop_events':info['drop_events'],
                        'normal_force':info.get('normal_force')},default=plain)+'\n')
            system=LearnedCarryPreparation(native,predictor,on_step=observe)
            preparation=system.prepare();summary['preparation_result']=preparation
            summary['prepared_handover']=preparation['prepared_handover']
            summary['proposal_diagnostics']=system.controller.stage_results.get('grasp_generation')
            if summary['prepared_handover']:
                summary['handover_observation']=system.handover_observation
                for attempt in summary['planned_attempts']:
                    active_attempt=attempt['attempt_index'];phase='transport';record=dict(attempt,success=False)
                    summary['attempts'].append(record)
                    if active_attempt:
                        before_restore=(native.step_count,native.physics_steps)
                        native.restore_state(system.handover_state,reset_counters=False)
                        system.controller.planner.set_payload()
                        record['snapshot_restore']={'physical_source':'saved original-start learned handover',
                            'restored_native_action':native.step_count,'restored_physics_ticks':native.physics_steps,
                            'previous_native_action':before_restore[0],'previous_native_physics_ticks':before_restore[1],
                            'new_physics_ticks':0,'preserves_grasp_history':True,'global_physical_action_ledger':physical_calls}
                    before=(native.step_count,native.physics_steps)
                    costs={'source_action_calls':0,'physics_ticks':0}
                    try:
                        carry=GeneralCarryEnv.bind(native,system.controller,horizon=config['horizon'],delta_scale=.075)
                        observation=carry._observation()
                        if any(not np.array_equal(observation[k],system.handover_observation[k]) for k in observation):
                            raise ValueError('Declared handover snapshot did not reproduce its measured observation')
                        record['initial_observation_digest']=__import__('hashlib').sha256(observation['observation'].tobytes()).hexdigest()
                        system._begin('teacher_carry','teacher')
                        with system._observe_native_steps():
                            rows,success,info,stopped=collect_episode(carry,observation,
                                np.random.default_rng(attempt['rng_seed']),attempt['noise_std'],costs,
                                out.parent/'stop_requested')
                        record.update(success=success,transitions=len(rows),failure_reason=info.get('failure_reason'),
                            goal_distance=info.get('goal_distance'),stopped=stopped,
                            final_native_metrics=native.evaluate())
                        if diverse:
                            attach_preparation_metadata(rows,case)
                        if rows:
                            if not (rows[-1]['terminated'] or rows[-1]['truncated']):
                                raise AssertionError('Carry trajectory is not closed')
                            episode={'case_id':case['case_id'],'layout_id':case['layout_id'],'maze_hash':case['maze_hash'],
                                'object_idx':case['object_idx'],'point_seed':case['env_kwargs']['seed'],
                                'full_route':True,'original_exit_goal':True,'collection_mode':'learned-handover',
                                'preparation_protocol':config['protocol'],'grasp_sha256':config['grasp_sha256'],
                                'grasp_proposal_mode':'clearance-v2','grasp_lift_m':.24,
                                'attempt_index':active_attempt,'state_origin':attempt['state_origin'],
                                'corrective_noise_std':attempt['noise_std'],'success':success,'transitions':rows}
                            if diverse:
                                episode.update(preparation_id=key,preparation_variant=case['preparation_variant'],
                                    preparation_variant_sha256=variant_signature(case['preparation_variant']),sampling_protocol=SAMPLING_PROTOCOL)
                            episodes.append(episode)  # Retain raw evidence if schema validation fails.
                            if diverse and validate_recorded_episode(episode,case)!=costs:
                                raise ValueError('Recorded transition costs differ from transport counters')
                    except Exception as exc:
                        record.update(error=f'{type(exc).__name__}: {exc}',success=False)
                        if diverse:summary['fatal_collection_error']=record['error']
                        if costs['source_action_calls']:
                            # Preserve known physics even if a schema check rejects recorded rows.
                            recorded=sum(r['info'].get('action_calls_delta',0) for e in episodes for r in e['transitions']) if diverse else 0
                            record['unrecorded_transport_calls']=max(0,costs['source_action_calls']-recorded)
                            if record['unrecorded_transport_calls']:summary['cost_accounting_complete']=False
                    finally:
                        record.update(source_action_calls=native.step_count-before[0],physics_ticks=native.physics_steps-before[1])
                        if record['source_action_calls']!=costs['source_action_calls']:
                            record['counter_disagreement']=costs;summary['cost_accounting_complete']=False
                        system._active_stage_start=None
                        summary['transitions']=sum(len(e['transitions']) for e in episodes)
                        atomic_pickle(out/(key+'_episodes.pkl'),episodes)
                        atomic_json(out/(key+'_partial.json'),json.loads(json.dumps(summary,default=plain)))
            else:
                summary['failure_stage']=preparation.get('failed_stage');summary['error']=preparation.get('error')
        except Exception as exc:
            summary.update(error=f'{type(exc).__name__}: {exc}',failure_stage=system.stage if system else 'environment_setup')
            if diverse:summary['fatal_collection_error']=summary['error']
        finally:
            if diverse and predictor is not None:
                try:summary['sampling_report']=predictor.sampling_report()
                except Exception as exc:summary['fatal_collection_error']=f'{type(exc).__name__}: {exc}'
            if native is not None:
                summary['scene_manifest']=native.scene_manifest()
                if system is not None:
                    summary['source_calls_by_controller']=system.calls_by_source
                    expected=summary['source_action_calls']+summary['preparation_source_action_calls']
                    if physical_calls!=expected or sum(system.calls_by_source.values())!=expected:
                        summary.update(cost_accounting_complete=False,ledger_error='Physical ledger/controller count disagreement')
                native.close()
            else:summary.update(cost_accounting_complete=False,cost_accounting_note='Partial constructor cost unknown')
            summary['all_source_action_calls']=summary['source_action_calls']+summary['preparation_source_action_calls']
            summary['all_physics_ticks']=summary['physics_ticks']+summary['preparation_physics_ticks']+summary['initialization_physics_ticks']
            if diverse and not summary['cost_accounting_complete']:
                summary['fatal_collection_error']=summary.get('fatal_collection_error','Physical accounting incomplete')
            if diverse and summary.get('fatal_collection_error'):(out.parent/'stop_requested').touch()
            summary['wall_seconds']=time.monotonic()-started
            summary['transitions']=sum(len(e['transitions']) for e in episodes)
            atomic_pickle(out/(key+'_episodes.pkl'),episodes)
            summary=json.loads(json.dumps(summary,default=plain,allow_nan=False))
            atomic_json(out/(key+'.json'),summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp',type=Path,required=True)
    parser.add_argument('--bank',type=Path,default=ROOT/'configs/phase2_scene_bank.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--collection-seed',type=int)
    parser.add_argument('--preparation-variants',type=int,choices=(1,3),default=1)
    parser.add_argument('--layout-limit',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('Use1–8 CPU workers')
    if args.output.exists() and any(args.output.iterdir()):parser.error('Use a new directory; preserve all attempts')
    if args.layout_limit<1:parser.error('Layout limit must be positive')
    diverse=args.preparation_variants==3
    if args.collection_seed is None:args.collection_seed=DEFAULT_COLLECTION_SEED if diverse else 8401
    bank=load_scene_bank(args.bank);cases=select_cases(bank,'train',OBJECTS,layout_limit=args.layout_limit)
    if len(cases)!=args.layout_limit*len(OBJECTS):parser.error('Bank lacks requested TRAIN cases')
    schedule=preparation_schedule(cases,args.preparation_variants,args.collection_seed)
    out=args.output/'artifacts';out.mkdir(parents=True)
    grasp=freeze_checkpoints({'grasp':args.grasp},out/'checkpoints')['grasp']
    candidate=json.loads((args.grasp.parent/'manifest.json').read_text())
    if candidate.get('grasp_proposal_mode')!='clearance-v2' or not any(r.get('sha256')==grasp['sha256'] for r in candidate['checkpoints'].values()):
        raise ValueError('Frozen grasp must match a declared clearance-v2 candidate')
    atomic_json(out/'grasp_candidate_manifest.json',candidate)
    hashes=freeze_sources(out/'source')
    for source in [Path(__file__).resolve(),ROOT/'scripts/phase2_carry_demos.py']:
        dest=out/'source'/'scripts'/source.name;shutil.copy2(source,dest);hashes['scripts/'+source.name]=sha256_file(source)
    atomic_json(out/'source/manifest.json',hashes)
    config={'protocol':DIVERSE_PROTOCOL if diverse else PROTOCOL,'run_id':args.output.name,'partition':'train','bank_sha256':bank['manifest_sha256'],
        'cases':cases,'preparation_schedule':schedule,'object_indices':OBJECTS,'planned_cases':len(cases),
        'planned_preparations':len(schedule),'planned_transport_opportunities':len(schedule)*(1 if diverse else 2),
        'preparation_variants':args.preparation_variants,'sampling_protocol':SAMPLING_PROTOCOL if diverse else None,
        'variant_seed_mapping':'collection_seed + 3*canonical_case_index + variant_index' if diverse else None,
        'selection':f'first{args.layout_limit} frozen TRAIN layouts cross all5 objects; no physical outcome filtering',
        'grasp':grasp,'grasp_sha256':grasp['sha256'],'grasp_proposal_mode':'clearance-v2','grasp_lift_m':.24,
        'planned_speed_scale':1.,'pick_hold_calls':50,'carry_goal_hold_decisions':5,'horizon':900,'gamma':.995,'shaping_weight':2.,'delta_scale':.075,'repeat':5,
        'attempts_per_retained_handover':1 if diverse else 2,'noise_stds':[0.] if diverse else [0.,.12],'collection_seed':args.collection_seed,
        'snapshot_restoration':'none; every preparation starts in an independent original world' if diverse else 'second TRAIN episode only, exact physically prepared learned handover, pickup history preserved',
        'grasp_at_original_start':True,'teacher_in_grasp':False,'teacher_in_carry':True,
        'full_task_evaluation':False,'retain_all_failed_preparations':True,'source_hashes':hashes,'workers':args.workers}
    config['plan_sha256']=plan_digest(config);atomic_json(out/'collection_plan.json',config)
    logger=RunLogger(args.output.name,root=args.output.parent,kind='demonstration',name='Five-object carry from learned grasps',
        summary=f'{len(schedule)} fixed TRAIN preparations; '+('native SAC variants and clean teacher routes.' if diverse else 'clean and noisy teacher routes.')+' All failures and costs retained.')
    logger.artifact('artifacts/collection_plan.json','Frozen TRAIN case and preparation protocol')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker,initargs=(grasp,)) as pool:
        futures={pool.submit(worker,(case,i,str(out),config)):case for i,case in enumerate(schedule)}
        for future in as_completed(futures):
            case=futures[future]
            try:result=future.result()
            except Exception as exc:
                result={'case_id':case['case_id'],'case':case,'prepared_handover':False,'attempts':[],
                    'error':repr(exc),'cost_accounting_complete':False,'failure_stage':'worker_exception',
                    'cost_accounting_note':'Audit any existing partial artifacts before claiming complete costs'}
                if diverse:
                    result.update(preparation_id=case['preparation_id'],preparation_variant=case['preparation_variant'],fatal_collection_error=repr(exc))
                    (out.parent/'stop_requested').touch()
                atomic_json(out/(preparation_key(case)+'_worker_exception.json'),result)
            results.append(result);summary=case_summary(results,schedule);atomic_json(out/'summary.json',summary)
            logger.metric(transitions=summary['transitions'],scope='training_data',completed_cases=summary['completed_cases'],
                completed_preparations=len(results),planned_preparations=len(schedule),planned_cases=len(cases),prepared_handovers=summary['prepared_handovers'],
                successful_transports=summary['successful_transports'],physics_ticks=summary['all_physics_ticks'])
    if diverse and (summary['fatal_collection_errors'] or not summary['cost_accounting_complete'] or (args.output/'stop_requested').exists()):
        logger.artifact('artifacts/summary.json','Incomplete collection; all physical evidence retained')
        logger.finish(status='failed',summary='Schema/sampling/cost error stopped collection; no merged dataset published.')
        raise RuntimeError('TRAIN collection integrity failure; inspect persisted case evidence')
    episodes=[]
    for case in schedule:
        path=out/(preparation_key(case)+'_episodes.pkl')
        if path.exists():episodes.extend(pickle.load(path.open('rb')))
    if diverse:
        try:
            schedule_by_id={preparation_key(case):case for case in schedule}
            checked={'source_action_calls':0,'physics_ticks':0}
            for episode in episodes:
                costs=validate_recorded_episode(episode,schedule_by_id[episode['preparation_id']])
                for key in checked:checked[key]+=costs[key]
            if any(checked[key]!=summary[key] for key in checked):raise ValueError('Dataset transition totals differ from native physical ledger')
            if len(episodes)!=summary['attempted_transports']:raise ValueError('Missing recorded transport episode')
        except Exception as exc:
            atomic_json(out/'dataset_validation_error.json',{'error':repr(exc),'dataset_published':False})
            logger.finish(status='failed',summary='Dataset validation failed; all case artifacts retained, no merged dataset published.')
            raise
    metadata={k:summary[k] for k in COST_FIELDS}
    metadata.update(protocol=config['protocol'],partition='train',bank_sha256=bank['manifest_sha256'],
        object_indices=OBJECTS,gamma=.995,shaping_weight=2.,delta_scale=.075,repeat=5,
        source_runs=[args.output.name],episodes=len(episodes),transitions=sum(len(e['transitions']) for e in episodes),
        configuration=config,summaries=results,preparation_in_replay=False,
        setup_failures_preserved=True,teacher_action_for_bc_only=True,actual_action_preserved=True,
        teacher_at_evaluation=False,full_task_evaluation=False,original_gripper=True,
        cost_accounting_complete=summary['cost_accounting_complete'])
    atomic_pickle(out/'demonstrations.pkl',{'schema_version':1,'metadata':metadata,'episodes':episodes})
    logger.artifact('artifacts/demonstrations.pkl','Actual multi-object carry transitions and failed attempts')
    logger.artifact('artifacts/summary.json','Every preparation and route attempt with all physics costs')
    logger.finish(summary=f"{summary['prepared_handovers']}/{len(schedule)} retained handovers;{summary['successful_transports']}/{summary['attempted_transports']} teacher routes completed. No learned-navigation result.")
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
