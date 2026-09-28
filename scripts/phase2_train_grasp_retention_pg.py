#!/usr/bin/env python3
"""Fixed fresh-batch latent-score pilot. Freeze/review before any physical launch."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer
from pnr.demonstrations import seed_replay
from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.general_manipulation import GeneralGraspEnv,GeneralGraspFeatures,GraspSchema,MultiObjectGrasp
from pnr.grasp_score_function import (RecordedGaussianPredictor,actor_digest,tree_digest,
    prepare_score_batch,guarded_score_update,protected_state)
from pnr.learned_carry import LearnedCarryPreparation
from pnr.policies import load_policy_checkpoint
from pnr.regularized_sac import RegularizedSAC
from pnr.sac_initialization import critic_only_warmup
from pnr.system import StageFailure
from pnr.telemetry import RunLogger,atomic_json
from scripts import phase2_grasp_qualified_demos as recorded
from scripts.phase2_carry_demos import collect_episode
from scripts.phase2_learned_carry_demos import atomic_pickle
from scripts.phase2_train_grasp import dataset_rows

PROPOSAL_SHA='59e67e27996ada5c95e35a7a29c76d3cc823ccf96e5963a4c8b5d02de5cce7bf'
RUN_IDS={'control':'phase2_grasp_retention_pg_control01','retention':'phase2_grasp_retention_pg_positive01'}
COMPARISON_ID='phase2_grasp_retention_pg_compare01'
_MODEL=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def plain(value):return recorded.serial(value)


class DeadlineReached(InterruptedError):pass


def check_deadline(deadline,stop_path):
    if time.time()>=deadline or Path(stop_path).exists():raise DeadlineReached('Prospective wall cap/stop; no replacement')


class AppendRows:
    def __init__(self,path):self.path=Path(path);self.count=0
    def __call__(self,rows):
        if len(rows)<self.count:raise ValueError('Append-only measured record shrank')
        with self.path.open('ab') as handle:
            for row in rows[self.count:]:pickle.dump(deepcopy(row),handle,protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        self.count=len(rows)


class NativeTupleRecorder:
    """Persist actual carry steps, including partial-action accounting on abort."""
    def __init__(self,env,path):self.env=env;self.rows=[];self.current=None;self.path=Path(path);self.append=AppendRows(path.with_suffix('.stream.pkl'))
    def __getattr__(self,name):return getattr(self.env,name)
    def _observation(self):
        result=self.env._observation();self.current=deepcopy(result);return result
    def step(self,action):
        before=(self.native.step_count,self.native.physics_steps)
        try:
            result=self.env.step(action)
        except BaseException as error:
            atomic_pickle(self.path.with_suffix('.partial.pkl'),{'observation':self.current,'action':np.asarray(action).copy(),
                'before':before,'after':(self.native.step_count,self.native.physics_steps),'error':repr(error),
                'actual_native_observation':self.native.observe(),'not_a_complete_transition':True})
            raise
        obs,reward,terminated,truncated,info=result
        self.rows.append({'observation':deepcopy(self.current),'action':np.asarray(action).copy(),'next_observation':deepcopy(obs),
            'reward':float(reward),'terminated':bool(terminated),'truncated':bool(truncated),'info':deepcopy(info),
            'native_call_before':before[0],'native_call_after':self.native.step_count,
            'physics_tick_before':before[1],'physics_tick_after':self.native.physics_steps})
        self.append(self.rows);self.current=deepcopy(obs)
        return result


class GraspRecorder(recorded.RecordingGrasp):
    def __init__(self,env,path):
        self.partial_path=Path(path).with_suffix('.partial.pkl');self.integrity_error=None
        super().__init__(env,AppendRows(Path(path).with_suffix('.stream.pkl')))
    def _observation(self):
        try:return super()._observation()
        except BaseException as error:self.integrity_error=repr(error);raise
    def seal(self,reason):
        boundary=super().seal(reason)
        boundary['note']='Original-goal retention label is separate; native tuples are neither rewritten nor imported into HER.'
        return boundary
    def step(self,action):
        before=(self.wrapped.native.step_count,self.wrapped.native.physics_steps)
        try:return super().step(action)
        except BaseException as error:
            self.integrity_error=repr(error)
            atomic_pickle(self.partial_path,{'observation':deepcopy(self.current),'action':np.asarray(action).copy(),
                'before':before,'after':(self.wrapped.native.step_count,self.wrapped.native.physics_steps),
                'error':repr(error),'actual_native_observation':self.wrapped.native.observe(),'not_a_complete_transition':True})
            raise


def initialize_suffix(checkpoint):
    global _MODEL
    torch.set_num_threads(1)
    if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Frozen suffix actor ZIP changed')
    _MODEL=load_policy_checkpoint(checkpoint['path'],device='cpu')
    if actor_digest(_MODEL.actor)!=checkpoint['actor_sha256']:raise ValueError('Frozen actor state differs')


def suffix_worker(task):
    case,slot,out,settings,deadline,expected_actor=task
    out=Path(out);key=slot['episode_id'];stop_path=out.parent/'STOP'
    native=system=grasp=carry=predictor=None;observed_calls=0;initialized=False;started=time.time()
    r={'case':case,'case_id':case['case_id'],'slot':slot,'replicate_index':slot.get('replicate_index'),
       'actor_sha256':expected_actor,'outcome_known':False,'Y':None,'prepared_handover':False,
       'route_completed':False,'final_hold_retained':False,'bc_eligible':False,'cost_accounting_complete':True,
       'source_action_calls':0,'physics_ticks':0,'initialization_physics_ticks':0,'total_physics_ticks_including_initialization':0,
       'original_start':True,'retry':False,'restores_between_grasp_and_suffix':0}
    with (out/(key+'.log')).open('x') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            check_deadline(deadline,stop_path)
            if actor_digest(_MODEL.actor)!=expected_actor:raise ValueError('Worker actor changed before fresh episode')
            if slot.get('deterministic',False):predictor=_MODEL
            else:predictor=RecordedGaussianPredictor(_MODEL,slot['torch_seed'],expected_actor,
                                    persist=AppendRows(out/(key+'_samples.stream.pkl')))
            # Keep a reference even if native construction fails; unknown settling
            # work is reported as incomplete accounting rather than invented.
            native=PickNavReachEnv.__new__(PickNavReachEnv)
            PickNavReachEnv.__init__(native,**case['env_kwargs']);initialized=True
            if native.scene_manifest()['maze_hash']!=case['maze_hash']:raise ValueError('Frozen TRAIN scene differs')
            def observe(current,observation,info):
                nonlocal observed_calls
                observed_calls+=1
                check_deadline(deadline,stop_path) # after the actual call was counted
            system=LearnedCarryPreparation(native,predictor,on_step=observe)
            def bind(n,controller,horizon):
                nonlocal grasp
                grasp=GraspRecorder(GeneralGraspEnv.bind(n,controller,horizon=horizon,goal_lift_m=settings['grasp_lift_m']),out/(key+'_grasp.pkl'))
                return grasp
            system.skill_factories['grasp']=bind
            prep=system.prepare();r['preparation_result']=prep;r['prepared_handover']=bool(prep['prepared_handover'])
            if getattr(predictor,'integrity_error',None) or getattr(grasp,'integrity_error',None):
                raise ValueError('Sampling/recording integrity error swallowed by physical preparation')
            if grasp:r['grasp_boundary']=grasp.seal('caller_handover_prefix_end' if prep.get('stages',{}).get('grasp',{}).get('success') else 'caller_stopped_prefix')
            if r['prepared_handover']:
                env=GeneralCarryEnv.bind(native,system.controller,horizon=settings['carry_horizon'],delta_scale=settings['delta_scale'])
                carry=NativeTupleRecorder(env,out/(key+'_transport.pkl'));costs={'source_action_calls':0,'physics_ticks':0}
                with system._observe_native_steps():
                    start=system._begin('teacher_carry','teacher')
                    _,success,info,stopped=collect_episode(carry,carry._observation(),np.random.default_rng(slot['torch_seed']),0.,costs,stop_path)
                    r['route_completed']=bool(success and info.get('is_success') and info.get('bilateral_contact') and not native.drop_events)
                    system._finish('teacher_carry',start,r['route_completed'],decisions=len(carry.rows),failure_reason=info.get('failure_reason'))
                    r['transport_costs']=costs
                    if stopped:raise DeadlineReached('Interrupted teacher suffix')
                    if r['route_completed']:
                        system._held_pause('transport_final_hold',settings['final_hold_calls']);r['final_hold_retained']=True
            else:r.update(failed_stage=prep.get('failed_stage'),error=prep.get('error'))
            r['outcome_known']=True
        except StageFailure as error:
            r.update(outcome_known=True,error=repr(error),failed_stage=system.stage if system else 'setup')
        except BaseException as error:
            r.update(outcome_known=False,error=repr(error),fatal_error=repr(error),failed_stage=system.stage if system else 'setup')
            stop_path.touch()
        finally:
            rows=grasp.rows if grasp is not None else []
            samples=predictor.samples if isinstance(predictor,RecordedGaussianPredictor) else []
            if grasp:
                if grasp.integrity_error:r.update(outcome_known=False,fatal_error=grasp.integrity_error);stop_path.touch()
                r['grasp_boundary']=grasp.seal('caller_stopped_prefix')
                try:r['grasp_costs']=recorded.validate_prefix(rows)
                except Exception as error:r.update(outcome_known=False,fatal_error=repr(error));stop_path.touch()
            if isinstance(predictor,RecordedGaussianPredictor):
                try:r['sampling_report']=predictor.sampling_report()
                except Exception as error:r.update(outcome_known=False,fatal_error=repr(error));stop_path.touch()
                if len(samples)!=len(rows) and r['outcome_known']:
                    r.update(outcome_known=False,fatal_error='Partial/unmatched policy action');stop_path.touch()
            if initialized:
                r.update(source_action_calls=native.step_count,physics_ticks=native.physics_steps,
                    initialization_physics_ticks=native.settle_steps,total_physics_ticks_including_initialization=native.physics_steps+native.settle_steps,
                    actual_suffix_end_observation=native.observe(),actual_suffix_end_metrics=native.evaluate())
                r['cost_accounting_complete']=bool(system and observed_calls==native.step_count
                    and sum(system.calls_by_source.values())==native.step_count and native.physics_steps==5*native.step_count)
            elif native is not None:
                r.update(cost_accounting_complete=False,known_source_call_lower_bound=getattr(native,'step_count',0),
                         known_physics_tick_lower_bound=getattr(native,'physics_steps',0),initialization_cost_unknown=True)
            if system:r.update(stages=system.stage_results,source_calls_by_controller=system.calls_by_source,
                wall_contact_calls=system.wall_contact_calls,robot_table_contact_calls=system.robot_table_contact_calls)
            if not r['cost_accounting_complete']:r.update(outcome_known=False,fatal_error='Incomplete physical accounting');stop_path.touch()
            r['bc_eligible']=recorded.qualification(r['prepared_handover'],r['route_completed'],r['final_hold_retained'],
                wall_contacts=r.get('wall_contact_calls',0),table_contacts=r.get('robot_table_contact_calls',0),fatal_error=not r['outcome_known'])
            r['Y']=int(r['bc_eligible']) if r['outcome_known'] else None
            r['wall_seconds']=time.time()-started
            if native is not None:
                try:native.close()
                except Exception as error:r['close_error']=repr(error)
            atomic_pickle(out/(key+'_grasp.pkl'),rows);atomic_pickle(out/(key+'_samples.pkl'),samples)
            atomic_pickle(out/(key+'_transport.pkl'),carry.rows if carry else [])
            r['grasp_sha256']=sha(out/(key+'_grasp.pkl'));r['samples_sha256']=sha(out/(key+'_samples.pkl'))
            r['transport_sha256']=sha(out/(key+'_transport.pkl'))
            episode={'case_id':case['case_id'],'replicate_index':slot.get('replicate_index'),'actor_sha256':expected_actor,
                     'outcome_known':r['outcome_known'],'Y':r['Y'],'samples':samples,'transitions':rows,'summary':plain(r)}
            atomic_pickle(out/(key+'_episode.pkl'),episode);atomic_json(out/(key+'.json'),plain(r))
    return plain(r)


def collect_batch(model,directory,cases,slots,settings,deadline,workers,logger=None):
    directory=Path(directory)
    if directory.exists():raise ValueError('No repeated/replaced physical batch')
    directory.mkdir(parents=True);out=directory/'artifacts';out.mkdir()
    checkpoint=directory/'generating_actor.zip';model.save(checkpoint)
    record={'path':str(checkpoint.resolve()),'sha256':sha(checkpoint),'actor_sha256':actor_digest(model.actor)}
    before=actor_digest(model.actor)
    frozen={'checkpoint':record,'cases':cases,'schedule':slots,'settings':settings,'deadline':deadline,'workers':workers,
            'existing_handover_save_allowed':True,'restores_between_grasp_and_suffix':0}
    atomic_json(out/'batch_plan.json',frozen)
    byid={c['case_id']:c for c in cases};results=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_suffix,initargs=(record,)) as pool:
        futures={pool.submit(suffix_worker,(byid[s['case_id']],s,str(out),settings,deadline,before)):s for s in slots}
        for future in as_completed(futures):
            try:results.append(future.result())
            except BaseException as error:
                (directory/'STOP').touch()
                errors.append({'slot':futures[future],'error':repr(error),'outcome_known':False,
                               'cost_accounting_complete':False,'available_worker_artifacts_retained':True})
            summary={'planned':len(slots),'completed':len(results),'known':sum(x['outcome_known'] for x in results),
                'successes':sum(x['Y']==1 for x in results),'cost_accounting_complete':not errors and all(x['cost_accounting_complete'] for x in results),
                'total_physics_ticks_including_initialization':sum(x['total_physics_ticks_including_initialization'] for x in results),
                'recorded_cost_is_lower_bound':bool(errors),'worker_errors':errors,'results':results}
            atomic_json(out/'summary.json',summary)
            if logger:logger.metric(scope='fresh_retention_collection',batch_completed=len(results),batch_planned=len(slots),
                auxiliary_batch_physics_ticks=summary['total_physics_ticks_including_initialization'])
    if actor_digest(model.actor)!=before:raise ValueError('Learner actor changed during the frozen physical batch')
    if errors or len(results)!=len(slots) or not all(x['outcome_known'] and x['cost_accounting_complete'] for x in results):
        raise RuntimeError('Incomplete/unknown batch: no auxiliary update or replacement')
    episodes=[pickle.loads((out/(s['episode_id']+'_episode.pkl')).read_bytes()) for s in slots]
    return episodes,summary,record


class CountedPickup(gym.Wrapper):
    """Unchanged pickup dynamics with independent lifetime and partial-call ledgers."""
    def __init__(self,cases,index,seed,directory,deadline):
        super().__init__(MultiObjectGrasp(cases,seed=seed,horizon=100,cache_size=4,episodes_per_case=4,
                                         lift_range=(.20,.26),grasp_proposal_mode='clearance-v2'))
        self.path=Path(directory)/f'pickup_worker_{index:02d}.json';self.deadline=deadline
        self.stop_path=Path(directory).parent/'STOP';self.calls=0;self.decisions=0;self.partial=[];self.error=None
        self.in_reset=False
        self.original_step=PickNavReachEnv.step
        def timed_step(native,action):
            check_deadline(self.deadline,self.stop_path)
            before=native.step_count
            try:return self.original_step(native,action)
            finally:
                if not self.in_reset:self.calls+=native.step_count-before
                check_deadline(self.deadline,self.stop_path)
        self.timed_step=timed_step;PickNavReachEnv.step=timed_step
        self._write()
    def _write(self):
        setup=int(self.env.setup_action_calls);settling=int(self.env.initialization_physics_ticks)
        atomic_json(self.path,{'returned_worker_decisions':self.decisions,'native_pickup_calls':self.calls,
            'native_pickup_physics_ticks':self.calls*5,'setup_source_action_calls':setup,
            'setup_physics_ticks':int(self.env.setup_physics_ticks),'initialization_physics_ticks':settling,
            'total_physics_ticks_including_initialization':self.calls*5+int(self.env.setup_physics_ticks)+settling,
            'failed_setup_attempts':plain(self.env.failed_setup_attempts),'partial_actions':self.partial,
            'error':self.error,'setup_deadline_checks':'worker-local native step hook before/after each native call; settling construction checked before/after',
            'cost_accounting_complete':all(bool(x.get('costs')) for x in self.env.failed_setup_attempts)})
    def reset(self,**kwargs):
        try:
            check_deadline(self.deadline,self.stop_path)
            self.in_reset=True
            result=self.env.reset(**kwargs)
            check_deadline(self.deadline,self.stop_path)
            return result
        except BaseException as error:self.error=repr(error);raise
        finally:self.in_reset=False;self._write()
    def step(self,action):
        before=self.calls
        try:
            check_deadline(self.deadline,self.stop_path)
            result=self.env.step(action);self.decisions+=1;return result
        except BaseException as error:
            self.error=repr(error)
            native=self.env.active.native if self.env.active else None
            self.partial.append({'action':np.asarray(action).tolist(),'native_calls_completed':self.calls-before,
                'native_physics_ticks_completed':5*(self.calls-before),'error':repr(error),
                'actual_native_observation':plain(native.observe()) if native else None,'not_a_complete_transition':True})
            raise
        finally:self._write()
    def compute_reward(self,*args,**kwargs):return self.env.compute_reward(*args,**kwargs)
    def close(self):
        self._write();self.env.close()
        if PickNavReachEnv.step is self.timed_step:PickNavReachEnv.step=self.original_step


def pickup_totals(artifacts):
    rows=[json.loads(p.read_text()) for p in sorted(Path(artifacts).glob('pickup_worker_*.json'))]
    fields=('returned_worker_decisions','native_pickup_calls','native_pickup_physics_ticks','setup_source_action_calls',
            'setup_physics_ticks','initialization_physics_ticks','total_physics_ticks_including_initialization')
    return {**{k:sum(r[k] for r in rows) for k in fields},'workers_reporting':len(rows),'cost_accounting_complete':len(rows)==6 and all(r['cost_accounting_complete'] for r in rows),'worker_ledgers':rows}


class PickupProgress(BaseCallback):
    def __init__(self,logger,deadline,stop):super().__init__();self.run=logger;self.deadline=deadline;self.stop=stop
    def _on_step(self):
        if self.num_timesteps%600==0:
            self.run.metric(self.num_timesteps,scope='skill',metric_source='training',policy_decisions=self.num_timesteps,
                            regularization=getattr(self.model,'last_regularization_metrics',{}))
        return time.time()<self.deadline and not self.stop.exists()


def verify_execution(path):
    plan=json.loads(Path(path).read_text());actual=digest({k:v for k,v in plan.items() if k!='execution_sha256'})
    if plan['execution_sha256']!=actual:raise ValueError('Execution plan digest differs')
    for name,h in plan['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen implementation/source changed: '+name)
    for record in (plan['proposal']['source_grasp'],plan['proposal']['demonstrations']):
        if sha(record['path'])!=record['sha256']:raise ValueError('Frozen actor/demo changed')
    if sha(plan['proposal']['scene_bank']['path'])!=plan['proposal']['scene_bank']['sha256']:raise ValueError('TRAIN bank changed')
    return plan


def verify_authorization(plan,path):
    auth=json.loads(Path(path).read_text())
    if not (auth.get('authorized') is True and auth.get('authorized_by')=='/root'
            and auth.get('scope')=='launch_full_retention_pg_pilot'
            and auth.get('execution_sha256')==plan['execution_sha256']):
        raise ValueError('Exact root physical-launch authorization is required')
    return auth


def native_model(proposal,artifacts,logger):
    cfg=proposal['shared_native_sac'];seed=proposal['study_seed']
    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    dataset=pickle.loads(Path(proposal['demonstrations']['path']).read_bytes())
    rows,groups,report=dataset_rows(dataset)
    demo_obs={k:np.stack([r['observation'][k] for r in rows]) for k in ('observation','achieved_goal','desired_goal')}
    demo_actions=np.stack([r['action'] for r in rows])
    schema=DummyVecEnv([lambda:GraspSchema() for _ in range(6)])
    model=RegularizedSAC('MultiInputPolicy',schema,bc_weight=200.,demo_batch_size=400,gradient_diagnostics_interval=100,
        learning_rate=3e-5,buffer_size=150000,learning_starts=0,batch_size=400,tau=.01,gamma=.99,
        train_freq=1,gradient_steps=1,ent_coef='auto_0.005',target_entropy=-8.,
        replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs={'n_sampled_goal':1,'goal_selection_strategy':'future','copy_info_dict':True},
        policy_kwargs={'net_arch':[128,128],'features_extractor_class':GeneralGraspFeatures},device='cpu',seed=seed)
    caller_rng=(torch.get_rng_state().clone(),deepcopy(np.random.get_state()),random.getstate())
    try:source=load_policy_checkpoint(proposal['source_grasp']['path'],device='cpu')
    finally:
        torch.set_rng_state(caller_rng[0]);np.random.set_state(caller_rng[1]);random.setstate(caller_rng[2])
    source_actor=actor_digest(source.actor);model.actor.load_state_dict(source.actor.state_dict())
    if actor_digest(model.actor)!=source_actor:raise AssertionError('Initial actor/std transfer changed tensors')
    model._pnr_grasp_protocol_signature=deepcopy(source._pnr_grasp_protocol_signature)
    model._pnr_retention_pg_ancestry={'source_sha256':proposal['source_grasp']['sha256'],
        'historical_online_decisions':12000,'historical_sac_updates':2000,'new_auxiliary_attempted':0,'new_auxiliary_accepted':0}
    del source
    model.set_demonstrations(demo_obs,demo_actions,groups,seed=seed)
    model.set_logger(configure(str(Path(artifacts).parent/'sb3'),['csv']))
    seeding=seed_replay(model,proposal['demonstrations']['path'])
    warmup=critic_only_warmup(model,2000,400,learning_rate=3e-4,warmup_tau=.01,log_every=500,
        on_log=lambda record:logger.metric(0,scope='critic_initialization',**record))
    if actor_digest(model.actor)!=source_actor or model.num_timesteps!=0 or model._n_updates!=0:
        raise AssertionError('Native critic warmup changed actor or online counters')
    init={'actor_sha256':source_actor,'protected_state':protected_state(model),'critic_warmup':warmup,
          'replay':{k:v for k,v in seeding.items() if k!='dataset_metadata'},'bc_data':report,
          'numpy_rng':tree_digest(np.random.get_state()),'torch_rng':tree_digest(torch.get_rng_state()),
          'python_rng':tree_digest(random.getstate()),'demo_rng':tree_digest(model._demo_rng.bit_generator.state)}
    atomic_json(Path(artifacts)/'initializer.json',plain(init))
    return model,schema,rows,groups


def fixed_bc_batch(rows,groups,round_index):
    rng=np.random.default_rng(119001+round_index);strata=list(groups.values())
    which=rng.integers(0,len(strata),size=400)
    indices=np.array([rng.choice(strata[group]) for group in which])
    return ({k:np.stack([rows[i]['observation'][k] for i in indices]) for k in ('observation','achieved_goal','desired_goal')},
            np.stack([rows[i]['action'] for i in indices]),indices)


def run_arm(execution_path,authorization,branch,start_time):
    execution=verify_execution(execution_path);verify_authorization(execution,authorization);proposal=execution['proposal']
    deadline=start_time+proposal['time_budget']['training_collection_dispatch_cap_seconds']
    run=ROOT/'runs'/RUN_IDS[branch]
    if run.exists():raise ValueError('No repeated arm or endpoint replacement')
    logger=RunLogger(run.name,kind='train',skill='grasp',budget=12000,
        algorithm='SAC+HER+BC plus fresh original-goal retention score' if branch=='retention' else 'SAC+HER+BC with matched extra BC-only steps',
        name='Retention score · '+branch,summary='Four fixed rounds; no intermediate physical checkpoint selection.')
    artifacts=run/'artifacts';artifacts.mkdir();atomic_json(artifacts/'execution_plan.json',execution)
    physical=schema=model=None;rounds=[];weight=1. if branch=='retention' else 0.
    try:
        model,schema,rows,groups=native_model(proposal,artifacts,logger)
        cases=proposal['shared_native_sac']['training_cases'];seed=proposal['study_seed']
        physical=SubprocVecEnv([partial(CountedPickup,cases[i::6],i,seed+i,str(artifacts),deadline) for i in range(6)],start_method='spawn')
        model.set_env(physical);model.replay_buffer.env=None;model.replay_buffer.set_env(physical)
        callback=PickupProgress(logger,deadline,run/'STOP')
        for r in range(4):
            check_deadline(deadline,run/'STOP');before_steps,before_updates=model.num_timesteps,model._n_updates
            model.learn(total_timesteps=3000,reset_num_timesteps=False,callback=callback)
            if (model.num_timesteps-before_steps,model._n_updates-before_updates)!=(3000,500):
                raise RuntimeError('Incomplete exact3k/500 round; no shortened endpoint')
            expected=(3000*(r+1),500*(r+1))
            if (model.num_timesteps,model._n_updates)!=expected:raise AssertionError('Native cumulative counters drifted')
            rng_before={'torch':torch.get_rng_state().clone(),'numpy':deepcopy(np.random.get_state()),'python':random.getstate(),
                        'demo':deepcopy(model._demo_rng.bit_generator.state)}
            batchpath=run/f'round_{r+1:02d}'
            slots=[s for s in proposal['retention_collection']['schedule'] if s['branch']==branch and s['round_index']==r]
            if r==0:
                atomic_json(artifacts/'pre_first_auxiliary.json',{'actor_sha256':actor_digest(model.actor),'protected_state':protected_state(model),
                    'rng':{k:tree_digest(v) for k,v in rng_before.items()}})
            episodes,summary,frozen=collect_batch(model,batchpath,proposal['retention_collection']['cases'],slots,
                proposal['retention_collection']['source_settings'],deadline,12,logger)
            rng_after={'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),'demo':model._demo_rng.bit_generator.state}
            if any(tree_digest(rng_before[k])!=tree_digest(rng_after[k]) for k in rng_before):
                raise AssertionError('Fresh suffix collection changed native learner/demo RNG')
            check_deadline(deadline,run/'STOP')
            demo,actions,indices=fixed_bc_batch(rows,groups,r)
            update=guarded_score_update(model,episodes,demo,actions,weight,expected_case_ids=[c['case_id'] for c in proposal['retention_collection']['cases']])
            update.update(round_index=r,bc_indices_sha256=tree_digest(indices),batch_actor=frozen,
                          batch_physics_ticks=summary['total_physics_ticks_including_initialization'])
            model._pnr_retention_pg_ancestry['new_auxiliary_attempted']+=1
            model._pnr_retention_pg_ancestry['new_auxiliary_accepted']+=int(update['accepted'])
            atomic_json(batchpath/'artifacts/update.json',plain(update));rounds.append(update)
            logger.metric(model.num_timesteps,scope='original_goal_retention_training',native_sac_updates=model._n_updates,
                auxiliary_updates_attempted=len(rounds),auxiliary_updates_accepted=sum(x['accepted'] for x in rounds),
                latest_auxiliary=plain(update),accounting={'pickup':pickup_totals(artifacts),
                    'auxiliary_physics_ticks':sum(x['batch_physics_ticks'] for x in rounds)})
            model.save(run/f'round_{r+1:02d}/post_update.zip') # provenance only; never selected/evaluated
        endpoint=run/'checkpoints/final.zip';endpoint.parent.mkdir();model.save(endpoint)
        loaded=load_policy_checkpoint(endpoint,device='cpu')
        if actor_digest(loaded.actor)!=actor_digest(model.actor) or protected_state(loaded)!=protected_state(model) or loaded._pnr_retention_pg_ancestry!=model._pnr_retention_pg_ancestry:
            raise AssertionError('Endpoint reload changed actor, protected optimizer/temperature state, or ancestry')
        if (loaded.num_timesteps,loaded._n_updates)!=(12000,2000):raise AssertionError('Endpoint counters changed')
        result={'status':'complete','branch':branch,'checkpoint':str(endpoint.resolve()),'checkpoint_sha256':sha(endpoint),
            'actor_sha256':actor_digest(model.actor),'new_native_pickup_decisions':12000,'new_native_sac_updates':2000,
            'source_ancestry':model._pnr_retention_pg_ancestry,'rounds':rounds,'pickup_accounting':pickup_totals(artifacts),
            'auxiliary_physics_ticks':sum(x['batch_physics_ticks'] for x in rounds),'execution_sha256':execution['execution_sha256'],
            'full60_launched':False}
        atomic_json(artifacts/'result.json',plain(result));logger.finish('complete','Fixed round4 endpoint saved; deterministic TRAIN comparison pending.')
        return result
    except BaseException as error:
        (run/'STOP').touch();atomic_json(artifacts/'failure.json',{'error':repr(error),'rounds_completed':len(rounds),
            'pickup_accounting':pickup_totals(artifacts),'native_decisions':getattr(model,'num_timesteps',0),
            'native_sac_updates':getattr(model,'_n_updates',0),'no_earlier_endpoint_selected':True,'retry':False})
        logger.finish('failed','Incomplete fixed pilot; all available physical evidence retained, no replacement or endpoint selection.')
        raise
    finally:
        if physical is not None:
            try:physical.close()
            except (EOFError,BrokenPipeError,ConnectionError):
                for process in physical.processes:
                    if process.is_alive():process.terminate()
                    process.join(timeout=5)
        if schema is not None:schema.close()
        if (artifacts/'failure.json').exists():
            failure=json.loads((artifacts/'failure.json').read_text())
            failure['pickup_accounting_after_worker_shutdown']=pickup_totals(artifacts)
            failure['external_interruption_may_leave_lower_bounds']=True
            atomic_json(artifacts/'failure.json',failure)


def endpoint_gate(results,case_ids):
    """Fixed deterministic TRAIN gate; missing/unknown cases never become losses."""
    by={};duplicates=[]
    for r in results:
        key=(r['branch'],r['case_id'])
        if key in by:duplicates.append(key)
        by[key]=r
    expected={(branch,c) for branch in RUN_IDS for c in case_ids}
    complete=set(by)==expected and not duplicates and all(r['outcome_known'] and r['cost_accounting_complete'] for r in results)
    if not complete:return {'passed':False,'complete':False,'reason':'Incomplete/unknown/duplicated fixed comparison','duplicates':duplicates}
    control=sum(by['control',c]['Y'] for c in case_ids);candidate=sum(by['retention',c]['Y'] for c in case_ids)
    wins=[c for c in case_ids if by['retention',c]['Y'] and not by['control',c]['Y']]
    losses=[c for c in case_ids if by['control',c]['Y'] and not by['retention',c]['Y']]
    prep_losses=[c for c in case_ids if by['control',c]['prepared_handover'] and not by['retention',c]['prepared_handover']]
    margin=(candidate-control>=3) if control<=37 else (candidate==40 and candidate>control)
    return {'passed':candidate>=36 and margin and not prep_losses,'complete':True,'control_successes':control,
        'retention_successes':candidate,'net_gain':candidate-control,'candidate_only_successes':wins,
        'control_only_successes':losses,'new_preparation_failures':prep_losses,
        'absolute_quality_pass':candidate>=36,'positive_margin_pass':margin,'full_task_evaluation':False,
        'full60_automatic_launch':False}


def matched_initial_states(control_dir,retention_dir):
    """No outcome selection: enforce the declared pre-treatment match."""
    records={}
    for name in ('initializer','pre_first_auxiliary'):
        a=json.loads((Path(control_dir)/'artifacts'/f'{name}.json').read_text())
        b=json.loads((Path(retention_dir)/'artifacts'/f'{name}.json').read_text())
        keys=('actor_sha256','protected_state','numpy_rng','torch_rng','python_rng','demo_rng') if name=='initializer' else ('actor_sha256','protected_state','rng')
        comparison={k:a[k]==b[k] for k in keys}
        if not all(comparison.values()):raise ValueError('Unmatched pre-treatment learner/RNG state: '+name+' '+str(comparison))
        records[name]=comparison
    return records


def compare(execution_path,authorization,start_time):
    execution=verify_execution(execution_path);verify_authorization(execution,authorization);proposal=execution['proposal']
    deadline=start_time+proposal['time_budget']['whole_physical_pilot_wall_cap_seconds']
    runs={branch:ROOT/'runs'/name for branch,name in RUN_IDS.items()}
    matching=matched_initial_states(runs['control'],runs['retention'])
    endpoints={}
    for branch,run in runs.items():
        r=json.loads((run/'artifacts/result.json').read_text())
        if r['status']!='complete' or r['execution_sha256']!=execution['execution_sha256'] or (r['new_native_pickup_decisions'],r['new_native_sac_updates'])!=(12000,2000):
            raise ValueError('Only the two complete declared fixed endpoints may be compared')
        if sha(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Fixed endpoint ZIP changed')
        endpoints[branch]=r
    directory=ROOT/'runs'/COMPARISON_ID
    if directory.exists():raise ValueError('No repeated/replaced endpoint comparison')
    check_deadline(deadline,directory/'STOP')
    logger=RunLogger(COMPARISON_ID,kind='evaluation',skill='grasp',planned_cases=80,
        name='Retention score · 40 deterministic TRAIN pairs',scope='TRAIN_teacher_transport_diagnostic',
        summary='Fixed endpoints; actual teacher transport plus hold, not full-task or heldout.')
    out=directory/'artifacts';out.mkdir();results=[];summaries={}
    atomic_json(out/'comparison_plan.json',{'execution_sha256':execution['execution_sha256'],
        'source_hashes':execution['source_hashes'],'endpoints':endpoints,'matched_initial_states':matching,
        'cases':proposal['endpoint_gate']['cases'],'settings':proposal['retention_collection']['source_settings'],
        'workers':12,'deadline':deadline,'sequential_branch_order':['control','retention'],'retries':0})
    try:
        for branch in RUN_IDS:
            check_deadline(deadline,directory/'STOP')
            model=load_policy_checkpoint(endpoints[branch]['checkpoint'],device='cpu')
            if actor_digest(model.actor)!=endpoints[branch]['actor_sha256']:raise ValueError('Fixed endpoint actor changed')
            slots=[{'case_id':c['case_id'],'episode_id':branch+'_'+c['case_id'],'deterministic':True,
                    'torch_seed':c['env_kwargs']['seed'],'branch':branch} for c in proposal['endpoint_gate']['cases']]
            _,summary,_=collect_batch(model,directory/branch,proposal['endpoint_gate']['cases'],slots,
                proposal['retention_collection']['source_settings'],deadline,12,logger)
            for r in summary['results']:r['branch']=branch
            results.extend(summary['results']);summaries[branch]=summary
            total_calls=sum(r['source_action_calls'] for r in results)
            logger.metric(total_calls,scope='TRAIN_teacher_transport_diagnostic',eval_episodes=len(results),planned_cases=80,
                planned_pairs=40,accounting={'total_physics_ticks_including_initialization':sum(r['total_physics_ticks_including_initialization'] for r in results)})
            atomic_json(out/'partial_results.json',{'results':results,'summaries':summaries})
        gate=endpoint_gate(results,[c['case_id'] for c in proposal['endpoint_gate']['cases']])
        report={'execution_sha256':execution['execution_sha256'],'gate':gate,'results':results,
            'total_physics_ticks_including_initialization':sum(r['total_physics_ticks_including_initialization'] for r in results),
            'all_attempts_retained':True,'retries':0,'new_heldout_rollouts':0,'full60_launched':False}
        atomic_json(out/'summary.json',report)
        logger.finish('complete','Fixed TRAIN gate '+('passed; root review required before any full60.' if gate['passed'] else 'failed; pilot closed without full60.'))
        return report
    except BaseException as error:
        (directory/'STOP').touch()
        atomic_json(out/'failure.json',{'error':repr(error),'completed_branch_results':results,
            'batch_artifacts_retained':True,'gate':{'passed':False,'complete':False},'retry':False,'full60_launched':False})
        logger.finish('failed','Incomplete comparison; no replacement or full60.')
        raise


def freeze_execution(proposal_path,output):
    proposal=json.loads(Path(proposal_path).read_text())
    actual=digest({k:v for k,v in proposal.items() if k!='proposal_sha256'})
    if actual!=PROPOSAL_SHA or proposal['proposal_sha256']!=PROPOSAL_SHA:raise ValueError('Approved proposal changed')
    prior=json.loads((ROOT/'runs/phase2_grasp_qualified_plan03/artifacts/plan.json').read_text())
    names=set(prior['source_hashes'])|{'pnr/grasp_score_function.py','scripts/phase2_train_grasp_retention_pg.py',
        'scripts/phase2_train_grasp.py'}
    sources={name:sha(ROOT/name) for name in sorted(names)}
    for name,h in prior['source_hashes'].items():
        if sources[name]!=h:raise ValueError('Previously frozen physical/dependency source changed: '+name)
    execution={'protocol':'retention-score-execution-v1','proposal':proposal,'proposal_path':str(Path(proposal_path).resolve()),
        'source_hashes':sources,'run_ids':RUN_IDS,'comparison_run_id':COMPARISON_ID,
        'clarifications':{'native_learning_starts':0,'native_train_freq':1,'native_gradient_steps':1,
            'save_unused_shared_handover_snapshot':True,'restore_between_grasp_and_suffix':False,
            'deadline_hooks':'before/after every native step including setup; native initialization settling counted intact',
            'read_only_source_load_restores_torch_numpy_python_rng':True,
            'comparison_sequential_branch_order':['control','retention'],'comparison_workers':12,
            'score_update_interpretation':'whole clipped/BC/curvature-scaled/KL-guarded heuristic, not guaranteed improvement'},
        'commands':{'arm':'.venv/bin/python scripts/phase2_train_grasp_retention_pg.py arm --execution PLAN --authorization AUTH --start-time EPOCH --branch control|retention',
            'compare':'.venv/bin/python scripts/phase2_train_grasp_retention_pg.py compare --execution PLAN --authorization AUTH --start-time EPOCH'},
        'outer_supervision':'root-owned exact command record; common start, stop peers on failure, graceful STOP before separately accounted external kill',
        'validation':{'tests_passed':66,'command':'.venv/bin/python -m pytest -q tests/test_grasp_retention_pg_runner.py tests/test_grasp_score_function.py tests/test_grasp_qualified_demos.py tests/test_regularized_sac.py',
            'physical_interactions':0,'test_source_hashes':{name:sha(ROOT/name) for name in
                ('tests/test_grasp_retention_pg_runner.py','tests/test_grasp_score_function.py','tests/test_grasp_qualified_demos.py','tests/test_regularized_sac.py')}},
        'physical_interactions_at_freeze':0,'optimizer_updates_at_freeze':0,'automatic_full60':False,
        'software':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__}}
    execution['execution_sha256']=digest(execution)
    output=Path(output)
    if output.exists():raise ValueError('Execution manifest is immutable; use a new prelaunch version')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,execution)
    verify_execution(output)
    return execution


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('freeze-execution');freeze.add_argument('--proposal',type=Path,default=ROOT/'runs/phase2_grasp_retention_pg_plan01/artifacts/proposal.json')
    freeze.add_argument('--output',type=Path,required=True)
    for command in ('arm','compare'):
        p=sub.add_parser(command);p.add_argument('--execution',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
        p.add_argument('--start-time',type=float,required=True)
        if command=='arm':p.add_argument('--branch',choices=list(RUN_IDS),required=True)
    args=parser.parse_args()
    if args.command=='freeze-execution':result=freeze_execution(args.proposal,args.output);print(result['execution_sha256'])
    elif args.command=='arm':run_arm(args.execution,args.authorization,args.branch,args.start_time)
    else:compare(args.execution,args.authorization,args.start_time)


if __name__=='__main__':main()
