#!/usr/bin/env python3
"""Spatial BC initialization, calibrated exploration, then ordinary SAC+HER.

Development runs use only declared train/validation layouts. The milestone test
bank is inaccessible here. Physical and supervised costs are logged separately.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pnr.general_carry import GeneralCarryEnv
from pnr.multi_carry import CarrySchema, MultiSceneCarry
from pnr.scenes import load_scene_bank, scene_records, case_specs
from pnr.spatial_features import CarrySpatialExtractor
from pnr.navigation_features import NavigationSpatialExtractor
from pnr.sac_initialization import calibrate_actor_std,critic_only_warmup
from pnr.regularized_sac import RegularizedSAC
from pnr.demonstrations import seed_replay, carry_supervision
from pnr.telemetry import RunLogger,atomic_json
from pnr.train import GracefulStop,ignore_worker_stop_signals
from pnr.phase2_checkpoints import (signature,load_checkpoint,publish_checkpoint,
                                   restore_rng,validate_loaded_model)


def encoder_policy_kwargs(encoder='cnn32', navigation_only=False):
    """Two fixed learned encoders; no input, action, or physical changes."""
    if encoder == 'cnn32':
        extractor = NavigationSpatialExtractor if navigation_only else CarrySpatialExtractor
        settings = {'features_dim': 128, 'channels': 32}
    elif encoder == 'gated48-k20-f5':
        from pnr.gated_spatial_features import GatedCarrySpatialExtractor
        extractor = GatedCarrySpatialExtractor
        settings = {'features_dim': 128, 'channels': 48, 'iterations': 20, 'kernel_size': 5}
    else:
        raise ValueError('Unknown carry encoder: ' + str(encoder))
    return {'net_arch': [128, 128], 'features_extractor_class': extractor,
            'features_extractor_kwargs': settings}


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def archive_sources(directory):
    out=directory/'sources';out.mkdir();manifest={}
    for path in list((ROOT/'pnr').glob('*.py'))+[Path(__file__).resolve()]:
        relative=path.relative_to(ROOT);target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);manifest[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(directory/'source_manifest.json',manifest)


class Progress(BaseCallback):
    def __init__(self,logger,layouts,args,stop):
        super().__init__();self.run_logger=logger;self.layouts=layouts;self.args=args;self.stop=stop
        self.evaluation={};self.last_eval=self.last_log=self.last_save=0;self.started=time.monotonic()
        self.train_calls=self.train_ticks=self.eval_calls=self.eval_ticks=0
        self.eval_setup_calls=self.eval_setup_ticks=self.eval_settling_ticks=0
        self.episodes=0;self.recent=deque(maxlen=100);self.best=(-1.,-float('inf'));self.initialization={}
        self._checkpoint_due=False;self._pending_replay_position=None
        self.resume_signature=None;self.ancestry=[];self.best_checkpoint=None
        self.inherited_accounting={};self.setup_base=self.setup_ticks_base=self.settling_base=0
    def restore(self,checkpoint):
        state=checkpoint['state'];account=state['accounting'];history=state.get('callback_state',{})
        self.inherited_accounting=dict(account)
        self.train_calls=account['training_source_action_calls'];self.train_ticks=account['training_physics_ticks']
        self.eval_calls=account['evaluation_source_action_calls'];self.eval_ticks=account['evaluation_physics_ticks']
        self.eval_setup_calls=account['evaluation_setup_source_action_calls'];self.eval_setup_ticks=account['evaluation_setup_physics_ticks']
        self.eval_settling_ticks=account['evaluation_initialization_physics_ticks']
        self.setup_base=account['training_setup_source_action_calls'];self.setup_ticks_base=account['training_setup_physics_ticks']
        self.settling_base=account['training_initialization_physics_ticks']
        self.initialization=state['initialization'];self.episodes=int(history.get('episodes',self.model._episode_num))
        self.recent=deque(history.get('recent',[]),maxlen=100)
        best=history.get('best');self.best=tuple(best) if best is not None else (-1.,-float('inf'))
        self.best_checkpoint=history.get('best_checkpoint')
        self.last_eval=int(history.get('last_eval',self.model.num_timesteps))
        self.last_log=self.last_save=int(self.model.num_timesteps)
        self.ancestry=list(state.get('ancestry',[]))+[{'run':checkpoint['run_directory'].name,
            'generation':checkpoint['generation'],'manifest_sha256':checkpoint.get('manifest_sha256'),
            'legacy_file_hashes':checkpoint.get('legacy_file_hashes'),
            'online_decisions':self.model.num_timesteps,'online_updates':self.model._n_updates,
            'accounting':account,'initialization_cost_inherited':True,'legacy_missing':state.get('legacy_missing',[])}]
    def accounting(self):
        physical=self.model.env
        # Schema-only env has no setup; no physical cost is silently fabricated.
        try:
            setup=sum(physical.get_attr('setup_action_calls'));ticks=sum(physical.get_attr('setup_physics_ticks'))
            settling=sum(physical.get_attr('initialization_physics_ticks'))
        except AttributeError:setup=ticks=settling=0
        result={'online_decisions':int(self.model.num_timesteps),'training_source_action_calls':self.train_calls,
            'training_physics_ticks':self.train_ticks,'training_setup_source_action_calls':self.setup_base+setup,
            'training_setup_physics_ticks':self.setup_ticks_base+ticks,'training_initialization_physics_ticks':self.settling_base+settling,
            'cost_scope':'cumulative_lineage','inherited_online_decisions':int(self.inherited_accounting.get('online_decisions',0)),
            'incremental_online_decisions':int(self.model.num_timesteps-self.inherited_accounting.get('online_decisions',0)),
            'evaluation_source_action_calls':self.eval_calls,'evaluation_physics_ticks':self.eval_ticks,
            'evaluation_setup_source_action_calls':self.eval_setup_calls,'evaluation_setup_physics_ticks':self.eval_setup_ticks,
            'evaluation_initialization_physics_ticks':self.eval_settling_ticks}
        result['incremental_accounting']={key:value-int(self.inherited_accounting.get(key,0))
            for key,value in result.items() if key=='online_decisions' or key.endswith(('_calls','_ticks'))}
        return result
    def evaluate(self,label):
        records=[]
        for layout in self.layouts:
            if self.stop.requested:break
            key=layout['layout_id']
            identity={'layout_id':key,'partition':key.split('_')[0],'maze_seed':layout['maze_seed'],
                      'maze_hash':layout['maze_hash']}
            if key not in self.evaluation:
                try:
                    env=GeneralCarryEnv(maze_seed=layout['maze_seed'],scene_seed=42,object_idx=14,
                        evaluation=True,build_bank=False,horizon=self.args.horizon,delta_scale=self.args.delta_scale,
                        gamma=getattr(self.args,'gamma',.995))
                except Exception as exc:
                    if not hasattr(exc,'setup_accounting'):raise
                    costs=exc.setup_accounting
                    self.eval_setup_calls+=costs.get('source_action_calls',0)
                    self.eval_setup_ticks+=costs.get('physics_ticks',0)
                    self.eval_settling_ticks+=costs.get('initialization_physics_ticks',0)
                    records.append(dict(identity,success=False,evaluation_complete=True,decisions=0,
                        goal_distance=None,selection_distance_penalty=30.,failure_reason='physical_setup_failed',
                        error=repr(exc),setup_accounting=costs,source_action_calls=0,physics_ticks=0,trajectory=[]))
                    self.run_logger.event(f'Validation setup failed for {key}: {exc}','error')
                    continue
                self.evaluation[key]=env
                self.eval_setup_calls+=env.setup_action_calls;self.eval_setup_ticks+=env.setup_physics_ticks
                self.eval_settling_ticks+=env.initialization_physics_ticks
            env=self.evaluation[key];obs,_=env.reset(options={'full':True});positions=[];success=False;complete=False
            info={};decisions=0
            for decision in range(env.horizon):
                if self.stop.requested:break
                action,_=self.model.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action);decisions=decision+1
                self.eval_calls+=info['action_calls_delta'];self.eval_ticks+=info['physics_ticks_delta']
                positions.append(env.controller.base_position().tolist())
                if info['is_success']:success=True;complete=True;break
                if terminated or truncated:complete=True;break
            if decisions==env.horizon:complete=True
            distance=info.get('goal_distance')
            if distance is not None and not np.isfinite(distance):distance=None
            records.append(dict(identity,success=success,evaluation_complete=complete,decisions=decisions,
                goal_distance=distance,selection_distance_penalty=30. if distance is None else float(distance),
                failure_reason=info.get('failure_reason') if complete else 'evaluation_interrupted',
                source_action_calls=env.native.step_count,physics_ticks=env.native.physics_steps,trajectory=positions))
        validation=[r for r in records if r['partition']=='validation' and r['evaluation_complete']]
        training=[r for r in records if r['partition']=='train' and r['evaluation_complete']]
        expected={layout['layout_id'] for layout in self.layouts if layout['layout_id'].startswith('validation_')}
        complete=bool(expected) and {r['layout_id'] for r in validation}==expected
        rate=float(np.mean([r['success'] for r in validation])) if validation else None
        actual=[r['goal_distance'] for r in validation if r['goal_distance'] is not None]
        distance=float(np.mean(actual)) if actual else None
        selection=float(np.mean([r['selection_distance_penalty'] for r in validation])) if validation else None
        out=self.run_logger.directory/'artifacts'/f'evaluation_{label}.json'
        atomic_json(out,{'scope':'full_carry_validation','online_decisions':int(self.model.num_timesteps),
            'records':records,'success_rate':rate,'validation_complete':complete,'expected_validation_cases':len(expected),
            'mean_measured_goal_distance':distance,'selection_distance_penalty':selection,
            'undefined_distance_selection_penalty':30.,'evaluation_is_full_task':False,'teacher_at_inference':False})
        self.run_logger.artifact('artifacts/'+out.name,'Navigation validation '+label)
        self.run_logger.metric(self.model.num_timesteps,scope='skill',evaluation_scope='full_carry_validation',
            metric_source='evaluation',success_rate=rate,eval_episodes=len(validation),mean_goal_distance=distance,
            validation_complete=complete,selection_distance_penalty=selection,
            training_layout_success_rate=float(np.mean([r['success'] for r in training])) if training else None,
            accounting=self.accounting())
        if complete and self.model.num_timesteps>0 and (rate,-selection)>self.best:
            self.best=(rate,-selection)
            checkpoint=self.run_logger.directory/'checkpoints/best_online.zip'
            temporary=checkpoint.with_name('best_online.pending.zip');self.model.save(temporary);os.replace(temporary,checkpoint)
            self.best_checkpoint=str(checkpoint)
            atomic_json(checkpoint.with_name('best_online_evaluation.json'),{'success_rate':rate,
                'goal_distance':distance,'selection_distance_penalty':selection,'online_decisions':int(self.model.num_timesteps),
                'validation_complete':True,'evaluation':str(out)})
        return records
    def checkpoint(self,label):
        if getattr(self.model.env,'waiting',False):raise RuntimeError('Checkpoint while waiting for worker replies')
        if self._pending_replay_position is not None:
            replay=self.model.replay_buffer
            if replay.pos==self._pending_replay_position:
                fields=self.locals
                self.model._store_transition(replay,fields['buffer_actions'],fields['new_obs'],fields['rewards'],fields['dones'],fields['infos'])
            elif replay.pos!=(self._pending_replay_position+1)%replay.buffer_size:
                raise RuntimeError('Unexpected replay position while flushing interrupted step')
            self._pending_replay_position=None
        try:states=self.model.env.env_method('get_training_state')
        except AttributeError:states=[]
        state=plain({'accounting':self.accounting(),'training_states':states,'initialization':self.initialization,
            'ancestry':self.ancestry,'inherited_accounting':self.inherited_accounting,
            'callback_state':{'episodes':self.episodes,'recent':list(self.recent),
                'best':self.best if np.isfinite(self.best[1]) else None,'best_checkpoint':self.best_checkpoint,
                'last_eval':self.last_eval,'last_log':self.last_log,'last_save':self.num_timesteps}})
        if getattr(self.args,'min_demo_fraction',None) is not None:
            state['replay_sampling']=self.model.replay_buffer.sampling_report()
        return publish_checkpoint(self.model,self.run_logger.directory,state,self.resume_signature,label)
    def _on_step(self):
        self._pending_replay_position=self.model.replay_buffer.pos
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            self.train_calls+=info['action_calls_delta'];self.train_ticks+=info['physics_ticks_delta']
            if done:self.episodes+=1;self.recent.append(float(info.get('episode_success',info['is_success'])))
        if self.stop.requested:return False
        if self.num_timesteps-self.last_log>=1000:
            self.run_logger.metric(self.num_timesteps,scope='skill',metric_source='training',
                train_success_rate=float(np.mean(self.recent)) if self.recent else 0.,episodes=self.episodes,
                source_action_calls=self.train_calls,physics_ticks=self.train_ticks,updates=self.model._n_updates,
                wall_seconds=time.monotonic()-self.started,accounting=self.accounting(),
                regularization=getattr(self.model,'last_regularization_metrics',{}))
            self.last_log=self.num_timesteps
        # Fixed curriculum schedule; validation outcomes do not modify training cases.
        level=min(5,2+self.num_timesteps//10000)
        if self.num_timesteps%1000<self.model.n_envs:self.model.env.env_method('set_curriculum_level',level)
        if self.num_timesteps-self.last_eval>=self.args.eval_every:
            self.evaluate(f'online_{self.num_timesteps:07d}');self.last_eval=self.num_timesteps
        if self.num_timesteps-self.last_save>=self.args.checkpoint_every:self._checkpoint_due=True
        return not self.stop.requested
    def _on_rollout_end(self):
        self._pending_replay_position=None
        if self._checkpoint_due:
            self.checkpoint(f'online_{self.num_timesteps:07d}');self.last_save=self.num_timesteps;self._checkpoint_due=False
    def close(self):
        for env in self.evaluation.values():env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--demonstrations');p.add_argument('--resume-run')
    p.add_argument('--bank',default='configs/phase2_scene_bank.json');p.add_argument('--layouts',type=int,default=16)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=7401)
    p.add_argument('--initial-checkpoint');p.add_argument('--bc-updates',type=int,default=12000);p.add_argument('--warmup-updates',type=int,default=2000)
    p.add_argument('--critic-warmup-tau',type=float,default=.01,help='Initialization target-update rate only; online SAC tau remains .01')
    p.add_argument('--steps',type=int,default=40000,help='Additional online decisions; zero runs fresh initialization and validation only');p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--gamma',type=float,choices=(.995,.999),default=.995,help='Controlled carry discount ablation; failure cost remains 1/(1-gamma)')
    p.add_argument('--relabel-demo-rewards',action='store_true',help='Verify stored source rewards and relabel replay under the target discount')
    p.add_argument('--learning-rate',type=float,default=3e-5);p.add_argument('--delta-scale',type=float,default=.075)
    p.add_argument('--horizon',type=int,default=900);p.add_argument('--eval-every',type=int,default=10000)
    p.add_argument('--checkpoint-every',type=int,default=10000);p.add_argument('--device',default='cuda')
    p.add_argument('--validation-layouts',type=int,default=4)
    p.add_argument('--encoder',choices=('cnn32','gated48-k20-f5'),default='cnn32',
                   help='Opt-in fixed GPPN-inspired map core; default CNN architecture is unchanged')
    p.add_argument('--navigation-only',action='store_true')
    p.add_argument('--balance-objects',action='store_true',help='Uniform TRAIN-object BC sampling before per-object direction strata')
    p.add_argument('--balance-replay-objects',action='store_true',help='Uniform original-object sampling before standard HER relabeling')
    p.add_argument('--balance-replay-sources',action='store_true',help='Uniform fixed preparation families within each replay object')
    p.add_argument('--min-demo-fraction',type=float,choices=(.5,),default=None,
                   help='Opt-in per-object minimum demonstration replay share; preserves larger natural demo fractions')
    p.add_argument('--preparation-grasp',help='Opt-in frozen learned grasp for conditional five-object carry training')
    p.add_argument('--preparation-variants',type=int,choices=(1,3),default=1,help='Three uses independent deterministic and two native-SAC sampled grasps per TRAIN case')
    p.add_argument('--preparation-seed',type=int,default=9401)
    p.add_argument('--training-layouts',type=int,help='Number of first selected TRAIN layouts used for learned handovers; default8')
    p.add_argument('--training-objects',type=int,nargs='+',default=[14,5,2,7,10])
    p.add_argument('--buffer-size',type=int,default=100000)
    p.add_argument('--bc-weight',type=float,default=0.);p.add_argument('--demo-batch-size',type=int,default=256)
    args=p.parse_args()
    if not np.isfinite(args.bc_weight) or args.bc_weight<0 or args.demo_batch_size<1:p.error('Invalid demonstration regularization settings')
    if not np.isfinite(args.critic_warmup_tau) or not 0<args.critic_warmup_tau<=1:p.error('Critic warmup tau must be within (0,1]')
    if args.resume_run and args.initial_checkpoint:p.error('--resume-run and --initial-checkpoint are exclusive')
    if args.steps<0 or min(args.workers,args.horizon,args.eval_every,args.checkpoint_every)<1:p.error('Nonnegative steps and positive workers/horizon/intervals required')
    if args.steps==0 and args.resume_run:p.error('Zero-step initialization requires a fresh run; resume with a positive online budget')
    if args.resume_run:
        if Path(args.resume_run).name!=args.resume_run:p.error('Use a run ID for --resume-run')
        parent_config=json.loads((ROOT/'runs'/args.resume_run/'config.json').read_text())
        explicit={token.split('=')[0] for token in sys.argv[1:] if token.startswith('--')}
        for key in ('demonstrations','bank','layouts','workers','seed','horizon','delta_scale','batch_size','learning_rate','bc_weight','demo_batch_size','validation_layouts','encoder','navigation_only','buffer_size','critic_warmup_tau','balance_objects','preparation_grasp','training_layouts','training_objects','balance_replay_objects','balance_replay_sources','preparation_variants','preparation_seed','gamma','relabel_demo_rewards','min_demo_fraction'):
            if '--'+key.replace('_','-') not in explicit:setattr(args,key,parent_config.get(key,getattr(args,key)))
        if '--preparation-grasp' not in explicit and parent_config.get('frozen_preparation_grasp'):
            args.preparation_grasp=parent_config['frozen_preparation_grasp']
        if args.critic_warmup_tau!=parent_config.get('critic_warmup_tau',.01):
            p.error('Cannot change completed critic-warmup provenance on resume')
    if (args.gamma!=.995)!=bool(args.relabel_demo_rewards):p.error('The gamma .999 ablation requires explicit --relabel-demo-rewards; default gamma does not relabel')
    if not args.demonstrations:p.error('--demonstrations required for a fresh run')
    if args.buffer_size<1:p.error('--buffer-size must be positive')
    if args.balance_replay_sources and not args.balance_replay_objects:p.error('Source balancing requires explicit object-balanced replay')
    if args.min_demo_fraction is not None:
        if not args.balance_replay_objects:p.error('--min-demo-fraction requires object-balanced replay')
        if args.balance_replay_sources:p.error('Demonstration-floor and preparation-family samplers are separate ablations')
        if args.resume_run and parent_config.get('min_demo_fraction') is None:
            p.error('Demonstration-floor replay requires a fresh branch; old replay origins cannot be migrated')
    if args.preparation_variants!=1 and not args.preparation_grasp:p.error('Preparation variants require a frozen learned grasp')
    if args.balance_replay_objects and not args.preparation_grasp:p.error('Object-balanced replay requires declared learned five-object handover training')
    if args.workers>args.layouts:p.error('Each worker needs at least one selected layout')
    torch.set_num_threads(1);np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed)
    bank=load_scene_bank(args.bank);available_layouts=scene_records(bank,'train')
    if not 1<=args.layouts<=len(available_layouts):p.error('Selected layout count must fit the frozen TRAIN partition')
    layouts=available_layouts[:args.layouts]
    preparation_cases=None;preparation_settings=None;preparation_sha=None;cases_sha=None
    if args.preparation_grasp:
        from pnr.learned_carry import PREPARATION_SETTINGS, CACHE_RESTORE_SETTINGS
        args.training_layouts=min(8,args.layouts) if args.training_layouts is None else args.training_layouts
        if not 1<=args.training_layouts<=len(layouts):p.error('Learned preparation layout count must fit selected TRAIN layouts')
        if set(args.training_objects)!={14,5,2,7,10} or len(args.training_objects)!=5:p.error('Learned handovers require all five declared original objects')
        selected_ids={layout['layout_id'] for layout in layouts[:args.training_layouts]}
        preparation_cases=[case for case in case_specs(bank,'train',object_indices=args.training_objects) if case['layout_id'] in selected_ids]
        if len(preparation_cases)<args.workers:p.error('Every worker needs at least one declared learned-handover case')
        preparation_settings=dict(PREPARATION_SETTINGS,cache_restore=dict(CACHE_RESTORE_SETTINGS))
        if args.preparation_variants==3:
            from pnr.diverse_learned_carry import expand_preparation_cases, CACHE_PROTOCOL as VARIANT_CACHE_PROTOCOL
            from pnr.grasp_sampling import SAMPLING_PROTOCOL
            preparation_cases=expand_preparation_cases(preparation_cases,seed=args.preparation_seed)
            preparation_settings.update(variants_per_case=3,preparation_seed=args.preparation_seed,
                sampling_protocol=SAMPLING_PROTOCOL,cache_protocol=VARIANT_CACHE_PROTOCOL)
        if args.delta_scale!=preparation_settings['carry_delta_scale']:p.error('Learned preparation requires its frozen carry action scale')
        preparation_sha=hashlib.sha256(Path(args.preparation_grasp).read_bytes()).hexdigest()
        cases_sha=hashlib.sha256(json.dumps(preparation_cases,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    elif args.training_layouts is not None:p.error('--training-layouts requires --preparation-grasp')
    validation=scene_records(bank,'validation')[:args.validation_layouts]
    if not 1<=args.validation_layouts<=len(scene_records(bank,'validation')):p.error('Invalid validation layout count')
    config=dict(vars(args),preparation_mode=('learned-grasp-variants' if args.preparation_variants==3 else 'learned-grasp') if preparation_cases else 'reference',
        preparation_grasp_sha256=preparation_sha,preparation_settings=preparation_settings,training_cases_sha256=cases_sha,
        carry_training_scope='conditional on physically prepared learned grasp; failures and costs retained' if preparation_cases else 'reference-prepared foam',
        gamma=args.gamma,shaping_weight=2.,reward_relabel_gamma=args.gamma if args.relabel_demo_rewards else None,alpha_initial=.005,target_entropy=-4.,online_tau=.01,
        n_sampled_goal=1,demonstrations_sha256=hashlib.sha256(Path(args.demonstrations).read_bytes()).hexdigest(),
        scene_bank_sha256=bank['manifest_sha256'],supervised_loss_during_online_training=args.bc_weight>0,
        independent_final_study_seed=False,scene_cache_policy='prime declared learned handovers once; counted restore hold before each full-goal episode' if preparation_cases else 'prime all assigned physical scenes once before learning',critic_initialization='fresh with fixed-actor Bellman warmup',
        gpu_numerics='cuDNN benchmark enabled; non-bitwise GPU training; float32 matmul high',
        initial_checkpoint_sha256=None if not args.initial_checkpoint else hashlib.sha256(Path(args.initial_checkpoint).read_bytes()).hexdigest())
    # Legacy resumes retain their original signature vocabulary. New runs record the encoder.
    if args.resume_run and 'encoder' not in parent_config and '--encoder' not in explicit:
        config.pop('encoder')
    if args.min_demo_fraction is None:
        config.pop('min_demo_fraction',None)  # Preserve existing default signature vocabulary.
    else:
        from pnr.demonstration_floor_her import SAMPLING_PROTOCOL as DEMO_FLOOR_PROTOCOL
        config['demo_floor_protocol']=DEMO_FLOOR_PROTOCOL
    preparation_source_mapping=None
    if args.balance_replay_sources:
        from pnr.preparation_metadata import PREPARATION_SOURCES_BY_MODE, PREPARATION_SOURCES_BY_OBJECT
        preparation_source_mapping=dict(PREPARATION_SOURCES_BY_MODE)
        config['preparation_sources_by_mode']=preparation_source_mapping
        config['preparation_sources_by_object']=PREPARATION_SOURCES_BY_OBJECT
    current_signature=signature(config,ROOT)
    resumed=load_checkpoint(ROOT/'runs'/args.resume_run,current_signature) if args.resume_run else None
    inherited=0 if resumed is None else resumed['state']['online_decisions']
    config.update(steps_are_additional_online_decisions=True,inherited_online_decisions=inherited,
                  cumulative_online_budget=inherited+args.steps)
    algorithm='SAC+HER+demonstration regularization' if args.bc_weight>0 else 'BC-initialized SAC+HER'
    learner=RegularizedSAC if args.bc_weight>0 else SAC
    logger=RunLogger(args.run_id,kind='train',skill='carry',algorithm=algorithm,budget=inherited+args.steps,
        transitions=inherited,name=f'Carry · regularized SAC (weight {args.bc_weight:g})' if args.bc_weight>0 else 'Spatial multi-maze SAC + HER',
        summary='Resuming ordinary SAC+HER.' if resumed else 'Initializing map-aware actor from diverse physical demonstrations.')
    (logger.directory/'artifacts').mkdir();(logger.directory/'checkpoints').mkdir();archive_sources(logger.directory)
    frozen_grasp=None
    if preparation_cases is not None:
        frozen_grasp=logger.directory/'artifacts/preparation_grasp.zip'
        shutil.copy2(args.preparation_grasp,frozen_grasp)
        if hashlib.sha256(frozen_grasp.read_bytes()).hexdigest()!=preparation_sha:raise ValueError('Preparation grasp changed while freezing')
        atomic_json(logger.directory/'artifacts/training_cases.json',{'cases':preparation_cases,'sha256':cases_sha,'settings':preparation_settings})
        config['frozen_preparation_grasp']=str(frozen_grasp.resolve())
    atomic_json(logger.directory/'config.json',config)
    dataset=pickle.loads(Path(args.demonstrations).read_bytes())
    allowed={l['maze_hash'] for l in layouts}
    if any(e['maze_hash'] not in allowed for e in dataset['episodes']):raise ValueError('Demonstration layout outside selected training partition')
    if resumed is None or args.bc_weight>0:
        rows,groups,probabilities,supervision=carry_supervision(dataset,balance_objects=args.balance_objects)
        # Preserve the established sampler group name for old regularized resumes.
        groups={('turn' if name=='turn_or_vertical' else name):indices for name,indices in groups.items()}
        strata=list(groups.values())
        atomic_json(logger.directory/'artifacts/supervision.json',supervision)
    if resumed is None:
        obs={key:torch.as_tensor(np.stack([r['observation'][key] for r in rows]),device=args.device) for key in CarrySchema().observation_space.spaces}
        targets=torch.as_tensor(np.stack([r['teacher_action'] for r in rows]),device=args.device)
    schema=DummyVecEnv([lambda:CarrySchema(gamma=args.gamma) for _ in range(args.workers)])
    replay_class=HerReplayBuffer
    replay_kwargs={'n_sampled_goal':1,'goal_selection_strategy':'future','copy_info_dict':True}
    if args.balance_replay_objects:
        from pnr.object_balanced_her import ObjectBalancedHerReplayBuffer
        replay_class=ObjectBalancedHerReplayBuffer
        replay_kwargs['object_indices']=list(args.training_objects)
    if args.balance_replay_sources:
        from pnr.preparation_balanced_her import PreparationBalancedHerReplayBuffer
        replay_class=PreparationBalancedHerReplayBuffer
        replay_kwargs['preparation_sources_by_object']=PREPARATION_SOURCES_BY_OBJECT
    if args.min_demo_fraction is not None:
        from pnr.demonstration_floor_her import DemonstrationFloorHerReplayBuffer
        replay_class=DemonstrationFloorHerReplayBuffer
        replay_kwargs['min_demo_fraction']=args.min_demo_fraction
    if resumed is not None:
        model=learner.load(resumed['model'],env=schema,device=args.device)
        model.load_replay_buffer(resumed['replay'],truncate_last_traj=True)
        validate_loaded_model(model,resumed,args.workers)
    else:
        model=learner('MultiInputPolicy',schema,learning_rate=args.learning_rate,buffer_size=args.buffer_size,
            learning_starts=0,batch_size=args.batch_size,tau=.01,gamma=args.gamma,train_freq=1,gradient_steps=1,
            ent_coef='auto_0.005',target_entropy=-4.,replay_buffer_class=replay_class,replay_buffer_kwargs=replay_kwargs,
            policy_kwargs=encoder_policy_kwargs(args.encoder,args.navigation_only),device=args.device,seed=args.seed)
        if args.initial_checkpoint:
            source=SAC.load(args.initial_checkpoint,env=schema,device=args.device)
            if source.num_timesteps != 0:
                raise ValueError('This initializer accepts only a zero-online BC checkpoint')
            incompatible=model.actor.load_state_dict(source.actor.state_dict(),strict=False)
            allowed={'features_extractor.navigation_grip_mask'} if args.navigation_only else set()
            if set(incompatible.missing_keys)-allowed or incompatible.unexpected_keys:
                raise ValueError('Actor initializer architecture does not match the declared features')
            del source
    # SB3 enables deterministic cuDNN kernels; benchmarked alternatives may be much faster.
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision('high')
    if args.bc_weight>0:
        demo_obs={key:np.stack([r['observation'][key] for r in rows]) for key in CarrySchema().observation_space.spaces}
        demo_actions=np.stack([r['teacher_action'] for r in rows])
        model.bc_weight=args.bc_weight;model.demo_batch_size=args.demo_batch_size
        manifest=model.set_demonstrations(demo_obs,demo_actions,groups,
            group_probabilities=probabilities,seed=args.seed+9000 if resumed is None else None)
        atomic_json(logger.directory/'artifacts/actor_regularization.json',manifest|{'weight':args.bc_weight,
            'demo_batch_size':args.demo_batch_size,'objective':'SAC actor loss + weight * original-goal demonstration action MSE'})
    if resumed is None:
        model.actor.features_extractor.set_carry_pose_mask(True)
        if args.navigation_only:model.actor.features_extractor.set_navigation_only(True)
    model.set_logger(configure(str(logger.directory/'sb3'),['csv']))
    callback=Progress(logger,layouts[:2]+validation,args,None);callback.model=model;callback.resume_signature=current_signature
    if resumed is not None:
        callback.restore(resumed)
        atomic_json(logger.directory/'artifacts/resume_ancestry.json',plain({'ancestry':callback.ancestry,'inherited_accounting':callback.inherited_accounting}))
    physical=None
    with GracefulStop() as stop:
        callback.stop=stop
        try:
            if resumed is None:
                optimizer=torch.optim.Adam(model.actor.parameters(),lr=3e-4)
                rng=np.random.default_rng(args.seed);losses=[];bc_started=time.monotonic()
                update=0
                for update in range(1,args.bc_updates+1):
                    if args.balance_objects:
                        sizes=rng.multinomial(args.batch_size,probabilities)
                    else:
                        sizes=[int(args.batch_size*probability) for probability in probabilities[:-1]]
                        sizes.append(args.batch_size-sum(sizes))
                    indices=np.concatenate([rng.choice(s,size=n,replace=True) for s,n in zip(strata,sizes)])
                    batch={k:v[indices] for k,v in obs.items()}
                    predicted=model.actor(batch,deterministic=True)
                    loss=torch.mean((predicted-targets[indices])**2)
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),10.);optimizer.step()
                    losses.append(float(loss.detach()))
                    if update%500==0 or update==args.bc_updates:
                        logger.metric(0,scope='actor_initialization',metric_source='supervised',bc_updates=update,
                            bc_training_loss=float(np.mean(losses[-500:])),supervised_examples=update*args.batch_size,
                            wall_seconds=time.monotonic()-bc_started)
                    if stop.requested:break
                callback.initialization['bc']={'parent_checkpoint':args.initial_checkpoint,
                    'parent_checkpoint_sha256':config['initial_checkpoint_sha256'],'updates':update,'batch_size':args.batch_size,'examples':update*args.batch_size,
                    'wall_seconds':time.monotonic()-bc_started,'supervision':supervision,
                    'batch_strata_sampler':'multinomial' if args.balance_objects else 'fixed_floor_last_remainder',
                    'encoder':args.encoder,'encoder_settings':encoder_policy_kwargs(args.encoder,args.navigation_only)['features_extractor_kwargs']}
                callback.initialization['std_calibration']=calibrate_actor_std(model,-3.)
                model.save(logger.directory/'checkpoints/bc_initialized')
                atomic_json(logger.directory/'artifacts/initialization.json',plain(callback.initialization))
                if stop.requested:return
                logger.update(summary='Measuring initialized actor on training and separate validation layouts.')
                callback.evaluate('bc_zero_online')
                logger.update(summary='Seeding real HER replay and fitting fresh critics with actor fixed.')
                import_context=(model.replay_buffer.demonstration_import()
                    if args.min_demo_fraction is not None else nullcontext())
                with import_context:
                    callback.initialization['demonstrations']=seed_replay(model,args.demonstrations,object_labels=args.balance_replay_objects,
                        preparation_sources_by_mode=preparation_source_mapping,
                        reward_relabel_gamma=args.gamma if args.relabel_demo_rewards else None)
                if args.balance_replay_objects:
                    callback.initialization['replay_sampling_before_warmup']=model.replay_buffer.sampling_report()
                callback.initialization['critic']=critic_only_warmup(model,args.warmup_updates,args.batch_size,
                    learning_rate=3e-4,warmup_tau=args.critic_warmup_tau,log_every=200,
                    on_log=lambda row:logger.metric(0,scope='actor_initialization',metric_source='critic_initialization',**row))
                if args.balance_replay_objects:
                    callback.initialization['replay_sampling_after_warmup']=model.replay_buffer.sampling_report()
                callback.checkpoint('initialized')
                if stop.requested:return
                if args.steps==0:
                    atomic_json(logger.directory/'artifacts/result.json',plain({
                        'status':'complete','initialization_only':True,'online_decisions':0,'online_updates':0,
                        'initialization':callback.initialization,'accounting':callback.accounting(),
                        'ancestry':callback.ancestry,'best_checkpoint':None,
                        'validation':'artifacts/evaluation_bc_zero_online.json'}))
                    logger.artifact('artifacts/result.json','Initialization and validation accounting')
                    logger.finish('complete','Initialization and validation complete; zero online SAC decisions. Resumable checkpoint saved.')
                    return
            parent_pid=os.getpid()
            def make(index):
                torch.set_num_threads(1)
                ignore_worker_stop_signals(parent_pid)
                if preparation_cases is not None:
                    from pnr.learned_carry import LearnedMultiCaseCarry
                    preparation_class=LearnedMultiCaseCarry
                    if args.preparation_variants==3:
                        from pnr.diverse_learned_carry import DiverseLearnedMultiCaseCarry
                        preparation_class=DiverseLearnedMultiCaseCarry
                    env=preparation_class(preparation_cases[index::args.workers],grasp_checkpoint=str(frozen_grasp.resolve()),
                        grasp_sha256=preparation_sha,seed=args.seed+index,episodes_per_case=1,
                        horizon=args.horizon,delta_scale=args.delta_scale,gamma=args.gamma)
                    if args.balance_replay_sources:
                        from pnr.preparation_metadata import LearnedPreparationMetadata
                        env=LearnedPreparationMetadata(env)
                    return env
                return MultiSceneCarry(layouts[index::args.workers],seed=args.seed+index,cache_size=4,
                    episodes_per_scene=4,horizon=args.horizon,delta_scale=args.delta_scale,gamma=args.gamma)
            physical=SubprocVecEnv([lambda i=i:make(i) for i in range(args.workers)],start_method='spawn')
            model.set_env(physical)
            # The reward implementation/spaces are identical; replace the initialization schema.
            model.replay_buffer.env = None
            model.replay_buffer.set_env(physical)
            if resumed is not None and resumed['state'].get('training_states'):
                for index,state in enumerate(resumed['state']['training_states']):
                    physical.env_method('set_training_state',state,indices=[index])
            else:physical.env_method('set_curriculum_level',2)
            logger.update(summary='Preparing all training scenes in parallel; each physical setup is counted.')
            priming=physical.env_method('prime_cache')
            atomic_json(logger.directory/'artifacts/scene_cache_setup.json',plain(priming))
            if preparation_cases is not None:
                counts={str(obj):sum(int(row['valid_object_counts'].get(str(obj),0)) for row in priming) for obj in args.training_objects}
                atomic_json(logger.directory/'artifacts/preparation_coverage.json',{'declared_preparation_slots':len(preparation_cases),
                    'declared_cases':len({case['case_id'] for case in preparation_cases}),
                    'variants_per_case':args.preparation_variants,'valid_object_counts':counts,'all_declared_failures_retained':True,'full_task_success':False})
                if any(not row.get('ready_for_training',False) for row in priming):raise RuntimeError('A learned-handover worker has no valid prepared case; see persisted priming reports')
                if any(count==0 for count in counts.values()):raise RuntimeError('No valid learned handover for a declared object: '+str(counts))
            if resumed is not None:
                restore_rng(resumed['rng'])
                callback.checkpoint('resumed_before_additional_online')
            logger.update(summary=('Online SAC + HER with continuing demonstration action regularization; no teacher at inference.'
                if args.bc_weight>0 else 'Ordinary online SAC + HER; no imitation loss or teacher action during training.'))
            model.learn(total_timesteps=args.steps,callback=callback,reset_num_timesteps=resumed is None)
            callback.checkpoint('final')
            if not stop.requested:
                callback.evaluate(f'final_{model.num_timesteps:07d}')
                callback.checkpoint('final')
            atomic_json(logger.directory/'artifacts/result.json',plain({'status':'stopped' if stop.requested else 'complete',
                'online_decisions':model.num_timesteps,'online_updates':model._n_updates,'initialization':callback.initialization,
                'accounting':callback.accounting(),'ancestry':callback.ancestry,'inherited_accounting':callback.inherited_accounting,
                'best_checkpoint':callback.best_checkpoint,'best_validation':callback.best if np.isfinite(callback.best[1]) else None,
                'replay_sampling':model.replay_buffer.sampling_report() if args.balance_replay_objects else None}))
            logger.artifact('artifacts/result.json','Training and accounting result')
            logger.finish('stopped' if stop.requested else 'complete','Bounded multi-maze SAC+HER experiment finished; see validation results.')
        except BaseException as exc:
            logger.event(repr(exc),'error');logger.finish('failed',str(exc));raise
        finally:
            callback.close()
            if physical is not None:physical.close()
            schema.close()
            if stop.requested and logger.state['status']=='running':logger.finish('stopped','Stopped after current safe stage.')

if __name__=='__main__':main()
