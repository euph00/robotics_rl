#!/usr/bin/env python3
"""Explicit long-route development transfer from the measured short-carry pilot.

Copies actor/critic parameters and actual replay; resets optimizers, changes the
replay-safe goal feature scale, and recomputes ALL real rewards at gamma=.995's
matched terminal failure cost. Source interactions are inherited, never claimed
as a fresh independent training seed. A* is used only to collect reset states.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from pnr.carry import CarrySkillEnv
from pnr.features import BoundedGoalExtractor
from pnr.policies import load_policy_checkpoint
from pnr.telemetry import RunLogger, atomic_json
from pnr.train import (ProgressCallback, GracefulStop, ignore_worker_stop_signals,
                       make_resume_signature, resume_preflight, validate_replay_layout)

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'gamma':.995,'failure_cost':200.,'horizon':1000,'curriculum_window':5,
          'adaptive_horizon':True,'goal_transform':'delta/(0.3m+norm(delta))',
          'evaluation':'full_carry_first_retained_goal','version':1}


def recompute_real_rewards(model):
    """Real and virtual replay must share the current physical-failure cost."""
    replay=model.replay_buffer
    count=replay.buffer_size if replay.full else replay.pos
    changed=0
    for lo in range(0,count,256):
        hi=min(count,lo+256)
        next_goals=replay.next_observations['achieved_goal'][lo:hi].reshape(-1,2)
        desired=replay.observations['desired_goal'][lo:hi].reshape(-1,2)
        infos=replay.infos[lo:hi].reshape(-1)
        rewards=model.env.env_method('compute_reward',next_goals,desired,infos,indices=[0])[0]
        rewards=np.asarray(rewards).reshape(hi-lo,model.n_envs)
        changed+=int(np.count_nonzero(rewards!=replay.rewards[lo:hi]))
        replay.rewards[lo:hi]=rewards
    return {'real_transitions_checked':count*model.n_envs,'real_rewards_changed':changed,
            'virtual_rewards':'HER recomputes with the same active environment'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',default='sac_carry_corrected01')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--steps',type=int,default=100000,help='Cumulative online transitions including the source pilot')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    source=ROOT/'runs'/args.source_run
    source_config=json.loads((source/'config.json').read_text())
    if source_config['skill']!='carry' or not source_config.get('goal_features'):
        raise ValueError('Expected the measured goal-conditioned carry pilot')
    config=dict(source_config,run_id=args.run_id,steps=args.steps,device=args.device,resume=args.resume,
                horizon=PROTOCOL['horizon'],demonstrations=None,demonstrations_sha256=None,
                transfer_source_run=args.source_run,transfer_protocol=PROTOCOL,eval_episodes=1,eval_every=5000)
    directory=ROOT/'runs'/args.run_id
    if args.resume:
        saved_config=json.loads((directory/'config.json').read_text())
        if saved_config['transfer_protocol']!=PROTOCOL or saved_config['transfer_source_run']!=args.source_run:
            raise ValueError('Resume requires the exact recorded transfer protocol and source run')
        config=dict(saved_config,steps=args.steps,device=args.device,resume=True)
    preflight=resume_preflight(directory,config,ROOT) if args.resume else None
    pointer=({'generation':json.loads((directory/'transfer.json').read_text())['generation']} if args.resume
             else json.loads((source/'checkpoints/latest.json').read_text()))
    origin=source/'checkpoints'/pointer['generation']
    origin_counts=json.loads((origin/'progress.json').read_text())
    signature=make_resume_signature(config,ROOT)
    signature['sources']['scripts/train_carry_long.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if preflight and preflight['counts']['resume_signature']['sources'].get('scripts/train_carry_long.py')!=signature['sources']['scripts/train_carry_long.py']:
        raise ValueError('Resume requires the exact transfer script')
    torch.set_num_threads(1)
    np.random.seed(config['seed']);random.seed(config['seed']);torch.manual_seed(config['seed'])
    logger=RunLogger(args.run_id,resume=args.resume,name='SAC + HER · long carry',kind='train',
        algorithm='SAC + HER',skill='carry',seed=config['seed'],budget=args.steps,status='running',
        summary='Preparing long-route curriculum; inherited short-carry training is counted.')
    archive=directory/'invocations'/str(time.time_ns());archive.mkdir(parents=True)
    manifest={}
    for path in list((ROOT/'pnr').glob('*.py'))+[Path(__file__).resolve()]:
        relative=path.relative_to(ROOT);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);manifest[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(archive/'source_manifest.json',manifest);atomic_json(archive/'config.json',config)
    atomic_json(directory/'config.json',config)
    env=evaluation_env=None
    parent_pid=os.getpid()
    with GracefulStop() as stop_request:
        try:
            def make(index):
                ignore_worker_stop_signals(parent_pid)
                return CarrySkillEnv(scene_seed=config['scene_seed'],object_idx=config['object_idx'],
                    seed=config['seed']+index,horizon=PROTOCOL['horizon'],failure_cost=PROTOCOL['failure_cost'],
                    curriculum_window=PROTOCOL['curriculum_window'],adaptive_horizon=True)
            env=SubprocVecEnv([lambda index=i:make(index) for i in range(config['num_envs'])],start_method='spawn')
            evaluation_env=CarrySkillEnv(scene_seed=config['scene_seed'],object_idx=config['object_idx'],
                seed=10000,horizon=1000,evaluation='full',failure_cost=PROTOCOL['failure_cost'])
            if args.resume:
                checkpoint=preflight['checkpoint'];counts=preflight['counts']
                model=SAC.load(checkpoint/'model.zip',env=env,device=args.device)
                model.load_replay_buffer(checkpoint/'replay.pkl',truncate_last_traj=True)
                previous_status=preflight['previous_status']
            else:
                parent=load_policy_checkpoint(origin/'model.zip',device=args.device)
                model=SAC('MultiInputPolicy',env,learning_rate=3e-4,buffer_size=100000,
                    learning_starts=1000,batch_size=256,tau=.05,gamma=PROTOCOL['gamma'],train_freq=1,
                    gradient_steps=-1,policy_kwargs={'net_arch':[128,128],
                        'features_extractor_class':BoundedGoalExtractor,'features_extractor_kwargs':{'goal_scale':20.}},
                    device=args.device,seed=config['seed'],replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs={'n_sampled_goal':4,'goal_selection_strategy':'future','copy_info_dict':True})
                model.policy.load_state_dict(parent.policy.state_dict())
                with torch.no_grad():model.log_ent_coef.copy_(parent.log_ent_coef)
                model.load_replay_buffer(origin/'replay.pkl',truncate_last_traj=True)
                validate_replay_layout(model.replay_buffer,config['num_envs'],'carry')
                model.num_timesteps=parent.num_timesteps;model._n_updates=parent._n_updates
                counts=origin_counts;previous_status=json.loads((source/'status.json').read_text())
                del parent
                reward_report=recompute_real_rewards(model)
                atomic_json(directory/'transfer.json',{'source_run':args.source_run,'generation':pointer['generation'],
                    'source_model_sha256':hashlib.sha256((origin/'model.zip').read_bytes()).hexdigest(),
                    'source_progress':counts,'protocol':PROTOCOL,'replay_conversion':reward_report,
                    'parameters':'actor and both critics copied; optimizer states reset; entropy coefficient copied',
                    'policy_goal_features':'bounded displacement computed after HER relabeling',
                    'independent_seed':False,'teacher_waypoints_at_inference':False})
            validate_replay_layout(model.replay_buffer,config['num_envs'],'carry')
            if model.num_timesteps!=counts['transitions'] or model._n_updates!=counts['updates']:
                raise ValueError('Model and recorded progress counters differ')
            if args.steps<model.num_timesteps:
                raise ValueError('The cumulative budget cannot be below inherited transitions')
            if model.gamma!=PROTOCOL['gamma'] or not isinstance(model.actor.features_extractor,BoundedGoalExtractor):
                raise ValueError('Loaded policy does not match the long-route protocol')
            for i,state in enumerate(counts['training_states']):env.env_method('set_training_state',state,indices=[i])
            callback=ProgressCallback(logger,evaluation_env,5000,1,stop_request=stop_request)
            callback.restore_progress(counts,previous_status)
            callback.resume_signature=signature;callback.invocation_name=archive.name
            callback.training_setup_actions+=sum(env.get_attr('setup_action_calls'))
            callback.training_setup_ticks+=sum(env.get_attr('setup_physics_ticks'))
            callback.evaluation_setup_actions+=evaluation_env.setup_action_calls
            callback.evaluation_setup_ticks+=evaluation_env.setup_physics_ticks
            callback.training_initialization_ticks+=config['num_envs']*480
            callback.evaluation_initialization_ticks+=evaluation_env.native.settle_steps
            callback.model=model;callback.num_timesteps=model.num_timesteps
            callback.last_eval=callback.last_checkpoint=callback.last_log=model.num_timesteps
            callback.segment_start_transitions=model.num_timesteps
            best=directory/'checkpoints/best/evaluation.json'
            if args.resume and best.exists():
                old=json.loads(best.read_text());callback.best_score=(old['success_rate'],-old['mean_goal_distance'])
            logger.update(summary='Training fixed episode goals with physically collected starts; no inference waypoints.',
                          accounting=callback.accounting(),inherited_transitions=origin_counts['transitions'])
            if not stop_request.requested:callback.evaluate()
            if not stop_request.requested:
                model.learn(total_timesteps=max(0,args.steps-model.num_timesteps),callback=callback,reset_num_timesteps=False)
            if not stop_request.requested:callback.evaluate()
            callback.checkpoint()
            logger.metric(model.num_timesteps,metric_source='training_end',scope='skill',
                source_action_calls=callback.native_actions,physics_ticks=callback.physics_ticks,
                updates=model._n_updates,curriculum_levels=env.get_attr('curriculum_level'),accounting=callback.accounting())
            logger.finish('stopped' if stop_request.requested else 'complete',
                'Graceful stop; checkpoint saved.' if stop_request.requested else 'Bounded long-route transfer complete.')
        except BaseException as error:
            logger.finish('failed',f'{type(error).__name__}: {error}');raise
        finally:
            if env is not None:env.close()
            if evaluation_env is not None:evaluation_env.close()


if __name__=='__main__':main()
