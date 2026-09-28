"""Bounded SAC+HER pilots with durable metrics, evaluation and checkpoints."""
from __future__ import annotations

import argparse
from collections import deque
import json
import hashlib
import shutil
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pnr.telemetry import RunLogger, atomic_json


EVALUATION_PROTOCOL = 'stop_on_first_retained_goal'
IMMUTABLE_RESUME_FIELDS = ('skill','scene_seed','object_idx','seed','num_envs','horizon','goal_features','demonstrations','demonstrations_sha256')



class GracefulStop:
    """Defer process signals until a vector step has returned all worker replies."""
    def __init__(self):
        self.signal_number = None
        self._previous = {}

    @property
    def requested(self):
        return self.signal_number is not None

    @property
    def signal_name(self):
        return None if self.signal_number is None else signal.Signals(self.signal_number).name

    def _handle(self, number, frame):
        # Never raise, perform IO, or issue a VecEnv RPC from a signal handler.
        if self.signal_number is None:
            self.signal_number = number

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *_):
        for number, previous in self._previous.items():
            signal.signal(number, previous)
        self._previous.clear()


def ignore_worker_stop_signals(parent_pid):
    """The parent drains replies and closes workers after a requested stop."""
    if os.getpid() != parent_pid:
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, signal.SIG_IGN)


def make_resume_signature(configuration, source_root):
    """Freeze semantics relevant to this skill, not unrelated logging changes."""
    critical=['pnr/env.py','pnr/planning.py','pnr/reference.py']
    critical += ['pnr/carry.py'] if configuration['skill']=='carry' else ['pnr/skills.py','pnr/local_control.py']
    if configuration['goal_features']:
        critical.append('pnr/features.py')
    if configuration.get('demonstrations'):
        critical.append('pnr/demonstrations.py')
    root=Path(source_root)
    return {'version':1,'configuration':{key:configuration.get(key) for key in IMMUTABLE_RESUME_FIELDS},
            'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in critical}}


def validate_replay_layout(replay, num_envs, skill):
    """Reject worker/space mismatches before HER can broadcast stale replay."""
    if not isinstance(replay,HerReplayBuffer) or not replay.copy_info_dict:
        raise ValueError('Resume requires HER replay with copied physical transition info')
    if replay.n_envs != num_envs:
        raise ValueError(f'Replay worker count {replay.n_envs} != requested {num_envs}')
    feature_dim,goal_dim,action_dim=(580,2,2) if skill=='carry' else (50,3,4 if skill=='grasp' else 3)
    expected={'observation':(feature_dim,),'achieved_goal':(goal_dim,),'desired_goal':(goal_dim,)}
    if replay.action_space.shape != (action_dim,):
        raise ValueError('Replay action space does not match the requested skill')
    for key,shape in expected.items():
        if replay.observation_space.spaces[key].shape != shape:
            raise ValueError(f'Replay observation space mismatch: {key}')
        for arrays in (replay.observations,replay.next_observations):
            if arrays[key].shape != (replay.buffer_size,num_envs,*shape):
                raise ValueError(f'Replay array layout mismatch: {key}')
    if replay.actions.shape != (replay.buffer_size,num_envs,action_dim):
        raise ValueError('Replay action array layout mismatch')
    for name in ('rewards','dones','timeouts','infos','ep_start','ep_length'):
        if getattr(replay,name).shape != (replay.buffer_size,num_envs):
            raise ValueError(f'Replay metadata layout mismatch: {name}')


def resume_preflight(directory, configuration, source_root):
    """Read-only compatibility checks run before changing run metadata/config."""
    directory=Path(directory)
    pointer=json.loads((directory/'checkpoints/latest.json').read_text())
    checkpoint=directory/'checkpoints'/pointer['generation']
    counts=json.loads((checkpoint/'progress.json').read_text())
    expected=make_resume_signature(configuration,source_root)
    saved=counts.get('resume_signature')
    if saved is None:
        # Legacy runs have invocation snapshots or a single preserved source tree.
        stamp=int(checkpoint.name.rsplit('_',1)[-1])
        candidates=sorted(path for path in (directory/'invocations').glob('*')
                          if path.name.isdigit() and int(path.name)<=stamp)
        provenance=candidates[-1] if candidates else directory
        previous=json.loads((provenance/'config.json').read_text())
        manifest_path=provenance/'source_manifest.json'
        if not manifest_path.exists():
            raise ValueError('Resume refused: no preserved source signature; start a new run')
        saved={'configuration':previous,'sources':json.loads(manifest_path.read_text())}
    mismatches=[]
    for key,value in expected['configuration'].items():
        if saved['configuration'].get(key) != value:
            mismatches.append('configuration '+key)
    for name,digest in expected['sources'].items():
        if saved['sources'].get(name) != digest:
            mismatches.append('source '+name)
    if mismatches:
        raise ValueError('Resume refused: changed '+', '.join(mismatches)+'. Start a new run for changed task semantics.')
    if len(counts['training_states']) != configuration['num_envs']:
        raise ValueError('Resume refused: saved worker curriculum count differs')
    replay=load_from_pkl(checkpoint/'replay.pkl')
    validate_replay_layout(replay,configuration['num_envs'],configuration['skill'])
    status_path=directory/'status.json'
    previous_status=json.loads(status_path.read_text()) if status_path.exists() else {}
    return {'checkpoint':checkpoint,'counts':counts,'signature':expected,'previous_status':previous_status}


class ProgressCallback(BaseCallback):
    def __init__(self, logger, evaluation_env, eval_every=5000, eval_episodes=5, checkpoint_every=10000, stop_request=None):
        super().__init__()
        self.run_logger, self.eval_env = logger, evaluation_env
        self.stop_request = stop_request or GracefulStop()
        self.eval_every, self.eval_episodes = eval_every, eval_episodes
        self.checkpoint_every = checkpoint_every
        self.last_eval = self.last_checkpoint = self.last_log = 0
        self.started = time.monotonic()
        self.segment_start_transitions = 0
        self.episodes = 0
        self.recent = deque(maxlen=40)
        self.native_actions = 0
        self.physics_ticks = 0
        self.best_score = (-1., -float('inf'))
        self.evaluation_actions = self.evaluation_ticks = 0
        self.training_setup_actions = self.evaluation_setup_actions = 0
        self.training_setup_ticks = self.evaluation_setup_ticks = 0
        self.accounting_missing = []
        self.demonstration_actions = self.demonstration_ticks = 0
        self.demonstration_setup_actions = self.demonstration_setup_ticks = 0
        self.demonstration_report = None
        self.training_initialization_ticks = self.evaluation_initialization_ticks = 0
        self.demonstration_initialization_ticks = 0
        self.resume_signature = None
        self.invocation_name = None
        self._checkpoint_due = False
        self._pending_replay_position = None

    def _on_step(self):
        self._pending_replay_position=self.model.replay_buffer.pos
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            self.native_actions += int(info.get('action_calls_delta', 1))
            self.physics_ticks += int(info.get('physics_ticks_delta', 5))
            if done:
                self.episodes += 1
                self.recent.append(float(info.get('episode_success', info.get('is_success', False))))
        if self.stop_request.requested:
            return False
        if self.num_timesteps - self.last_log >= 500:
            elapsed = max(time.monotonic() - self.started, 1e-6)
            self.run_logger.metric(self.num_timesteps, scope='skill', metric_source='training',
                train_success_rate=float(np.mean(self.recent)) if self.recent else 0.,
                episodes=self.episodes, source_action_calls=self.native_actions,
                physics_ticks=self.physics_ticks, wall_seconds=elapsed,
                transitions_per_second=(self.num_timesteps-self.segment_start_transitions)/elapsed,
                updates=int(getattr(self.model, '_n_updates', 0)),
                curriculum_levels=self.model.env.get_attr('curriculum_level'),accounting=self.accounting())
            self.last_log = self.num_timesteps
        if self.num_timesteps - self.last_eval >= self.eval_every:
            self.evaluate()
            self.last_eval = self.num_timesteps
        if self.stop_request.requested:
            return False
        if self.num_timesteps - self.last_checkpoint >= self.checkpoint_every:
            self._checkpoint_due=True
        return True

    def _on_rollout_end(self):
        # SB3 _on_step precedes _store_transition; checkpoint only after storage.
        self._pending_replay_position=None
        if self._checkpoint_due:
            self.checkpoint()
            self.last_checkpoint=self.num_timesteps
            self._checkpoint_due=False

    def accounting(self):
        values={'accounting_version':2,'training_source_action_calls':self.native_actions,'training_physics_ticks':self.physics_ticks,
            'evaluation_source_action_calls':self.evaluation_actions,'evaluation_physics_ticks':self.evaluation_ticks,
            'training_setup_source_action_calls':self.training_setup_actions,
            'evaluation_setup_source_action_calls':self.evaluation_setup_actions,
            'training_setup_physics_ticks':self.training_setup_ticks,
            'evaluation_setup_physics_ticks':self.evaluation_setup_ticks,
            'demonstration_source_action_calls':self.demonstration_actions,
            'demonstration_physics_ticks':self.demonstration_ticks,
            'demonstration_setup_source_action_calls':self.demonstration_setup_actions,
            'demonstration_setup_physics_ticks':self.demonstration_setup_ticks,
            'training_initialization_physics_ticks':self.training_initialization_ticks,
            'evaluation_initialization_physics_ticks':self.evaluation_initialization_ticks,
            'demonstration_initialization_physics_ticks':self.demonstration_initialization_ticks,
            'demonstration_transitions_imported':(0 if self.demonstration_report is None else self.demonstration_report['transitions_imported']),
            'complete':not self.accounting_missing,'legacy_missing':list(self.accounting_missing)}
        values['total_source_action_calls_known']=sum(values[name] for name in (
            'training_source_action_calls','evaluation_source_action_calls',
            'training_setup_source_action_calls','evaluation_setup_source_action_calls',
            'demonstration_source_action_calls','demonstration_setup_source_action_calls'))
        values['total_physics_ticks_known']=sum(values[name] for name in (
            'training_physics_ticks','evaluation_physics_ticks','training_setup_physics_ticks','evaluation_setup_physics_ticks',
            'demonstration_physics_ticks','demonstration_setup_physics_ticks',
            'training_initialization_physics_ticks','evaluation_initialization_physics_ticks','demonstration_initialization_physics_ticks'))
        return values

    def add_demonstrations(self,report):
        if self.demonstration_report is not None:
            raise ValueError('Demonstration replay accounting must not be added twice')
        metadata=report['dataset_metadata']
        required=('demonstration_source_action_calls','demonstration_physics_ticks',
                  'setup_source_action_calls','setup_physics_ticks')
        if any(key not in metadata or int(metadata[key])<0 for key in required):
            raise ValueError('Demonstration data requires complete nonnegative collection/setup counters')
        self.demonstration_report=report
        self.demonstration_actions=int(metadata['demonstration_source_action_calls'])
        self.demonstration_ticks=int(metadata['demonstration_physics_ticks'])
        self.demonstration_setup_actions=int(metadata['setup_source_action_calls'])
        self.demonstration_setup_ticks=int(metadata['setup_physics_ticks'])
        self.demonstration_initialization_ticks=int(metadata.get('initialization_physics_ticks',0))
        if 'initialization_physics_ticks' not in metadata:
            self.accounting_missing.append('demonstration_initialization_physics_ticks')

    def restore_progress(self,counts,previous_status=None):
        self.native_actions=counts['source_action_calls'];self.physics_ticks=counts['physics_ticks']
        account=counts.get('accounting')
        if account is None:
            # Keep known previous setup values, explicitly flag unavailable history.
            previous_status=previous_status or {}
            self.training_setup_actions=int(previous_status.get('setup_action_calls',0))
            self.evaluation_setup_actions=int(previous_status.get('evaluation_setup_action_calls',0))
            self.training_setup_ticks=self.training_setup_actions*5
            self.evaluation_setup_ticks=self.evaluation_setup_actions*5
            self.accounting_missing=['historical_evaluation_interactions','historical_setup_totals_before_checkpoint','historical_initialization_physics_ticks']
        else:
            self.evaluation_actions=account['evaluation_source_action_calls']
            self.evaluation_ticks=account['evaluation_physics_ticks']
            self.training_setup_actions=account['training_setup_source_action_calls']
            self.evaluation_setup_actions=account['evaluation_setup_source_action_calls']
            self.training_setup_ticks=account['training_setup_physics_ticks']
            self.evaluation_setup_ticks=account['evaluation_setup_physics_ticks']
            self.accounting_missing=list(account.get('legacy_missing',[]))
            self.demonstration_actions=int(account.get('demonstration_source_action_calls',0))
            self.demonstration_ticks=int(account.get('demonstration_physics_ticks',0))
            self.demonstration_setup_actions=int(account.get('demonstration_setup_source_action_calls',0))
            self.demonstration_setup_ticks=int(account.get('demonstration_setup_physics_ticks',0))
            self.demonstration_report=counts.get('demonstrations')
            self.training_initialization_ticks=int(account.get('training_initialization_physics_ticks',0))
            self.evaluation_initialization_ticks=int(account.get('evaluation_initialization_physics_ticks',0))
            self.demonstration_initialization_ticks=int(account.get('demonstration_initialization_physics_ticks',0))
            if account.get('accounting_version',1)<2:
                self.accounting_missing.append('historical_initialization_physics_ticks')
        history=counts.get('callback_state',{})
        self.episodes=int(history.get('episodes',0))
        self.recent=deque(history.get('recent',[]),maxlen=40)

    def evaluate(self):
        outcomes, distances = [], []
        for episode in range(self.eval_episodes):
            if self.stop_request.requested:
                return False
            observation, _ = self.eval_env.reset(seed=10000+episode)
            success = False
            for _ in range(self.eval_env.horizon):
                if self.stop_request.requested:
                    return False
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = self.eval_env.step(action)
                self.evaluation_actions+=int(info.get('action_calls_delta',1))
                self.evaluation_ticks+=int(info.get('physics_ticks_delta',5))
                if self.stop_request.requested:
                    return False
                failure=bool(terminated or info.get('physical_failure',False)
                             or info.get('drop_events',0) or not info.get('retention_ok',True))
                success=bool(info['is_success'] and info.get('bilateral_contact',False) and not failure)
                if success or terminated or truncated:
                    break
            outcomes.append(success)
            distances.append(float(info['goal_distance']))
        self.run_logger.metric(self.num_timesteps, scope='skill', metric_source='evaluation',
            success_rate=float(np.mean(outcomes)), eval_episodes=len(outcomes),
            mean_goal_distance=float(np.mean(distances)),
            evaluation_scope=info.get('evaluation_scope','easy_manipulation_diagnostic'),
            eval_outcomes=[bool(x) for x in outcomes],
            evaluation_protocol=EVALUATION_PROTOCOL,accounting=self.accounting())
        score=(float(np.mean(outcomes)),-float(np.mean(distances)))
        if score > self.best_score:
            self.best_score=score
            best_dir=self.run_logger.directory/'checkpoints'/'best'
            best_dir.mkdir(parents=True,exist_ok=True)
            temporary=best_dir/'model.pending.zip'
            self.model.save(temporary)
            os.replace(temporary,best_dir/'model.zip')
            atomic_json(best_dir/'evaluation.json',{'transitions':self.num_timesteps,
                'success_rate':score[0],'mean_goal_distance':-score[1],
                'episodes':self.eval_episodes,'evaluation_scope':info.get('evaluation_scope','easy_manipulation_diagnostic'),
                'evaluation_protocol':EVALUATION_PROTOCOL})
        self.run_logger.event('Skill evaluation: %d/%d successful.' % (sum(outcomes), len(outcomes)))
        print(json.dumps({'transitions': self.num_timesteps, 'skill_success_rate': float(np.mean(outcomes)),
                          'goal_distance': float(np.mean(distances))}), flush=True)
        return True

    def checkpoint(self):
        if getattr(self.model.env, 'waiting', False):
            raise RuntimeError('Checkpoint refused while a vector step is waiting for worker replies')
        # An interrupt can arrive during on_step evaluation, before SB3 stores
        # this already-executed vector transition. Flush it exactly once.
        if self._pending_replay_position is not None:
            replay=self.model.replay_buffer
            if replay.pos==self._pending_replay_position:
                fields=self.locals
                self.model._store_transition(replay,fields['buffer_actions'],fields['new_obs'],
                    fields['rewards'],fields['dones'],fields['infos'])
            elif replay.pos != (self._pending_replay_position+1)%replay.buffer_size:
                raise RuntimeError('Checkpoint refused: unexpected pending replay position')
            self._pending_replay_position=None
        training_states = self.model.env.env_method('get_training_state')
        if len(training_states) != self.model.n_envs or not all(isinstance(state, dict) for state in training_states):
            raise RuntimeError('Checkpoint refused: invalid worker training-state replies')
        root = self.run_logger.directory/'checkpoints'
        root.mkdir(exist_ok=True)
        generation = 'step_%09d_%d' % (self.num_timesteps,time.time_ns())
        folder = root/generation
        folder.mkdir()
        self.model.save(folder/'model')
        self.model.save_replay_buffer(folder/'replay.pkl')
        atomic_json(folder/'progress.json', {'transitions': self.num_timesteps,
            'source_action_calls': self.native_actions, 'physics_ticks': self.physics_ticks,
            'updates': int(getattr(self.model, '_n_updates', 0)),
            'training_states':training_states,
            'stop_requested':self.stop_request.requested,'stop_signal':self.stop_request.signal_name,
            'resume_signature':self.resume_signature,'invocation':self.invocation_name,
            'accounting':self.accounting(),'demonstrations':self.demonstration_report,'callback_state':{'episodes':self.episodes,'recent':list(self.recent)},
            'replay_position':self.model.replay_buffer.pos,'replay_n_envs':self.model.replay_buffer.n_envs,
            'resume_policy': 'fresh physical episode with worker curriculum/RNG restored; learner RNG reseeded; truncate unfinished replay episode'})
        # Publish a complete generation only after every member was written.
        atomic_json(root/'latest.json', {'generation':generation})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill', choices=['grasp','carry','reach'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scene-seed', type=int, default=42)
    parser.add_argument('--object-idx', type=int, default=14)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gradient-steps',type=int,default=-1)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--goal-features', action='store_true')
    parser.add_argument('--demonstrations',help='Physical replay-seed dataset, imported only for a fresh run')
    args = parser.parse_args()
    if min(args.num_envs,args.horizon,args.eval_every,args.eval_episodes)<1 or args.steps<0:
        parser.error('Worker count, horizon and evaluation settings must be positive; steps must be nonnegative')
    if args.gradient_steps < -1:
        parser.error('--gradient-steps must be -1 or nonnegative')
    with GracefulStop() as stop_request:
        run_training(args, stop_request)


def run_training(args, stop_request):
    from pnr.skills import make_skill_env
    if args.demonstrations:
        args.demonstrations=str(Path(args.demonstrations).expanduser().resolve())

    torch.set_num_threads(1)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    configuration=vars(args) | {
        'phase': 'development', 'net_arch': [128,128], 'batch_size':256,
        'learning_rate':3e-4, 'buffer_size':100000, 'n_sampled_goal':4,
        'train_freq':1,'gradient_steps':args.gradient_steps,
        'training_reward': 'See skill compute_reward; fixed retention obligations are not relabeled.',
        'upstream_commit':'54ab2d1faa54d1aa7e2738f114531a76c6514cd3',
        'demonstrations_sha256':(hashlib.sha256(Path(args.demonstrations).read_bytes()).hexdigest() if args.demonstrations else None)}
    source_root=Path(__file__).resolve().parents[1]
    run_directory=source_root/'runs'/args.run_id
    preflight=resume_preflight(run_directory,configuration,source_root) if args.resume else None
    signature=make_resume_signature(configuration,source_root)
    logger = RunLogger(args.run_id, resume=args.resume, name='SAC + HER · '+args.skill,
        kind='train', algorithm='SAC + HER', skill=args.skill, seed=args.seed,
        status='running', budget=args.steps, summary='Initializing physical skill states.')
    # Every invocation retains its exact source, including resumed segments.
    archive=logger.directory/'invocations'/str(time.time_ns())
    archive.mkdir(parents=True)
    source_root=Path(__file__).resolve().parents[1]
    manifest={}
    for path in sorted((source_root/'pnr').glob('*.py')):
        relative=path.relative_to(source_root)
        target=archive/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        manifest[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(archive/'config.json',configuration)
    atomic_json(archive/'source_manifest.json',manifest)
    atomic_json(logger.directory/'config.json',configuration)
    env = evaluation_env = None
    try:
        parent_pid=os.getpid()
        def make(index, evaluation=False):
            ignore_worker_stop_signals(parent_pid)
            instance=make_skill_env(args.skill, scene_seed=args.scene_seed, object_idx=args.object_idx,
                seed=args.seed+index, horizon=args.horizon, evaluation=evaluation)
            instance.initialization_physics_ticks=instance.native.settle_steps
            return instance
        factories=[lambda index=index: make(index) for index in range(args.num_envs)]
        env = DummyVecEnv(factories) if args.num_envs==1 else SubprocVecEnv(factories,start_method='spawn')
        evaluation_env = make(10000, evaluation=True)
        demonstration_report=None
        if args.resume:
            checkpoint=preflight['checkpoint']
            model = SAC.load(checkpoint/'model.zip', env=env, device=args.device)
            model.load_replay_buffer(checkpoint/'replay.pkl', truncate_last_traj=True)
            model.gradient_steps=args.gradient_steps
            counts=preflight['counts']
            if model.num_timesteps!=counts['transitions'] or model._n_updates!=counts['updates']:
                raise ValueError('Checkpoint model/progress counters differ')
            validate_replay_layout(model.replay_buffer,args.num_envs,args.skill)
            for index,state in enumerate(counts['training_states']):
                env.env_method('set_training_state',state,indices=[index])
        else:
            policy_kwargs={'net_arch':[128,128]}
            if args.goal_features:
                from pnr.features import RelativeGoalExtractor
                policy_kwargs.update(features_extractor_class=RelativeGoalExtractor,
                    features_extractor_kwargs={'goal_scale':20. if args.skill=='carry' else 5.})
            model = SAC('MultiInputPolicy', env, learning_rate=3e-4, buffer_size=100000,
                learning_starts=max(1000,args.horizon*args.num_envs+1), batch_size=256,
                tau=.05, gamma=.98, train_freq=1, gradient_steps=args.gradient_steps,
                policy_kwargs=policy_kwargs, device=args.device, seed=args.seed,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs={'n_sampled_goal':4,'goal_selection_strategy':'future',
                                      'copy_info_dict':True}, verbose=0)
            if args.demonstrations:
                from pnr.demonstrations import seed_replay
                demonstration_report=seed_replay(model,args.demonstrations)
                if demonstration_report['sha256']!=configuration['demonstrations_sha256']:
                    raise ValueError('Demonstration dataset changed while the run initialized')
                atomic_json(logger.directory/'demonstrations.json',demonstration_report)
                logger.event('Seeded replay with %d physical demonstration transitions; no actor cloning.' % demonstration_report['transitions_imported'])
        setup_calls=sum(env.get_attr('setup_action_calls'))
        logger.update(summary='Training on valid physical starts.',
                      setup_action_calls=setup_calls, evaluation_setup_action_calls=evaluation_env.setup_action_calls)
        logger.event('Physical start states ready; %d setup action calls.' % setup_calls)
        callback = ProgressCallback(logger,evaluation_env,args.eval_every,args.eval_episodes,stop_request=stop_request)
        callback.resume_signature=signature
        callback.invocation_name=archive.name
        best_file=logger.directory/'checkpoints'/'best'/'evaluation.json'
        if best_file.exists():
            best=json.loads(best_file.read_text())
            if best.get('evaluation_protocol')==EVALUATION_PROTOCOL:
                callback.best_score=(best['success_rate'],-best['mean_goal_distance'])
            else:
                previous_best=best_file.parent.parent/'historical_best'/str(time.time_ns())
                previous_best.mkdir(parents=True)
                shutil.copy2(best_file,previous_best/'evaluation.json')
                if (best_file.parent/'model.zip').exists():
                    shutil.copy2(best_file.parent/'model.zip',previous_best/'model.zip')
                logger.event('Preserved previous best under its historical evaluation protocol; new protocol comparison starts fresh.')
        if args.resume:
            callback.restore_progress(counts,preflight['previous_status'])
            callback.last_eval=callback.last_checkpoint=callback.last_log=model.num_timesteps
        if demonstration_report is not None:
            callback.add_demonstrations(demonstration_report)
        callback.training_initialization_ticks+=sum(env.get_attr('initialization_physics_ticks'))
        callback.evaluation_initialization_ticks+=evaluation_env.initialization_physics_ticks
        callback.training_setup_actions+=setup_calls
        callback.evaluation_setup_actions+=evaluation_env.setup_action_calls
        callback.training_setup_ticks+=setup_calls*5
        callback.evaluation_setup_ticks+=evaluation_env.setup_action_calls*5
        atomic_json(archive/'setup_accounting.json',{'training_source_action_calls':setup_calls,
            'evaluation_source_action_calls':evaluation_env.setup_action_calls,'ticks_per_action':5,
            'training_initialization_physics_ticks':sum(env.get_attr('initialization_physics_ticks')),
            'evaluation_initialization_physics_ticks':evaluation_env.initialization_physics_ticks})
        logger.update(accounting=callback.accounting())
        callback.model=model
        callback.num_timesteps=model.num_timesteps
        callback.segment_start_transitions=model.num_timesteps
        if not stop_request.requested:
            callback.evaluate()
        remaining=max(0,args.steps-model.num_timesteps) if args.resume else args.steps
        if remaining and not stop_request.requested:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not args.resume)
        if not stop_request.requested:
            callback.evaluate()
        callback.checkpoint()
        logger.metric(model.num_timesteps,metric_source='training_end',scope='skill',
            source_action_calls=callback.native_actions,physics_ticks=callback.physics_ticks,
            updates=int(model._n_updates),curriculum_levels=env.get_attr('curriculum_level'),accounting=callback.accounting())
        if stop_request.requested:
            logger.finish('stopped',f'{stop_request.signal_name} requested; drained vector steps and saved a complete checkpoint.')
        else:
            logger.finish(summary='Bounded pilot complete; skill evaluation recorded.')
    except KeyboardInterrupt:
        # Unexpected explicit KeyboardInterrupt is not a safe VecEnv boundary.
        # Normal SIGINT/SIGTERM use GracefulStop and reach the checkpoint above.
        logger.finish('stopped','Unexpected interruption; preserving the last published checkpoint without an unsafe worker query.')
        raise
    except BaseException as error:
        logger.finish('failed',f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None: env.close()
        if evaluation_env is not None: evaluation_env.close()


if __name__ == '__main__':
    main()
