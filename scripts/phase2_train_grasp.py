#!/usr/bin/env python3
"""Balanced multi-object BC initialization, critic calibration, ordinary SAC+HER."""
from __future__ import annotations
import argparse
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pnr.demonstrations import seed_replay
from pnr.general_manipulation import GraspSchema, GeneralGraspFeatures, MultiObjectGrasp, make_grasp_case
from pnr.sac_initialization import calibrate_actor_std, critic_only_warmup
from pnr.telemetry import RunLogger, atomic_json
from pnr.train import GracefulStop, ignore_worker_stop_signals


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [serial(item) for item in value]
    return value


def dataset_rows(dataset):
    rows, labels, object_ids = [], [], []
    success_episodes = 0
    for episode in dataset['episodes']:
        summary = episode['summary']
        if not summary['successful_physical_lift_and_hold']:
            continue
        success_episodes += 1
        for item in episode['transitions']:
            phase = item['teacher_phase']
            if phase == 'held_goal_correction':
                phase = ('settle' if summary['first_goal_decision'] is not None
                         and item['step'] >= summary['first_goal_decision'] else 'lift')
            rows.append(item)
            labels.append(phase)
            object_ids.append(summary['object_idx'])
    labels, object_ids = np.asarray(labels), np.asarray(object_ids)
    groups = {}
    for obj in sorted(set(object_ids)):
        for phase in ('approach', 'close', 'lift', 'settle'):
            indices = np.flatnonzero((object_ids == obj) & (labels == phase))
            if len(indices): groups[f'{obj}:{phase}'] = indices
    return rows, groups, {'successful_episodes_for_bc': success_episodes,
                         'failed_episodes_excluded_from_bc': len(dataset['episodes'])-success_episodes,
                         'all_episodes_retained_for_replay': True,
                         'bc_transitions': len(rows), 'strata_counts': {key: len(value) for key, value in groups.items()}}


class Progress(BaseCallback):
    def __init__(self, logger, cases, args, stop):
        super().__init__()
        self.run_logger, self.cases, self.args, self.stop = logger, cases, args, stop
        self.evaluation = {}
        self.train_calls = self.train_ticks = self.eval_calls = self.eval_ticks = 0
        self.eval_setup_calls = self.eval_setup_ticks = self.eval_settling_ticks = 0
        self.last_log = self.last_eval = self.last_save = 0
        self.episodes = 0
        self.recent = deque(maxlen=100)
        self.best_online = (-1., -1e9)
        self.initialization = {}
        self.started = time.monotonic()
        self.checkpoint_due = False
        self.initialized_validation_rate = None
        self.collapse_count = 0
        self.early_stop_reason = None

    def accounting(self):
        try:
            setup = sum(self.model.env.get_attr('setup_action_calls'))
            ticks = sum(self.model.env.get_attr('setup_physics_ticks'))
            settling = sum(self.model.env.get_attr('initialization_physics_ticks'))
        except AttributeError:
            setup = ticks = settling = 0
        return {'online_decisions': int(self.model.num_timesteps), 'online_gradient_updates': int(self.model._n_updates),
                'training_source_action_calls': self.train_calls, 'training_physics_ticks': self.train_ticks,
                'training_setup_source_action_calls': setup, 'training_setup_physics_ticks': ticks,
                'training_initialization_physics_ticks': settling,
                'evaluation_source_action_calls': self.eval_calls, 'evaluation_physics_ticks': self.eval_ticks,
                'evaluation_setup_source_action_calls': self.eval_setup_calls,
                'evaluation_setup_physics_ticks': self.eval_setup_ticks,
                'evaluation_initialization_physics_ticks': self.eval_settling_ticks}

    def evaluate(self, label):
        records = []
        for case in self.cases:
            if self.stop.requested: break
            key = (case['object_idx'], case['scene_seed'])
            try:
                if key not in self.evaluation:
                    env = make_grasp_case(*key, seed=71000, horizon=self.args.eval_horizon, evaluation=True)
                    self.evaluation[key] = env
                    self.eval_setup_calls += env.setup_action_calls
                    self.eval_setup_ticks += env.setup_action_calls*env.native.substeps
                    self.eval_settling_ticks += env.native.settle_steps
                env = self.evaluation[key]
                env.reset(seed=71000, options={'full_pregrasp': True})
                env.goal = env.base_goal.copy()
                observation = env._observation()
                trajectory, held = [], []
                success = False
                for decision in range(self.args.eval_horizon):
                    action, _ = self.model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    self.eval_calls += info['action_calls_delta']
                    self.eval_ticks += info['physics_ticks_delta']
                    trajectory.append({'decision': decision+1, 'action': action.tolist(),
                                       'object_position': env.native.observe()['object_pos'].tolist(),
                                       'goal_distance': info['goal_distance'], 'bilateral_contact': info['bilateral_contact'],
                                       'action_rejected': info['action_rejected']})
                    if info['is_success'] and not terminated and not env.native.drop_events:
                        target = env.native.agent_qpos.copy()
                        target[env.native.finger_slots] = 0.
                        for _ in range(50):
                            state, _ = env.native.step(target)
                            held.append(float(state['object_pos'][2]))
                        self.eval_calls += 50
                        self.eval_ticks += 50*env.native.substeps
                        metrics = env.native.evaluate()
                        success = bool(metrics['bilateral_contact'] and metrics['lifted']
                                       and metrics['stable_hold_steps'] >= 50 and metrics['drop_events'] == 0
                                       and np.ptp(held) < .02)
                        break
                    if terminated or truncated or env.native.drop_events: break
                record = dict(case, success=success, goal_distance=trajectory[-1]['goal_distance'],
                              decisions=len(trajectory), source_action_calls=env.native.step_count,
                              physics_ticks=env.native.physics_steps, metrics=env.native.evaluate(),
                              height_range_final_hold_m=float(np.ptp(held)) if held else None,
                              trajectory=trajectory)
            except Exception as error:
                costs = getattr(error, 'setup_accounting', {})
                self.eval_setup_calls += costs.get('source_action_calls', 0)
                self.eval_setup_ticks += costs.get('physics_ticks', 0)
                self.eval_settling_ticks += costs.get('initialization_physics_ticks', 0)
                record = dict(case, success=False, goal_distance=None, error=repr(error), failed_setup_costs=costs)
            records.append(record)
            atomic_json(self.run_logger.directory/'artifacts'/f'evaluation_{label}_partial.json',
                        {'scope': 'multi_object_full_pregrasp', 'status': 'running',
                         'online_decisions': int(self.model.num_timesteps), 'records': records,
                         'accounting': self.accounting()})
            self.run_logger.update(evaluation_cases_completed=len(records), evaluation_cases_total=len(self.cases))
        validation = [r for r in records if r['partition'] == 'validation']
        train = [r for r in records if r['partition'] == 'train']
        rate = float(np.mean([r['success'] for r in validation])) if validation else 0.
        distance = float(np.mean([r['goal_distance'] if r['goal_distance'] is not None else 1. for r in validation])) if validation else 1.
        path = self.run_logger.directory/'artifacts'/f'evaluation_{label}.json'
        atomic_json(path, {'scope': 'multi_object_full_pregrasp', 'online_decisions': int(self.model.num_timesteps),
                           'teacher_at_inference': False, 'records': records,
                           'validation_success_rate': rate, 'validation_mean_goal_distance': distance})
        self.run_logger.artifact('artifacts/'+path.name, 'Physical grasp evaluation '+label)
        self.run_logger.metric(self.model.num_timesteps, scope='skill', metric_source='evaluation',
                               evaluation_scope='multi_object_full_pregrasp', success_rate=rate,
                               mean_goal_distance=distance, eval_episodes=len(validation),
                               training_pointseed_success_rate=float(np.mean([r['success'] for r in train])) if train else None,
                               accounting=self.accounting())
        if self.model.num_timesteps > 0:
            if (rate, -distance) > self.best_online:
                self.best_online = (rate, -distance)
                self.model.save(self.run_logger.directory/'checkpoints/best_online')
                atomic_json(self.run_logger.directory/'checkpoints/best_online_evaluation.json',
                            {'success_rate': rate, 'goal_distance': distance,
                             'online_decisions': int(self.model.num_timesteps), 'evaluation': str(path)})
            if self.initialized_validation_rate is not None and self.initialized_validation_rate >= .8 and rate < .4:
                self.collapse_count += 1
            else:
                self.collapse_count = 0
            if self.collapse_count >= 2:
                self.early_stop_reason = 'Two consecutive validation checks below40% after>=80% initialization; retain failed run and diagnose.'
        return rate, distance

    def checkpoint(self, label):
        directory = self.run_logger.directory/'checkpoints'
        self.model.save(directory/label)
        self.model.save_replay_buffer(directory/'latest_replay.pkl')
        try: states = self.model.env.env_method('get_training_state')
        except AttributeError: states = []
        atomic_json(directory/'latest.json', serial({'model': label+'.zip', 'accounting': self.accounting(),
                    'initialization': self.initialization, 'training_states': states,
                    'resume_note': 'Restart physical episodes and truncate open HER tails; no bitwise physical continuation.'}))
        torch.save({'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}, directory/'rng.pt')

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            self.train_calls += info['action_calls_delta']
            self.train_ticks += info['physics_ticks_delta']
            if done:
                self.episodes += 1
                self.recent.append(float(info['episode_success']))
        if self.stop.requested: return False
        if self.num_timesteps-self.last_log >= 1000:
            self.run_logger.metric(self.num_timesteps, scope='skill', metric_source='training',
                                   train_success_rate=float(np.mean(self.recent)) if self.recent else 0.,
                                   episodes=self.episodes, wall_seconds=time.monotonic()-self.started,
                                   accounting=self.accounting())
            self.last_log = self.num_timesteps
        early_due = any(self.last_eval < point <= self.num_timesteps for point in (256, 1024))
        if early_due or self.num_timesteps-self.last_eval >= self.args.eval_every:
            self.evaluate(f'online_{self.num_timesteps:07d}')
            self.last_eval = self.num_timesteps
        if self.num_timesteps-self.last_save >= self.args.checkpoint_every:
            self.checkpoint_due = True
        return not self.stop.requested and self.early_stop_reason is None

    def _on_rollout_end(self):
        if self.checkpoint_due:
            self.checkpoint(f'online_{self.num_timesteps:07d}')
            self.last_save = self.num_timesteps
            self.checkpoint_due = False

    def close(self):
        for env in self.evaluation.values(): env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--demonstrations', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--bc-updates', type=int, default=8000)
    parser.add_argument('--bc-eval-every', type=int, default=4000)
    parser.add_argument('--warmup-updates', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--eval-horizon', type=int, default=120)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--validation-seeds', nargs='+', type=int, default=[43001, 43002])
    parser.add_argument('--seed', type=int, default=8101)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.steps > 40000:
        raise ValueError('Initial pilot limited to eight workers and40k online decisions')
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.set_num_threads(1)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = pickle.loads(args.demonstrations.read_bytes())
    training_cases = [{'object_idx': r['object_idx'], 'scene_seed': r['scene_seed']}
                      for r in dataset['metadata']['cases'] if r['case_status'] == 'complete']
    training_cases.sort(key=lambda case: (case['scene_seed'], case['object_idx']))
    if not training_cases or len(training_cases) < args.workers:
        raise ValueError('Need at least one prepared training case per worker')
    if set(args.validation_seeds) & {case['scene_seed'] for case in training_cases}:
        raise ValueError('Validation point-cloud seeds overlap training')
    objects = sorted({case['object_idx'] for case in training_cases})
    evaluation_cases = [{'object_idx': index, 'scene_seed': seed, 'partition': 'validation'}
                        for index in objects for seed in args.validation_seeds]
    evaluation_cases += [{'object_idx': index, 'scene_seed': 42, 'partition': 'train'} for index in objects]
    logger = RunLogger(args.run_id, kind='train', skill='grasp', algorithm='BC-initialized SAC+HER',
                       name='Phase 2 · multi-object SAC + HER grasp', budget=args.steps,
                       summary='Balanced object/phase actor initialization; physical validation before ordinary SAC.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir()
    checkpoints = logger.directory/'checkpoints'; checkpoints.mkdir()
    sources = logger.directory/'sources'; sources.mkdir()
    source_hashes = {}
    for path in list((ROOT/'pnr').glob('*.py'))+[Path(__file__).resolve()]:
        relative = path.relative_to(ROOT)
        target = sources/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(logger.directory/'source_manifest.json', source_hashes)
    config = serial(vars(args) | {'demonstrations': str(args.demonstrations),
        'demonstrations_sha256': hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
        'training_cases': training_cases, 'evaluation_cases': evaluation_cases,
        'observation_version': 'antipodal-relative-grasp-v1', 'gamma': .99,
        'n_sampled_goal': 1, 'alpha_initial': .005, 'target_entropy': -8.,
        'supervised_loss_during_online_training': False, 'independent_final_study_seed': False,
        'bc_selection': 'Best validation success then goal distance at declared initialization checks.',
        'online_collapse_rule': 'Stop after two checks below40% if BC initial success at least80%; retain outcomes.'})
    atomic_json(logger.directory/'config.json', config)
    rows, groups, bc_metadata = dataset_rows(dataset)
    obs = {key: torch.as_tensor(np.stack([r['observation'][key] for r in rows]), device=args.device)
           for key in GraspSchema().observation_space.spaces}
    actions = torch.as_tensor(np.stack([r['action'] for r in rows]), device=args.device)
    schema = DummyVecEnv([lambda: GraspSchema() for _ in range(args.workers)])
    model = SAC('MultiInputPolicy', schema, learning_rate=args.learning_rate, buffer_size=150000,
                learning_starts=0, batch_size=args.batch_size, tau=.01, gamma=.99,
                train_freq=1, gradient_steps=1, ent_coef='auto_0.005', target_entropy=-8.,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs={'n_sampled_goal': 1, 'goal_selection_strategy': 'future', 'copy_info_dict': True},
                policy_kwargs={'net_arch': [128, 128], 'features_extractor_class': GeneralGraspFeatures},
                device=args.device, seed=args.seed)
    model.set_logger(configure(str(logger.directory/'sb3'), ['csv']))
    progress = Progress(logger, evaluation_cases, args, None); progress.model = model
    physical = None
    with GracefulStop() as stop:
        progress.stop = stop
        try:
            optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
            rng = np.random.default_rng(args.seed)
            losses = []
            best = (-1., -1e9); best_state = None; best_update = None
            started = time.monotonic()
            strata = list(groups.values())
            for update in range(1, args.bc_updates+1):
                group_ids = rng.integers(0, len(strata), size=args.batch_size)
                indices = np.array([rng.choice(strata[group]) for group in group_ids])
                batch = {key: value[indices] for key, value in obs.items()}
                predictions = model.actor(batch, deterministic=True)
                loss = torch.mean((predictions-actions[indices])**2)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.)
                optimizer.step(); losses.append(float(loss.detach()))
                if update % 500 == 0:
                    logger.metric(0, scope='actor_initialization', metric_source='supervised', bc_updates=update,
                                  bc_training_loss=float(np.mean(losses[-500:])), supervised_examples=update*args.batch_size)
                if update % args.bc_eval_every == 0 or update == args.bc_updates:
                    model.save(checkpoints/f'bc_{update:06d}')
                    rate, distance = progress.evaluate(f'bc_{update:06d}')
                    if (rate, -distance) > best:
                        best = (rate, -distance)
                        best_state = copy.deepcopy(model.actor.state_dict()); best_update = update
                if stop.requested: break
            if stop.requested: return
            if best_state is None: raise RuntimeError('No completed physical BC evaluation')
            model.actor.load_state_dict(best_state)
            progress.initialized_validation_rate = best[0]
            progress.initialization['bc'] = bc_metadata | {'updates_completed': update, 'selected_update': best_update,
                'selected_validation_rate': best[0], 'batch_size': args.batch_size,
                'supervised_examples': update*args.batch_size, 'wall_seconds': time.monotonic()-started}
            progress.initialization['std_calibration'] = calibrate_actor_std(model, -3.)
            model.save(checkpoints/'bc_initialized')
            logger.update(summary='Importing all measured demonstrations and calibrating fresh critics with actor fixed.')
            replay_metadata = seed_replay(model, args.demonstrations)
            progress.initialization['replay'] = {key: value for key, value in replay_metadata.items() if key != 'dataset_metadata'}
            atomic_json(artifacts/'demonstration_accounting.json', dataset['metadata'])
            logger.artifact('artifacts/demonstration_accounting.json', 'All demonstration attempts and collection costs')
            progress.initialization['critic_warmup'] = critic_only_warmup(model, args.warmup_updates,
                args.batch_size, learning_rate=3e-4, log_every=200,
                on_log=lambda row: logger.metric(0, scope='actor_initialization', metric_source='critic_initialization', **row))
            progress.evaluate('initialized_zero_online')
            progress.checkpoint('initialized')
            if stop.requested: return
            parent_pid = os.getpid()
            def make(index):
                ignore_worker_stop_signals(parent_pid)
                return MultiObjectGrasp(training_cases[index::args.workers], seed=args.seed+index,
                                         horizon=args.horizon, cache_size=4, episodes_per_case=4)
            physical = SubprocVecEnv([lambda index=index: make(index) for index in range(args.workers)], start_method='spawn')
            model.set_env(physical)
            model.replay_buffer.env = None; model.replay_buffer.set_env(physical)
            logger.update(summary='Ordinary online SAC + HER over physical multi-object starts; no imitation loss or teacher actions.')
            model.learn(total_timesteps=args.steps, callback=progress)
            progress.evaluate(f'final_{model.num_timesteps:07d}')
            progress.checkpoint('final')
            result = {'online_decisions': model.num_timesteps, 'online_gradient_updates': model._n_updates,
                      'initialization': progress.initialization, 'accounting': progress.accounting(),
                      'best_online_validation': progress.best_online, 'early_stop_reason': progress.early_stop_reason}
            atomic_json(artifacts/'result.json', serial(result))
            logger.artifact('artifacts/result.json', 'Initialization, ordinary SAC performance and physical costs')
            logger.finish('stopped' if stop.requested or progress.early_stop_reason else 'complete',
                          progress.early_stop_reason or 'Bounded multi-object SAC+HER grasp run complete; see physical validation.')
        except BaseException as error:
            logger.finish('failed', f'{type(error).__name__}: {error}')
            raise
        finally:
            progress.close()
            if physical is not None: physical.close()
            schema.close()
            if stop.requested and logger.state['status'] == 'running':
                logger.finish('stopped', 'Stopped after a safe manipulation training stage.')


if __name__ == '__main__':
    main()
