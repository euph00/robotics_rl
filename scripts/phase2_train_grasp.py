#!/usr/bin/env python3
"""Balanced multi-object BC initialization, critic calibration and labeled SAC+HER pilots."""
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
from pnr.regularized_sac import RegularizedSAC
from pnr.grasp_actor_ablation import CompactGraspActorPolicy
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


def action_fit(model, observations, actions, groups):
    """Original-goal teacher action agreement; not physical motion accuracy."""
    differences = []
    with torch.no_grad():
        for start in range(0, len(actions), 1024):
            batch = {key: value[start:start+1024] for key, value in observations.items()}
            predicted = model.actor(batch, deterministic=True)
            differences.append((predicted-actions[start:start+1024]).cpu().numpy())
    error = np.concatenate(differences)
    labels = actions.detach().cpu().numpy()
    result = {}
    for key, indices in groups.items():
        selected = error[indices]
        result[key] = {'examples': len(indices), 'mean_squared_action_error': float(np.mean(selected**2)),
                       'mean_absolute_error_by_action': np.mean(np.abs(selected), axis=0).tolist(),
                       'mean_nominal_cartesian_target_error_m': float(np.linalg.norm(selected[:,:3]*.012, axis=1).mean()),
                       'finger_open_close_sign_disagreement': float(np.mean(np.sign(labels[indices,3]+selected[:,3]) != np.sign(labels[indices,3])))}
    return {'scope': 'training_demonstration_action_fit', 'physical_interactions': 0, 'groups': result}


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
        self.demo_fit_data = None

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
                'evaluation_initialization_physics_ticks': self.eval_settling_ticks,
                'regularization': getattr(self.model, 'last_regularization_metrics', {})}

    def evaluate(self, label):
        records = []
        if self.demo_fit_data is not None:
            atomic_json(self.run_logger.directory/'artifacts'/f'action_fit_{label}.json',
                        action_fit(self.model, *self.demo_fit_data))
        for case in self.cases:
            if self.stop.requested: break
            key = (case['object_idx'], case['scene_seed'])
            try:
                if key not in self.evaluation:
                    env = make_grasp_case(*key, seed=71000, horizon=self.args.eval_horizon, evaluation=True,
                                          goal_lift_m=self.args.eval_lift_m,
                                          grasp_proposal_mode=self.args.grasp_proposal_mode)
                    self.evaluation[key] = env
                    self.eval_setup_calls += env.setup_action_calls
                    self.eval_setup_ticks += env.setup_action_calls*env.native.substeps
                    self.eval_settling_ticks += env.native.settle_steps
                env = self.evaluation[key]
                env.reset(seed=71000, options={'full_pregrasp': True})
                env.goal = env.base_goal.copy()
                observation = env._observation()
                fallback_before = (env.local_ik.fallback_attempts, env.local_ik.fallback_successes)
                trajectory, held = [], []
                first_contact_geometry = None
                success = False
                for decision in range(self.args.eval_horizon):
                    action, _ = self.model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    if info['bilateral_contact'] and first_contact_geometry is None:
                        from scipy.spatial.transform import Rotation
                        ee, quat = env.controller.planner.ee_pose()
                        obj = env.native.observe()['object_pos']
                        first_contact_geometry = {
                            'decision': decision+1,
                            'object_com_in_hand_frame_m': Rotation.from_quat(quat).inv().apply(obj-ee).tolist(),
                            'proposal_target_minus_ee_m': (env.measured_grasp_target()-ee).tolist()}
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
                record = dict(case, success=success, goal_lift_m=self.args.eval_lift_m,
                              grasp_proposal_mode=env.grasp_proposal_mode,
                              control_version=env.control_version, fallback_attempts=env.local_ik.fallback_attempts-fallback_before[0],
                              fallback_successes=env.local_ik.fallback_successes-fallback_before[1], goal_distance=trajectory[-1]['goal_distance'],
                              decisions=len(trajectory), source_action_calls=env.native.step_count,
                              physics_ticks=env.native.physics_steps, metrics=env.native.evaluate(),
                              height_range_final_hold_m=float(np.ptp(held)) if held else None,
                              first_bilateral_contact_geometry=first_contact_geometry, trajectory=trajectory)
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
        if len(records) != len(self.cases):
            path = self.run_logger.directory/'artifacts'/f'evaluation_{label}_interrupted.json'
            atomic_json(path, {'scope': 'multi_object_full_pregrasp', 'status': 'incomplete',
                'online_decisions': int(self.model.num_timesteps), 'records': records,
                'expected_cases': len(self.cases), 'completed_cases': len(records),
                'validation_success_rate': None, 'validation_mean_goal_distance': None,
                'accounting': self.accounting()})
            self.run_logger.artifact('artifacts/'+path.name, 'Incomplete grasp evaluation '+label)
            self.run_logger.update(evaluation_status='incomplete')
            return -float('inf'), float('inf')
        validation = [r for r in records if r['partition'] == 'validation']
        train = [r for r in records if r['partition'] == 'train']
        rate = float(np.mean([r['success'] for r in validation])) if validation else 0.
        distance = float(np.mean([r['goal_distance'] if r['goal_distance'] is not None else 1. for r in validation])) if validation else 1.
        path = self.run_logger.directory/'artifacts'/f'evaluation_{label}.json'
        atomic_json(path, {'scope': 'multi_object_full_pregrasp', 'online_decisions': int(self.model.num_timesteps),
                           'teacher_at_inference': False, 'records': records,
                           'validation_success_rate': rate, 'validation_mean_goal_distance': distance})
        self.run_logger.artifact('artifacts/'+path.name, 'Physical grasp evaluation '+label)
        self.run_logger.update(evaluation_status='complete')
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
                    'grasp_protocol_signature': getattr(self.model, '_pnr_grasp_protocol_signature', None),
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
                                   accounting=self.accounting(),
                                   regularization=getattr(self.model, 'last_regularization_metrics', {}))
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
    parser.add_argument('--bc-only', action='store_true', help='Stop after BC selection/physical check; no critic calibration or online environments')
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
    parser.add_argument('--shuffle-training-cases', action='store_true',
                        help='Seeded case order broadens physical coverage in short pilots; exact order is logged')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bc-weight', type=float, default=0., help='Positive enables explicitly labeled ongoing demonstration regularization')
    parser.add_argument('--demo-batch-size', type=int, default=400)
    parser.add_argument('--actor-checkpoint', type=Path, help='Transfer actor only; critics and all optimizers remain fresh')
    parser.add_argument('--actor-features', choices=['full70','compact28'], default='full70',
                        help='Opt-in compact relational actor; critic and world-coordinate actions stay unchanged')
    parser.add_argument('--lift-range', nargs=2, type=float, default=[.20, .20])
    parser.add_argument('--eval-lift-m', type=float, default=.20)
    parser.add_argument('--grasp-proposal-mode', choices=['legacy','clearance-v2'], default='legacy')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.steps > 40000:
        raise ValueError('Initial pilot limited to eight workers and40k online decisions')
    if args.bc_weight < 0 or not .15 <= args.lift_range[0] <= args.lift_range[1] <= .30 or not .15 <= args.eval_lift_m <= .30:
        raise ValueError('Invalid regularization weight or lift-goal range')
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.set_num_threads(1)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = pickle.loads(args.demonstrations.read_bytes())
    dataset_mode = dataset['metadata'].get('configuration', {}).get('grasp_proposal_mode', 'legacy')
    if dataset_mode != args.grasp_proposal_mode:
        raise ValueError('Dataset proposal mode differs from declared physical training/evaluation mode')
    training_cases = [{'object_idx': r['object_idx'], 'scene_seed': r['scene_seed']}
                      for r in dataset['metadata']['cases'] if r['case_status'] == 'complete']
    training_cases.sort(key=lambda case: (case['scene_seed'], case['object_idx']))
    if args.shuffle_training_cases:
        random.Random(args.seed).shuffle(training_cases)
    if not training_cases or len(training_cases) < args.workers:
        raise ValueError('Need at least one prepared training case per worker')
    if set(args.validation_seeds) & {case['scene_seed'] for case in training_cases}:
        raise ValueError('Validation point-cloud seeds overlap training')
    objects = sorted({case['object_idx'] for case in training_cases})
    evaluation_cases = [{'object_idx': index, 'scene_seed': seed, 'partition': 'validation'}
                        for index in objects for seed in args.validation_seeds]
    evaluation_cases += [{'object_idx': index, 'scene_seed': 42, 'partition': 'train'} for index in objects]
    algorithm = ('Behavior cloning initializer; zero online updates' if args.bc_only else
                 'SAC+HER+demonstration regularization' if args.bc_weight > 0 else 'BC-initialized SAC+HER')
    logger = RunLogger(args.run_id, kind='train', skill='grasp', algorithm=algorithm,
                       name='Phase 2 · multi-object SAC + HER grasp', budget=args.steps,
                       summary='Balanced object/phase actor initialization and measured full-pregrasp validation before '+algorithm+'.')
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
        'actor_checkpoint': str(args.actor_checkpoint) if args.actor_checkpoint else None,
        'actor_checkpoint_sha256': hashlib.sha256(args.actor_checkpoint.read_bytes()).hexdigest() if args.actor_checkpoint else None,
        'algorithm': algorithm, 'control_version': 'bounded_local_global_ik-v1',
        'demonstration_teacher_approach_control': dataset['metadata'].get('configuration', {}).get('teacher_approach_control', 'component'),
        'demonstrations_sha256': hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
        'training_cases': training_cases, 'evaluation_cases': evaluation_cases,
        'observation_version': 'antipodal-relative-grasp-v1', 'gamma': .99,
        'n_sampled_goal': 1, 'alpha_initial': .005, 'target_entropy': -8.,
        'supervised_loss_during_online_training': args.bc_weight > 0 and not args.bc_only, 'independent_final_study_seed': False,
        'bc_selection': 'Best validation success then goal distance at declared initialization checks.',
        'online_collapse_rule': 'Stop after two checks below40% if BC initial success at least80%; retain outcomes.'})
    protocol = {key: config[key] for key in ['observation_version','control_version','grasp_proposal_mode','actor_features',
                'lift_range','eval_lift_m','demonstration_teacher_approach_control','demonstrations_sha256']}
    protocol['semantic_sources'] = {name: source_hashes[name] for name in [
        'pnr/env.py','pnr/skills.py','pnr/general_manipulation.py','pnr/reference.py',
        'pnr/planning.py','pnr/local_control.py','pnr/clearance_grasps.py']}
    if args.actor_features == 'compact28':
        protocol['semantic_sources']['pnr/grasp_actor_ablation.py'] = source_hashes['pnr/grasp_actor_ablation.py']
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    config['grasp_protocol_signature'] = protocol
    atomic_json(logger.directory/'config.json', config)
    rows, groups, bc_metadata = dataset_rows(dataset)
    obs = {key: torch.as_tensor(np.stack([r['observation'][key] for r in rows]), device=args.device)
           for key in GraspSchema().observation_space.spaces}
    actions = torch.as_tensor(np.stack([r['action'] for r in rows]), device=args.device)
    schema = DummyVecEnv([lambda: GraspSchema() for _ in range(args.workers)])
    model_class = RegularizedSAC if args.bc_weight > 0 else SAC
    regularization = {'bc_weight': args.bc_weight, 'demo_batch_size': args.demo_batch_size,
                      'gradient_diagnostics_interval': 25} if args.bc_weight > 0 else {}
    policy_class = CompactGraspActorPolicy if args.actor_features == 'compact28' else 'MultiInputPolicy'
    model = model_class(policy_class, schema, **regularization, learning_rate=args.learning_rate, buffer_size=150000,
                learning_starts=0, batch_size=args.batch_size, tau=.01, gamma=.99,
                train_freq=1, gradient_steps=1, ent_coef='auto_0.005', target_entropy=-8.,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs={'n_sampled_goal': 1, 'goal_selection_strategy': 'future', 'copy_info_dict': True},
                policy_kwargs={'net_arch': [128, 128], 'features_extractor_class': GeneralGraspFeatures},
                device=args.device, seed=args.seed)
    model._pnr_grasp_protocol_signature = protocol
    transfer_reference = transfer_observations = transfer_verification = None
    if args.actor_checkpoint:
        source_model = SAC.load(args.actor_checkpoint, env=schema, device=args.device)
        if source_model.actor.features_extractor.features_dim != model.actor.features_extractor.features_dim:
            raise ValueError('Actor transfer cannot cross feature contracts; start a fresh actor for the ablation')
        if type(source_model.actor.features_extractor) is not type(model.actor.features_extractor):
            raise ValueError('Actor transfer feature classes differ')
        model.actor.load_state_dict(source_model.actor.state_dict())
        tensor_equal = all(torch.equal(value, model.actor.state_dict()[key])
                           for key,value in source_model.actor.state_dict().items())
        indices = np.linspace(0, len(actions)-1, min(1024,len(actions)), dtype=int)
        transfer_observations = {key:value[indices] for key,value in obs.items()}
        with torch.no_grad():
            transfer_reference = source_model.actor(transfer_observations, deterministic=True).clone()
            action_equal = torch.equal(transfer_reference, model.actor(transfer_observations, deterministic=True))
        if not tensor_equal or not action_equal:
            raise AssertionError('Actor-only transfer did not preserve its state and deterministic TRAIN actions')
        transfer_verification = {'source':str(args.actor_checkpoint),
            'source_sha256':config['actor_checkpoint_sha256'], 'all_actor_state_tensors_identical_after_copy':True,
            'deterministic_actions_identical_after_copy':True,'training_probe_examples':len(indices),
            'validation_examples_used':0,'actor_features':args.actor_features}
        model._pnr_actor_transfer_verification = transfer_verification
        atomic_json(artifacts/'actor_transfer_verification.json', transfer_verification)
        del source_model
    if args.bc_weight > 0:
        demo_manifest = model.set_demonstrations(obs, actions, groups, seed=args.seed)
        atomic_json(artifacts/'regularization_demonstrations.json', demo_manifest)
    model.set_logger(configure(str(logger.directory/'sb3'), ['csv']))
    progress = Progress(logger, evaluation_cases, args, None); progress.model = model
    progress.demo_fit_data = (obs, actions, groups)
    physical = None
    with GracefulStop() as stop:
        progress.stop = stop
        try:
            optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
            rng = np.random.default_rng(args.seed)
            losses = []
            best = (-1., -1e9); best_state = None; best_update = None
            started = time.monotonic()
            if args.actor_checkpoint:
                rate, distance = progress.evaluate('transferred_actor_zero_refinement')
                best = (rate, -distance)
                best_state = copy.deepcopy(model.actor.state_dict()); best_update = 0
                model.save(checkpoints/'transferred_actor')
            update = 0
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
                'supervised_examples': update*args.batch_size,
                'actor_transfer_source': str(args.actor_checkpoint) if args.actor_checkpoint else None,
                'wall_seconds': time.monotonic()-started}
            if args.bc_only:
                model.save(checkpoints/'bc_initialized')
                record = {'stage': 'bc_only', 'online_decisions': 0, 'online_gradient_updates': 0,
                    'critic_initialization_updates': 0, 'initialization': progress.initialization,
                    'accounting': progress.accounting(), 'algorithm': algorithm,
                    'actor_features': args.actor_features, 'critic_features': 70}
                atomic_json(artifacts/'result.json', serial(record))
                atomic_json(checkpoints/'latest.json', serial({'model': 'bc_initialized.zip',
                    'initialization': progress.initialization, 'accounting': progress.accounting(),
                    'grasp_protocol_signature': protocol, 'stage': 'bc_only',
                    'replay_collected': False, 'critic_calibration_completed': False}))
                logger.artifact('artifacts/result.json', 'BC-only actor and fixed physical evaluation; zero online updates')
                logger.finish(summary='Fresh '+args.actor_features+' BC initializer physically checked; no critic calibration or online training.')
                return
            progress.initialization['std_calibration'] = calibrate_actor_std(model, -3.)
            if transfer_verification is not None and args.bc_updates == 0:
                with torch.no_grad():
                    unchanged = torch.equal(transfer_reference, model.actor(transfer_observations, deterministic=True))
                if not unchanged: raise AssertionError('Std calibration changed deterministic actor actions')
                transfer_verification['deterministic_actions_identical_after_std_calibration'] = True
                atomic_json(artifacts/'actor_transfer_verification.json', transfer_verification)
            model.save(checkpoints/'bc_initialized')
            logger.update(summary='Importing all measured demonstrations and calibrating fresh critics with actor fixed.')
            replay_metadata = seed_replay(model, args.demonstrations)
            progress.initialization['replay'] = {key: value for key, value in replay_metadata.items() if key != 'dataset_metadata'}
            atomic_json(artifacts/'demonstration_accounting.json', dataset['metadata'])
            logger.artifact('artifacts/demonstration_accounting.json', 'All demonstration attempts and collection costs')
            progress.initialization['critic_warmup'] = critic_only_warmup(model, args.warmup_updates,
                args.batch_size, learning_rate=3e-4, log_every=200,
                on_log=lambda row: logger.metric(0, scope='actor_initialization', metric_source='critic_initialization', **row))
            if transfer_verification is not None and args.bc_updates == 0:
                with torch.no_grad():
                    unchanged = torch.equal(transfer_reference, model.actor(transfer_observations, deterministic=True))
                if not unchanged: raise AssertionError('Critic calibration changed deterministic actor actions')
                transfer_verification['deterministic_actions_identical_after_critic_calibration'] = True
                atomic_json(artifacts/'actor_transfer_verification.json', transfer_verification)
                logger.artifact('artifacts/actor_transfer_verification.json', 'BC actor preserved before online updates')
            progress.evaluate('initialized_zero_online')
            progress.checkpoint('initialized')
            if stop.requested: return
            parent_pid = os.getpid()
            def make(index):
                ignore_worker_stop_signals(parent_pid)
                return MultiObjectGrasp(training_cases[index::args.workers], seed=args.seed+index,
                                         horizon=args.horizon, cache_size=4, episodes_per_case=4, lift_range=args.lift_range,
                                         grasp_proposal_mode=args.grasp_proposal_mode)
            physical = SubprocVecEnv([lambda index=index: make(index) for index in range(args.workers)], start_method='spawn')
            model.set_env(physical)
            model.replay_buffer.env = None; model.replay_buffer.set_env(physical)
            logger.update(summary=algorithm+' on physical multi-object starts; original-goal demonstrations constrain the actor.' if args.bc_weight > 0
                          else 'Ordinary online SAC + HER over physical multi-object starts.')
            model.learn(total_timesteps=args.steps, callback=progress)
            progress.evaluate(f'final_{model.num_timesteps:07d}')
            progress.checkpoint('final')
            result = {'online_decisions': model.num_timesteps, 'online_gradient_updates': model._n_updates,
                      'initialization': progress.initialization, 'accounting': progress.accounting(),
                      'best_online_validation': progress.best_online, 'early_stop_reason': progress.early_stop_reason,
                      'algorithm': algorithm, 'regularization': getattr(model, 'last_regularization_metrics', {})}
            atomic_json(artifacts/'result.json', serial(result))
            logger.artifact('artifacts/result.json', 'Initialization, '+algorithm+' performance and physical costs')
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
