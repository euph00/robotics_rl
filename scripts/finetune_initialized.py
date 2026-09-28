#!/usr/bin/env python3
"""Bounded ordinary SAC+HER continuation after a disclosed actor BC stage.

This is a development transfer, not an independent experimental replication.
The actor receives no further supervised loss. Real replay and accounting are
inherited from the stated SAC parent. Demonstrations are not counted twice.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from pnr.skills import ManipulationSkillEnv
from pnr.carry import CarrySkillEnv
from pnr.telemetry import RunLogger, atomic_json
from pnr.replay_transfer import append_demonstrations
from pnr.train import (ProgressCallback, GracefulStop, ignore_worker_stop_signals,
                       make_resume_signature, validate_replay_layout)


class FullPregraspEvaluation(ManipulationSkillEnv):
    """One fixed development case matching the original planned handover."""
    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options={'full_pregrasp':True})
        self.goal = self.base_goal.copy()
        self.evaluation_scope = 'full_pregrasp_diagnostic'
        return self._observation(), info



class InitializedProgressCallback(ProgressCallback):
    """Keep BC validation costs in durable totals, with a separate breakdown."""
    def add_initialization(self, record):
        if getattr(self, 'bc_initialization', None) is not None:
            raise ValueError('BC initialization accounting must not be added twice')
        costs = record['incremental_accounting']
        self.evaluation_actions += costs['new_evaluation_source_action_calls']
        self.evaluation_ticks += costs['new_evaluation_physics_ticks']
        self.evaluation_setup_actions += costs['new_setup_source_action_calls']
        self.evaluation_setup_ticks += costs['new_setup_physics_ticks']
        self.evaluation_initialization_ticks += costs['new_initialization_physics_ticks']
        preparation = record.get('collector_policy_preparation')
        if preparation is not None:
            self.evaluation_actions += preparation['evaluation_source_action_calls']
            self.evaluation_ticks += preparation['evaluation_physics_ticks']
            self.evaluation_setup_actions += preparation['setup_source_action_calls']
            self.evaluation_setup_ticks += preparation['setup_physics_ticks']
            self.evaluation_initialization_ticks += preparation['initialization_physics_ticks']
        if record.get('new_demonstration_report') is not None:
            report = record['new_demonstration_report']
            # Training-data teacher validation is a physical evaluation cost,
            # distinct from the actual demonstrations imported into replay.
            metadata = report['dataset_metadata']
            self.evaluation_actions += metadata.get('preflight_source_action_calls', 0)
            self.evaluation_ticks += metadata.get('preflight_physics_ticks', 0)
            self.add_demonstrations(report)
        self.bc_initialization = record
        self.accounting_missing = list(dict.fromkeys(self.accounting_missing + record['legacy_missing']))

    def accounting(self):
        result = super().accounting()
        if getattr(self, 'bc_initialization', None) is not None:
            # These are a breakdown of the evaluation/setup totals above, not
            # additional demonstration collection or separately summed costs.
            result['bc_initialization'] = self.bc_initialization
        return result


def initialization_record(bc, bc_config, source_counts, model_digest):
    """Require finalized BC costs before publishing a continuing experiment."""
    result_path = bc/'artifacts/result.json'
    if not result_path.is_file():
        raise ValueError('BC run must finalize artifacts/result.json before continuation')
    result = json.loads(result_path.read_text())
    if (result['source_run'] != bc_config['source_run'] or
            result['source_generation'] != bc_config['source_generation'] or
            result['source_model_sha256'] != bc_config['source_model_sha256'] or
            result['source_online_transitions'] != source_counts['transitions'] or
            result.get('new_online_transitions') != 0):
        raise ValueError('BC result does not match the immutable SAC parent')
    if source_counts.get('accounting', {}).get('bc_initialization'):
        raise ValueError('This bounded script supports one BC transfer, not repeated initialization')
    costs = result['accounting']
    keys = ('new_setup_source_action_calls', 'new_setup_physics_ticks',
            'new_evaluation_source_action_calls', 'new_evaluation_physics_ticks',
            'new_initialization_physics_ticks')
    if any(key not in costs or not isinstance(costs[key], int) or costs[key] < 0 for key in keys):
        raise ValueError('BC initialization requires explicit nonnegative integer physical counters')
    preparation = costs.get('collector_policy_preparation')
    if preparation is not None:
        required = ('setup_source_action_calls','setup_physics_ticks',
                    'evaluation_source_action_calls','evaluation_physics_ticks','initialization_physics_ticks')
        if any(not isinstance(preparation.get(key),int) or preparation[key]<0 for key in required):
            raise ValueError('Collector policy preparation needs complete nonnegative physical counters')
    metadata = costs.get('demonstration_collection_metadata', {})
    for key in ('preflight_source_action_calls', 'preflight_physics_ticks'):
        if not isinstance(metadata.get(key, 0), int) or metadata.get(key, 0) < 0:
            raise ValueError('Demonstration preflight costs must be nonnegative integer counters')
    selected = json.loads((bc/'checkpoints/best/evaluation.json').read_text())
    demonstration_report = None
    if costs.get('demonstration_generation_cost_newly_attributed', False):
        if source_counts.get('demonstrations') is not None:
            raise ValueError('Multiple demonstration datasets need an explicit merged accounting protocol')
        demonstration_report = {'method':'physical_demonstrations_for_actor_initialization',
            'dataset_metadata':costs['demonstration_collection_metadata'],
            'sha256':result['demonstrations_sha256'], 'transitions_imported':0}
    return {'collector_policy_preparation':preparation,
        'new_demonstration_report':demonstration_report, 'run':bc.name, 'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'selected_model_sha256':model_digest, 'selected_bc_updates':selected['bc_updates'],
        'total_bc_updates':result['bc_updates'], 'supervised_batch_size':bc_config['batch_size'],
        'supervised_examples_processed':result['bc_updates']*bc_config['batch_size'],
        'wall_seconds':result['wall_seconds'],
        'incremental_accounting':{key:costs[key] for key in keys},
        'legacy_missing':list(costs.get('legacy_missing', [])),
        'included_in_cumulative_evaluation_totals':True,
        'demonstrations_recounted':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--bc-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--steps', type=int, default=4000, help='Additional online decisions')
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--append-demonstrations', type=Path,
                        help='Append the BC physical dataset to existing HER replay, preserving executed actions')
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0 or args.eval_every < 1:
        parser.error('Positive budgets and learning rate required')
    source = ROOT/'runs'/args.source_run
    bc = ROOT/'runs'/args.bc_run
    source_config = json.loads((source/'config.json').read_text())
    bc_config = json.loads((bc/'config.json').read_text())
    # The caller names the explicit immutable parent through the BC record.
    source_generation = bc_config['source_generation']
    if bc_config['source_run'] != args.source_run:
        raise ValueError('BC and online continuation must share the exact SAC parent')
    generation = source/'checkpoints'/source_generation
    counts = json.loads((generation/'progress.json').read_text())
    initialized_model = bc/'checkpoints/best/model.zip'
    if not initialized_model.is_file():
        raise ValueError('BC initialization has not published a checkpoint')
    if hashlib.sha256((generation/'model.zip').read_bytes()).hexdigest() != bc_config['source_model_sha256']:
        raise ValueError('Immutable SAC parent model hash differs from the BC record')
    if len(counts['training_states']) != source_config['num_envs']:
        raise ValueError('SAC parent worker-state count does not match its configuration')
    model_digest = hashlib.sha256(initialized_model.read_bytes()).hexdigest()
    bc_record = initialization_record(bc, bc_config, counts, model_digest)
    demo_path = args.append_demonstrations.resolve() if args.append_demonstrations else None
    if demo_path and hashlib.sha256(demo_path.read_bytes()).hexdigest() != bc_config['demonstrations_sha256']:
        raise ValueError('Replay append must use the same immutable BC dataset')
    if demo_path and counts.get('demonstrations') is not None:
        raise ValueError('Dataset already present in the parent; repeated replay weighting is not this protocol')
    skill = source_config['skill']
    config = dict(source_config, run_id=args.run_id, device=args.device, resume=False,
        steps=counts['transitions']+args.steps, learning_rate=args.learning_rate,
        source_run=args.source_run, source_generation=source_generation, bc_run=args.bc_run,
        bc_model_sha256=model_digest, bc_initialization=bc_record,
        method='BC-initialized SAC+HER', eval_every=args.eval_every, eval_episodes=1,
        independent_seed=False, supervised_loss_during_online_training=False,
        actor_optimizer='reset after BC', online_learning_rate=args.learning_rate,
        appended_demonstrations=None if demo_path is None else str(demo_path))
    logger = RunLogger(args.run_id, name='BC-initialized SAC + HER · '+skill,
        kind='train', algorithm='BC-initialized SAC+HER', skill=skill, seed=config['seed'],
        budget=config['steps'], status='running',
        summary='Preparing ordinary SAC+HER continuation; BC-only behavior is measured first.')
    directory = logger.directory
    archive = directory/'invocations'/str(time.time_ns())
    archive.mkdir(parents=True)
    manifest = {}
    for path in list((ROOT/'pnr').glob('*.py'))+[Path(__file__).resolve()]:
        relative = path.relative_to(ROOT)
        target = archive/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(archive/'source_manifest.json', manifest)
    atomic_json(archive/'config.json', config)
    atomic_json(directory/'config.json', config)
    inputs = directory/'artifacts/inputs'; inputs.mkdir(parents=True)
    shutil.copy2(initialized_model, inputs/'bc_model.zip')
    shutil.copy2(bc/'config.json', inputs/'bc_config.json')
    shutil.copy2(bc/'artifacts/result.json', inputs/'bc_result.json')
    shutil.copy2(bc/'checkpoints/best/evaluation.json', inputs/'bc_selected_evaluation.json')
    if hashlib.sha256((inputs/'bc_model.zip').read_bytes()).hexdigest() != model_digest:
        raise ValueError('BC selected model changed while freezing continuation inputs')
    atomic_json(inputs/'source_progress.json', counts)
    torch.set_num_threads(1)
    np.random.seed(config['seed']); random.seed(config['seed']); torch.manual_seed(config['seed'])
    env = evaluation = None
    parent_pid = os.getpid()
    with GracefulStop() as stop:
        try:
            def make(index):
                ignore_worker_stop_signals(parent_pid)
                common = dict(scene_seed=config['scene_seed'], object_idx=config['object_idx'],
                    seed=config['seed']+index, horizon=config['horizon'])
                if skill == 'grasp':
                    instance = ManipulationSkillEnv(skill='grasp', **common)
                elif skill == 'carry':
                    protocol = config['transfer_protocol']
                    instance = CarrySkillEnv(**common, failure_cost=protocol['failure_cost'],
                        curriculum_window=protocol['curriculum_window'],
                        adaptive_horizon=protocol['adaptive_horizon'])
                else:
                    raise ValueError('This development transfer supports grasp or carry')
                instance.setup_physics_ticks = instance.setup_action_calls*instance.native.substeps
                instance.initialization_physics_ticks = instance.native.settle_steps
                return instance
            env = SubprocVecEnv([lambda index=i:make(index) for i in range(config['num_envs'])], start_method='spawn')
            common = dict(scene_seed=config['scene_seed'], object_idx=config['object_idx'],
                          seed=10000, horizon=config['horizon'])
            evaluation = (FullPregraspEvaluation(skill='grasp', evaluation=True, **common)
                if skill == 'grasp' else CarrySkillEnv(evaluation='full', **common,
                    failure_cost=config['transfer_protocol']['failure_cost']))
            model = SAC.load(inputs/'bc_model.zip', env=env, device=args.device,
                             learning_rate=args.learning_rate)
            if model.replay_buffer_class is not HerReplayBuffer:
                raise ValueError('BC checkpoint must retain normal SAC/HER save semantics')
            model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate)
            model.load_replay_buffer(generation/'replay.pkl', truncate_last_traj=True)
            validate_replay_layout(model.replay_buffer, config['num_envs'], skill)
            if model.num_timesteps != counts['transitions'] or model._n_updates != counts['updates']:
                raise ValueError('BC model must preserve source online/update counters')
            append_report = append_demonstrations(model, demo_path) if demo_path else None
            if append_report:
                atomic_json(inputs/'replay_append.json', append_report)
            for i, state in enumerate(counts['training_states']):
                env.env_method('set_training_state', state, indices=[i])
            callback = InitializedProgressCallback(logger, evaluation, args.eval_every, 1,
                checkpoint_every=1000, stop_request=stop)
            callback.restore_progress(counts, json.loads((source/'status.json').read_text()))
            callback.add_initialization(bc_record)
            if append_report:
                if callback.demonstration_report is None:
                    callback.add_demonstrations(append_report)
                else:
                    if callback.demonstration_report['sha256'] != append_report['sha256']:
                        raise ValueError('BC and replay demonstration provenance differ')
                    callback.demonstration_report = append_report
            callback.resume_signature = make_resume_signature(config, ROOT)
            callback.resume_signature['sources']['scripts/finetune_initialized.py'] = manifest['scripts/finetune_initialized.py']
            for relative in ('pnr/geometry_features.py','pnr/carry_features.py','pnr/carry_periodic_features.py','pnr/replay_transfer.py'):
                if relative in manifest:
                    callback.resume_signature['sources'][relative] = manifest[relative]
            callback.invocation_name = archive.name
            callback.training_setup_actions += sum(env.get_attr('setup_action_calls'))
            callback.training_setup_ticks += sum(env.get_attr('setup_physics_ticks'))
            callback.evaluation_setup_actions += evaluation.setup_action_calls
            callback.evaluation_setup_ticks += evaluation.setup_action_calls*evaluation.native.substeps
            callback.training_initialization_ticks += sum(env.get_attr('initialization_physics_ticks'))
            callback.evaluation_initialization_ticks += evaluation.native.settle_steps
            callback.model = model; callback.num_timesteps = model.num_timesteps
            callback.last_eval = callback.last_checkpoint = callback.last_log = model.num_timesteps
            callback.segment_start_transitions = model.num_timesteps
            logger.update(inherited_transitions=counts['transitions'], accounting=callback.accounting(),
                          summary='Ordinary SAC+HER after actor initialization; no supervised online loss.')
            # Publish a complete starting model/replay/cost record before any
            # potentially interrupted evaluation or first online rollout.
            callback.checkpoint()
            if not stop.requested and callback.evaluate():
                zero = directory/'artifacts/bc_zero_online'
                os.replace(directory/'checkpoints/best', zero)
                logger.artifact('artifacts/bc_zero_online/evaluation.json',
                    'BC initialization before additional online learning', 'file')
                # The online-best directory now starts empty. An interruption
                # cannot leave a BC-only checkpoint labeled best-online.
                callback.best_score = (-float('inf'), -float('inf'))
            if not stop.requested:
                model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=False)
            if not stop.requested:
                callback.evaluate()
            callback.checkpoint()
            logger.metric(model.num_timesteps, scope='skill', metric_source='training_end',
                source_action_calls=callback.native_actions, physics_ticks=callback.physics_ticks,
                updates=model._n_updates, accounting=callback.accounting())
            logger.finish('stopped' if stop.requested else 'complete',
                'Stopped safely with checkpoint.' if stop.requested else 'Bounded ordinary SAC+HER continuation complete.')
        except KeyboardInterrupt:
            logger.finish('stopped', 'Unexpected interruption; retained the last published complete checkpoint.')
            raise
        except BaseException as error:
            logger.finish('failed', f'{type(error).__name__}: {error}')
            raise
        finally:
            if env is not None: env.close()
            if evaluation is not None: evaluation.close()


if __name__ == '__main__':
    main()
