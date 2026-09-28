#!/usr/bin/env python3
"""Collect actual mixed-policy transitions and separate geometric teacher labels.

The teacher is training-only privileged assistance. Its action is a supervised
label; the action passed to physics and HER is the recorded 50/50 blend.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from pnr.policies import load_policy_checkpoint
from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json
from pnr.train import GracefulStop
from scripts.generate_grasp_demos import GeometricContactTeacher


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_dataset(path, episodes, metadata):
    """Atomic trusted-local pickle plus plain NPZ; never relabel action fields."""
    dataset = {'schema_version':1, 'kind':'physical_mixed_policy_corrections',
               'training_only':True, 'teacher_version':'geometric-contact-v1',
               'metadata':metadata, 'episodes':episodes}
    temporary = path.with_suffix('.pending')
    with temporary.open('wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    transitions = [row for episode in episodes for row in episode['transitions']]
    if transitions:
        arrays = {f'{prefix}_{key}':np.stack([row[prefix][key] for row in transitions])
                  for prefix in ('observation','next_observation') for key in transitions[0]['observation']}
        arrays.update(actions=np.stack([row['action'] for row in transitions]),
            teacher_actions=np.stack([row['teacher_action'] for row in transitions]),
            learner_actions=np.stack([row['learner_action'] for row in transitions]),
            teacher_phases=np.array([row['teacher_phase'] for row in transitions]),
            rewards=np.array([row['reward'] for row in transitions],dtype=np.float32),
            terminated=np.array([row['terminated'] for row in transitions],dtype=bool),
            truncated=np.array([row['truncated'] for row in transitions],dtype=bool),
            episode=np.array([row['episode'] for row in transitions],dtype=np.int32),
            step=np.array([row['step'] for row in transitions],dtype=np.int32),
            info_json=np.array([json.dumps(row['info']) for row in transitions]))
        array_pending = path.with_suffix('.pending.npz')
        np.savez_compressed(array_pending, **arrays)
        array_pending.replace(path.with_suffix('.npz'))
    return digest(path)


def validate_dataset(env, episodes):
    count = 0
    for episode in episodes:
        rows = episode['transitions']
        if not rows or not (rows[-1]['terminated'] or rows[-1]['truncated']):
            raise AssertionError('Every collected episode must have a real ending')
        for index,row in enumerate(rows):
            expected = (.5*row['teacher_action']+.5*row['learner_action']).astype(np.float32)
            if not np.array_equal(expected,row['action']):
                raise AssertionError('HER action differs from the executed blend')
            for field in ('action','teacher_action','learner_action'):
                if row[field].shape != (4,) or not np.isfinite(row[field]).all() or np.max(np.abs(row[field]))>1.:
                    raise AssertionError('Malformed action field')
            for obs_name in ('observation','next_observation'):
                if not env.observation_space.contains(row[obs_name]):
                    raise AssertionError('Invalid recorded observation')
            if index and any(not np.array_equal(rows[index-1]['next_observation'][key],row['observation'][key]) for key in row['observation']):
                raise AssertionError('Discontinuous physical trajectory')
            if index<len(rows)-1 and (row['terminated'] or row['truncated']):
                raise AssertionError('Trajectory continues after its ending')
            if row['info']['TimeLimit.truncated'] != (row['truncated'] and not row['terminated']):
                raise AssertionError('Collector/environment timeout is not bootstrap-safe')
            count += 1
        recomputed = env.compute_reward(np.stack([row['next_observation']['achieved_goal'] for row in rows]),
            np.stack([row['observation']['desired_goal'] for row in rows]),[row['info'] for row in rows])
        if not np.array_equal(recomputed,np.asarray([row['reward'] for row in rows],dtype=np.float32)):
            raise AssertionError('Recorded rewards differ from the actual HER contract')
    return {'validated_transitions':count, 'exact_blended_actions':True,
            'teacher_labels_separate_from_her_actions':True, 'contiguous_actual_observations':True,
            'recomputed_rewards_match':True, 'all_episodes_end_explicitly':True,
            'padding_transitions':0, 'failed_episodes_discarded':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='grasp_correction_demos_v1')
    parser.add_argument('--bc-run',default='grasp_bc_init01')
    parser.add_argument('--seed',type=int,default=52000)
    parser.add_argument('--timeout-seconds',type=float,default=600.)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 900:
        parser.error('Collector wall budget must be between1 and900 seconds')
    torch.set_num_threads(1)
    source = ROOT/'runs'/args.bc_run
    source_model = source/'checkpoints/best/model.zip'
    if not (source/'artifacts/result.json').is_file():
        raise ValueError('Freeze a completed BC initialization run before collecting corrections')
    model_hash = digest(source_model)
    logger = RunLogger(args.run_id,kind='validation',skill='grasp',algorithm='Mixed-policy correction collection',
        name='Physical grasp corrections',budget=1600,
        summary='16 training-only episodes; actual actions are half teacher, half frozen learned policy.')
    artifacts=logger.directory/'artifacts';inputs=artifacts/'inputs';inputs.mkdir(parents=True)
    shutil.copy2(source_model,inputs/'learner.zip')
    for origin,name in ((source/'config.json','bc_config.json'),(source/'artifacts/result.json','bc_result.json'),
                        (source/'checkpoints/best/evaluation.json','bc_selected_evaluation.json')):
        shutil.copy2(origin,inputs/name)
    if digest(inputs/'learner.zip') != model_hash:
        raise ValueError('Learner checkpoint changed while freezing input')
    source_files=('scripts/collect_grasp_corrections.py','scripts/generate_grasp_demos.py',
                  'pnr/skills.py','pnr/env.py','pnr/local_control.py','pnr/reference.py',
                  'pnr/planning.py','pnr/features.py','pnr/policies.py')
    frozen_sources=artifacts/'source';frozen_sources.mkdir()
    source_hashes={}
    for name in source_files:
        content=(ROOT/name).read_bytes();source_hashes[name]=hashlib.sha256(content).hexdigest()
        (frozen_sources/name.replace('/','__')).write_bytes(content)
    metadata={'collection_status':'running','metadata_finalized':False,'skill':'grasp',
        'scene_seed':42,'object_idx':14,'horizon':100,'planned_episodes':16,
        'start_heights_m':[.008,.04,.09,.17],'episodes_per_height':4,'goal_lift_m':.20,
        'goal_xy_jitter_m':.015,'collection_seed':args.seed,'teacher_weight':.5,'learner_weight':.5,
        'learner_deterministic':True,'learner_source_run':args.bc_run,'learner_sha256':model_hash,
        'source_hashes':source_hashes,'source_snapshot_directory':'source',
        'policy_decisions':0,'demonstration_source_action_calls':0,'demonstration_physics_ticks':0,
        'setup_source_action_calls':0,'setup_physics_ticks':0,'initialization_physics_ticks':0,
        'collector_timeouts':0,'training_only':True,'teacher_at_learned_inference':False,
        'action_semantics':'Actual executed .5*teacher_action+.5*learner_action; use action for HER.',
        'supervised_target_semantics':'Use separate teacher_action with the original observation desired goal for BC.',
        'privileged_teacher_inputs':['geometric grasp proposal','current object pose','physical finger contact'],
        'new_actor_or_critic_updates':0,'object_attachment':False}
    atomic_json(logger.directory/'config.json',vars(args)|metadata)
    atomic_json(artifacts/'source_hashes.json',source_hashes)
    env=None;episodes=[];started=time.perf_counter()
    dataset_path=artifacts/'corrections.pkl'
    with GracefulStop() as stop:
        try:
            model=load_policy_checkpoint(inputs/'learner.zip',device='cpu')
            env=ManipulationSkillEnv(skill='grasp',scene_seed=42,object_idx=14,seed=args.seed,horizon=100,evaluation=True)
            metadata.update(setup_source_action_calls=env.setup_action_calls,
                setup_physics_ticks=env.setup_action_calls*env.native.substeps,
                initialization_physics_ticks=env.native.settle_steps,
                scene=env.native.scene_manifest(),curriculum_protocol=env.curriculum_protocol)
            teacher=GeometricContactTeacher(env)
            for episode_index in range(16):
                if stop.requested or time.perf_counter()-started>=args.timeout_seconds:
                    break
                env.reset(seed=args.seed+episode_index)
                height,snapshot=env.snapshots[episode_index%4]
                env.native.restore_state(snapshot,reset_counters=True)
                observation=env._observation()
                initial_object=env.native.observe()['object_pos'].copy()
                transitions=[];first_goal=first_stable=None;max_lift=0.
                for step in range(100):
                    teacher_action,phase=teacher.action(observation)
                    learner_action=np.asarray(model.predict(observation,deterministic=True)[0],dtype=np.float32)
                    action=(.5*teacher_action+.5*learner_action).astype(np.float32)
                    following,reward,terminated,truncated,info=env.step(action)
                    elapsed=time.perf_counter()-started
                    collector_timeout=bool(not terminated and not truncated and (stop.requested or elapsed>=args.timeout_seconds))
                    info=copy.deepcopy(info)
                    if collector_timeout:
                        truncated=True;info['collector_truncated']=True
                        info['collector_truncation_reason']=stop.signal_name if stop.requested else 'wall_time_budget'
                        metadata['collector_timeouts']+=1
                    info['TimeLimit.truncated']=bool(truncated and not terminated)
                    transitions.append({'episode':episode_index,'step':step,
                        'observation':copy.deepcopy(observation),'action':action.copy(),
                        'teacher_action':teacher_action.copy(),'learner_action':learner_action.copy(),
                        'teacher_phase':phase,'teacher_weight':.5,
                        'next_observation':copy.deepcopy(following),'reward':float(reward),
                        'terminated':bool(terminated),'truncated':bool(truncated),'info':info})
                    max_lift=max(max_lift,float(env.native.observe()['object_pos'][2]-initial_object[2]))
                    if info['is_success'] and first_goal is None:first_goal=step+1
                    if info['ever_stably_held'] and first_stable is None:first_stable=step+1
                    observation=following
                    if terminated or truncated:break
                final=transitions[-1]['info']
                success=bool(first_goal is not None and first_stable is not None and final['bilateral_contact']
                    and final['goal_distance']<env.tolerance and max_lift>=.16 and not transitions[-1]['terminated'])
                summary={'episode':episode_index,'seed':args.seed+episode_index,'start_height_m':float(height),
                    'goal_lift_m':.20,'desired_goal':env.goal.tolist(),'initial_object_position':initial_object.tolist(),
                    'decisions':len(transitions),'successful_physical_lift_and_hold':success,
                    'first_goal_decision':first_goal,'first_stable_hold_decision':first_stable,
                    'maximum_lift_m':max_lift,'final_goal_distance_m':float(final['goal_distance']),
                    'final_bilateral_contact':bool(final['bilateral_contact']),
                    'source_action_calls':sum(row['info']['action_calls_delta'] for row in transitions),
                    'physics_ticks':sum(row['info']['physics_ticks_delta'] for row in transitions),
                    'physical_failure':transitions[-1]['terminated'],
                    'collector_timeout':bool(final.get('collector_truncated',False)),
                    'bilateral_contact_fraction':float(np.mean([row['info']['bilateral_contact'] for row in transitions])),
                    'action_rejection_fraction':float(np.mean([row['info']['action_rejected'] for row in transitions])),
                    'learner_teacher_action_mse':float(np.mean([(row['learner_action']-row['teacher_action'])**2 for row in transitions]))}
                episodes.append({'summary':summary,'transitions':transitions})
                metadata.update(policy_decisions=sum(ep['summary']['decisions'] for ep in episodes),
                    demonstration_source_action_calls=sum(ep['summary']['source_action_calls'] for ep in episodes),
                    demonstration_physics_ticks=sum(ep['summary']['physics_ticks'] for ep in episodes),
                    wall_seconds=time.perf_counter()-started)
                save_dataset(dataset_path,episodes,metadata)
                logger.metric(metadata['policy_decisions'],scope='training_data',metric_source='correction_collection',
                    episodes=len(episodes),success_rate=float(np.mean([ep['summary']['successful_physical_lift_and_hold'] for ep in episodes])),
                    source_action_calls=metadata['demonstration_source_action_calls'],physics_ticks=metadata['demonstration_physics_ticks'],
                    setup_source_action_calls=metadata['setup_source_action_calls'],setup_physics_ticks=metadata['setup_physics_ticks'])
                print(json.dumps(summary),flush=True)
            validation=validate_dataset(env,episodes)
            if digest(source_model)!=model_hash or digest(inputs/'learner.zip')!=model_hash:
                raise RuntimeError('Frozen learner input changed during collection')
            if any(digest(ROOT/name)!=expected for name,expected in source_hashes.items()):
                raise RuntimeError('Source changed while collecting; preserve data and inspect provenance')
            metadata.update(collection_status='complete' if len(episodes)==16 else 'stopped',metadata_finalized=True,
                wall_seconds=time.perf_counter()-started,episodes_collected=len(episodes),
                total_source_action_calls=metadata['setup_source_action_calls']+metadata['demonstration_source_action_calls'],
                total_physics_ticks=metadata['setup_physics_ticks']+metadata['demonstration_physics_ticks']+metadata['initialization_physics_ticks'])
            dataset_hash=save_dataset(dataset_path,episodes,metadata)
            report={**metadata,'dataset_sha256':dataset_hash,'pickle_file':'corrections.pkl','array_file':'corrections.npz',
                'episodes':[ep['summary'] for ep in episodes],'validation':validation,
                'limitation':'Mixed-policy training collection; successes are not learned-policy evaluation results.'}
            atomic_json(artifacts/'manifest.json',report)
            atomic_json(artifacts/'validation.json',validation)
            logger.artifact('artifacts/corrections.pkl','Trusted local actual transitions and separate teacher labels','file')
            logger.artifact('artifacts/corrections.npz','Actual, teacher and learner action arrays','file')
            logger.artifact('artifacts/manifest.json','All outcomes, provenance and physical interaction costs','file')
            logger.finish('complete' if len(episodes)==16 else 'stopped',
                f"Collected {len(episodes)}/16 actual mixed-policy episodes; {sum(ep['summary']['successful_physical_lift_and_hold'] for ep in episodes)} retained-lift outcomes. No actor training.")
            print(json.dumps({'dataset_sha256':dataset_hash,'metadata':metadata,'validation':validation}),flush=True)
        except BaseException as error:
            metadata.update(collection_status='failed',error=f'{type(error).__name__}: {error}',wall_seconds=time.perf_counter()-started)
            if episodes:save_dataset(dataset_path,episodes,metadata)
            atomic_json(artifacts/'manifest.json',{**metadata,'episodes':[ep['summary'] for ep in episodes]})
            logger.finish('failed',metadata['error']);raise
        finally:
            if env is not None:env.close()


if __name__=='__main__':main()
