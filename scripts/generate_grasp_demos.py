#!/usr/bin/env python3
"""Generate counted physical grasp demonstrations for disclosed replay seeding.

The teacher uses geometric grasp proposals and contact information. It is used
only for training data generation, never added to learned policy observations
or inference. Every recorded action passes through the real skill environment;
there is no object teleport, weld, padded transition, or invented success flag.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation

from pnr.skills import ManipulationSkillEnv
from pnr.telemetry import RunLogger, atomic_json


class GeometricContactTeacher:
    """Markov approach/close/lift controller using current measured geometry."""
    def __init__(self, env):
        self.env = env
        obs = env.native.observe()
        self.grasp_local = Rotation.from_quat(obs['object_xyzw']).inv().apply(
            env.controller.grasp.position-obs['object_pos'])
        self.max_action = .65

    def action(self, observation):
        env = self.env
        native = env.native.observe()
        ee, _ = env.controller.planner.ee_pose()
        contact = env.native.contacts()['bilateral_contact']
        if contact:
            # Translation of the held object is controlled through its observed
            # goal error; fingers stay closed at every native sub-action.
            delta = observation['desired_goal']-observation['achieved_goal']
            command = np.r_[np.clip(delta/env.delta_scale, -self.max_action, self.max_action), -1.]
            phase = 'held_goal_correction'
        else:
            target = native['object_pos']+Rotation.from_quat(native['object_xyzw']).apply(self.grasp_local)
            delta = target-ee
            near = np.linalg.norm(delta) <= .005
            # Geometry, rather than a hidden phase counter, determines when
            # closure is appropriate. Continue correcting alignment while closing.
            command = np.r_[np.clip(delta/env.delta_scale, -self.max_action, self.max_action), -1. if near else 1.]
            phase = 'close' if near else 'approach'
        return command.astype(np.float32), phase


def save_dataset(path, episodes, metadata):
    transitions = [transition for episode in episodes for transition in episode['transitions']]
    artifact = {'schema_version':1, 'kind':'physical_training_demonstrations', 'training_only':True,
        'teacher_version':'geometric-contact-v1', 'metadata':metadata, 'episodes':episodes}
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    if transitions:
        arrays = {f'{prefix}_{key}':np.stack([transition[prefix][key] for transition in transitions])
            for prefix in ('observation', 'next_observation') for key in transitions[0]['observation']}
        arrays.update(actions=np.stack([transition['action'] for transition in transitions]),
            rewards=np.array([transition['reward'] for transition in transitions], dtype=np.float32),
            terminated=np.array([transition['terminated'] for transition in transitions], dtype=bool),
            truncated=np.array([transition['truncated'] for transition in transitions], dtype=bool),
            episode=np.array([transition['episode'] for transition in transitions], dtype=np.int32),
            step=np.array([transition['step'] for transition in transitions], dtype=np.int32),
            info_json=np.array([json.dumps(transition['info']) for transition in transitions]))
        np.savez_compressed(path.with_suffix('.npz'), **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='grasp_physical_demos_v1')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--seed', type=int, default=22000)
    args = parser.parse_args()
    if args.episodes != 32 or args.horizon != 100:
        raise ValueError('This collection protocol is fixed at 32 episodes, each capped at 100 actual decisions')
    logger = RunLogger(args.run_id, kind='validation', skill='grasp', algorithm='Physical demonstration collection',
        name='Physical grasp demonstrations', budget=args.episodes*args.horizon,
        summary='Validating geometric/contact teacher on four physical start heights; no policy training.')
    artifacts = logger.directory/'artifacts'; artifacts.mkdir(exist_ok=True)
    sources = artifacts/'source'; sources.mkdir(exist_ok=True)
    source_hashes = {}
    for name in ('scripts/generate_grasp_demos.py', 'pnr/skills.py', 'pnr/env.py',
                 'pnr/local_control.py', 'pnr/reference.py', 'pnr/planning.py'):
        content = (ROOT/name).read_bytes()
        source_hashes[name] = hashlib.sha256(content).hexdigest()
        (sources/name.replace('/', '__')).write_bytes(content)
    atomic_json(artifacts/'source_hashes.json', source_hashes)
    atomic_json(logger.directory/'config.json', vars(args) | {
        'training_only':True, 'goal_lift_m':.20, 'teacher_version':'geometric-contact-v1',
        'privileged_teacher_inputs':['geometric grasp proposal', 'current object pose', 'current contact state'],
        'policy_observations_modified':False, 'teacher_at_learned_inference':False,
        'preflight':'First four physical episodes, one per height; continue to 32 only when all pass.'})
    env = None
    episodes = []
    started = time.perf_counter()
    metadata = {'collection_status':'running', 'policy_decisions':0, 'demonstration_source_action_calls':0,
        'demonstration_physics_ticks':0, 'setup_source_action_calls':0, 'setup_physics_ticks':0,
        'skill':'grasp', 'scene_seed':42, 'object_idx':14, 'horizon':100, 'goal_lift_m':.20,
        'source_hashes':source_hashes, 'source_snapshot_directory':'source'}
    try:
        env = ManipulationSkillEnv(skill='grasp', scene_seed=42, object_idx=14, seed=args.seed,
            horizon=args.horizon, evaluation=True)
        metadata.update(setup_source_action_calls=env.setup_action_calls,
            setup_physics_ticks=env.setup_action_calls*env.native.substeps, initialization_physics_ticks=env.native.settle_steps, scene=env.native.scene_manifest(),
            curriculum_protocol=env.curriculum_protocol, start_heights_m=[float(height) for height, _ in env.snapshots],
            observation_keys=list(env.observation_space.spaces), native_action_contract='15 absolute position targets, 5 physics ticks per source action',
            policy_action_contract='Cartesian xyz refinement and coupled gripper command, 3 native calls per decision',
            observation_semantics='Unmodified 50-state + achieved/desired goal dictionary',
            reward_semantics='Recorded current skill sparse goal+retention reward and physical failure penalty')
        logger.update(setup_action_calls=env.setup_action_calls, setup_physics_ticks=env.setup_action_calls*env.native.substeps)
        teacher = GeometricContactTeacher(env)
        for episode_index in range(args.episodes):
            _, _ = env.reset(seed=args.seed+episode_index)
            height, snapshot = env.snapshots[episode_index % len(env.snapshots)]
            env.native.restore_state(snapshot, reset_counters=True)
            observation = env._observation()
            initial_object = env.native.observe()['object_pos'].copy()
            transitions = []
            first_goal_step = first_stable_hold_step = None
            max_lift = 0.
            for step in range(args.horizon):
                action, phase = teacher.action(observation)
                before = copy.deepcopy(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                info = copy.deepcopy(info)
                info['TimeLimit.truncated'] = bool(truncated and not terminated)
                transitions.append({'episode':episode_index, 'step':step, 'observation':before,
                    'action':action.copy(), 'next_observation':copy.deepcopy(next_observation),
                    'reward':float(reward), 'terminated':bool(terminated), 'truncated':bool(truncated),
                    'info':info, 'teacher_phase':phase})
                current_lift = float(env.native.observe()['object_pos'][2]-initial_object[2])
                max_lift = max(max_lift, current_lift)
                if info['is_success'] and first_goal_step is None:
                    first_goal_step = step+1
                if info['ever_stably_held'] and first_stable_hold_step is None:
                    first_stable_hold_step = step+1
                observation = next_observation
                if terminated or truncated:
                    break
            final_info = transitions[-1]['info']
            passed = bool(first_goal_step is not None and first_stable_hold_step is not None
                and final_info['bilateral_contact'] and final_info['goal_distance'] < env.tolerance
                and max_lift >= .16 and not transitions[-1]['terminated'])
            summary = {'episode':episode_index, 'seed':args.seed+episode_index, 'start_height_m':float(height),
                'goal_lift_m':.20, 'desired_goal':env.goal.tolist(), 'initial_object_position':initial_object.tolist(),
                'decisions':len(transitions), 'successful_physical_lift_and_hold':passed,
                'first_goal_decision':first_goal_step, 'first_stable_hold_decision':first_stable_hold_step,
                'maximum_lift_m':max_lift, 'final_goal_distance_m':float(final_info['goal_distance']),
                'final_bilateral_contact':final_info['bilateral_contact'],
                'source_action_calls':sum(t['info']['action_calls_delta'] for t in transitions),
                'physics_ticks':sum(t['info']['physics_ticks_delta'] for t in transitions),
                'physical_failure':transitions[-1]['terminated'],
                'bilateral_contact_fraction':float(np.mean([t['info']['bilateral_contact'] for t in transitions])),
                'action_rejection_fraction':float(np.mean([t['info']['action_rejected'] for t in transitions]))}
            episodes.append({'summary':summary, 'transitions':transitions})
            metadata.update(policy_decisions=sum(e['summary']['decisions'] for e in episodes),
                demonstration_source_action_calls=sum(e['summary']['source_action_calls'] for e in episodes),
                demonstration_physics_ticks=sum(e['summary']['physics_ticks'] for e in episodes),
                wall_seconds=time.perf_counter()-started)
            logger.metric(metadata['policy_decisions'], scope='training_data', metric_source='demonstration',
                episodes=len(episodes), successful_episodes=sum(e['summary']['successful_physical_lift_and_hold'] for e in episodes),
                success_rate=float(np.mean([e['summary']['successful_physical_lift_and_hold'] for e in episodes])),
                source_action_calls=metadata['demonstration_source_action_calls'], physics_ticks=metadata['demonstration_physics_ticks'])
            print(json.dumps(summary), flush=True)
            # Persist measured trajectories after every episode, including failed
            # preflight cases. Never discard failed teacher attempts.
            save_dataset(artifacts/'demonstrations.pkl', episodes, metadata)
            if episode_index == 3:
                preflight_passed = all(e['summary']['successful_physical_lift_and_hold'] for e in episodes)
                metadata['preflight_passed'] = preflight_passed
                if not preflight_passed:
                    raise RuntimeError('Physical four-height preflight failed; retained actual trajectories and stopped before expanded collection')
                logger.event('Four-height physical preflight passed; collecting the remaining balanced episodes.')
                logger.update(summary='Validated physical teacher; collecting balanced 100-step demonstration episodes.')
        metadata.update(collection_status='complete', wall_seconds=time.perf_counter()-started)
        digest = save_dataset(artifacts/'demonstrations.pkl', episodes, metadata)
        manifest = {**metadata, 'episodes':[episode['summary'] for episode in episodes],
            'dataset_sha256':digest, 'pickle_file':'demonstrations.pkl', 'array_file':'demonstrations.npz',
            'replay_import':'Use actual obs/action/next_obs/info. Episode ending flags come directly from the environment; TimeLimit.truncated separates horizon from physical failure.',
            'fair_comparison':'Development demonstration initialization must be disclosed and its data/interaction budget matched across future algorithm comparisons.'}
        atomic_json(artifacts/'manifest.json', manifest)
        logger.artifact('artifacts/demonstrations.pkl', 'Trusted local physical replay data')
        logger.artifact('artifacts/demonstrations.npz', 'Demonstration arrays and recorded info')
        logger.artifact('artifacts/manifest.json', 'Physical validation and counted data generation')
        logger.finish(summary=f"Saved {len(episodes)} physical demonstration episodes; {sum(e['summary']['successful_physical_lift_and_hold'] for e in episodes)} passed lift/hold validation. No policy training.")
    except BaseException as error:
        metadata.update(collection_status='failed', error=f'{type(error).__name__}: {error}', wall_seconds=time.perf_counter()-started)
        if episodes:
            save_dataset(artifacts/'demonstrations.pkl', episodes, metadata)
            atomic_json(artifacts/'manifest.json', {**metadata, 'episodes':[episode['summary'] for episode in episodes]})
            logger.artifact('artifacts/demonstrations.pkl', 'Retained measured demonstration attempts')
            logger.artifact('artifacts/manifest.json', 'Collection failure and counted attempts')
        logger.finish('failed', f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
