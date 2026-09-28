"""Seed HER with explicitly counted physical demonstrations, without actor cloning."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import numpy as np


def seed_replay(model, path):
    """Import a trusted local dataset into a fresh vector HER replay buffer.

    Whole demonstration episodes are assigned to worker streams. Only real
    transitions are used. Unequal streams are cropped to the shortest stream;
    the final prefix of each unfinished episode gets a collector timeout, so
    subsequent online resets cannot connect unrelated physical trajectories.
    """
    path=Path(path).resolve()
    data=pickle.loads(path.read_bytes())
    if data.get('schema_version') != 1 or not data.get('episodes'):
        raise ValueError('Expected a schema-1 physical demonstration dataset')
    replay=model.replay_buffer
    if replay.pos or replay.full:
        raise ValueError('Demonstrations can only seed a fresh replay buffer')
    streams=[[] for _ in range(model.n_envs)]
    for episode in data['episodes']:
        transitions=episode['transitions']
        if not transitions:
            raise ValueError('Empty demonstration episode')
        if not (transitions[-1]['terminated'] or transitions[-1]['truncated']):
            raise ValueError('Dataset episodes must end explicitly')
        stream=min(streams,key=len)
        stream.extend(transitions)
    length=min(map(len,streams))
    if not length:
        raise ValueError('Need at least one physical episode per vector worker')
    if length>=replay.buffer_size:
        raise ValueError('Seed data must fit the configured replay capacity')
    action_calls=physics_ticks=collector_timeouts=0
    for index in range(length):
        records=[stream[index] for stream in streams]
        observations={key:np.stack([record['observation'][key] for record in records]).astype(np.float32)
                      for key in model.observation_space.spaces}
        following={key:np.stack([record['next_observation'][key] for record in records]).astype(np.float32)
                   for key in model.observation_space.spaces}
        actions=np.stack([record['action'] for record in records]).astype(np.float32)
        if actions.shape != (model.n_envs,*model.action_space.shape) or not np.isfinite(actions).all() or np.max(np.abs(actions))>1.+1e-6:
            raise ValueError('Demonstration action shape/range mismatch')
        for key,space in model.observation_space.spaces.items():
            if observations[key].shape != (model.n_envs,*space.shape) or following[key].shape != observations[key].shape:
                raise ValueError('Demonstration observation shape mismatch')
            if not np.isfinite(observations[key]).all() or not np.isfinite(following[key]).all():
                raise ValueError('Nonfinite demonstration observation')
        infos=[dict(record['info']) for record in records]
        dones=np.array([record['terminated'] or record['truncated'] for record in records],dtype=bool)
        for worker,(record,info) in enumerate(zip(records,infos)):
            timeout=bool(record['truncated'] and not record['terminated'])
            if index==length-1 and not dones[worker]:
                dones[worker]=True;timeout=True;collector_timeouts+=1
            info['TimeLimit.truncated']=timeout
            action_calls+=int(info['action_calls_delta'])
            physics_ticks+=int(info['physics_ticks_delta'])
        rewards=np.asarray([record['reward'] for record in records],dtype=np.float32)
        recomputed=model.env.env_method('compute_reward',following['achieved_goal'],observations['desired_goal'],infos,indices=[0])[0]
        if not np.allclose(rewards,recomputed,atol=1e-5,rtol=0.):
            raise ValueError('Dataset reward differs from the active environment contract')
        replay.add(observations,following,actions,rewards,dones,infos)
    available=sum(len(stream) for stream in streams)
    return {'method':'physical_demonstration_replay_seed','actor_behavior_cloning':False,
        'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'dataset_metadata':data.get('metadata',{}),'episodes_collected':len(data['episodes']),
        'transitions_collected':available,'transitions_imported':length*model.n_envs,
        'transitions_omitted_from_replay':available-length*model.n_envs,
        'imported_source_action_calls':action_calls,'imported_physics_ticks':physics_ticks,
        'collector_timeouts':collector_timeouts,
        'teacher_at_evaluation':False,'hindsight_at_evaluation':False,'reward_roundoff_atol':1e-5}
