"""Append counted physical demonstrations to an existing HER development replay."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pickle
import numpy as np


def append_demonstrations(model, path):
    """Continue at an explicit episode boundary; use only executed actions.

    Longest episodes are assigned first to the shortest worker stream. Shorter
    streams set the shared import length; any final real prefix receives a
    collector timeout. Optional teacher_action labels are deliberately ignored:
    they supervise an actor but are not the action that produced a transition.
    """
    path = Path(path).resolve()
    content = path.read_bytes()
    data = pickle.loads(content)
    if data.get('schema_version') != 1 or not data.get('episodes'):
        raise ValueError('Expected a trusted local schema-1 physical dataset')
    replay = model.replay_buffer
    if replay.n_envs != model.n_envs:
        raise ValueError('Replay worker count differs')
    streams = [[] for _ in range(model.n_envs)]
    for episode in sorted(data['episodes'], key=lambda item:-len(item['transitions'])):
        records = episode['transitions']
        if not records or not (records[-1]['terminated'] or records[-1]['truncated']):
            raise ValueError('Each physical episode must end explicitly')
        if any(row['terminated'] or row['truncated'] for row in records[:-1]):
            raise ValueError('Dataset episode crosses a physical reset')
        min(streams, key=len).extend(records)
    length = min(map(len, streams))
    if not 0 < length < replay.buffer_size:
        raise ValueError('Need data for each worker and room for complete episodes')
    # Parent loading normally already does this; direct callers also get safe
    # boundaries without inventing a physical terminal failure.
    replay.truncate_last_trajectory()
    if np.any(replay._current_ep_start != replay.pos):
        raise ValueError('Replay still contains an open tail')
    initial_pos, initial_full = int(replay.pos), bool(replay.full)
    calls = ticks = timeouts = 0
    for index in range(length):
        records = [stream[index] for stream in streams]
        before = {key:np.stack([row['observation'][key] for row in records]).astype(np.float32)
                  for key in model.observation_space.spaces}
        after = {key:np.stack([row['next_observation'][key] for row in records]).astype(np.float32)
                 for key in model.observation_space.spaces}
        actions = np.stack([row['action'] for row in records]).astype(np.float32)
        if actions.shape != (model.n_envs,*model.action_space.shape) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.+1e-6:
            raise ValueError('Executed action shape/range mismatch')
        for key, space in model.observation_space.spaces.items():
            if before[key].shape != (model.n_envs,*space.shape) or after[key].shape != before[key].shape:
                raise ValueError('Observation shape mismatch')
            if not np.isfinite(before[key]).all() or not np.isfinite(after[key]).all():
                raise ValueError('Nonfinite physical observation')
        infos = [dict(row['info']) for row in records]
        dones = np.array([row['terminated'] or row['truncated'] for row in records], dtype=bool)
        for worker, (row, info) in enumerate(zip(records, infos)):
            timeout = bool(row['truncated'] and not row['terminated'])
            if index == length-1 and not dones[worker]:
                dones[worker] = True; timeout = True; timeouts += 1
            info['TimeLimit.truncated'] = timeout
            calls += int(info['action_calls_delta']); ticks += int(info['physics_ticks_delta'])
        rewards = np.asarray([row['reward'] for row in records], dtype=np.float32)
        expected = model.env.env_method('compute_reward', after['achieved_goal'], before['desired_goal'], infos, indices=[0])[0]
        if not np.allclose(rewards, expected, atol=1e-6, rtol=0.):
            raise ValueError('Dataset and current physical reward differ')
        replay.add(before, after, actions, rewards, dones, infos)
    if np.any(replay._current_ep_start != replay.pos):
        raise RuntimeError('Appended data did not close every worker stream')
    available = sum(map(len, streams))
    return {'method':'physical_demonstration_replay_append', 'actor_behavior_cloning':False,
        'path':str(path), 'sha256':hashlib.sha256(content).hexdigest(),
        'dataset_metadata':data.get('metadata',{}), 'episodes_collected':len(data['episodes']),
        'transitions_collected':available, 'transitions_imported':length*model.n_envs,
        'transitions_omitted_from_replay':available-length*model.n_envs,
        'imported_source_action_calls':calls, 'imported_physics_ticks':ticks,
        'collector_timeouts':timeouts, 'worker_stream_lengths':list(map(len,streams)),
        'replay_position_before':initial_pos, 'replay_full_before':initial_full,
        'replay_position_after':int(replay.pos), 'replay_full_after':bool(replay.full),
        'teacher_labels_used_as_executed_actions':False,
        'teacher_at_evaluation':False, 'hindsight_at_evaluation':False}
