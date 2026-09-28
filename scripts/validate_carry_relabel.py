#!/usr/bin/env python3
"""Offline integrity and eight-worker actual-checkpoint HER append validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from pnr.carry import CarrySkillEnv
from pnr.replay_transfer import append_demonstrations
from pnr.telemetry import RunLogger,atomic_json
from scripts.pretrain_carry_actor import dataset_arrays


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def same(first,second):
    if isinstance(first,np.ndarray):return isinstance(second,np.ndarray) and first.dtype==second.dtype and np.array_equal(first,second)
    if isinstance(first,dict):return first.keys()==second.keys() and all(same(first[key],second[key]) for key in first)
    if isinstance(first,(list,tuple)):return type(first) is type(second) and len(first)==len(second) and all(same(a,b) for a,b in zip(first,second))
    return first==second


class RewardOnlyEnv(gym.Env):
    """Canonical spaces/reward; stepping is forbidden and no simulator exists."""
    goal_scale=20.;tolerance=.15;failure_cost=200.
    observation_space=spaces.Dict({'observation':spaces.Box(-np.inf,np.inf,(580,),np.float32),
        'achieved_goal':spaces.Box(-np.inf,np.inf,(2,),np.float32),
        'desired_goal':spaces.Box(-np.inf,np.inf,(2,),np.float32)})
    action_space=spaces.Box(-1.,1.,(2,),np.float32)
    compute_reward=CarrySkillEnv.compute_reward
    def reset(self,*,seed=None,options=None):
        return {key:np.zeros(value.shape,np.float32) for key,value in self.observation_space.spaces.items()},{}
    def step(self,action):raise AssertionError('Offline validation cannot execute a physical or synthetic transition')


def array_digest(replay,indices):
    result=hashlib.sha256()
    arrays=[*replay.observations.values(),*replay.next_observations.values(),replay.actions,replay.rewards]
    for values in arrays:result.update(np.ascontiguousarray(values[indices]).tobytes())
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--sampling-labels',type=Path,required=True)
    parser.add_argument('--run-id',default='carry_relabel_replay_validation01')
    args=parser.parse_args()
    torch.set_num_threads(1);np.random.seed(8143)
    source=ROOT/'runs/sac_carry_long01/checkpoints/step_000090000_1790558258297679854'
    originals=[ROOT/'runs/carry_physical_demos_v1/artifacts/demonstrations.pkl',
        ROOT/'runs/carry_lookahead_demos_v1/artifacts/corrections.pkl']
    protected_paths=[args.dataset,args.sampling_labels,*originals,source/'model.zip',source/'replay.pkl',source/'progress.json']
    hashes={str(path):digest(path) for path in protected_paths}
    data=pickle.loads(args.dataset.read_bytes());parts=[pickle.loads(path.read_bytes()) for path in originals]
    metadata=data['metadata']
    assert data['schema_version']==1 and metadata['metadata_finalized'] and metadata['collection_status']=='complete'
    assert (metadata['skill'],metadata['scene_seed'],metadata['object_idx'])==('carry',42,14)
    actual=[ep for part in parts for ep in part['episodes']]
    assert len(actual)==len(data['episodes'])==40
    protected=('observation','next_observation','action','effective_action','learner_action','reward','terminated','truncated','info')
    count=labels_changed=source_terminal_failures=incoming_terminated=0
    for old,new in zip(actual,data['episodes']):
        assert len(old['transitions'])==len(new['transitions'])
        goal=new['transitions'][0]['observation']['desired_goal']
        for index,(before,after) in enumerate(zip(old['transitions'],new['transitions'])):
            for key in protected:
                if key in before:assert same(before[key],after[key]),(count,key)
            for key in ('observation','next_observation'):
                assert np.array_equal(after[key]['desired_goal'],goal)
            if index:
                previous=new['transitions'][index-1]['next_observation']
                assert same(previous,after['observation'])
            terminal=after['terminated'] or after['truncated']
            assert terminal==(index==len(new['transitions'])-1)
            assert after['info']['TimeLimit.truncated']==bool(after['truncated'] and not after['terminated'])
            assert np.isfinite(after['teacher_action']).all() and np.linalg.norm(after['teacher_action'])<=1.+1e-6
            labels_changed+=int(not np.array_equal(before.get('teacher_action',before['action']),after['teacher_action']))
            incoming_terminated+=int(after['terminated']);count+=1
    assert count==sum(len(ep['transitions']) for ep in actual)
    labels=np.load(args.sampling_labels,allow_pickle=False)
    assert str(labels['dataset_sha256'])==hashes[str(args.dataset)]
    _,_,_,_,split=dataset_arrays(data,labels,7400,validation_episode_ids=metadata['validation_episode_ids'])
    assert set(metadata['validation_episode_ids'])=={1,5,9,24}
    expected={}
    for key in ('demonstration_source_action_calls','demonstration_physics_ticks','setup_source_action_calls',
                'setup_physics_ticks','initialization_physics_ticks','preflight_source_action_calls','preflight_physics_ticks'):
        expected[key]=sum(int(part['metadata'].get(key,0)) for part in parts)
        assert int(metadata.get(key,0))==expected[key],key
    expected['total_source_action_calls']=sum(expected[key] for key in ('demonstration_source_action_calls','setup_source_action_calls','preflight_source_action_calls'))
    expected['total_physics_ticks']=sum(expected[key] for key in ('demonstration_physics_ticks','setup_physics_ticks','initialization_physics_ticks','preflight_physics_ticks'))
    for key in ('total_source_action_calls','total_physics_ticks'):assert metadata[key]==expected[key]
    vec=DummyVecEnv([RewardOnlyEnv for _ in range(8)])
    try:
        model=SAC.load(source/'model.zip',env=vec,device='cpu')
        assert model.n_envs==8 and model.num_timesteps==90000 and model._n_updates==88992
        model.load_replay_buffer(source/'replay.pkl',truncate_last_traj=True)
        replay=model.replay_buffer
        assert replay.n_envs==8 and replay.pos==11250
        old_pos=replay.pos;old_full=replay.full
        old_indices=np.arange(replay.buffer_size if old_full else old_pos)
        old_states=array_digest(replay,old_indices)
        source_terminal_failures=sum(bool(info.get('physical_failure',False)) for row in replay.infos[old_indices] for info in row)
        report=append_demonstrations(model,args.dataset)
        streams=[[] for _ in range(8)]
        for episode in sorted(data['episodes'],key=lambda ep:-len(ep['transitions'])):min(streams,key=len).extend(episode['transitions'])
        length=min(map(len,streams))
        assert not old_full and old_pos+length<replay.buffer_size,'Audit expects no overwrite of90k source rows'
        assert array_digest(replay,old_indices)==old_states
        assert source_terminal_failures==sum(bool(info.get('physical_failure',False)) for row in replay.infos[old_indices] for info in row)
        distinct_teacher=0;mixed_rows=0
        for step in range(length):
            position=old_pos+step
            for worker in range(8):
                row=streams[worker][step]
                np.testing.assert_array_equal(replay.actions[position,worker],row['action'])
                np.testing.assert_array_equal(replay.observations['desired_goal'][position,worker],row['observation']['desired_goal'])
                for key in row['observation']:
                    np.testing.assert_array_equal(replay.observations[key][position,worker],row['observation'][key])
                    np.testing.assert_array_equal(replay.next_observations[key][position,worker],row['next_observation'][key])
                assert replay.rewards[position,worker]==row['reward']
                distinct_teacher+=int(not np.array_equal(row['teacher_action'],row['action']))
                mixed_rows+=int('learner_action' in row)
        assert np.all(replay._current_ep_start==replay.pos)
        rng=np.random.default_rng(8195)
        positions=rng.integers(old_pos,replay.pos,size=512);workers=rng.integers(0,8,size=512)
        virtual=replay._get_virtual_samples(positions,workers,env=None)
        info=[replay.infos[p,w] for p,w in zip(positions,workers)]
        expected_rewards=vec.envs[0].compute_reward(virtual.next_observations['achieved_goal'].numpy(),virtual.observations['desired_goal'].numpy(),info)
        np.testing.assert_array_equal(virtual.rewards.numpy().reshape(-1),expected_rewards)
        goals=replay._sample_goals(positions,workers)
        for position,worker,goal in zip(positions,workers,goals):
            start=replay.ep_start[position,worker];size=replay.ep_length[position,worker]
            assert size>0 and start>=old_pos
            future=np.arange(position,start+size)%replay.buffer_size
            allowed=replay.next_observations['achieved_goal'][future,worker]
            assert np.any(np.all(allowed==goal,axis=1)),'HER future goal escaped its physical episode'
        sampled=replay.sample(256)
        assert all(torch.isfinite(values).all() for values in sampled.observations.values())
        assert model.num_timesteps==90000 and model._n_updates==88992
        for path in protected_paths:assert digest(path)==hashes[str(path)],f'Immutable input changed: {path}'
        final={'scope':'offline_physical_data_and_actual_90k_HER_validation','simulator_calls':0,'physics_ticks':0,'training_updates':0,
            'dataset_sha256':hashes[str(args.dataset)],'source_hashes':hashes,'source_num_envs':8,'source_transitions':90000,
            'episodes':40,'actual_transitions':count,'protected_physical_records_unchanged':True,'fixed_goals_verified':True,
            'contiguous_states_verified':True,'changed_teacher_label_rows':labels_changed,
            'incoming_physical_terminal_failures':incoming_terminated,'source_replay_physical_failures_preserved':source_terminal_failures,
            'source_replay_observation_action_reward_arrays_unchanged':True,'source_files_and_counters_unchanged':True,
            'split':split,'generation_accounting':expected,
            'collector_policy_preparation':metadata.get('collector_policy_preparation'),
            'append':{key:value for key,value in report.items() if key!='dataset_metadata'},
            'teacher_label_differs_from_imported_executed_action_rows':distinct_teacher,'mixed_policy_rows_imported':mixed_rows,
            'virtual_reward_samples_verified':512,'future_goals_within_original_episode_verified':512,'mixed_replay_samples':256,
            'limitation':'Compute-only integrity evidence; no new physical success claim. Preflight costs are included in generation totals but preflight episodes are not appended.'}
        logger=RunLogger(args.run_id,kind='validation',skill='carry',name='Carry relabel replay integrity',summary='Offline check of physical records and8-worker90k HER append; no simulation.')
        atomic_json(logger.directory/'artifacts/result.json',final)
        (logger.directory/'artifacts/validate_carry_relabel.py').write_bytes(Path(__file__).read_bytes())
        logger.artifact('artifacts/result.json','Unchanged physics records, split, actual8-worker HER and generation costs')
        logger.artifact('artifacts/validate_carry_relabel.py','Independent offline validation source')
        logger.finish('complete','Relabeled data and actual90k eight-worker HER append verified; zero physics or training calls.')
        print(json.dumps({key:value for key,value in final.items() if key not in ('source_hashes','split','collector_policy_preparation')},indent=2))
    finally:vec.close()


if __name__=='__main__':main()
