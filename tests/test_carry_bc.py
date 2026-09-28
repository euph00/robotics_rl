"""Compute-only checks for physical-data labels, boundaries, and grouped splits."""
import copy

import numpy as np

from scripts.pretrain_carry_actor import (
    STRATA, dataset_arrays, merged_training_data, sampling_labels_for_dataset,
    sharp_turn_groups, training_batch,
)


def episode(index, length, case=0, correction=False):
    actions=np.array([[1.,0.]]*6+[[0.,1.]]*2+[[0.,0.]],np.float32)
    positions=np.array([[i*.2,0.] for i in range(6)]+[[1.,.2],[1.,.4],[1.,.6]])
    rows=[]
    for step,(target,position) in enumerate(zip(actions,positions)):
        observation={'observation':np.zeros(3,np.float32),
            'achieved_goal':position.astype(np.float32)/20.,
            'desired_goal':np.array([.3,.4],np.float32)}
        row={'episode':index,'step':step,'observation':observation,
            'next_observation':copy.deepcopy(observation),
            'action':target*.85 if correction else target.copy(),
            'reward':-1.,'terminated':False,'truncated':step==8,
            'teacher_phase':'retained_goal_hold' if step==8 else 'astar_path_following',
            'info':{'TimeLimit.truncated':step==8,'action_calls_delta':5,'physics_ticks_delta':25}}
        if correction:
            row['teacher_action']=target.copy()
            row['teacher_waypoint_index']=1 if step<6 else 2
        rows.append(row)
    return {'summary':{'episode':index,'kind':'full_route' if length is None else 'segment',
        'segment_length_m':30. if length is None else length,'start_arc_length_m':float(case),
        'goal_xy':[6.,8.]},'transitions':rows}


def datasets():
    metadata={'metadata_finalized':True,'skill':'carry','scene_seed':42,'object_idx':14,
        'scene':{},'demonstration_source_action_calls':45,'demonstration_physics_ticks':225,
        'setup_source_action_calls':3,'setup_physics_ticks':15,'initialization_physics_ticks':480}
    original=[]
    for length in (1.2,2.5,5.,10.):
        for case in range(3):original.append(episode(len(original),length,case))
    original.append(episode(len(original),None))
    correction=[episode(0,1.2,0,True),episode(1,None,correction=True)]
    return ({'schema_version':1,'metadata':copy.deepcopy(metadata),'episodes':original},
            {'schema_version':1,'metadata':copy.deepcopy(metadata),'episodes':correction})


def test_merge_preserves_executed_actions_goals_and_real_boundaries():
    original,correction=datasets()
    actual=correction['episodes'][0]['transitions'][0]['action'].copy()
    merged=merged_training_data(original,correction,[{'name':'original'},{'name':'corrections'}])
    record=merged['episodes'][len(original['episodes'])]['transitions'][0]
    np.testing.assert_array_equal(record['action'],actual)
    np.testing.assert_array_equal(record['teacher_action'],[1.,0.])
    assert record['episode']==len(original['episodes']) and record['source_episode']==0
    for before,after in zip(original['episodes']+correction['episodes'],merged['episodes']):
        for old,new in zip(before['transitions'],after['transitions']):
            assert old['terminated']==new['terminated'] and old['truncated']==new['truncated']
            np.testing.assert_array_equal(old['observation']['desired_goal'],new['observation']['desired_goal'])
            np.testing.assert_array_equal(old['action'],new['action'])
    assert 'teacher_action' not in original['episodes'][0]['transitions'][0]
    assert merged['metadata']['total_source_action_calls']==96
    assert merged['metadata']['total_physics_ticks']==1440


def test_grouped_split_keeps_cross_dataset_cases_together_and_uses_teacher_targets():
    original,correction=datasets()
    merged=merged_training_data(original,correction,[{'name':'original'},{'name':'corrections'}])
    labels=sampling_labels_for_dataset(merged,'test-digest')
    observations,targets,training,validation,split=dataset_arrays(merged,labels,7400)
    assert set(labels['stratum'])==set(STRATA)
    assert len(training)==15 and len(validation)==12
    assert targets.shape==(15*9,2)
    np.testing.assert_array_equal(targets[13*9],[1.,0.])
    assert len(observations['observation'])==len(targets)
    for group in set(labels['group_id']):
        ids=set(labels['episode_id'][labels['group_id']==group])
        assert ids<=set(split['train_episode_ids']) or ids<=set(split['validation_episode_ids'])
    assert {12,14}<=set(split['train_episode_ids'])
    assert (0 in split['train_episode_ids'])==(13 in split['train_episode_ids'])
    _,_,_,_,fixed=dataset_arrays(merged,labels,7400,validation_episode_ids=[0,3,6,9])
    assert fixed['validation_episode_ids']==[0,3,6,9,13]


def test_merge_rejects_changing_original_goal():
    original,correction=datasets()
    correction['episodes'][0]['transitions'][-1]['next_observation']['desired_goal'][0]+=.1
    import pytest
    with pytest.raises(ValueError,match='changed its desired goal'):
        merged_training_data(original,correction,[{'name':'original'},{'name':'corrections'}])


def test_declared_sharp_sampler_keeps_five_strata_and_total_phase_balance():
    original,correction=datasets()
    merged=merged_training_data(original,correction,[{'name':'original'},{'name':'corrections'}])
    labels=sampling_labels_for_dataset(merged,'test-digest')
    _,_,training,_,split=dataset_arrays(merged,labels,7400)
    sharp,counts=sharp_turn_groups(merged,labels,split)
    assert len(counts)==10 and all(value>0 for value in counts.values())
    indices=training_batch(np.random.default_rng(77),training,sharp)
    assert len(indices)==400
    for stratum in STRATA:
        chosen=indices[labels['stratum'][indices]==stratum]
        assert len(chosen)==80
        assert (labels['phase'][chosen]=='turn_approach').sum()==36
        assert (labels['phase'][chosen]=='straight').sum()==36
        assert (labels['phase'][chosen]=='goal_hold').sum()==8
        assert np.isin(chosen,sharp[f'{stratum}:exact_switch']).sum()>=12
        assert np.isin(chosen,sharp[f'{stratum}:switch_neighbors']).sum()>=12
