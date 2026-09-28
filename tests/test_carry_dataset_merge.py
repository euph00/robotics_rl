"""Carry merge provenance, immutable sources and correction-prefix sampling."""
import copy
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from pnr.demonstrations import carry_supervision,merge_carry_datasets
from pnr.multi_carry import CarrySchema
from pnr.scenes import load_scene_bank,scene_records

ROOT=Path(__file__).resolve().parents[1]


def episode(recovery=False,success=True):
    layout=scene_records(load_scene_bank(ROOT/'configs/phase2_scene_bank.json'),'train')[0]
    wall=np.asarray(layout['maze_map'],np.float32).ravel()
    goal=np.array(layout['goal_pos'][:2],np.float32)/20
    positions=[1.5,2.5,3.6,3.,3.] if recovery else [1.5,1.6,1.7,1.8]
    actions=[[0.,1.],[1.,0.],[0.,0.],[0.,0.]] if recovery else [[0.,1.],[1.,0.],[0.,0.]]
    def observation(x):
        raw=np.zeros(580,np.float32);raw[:264]=wall
        return {'observation':raw,'achieved_goal':np.array([x,1.5],np.float32)/20,'desired_goal':goal.copy()}
    rows=[];schema=CarrySchema()
    for index,action in enumerate(actions):
        before,after=observation(positions[index]),observation(positions[index+1])
        last=index==len(actions)-1
        info={'action_calls_delta':5,'physics_ticks_delta':25,'bilateral_contact':True,
              'retention_ok':success or not last,'physical_failure':not success and last,
              'previous_achieved_goal':before['achieved_goal'].copy()}
        rows.append({'observation':before,'next_observation':after,'action':np.array(action,np.float32),
            'teacher_action':np.array(action,np.float32),'reward':float(schema.compute_reward(after['achieved_goal'],goal,info)),
            'terminated':not success and last,'truncated':success and last,'info':info})
    return {'layout_id':layout['layout_id'],'maze_hash':layout['maze_hash'],'object_idx':14,
            'success':success,'collection_mode':'recovery' if recovery else 'standard','transitions':rows}


def dataset(episodes,preparation=0):
    n=sum(len(e['transitions']) for e in episodes)
    return {'schema_version':1,'episodes':episodes,'metadata':{'partition':'train','gamma':.995,
        'shaping_weight':2.,'delta_scale':.075,'repeat':5,'source_action_calls':n*5,'physics_ticks':n*25,
        'setup_source_action_calls':7,'setup_physics_ticks':35,'initialization_physics_ticks':10,
        'preparation_source_action_calls':preparation,'preparation_physics_ticks':preparation*5,
        'summaries':[{'failed_attempts':[{'reason':'test-only preparation failure','calls':preparation}]}]}}


def test_supervision_prefix_stops_at_first_crossing_and_excludes_failed_bc():
    data=dataset([episode(),episode(True),episode(True,False)])
    rows,groups,p,stats=carry_supervision(data)
    assert len(rows)==7 and stats['failed_episodes_excluded_from_bc']==1
    assert groups['recovery_prefix'].tolist()==[3,4]
    # Later returning below3.5m does not become a new correction prefix.
    assert 6 in groups['hold'] and 6 not in groups['recovery_prefix']
    np.testing.assert_allclose(p,[.3,.35,.245,.105])
    assert sorted(np.concatenate(list(groups.values())).tolist())==list(range(7))
    _,plain_groups,plain_p,_=carry_supervision(dataset([episode()]))
    assert list(plain_groups)==['turn_or_vertical','straight','hold']
    np.testing.assert_allclose(plain_p,[.5,.35,.15])


def test_merge_preserves_sources_failed_attempts_costs_and_hash_lineage(tmp_path):
    left,right=tmp_path/'left.pkl',tmp_path/'right.pkl'
    left.write_bytes(pickle.dumps(dataset([episode()]),protocol=5))
    right.write_bytes(pickle.dumps(dataset([episode(True),episode(True,False)],preparation=17),protocol=5))
    original={p:p.read_bytes() for p in (left,right)}
    destination=tmp_path/'merged.pkl';report=merge_carry_datasets([left,right],destination)
    merged=pickle.loads(destination.read_bytes());metadata=merged['metadata']
    assert len(merged['episodes'])==3 and not merged['episodes'][-1]['success']
    assert metadata['preparation_source_action_calls']==17 and metadata['all_source_action_calls']==86
    assert metadata['all_physics_ticks']==450
    assert metadata['new_physical_interactions']==0 and report['new_physical_interactions']==0
    assert metadata['source_datasets'][1]['metadata']['summaries'][0]['failed_attempts'][0]['calls']==17
    assert [r['sha256'] for r in metadata['source_datasets']]==[hashlib.sha256(original[p]).hexdigest() for p in (left,right)]
    assert all(p.read_bytes()==raw for p,raw in original.items())
    assert report['sha256']==hashlib.sha256(destination.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):merge_carry_datasets([left,right],destination)


@pytest.mark.parametrize('mutation,match',[
    (lambda d:d['metadata'].update(gamma=.9),'contract'),
    (lambda d:d['metadata'].update(partition='validation'),'TRAIN'),
    (lambda d:d['episodes'][0]['transitions'][0].update(action=np.array([2.,0.])),'action labels'),
    (lambda d:d['episodes'][0]['transitions'][0].update(reward=55.),'rewards'),
    (lambda d:d['metadata'].update(source_action_calls=999),'accounting'),
])
def test_merge_rejects_semantic_or_cost_mismatch_without_writing(tmp_path,mutation,match):
    left,right=tmp_path/'left.pkl',tmp_path/'right.pkl';first=dataset([episode()]);second=dataset([episode(True)])
    mutation(second);left.write_bytes(pickle.dumps(first));right.write_bytes(pickle.dumps(second));destination=tmp_path/'out.pkl'
    with pytest.raises(ValueError,match=match):merge_carry_datasets([left,right],destination)
    assert not destination.exists()


def test_duplicate_content_cannot_double_count_costs(tmp_path):
    data=pickle.dumps(dataset([episode()]));paths=[tmp_path/'first.pkl',tmp_path/'renamed.pkl']
    for path in paths:path.write_bytes(data)
    with pytest.raises(ValueError,match='Duplicate'):merge_carry_datasets(paths,tmp_path/'out.pkl')
