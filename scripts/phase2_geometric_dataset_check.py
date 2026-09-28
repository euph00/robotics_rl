#!/usr/bin/env python3
"""Validate physical TRAIN branch/entry recovery records without new physics."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pnr.demonstrations import carry_supervision
from pnr.multi_carry import CarrySchema


def check(path):
    data=pickle.loads(path.read_bytes());meta=data['metadata'];schema=CarrySchema()
    assert meta['protocol']=='multimaze-physical-geometric-recovery-v1' and meta['partition']=='train'
    bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layouts={r['layout_id']:r for r in bank['partitions']['train']}
    records=[];calls=ticks=0;max_error=0.
    for ep in data['episodes']:
        layout=layouts[ep['layout_id']];assert layout['maze_hash']==ep['maze_hash']
        assert ep['collection_mode']=='geometric-recovery' and ep['original_exit_goal'] and not ep['full_route']
        rows=ep['transitions'];assert rows and (rows[-1]['terminated'] or rows[-1]['truncated'])
        goal=rows[0]['observation']['desired_goal'];np.testing.assert_allclose(goal*20,layout['goal_pos'][:2],rtol=0.,atol=1e-5)
        prep=ep['recovery_start'];np.testing.assert_allclose(rows[0]['observation']['achieved_goal']*20,
            prep['preparation_trace_xy'][-1],rtol=0.,atol=1e-6)
        assert prep['preparation_decisions']==len(prep['preparation_trace_xy'])-1
        west=0
        for i,row in enumerate(rows):
            for key in ('action','teacher_action'):
                action=np.asarray(row[key]);assert action.shape==(2,) and np.isfinite(action).all() and np.max(np.abs(action))<=1.
            for key in ('observation','next_observation'):
                obs=row[key];assert np.array_equal(obs['desired_goal'],goal)
                np.testing.assert_array_equal(obs['observation'][:264],np.asarray(layout['maze_map']).reshape(-1))
            if i<len(rows)-1:
                assert not row['terminated'] and not row['truncated']
                for key in row['next_observation']:
                    np.testing.assert_array_equal(row['next_observation'][key],rows[i+1]['observation'][key])
            reward=float(schema.compute_reward(row['next_observation']['achieved_goal'],goal,row['info']))
            error=abs(reward-row['reward']);max_error=max(max_error,error);assert error<1e-5
            calls+=row['info']['action_calls_delta'];ticks+=row['info']['physics_ticks_delta']
            west+=int(row['observation']['achieved_goal'][0]*20>3.5 and row['teacher_action'][0]<-.5)
        records.append(dict(layout_id=ep['layout_id'],kind=ep['recovery_kind'],success=ep['success'],rows=len(rows),
                            interior_west_rows=west,start_xy=prep['actual_start_xy'],
                            preparation_decisions=prep['preparation_decisions'],
                            failure_reason=rows[-1]['info'].get('failure_reason')))
    assert calls==meta['source_action_calls'] and ticks==meta['physics_ticks']
    totals={key:sum(s[key] for s in meta['summaries']) for key in
            ('source_action_calls','physics_ticks','preparation_source_action_calls','preparation_physics_ticks',
             'setup_source_action_calls','setup_physics_ticks','initialization_physics_ticks')}
    assert all(value==meta[key] for key,value in totals.items())
    assert meta['all_source_action_calls']==sum(totals[k] for k in
        ('source_action_calls','preparation_source_action_calls','setup_source_action_calls'))
    assert meta['all_physics_ticks']==sum(totals[k] for k in
        ('physics_ticks','preparation_physics_ticks','setup_physics_ticks','initialization_physics_ticks'))
    _,groups,probabilities,supervision=carry_supervision(data)
    attempts=[attempt for summary in meta['summaries'] for attempt in summary['recovery_attempts']]
    failed_preparation=[dict(layout_id=summary['layout_id'],**attempt)
        for summary in meta['summaries'] for attempt in summary['recovery_attempts'] if attempt['status']=='failed_preparation']
    return dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),layouts=meta['layouts_completed'],
        episodes=len(records),successful_episodes=sum(r['success'] for r in records),
        transitions=sum(r['rows'] for r in records),attempts=len(attempts),failed_preparation=failed_preparation,
        interior_episodes=sum(r['kind']=='interior_branch' for r in records),
        successful_interior_west_rows=sum(r['interior_west_rows'] for r in records if r['success'] and r['kind']=='interior_branch'),
        exact_observation_continuity=True,physical_preparation_endpoint_matches=True,reward_max_absolute_error=max_error,
        all_source_calls=meta['all_source_action_calls'],all_physics_ticks=meta['all_physics_ticks'],
        supervision=supervision,records=records,audit_new_physics_ticks=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh audit path')
    report=check(args.dataset);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','failed_preparation')}))
    print(json.dumps({'failed_preparations':len(report['failed_preparation'])}))


if __name__=='__main__':main()
