#!/usr/bin/env python3
"""Validate recorded on-policy recovery data offline; no simulator or learner updates."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pnr.multi_carry import CarrySchema


def check(path):
    data=pickle.loads(path.read_bytes());metadata=data['metadata'];schema=CarrySchema()
    assert metadata['protocol']=='multimaze-physical-onpolicy-recovery-v1'
    assert metadata['partition']=='train'
    bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layouts={r['layout_id']:r for r in bank['partitions']['train']}
    records=[];calls=ticks=total_rows=0;max_reward_error=0.
    for episode in data['episodes']:
        layout=layouts[episode['layout_id']];assert episode['maze_hash']==layout['maze_hash']
        assert episode['full_route'] and episode['original_exit_goal']
        rows=episode['transitions'];assert rows and (rows[-1]['terminated'] or rows[-1]['truncated'])
        checkpoint=Path(episode['learner']['frozen_path'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==episode['learner']['sha256']
        goal=rows[0]['observation']['desired_goal']
        np.testing.assert_allclose(goal*20,layout['goal_pos'][:2],atol=1e-5,rtol=0.)
        sources=[r['control_source'] for r in rows]
        switches=sum(a!=b for a,b in zip(sources[:-1],sources[1:]));assert switches<=1
        if switches:assert sources[0]=='learner' and sources[-1]=='expert_recovery'
        for i,row in enumerate(rows):
            for field in ('action','teacher_action'):
                action=np.asarray(row[field]);assert action.shape==(2,) and np.isfinite(action).all() and np.max(np.abs(action))<=1.
            for field in ('observation','next_observation'):
                obs=row[field]
                assert np.array_equal(obs['desired_goal'],goal)
                np.testing.assert_array_equal(obs['observation'][:264],np.asarray(layout['maze_map']).reshape(-1))
            if i<len(rows)-1:
                assert not row['terminated'] and not row['truncated']
                for key in row['next_observation']:
                    np.testing.assert_array_equal(row['next_observation'][key],rows[i+1]['observation'][key])
            reward=float(schema.compute_reward(row['next_observation']['achieved_goal'],goal,row['info']))
            error=abs(reward-row['reward']);max_reward_error=max(max_reward_error,error);assert error<1e-5
            calls+=int(row['info']['action_calls_delta']);ticks+=int(row['info']['physics_ticks_delta'])
            if row['control_source']=='expert_recovery':np.testing.assert_array_equal(row['action'],row['teacher_action'])
            elif row.get('learner_mean_action') is not None:
                np.testing.assert_allclose(row['action'],row['learner_mean_action']+row['applied_action_noise'],atol=1e-7,rtol=0.)
        total_rows+=len(rows);xy=np.stack([r['observation']['achieved_goal'] for r in rows])*20
        labels=np.stack([r['teacher_action'] for r in rows]);actual=np.stack([r['action'] for r in rows])
        inside=xy[:,0]>3.5;west=labels[:,0]<-.5
        records.append(dict(layout_id=episode['layout_id'],learner_source=episode['learner']['source'],
            success=episode['success'],rows=len(rows),takeover=episode['on_policy_recovery']['takeover'],
            collector_error=episode['on_policy_recovery']['collector_error'],
            physical_failure=bool(rows[-1]['terminated']),
            collision_transitions=sum(bool(r['info'].get('collision',False)) for r in rows),
            final_failure_reason=rows[-1]['info'].get('failure_reason'),
            noise_std=episode['on_policy_recovery'].get('noise_std',0.),
            learner_rows=sources.count('learner'),expert_recovery_rows=sources.count('expert_recovery'),
            interior_teacher_west_rows=int((inside&west).sum()),
            interior_actual_west_rows=int((inside&(actual[:,0]<-.5)).sum()),
            action_teacher_difference_rows=int((np.linalg.norm(actual-labels,axis=1)>1e-5).sum()),
            position_min_m=xy.min(0).tolist(),position_max_m=xy.max(0).tolist()))
    assert calls==metadata['source_action_calls'] and ticks==metadata['physics_ticks']
    return dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),episodes=len(records),
                successful_episodes=sum(r['success'] for r in records),transitions=total_rows,
                exact_observation_continuity=True,original_goal_fixed=True,actual_actions_preserved=True,
                reward_max_absolute_error=max_reward_error,rollout_source_calls=calls,rollout_physics_ticks=ticks,
                all_source_calls=metadata['all_source_action_calls'],all_physics_ticks=metadata['all_physics_ticks'],
                interior_teacher_west_rows=sum(r['interior_teacher_west_rows'] for r in records),
                takeovers=sum(r['takeover'] is not None for r in records),records=records,
                audit_new_physics_ticks=0,audit_gradient_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh audit path')
    report=check(args.dataset);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
