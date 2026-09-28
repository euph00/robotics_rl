#!/usr/bin/env python3
"""Verify saved recovery data boundaries, rewards and exact physical accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pnr.multi_carry import CarrySchema


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
    args=p.parse_args();d=pickle.loads(args.dataset.read_bytes());m=d['metadata'];schema=CarrySchema()
    assert m['protocol']=='multimaze-physical-originalgoal-recovery-v1' and m['partition']=='train'
    bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text());layouts={l['layout_id']:l for l in bank['partitions']['train']}
    counts={'episodes':len(d['episodes']),'successful_episodes':0,'transitions':0,'source_action_calls':0,
            'physics_ticks':0,'overshoot_original_goal_transitions':0,'outer_gate_overshoot_transitions':0}
    for e in d['episodes']:
        layout=layouts[e['layout_id']]
        assert layout['maze_hash']==e['maze_hash'] and e['original_exit_goal'] and not e['full_route']
        assert e['collection_mode']=='recovery' and e['transitions']
        last=e['transitions'][-1];assert last['terminated'] or last['truncated']
        goal=e['transitions'][0]['observation']['desired_goal'];expected=np.asarray(layout['goal_pos'][:2])/20
        assert np.allclose(goal,expected,atol=1e-7,rtol=0)
        rows=e['transitions'];nxt=np.stack([r['next_observation']['achieved_goal'] for r in rows]);infos=[r['info'] for r in rows]
        rewards=schema.compute_reward(nxt,np.broadcast_to(goal,nxt.shape),infos)
        assert np.allclose(rewards,[r['reward'] for r in rows],atol=1e-6,rtol=0)
        for index,r in enumerate(rows):
            assert np.array_equal(r['observation']['desired_goal'],goal) and np.array_equal(r['next_observation']['desired_goal'],goal)
            assert np.isfinite(r['action']).all() and np.max(np.abs(r['action']))<=1.
            assert np.isfinite(r['teacher_action']).all() and np.max(np.abs(r['teacher_action']))<=1.
            if index<len(rows)-1:
                assert not r['terminated'] and not r['truncated']
                assert all(np.array_equal(r['next_observation'][key],rows[index+1]['observation'][key]) for key in r['next_observation'])
            if e['success']:assert not r['info']['physical_failure'] and r['info']['retention_ok']
            counts['transitions']+=1;counts['source_action_calls']+=r['info']['action_calls_delta'];counts['physics_ticks']+=r['info']['physics_ticks_delta']
            xy=r['observation']['achieved_goal']*20;entry_y=5.5-layout['maze_entrance_cell'][0]
            overshoot=abs(xy[0]-1.5)<.2 and np.sign(entry_y)*(xy[1]-entry_y)>.5
            counts['overshoot_original_goal_transitions']+=int(overshoot)
            counts['outer_gate_overshoot_transitions']+=int(overshoot and abs(entry_y)>=1.5)
        counts['successful_episodes']+=int(e['success'])
        recovery=e['recovery_start'];assert recovery['start_error_m']<.075
        assert np.allclose(rows[0]['observation']['achieved_goal']*20,recovery['actual_start_xy'],atol=1e-6)
    assert counts['source_action_calls']==m['source_action_calls'] and counts['physics_ticks']==m['physics_ticks']
    costs={key:m[key] for key in ('source_action_calls','physics_ticks','preparation_source_action_calls','preparation_physics_ticks',
           'setup_source_action_calls','setup_physics_ticks','initialization_physics_ticks')}
    for key,value in costs.items():assert value==sum(s[key] for s in m['summaries'])
    result={'status':'passed','dataset':str(args.dataset),'sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'physical_interactions_for_verification':0,'counts':counts,'costs':costs,
            'all_source_action_calls':m['all_source_action_calls'],'all_physics_ticks':m['all_physics_ticks'],
            'failed_preparations':sum(a['status']=='failed_preparation' for s in m['summaries'] for a in s['recovery_attempts']),
            'geometry_skips':sum(a['status']=='skipped_geometry' for s in m['summaries'] for a in s['recovery_attempts'])}
    output=args.dataset.with_name('verification.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
