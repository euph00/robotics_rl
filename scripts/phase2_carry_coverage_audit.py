#!/usr/bin/env python3
"""Count physical-demo coverage near maze entry turns; no simulator operations."""
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np


def main():
    path=ROOT/'runs/phase2_carry_dataset64/artifacts/demonstrations.pkl'
    data=pickle.loads(path.read_bytes());bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layouts={row['layout_id']:row for row in bank['partitions']['train']}
    total=collections.Counter();per_layout=collections.defaultdict(collections.Counter)
    episodes=collections.Counter();overshoot_layouts=collections.Counter()
    for episode in data['episodes']:
        layout=layouts[episode['layout_id']];rows=episode['transitions']
        xy=np.stack([r['observation']['achieved_goal'] for r in rows])*20
        goals=np.stack([r['observation']['desired_goal'] for r in rows])*20
        entrance_y=5.5-layout['maze_entrance_cell'][0]
        full=np.linalg.norm(goals-np.array(layout['goal_pos'][:2]),axis=1)<1e-4
        room=xy[:,0]<2.
        at_turn=(np.abs(xy[:,0]-1.5)<.25)&(np.abs(xy[:,1]-entrance_y)<.35)
        overshoot=(np.abs(xy[:,0]-1.5)<.2)&(np.sign(entrance_y)*(xy[:,1]-entrance_y)>.5)
        # Entries at +/-0.5 m sometimes require table-avoidance paths that pass
        # beyond the entrance. Keep that special geometry separate.
        away_from_center=abs(entrance_y)>=1.5
        masks={'transitions':np.ones(len(rows),bool),'full_exit_goal':full,'room':room,
               'room_full_exit_goal':room&full,'entrance_turn':at_turn,
               'entrance_turn_full_exit_goal':at_turn&full,'overshoot':overshoot,
               'overshoot_full_exit_goal':overshoot&full,
               'outer_gate_overshoot_full_exit_goal':overshoot&full&away_from_center}
        for key,mask in masks.items():
            count=int(mask.sum());total[key]+=count;per_layout[episode['layout_id']][key]+=count
        episodes['all']+=1;episodes['declared_full']+=bool(episode['full_route'])
        episodes['actual_full_exit_goal']+=bool(full.all())
        if np.any(overshoot&full):overshoot_layouts[episode['layout_id']]+=int((overshoot&full).sum())
    result={'dataset':str(path),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'physical_interactions':0,'gradient_updates':0,'layouts':len(per_layout),
            'episodes':dict(episodes),'transitions':dict(total),
            'full_exit_goal_transition_fraction':total['full_exit_goal']/total['transitions'],
            'overshoot_fullgoal_layouts':dict(overshoot_layouts),
            'per_layout':{key:dict(value) for key,value in per_layout.items()},
            'definitions':{'entrance_turn':'|x-1.5|<.25 m, |y-entry_y|<.35 m',
                'overshoot':'|x-1.5|<.2 m, sign(entry_y)*(y-entry_y)>.5 m',
                'outer_gate':'|entry_y|>=1.5 m; separates table-avoidance entries at +/-0.5 m',
                'full_exit_goal':'actual desired goal matches that layout exit within .1 mm'},
            'limits':['Spatial counts describe recorded states only; they do not prove why a network generalizes poorly.',
                      'No validation/test transitions are added to training data.']}
    out=ROOT/'runs/phase2_carry_trace_audit01/artifacts/demo_coverage.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_layout','definitions','limits')},indent=2))


if __name__=='__main__':main()
