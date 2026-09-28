#!/usr/bin/env python3
"""Audit saved carry failures and TRAIN demonstration coverage; no physics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def first(mask):
    found=np.flatnonzero(mask)
    return int(found[0]+1) if len(found) else None


def trace_summary(record,layout):
    xy=np.asarray(record['trajectory']); moves=np.diff(xy,axis=0)
    gate_y=5.5-layout['maze_entrance_cell'][0]
    interior=xy[:,0]>3.5
    region=(np.abs(xy[:,0]-4.5)<.2) if interior.any() else (np.abs(xy[:,0]-1.5)<.2)
    indices=np.flatnonzero(region)
    directed=np.sign(moves[np.maximum(indices[indices>0]-1,0),1])
    directed=directed[np.abs(moves[np.maximum(indices[indices>0]-1,0),1])>.01]
    return dict(layout_id=record['layout_id'],success=record['success'],decisions=record['decisions'],
                failure_reason=record['failure_reason'],goal_distance_m=record['goal_distance'],
                first_x_above_3_5_decision=first(interior),gate_y_m=gate_y,
                position_range_m=[xy.min(0).tolist(),xy.max(0).tolist()],
                median_move_m=float(np.median(np.linalg.norm(moves,axis=1))),
                oscillation_region='first_interior_branch_x4.5' if interior.any() else 'entrance_room_x1.5',
                region_decisions=int(region.sum()),region_y_range_m=[float(xy[region,1].min()),float(xy[region,1].max())],
                region_y_reversals=int((np.diff(directed)!=0).sum()),
                room_wrong_side_y_above0_decisions=int(((xy[:,0]<2.)&(xy[:,1]>0)).sum()),
                maximum_gate_y_offset_m=float((xy[:,1]-gate_y).max()))


def train_coverage(layouts):
    counts=Counter();by_gate={};runtimes=[];all_costs=Counter();shards=[]
    for run in ('phase2_carry_demos01','phase2_carry_demos02','phase2_carry_recovery64'):
        for path in sorted((ROOT/'runs'/run/'artifacts').glob('layout_*.pkl')):
            data=pickle.loads(path.read_bytes());summary=data['summary'];runtimes.append(summary['wall_seconds'])
            for key in ('source_action_calls','preparation_source_action_calls','setup_source_action_calls',
                        'physics_ticks','preparation_physics_ticks','setup_physics_ticks','initialization_physics_ticks'):
                all_costs[key]+=summary.get(key,0)
            shards.append(str(path.relative_to(ROOT)))
            for ep in data['episodes']:
                assert ep['layout_id'].startswith('train_')
                counts['episodes']+=1;counts['failed_episodes']+=not ep['success']
                if not ep['success']:continue
                layout=layouts[ep['layout_id']];gate_row=layout['maze_entrance_cell'][0];gate_y=5.5-gate_row
                rows=ep['transitions'];xy=np.stack([r['observation']['achieved_goal'] for r in rows])*20
                actions=np.stack([r['teacher_action'] for r in rows]);strip=np.abs(xy[:,0]-1.5)<.2
                interior=xy[:,0]>3.5
                counts['successful_rows']+=len(rows);counts['interior_rows']+=int(interior.sum())
                counts['interior_teacher_west_rows']+=int((interior&(actions[:,0]<-.5)).sum())
                stats=by_gate.setdefault(str(gate_row),dict(layout_ids=set(),successful_episodes=0,room_strip_rows=0,
                    room_strip_y_min_m=None,room_strip_y_max_m=None,room_strip_y_above0_rows=0,
                    recovery_start_y_min_m=None,recovery_start_y_max_m=None,recovery_episodes=0))
                stats['layout_ids'].add(ep['layout_id']);stats['successful_episodes']+=1
                stats['room_strip_rows']+=int(strip.sum());stats['room_strip_y_above0_rows']+=int((strip&(xy[:,1]>0)).sum())
                if strip.any():
                    lo,hi=float(xy[strip,1].min()),float(xy[strip,1].max())
                    stats['room_strip_y_min_m']=lo if stats['room_strip_y_min_m'] is None else min(lo,stats['room_strip_y_min_m'])
                    stats['room_strip_y_max_m']=hi if stats['room_strip_y_max_m'] is None else max(hi,stats['room_strip_y_max_m'])
                if ep.get('collection_mode')=='recovery':
                    counts['recovery_successful_rows']+=len(rows);stats['recovery_episodes']+=1
                    y=float(xy[0,1]);lo=stats['recovery_start_y_min_m'];hi=stats['recovery_start_y_max_m']
                    stats['recovery_start_y_min_m']=y if lo is None else min(lo,y)
                    stats['recovery_start_y_max_m']=y if hi is None else max(hi,y)
                    counts['recovery_starts_inside_maze']+=int(xy[0,0]>3.5)
    for stats in by_gate.values():stats['layout_ids']=sorted(stats['layout_ids'])
    return dict(counts=dict(counts),by_gate_row=by_gate,source_shards=shards,
                historical_worker_seconds_sum=sum(runtimes),source_costs=dict(all_costs))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh audit path')
    bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layouts={r['layout_id']:r for values in bank['partitions'].values() for r in values}
    report=dict(new_physics_ticks=0,new_gradient_updates=0,validation_used_only_for_diagnosis=True,runs=[])
    for run in ('phase2_carry_recovery01','phase2_carry_navigation01'):
        path=ROOT/'runs'/run/'artifacts/evaluation_bc_zero_online.json';data=json.loads(path.read_text())
        report['runs'].append(dict(run=run,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            failures=[trace_summary(r,layouts[r['layout_id']]) for r in data['records'] if not r['success']]))
    report['train_coverage']=train_coverage(layouts)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='train_coverage'},indent=2))
    print(json.dumps({'train_counts':report['train_coverage']['counts'],
                      'low_gate_row9':report['train_coverage']['by_gate_row'].get('9')},indent=2))


if __name__=='__main__':main()
