#!/usr/bin/env python3
"""Summarize existing physical carry traces without replaying the simulator."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=ROOT/'runs/phase2_carry_spatial02/artifacts/evaluation_bc_zero_online.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/phase2_carry_trace_audit01/artifacts')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.evaluation.read_text());bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layouts={r['layout_id']:r for values in bank['partitions'].values() for r in values}
    summaries=[]
    for record in data['records']:
        xy=np.asarray(record['trajectory'],dtype=float);step=np.diff(xy,axis=0);speed=np.linalg.norm(step,axis=1)
        cells=np.rint(np.c_[5.5-xy[:,1],xy[:,0]+3.5]).astype(int)
        unique,counts=np.unique(cells,axis=0,return_counts=True)
        hot=[]
        for index in counts.argsort()[::-1][:8]:
            selected=np.flatnonzero((cells==unique[index]).all(1))
            split=np.r_[0,np.flatnonzero(np.diff(selected)>1)+1,len(selected)]
            spans=[[int(selected[a]+1),int(selected[b-1]+1)] for a,b in zip(split[:-1],split[1:])]
            hot.append({'cell':unique[index].tolist(),'decisions':int(counts[index]),'visits':len(spans),'decision_spans':spans})
        gate=np.asarray(layouts[record['layout_id']]['maze_entrance_cell'])
        entries=np.flatnonzero((cells==gate).all(1))
        summary={'layout_id':record['layout_id'],'success':record['success'],'decisions':record['decisions'],
                 'measured_interpoint_moves':len(speed),'path_length_m':float(speed.sum()),
                 'median_m_per_decision':float(np.median(speed)),'mean_m_per_decision':float(speed.mean()),
                 'moves_below_1mm':int((speed<.001).sum()),'moves_below_1cm':int((speed<.01).sum()),
                 'moves_above_5cm':int((speed>.05).sum()),
                 'first_gate_cell_decision':int(entries[0]+1) if len(entries) else None,'hotspots':hot}
        if record['layout_id']=='validation_002':
            region=(cells[:,1]==5)&np.isin(cells[:,0],[1,2,3])
            inds=np.flatnonzero(region);moves=step[inds[inds>0]-1]
            directed=np.sign(moves[np.abs(moves[:,1])>.01,1])
            summary['wrong_way_strip']={'cell_column':5,'cell_rows':[1,2,3],
                'decisions':int(region.sum()),'first_decision':int(inds[0]+1),'last_decision':int(inds[-1]+1),
                'path_length_m':float(np.linalg.norm(moves,axis=1).sum()),
                'vertical_direction_reversals_above_1cm':int((np.diff(directed)!=0).sum()),
                'x_range_m':[float(xy[region,0].min()),float(xy[region,0].max())],
                'y_range_m':[float(xy[region,1].min()),float(xy[region,1].max())]}
            summary['after_gate']={'decisions_from_first_gate_position':len(xy)-int(entries[0])-1,
                'path_length_m':float(speed[entries[0]:].sum()),
                'median_m_per_decision':float(np.median(speed[entries[0]:]))}
            figure(xy,np.asarray(layouts[record['layout_id']]['maze_map']),speed,args.output_dir/'validation_002_trace.png')
        summaries.append(summary)
    result={'source':str(args.evaluation),'physical_interactions':0,'gradient_updates':0,
            'limits':['Only post-decision XY was saved; the first displacement is unavailable.',
                      'A map cell visit is a descriptive location bin, not a collision test.',
                      'No action/state/contact replay or optimal physical trajectory is inferred.'],
            'records':summaries}
    (args.output_dir/'trace_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(next(r for r in summaries if r['layout_id']=='validation_002'),indent=2))


def figure(xy,maze,speed,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].imshow(maze,cmap='Greys',origin='upper',extent=(-4,18,-6,6),vmin=0,vmax=1)
    axes[0].plot(xy[:,0],xy[:,1],color='#277da1',linewidth=1.5)
    axes[0].scatter(xy[::100,0],xy[::100,1],s=15,color='#d65f32')
    for i in range(0,len(xy),100):axes[0].annotate(str(i+1),xy[i],fontsize=7)
    axes[0].set(title='Validation 002: repeated wrong-way motion',xlabel='World x (m)',ylabel='World y (m)')
    axes[1].plot(np.arange(1,len(xy)+1),xy[:,1],linewidth=1.2,color='#277da1')
    axes[1].axvspan(69,616,color='#d65f32',alpha=.13,label='548 decisions near x = 1.5 m')
    axes[1].axvline(634,color='#777777',linewidth=1,linestyle='--',label='First gate entry')
    axes[1].set(xlabel='Decision',ylabel='World y (m)',title='Actuator moves; route choice repeats')
    axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':main()
