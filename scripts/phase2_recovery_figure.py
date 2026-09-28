#!/usr/bin/env python3
"""Plot recorded physical preparation and recovery; never replay the simulator."""
import argparse
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--layout',default='train_000')
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new figure path')
    data=pickle.loads(args.dataset.read_bytes())
    selected=[]
    for kind in ('interior_branch','entry_extreme'):
        selected.append(next(e for e in data['episodes'] if e['layout_id']==args.layout and e['recovery_kind']==kind and e['success']))
    bank=json.loads((ROOT/'configs/phase2_scene_bank.json').read_text())
    layout=next(r for r in bank['partitions']['train'] if r['layout_id']==args.layout)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained',sharey=True)
    for ax,episode,title in zip(axes,selected,('Return from a dead end','Return to the entrance')):
        prep=np.array(episode['recovery_start']['preparation_trace_xy'])
        xy=np.stack([r['observation']['achieved_goal'] for r in episode['transitions']])*20
        labels=np.stack([r['teacher_action'] for r in episode['transitions']])
        west=(labels[:,0]<-.5)&(xy[:,0]>3.5)
        ax.imshow(layout['maze_map'],origin='upper',extent=(-4,18,-6,6),cmap='Greys',vmin=0,vmax=1,alpha=.35)
        ax.plot(prep[:,0],prep[:,1],color='#9ca3af',linewidth=1.6,label='Preparation')
        ax.plot(xy[:,0],xy[:,1],color='#2563eb',linewidth=1.7,label='Recovery')
        if west.any():ax.scatter(xy[west,0],xy[west,1],s=12,color='#ea580c',label='Backtracking',zorder=3)
        ax.scatter(*xy[0],marker='o',s=35,color='#2563eb',zorder=4)
        ax.scatter(*layout['goal_pos'][:2],marker='*',s=85,color='#16a34a',zorder=4,label='Goal')
        ax.set(title=title,xlabel='World x (m)',xlim=(-3,18),ylim=(-5.5,5.5));ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('World y (m)');axes[0].legend(loc='upper left',fontsize=7,frameon=False)
    fig.suptitle('Training recovery routes · maze'+args.layout.rsplit('_',1)[-1],fontsize=11)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=160);plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
