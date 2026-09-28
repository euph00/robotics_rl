#!/usr/bin/env python3
"""Plot recorded validation routes; never run or relabel an experiment."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pnr.scenes import load_scene_bank,scene_records
from pnr.telemetry import RunLogger,atomic_json
p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
p.add_argument('--bank',default='configs/phase2_scene_bank.json');args=p.parse_args()
bank=load_scene_bank(args.bank);maps={r['layout_id']:r for r in scene_records(bank,'validation')}
result=json.loads(Path(args.evaluation).read_text());records=[r for r in result['records'] if r['layout_id'] in maps]
logger=RunLogger(args.output,kind='diagnostic',skill='carry',name='Navigation validation routes',summary='Recorded navigation-only evaluation; figures add no physical trials.')
out=logger.directory/'artifacts';out.mkdir()
fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True,sharey=True)
for ax,record in zip(axes.flat,records):
 layout=maps[record['layout_id']];walls=np.array(layout['maze_map']);rows,cols=np.where(walls)
 for row,col in zip(rows,cols):ax.add_patch(Rectangle((col-4,5-row),1,1,color='#d9e0e7',lw=0))
 ax.add_patch(Rectangle((-.75,-.5),1.5,1.,color='#bd9d7c',alpha=.8,lw=0))
 xy=np.array(record['trajectory']);ax.plot(xy[:,0],xy[:,1],color='#0f766e',lw=1.5)
 ax.scatter(*xy[0],color='#334155',s=15);ax.scatter(*layout['goal_pos'][:2],color='#16a34a',marker='*',s=50)
 ax.set_title(f"{record['layout_id']} · {'success' if record['success'] else 'failed'} · {record['physics_ticks']:,} navigation ticks",fontsize=10)
 ax.set(xlim=(-4,18.5),ylim=(-6,6),aspect='equal');ax.spines[['top','right']].set_visible(False)
for ax in axes[-1]:ax.set_xlabel('x (m)')
for ax in axes[:,0]:ax.set_ylabel('y (m)')
fig.suptitle('Four development-validation mazes · before online SAC',fontsize=13)
fig.tight_layout();fig.savefig(out/'validation_routes.png',dpi=150);plt.close(fig)
atomic_json(out/'source.json',{'source_evaluation':str(Path(args.evaluation).resolve()),'online_decisions':result['online_decisions'],'new_physical_interactions':0,'scope':'navigation only; not full-task evidence'})
logger.artifact('artifacts/validation_routes.png','Recorded validation routes','image');logger.artifact('artifacts/source.json','Source evaluation')
logger.finish('complete','Four recorded development-validation routes; one exceeds the timing target.')
print(out/'validation_routes.png')
