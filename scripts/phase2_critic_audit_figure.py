#!/usr/bin/env python3
"""Draw a concise figure from recorded offline critic probes; no model execution."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=Path('runs/phase2_critic_return_audit01/artifacts/result.json'))
    p.add_argument('--warmup',type=Path,default=Path('runs/phase2_critic_warmup_probe02/artifacts/result.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit=json.loads(a.audit.read_text());warm=json.loads(a.warmup.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
    colors=['#3c73a8','#da8243']
    for c,color in zip(audit['checkpoints'],colors):
        x=[];q=[];reference=[]
        for lo,hi in ((0,20),(20,100),(100,200),(200,10000)):
            rows=[r for r in c['records'] if lo<r['remaining_decisions']<=hi]
            x.append(np.median([r['remaining_decisions'] for r in rows]));q.append(np.median([r['q_expert'] for r in rows]));reference.append(np.median([r['teacher_return_with_stationary_goal'] for r in rows]))
        label='Initialized critic' if c['name']=='initialized' else 'After 4k decisions'
        axes[0].plot(x,q,'o-',color=color,label=label,lw=2,ms=4)
    axes[0].plot(x,reference,'o--',color='#56636b',label='Demonstrated hard return',lw=1.5,ms=3)
    axes[0].set(title='Long-route value estimates',xlabel='Demonstrated decisions remaining',ylabel='Q or discounted return')
    axes[0].legend(frameon=False,fontsize=8)
    baseline=warm['fresh_critic_probe']['remaining_decision_bins']['201_to_10000_remaining_decisions']['q_minus_teacher_return']['median']
    for arm,color in zip(warm['arms'],['#3c73a8','#339578','#b06eaf']):
        x=[0];gap=[baseline]
        for probe in arm['probes']:
            x.append(probe['critic_warmup_updates']);gap.append(probe['remaining_decision_bins']['201_to_10000_remaining_decisions']['q_minus_teacher_return']['median'])
        axes[1].plot(x,gap,'o-',color=color,label=f"Warmup tau {arm['warmup_tau']:g}",lw=2,ms=4)
    axes[1].axhline(0,color='#a0a7ad',lw=.8)
    axes[1].set(title='Frozen-actor warmup',xlabel='Critic updates',ylabel='Median Q minus hard return')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Critic estimates on successful training routes',fontsize=13)
    fig.supxlabel('Hard-return reference uses demonstrated actions; it is not exact soft-Q ground truth.',fontsize=8,color='#56636b')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);plt.close(fig)


if __name__=='__main__':main()
