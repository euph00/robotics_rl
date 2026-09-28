"""Small static figures from measured physical trajectories."""
from pathlib import Path
import json
import numpy as np


def measured_route(trace_path, scene, walls, root_xy, destination, *, title):
    """Plot measured base XY; never substitute a planner path for execution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rows=[json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
    if not rows:
        return None
    xy=np.asarray([row['qpos'][:2] for row in rows])+np.asarray(root_xy)
    fig,ax=plt.subplots(figsize=(10,4.4))
    for x,y,*_ in walls:
        ax.add_patch(Rectangle((x-.5,y-.5),1,1,color='#e2e8f0',linewidth=0))
    ax.add_patch(Rectangle((-.5,-.75),1,1.5,color='#c8a783',alpha=.6))
    ax.plot(xy[:,0],xy[:,1],color='#0f766e',linewidth=1.8,label='Measured base route')
    ax.scatter(*np.asarray(root_xy),color='#334155',s=25,label='Original start',zorder=3)
    ax.scatter(*xy[-1],color='#0f766e',s=35,marker='s',label='Final base',zorder=3)
    ax.scatter(*scene['goal_pos'][:2],color='#d97706',s=65,marker='*',label='Object goal',zorder=3)
    ax.set(xlim=(-4,18.5),ylim=(-6,6),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='lower center',ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig(destination,dpi=150);plt.close(fig)
    return str(destination)


def system_matrix(results, cases, destination, *, title='Full-system validation'):
    """Compact object×maze outcomes; timing includes initialization settling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    names={14:'Foam brick',5:'Banana',2:'Tomato can',7:'Apple',10:'Mug'}
    objects=list(dict.fromkeys(case['object_idx'] for case in cases))
    layouts=list(dict.fromkeys(case['case_id'].rsplit('_object_',1)[0] for case in cases))
    by_case={r['case_id']:r for r in results}
    colors=['#e2e8f0','#fbbf24','#99d5bb'];values=np.zeros((len(objects),len(layouts)))
    labels={};successes=timed=0
    for iy,obj in enumerate(objects):
        for ix,layout in enumerate(layouts):
            row=by_case.get(f'{layout}_object_{obj:02d}')
            if row is None:labels[iy,ix]='pending';continue
            if row.get('success'):
                successes+=1;on_time=bool(row.get('success_within_deadline_including_initialization'))
                timed+=int(on_time);values[iy,ix]=2 if on_time else 1
                labels[iy,ix]='pass' if on_time else 'late'
            else:
                error=row.get('error','') or ''
                stage=row.get('failed_stage','')
                labels[iy,ix]=('grip' if 'grasp' in error and stage!='grasp' else
                    'route' if stage=='carry' and 'horizon' in error else
                    'contact' if 'collision' in error else 'pick' if stage=='grasp' else
                    'reach' if stage in ('reach','final_hold') else 'setup')
    fig,ax=plt.subplots(figsize=(max(6.,len(layouts)*.72+1.8),len(objects)*.62+1.7))
    ax.imshow(values,cmap=ListedColormap(colors),vmin=0,vmax=2,aspect='auto')
    for (iy,ix),label in labels.items():ax.text(ix,iy,label,ha='center',va='center',fontsize=9,color='#1e293b')
    ax.set_xticks(range(len(layouts)),[name.replace('validation_','') for name in layouts],fontsize=9)
    ax.set_yticks(range(len(objects)),[names.get(obj,str(obj)) for obj in objects])
    ax.set_xlabel('Maze');ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,len(layouts)),minor=True);ax.set_yticks(np.arange(-.5,len(objects)),minor=True)
    ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle(f'{title} · {successes}/{len(cases)} complete · {timed}/{len(cases)} within 15,000 ticks',fontsize=12)
    ax.legend(handles=[Patch(facecolor=colors[2],label='Success within deadline'),
        Patch(facecolor=colors[1],label='Success after deadline'),Patch(facecolor=colors[0],label='Failed / pending')],
        loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(destination,dpi=150,bbox_inches='tight');plt.close(fig)
    return str(destination)
