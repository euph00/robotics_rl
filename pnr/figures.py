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
