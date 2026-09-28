#!/usr/bin/env python3
"""Run the real-contact geometric reference and save a truthful result/video."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pnr.env import PickNavReachEnv
from pnr.reference import ReferenceController
from pnr.telemetry import RunLogger


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--object-idx',type=int,default=14)
    parser.add_argument('--stop-after',choices=['pregrasp','pick','carry_posture','reach'],default='reach')
    parser.add_argument('--output',default=None)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--speed-scale',type=float,default=1.)
    args=parser.parse_args()
    out=Path(args.output or f'runs/reference_{time.strftime("%Y%m%d_%H%M%S")}')
    out.mkdir(parents=True,exist_ok=True)
    logger=RunLogger(out.name,root=out.parent,kind='reference',name=f'Physical reference · object {args.object_idx} · seed {args.seed}',
                     stage='initial',seed=args.seed,object_idx=args.object_idx,
                     summary='One uninterrupted geometric reference episode; not learned-policy evaluation.')
    artifact_dir=out/'artifacts'; artifact_dir.mkdir(exist_ok=True)
    env=PickNavReachEnv(seed=args.seed,object_idx=args.object_idx)
    writer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(artifact_dir/'rollout.mp4',fps=12,codec='libx264',quality=7,macro_block_size=8)
    trace=open(out/'trajectory.jsonl','w')
    counter=0
    previous_stage=None
    def record(ctrl,obs,info):
        nonlocal counter,previous_stage
        counter+=1
        if ctrl.stage!=previous_stage:
            print(json.dumps({'stage':ctrl.stage,'action':counter,
                              'object_position':obs['object_pos'].tolist()}),flush=True)
            previous_stage=ctrl.stage
            logger.update(stage=ctrl.stage)
            logger.event(f'Stage: {ctrl.stage}')
        if counter%100==0:
            logger.metric(transitions=counter,goal_distance=float(np.linalg.norm(obs['object_pos']-env.goal_pos)),
                          physics_ticks=counter*env.substeps,drop_events=int(info['drop_events']))
        if counter%5==0:
            trace.write(json.dumps({'action':counter,'stage':ctrl.stage,
                                    'qpos':obs['qpos'].tolist(),'object_position':obs['object_pos'].tolist(),
                                    'goal_distance':float(np.linalg.norm(obs['object_pos']-env.goal_pos))})+'\n')
        if writer and counter%4==0:
            camera='grasp' if ctrl.stage in ('pregrasp','descend','close','lift','hold') else 'follow'
            frame=env.render_rgb(camera=camera,width=640,height=360,shadow=False,
                                 pitch=-78 if camera=='follow' and ctrl.stage in ('carry','reach') else None)
            from PIL import Image, ImageDraw, ImageFont
            canvas=Image.fromarray(frame); draw=ImageDraw.Draw(canvas)
            try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
            except OSError: font=ImageFont.load_default()
            draw.rectangle((0,0,640,30),fill=(15,23,42))
            distance=float(np.linalg.norm(obs['object_pos']-env.goal_pos))
            label=f'Geometric reference  |  {ctrl.stage.replace("_"," ")}  |  {counter*env.substeps*env.physics_timestep:.1f} s  |  goal {distance:.2f} m'
            draw.text((12,8),label,fill='white',font=font)
            writer.append_data(np.asarray(canvas))
    try:
        ctrl=ReferenceController(env,on_step=record,seed=args.seed,speed_scale=args.speed_scale)
        result=ctrl.run(args.stop_after)
        result.update(seed=args.seed,object_idx=args.object_idx,controller='geometric_reference',
                      environment_calls=counter,physics_ticks=counter*env.substeps)
        (out/'result.json').write_text(json.dumps(result,indent=2,default=plain)+'\n')
        import imageio.v2 as imageio
        imageio.imwrite(artifact_dir/'final.png',env.render_rgb(camera='follow',pitch=-78,shadow=False))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig,ax=plt.subplots(figsize=(10,5))
        for x,y,*_ in env.cube_positions:
            ax.add_patch(Rectangle((x-.5,y-.5),1,1,color='#e2e8f0',linewidth=0))
        if ctrl.path:
            xy=np.array(ctrl.path); ax.plot(xy[:,0],xy[:,1],color='#0f766e',linewidth=2,label='Planned carry route')
        ax.scatter(*env.root_position[:2],color='#334155',s=35,label='Original start')
        ax.scatter(*env.goal_pos[:2],color='#16a34a',s=65,marker='*',label='Goal')
        ax.add_patch(Rectangle((-.5,-.75),1,1.5,color='#c8a783',alpha=.6))
        ax.set(xlim=(-4,18.5),ylim=(-6,6),aspect='equal',xlabel='x (m)',ylabel='y (m)')
        ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,loc='lower center',ncol=3)
        fig.tight_layout(); fig.savefig(artifact_dir/'route.png',dpi=140); plt.close(fig)
        (artifact_dir/'result.json').write_text(json.dumps(result,indent=2,default=plain)+'\n')
        logger.metric(transitions=counter,scope='full_task',success_rate=float(result['success']),eval_episodes=1,
                      goal_distance=float(result['metrics']['dist_to_goal']),
                      physics_ticks=counter*env.substeps,drop_events=int(result['metrics']['drop_events']))
        logger.artifact('artifacts/result.json','Physical result')
        logger.artifact('artifacts/final.png','Final physical pose',kind='image')
        logger.artifact('artifacts/route.png','Maze route',kind='image')
        logger.finish(status='complete' if result['success'] else 'failed',
                      summary='One physical reference episode: '+('success.' if result['success'] else result.get('error',result['status'])))
        print(json.dumps(result,default=plain),flush=True)
    finally:
        trace.close()
        if writer:
            writer.close()
            from pnr.video import webm_copy
            portable=webm_copy(artifact_dir/'rollout.mp4')
            logger.artifact('artifacts/'+portable.name,'Physical reference rollout',kind='video')
            logger.artifact('artifacts/rollout.mp4','Physical reference MP4 download')
        env.close()


if __name__=='__main__': main()
