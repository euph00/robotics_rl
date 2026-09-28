#!/usr/bin/env python3
"""Evaluate three actual SAC checkpoints from the original uninterrupted task start."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pnr.policies import load_policy_checkpoint
from pnr.env import PickNavReachEnv
from pnr.system import LearnedSystem
from pnr.telemetry import RunLogger


def resolve_checkpoint(value):
    path=Path(value).expanduser().resolve()
    if path.is_dir():
        if (path/'model.zip').is_file():
            return path/'model.zip'
        path=path/('latest.json' if (path/'latest.json').is_file() else 'checkpoints/latest.json')
    if path.suffix=='.json':
        manifest=json.loads(path.read_text())
        path=path.parent/manifest['generation']/'model.zip'
    if not path.is_file() or path.suffix!='.zip':
        raise ValueError(f'No model.zip checkpoint found from {value}')
    return path



def freeze_checkpoints(paths,directory):
    """Capture the exact bytes evaluated, including mutable best checkpoints."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    records={}
    for name,source in paths.items():
        destination=directory/(name+'.zip')
        digest=hashlib.sha256()
        with Path(source).open('rb') as reader,destination.open('xb') as writer:
            while chunk := reader.read(1024*1024):
                writer.write(chunk);digest.update(chunk)
        records[name]={'source':str(source),'frozen_path':str(destination.resolve()),
                       'sha256':digest.hexdigest()}
    return records


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser()
    for skill in ('grasp','carry','reach'):
        parser.add_argument('--'+skill,required=True,help='Model ZIP, generation folder, run folder, or latest.json')
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--scene-seeds',default='42',help='Comma-separated seed bank, cycled across episodes')
    parser.add_argument('--object-idx',type=int,default=14)
    parser.add_argument('--output',default=None)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--grasp-horizon',type=int,default=600)
    parser.add_argument('--carry-horizon',type=int,default=1500)
    parser.add_argument('--reach-horizon',type=int,default=150)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--algorithm-label',default='SAC+HER',
                        help='Declared training method shown in results and video, including demonstration initialization')
    args=parser.parse_args()
    if args.episodes<1: parser.error('--episodes must be positive')
    if args.torch_threads<1: parser.error('--torch-threads must be positive')
    if min(args.grasp_horizon,args.carry_horizon,args.reach_horizon)<1:
        parser.error('Skill horizons must be positive')
    import torch
    torch.set_num_threads(args.torch_threads)
    try: seeds=[int(value) for value in args.scene_seeds.split(',')]
    except ValueError: parser.error('--scene-seeds must be comma-separated integers')
    episode_seeds=[seeds[i%len(seeds)] for i in range(args.episodes)]
    paths={name:resolve_checkpoint(getattr(args,name)) for name in ('grasp','carry','reach')}
    out=Path(args.output or f'runs/system_eval_{time.strftime("%Y%m%d_%H%M%S")}')
    artifact_dir=out/'artifacts';artifact_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_records=freeze_checkpoints(paths,artifact_dir/'checkpoints')
    source_dir=artifact_dir/'source'; source_dir.mkdir()
    repository=Path(__file__).resolve().parents[1]
    source_hashes={}
    for source in list((repository/'pnr').glob('*.py'))+[Path(__file__).resolve()]:
        relative=source.relative_to(repository)
        target=source_dir/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        source_hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (source_dir/'manifest.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    logger=RunLogger(out.name,root=out.parent,kind='evaluation',name=args.algorithm_label+' · complete system',
        algorithm=args.algorithm_label,seed_bank=seeds,object_idx=args.object_idx,
        checkpoints=checkpoint_records,
        summary=f'{args.episodes} uninterrupted episodes across {len(set(episode_seeds))} declared maze seeds; geometric pregrasp and carry-posture assistance.')
    results=[];total_calls=0
    try:
        policies={name:load_policy_checkpoint(record['frozen_path'],device=args.device) for name,record in checkpoint_records.items()}
        for name,dimensions in [('grasp',4),('carry',2),('reach',3)]:
            if policies[name].action_space.shape!=(dimensions,):
                raise ValueError(f'{name} checkpoint has the wrong action interface')
        for episode in range(args.episodes):
            env=PickNavReachEnv(seed=episode_seeds[episode],object_idx=args.object_idx)
            writer=None
            trace_path=artifact_dir/f'episode_{episode:03d}_trajectory.jsonl'
            trace=trace_path.open('w')
            if args.video:
                import imageio.v2 as imageio
                writer=imageio.get_writer(artifact_dir/f'episode_{episode:03d}.mp4',fps=12,codec='libx264',quality=7,macro_block_size=8)
            last_stage=None
            def observe(system,observation,info):
                nonlocal last_stage
                stage=system.display_stage
                if stage != last_stage:
                    logger.update(stage=stage,episode=episode+1)
                    logger.event(f'Episode {episode+1}: {stage}')
                    last_stage=stage
                if env.step_count%5==0:
                    trace.write(json.dumps({'source_action':env.step_count,'physics_tick':env.physics_steps,
                        'stage':stage,'controller':system.control_source,
                        'qpos':observation['qpos'].tolist(),'object_position':observation['object_pos'].tolist(),
                        'object_quaternion':observation['object_xyzw'].tolist(),
                        'goal_distance':info['dist_to_goal'],'bilateral_contact':info['bilateral_contact']})+'\n')
                if env.step_count%100==0:
                    logger.metric(transitions=total_calls+env.step_count,goal_distance=info['dist_to_goal'],
                                  physics_ticks=sum(r['physics_ticks'] for r in results)+env.physics_steps)
                if writer and env.step_count%4==0:
                    from PIL import Image,ImageDraw,ImageFont
                    camera='grasp' if stage in ('pregrasp','grasp','pick_hold') else 'follow'
                    frame=env.render_rgb(camera=camera,width=640,height=360,shadow=False,
                        pitch=-78 if stage in ('carry','reach','final_hold') else None)
                    canvas=Image.fromarray(frame);draw=ImageDraw.Draw(canvas)
                    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
                    except OSError: font=ImageFont.load_default()
                    draw.rectangle((0,0,640,30),fill=(15,23,42))
                    text=f'{args.algorithm_label} | {stage.replace("_"," ")} | {env.physics_steps*env.physics_timestep:.1f} s | goal {info["dist_to_goal"]:.2f} m'
                    draw.text((12,8),text,fill='white',font=font);writer.append_data(np.asarray(canvas))
            try:
                system=LearnedSystem(env,policies,grasp_horizon=args.grasp_horizon,
                    carry_horizon=args.carry_horizon,reach_horizon=args.reach_horizon,on_step=observe)
                result=system.run();result['episode']=episode
                if env.step_count%5:
                    final_observation=env.observe();final_info=env.evaluate()
                    trace.write(json.dumps({'source_action':env.step_count,'physics_tick':env.physics_steps,
                        'stage':system.display_stage,'controller':system.control_source,
                        'qpos':final_observation['qpos'].tolist(),
                        'object_position':final_observation['object_pos'].tolist(),
                        'object_quaternion':final_observation['object_xyzw'].tolist(),
                        'goal_distance':final_info['dist_to_goal'],
                        'bilateral_contact':final_info['bilateral_contact']})+'\n')
                result['algorithm']=args.algorithm_label
                result['initialization_physics_ticks']=env.settle_steps
                result['total_physics_ticks_including_initialization']=env.physics_steps+env.settle_steps
                result['checkpoints']=checkpoint_records
                result['source_hashes']=source_hashes
                results.append(result);total_calls+=env.step_count
                filename=f'episode_{episode:03d}.json'
                (artifact_dir/filename).write_text(json.dumps(result,indent=2,default=plain)+'\n')
                logger.artifact('artifacts/'+filename,f'Episode {episode+1} result')
                trace.flush()
                from pnr.figures import measured_route
                route_name=f'episode_{episode:03d}_route.png'
                measured_route(trace_path,result['scene_manifest'],env.cube_positions,
                    env.root_position[:2],artifact_dir/route_name,
                    title=f'Episode {episode+1} · '+('complete task passed' if result['success']
                        else 'failed at '+result.get('failed_stage','final check')))
                logger.artifact('artifacts/'+route_name,f'Episode {episode+1} measured route',kind='image')
                logger.metric(transitions=total_calls,scope='full_task',
                    success_rate=float(np.mean([item['success'] for item in results])),eval_episodes=len(results),
                    goal_distance=result['metrics']['dist_to_goal'],
                    drop_events=sum(item['metrics']['drop_events'] for item in results),
                    physics_ticks=sum(item['physics_ticks'] for item in results))
                print(json.dumps({'episode':episode,'success':result['success'],
                    'failed_stage':result.get('failed_stage'),'error':result.get('error'),
                    'goal_distance':result['metrics']['dist_to_goal']},default=plain),flush=True)
            finally:
                trace.close()
                if writer:
                    writer.close()
                    from pnr.video import webm_copy
                    movie=artifact_dir/f'episode_{episode:03d}.mp4'
                    portable=webm_copy(movie)
                    imageio.imwrite(artifact_dir/f'episode_{episode:03d}_preview.png',
                        env.render_rgb(camera='follow',width=640,height=360,shadow=False,pitch=-78))
                    logger.artifact('artifacts/'+portable.name,f'Episode {episode+1} rollout',kind='video')
                    logger.artifact('artifacts/'+movie.name,f'Episode {episode+1} MP4 download')
                    logger.artifact(f'artifacts/episode_{episode:03d}_preview.png',f'Episode {episode+1} physical final frame',kind='image')
                env.close()
        summary={'scope':'full_task','algorithm':args.algorithm_label,'episodes':len(results),
                 'successes':sum(item['success'] for item in results),
                 'success_rate':float(np.mean([item['success'] for item in results])),
                 'scene_seeds':seeds,'episode_seeds':episode_seeds,
                 'distinct_maze_hashes':len({item['scene_manifest']['maze_hash'] for item in results}),
                 'object_idx':args.object_idx,'source_action_calls':total_calls,
                 'physics_ticks':sum(item['physics_ticks'] for item in results),
                 'initialization_physics_ticks':sum(item['initialization_physics_ticks'] for item in results),
                 'total_physics_ticks_including_initialization':sum(item['total_physics_ticks_including_initialization'] for item in results),
                 'checkpoints':checkpoint_records,'source_hashes':source_hashes}
        (artifact_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        logger.artifact('artifacts/summary.json','Evaluation summary')
        logger.finish(summary=f'{summary["successes"]}/{len(results)} uninterrupted complete-task successes.')
    except KeyboardInterrupt:
        logger.finish('stopped','Evaluation interrupted; completed episodes retained.')
        raise
    except BaseException as exc:
        logger.finish('failed',str(exc));raise


if __name__=='__main__': main()
