#!/usr/bin/env python3
"""Recheck three already exposed development failures with integrated bounded IK."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def probe(config):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from pnr.env import PickNavReachEnv
    from pnr.reference import ReferenceController
    from pnr.general_manipulation import GeneralGraspEnv
    from pnr.telemetry import atomic_json
    torch.set_num_threads(1)
    native=PickNavReachEnv(**config['env_kwargs'])
    counter={'calls':0,'wall_contact_calls':0,'robot_table_contact_calls':0}
    original=native.step
    def audited(action):
        result=original(action);counter['calls']+=1
        points=native.p.getContactPoints(native.robot_id,native.maze_id)+native.p.getContactPoints(native.object_id,native.maze_id)
        counter['wall_contact_calls']+=int(any(c[9]>0 for c in points))
        counter['robot_table_contact_calls']+=int(any(c[9]>0 for c in native.p.getContactPoints(native.robot_id,native.table_id)))
        return result
    native.step=audited
    results=[]
    try:
        controller=ReferenceController(native,seed=native.seed,speed_scale=1.)
        controller.prepare_grasp()
        setup=counter.copy();snapshot=native.save_state()
        for lift in [.20,.24]:
            native.restore_state(snapshot,reset_counters=True)
            env=GeneralGraspEnv.bind(native,controller,horizon=150)
            env.goal[2]+=lift-.20
            model=SAC.load(config['checkpoint'],env=env,device='cpu')
            before=counter.copy();obs=env._observation();trace=[];held=[];reached=False
            for decision in range(150):
                action,_=model.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action)
                trace.append({'decision':decision+1,'action':action.tolist(),'goal_distance':info['goal_distance'],
                              'action_rejected':info['action_rejected'],'bilateral_contact':info['bilateral_contact'],
                              'table_clearance':info['table_clearance'],'fallback_used':env.local_ik.last_used_fallback})
                reached=bool(info['is_success'] and not terminated and not native.drop_events)
                if reached:
                    q=native.agent_qpos.copy();q[native.finger_slots]=0.
                    for _ in range(50):
                        state,_=native.step(q);held.append(float(state['object_pos'][2]))
                    break
                if terminated or truncated or native.drop_events:break
            metrics=native.evaluate();counts={key:counter[key]-before[key] for key in counter}
            result={'case':config['case'],'goal_lift_m':lift,'success':bool(reached and metrics['lifted'] and metrics['bilateral_contact']
                      and metrics['stable_hold_steps']>=50 and not metrics['drop_events'] and np.ptp(held)<.02),
                    'metrics':metrics,'trace':trace,'accounting':counts|{'physics_ticks':counts['calls']*native.substeps},
                    'fallback_attempts':env.local_ik.fallback_attempts,'fallback_successes':env.local_ik.fallback_successes}
            results.append(result)
            atomic_json(Path(config['output'])/(config['case']+'.json'),{'setup':setup,'initialization_physics_ticks':native.settle_steps,'results':results})
        return {'setup':setup,'initialization_physics_ticks':native.settle_steps,'results':results}
    finally:native.close()


def main():
    from pnr.telemetry import RunLogger,atomic_json
    logger=RunLogger('phase2_grasp_failure_recheck01',kind='validation',skill='grasp',budget=6,
        name='Integrated bounded IK development recheck',algorithm='Frozen BC actor; bounded IK fix',
        summary='Three already exposed grasp failures at20/24cm goals, actual contact monitoring every native call.')
    out=logger.directory/'artifacts';out.mkdir()
    for name in ['scripts/phase2_grasp_failure_recheck.py','pnr/general_manipulation.py']:
        (out/name.replace('/','__')).write_bytes((ROOT/name).read_bytes())
    cases=[('validation_000_object_05',352965787,5,141130023),('validation_001_object_05',1532150578,5,1564340550),('validation_000_object_07',2624414159,7,141130023)]
    results=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(probe,{'case':name,'env_kwargs':{'seed':seed,'object_idx':obj,'maze_seed':maze,'seed_mode':'independent-maze-v1'},
                  'checkpoint':'runs/phase2_sac_grasp01/checkpoints/bc_initialized.zip','output':str(out)}) for name,seed,obj,maze in cases]
        for future in as_completed(futures):
            result=future.result();results.append(result);rows=[r for result in results for r in result['results']]
            atomic_json(out/'summary.json',{'results':results})
            logger.metric(sum(len(r['trace']) for r in rows),scope='integrated_ik_recheck',success_rate=sum(r['success'] for r in rows)/len(rows),eval_episodes=len(rows))
            print(json.dumps([{k:r[k] for k in ['case','goal_lift_m','success','fallback_attempts','fallback_successes','accounting']} for r in result['results']]),flush=True)
    logger.artifact('artifacts/summary.json','Integrated solver physical/contact audit')
    logger.finish(summary='Completed six frozen-actor development grasp rechecks with actual physics contact counts.')

if __name__=='__main__':main()
