#!/usr/bin/env python3
"""Paired TRAIN-only teacher/continuous/binary closure diagnostic, no policy updates."""
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
    from scipy.spatial.transform import Rotation
    from stable_baselines3 import SAC
    from pnr.env import PickNavReachEnv
    from pnr.reference import ReferenceController
    from pnr.general_manipulation import GeneralGraspEnv,ContactGraspTeacher
    from pnr.telemetry import atomic_json
    torch.set_num_threads(1)
    native=PickNavReachEnv(seed=config['seed'],object_idx=config['object_idx'])
    results=[]
    try:
        controller=ReferenceController(native,seed=61000)
        controller.prepare_grasp();setup=native.step_count;snapshot=native.save_state()
        def geometry(env):
            ee,quat=controller.planner.ee_pose();rotation=Rotation.from_quat(quat).inv()
            obj=native.observe()['object_pos'];target=env.measured_grasp_target()-ee
            pts=native.p.getContactPoints(native.robot_id,native.object_id)
            fingers=[]
            for link in native.finger_links:
                active=[c for c in pts if c[3]==link and c[9]>0]
                total=sum(c[9] for c in active)
                centroid=sum((np.asarray(c[5])*c[9] for c in active),start=np.zeros(3))/total if total else None
                fingers.append({'normal_force':total,'centroid_in_hand_frame_m':rotation.apply(centroid-ee).tolist() if centroid is not None else None})
            return {'target_delta_world_m':target.tolist(),'target_distance_m':float(np.linalg.norm(target)),
                    'object_com_in_hand_frame_m':rotation.apply(obj-ee).tolist(),
                    'measured_finger_opening_m':float(np.sum(native.agent_qpos[native.finger_slots])),
                    'fingers':fingers,'bilateral_contact':native.contacts()['bilateral_contact']}
        for mode in ['teacher','bc_continuous','bc_binary_diagnostic']:
            native.restore_state(snapshot,reset_counters=True)
            env=GeneralGraspEnv.bind(native,controller,horizon=120,goal_lift_m=.24)
            teacher=ContactGraspTeacher(env)
            model=SAC.load(config['checkpoint'],env=env,device='cpu') if mode!='teacher' else None
            obs=env._observation();trace=[];held=[];reached=False
            first_partial=first_negative=first_contact=None
            for decision in range(120):
                action,_=teacher.action(obs) if mode=='teacher' else model.predict(obs,deterministic=True)
                raw_finger=float(action[3])
                if mode=='bc_binary_diagnostic':action[3]=1. if action[3]>=0 else -1.
                before=geometry(env)
                if action[3]<.95 and first_partial is None:first_partial={'decision':decision+1,'action':action.tolist(),**before}
                if action[3]<0 and first_negative is None:first_negative={'decision':decision+1,'action':action.tolist(),**before}
                obs,_,terminated,truncated,info=env.step(action);after=geometry(env)
                trace.append({'decision':decision+1,'action':action.tolist(),'raw_actor_finger':raw_finger,'before':before,'after':after})
                if after['bilateral_contact'] and first_contact is None:first_contact=trace[-1]
                reached=bool(info['is_success'] and not terminated and not native.drop_events)
                if reached:
                    q=native.agent_qpos.copy();q[native.finger_slots]=0.
                    for _ in range(50):
                        state,_=native.step(q);held.append(float(state['object_pos'][2]))
                    break
                if terminated or truncated or native.drop_events:break
            metrics=native.evaluate()
            results.append({'mode':mode,'success':bool(reached and metrics['lifted'] and metrics['bilateral_contact'] and metrics['stable_hold_steps']>=50 and not metrics['drop_events'] and np.ptp(held)<.02),
                            'first_partial_command':first_partial,'first_negative_command':first_negative,'first_bilateral_contact':first_contact,
                            'handover_geometry':geometry(env),'metrics':metrics,'trace':trace,
                            'source_action_calls':native.step_count,'physics_ticks':native.physics_steps})
            atomic_json(Path(config['output'])/f"object_{config['object_idx']:02d}_seed_{config['seed']}.json",
                        {'case':config,'setup_source_action_calls':setup,'initialization_physics_ticks':native.settle_steps,'results':results})
        return {'case':config,'setup_source_action_calls':setup,'initialization_physics_ticks':native.settle_steps,'results':results}
    finally:native.close()


def main():
    from pnr.telemetry import RunLogger,atomic_json
    logger=RunLogger('phase2_grasp_closure_probe01',kind='validation',skill='grasp',budget=12,
        name='Paired TRAIN closure diagnostic',algorithm='Frozen BC and physical teacher comparison',
        summary='Same physical pregrasp snapshots; binary closure is diagnostic only, with unchanged XYZ actor.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(probe,{'object_idx':obj,'seed':seed,'checkpoint':'runs/phase2_sac_grasp01/checkpoints/bc_initialized.zip','output':str(out)}) for obj,seed in [(2,42),(14,42),(10,42),(2,42001)]]
        for future in as_completed(futures):
            result=future.result();results.append(result);rows=[r for result in results for r in result['results']]
            atomic_json(out/'summary.json',{'results':results})
            logger.metric(sum(len(r['trace']) for r in rows),scope='paired_training_closure',success_rate=sum(r['success'] for r in rows)/len(rows),eval_episodes=len(rows))
            print(json.dumps({'case':result['case'],'results':[{k:r[k] for k in ['mode','success','first_negative_command','handover_geometry']} for r in result['results']]}),flush=True)
    logger.artifact('artifacts/summary.json','Physical closure traces and contact centroids')
    logger.finish(summary='Completed12 paired TRAIN closure probes; binary mapping remains diagnostic only.')

if __name__=='__main__':main()
