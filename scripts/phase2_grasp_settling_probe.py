#!/usr/bin/env python3
"""TRAIN-only equivalence of zero-XYZ close actions and native joint settling."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC
from pnr.general_manipulation import make_grasp_case
from pnr.telemetry import RunLogger,atomic_json


def main():
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    logger=RunLogger(args.run_id,kind='validation',skill='grasp',budget=9,
        name='TRAIN grasp settling equivalence',algorithm='Frozen actor physical diagnostic',
        summary='Three fixed TRAIN apple starts, shared first-contact snapshots, no training or production pauses.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    for seed in [42,42001,42002]:
        env=None;case={'object_idx':7,'scene_seed':seed,'training_only':True,'branches':[]}
        try:
            env=make_grasp_case(7,seed,seed=71000,horizon=180,evaluation=True,goal_lift_m=.24,grasp_proposal_mode='legacy')
            n=env.native;c=env.controller;model=SAC.load(args.checkpoint,env=env,device='cpu')
            case['setup_source_action_calls']=env.setup_action_calls
            case['initialization_physics_ticks']=n.settle_steps
            env.reset(seed=71000,options={'full_pregrasp':True});env.goal=env.base_goal.copy()
            def geometry():
                state=n.observe();ee,quat=c.planner.ee_pose();rotation=Rotation.from_quat(quat).inv()
                points=n.p.getContactPoints(n.robot_id,n.object_id);fingers=[]
                for link in n.finger_links:
                    selected=[p for p in points if p[3]==link and p[9]>0];force=sum(p[9] for p in selected)
                    center=sum((np.asarray(p[6])*p[9] for p in selected),start=np.zeros(3))/force if force else None
                    fingers.append({'force':force,'centroid_hand_m':rotation.apply(center-ee).tolist() if center is not None else None})
                linear,_=n.p.getBaseVelocity(n.object_id)
                return {'object_position':state['object_pos'].tolist(),'qpos':state['qpos'].tolist(),
                    'finger_positions':state['qpos'][n.finger_slots].tolist(),
                    'finger_velocities':state['qvel'][n.finger_slots].tolist(),
                    'object_linear_speed':float(np.linalg.norm(linear)),
                    'object_com_hand_m':rotation.apply(state['object_pos']-ee).tolist(),
                    'target_delta_m':(env.measured_grasp_target()-ee).tolist(),
                    'bilateral_contact':n.contacts()['bilateral_contact'],'fingers':fingers}
            prefix=[]
            for decision in range(120):
                action,_=model.predict(env._observation(),deterministic=True)
                _,_,terminated,truncated,info=env.step(action);prefix.append({'action':action.tolist(),'after':geometry()})
                if info['bilateral_contact']:break
                if terminated or truncated:raise RuntimeError('No first-contact snapshot available')
            else:raise RuntimeError('No bilateral contact in120 decisions')
            case['prefix']=prefix;case['prefix_source_action_calls']=n.step_count
            snapshot=n.save_state();first=geometry();case['shared_first_contact']=first
            prefix_steps=env._episode_steps
            for branch in ['immediate_lift','native_joint_hold30','zero_xyz_close10']:
                n.restore_state(snapshot,reset_counters=True);env._episode_steps=prefix_steps;env._episode_success=False
                row={'branch':branch,'start':geometry(),'settling_trace':[],'success':False}
                assert np.max(np.abs(np.array(row['start']['qpos'])-first['qpos']))<1e-10
                try:
                    if branch=='native_joint_hold30':
                        target=n.agent_qpos.copy();target[n.finger_slots]=0.
                        for _ in range(30):n.step(target);row['settling_trace'].append(geometry())
                    elif branch=='zero_xyz_close10':
                        for _ in range(10):
                            _,_,terminated,truncated,info=env.step(np.array([0.,0.,0.,-1.],dtype=np.float32))
                            row['settling_trace'].append(geometry()|{'action_rejected':info['action_rejected']})
                            if terminated or truncated:raise RuntimeError('Settling action terminated')
                    row['after_settling']=geometry();reached=False
                    for _ in range(120):
                        action,_=model.predict(env._observation(),deterministic=True)
                        _,_,terminated,truncated,info=env.step(action)
                        if info['is_success'] and not terminated and not n.drop_events:reached=True;break
                        if terminated or truncated or n.drop_events:break
                    if reached:
                        target=n.agent_qpos.copy();target[n.finger_slots]=0.
                        for _ in range(50):n.step(target)
                        row['after_pick_hold']=geometry();row['lift_success']=bool(n.contacts()['bilateral_contact'] and not n.drop_events)
                        c.prepare_carry();row['after_carry_posture']=geometry()
                        row['success']=bool(n.contacts()['bilateral_contact'] and not n.drop_events)
                except Exception as error:row['error']=repr(error)
                row.update(source_action_calls=n.step_count,physics_ticks=n.physics_steps,metrics=n.evaluate())
                case['branches'].append(row)
                atomic_json(out/f'apple_{seed}.json',case)
        except Exception as error:
            case['error']=repr(error)
            case['failed_setup_costs']=getattr(error,'setup_accounting',{})
            if env is not None:case['unbranched_source_action_calls']=env.native.step_count
        finally:
            if env is not None:env.close()
        results.append(case);rows=[b for c in results for b in c['branches']]
        atomic_json(out/'summary.json',{'checkpoint':str(args.checkpoint),'cases':results,
            'scope':'TRAIN_first_contact_through_planned_carry_posture','full_task_success_claim':False,
            'accounting':'Count each setup and shared approach prefix once, then all executed branch calls; snapshots are physical.'})
        logger.metric(len(rows),scope='training_diagnostic',branches=len(rows),success_rate=sum(r['success'] for r in rows)/max(1,len(rows)))
    logger.artifact('artifacts/summary.json','Settling action equivalence and measured contact geometry')
    logger.finish(summary='TRAIN-only settling branches complete; no dataset or production action changes.')
    print(json.dumps([(c['scene_seed'],[(b['branch'],b['success']) for b in c['branches']]) for c in results]))

if __name__=='__main__':main()
