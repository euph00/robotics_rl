#!/usr/bin/env python3
"""Locate the physical/kinematic cause of a frozen grasp action rejection."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import torch
from stable_baselines3 import SAC
from pnr.general_manipulation import make_grasp_case
from pnr.telemetry import RunLogger,atomic_json


def main():
    torch.set_num_threads(1)
    logger=RunLogger('phase2_grasp_ik_probe02',kind='validation',skill='grasp',name='Banana local IK rejection diagnosis',
                     algorithm='Frozen actor and read-only collision queries',budget=120,
                     summary='Locate the first action rejection in a previously used validation point seed; no training.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    env=make_grasp_case(5,43001,seed=71000,horizon=120,evaluation=True)
    model=SAC.load(ROOT/'runs/phase2_sac_grasp01/checkpoints/bc_initialized.zip',env=env,device='cpu')
    try:
        env.reset(seed=71000,options={'full_pregrasp':True});env.goal=env.base_goal.copy()
        observation=env._observation();records=[]
        for decision in range(120):
            action,_=model.predict(observation,deterministic=True)
            ee,_=env.controller.planner.ee_pose();requested=ee+env.delta_scale*action[:3]
            requested[:2]=np.clip(requested[:2],env.base_xy+[.10,-.28],env.base_xy+[.95,.28])
            requested[2]=np.clip(requested[2],env.native.table_height+.037,1.35)
            record={'decision':decision+1,'ee':ee.tolist(),'requested':requested.tolist(),
                    'target_minus_ee':observation['observation'][50:53].tolist(),
                    'action':action.tolist(),'qpos':env.native.agent_qpos.tolist()}
            rejected=False
            try:
                target=env.local_ik.ik(requested,env.orientation)
                free=env.controller.planner.collision_free(target,include_object=False,margin=-.0005)
                record['target_qpos']=target.tolist();record['local_ik_error']=env.local_ik.last_ik_error
                if not free:
                    rejected=True;record['rejection']='collision'
                    points=[]
                    with env.controller.planner.query():
                        env.controller.planner._put(target);env.native.p.performCollisionDetection()
                        for name,body in [('table',env.native.table_id),('maze',env.native.maze_id)]:
                            for contact in env.native.p.getClosestPoints(env.native.robot_id,body,distance=.002):
                                link=contact[3]
                                link_name=env.native.p.getJointInfo(env.native.robot_id,link)[12].decode() if link>=0 else 'root'
                                points.append({'body':name,'link':link,'link_name':link_name,'signed_distance':contact[8],
                                               'robot_point':list(contact[5]),'obstacle_point':list(contact[6])})
                    record['closest_points']=points
            except ValueError as error:
                rejected=True;record['rejection']='local_ik';record['error']=str(error)
                from pnr.local_control import LocalCartesianIK
                trials=[]
                for damping,bound,iterations in [(.005,.35,48),(.003,.35,64),(.003,.7,64),(.001,.7,96)]:
                    solver=LocalCartesianIK(env.native,max_iterations=iterations,damping=damping,max_joint_change=bound)
                    trial={'damping':damping,'joint_change_bound':bound,'iterations':iterations}
                    try:
                        q=solver.ik(requested,env.orientation)
                        trial.update(success=True,max_joint_change=float(np.max(np.abs(q-env.native.agent_qpos))),
                                     collision_free=env.controller.planner.collision_free(q,margin=-.0005),qpos=q.tolist())
                    except ValueError as err:trial.update(success=False,error=str(err))
                    trials.append(trial)
                try:
                    q=env.controller.planner.ik(requested,env.orientation,tolerance=.0015)
                    trials.append({'method':'global_ik','success':True,'max_joint_change':float(np.max(np.abs(q-env.native.agent_qpos))),
                                   'collision_free':env.controller.planner.collision_free(q,margin=-.0005),'qpos':q.tolist()})
                except ValueError as err:trials.append({'method':'global_ik','success':False,'error':str(err)})
                record['read_only_solver_trials']=trials
            observation,_,terminated,truncated,info=env.step(action)
            record['actual_rejected']=info['action_rejected'];records.append(record)
            if rejected or terminated or truncated:break
        result={'records':records,'setup_action_calls':env.setup_action_calls,
                'actual_probe_action_calls':env.native.step_count,'actual_probe_physics_ticks':env.native.physics_steps,
                'initialization_physics_ticks':env.native.settle_steps,'grasp':env.controller.grasp.as_dict()}
        atomic_json(out/'result.json',result);logger.artifact('artifacts/result.json','First rejection geometry and measured physical state')
        logger.metric(len(records),scope='kinematic_diagnostic',rejection=records[-1].get('rejection'),
                      actual_action_calls=env.native.step_count,setup_action_calls=env.setup_action_calls)
        logger.finish(summary='First frozen banana approach rejection located and recorded; no training.')
        print(json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':main()
