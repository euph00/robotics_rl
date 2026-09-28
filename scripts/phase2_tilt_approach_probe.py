#!/usr/bin/env python3
"""Counted TRAIN can42001 approach-stall diagnosis; no production edits."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pnr.general_manipulation import make_grasp_case,ContactGraspTeacher
from pnr.telemetry import RunLogger,atomic_json


def main():
    logger=RunLogger('phase2_tilt_approach_probe01',kind='validation',skill='grasp',budget=4,
        name='Tilted TRAIN approach conditioning',algorithm='Physical teacher diagnostic',
        summary='Exact recorded failed perturbations; compare Cartesian component/ray controls with measured IK rejection causes.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    source=json.loads(Path('runs/phase2_grasp_demos05/artifacts/object_02_pointseed_42001.json').read_text())
    env=make_grasp_case(2,42001,seed=61000,horizon=100,evaluation=True,grasp_proposal_mode='clearance-v2')
    teacher=ContactGraspTeacher(env);results=[]
    try:
        for ep in [e for e in source['episodes'] if not e['successful_physical_lift_and_hold']]:
            env.reset(seed=ep['seed'],options={'full_pregrasp':True})
            ee,orientation=env.controller.planner.ee_pose()
            before=env.native.step_count
            env.controller.move_ee(ee+ep['start_jitter_m'],orientation,stage='training_start_perturbation',closed=False)
            perturbation=env.native.step_count-before;snapshot=env.native.save_state()
            for variant in ['component','approach_ray']:
                env.native.restore_state(snapshot,reset_counters=True)
                env._episode_steps=0;env._episode_success=False
                env.goal=np.array(ep['desired_goal']);obs=env._observation();trace=[]
                for decision in range(100):
                    action,phase=teacher.action(obs)
                    if variant=='approach_ray' and not env.native.contacts()['bilateral_contact']:
                        delta=env.measured_grasp_target()-env.controller.planner.ee_pose()[0]
                        action[:3]=delta/max(env.delta_scale,np.linalg.norm(delta)/teacher.max_action)
                    ee,_=env.controller.planner.ee_pose();requested=ee+env.delta_scale*action[:3]
                    requested[:2]=np.clip(requested[:2],env.base_xy+[.10,-.28],env.base_xy+[.95,.28])
                    requested[2]=np.clip(requested[2],env.native.table_height+.037,1.35)
                    diagnostic={}
                    if decision<5:
                        initial=env.native.agent_qpos.copy()
                        try:
                            q=env.local_ik.ik(requested,env.orientation)
                            diagnostic['ik_ok']=True
                            diagnostic['collision_free']=env.controller.planner.collision_free(q,include_object=False,margin=-.0005)
                        except ValueError as error:
                            diagnostic['ik_error']=str(error)
                            try:
                                q=env.controller.planner.ik(requested,env.orientation,seed=initial,tolerance=.0015)
                                diagnostic['global_delta']=np.abs(q[env.local_ik.active]-initial[env.local_ik.active]).tolist()
                                diagnostic['global_collision_free']=env.controller.planner.collision_free(q,include_object=False,margin=-.0005)
                            except ValueError as global_error:diagnostic['global_error']=str(global_error)
                    obs,_,terminated,truncated,info=env.step(action)
                    trace.append({'decision':decision+1,'action':action.tolist(),'phase':phase,'diagnostic':diagnostic,
                                  'rejected':info['action_rejected'],'goal_distance':info['goal_distance'],'bilateral':info['bilateral_contact']})
                    if terminated or truncated:break
                metrics=env.native.evaluate()
                success=bool(info['goal_distance']<env.tolerance and metrics['lifted'] and metrics['bilateral_contact'] and metrics['stable_hold_steps']>=50 and not metrics['drop_events'])
                row={'episode':ep['episode'],'variant':variant,'success':success,'trace':trace,'metrics':metrics,
                     'perturbation_calls':perturbation,'source_action_calls':env.native.step_count,'physics_ticks':env.native.physics_steps}
                results.append(row);atomic_json(out/'summary.json',{'results':results,'setup_action_calls':env.setup_action_calls,'initialization_physics_ticks':env.native.settle_steps})
                logger.metric(sum(len(r['trace']) for r in results),scope='tilt_approach',eval_episodes=len(results),success_rate=sum(r['success'] for r in results)/len(results))
                print(json.dumps({'episode':ep['episode'],'variant':variant,'success':success,'first_steps':trace[:5]}),flush=True)
        logger.artifact('artifacts/summary.json','Measured rejected target and physical ray approach comparison')
        logger.finish(summary='Four counted TRAIN approach probes complete; no production teacher/control edits.')
    finally:env.close()

if __name__=='__main__':main()
