#!/usr/bin/env python3
"""TRAIN-only measured finger-speed settling teacher, including actor seating."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def plain(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def worker(task):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from pnr.general_manipulation import make_grasp_case,ContactGraspTeacher
    from pnr.telemetry import atomic_json
    obj,checkpoint,out,max_vertical_speed=task;out=Path(out);torch.set_num_threads(1)
    result={'object_idx':obj,'scene_seed':42,'training_only':True,'branches':[],'prefixes':[]}
    env=None;executed=[]
    with (out/f'object_{obj:02d}.log').open('w') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            env=make_grasp_case(obj,42,seed=71000,horizon=200,evaluation=True,goal_lift_m=.24,grasp_proposal_mode='clearance-v2')
            n=env.native;teacher=ContactGraspTeacher(env,approach_control='ray-tilted')
            model=SAC.load(checkpoint,env=env,device='cpu')
            result.update(setup_source_action_calls=env.setup_action_calls,initialization_physics_ticks=n.settle_steps)
            for prefix_kind in ['clean_teacher','frozen_actor']:
                env.reset(seed=71000,options={'full_pregrasp':True});env.goal=env.base_goal.copy()
                found=False
                for decision in range(100):
                    obs=env._observation()
                    action,_=(teacher.action(obs) if prefix_kind=='clean_teacher' else model.predict(obs,deterministic=True))
                    _,_,terminated,truncated,info=env.step(action)
                    if info['bilateral_contact']:found=True;break
                    if terminated or truncated:break
                prefix={'kind':prefix_kind,'first_contact':found,'source_action_calls':n.step_count,
                        'physics_ticks':n.physics_steps,'decision':decision+1}
                result['prefixes'].append(prefix)
                if not found:continue
                snapshot=n.save_state();contact_z=float(n.observe()['object_pos'][2])
                for mode in ['ordinary_teacher','finger_velocity_gate']:
                    n.restore_state(snapshot,reset_counters=True);env._episode_steps=0;env._episode_success=False
                    row={'prefix':prefix_kind,'mode':mode,'trace':[],'success':False,'threshold_m_s':.002,'max_vertical_speed_m_s':max_vertical_speed}
                    prior_gate=False;reached=False;held=[];transitions=[]
                    for step in range(100):
                        obs=env._observation();action,phase=teacher.action(obs)
                        speed=float(np.max(np.abs(n.observe()['qvel'][n.finger_slots])))
                        bilateral=bool(n.contacts()['bilateral_contact'])
                        vertical_speed=float(n.p.getBaseVelocity(n.object_id)[0][2])
                        gate=bool(mode=='finger_velocity_gate' and bilateral and speed>.002
                                  and (max_vertical_speed is None or abs(vertical_speed)<max_vertical_speed))
                        rise=float(n.observe()['object_pos'][2])-contact_z
                        if gate:action=np.array([0.,0.,0.,-1.],dtype=np.float32);phase='settle_finger_velocity'
                        nxt,reward,terminated,truncated,info=env.step(action)
                        row['trace'].append({'decision':step+1,'gate':gate,'phase':phase,'finger_speed_before_m_s':speed,
                            'object_vertical_speed_before_m_s':vertical_speed,'rise_before_m':rise,'gate_transition':bool(gate!=prior_gate),'bilateral_before':bilateral,
                            'action_rejected':bool(info['action_rejected']),'goal_distance':float(info['goal_distance'])})
                        transitions.append({'observation':obs,'action':action.copy(),'next_observation':nxt,'reward':reward,
                            'terminated':terminated,'truncated':truncated,'info':info,'teacher_phase':phase})
                        prior_gate=gate
                        reached=bool(info['is_success'] and not terminated and not n.drop_events)
                        if reached:
                            q=n.agent_qpos.copy();q[n.finger_slots]=0.
                            for _ in range(50):state,_=n.step(q);held.append(float(state['object_pos'][2]))
                            break
                        if terminated or truncated or n.drop_events:break
                    metrics=n.evaluate()
                    row.update(success=bool(reached and metrics['bilateral_contact'] and metrics['lifted']
                        and metrics['stable_hold_steps']>=50 and not metrics['drop_events'] and np.ptp(held)<.02),
                        source_action_calls=n.step_count,physics_ticks=n.physics_steps,metrics=metrics,
                        triggered_decisions=sum(t['gate'] for t in row['trace']),
                        late_triggered_decisions=sum(t['gate'] and t['rise_before_m']>.02 for t in row['trace']),
                        gate_switches_after_2cm_rise=sum(t['gate_transition'] and t['rise_before_m']>.02 for t in row['trace']))
                    result['branches'].append(row)
                    executed.append({'prefix':prefix_kind,'mode':mode,'transitions':transitions,'summary':row})
                    with (out/f'object_{obj:02d}_executed.pkl').open('wb') as handle:pickle.dump(executed,handle)
                    atomic_json(out/f'object_{obj:02d}.json',json.loads(json.dumps(result,default=plain)))
        except Exception as error:
            result['error']=repr(error);result['failed_setup_costs']=getattr(error,'setup_accounting',{})
            if env is not None:result['last_native_counter_at_error']=env.native.step_count
        finally:
            if env is not None:env.close()
            with (out/f'object_{obj:02d}_result.pkl').open('wb') as backup:pickle.dump(result,backup)
            result=json.loads(json.dumps(result,default=plain));atomic_json(out/f'object_{obj:02d}.json',result)
    return result


def main():
    from pnr.telemetry import RunLogger,atomic_json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--max-vertical-speed',type=float,default=None)
    args=parser.parse_args()
    if args.max_vertical_speed is not None and args.max_vertical_speed<=0:raise ValueError('Velocity threshold must be positive')
    objects=[14,5,2,7,10]
    logger=RunLogger(args.run_id,kind='validation',skill='grasp',budget=20,
        name='TRAIN observable settling teacher probe',algorithm='Diagnostic measured finger-speed teacher',
        summary='Five frozen TRAIN objects, clean/learned first-contact seating, ordinary versus velocity-gated teacher; no inference changes.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(logger.directory/'config.json',{'objects':objects,'scene_seed':42,'checkpoint':str(args.checkpoint),
        'grasp_proposal_mode':'clearance-v2','gate_m_s':.002,'max_vertical_speed_m_s':args.max_vertical_speed,'no_fixed_timer':True,'no_production_inference_change':True})
    rows=[]
    try:
        with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(worker,(obj,str(args.checkpoint),str(out),args.max_vertical_speed)) for obj in objects]):
                rows.append(future.result());branches=[b for r in rows for b in r['branches']]
                atomic_json(out/'summary.json',{'cases':rows,'training_only':True,'production_teacher_changed':False,
                    'source_action_calls':sum(r.get('setup_source_action_calls',0)+sum(p['source_action_calls'] for p in r['prefixes'])+sum(b['source_action_calls'] for b in r['branches']) for r in rows),
                    'initialization_physics_ticks':sum(r.get('initialization_physics_ticks',0) for r in rows)})
                logger.metric(len(branches),scope='training_teacher_diagnostic',branches=len(branches),
                    success_rate=sum(b['success'] for b in branches)/max(1,len(branches)))
        logger.artifact('artifacts/summary.json','All teacher branches, settling triggers, chatter and physical costs')
        logger.finish(summary='Bounded TRAIN velocity-gate test completed; every executed transition preserved for audit only.')
    except BaseException as error:logger.finish('failed',repr(error));raise

if __name__=='__main__':main()
