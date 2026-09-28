#!/usr/bin/env python3
"""Bounded frozen-actor lift-height/settling diagnostic; real physics only."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def probe(config):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from pnr.general_manipulation import make_grasp_case
    from pnr.telemetry import atomic_json
    torch.set_num_threads(1)
    env=make_grasp_case(5, config['scene_seed'], seed=71000, horizon=120, evaluation=True, bounded_ik_fallback=False)
    model=SAC.load(config['checkpoint'], env=env, device='cpu')
    if config.get('fallback'):
        from pnr.local_control import LocalCartesianIK
        class BoundedFallback(LocalCartesianIK):
            fallback_attempts=0
            fallback_successes=0
            def ik(self,position,quaternion,seed=None,**kwargs):
                initial=np.asarray(self.env.agent_qpos if seed is None else seed,dtype=float).copy()
                try:return super().ik(position,quaternion,seed=initial,**kwargs)
                except ValueError as original:
                    self.fallback_attempts+=1
                    try:
                        q=env.controller.planner.ik(position,quaternion,seed=initial,tolerance=.0015)
                        limit=np.full(len(self.active),self.max_joint_change);limit[0]=.04
                        if np.any(np.abs(q[self.active]-initial[self.active])>limit+1e-7):raise original
                        self.fallback_successes+=1
                        return q
                    except ValueError:raise original
        env.local_ik=BoundedFallback(env.native)
    results=[]
    try:
        for lift,settle in [(0.20,0),(0.24,0),(0.26,0),(0.24,30)]:
            env.reset(seed=71000,options={'full_pregrasp':True})
            env.goal=env.base_goal.copy();env.goal[2]+=lift-.20
            if settle:
                q=env.native.agent_qpos.copy();q[env.native.finger_slots]=.05
                for _ in range(settle):env.native.step(q)
            obs=env._observation();trace=[];held=[];reached=False
            for decision in range(120):
                action,_=model.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action)
                trace.append({'decision':decision+1,'action':action.tolist(),'goal_distance':info['goal_distance'],
                              'rejected':info['action_rejected'],'object_position':env.native.observe()['object_pos'].tolist(),
                              'bilateral_contact':info['bilateral_contact'],'table_clearance':info['table_clearance']})
                reached=bool(info['is_success'] and not terminated and not env.native.drop_events)
                if reached:
                    q=env.native.agent_qpos.copy();q[env.native.finger_slots]=0.
                    for _ in range(50):
                        state,_=env.native.step(q);held.append(float(state['object_pos'][2]))
                    break
                if terminated or truncated or env.native.drop_events:break
            m=env.native.evaluate()
            results.append({'scene_seed':config['scene_seed'],'goal_lift_m':lift,'pregrasp_settle_calls':settle,
                            'success':bool(reached and m['lifted'] and m['bilateral_contact'] and m['stable_hold_steps']>=50
                                           and not m['drop_events'] and np.ptp(held)<.02),
                            'reached_goal':reached,'metrics':m,'trace':trace,
                            'source_action_calls':env.native.step_count,'physics_ticks':env.native.physics_steps,
                            'cumulative_fallback_attempts':getattr(env.local_ik,'fallback_attempts',0),
                            'cumulative_fallback_successes':getattr(env.local_ik,'fallback_successes',0)})
            atomic_json(Path(config['output'])/f"seed_{config['scene_seed']}.json",{'results':results,
                       'setup_source_action_calls':env.setup_action_calls,'initialization_physics_ticks':env.native.settle_steps})
        return {'results':results,'setup_source_action_calls':env.setup_action_calls,
                'initialization_physics_ticks':env.native.settle_steps}
    finally:env.close()


def main():
    from pnr.telemetry import RunLogger,atomic_json
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='phase2_grasp_goal_probe01')
    p.add_argument('--checkpoint',default='runs/phase2_sac_grasp01/checkpoints/bc_initialized.zip')
    p.add_argument('--fallback',action='store_true')
    a=p.parse_args();logger=RunLogger(a.run_id,kind='validation',skill='grasp',name='Banana lift target diagnosis',
        algorithm='Frozen BC-initialized SAC actor',budget=12,summary='Measured lift-height and pregrasp settling variants; no learning.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    all_results=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(probe,{'scene_seed':s,'checkpoint':a.checkpoint,'output':str(out),'fallback':a.fallback}) for s in [42,43001,43002]]
        for f in as_completed(futures):
            result=f.result();all_results.append(result)
            rows=[r for result in all_results for r in result['results']]
            logger.metric(sum(len(r['trace']) for r in rows),scope='goal_diagnostic',eval_episodes=len(rows),
                          success_rate=sum(r['success'] for r in rows)/len(rows),
                          evaluation_action_calls=sum(r['source_action_calls'] for r in rows),
                          setup_action_calls=sum(r['setup_source_action_calls'] for r in all_results))
            atomic_json(out/'summary.json',{'checkpoint':a.checkpoint,'fallback':a.fallback,'results':all_results})
            print(json.dumps([(r['scene_seed'],r['goal_lift_m'],r['pregrasp_settle_calls'],r['success'],r['reached_goal'],r['metrics']['table_clearance']) for r in result['results']]),flush=True)
    logger.artifact('artifacts/summary.json','Frozen policy goal/settling physical diagnosis')
    logger.finish(summary='Completed twelve frozen banana lift/settling probes; no training or hidden maze evaluation.')

if __name__=='__main__':main()
