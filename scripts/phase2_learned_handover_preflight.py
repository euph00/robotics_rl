#!/usr/bin/env python3
"""TRAIN-only feasibility of frozen learned grasps feeding short teacher carry.

Reuses LearnedSystem.run's exact original-start approach/grasp/hold/posture
prefix, then stops before learned navigation/reach. The carry controller is a
TRAIN data teacher, explicitly not an evaluated navigation policy. No resets,
attachments, alternative physics, production control changes, or failed-case
replacement are permitted.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from pnr.env import PickNavReachEnv
from pnr.general_carry import GeneralCarryEnv, route_action
from pnr.general_manipulation import GeneralGraspEnv
from pnr.planning import maze_astar
from pnr.policies import load_policy_checkpoint
from pnr.scenes import load_scene_bank
from pnr.system import LearnedSystem, StageFailure
from pnr.telemetry import RunLogger, atomic_json
from scripts.evaluate_system import freeze_checkpoints, plain
from scripts.phase2_evaluate_system import freeze_sources, plan_digest, select_cases, sha256_file
from scripts.phase2_apple_retention_branches import contact_record

_GRASP=None

class _DiagnosticComplete(Exception):
    """Unwinds the shared run before reach; never a full-task success signal."""

class _UnusedPolicy:
    def predict(self,*args,**kwargs):
        raise AssertionError('No learned carry/reach policy belongs in this TRAIN preflight')

class TeacherCarryPreflight(LearnedSystem):
    """Shared learned prefix, with a bounded, explicitly identified carry teacher."""
    def __init__(self,native,grasp,*,teacher_decisions=100,delta_scale=.075,on_step=None):
        super().__init__(native,{'grasp':grasp,'carry':_UnusedPolicy(),'reach':_UnusedPolicy()},
            grasp_horizon=600,carry_horizon=teacher_decisions,reach_horizon=1,on_step=on_step,
            controller_speed_scale=1.,grasp_proposal_mode='clearance-v2',reach_tolerance_m=.08,
            skill_factories={'grasp':partial(GeneralGraspEnv.bind,goal_lift_m=.24)})
        if teacher_decisions<1:raise ValueError('Positive teacher decision budget required')
        self.teacher_budget=int(teacher_decisions);self.delta_scale=float(delta_scale)
        self.calls_by_source['teacher']=0
        self.teacher_transitions=[];self.retained_short_transport=False
        self.handover_observation=None;self.preparation_calls=None

    def _skill(self,name):
        if name=='grasp':return super()._skill(name)
        if name!='carry':raise AssertionError('Preflight must stop before reaching')
        start=self._begin('teacher_carry','teacher')
        self.navigation_start_action=self.native.step_count
        self.preparation_calls=self.native.step_count
        before=(self.native.step_count,self.native.physics_steps)
        carry=GeneralCarryEnv.bind(self.native,self.controller,horizon=self.teacher_budget,
                                   delta_scale=self.delta_scale)
        if before!=(self.native.step_count,self.native.physics_steps):
            raise AssertionError('Binding must not advance or reset physics')
        observation=carry._observation()
        self.handover_observation=observation
        initial_goal=observation['desired_goal'].copy()
        path=maze_astar(self.native.cube_positions,self.controller.base_position(),
                       self.native.goal_pos[:2],table_bounds=self.native.p.getAABB(self.native.table_id))
        last_info={};terminated=False
        for decision in range(self.teacher_budget):
            action=route_action(path,observation['achieved_goal']*carry.goal_scale,
                                self.native.goal_pos[:2],carry.delta_scale)
            before_step=self.native.step_count
            next_observation,reward,terminated,truncated,last_info=carry.step(action)
            if not np.array_equal(next_observation['desired_goal'],initial_goal):
                raise AssertionError('Teacher changed the fixed original exit goal')
            self.teacher_transitions.append({'observation':observation,'action':action,
                'next_observation':next_observation,'reward':reward,'terminated':terminated,
                'truncated':truncated,'info':dict(last_info),'source_action_calls':self.native.step_count-before_step})
            observation=next_observation
            if terminated or self.native.drop_events or truncated:break
        reached_budget=len(self.teacher_transitions)==self.teacher_budget
        self.retained_short_transport=bool(not terminated and self.native.drop_events==0
            and last_info.get('bilateral_contact',False) and (reached_budget or last_info.get('is_success',False)))
        self._finish('teacher_carry',start,self.retained_short_transport,
            teacher_decisions=len(self.teacher_transitions),budget=self.teacher_budget,
            reason=last_info.get('failure_reason') or ('retained_budget' if reached_budget else 'exit_reached'),
            goal_distance=last_info.get('goal_distance'),original_goal=initial_goal.tolist())
        if not self.retained_short_transport:
            raise StageFailure('teacher_carry: '+str(last_info.get('failure_reason') or 'retention_failed'))
        raise _DiagnosticComplete()

    def run_preflight(self):
        try:
            result=super().run()
        except _DiagnosticComplete:
            result={'status':'complete','stages':self.stage_results}
        result.update(kind='learned_handover_preflight',scope='train_preparation_and_short_teacher_carry',
            full_task_evaluation=False,success=self.retained_short_transport,
            retained_short_transport=self.retained_short_transport,
            prepared_handover=bool(self.stage_results.get('planned_carry_posture',{}).get('success')),
            original_start=True,inter_skill_resets=False,teacher_in_carry=True,
            teacher_in_grasp=False,production_controller_changed=False,
            teacher_decisions=len(self.teacher_transitions),preparation_source_action_calls=self.preparation_calls,
            stages=self.stage_results,policy_decisions=dict(self.policy_decisions),
            source_calls_by_controller=dict(self.calls_by_source),
            wall_contact_calls=self.wall_contact_calls,robot_table_contact_calls=self.robot_table_contact_calls,
            first_lift_action=self.first_lift_action,first_grasped_lift_action=self.first_grasped_lift_action,
            navigation_start_action=self.navigation_start_action)
        return result

def initialize_worker(record):
    global _GRASP
    import torch
    torch.set_num_threads(1)
    if sha256_file(record['frozen_path'])!=record['sha256']:raise ValueError('Frozen grasp digest mismatch')
    _GRASP=load_policy_checkpoint(record['frozen_path'],device='cpu')
    if (_GRASP.action_space.shape!=(4,) or _GRASP.observation_space['observation'].shape!=(54,)):
        raise ValueError('Expected54-state learned grasp')

def aggregate_results(results,cases):
    expected={c['case_id'] for c in cases};seen={r['case_id'] for r in results}
    if len(seen)!=len(results) or seen-expected:raise ValueError('Duplicate or undeclared attempt')
    fields=('source_action_calls','physics_ticks','initialization_physics_ticks',
            'total_physics_ticks_including_initialization','teacher_decisions')
    return {'planned_attempts':len(cases),'completed_attempts':len(results),
        'prepared_handovers':sum(r.get('prepared_handover',False) for r in results),
        'retained_short_transport':sum(r.get('retained_short_transport',False) for r in results),
        'retained_fraction_of_planned':sum(r.get('retained_short_transport',False) for r in results)/len(cases),
        'missing_case_ids':sorted(expected-seen),'all_failures_retained':True,
        'cost_accounting_complete':all(r.get('cost_accounting_complete',False) for r in results),
        'full_task_evaluation':False,'partition':'train',
        **{k:sum(r.get(k,0) for r in results) for k in fields},'results':results}

def worker(task):
    case,output,budget=task;out=Path(output);key=case['case_id']
    native=system=None;rows=[];started=time.monotonic()
    result={'case_id':key,'case':case,'partition':'train','success':False,
            'retained_short_transport':False,'prepared_handover':False,'full_task_evaluation':False,
            'source_action_calls':0,'physics_ticks':0,'initialization_physics_ticks':0}
    with (out/(key+'.log')).open('w') as log,(out/(key+'_contacts.jsonl')).open('w') as trace:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            native=PickNavReachEnv(**case['env_kwargs'])
            if native.scene_manifest()['maze_hash']!=case['maze_hash']:raise ValueError('Frozen TRAIN map mismatch')
            def observe(current,observation,info):
                row=contact_record(native,current.controller,observation,info,current)
                rows.append(row);trace.write(json.dumps(row,default=plain,allow_nan=False)+'\n')
            system=TeacherCarryPreflight(native,_GRASP,teacher_decisions=budget,on_step=observe)
            result.update(system.run_preflight())
        except Exception as exc:
            result.update(error=f'{type(exc).__name__}: {exc}',status='failed',
                          failed_stage=system.stage if system is not None else 'environment_setup')
            if system is not None and system._active_stage_start is not None:
                system._finish(system.stage,system._active_stage_start,False,error=str(exc))
        finally:
            if system is not None:
                result.update(stages=system.stage_results,policy_decisions=system.policy_decisions,
                    source_calls_by_controller=system.calls_by_source,
                    prepared_handover=bool(system.stage_results.get('planned_carry_posture',{}).get('success')),
                    proposal_diagnostics=system.controller.stage_results.get('grasp_generation'),
                    teacher_decisions=len(system.teacher_transitions),
                    handover_observation=system.handover_observation)
                if system.teacher_transitions:
                    t=system.teacher_transitions
                    np.savez_compressed(out/(key+'_teacher_transitions.npz'),
                        observation=np.stack([x['observation']['observation'] for x in t]),
                        achieved_goal=np.stack([x['observation']['achieved_goal'] for x in t]),
                        desired_goal=np.stack([x['observation']['desired_goal'] for x in t]),
                        next_observation=np.stack([x['next_observation']['observation'] for x in t]),
                        next_achieved_goal=np.stack([x['next_observation']['achieved_goal'] for x in t]),
                        action=np.stack([x['action'] for x in t]),reward=np.array([x['reward'] for x in t]),
                        terminated=np.array([x['terminated'] for x in t]),truncated=np.array([x['truncated'] for x in t]),
                        source_action_calls=np.array([x['source_action_calls'] for x in t]))
                    atomic_json(out/(key+'_teacher_info.json'),json.loads(json.dumps([x['info'] for x in t],default=plain)))
            if native is not None:
                result.update(scene_manifest=native.scene_manifest(),metrics=native.evaluate(),
                    source_action_calls=native.step_count,physics_ticks=native.physics_steps,
                    initialization_physics_ticks=native.settle_steps,
                    total_physics_ticks_including_initialization=native.physics_steps+native.settle_steps,
                    cost_accounting_complete=True)
                if rows:result['final_trace']=rows[-1]
                if system is not None:
                    if len(rows)!=native.step_count or sum(system.calls_by_source.values())!=native.step_count:
                        result.update(cost_accounting_complete=False,accounting_error='Native trace/source count mismatch')
                    held=[r for r in rows if r['stage']=='pick_hold']
                    carry=[r for r in rows if r['stage']=='teacher_carry']
                    if held:result['after_pick_hold']=held[-1]
                    if carry:
                        first=rows.index(carry[0]);result['before_teacher_carry']=rows[first-1] if first else None
                        anchor=np.asarray(result['before_teacher_carry']['object_in_hand'])
                        result['peak_teacher_payload_shift_m']=float(max(np.linalg.norm(np.asarray(r['object_in_hand'])-anchor) for r in carry))
                native.close()
            else:
                result.update(cost_accounting_complete=False,cost_accounting_note='No live environment; partial constructor cost unknown')
            result['wall_seconds']=time.monotonic()-started
            result=json.loads(json.dumps(result,default=plain,allow_nan=False))
            atomic_json(out/(key+'.json'),result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp',type=Path,required=True)
    parser.add_argument('--bank',type=Path,default=ROOT/'configs/phase2_scene_bank.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--teacher-decisions',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('Use1–8 CPU workers')
    if not 1<=args.teacher_decisions<=100:parser.error('Bounded preflight permits at most100 teacher decisions')
    if args.output.exists() and any(args.output.iterdir()):parser.error('Use a fresh output directory; preserve every attempt')
    bank=load_scene_bank(args.bank)
    cases=select_cases(bank,'train',[14,5,2,7,10],layout_limit=2)
    out=args.output/'artifacts';out.mkdir(parents=True)
    record=freeze_checkpoints({'grasp':args.grasp},out/'checkpoints')['grasp']
    candidate_path=args.grasp.parent/'manifest.json'
    if candidate_path.exists():
        candidate=json.loads(candidate_path.read_text())
        if candidate.get('grasp_proposal_mode')!='clearance-v2':
            raise ValueError('Candidate must declare the clearance-v2 proposal used here')
        matches=[r for r in candidate.get('checkpoints',{}).values() if r.get('sha256')==record['sha256']]
        if not matches:raise ValueError('Grasp bytes not recorded in candidate manifest')
        record['training_selection']=matches[0]
        atomic_json(out/'grasp_candidate_manifest.json',candidate)
    hashes=freeze_sources(out/'source')
    for source in [Path(__file__).resolve(),ROOT/'scripts/phase2_apple_retention_branches.py']:
        target=out/'source'/'scripts'/source.name;shutil.copy2(source,target);hashes['scripts/'+source.name]=sha256_file(source)
    atomic_json(out/'source/manifest.json',hashes)
    plan={'protocol':'learned-handover-short-teacher-carry-v1','partition':'train','cases':cases,
          'planned_attempts':len(cases),'selection':'first2 frozen TRAIN layouts cross all5 fixed objects',
          'scene_bank_sha256':bank['manifest_sha256'],'checkpoint':record,'source_hashes':hashes,
          'teacher_decisions':args.teacher_decisions,'delta_scale':.075,'carry_repeat':5,
          'native_physics_ticks_per_call':5,'grasp_lift_m':.24,'grasp_horizon':600,
          'grasp_proposal_mode':'clearance-v2','planned_speed_scale':1.,'pick_hold_calls':50,
          'original_start':True,'inter_skill_resets':False,'physics_unchanged':True,
          'retain_failed_preparations':True,'teacher_in_carry':True,'teacher_in_grasp':False,
          'full_task_evaluation':False,'workers':args.workers}
    plan['plan_sha256']=plan_digest(plan);atomic_json(out/'diagnostic_plan.json',plan)
    logger=RunLogger(args.output.name,root=args.output.parent,kind='diagnostic',
        name='Learned grasp handover:10 TRAIN cases',summary='Frozen clearance grasp, original physical prefix, short teacher carry; all failures counted.')
    logger.artifact('artifacts/diagnostic_plan.json','Frozen TRAIN cases and grasp')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker,initargs=(record,)) as pool:
        futures={pool.submit(worker,(case,str(out),args.teacher_decisions)):case for case in cases}
        for future in as_completed(futures):
            case=futures[future]
            try:result=future.result()
            except Exception as exc:
                result={'case_id':case['case_id'],'case':case,'partition':'train','success':False,
                        'retained_short_transport':False,'prepared_handover':False,
                        'full_task_evaluation':False,'cost_accounting_complete':False,
                        'error':repr(exc),'failed_stage':'worker_exception',
                        'cost_accounting_note':'Worker failed; any contact trace must be audited before cost claims'}
                atomic_json(out/(case['case_id']+'_worker_exception.json'),result)
            results.append(result);summary=aggregate_results(results,cases);atomic_json(out/'summary.json',summary)
            logger.metric(transitions=summary['source_action_calls'],completed_attempts=len(results),
                planned_attempts=len(cases),retained_transport=summary['retained_short_transport'],
                physics_ticks=summary['total_physics_ticks_including_initialization'])
    logger.artifact('artifacts/summary.json','All10 attempts, preparation failures and costs')
    logger.finish(summary=f"{summary['retained_short_transport']}/10 retain through short teacher carry; not full-task evaluation.")
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
