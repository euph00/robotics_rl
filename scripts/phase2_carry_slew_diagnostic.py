#!/usr/bin/env python3
"""Fixed TRAIN raw-versus-slew learned-system diagnostic; freeze/review first."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from pnr.command_slew import CommandSlewPolicy,PROTOCOL
from pnr.grasp_sampling import actor_parameter_digest
from pnr.telemetry import RunLogger,atomic_json
from scripts import phase2_evaluate_system as evaluator

RUN_ID='phase2_learned_carry_slew01'
BASELINE=ROOT/'runs/phase2_system_encoder_gated192_frozen_040k01/artifacts/evaluation_plan.json'
BANK=ROOT/'configs/phase2_scene_bank_expanded.json'


def sha(path):return evaluator.sha256_file(path)
def digest(plan):return evaluator.plan_digest(plan)


class DiagnosticInterrupted(InterruptedError):pass


def initialize_worker(records):evaluator.initialize_worker(records,'cpu',1)


def evaluate_branch(task):
    case,branch,execution,directory,deadline,stop=task;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    policies=evaluator._WORKER_POLICIES;original=policies['carry'];before={k:actor_parameter_digest(v) for k,v in policies.items()}
    interrupted=False;native_step=evaluator.PickNavReachEnv.step
    def timed_step(native,action):
        nonlocal interrupted
        if time.time()>=deadline or Path(stop).exists():
            interrupted=True;raise DiagnosticInterrupted('Declared45-minute cap/stop; no replacement')
        result=native_step(native,action)
        if time.time()>=deadline:interrupted=True
        return result
    with (directory/(case['case_id']+'_carry_commands.jsonl')).open('x') as commands:
        def persist(row):
            commands.write(json.dumps(row,default=evaluator.plain,allow_nan=False)+'\n');commands.flush()
        adapter=CommandSlewPolicy(original,filtered=branch=='slew',limit=.18,persist=persist)
        policies['carry']=adapter;evaluator.PickNavReachEnv.step=timed_step
        try:
            if time.time()>=deadline or Path(stop).exists():
                interrupted=True;result=evaluator.failure_result(case,DiagnosticInterrupted('Undispatched at cap'),'not_started')
                result.update(cost_accounting_complete=True,source_action_calls=0,physics_ticks=0)
                result.update(evaluator.deadline_metrics(result,0,execution['deadline_physics_ticks']))
            else:result=evaluator.evaluate_case(case,execution,directory,False)
        finally:
            policies['carry']=original;evaluator.PickNavReachEnv.step=native_step
        after={k:actor_parameter_digest(v) for k,v in policies.items()}
        changed=before!=after
        completed_commands=int(result.get('policy_decisions',{}).get('carry',0))
        command_count_matches=completed_commands==len(adapter.records)
        scene_matches=result.get('scene_manifest',{}).get('maze_hash')==case['maze_hash']
        known=bool(scene_matches and not interrupted and not adapter.integrity_error and not changed and command_count_matches and result.get('cost_accounting_complete'))
        result.update(branch=branch,diagnostic_scene_verified=scene_matches,diagnostic_outcome_known=known,actor_unchanged=not changed,actor_hashes_before=before,
            actor_hashes_after=after,command_filter={'protocol':PROTOCOL,'enabled':branch=='slew','limit':.18,'initial_control':[0.,0.],
                'requested_commands':len(adapter.records),'completed_carry_decisions':completed_commands,
                'command_count_matches':command_count_matches,'unconfirmed_or_partial_tail_commands':len(adapter.records)-completed_commands,
                'record_semantics':'Requested adapter controls, not measured physical joint motion',
                'changed_decisions':sum(r['raw_to_requested_l2']>1e-12 for r in adapter.records),
                'maximum_command_change_l2':max((r['command_change_l2'] for r in adapter.records),default=0.),
                'integrity_error':adapter.integrity_error},wall_cap_interrupted=interrupted,retry=False)
        if not known:Path(stop).touch()
        atomic_json(directory/(case['case_id']+'.json'),json.loads(json.dumps(result,default=evaluator.plain)))
        return result


def gate(results,cases):
    ids={c['case_id'] for c in cases};by={(r['branch'],r['case_id']):r for r in results}
    expected={(b,c) for b in ('raw','slew') for c in ids}
    complete=len(results)==len(by)==80 and set(by)==expected and all(r.get('diagnostic_outcome_known') and r.get('cost_accounting_complete') for r in results)
    if not complete:return {'passed':False,'complete':False,'reason':'Missing/unknown/cost-invalid fixed branch'}
    full={b:sum(bool(by[b,c]['success']) for c in ids) for b in ('raw','slew')}
    timed={b:sum(bool(by[b,c]['success_within_deadline_including_initialization']) for c in ids) for b in ('raw','slew')}
    gain=full['slew']-full['raw'];margin=gain>=3 if full['raw']<=37 else full['slew']==40 and gain>0
    return {'complete':True,'passed':full['slew']>=36 and margin and timed['slew']>=timed['raw'],
        'full':full,'inclusive_timed':timed,'net_full_gain':gain,
        'slew_only_successes':sorted(c for c in ids if by['slew',c]['success'] and not by['raw',c]['success']),
        'raw_only_successes':sorted(c for c in ids if by['raw',c]['success'] and not by['slew',c]['success']),
        'automatic_full60':False}


def freeze(output):
    output=Path(output)
    if output.exists():raise ValueError('Frozen plan output must be new')
    output.mkdir(parents=True);artifacts=output/'artifacts';artifacts.mkdir()
    baseline=json.loads(BASELINE.read_text());bank=evaluator.load_scene_bank(BANK)
    all_cases=evaluator.select_cases(bank,'train',evaluator.DEFAULT_OBJECTS)
    layouts={f'train_{i:03d}' for i in range(16,24)};cases=[c for c in all_cases if c['layout_id'] in layouts]
    if len(cases)!=40:raise AssertionError('Exactly TRAIN016–023 crossed with five supplied objects required')
    records=evaluator.freeze_checkpoints({k:Path(v['frozen_path']) for k,v in baseline['checkpoints'].items()},artifacts/'checkpoints')
    selected=json.loads((ROOT/'configs/phase2_candidate02.json').read_text())
    if {k:v['sha256'] for k,v in records.items()}!=selected['checkpoint_sha256']:raise ValueError('Candidate02 checkpoint bytes differ')
    if baseline['execution']!=selected['execution']:raise ValueError('Candidate02 execution differs')
    sources=evaluator.freeze_sources(artifacts/'source')
    for name in ('scripts/phase2_carry_slew_diagnostic.py',):
        destination=artifacts/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes());sources[name]=sha(destination)
    atomic_json(artifacts/'source/manifest.json',sources)
    # No retired test trajectories or cases are read; selected checkpoint hashes/settings only.
    plan={'protocol':PROTOCOL,'status':'frozen_pending_independent_review_and_root_physics_authorization',
        'source_hashes':sources,'baseline_plan':str(BASELINE),'baseline_plan_sha256':baseline['plan_sha256'],
        'scene_bank':{'path':str(BANK),'sha256':sha(BANK),'manifest_sha256':bank['manifest_sha256']},
        'checkpoints':records,'execution':baseline['execution'],'cases':cases,'run_id':RUN_ID,
        'case_role':'Predetermined TRAIN development cases; ancestor training overlap possible; no heldout claim',
        'schedule':[{'branch':b,'case_id':c['case_id']} for c in cases for b in ('raw','slew')],
        'workers':8,'wall_cap_seconds':2700,'command_filter':{'limit_l2_per_decision':.18,'initial_control':[0.,0.],
            'input':'original adapter float64 clip[-1,1], then L2 normalization',
            'output_dtype':'float32, matching unchanged LearnedSystem cast',
            'rounding_guard':'If post-float32 effective command exceeds .18, one fixed inward fraction correction by1-2e-6; then assert bound within1e-12','previous':'previous adapter-normalized requested command; not measured joint motion',
            'baseline':'exact original actor prediction tuple/action, no numeric rewrite',
            'not_a_claim':'No guaranteed physical acceleration smoothing; within-native-call position-control pulses persist'},
        'physical_contract':{'original_start':True,'resets_between_skills':False,'teacher_or_oracle':False,'hardware_physics_changed':False,
            'carry_delta_scale':.075,'carry_repeat_native_calls':5,'physics_ticks_per_native_call':5,'all_failures_retained':True,'retries':0},
        'gate':{'candidate_full_minimum':36,'positive_margin':'control<=37: gain>=3; control38–39: candidate40; control40: close',
            'inclusive_15000_tick_successes_at_least_control':True,'incomplete_unknown_or_cost_error':'fail without replacement'},
        'followup':{'maximum_full60':1,'requires_gate_and_root_review':True,'same_cases_settings_checkpoints_as':str(BASELINE),
            'only_change':'declared carry command filter','promotion_minimum_full':57,'promotion_minimum_inclusive_timed':38,
            'automatic_launch':False,'selected_control_full':52,'selected_control_inclusive_timed':38},
        'new_heldout_or_reserve':False,'retired_test_trajectories_imported':False,'training':False,
        'physical_interactions_at_freeze':0,'commands':{'run':'.venv/bin/python scripts/phase2_carry_slew_diagnostic.py run --plan PLAN --authorization AUTH'},
        'validation':{'synthetic_tests_passed':19,'command':'.venv/bin/python -m pytest -q tests/test_command_slew.py','physical_interactions':0},
        'test_hashes':{'tests/test_command_slew.py':sha(ROOT/'tests/test_command_slew.py')}}
    plan['plan_sha256']=digest(plan);atomic_json(artifacts/'plan.json',plan);return plan


def verify(plan_path,authorization):
    plan=json.loads(Path(plan_path).read_text())
    if digest(plan)!=plan['plan_sha256']:raise ValueError('Frozen plan digest changed')
    for name,h in plan['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen source changed: '+name)
    if sha(plan['scene_bank']['path'])!=plan['scene_bank']['sha256']:raise ValueError('Scene bank changed')
    for r in plan['checkpoints'].values():
        if sha(r['frozen_path'])!=r['sha256']:raise ValueError('Checkpoint changed')
    auth=json.loads(Path(authorization).read_text())
    if not(auth.get('authorized') is True and auth.get('authorized_by')=='/root' and auth.get('plan_sha256')==plan['plan_sha256'] and auth.get('scope')=='learned_carry_slew_40_train_pairs'):
        raise ValueError('Exact root physical authorization required')
    return plan


def run(plan_path,authorization):
    plan=verify(plan_path,authorization);directory=ROOT/'runs'/plan['run_id']
    if directory.exists():raise ValueError('No repeated diagnostic run')
    logger=RunLogger(directory.name,kind='evaluation',name='Learned carry command slew · 40 TRAIN pairs',planned_cases=80,
        summary='Raw versus .18 command slew; unchanged selected policies and native task.')
    artifacts=directory/'artifacts';artifacts.mkdir();atomic_json(artifacts/'plan.json',plan)
    deadline=time.time()+plan['wall_cap_seconds'];stop=directory/'STOP';results=[]
    cases={c['case_id']:c for c in plan['cases']}
    try:
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=mp.get_context('spawn'),initializer=initialize_worker,initargs=(plan['checkpoints'],)) as pool:
            futures={pool.submit(evaluate_branch,(cases[s['case_id']],s['branch'],plan['execution'],str(artifacts/s['branch']),deadline,str(stop))):s for s in plan['schedule']}
            for future in as_completed(futures):
                slot=futures[future]
                try:r=future.result()
                except BaseException as error:
                    stop.touch();r=evaluator.failure_result(cases[slot['case_id']],error,'worker_execution')
                    r.update(branch=slot['branch'],diagnostic_outcome_known=False)
                    r.update(evaluator.deadline_metrics(r,0,plan['execution']['deadline_physics_ticks']))
                results.append(r)
                summaries={b:evaluator.summarize_results([v for v in results if v['branch']==b],plan['cases']) for b in ('raw','slew')}
                report={'plan_sha256':plan['plan_sha256'],'completed_branches':len(results),'planned_branches':80,'results':results,
                    'summaries':summaries,'gate':gate(results,plan['cases']),'full60_launched':False}
                atomic_json(artifacts/'summary.json',report)
                logger.metric(sum(v['source_action_calls'] for v in results),scope='full_task_TRAIN_diagnostic',eval_episodes=len(results),planned_cases=80,
                    branch_summaries=summaries,raw_full_successes=summaries['raw']['successes'],slew_full_successes=summaries['slew']['successes'],
                    raw_inclusive_timed_successes=summaries['raw']['successes_within_deadline_including_initialization'],
                    slew_inclusive_timed_successes=summaries['slew']['successes_within_deadline_including_initialization'],total_physics_ticks_including_initialization=sum(v['total_physics_ticks_including_initialization'] for v in results))
        logger.finish('complete' if report['gate']['complete'] else 'failed',
            'Fixed gate '+('passed; root review required.' if report['gate']['passed'] else 'failed; no full60.'))
        return report
    except BaseException:
        stop.touch();logger.finish('failed','Incomplete diagnostic; available outcomes/costs retained, no replacement.');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='freeze':print(freeze(args.output)['plan_sha256'])
    else:run(args.plan,args.authorization)

if __name__=='__main__':main()
