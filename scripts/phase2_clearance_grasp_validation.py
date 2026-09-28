#!/usr/bin/env python3
"""Counted TRAIN-only audit of opt-in clearance-v2 grasp proposals and fast carry."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import phase2_carry_retention_probe as retention
from pnr.reference import ReferenceController
from pnr.scenes import load_scene_bank,case_specs
from pnr.telemetry import RunLogger,atomic_json


def worker(task):
    case,out,proposal_mode=task
    retention.ReferenceController=partial(ReferenceController,grasp_proposal_mode=proposal_mode)
    return retention.worker((case,'general075','contact',out,1400,None,'original',0.,False,True,18.))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--include-legacy-can',action='store_true')
    parser.add_argument('--case-ids',nargs='+',help='Optional subset of the declared ten TRAIN cases')
    parser.add_argument('--proposal-mode',choices=['legacy','clearance-v2'],default='clearance-v2')
    args=parser.parse_args()
    if args.output.exists():parser.error('Use new output directory')
    if not 1<=args.workers<=8:parser.error('Use1..8 CPU workers')
    out=args.output/'artifacts';out.mkdir(parents=True)
    bank=load_scene_bank('configs/phase2_scene_bank.json')
    objects=[14,5,2,7,10]
    specs=case_specs(bank,'train',object_indices=objects)[:10]
    if args.case_ids:
        available={s['case_id'] for s in specs}
        if len(set(args.case_ids))!=len(args.case_ids) or any(x not in available for x in args.case_ids):
            parser.error('Case subset must use distinct IDs from the declared ten TRAIN cases')
        specs=[s for s in specs if s['case_id'] in args.case_ids]
    cases=[{'layout_id':s['layout_id'],'maze_seed':s['env_kwargs']['maze_seed'],
            'object_idx':s['object_idx'],'scene_seed':s['env_kwargs']['seed']} for s in specs]
    if args.include_legacy_can:
        cases.append({'layout_id':'course_legacy42','maze_seed':None,'object_idx':2,'scene_seed':42})
    sources={}
    for relative in ['scripts/phase2_clearance_grasp_validation.py','scripts/phase2_carry_retention_probe.py',
                     'pnr/reference.py','pnr/clearance_grasps.py','pnr/planning.py','pnr/env.py',
                     'pnr/general_manipulation.py','pnr/skills.py','pnr/general_carry.py','pnr/local_control.py']:
        f=ROOT/relative;sources[relative]=hashlib.sha256(f.read_bytes()).hexdigest()
        shutil.copy2(f,out/relative.replace('/','_'))
    atomic_json(args.output/'config.json',{'proposal_mode':args.proposal_mode,'bank_sha256':bank['manifest_sha256'],
        'cases':cases,'teacher_close_distance_mm':18,'carry_delta_m':.075,'teacher_grasp_and_navigation':True,
        'original_physics':True,'sources':sources})
    logger=RunLogger(args.output.name,root=args.output.parent,kind='reference',skill='grasp',
        name='Clearance grasp TRAIN audit',algorithm='Physical teacher diagnostic',
        summary='Shared point-cloud geometry rule; all five objects on two TRAIN layouts; original physics.')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(case,str(out),args.proposal_mode)) for case in cases]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            summary={'protocol':'clearance-proposal-train-audit','proposal_mode':args.proposal_mode,'planned_attempts':len(cases),
                'completed_attempts':len(results),'successes':sum(r['success'] for r in results),
                'source_action_calls':sum(r['source_action_calls'] for r in results),
                'physics_ticks':sum(r['physics_ticks'] for r in results),
                'initialization_physics_ticks':sum(r['initialization_physics_ticks'] for r in results),
                'results':results}
            atomic_json(out/'summary.json',summary)
            logger.metric(transitions=summary['source_action_calls'],scope='physical_teacher_diagnostic',
                eval_episodes=len(results),physics_ticks=summary['physics_ticks'],
                reference_success_rate=summary['successes']/len(results))
            print(json.dumps({key:r.get(key) for key in ['case_id','success','error','physics_ticks']}),flush=True)
    logger.artifact('artifacts/summary.json','All TRAIN audit outcomes and proposals')
    logger.finish(summary=f"{summary['successes']}/{len(cases)} TRAIN teacher pickup/transport attempts passed.")

if __name__=='__main__':main()
