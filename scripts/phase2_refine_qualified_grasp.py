#!/usr/bin/env python3
"""Fixed 2k grasp BC refinement and a separate 20-case TRAIN transport comparison.

No automatic fitting, SAC continuation, full-task evaluation or checkpoint
selection. Use inspect first; fit/compare are separate explicit root decisions.
Collection dependencies and the original physical helpers remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import torch
from pnr.grasp_sampling import fixed_variants
from pnr.policies import load_policy_checkpoint
from pnr.telemetry import RunLogger, atomic_json
from scripts.phase2_train_grasp import dataset_rows
from scripts import phase2_grasp_qualified_demos as collector


class Mixture:
    """Exactly half original labels and half qualified native-SAC actions."""
    def __init__(self, old_dataset, qualified_dataset, seed):
        old_rows, old_groups, self.old_report = dataset_rows(old_dataset)
        self.old_rows = old_rows
        self.old_groups = defaultdict(list)
        for key, indices in sorted(old_groups.items()):
            self.old_groups[int(key.split(':')[0])].append(np.asarray(indices, dtype=int))
        self.qualified = defaultdict(list)
        for episode in qualified_dataset['episodes']:
            if episode['summary']['bc_eligible']:
                if not episode['transitions']:
                    raise ValueError('Qualified grasp prefix has no actual actions')
                self.qualified[int(episode['case']['object_idx'])].append(episode['transitions'])
        self.objects = sorted(collector.OBJECTS)
        if sorted(self.old_groups) != self.objects or sorted(self.qualified) != self.objects:
            raise ValueError('Both supervision halves require all five objects; no replacement collection')
        self.rng = np.random.default_rng(seed)
        self.draws = {'old':defaultdict(int), 'qualified':defaultdict(int)}

    def sample(self, half=200):
        rows = []; provenance = []
        for origin in ('old', 'qualified'):
            objects = self.rng.choice(self.objects, size=half)
            for obj in objects:
                obj = int(obj)
                if origin == 'old':
                    phases = self.old_groups[obj]
                    phase = phases[int(self.rng.integers(len(phases)))]
                    row = self.old_rows[int(self.rng.choice(phase))]
                else:
                    episodes = self.qualified[obj]
                    episode = episodes[int(self.rng.integers(len(episodes)))]
                    row = episode[int(self.rng.integers(len(episode)))]
                rows.append(row); provenance.append((origin, obj))
                self.draws[origin][obj] += 1
        observed = {key:np.stack([r['observation'][key] for r in rows])
                    for key in ('observation', 'achieved_goal', 'desired_goal')}
        actions = np.stack([r['action'] for r in rows])
        return observed, actions, provenance

    def report(self):
        return {'examples_by_source_and_object':{k:{str(o):v for o,v in values.items()}
                    for k,values in self.draws.items()},
                'qualified_prefixes_by_object':{str(o):len(self.qualified[o]) for o in self.objects},
                'qualified_rows_by_object':{str(o):sum(len(e) for e in self.qualified[o]) for o in self.objects},
                'old_supervision':self.old_report}


def tensor_digest(module):
    result=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        header=json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()
        result.update(len(header).to_bytes(8,'big'));result.update(header);result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def fixed_state(model):
    return {'critic':tensor_digest(model.critic), 'critic_target':tensor_digest(model.critic_target),
            'log_std_head':tensor_digest(model.actor.log_std),
            'temperature':float(model.log_ent_coef.detach().cpu()),
            'online_decisions':int(model.num_timesteps), 'online_sac_updates':int(model._n_updates)}


def load_collection(directory):
    directory=Path(directory);art=directory/'artifacts'
    status=json.loads((directory/'status.json').read_text())
    summary=json.loads((art/'summary.json').read_text());plan=json.loads((art/'plan.json').read_text())
    if status['status']!='complete' or summary['completed_preparations']!=120 or summary['planned_preparations']!=120:
        raise ValueError('Require the complete predeclared 120-start collection')
    if summary['fatal_errors'] or summary['missing_preparation_ids'] or not summary['cost_accounting_complete']:
        raise ValueError('Incomplete or malformed collection cannot be fitted')
    if plan.get('development_phase')!=collector.DEVELOPMENT_PHASE:
        raise ValueError('Expected explicitly authorized successor collection')
    if collector.digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Collection plan digest mismatch')
    audit=json.loads((art/'independent_data_audit_scene_bank.json').read_text())
    if (audit.get('verdict')!='PASS' or audit.get('plan_sha256')!=plan['plan_sha256']
            or audit.get('dataset_sha256')!=collector.sha256_file(art/'grasp_prefixes.pkl')
            or audit.get('summary_sha256')!=collector.sha256_file(art/'summary.json')):
        raise ValueError('Completed-data audit must match exact dataset and outcomes')
    data=pickle.loads((art/'grasp_prefixes.pkl').read_bytes())
    if data['plan_sha256']!=plan['plan_sha256'] or not data['qualification_only_bc'] or data['directly_replay_seedable']:
        raise ValueError('Wrong grasp-prefix dataset semantics')
    if collector.summarize(summary['results'], plan['schedule']) != summary:
        raise ValueError('Collection accounting differs from all-attempt manifest')
    for path, expected_hash in plan['source_hashes'].items():
        if collector.sha256_file(ROOT/path) != expected_hash:
            raise ValueError('Frozen collection dependency changed: '+path)
    expected={c['preparation_id']:c for c in plan['schedule']}
    records={r['preparation_id']:r for r in summary['results']}
    seen=set()
    for e in data['episodes']:
        key=e['case']['preparation_id']
        if key in seen or e['case']!=expected.get(key) or e['summary']!=records[key]:
            raise ValueError('Prefix identity or qualified summary differs from retained manifest')
        seen.add(key);collector.validate_prefix(e['transitions'])
        raw_path=art/(key+'_grasp.pkl')
        if collector.sha256_file(raw_path)!=e['summary']['grasp_prefix_sha256']:
            raise ValueError('Raw grasp prefix changed')
        raw=pickle.loads(raw_path.read_bytes())
        if raw['case']!=e['case'] or collector.serial(raw['transitions'])!=collector.serial(e['transitions']):
            raise ValueError('Merged training tuples differ from immutable measured prefix')
        record=e['summary']
        eligible=collector.qualification(record['prepared_handover'], record['route_completed'],
            record['final_hold_retained'], wall_contacts=record['wall_contact_calls'],
            table_contacts=record['robot_table_contact_calls'], fatal_error=bool(record.get('fatal_error')))
        if record['bc_eligible'] != eligible:
            raise ValueError('Qualification label differs from retained physical outcome')
    if seen!=set(expected):raise ValueError('Not all attempts retained in dataset')
    old_path=Path(plan['old_demonstrations']['path'])
    if collector.sha256_file(old_path)!=plan['old_demonstrations']['sha256']:
        raise ValueError('Original grasp supervision changed')
    if collector.sha256_file(plan['source_grasp']['frozen_path'])!=plan['source_grasp']['sha256']:
        raise ValueError('Frozen source grasp changed')
    old=pickle.loads(old_path.read_bytes())
    return plan,summary,data,old


def bc_update(actor, optimizer, observations, actions, clip_norm):
    """One supervised update; this has no access to critic, replay or simulator."""
    tensors={k:torch.as_tensor(v,device='cpu') for k,v in observations.items()}
    labels=torch.as_tensor(actions,device='cpu')
    prediction=actor(tensors,deterministic=True)
    loss=torch.mean((prediction-labels)**2)
    if not torch.isfinite(loss):raise ValueError('Nonfinite BC loss; no replacement fit')
    optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(),clip_norm);optimizer.step()
    return float(loss.detach())


def fit(directory,output):
    plan,summary,data,old=load_collection(directory);method=plan['bc_refinement']
    if (method['updates'],method['batch_size'],method['old_examples_per_update'],method['qualified_examples_per_update'])!=(2000,400,200,200):
        raise ValueError('Only the predeclared fixed 2k BC budget is supported')
    output=Path(output)
    if output.exists():raise ValueError('No fitting retry or checkpoint replacement')
    torch.set_num_threads(1);torch.manual_seed(method['seed'])
    mixture=Mixture(old,data,method['seed'])
    model=load_policy_checkpoint(plan['source_grasp']['frozen_path'],device='cpu')
    if collector.checkpoint_contract(model)!=plan['source_grasp']['contract']:
        raise ValueError('Frozen schema/feature width/historical SAC counters differ')
    before=fixed_state(model);actor_before=tensor_digest(model.actor)
    logger=RunLogger(output.name,root=output.parent,kind='initialization',skill='grasp',budget=2000,
        accounting={'new_online_transitions':0,'new_evaluation_physics_ticks':0,'new_physical_interactions':0},
        algorithm='Fixed BC refinement of SAC+HER grasp; transport-qualified TRAIN prefixes',
        name='Transport-qualified grasp · fixed 2k BC',summary='No new online SAC updates; historical 12k decisions/2k SAC updates retained.')
    artifacts=output/'artifacts';artifacts.mkdir(exist_ok=True)
    specification={'plan_sha256':plan['plan_sha256'],'method':method,'source_grasp':plan['source_grasp'],
        'qualified_dataset_sha256':collector.sha256_file(Path(directory)/'artifacts/grasp_prefixes.pkl'),
        'collection_summary_sha256':collector.sha256_file(Path(directory)/'artifacts/summary.json'),
        'source_hashes':{str(Path(__file__).relative_to(ROOT)):collector.sha256_file(__file__),
                       'scripts/phase2_train_grasp.py':collector.sha256_file(ROOT/'scripts/phase2_train_grasp.py')},
        'parent_physical_costs':summary['total_physics_ticks_including_initialization'],
        'new_physical_interactions':0,'checkpoint_selection':'fixed endpoint only',
        'std_contract':'No direct log_std-head updates or calibration; shared actor layers may change conditional standard deviation.'}
    atomic_json(artifacts/'fit_specification.json',specification)
    # Separate BC optimizer; no SAC optimizer/critic/temperature update or counter reset.
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=method['learning_rate'])
    model.actor.train();losses=[];started=time.monotonic()
    try:
        for update in range(1,2001):
            obs,actions,_=mixture.sample(200)
            losses.append(bc_update(model.actor,optimizer,obs,actions,method['gradient_clip_norm']))
            if update%200==0:
                logger.metric(0,scope='actor_refinement',metric_source='supervised',updates=update,
                              supervised_updates=update,bc_training_loss=float(np.mean(losses[-200:])),
                              supervised_examples=update*400,new_online_sac_updates=0)
        model.actor.eval()
        if fixed_state(model)!=before:raise ValueError('BC changed a protected critic, std head, temperature or SAC counter')
        checkpoint=output/'checkpoints/grasp_bc_002000.zip';checkpoint.parent.mkdir(exist_ok=True)
        model.save(checkpoint)
        reloaded=load_policy_checkpoint(checkpoint,device='cpu')
        if tensor_digest(reloaded.actor)!=tensor_digest(model.actor) or fixed_state(reloaded)!=before:
            raise ValueError('Fixed endpoint serialization changed actor/ancestry')
        result={'status':'complete','checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':collector.sha256_file(checkpoint),
            'contract':collector.checkpoint_contract(reloaded),'new_supervised_updates':2000,'supervised_examples':800000,
            'new_online_decisions':0,'new_online_sac_updates':0,'new_physical_interactions':0,
            'historical_online_decisions':12000,'historical_online_sac_updates':2000,
            'actor_before_sha256':actor_before,'actor_after_sha256':tensor_digest(model.actor),
            'protected_state':before,'sampling':mixture.report(),'wall_seconds':time.monotonic()-started,
            'plan_sha256':plan['plan_sha256'],'paired_comparison_launched':False}
        atomic_json(artifacts/'result.json',result)
        logger.finish('complete','Fixed 2k BC endpoint saved; no online SAC, physical comparison or checkpoint selection.')
        return result
    except BaseException as error:
        atomic_json(artifacts/'failure.json',{'error':repr(error),'completed_bc_updates':len(losses),
                    'new_physical_interactions':0,'new_online_sac_updates':0,'retry':False})
        logger.finish('failed','Fixed refinement interrupted; evidence retained, no replacement fit.')
        raise



def paired_result(results,cases):
    grouped=defaultdict(dict)
    for r in results:
        branch=r['case']['comparison_branch'];key=r['case_id']
        if branch not in ('control','candidate'):raise ValueError('Unknown comparison branch')
        if branch in grouped[key]:raise ValueError('Duplicate comparison branch')
        grouped[key][branch]=r
    expected={c['case_id'] for c in cases}
    complete=set(grouped)==expected and all(set(v)=={'control','candidate'} for v in grouped.values())
    wins=[];losses=[];preparation_regressions=[]
    if complete:
        for key,pair in grouped.items():
            c,n=pair['control'],pair['candidate']
            if n['bc_eligible'] and not c['bc_eligible']:wins.append(key)
            if c['bc_eligible'] and not n['bc_eligible']:losses.append(key)
            if c['prepared_handover'] and not n['prepared_handover']:preparation_regressions.append(key)
    costs_complete=bool(results) and all(r['cost_accounting_complete'] and not r.get('fatal_error') for r in results)
    return {'scope':'paired TRAIN retained teacher transport plus hold','full_task_evaluation':False,
        'ancestry':'development cases may appear in ancestor training; no heldout claim',
        'planned_cases':len(cases),'completed_branches':len(results),'complete':complete,
        'candidate_only_successes':sorted(wins),'control_only_successes':sorted(losses),
        'net_retention_wins':len(wins)-len(losses),'new_preparation_failures':sorted(preparation_regressions),
        'gate_passed':bool(complete and costs_complete and len(wins)-len(losses)>=3 and not preparation_regressions),
        'branch_successes':{b:sum(r['bc_eligible'] for r in results if r['case']['comparison_branch']==b) for b in ['control','candidate']},
        'cost_accounting_complete':costs_complete,'total_physics_ticks_including_initialization':sum(r.get('total_physics_ticks_including_initialization',0) for r in results),
        'full60_launched':False,'results':results}


def compare(directory,fit_directory,output):
    plan,_,_,_=load_collection(directory)
    fitted=json.loads((Path(fit_directory)/'artifacts/result.json').read_text())
    if (fitted['plan_sha256']!=plan['plan_sha256'] or fitted['new_supervised_updates']!=2000
            or fitted['new_online_sac_updates']!=0):raise ValueError('Require exact fixed BC endpoint')
    if collector.sha256_file(fitted['checkpoint'])!=fitted['checkpoint_sha256']:raise ValueError('Candidate checkpoint changed')
    output=Path(output)
    if output.exists():raise ValueError('No paired comparison retries or replacement')
    cases=plan['development_comparison']['cases'];schedule=[]
    for i,case in enumerate(cases):
        for branch in ['control','candidate']:
            schedule.append(dict(case,comparison_branch=branch,
                preparation_id=case['case_id']+'__'+branch,
                preparation_variant=fixed_variants(40+i,9401)[0]))
    checkpoints={'control':plan['source_grasp'],'candidate':dict(frozen_path=fitted['checkpoint'],
                 sha256=fitted['checkpoint_sha256'],contract=fitted['contract'])}
    logger=RunLogger(output.name,root=output.parent,kind='evaluation',skill='grasp',budget=40,planned_cases=40,planned_pairs=20,
        name='Qualified grasp · 20 TRAIN pairs',algorithm='Deterministic learned grasp with explicit teacher transport diagnostic')
    out=output/'artifacts';out.mkdir(exist_ok=True)
    frozen={'plan_sha256':plan['plan_sha256'],'schedule':schedule,'checkpoints':checkpoints,
            'settings':plan['settings'],'gate':plan['development_comparison']['acceptance_gate'],
            'collector_source_sha256':collector.sha256_file(collector.__file__),
            'runner_source_sha256':collector.sha256_file(__file__),'full_task_evaluation':False}
    atomic_json(out/'comparison_plan.json',frozen)
    results=[]
    for branch in ['control','candidate']:
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),
                initializer=collector.initialize_worker,initargs=(checkpoints[branch],)) as pool:
            futures=[pool.submit(collector.worker,(c,str(out),plan['settings'])) for c in schedule if c['comparison_branch']==branch]
            for future in as_completed(futures):
                results.append(future.result());summary=paired_result(results,cases)
                atomic_json(out/'summary.json',summary)
                logger.metric(sum(r.get('source_action_calls',0) for r in results),scope='training_transport_diagnostic',
                    eval_episodes=len(results),planned_cases=40,planned_pairs=20,
                    source_action_calls=sum(r.get('source_action_calls',0) for r in results),
                    physics_ticks=summary['total_physics_ticks_including_initialization'])
        if any(r.get('fatal_error') for r in results):break
    summary=paired_result(results,cases);atomic_json(out/'summary.json',summary)
    logger.finish('complete' if summary['complete'] and summary['cost_accounting_complete'] else 'failed',
        f"Net retained-transport wins {summary['net_retention_wins']}; gate passed={summary['gate_passed']}. No full60 launched.")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['inspect','fit','compare'])
    p.add_argument('--collection',type=Path,default=ROOT/'runs/phase2_grasp_qualified_demos03')
    p.add_argument('--output',type=Path)
    p.add_argument('--fit-directory',type=Path)
    args=p.parse_args()
    if args.mode=='inspect':
        plan,summary,data,old=load_collection(args.collection)
        print(json.dumps({'planned_method':plan['bc_refinement'],'coverage':Mixture(old,data,10101).report(),
                          'new_physical_interactions':0,'new_optimizer_updates':0},default=collector.plain))
    elif args.mode=='fit':
        if args.output is None:p.error('Explicit new --output required')
        fit(args.collection,args.output)
    else:
        if args.output is None or args.fit_directory is None:p.error('Explicit --output and --fit-directory required')
        compare(args.collection,args.fit_directory,args.output)


if __name__=='__main__':main()
