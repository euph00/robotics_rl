#!/usr/bin/env python3
"""Compare one actual learned-grasp handover with recorded carry snapshot zero.

No navigation rollout or carry action is executed. The reference is an immutable
training observation already collected through physical reference motions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pnr.carry import CarrySkillEnv
from pnr.env import PickNavReachEnv
from pnr.policies import load_policy_checkpoint
from pnr.system import LearnedSystem, StageFailure
from pnr.telemetry import RunLogger,atomic_json

GROUPS={'wall_map':slice(0,264),'table_map':slice(264,528),'base_xy_scaled':slice(528,530),
        'joint_position_without_base_xy':slice(530,543),'joint_velocity':slice(543,558),
        'ee_local_position':slice(558,561),'ee_quaternion':slice(561,565),
        'object_local_position':slice(565,568),'object_quaternion':slice(568,572),
        'object_linear_velocity':slice(572,575),'object_angular_velocity':slice(575,578),
        'finger_contacts':slice(578,580)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(observation):
    features=observation['observation'];base=features[528:530].astype(float)*20.
    ee=features[558:561].astype(float).copy();ee[:2]+=base
    obj=features[565:568].astype(float).copy();obj[:2]+=base
    return {'base_world_xy':base.tolist(),'ee_world_position':ee.tolist(),'object_world_position':obj.tolist(),
        'joint_positions':np.r_[base-PickNavReachEnv.root_position[:2],features[530:543]].tolist(),
        **{key:features[part].tolist() for key,part in GROUPS.items() if not key.endswith('map')},
        'desired_goal_world_xy':(observation['desired_goal']*20.).tolist()}


class UnusedPolicy:
    def predict(self,*args,**kwargs):raise AssertionError('Only the learned grasp executes in this audit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='carry_handoff_observation_audit')
    args=parser.parse_args();torch.set_num_threads(1)
    logger=RunLogger(args.run_id,kind='validation',skill='carry',algorithm='Physical handover observation audit',
        name='Carry handover state comparison',summary='One learned pick and posture; no carry rollout or training.')
    artifacts=logger.directory/'artifacts';inputs=artifacts/'inputs';inputs.mkdir(parents=True)
    paths={'grasp':ROOT/'runs/sac_grasp_bc_finetune01/checkpoints/best/model.zip',
           'carry_bc500':ROOT/'runs/carry_bc_init01/checkpoints/best/model.zip'}
    hashes={}
    for name,path in paths.items():
        hashes[name]=sha(path);shutil.copy2(path,inputs/f'{name}.zip')
        shutil.copy2(path.parent/'evaluation.json',inputs/f'{name}_selection.json')
        if hashes[name]!=sha(inputs/f'{name}.zip'):raise RuntimeError('Checkpoint changed while freezing')
    if json.loads((inputs/'carry_bc500_selection.json').read_text())['bc_updates']!=500:
        raise ValueError('The requested carry diagnostic requires the selected BC500 checkpoint')
    dataset_path=ROOT/'runs/carry_physical_demos_v1/artifacts/demonstrations.pkl'
    dataset_hash=sha(dataset_path);dataset=pickle.loads(dataset_path.read_bytes())
    episode_index=next(i for i,e in enumerate(dataset['episodes'])
        if e['summary']['kind']=='full_route' and e['summary']['snapshot_index']==0)
    reference=copy.deepcopy(dataset['episodes'][episode_index]['transitions'][0]['observation'])
    provenance={'dataset':str(dataset_path.relative_to(ROOT)),'dataset_sha256':dataset_hash,
                'episode_index':episode_index,'transition_index':0,'snapshot_index':0,
                'summary':dataset['episodes'][episode_index]['summary']}
    atomic_json(inputs/'reference_provenance.json',provenance)
    source_hashes={name:sha(ROOT/name) for name in ('pnr/env.py','pnr/skills.py','pnr/carry.py',
        'pnr/system.py','pnr/reference.py','pnr/planning.py','scripts/audit_carry_handoff.py')}
    atomic_json(logger.directory/'config.json',{'training':False,'new_carry_policy_decisions':0,
        'policy_sha256':hashes,'reference':provenance,'source_hashes':source_hashes})
    native=None;started=time.perf_counter()
    try:
        grasp=load_policy_checkpoint(inputs/'grasp.zip',device='cpu')
        carry=load_policy_checkpoint(inputs/'carry_bc500.zip',device='cpu')
        native=PickNavReachEnv(seed=42,object_idx=14)
        system=LearnedSystem(native,{'grasp':grasp,'carry':UnusedPolicy(),'reach':UnusedPolicy()},grasp_horizon=200)
        with system._observe_native_steps():
            stage=system._begin('planned_approach','planned')
            proposal=system.controller.prepare_grasp()
            system._finish('planned_approach',stage,True,grasp=proposal.as_dict())
            system._skill('grasp')
            system._held_pause('pick_hold',50)
            stage=system._begin('planned_carry_posture','planned')
            system.controller.prepare_carry()
            held=native.contacts()['bilateral_contact']
            system._finish('planned_carry_posture',stage,held)
            if not held:raise StageFailure('Carry posture lost physical contact')
        wrapper=CarrySkillEnv.bind(native,system.controller,horizon=1000)
        actual=wrapper._observation()
        if not np.array_equal(actual['desired_goal'],reference['desired_goal']):
            raise AssertionError('Reference and learned handover do not have the same actual fixed goal')
        physical_calls_before=native.step_count
        reference_action=carry.predict(reference,deterministic=True)[0]
        actual_action=carry.predict(actual,deterministic=True)[0]
        comparisons={}
        for name,part in GROUPS.items():
            first=reference['observation'][part];second=actual['observation'][part]
            delta=second-first
            comparisons[name]={'max_abs_difference':float(np.max(np.abs(delta))),
                'l2_difference':float(np.linalg.norm(delta)),'different_entries':int(np.count_nonzero(delta))}
            if not name.endswith('map'):
                comparisons[name].update(reference=first.tolist(),actual=second.tolist(),delta=delta.tolist())
            changed=copy.deepcopy(actual);changed['observation'][part]=first
            action=carry.predict(changed,deterministic=True)[0]
            comparisons[name]['action_if_only_this_input_group_replaced_by_reference']=action.tolist()
            comparisons[name]['counterfactual_action_change_l2']=float(np.linalg.norm(action-actual_action))
        for name,part in [('ee_orientation',slice(561,565)),('object_orientation',slice(568,572))]:
            comparisons[name]={'rotation_difference_rad':float((Rotation.from_quat(reference['observation'][part]).inv()*
                Rotation.from_quat(actual['observation'][part])).magnitude())}
        if native.step_count!=physical_calls_before:raise AssertionError('Inference unexpectedly advanced physics')
        np.savez_compressed(artifacts/'observations.npz',**{f'{prefix}_{key}':value
            for prefix,obs in [('reference',reference),('actual',actual)] for key,value in obs.items()})
        report={'scope':'carry_handover_observation_audit','training':False,'full_task_evaluation':False,
            'reference':provenance,'policy_sha256':hashes,'source_hashes':source_hashes,
            'reference_state':decode(reference),'actual_state':decode(actual),'field_comparison':comparisons,
            'initial_carry_bc500_action':{'reference_snapshot_zero':reference_action.tolist(),
                'actual_learned_handover':actual_action.tolist(),'l2_difference':float(np.linalg.norm(actual_action-reference_action)),
                'executed':False},'actual_handover_contact':native.contacts(),'actual_stages':system.stage_results,
            'counts':{'learned_grasp_decisions':system.policy_decisions['grasp'],'learned_carry_decisions':0,
                'reference_reuse_new_source_action_calls':0,'source_calls_by_controller':system.calls_by_source,
                'new_source_action_calls':native.step_count,'new_action_physics_ticks':native.physics_steps,
                'new_initialization_physics_ticks':native.settle_steps,'total_new_physics_ticks':native.physics_steps+native.settle_steps},
            'wall_seconds':time.perf_counter()-started,
            'interpretation_limit':'One actual learned handover versus recorded reference training state. Input substitutions are offline nonphysical sensitivity queries, not rollout evidence.'}
        atomic_json(artifacts/'comparison.json',report)
        logger.artifact('artifacts/comparison.json','Matched carry observation fields and unexecuted BC500 actions','file')
        logger.artifact('artifacts/observations.npz','Exact recorded and actual observation dictionaries','file')
        logger.finish(summary=f"Compared recorded carry snapshot0 to actual learned handover; initial BC500 action difference {report['initial_carry_bc500_action']['l2_difference']:.3f}. No carry rollout.")
        print(json.dumps({key:value for key,value in report.items() if key not in ('reference','source_hashes','field_comparison','actual_stages','reference_state','actual_state')},indent=2),flush=True)
    except BaseException as error:
        atomic_json(artifacts/'failure.json',{'error':f'{type(error).__name__}: {error}',
            'new_source_action_calls':0 if native is None else native.step_count,
            'new_action_physics_ticks':0 if native is None else native.physics_steps,
            'initialization_physics_ticks':0 if native is None else native.settle_steps})
        logger.finish('failed',f'{type(error).__name__}: {error}');raise
    finally:
        if native is not None:native.close()


if __name__=='__main__':main()
