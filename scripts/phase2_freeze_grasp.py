#!/usr/bin/env python3
"""Freeze completed grasp pilot candidates with actual counters and contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stable_baselines3.common.save_util import load_from_zip_file
from scripts.evaluate_system import freeze_checkpoints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    status=json.loads((run/'status.json').read_text())
    if status['status']!='complete':raise ValueError('Only completed pilots may be frozen by this script')
    cfg=json.loads((run/'config.json').read_text())
    result=json.loads((run/'artifacts/result.json').read_text())
    if result.get('stage')=='bc_only':
        selected_update=result['initialization']['bc']['selected_update']
        sources={'grasp':run/'checkpoints/bc_initialized.zip'}
        evaluated={'grasp':run/'artifacts'/f'evaluation_bc_{selected_update:06d}.json'}
        selection='Selected completed BC check; no critic calibration or online updates.'
    else:
        selected=json.loads((run/'checkpoints/best_online_evaluation.json').read_text())
        sources={'grasp':run/'checkpoints/best_online.zip','grasp_final':run/'checkpoints/final.zip',
                 'grasp_zero_online':run/'checkpoints/initialized.zip'}
        evaluated={'grasp':Path(selected['evaluation']).resolve(),
            'grasp_final':run/'artifacts'/f"evaluation_final_{result['online_decisions']:07d}.json",
            'grasp_zero_online':run/'artifacts/evaluation_initialized_zero_online.json'}
        selection='Best online validation success, then lowest mean goal distance.'
    records={}
    for name,path in sources.items():
        data,_,_=load_from_zip_file(path,device='cpu')
        report=json.loads(evaluated[name].read_text())
        if report['online_decisions']!=data['num_timesteps']:
            raise ValueError('Evaluation and checkpoint decision counters differ: '+name)
        if data.get('_pnr_grasp_protocol_signature')!=cfg['grasp_protocol_signature']:
            raise ValueError('Saved inference/training protocol differs from run configuration')
        records[name]={'online_decisions':int(data['num_timesteps']),
            'online_gradient_updates':int(data['_n_updates']),
            'critic_initialization_updates':cfg['warmup_updates'],'training_workers':cfg['workers'],
            'physical_training_workers_created':0 if result.get('stage')=='bc_only' else cfg['workers'],
            'bc_weight':cfg['bc_weight'],'validation_success_rate':report['validation_success_rate'],
            'validation_mean_goal_distance':report['validation_mean_goal_distance'],
            'observation_shapes':{k:list(v.shape) for k,v in data['observation_space'].spaces.items()},
            'action_shape':list(data['action_space'].shape)}
    target=run/'candidate'
    frozen=freeze_checkpoints(sources,target)
    for name in frozen:frozen[name].update(records[name])
    copies={'config.json':run/'config.json','training_result.json':run/'artifacts/result.json',
            'source_manifest.json':run/'source_manifest.json'}
    copies.update({name+'_evaluation.json':path for name,path in evaluated.items()})
    provenance={}
    for name,path in copies.items():
        content=path.read_bytes()
        with (target/name).open('xb') as writer:writer.write(content)
        provenance[name]={'source':str(path),'sha256':hashlib.sha256(content).hexdigest()}
    manifest={'kind':'frozen_grasp_candidate','run':str(run),'checkpoints':frozen,
        'algorithm':cfg['algorithm'],'selection':selection,
        'actor_features':cfg.get('actor_features','full70'),'critic_features':70,
        'grasp_proposal_mode':cfg['grasp_proposal_mode'],'goal_lift_m':cfg['eval_lift_m'],
        'control_version':cfg['control_version'],'teacher_at_inference':False,
        'grasp_protocol_signature':cfg['grasp_protocol_signature'],'artifacts':provenance}
    with (target/'manifest.json').open('x') as writer:json.dump(manifest,writer,indent=2)
    for path in target.iterdir():path.chmod(0o444)
    print(json.dumps({'candidate':str(target),'checkpoints':frozen},indent=2))

if __name__=='__main__':main()
