#!/usr/bin/env python3
"""Bounded, disposable actor timing; no physics, policy selection, or checkpoints."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import SAC
from pnr.gated_spatial_features import GatedCarrySpatialExtractor
from pnr.multi_carry import CarrySchema
from pnr.spatial_features import CarrySpatialExtractor


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(fn,device,repeats,warmups=3):
    def synchronize():
        if device=='cuda':torch.cuda.synchronize()
    for _ in range(warmups):fn()
    synchronize()
    allocated=torch.cuda.memory_allocated() if device=='cuda' else None
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(repeats):
        synchronize();started=time.perf_counter();fn();synchronize()
        times.append((time.perf_counter()-started)*1000.)
    return {'median_ms':float(np.median(times)),'p90_ms':float(np.quantile(times,.9)),
        'mean_ms':float(np.mean(times)),'min_ms':min(times),'max_ms':max(times),
        'repeats':repeats,'warmup_passes':warmups,'individual_ms':times,
        'cuda_total_peak_allocated_mib':None if device!='cuda' else torch.cuda.max_memory_allocated()/2**20,
        'cuda_peak_above_start_mib':None if device!='cuda' else (torch.cuda.max_memory_allocated()-allocated)/2**20}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',nargs='+',choices=['cpu','cuda'],default=['cpu','cuda'])
    p.add_argument('--cpu-train-repeats',type=int,default=5)
    p.add_argument('--cuda-train-repeats',type=int,default=15)
    p.add_argument('--inference-repeats',type=int,default=25)
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh benchmark output')
    if min(args.cpu_train_repeats,args.cuda_train_repeats,args.inference_repeats)<1:p.error('Positive repeats required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    source=sorted((ROOT/'runs/phase2_carry_demos01/artifacts').glob('layout_*.pkl'))[0]
    with source.open('rb') as stream:data=pickle.load(stream)
    rows=[row for episode in data['episodes'] if episode['success'] for row in episode['transitions']]
    indices=np.random.default_rng(19031).choice(len(rows),256,replace=True)
    observations={key:np.stack([rows[index]['observation'][key] for index in indices]) for key in rows[0]['observation']}
    targets=np.stack([rows[index]['teacher_action'] for index in indices])
    cpu_name=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    report={'kind':'disposable_encoder_throughput','physics_ticks':0,'source_action_calls':0,
        'policy_checkpoints_created':0,'validation_rollouts':0,'cpu_name':cpu_name,'cpu_threads':4,
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
        'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'numerics':'float32; matmul high; cuDNN benchmark enabled and deterministic disabled',
        'measurement_note':'Shared host/GPU with project jobs. Per-call wall time includes synchronization. These are timings, not learned-policy outcomes.',
        'training_measurement':'Actual continuous SAC actor BC MSE, Adam3e-4 and gradient clipping10; no critic/temperature update or complete online SAC step.',
        'inference_measurement':'SB3 predict includes numpy observation conversion, transfers, continuous actor head and numpy action output.',
        'sample_source':str(source.relative_to(ROOT)),'sample_source_sha256':digest(source),'batch_size':256,'seed':19031,
        'sources':{name:digest(ROOT/name) for name in ('pnr/spatial_features.py','pnr/navigation_features.py','pnr/gated_spatial_features.py','scripts/phase2_gated_encoder_benchmark.py')},
        'results':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    started=time.monotonic();save()
    candidates=[('cnn32',CarrySpatialExtractor,{'channels':32}),
        ('gated32_k20_f5',GatedCarrySpatialExtractor,{'channels':32,'iterations':20,'kernel_size':5}),
        ('gated48_k20_f5',GatedCarrySpatialExtractor,{'channels':48,'iterations':20,'kernel_size':5})]
    for device in args.devices:
        if device=='cuda' and not torch.cuda.is_available():continue
        for name,extractor,kwargs in candidates:
            model=SAC('MultiInputPolicy',CarrySchema(),buffer_size=32,batch_size=256,seed=19031,
                policy_kwargs={'net_arch':[128,128],'features_extractor_class':extractor,
                    'features_extractor_kwargs':dict(features_dim=128,**kwargs)},device=device)
            # SAC's seeding forces deterministic cuDNN; match our actual runner.
            torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True
            torch.set_float32_matmul_precision('high')
            model.actor.features_extractor.set_carry_pose_mask(True)
            record={'name':name,'device':device,'encoder_parameters':sum(x.numel() for x in model.actor.features_extractor.parameters()),
                'actor_parameters':sum(x.numel() for x in model.actor.parameters()),
                'critic_parameters':sum(x.numel() for x in model.critic.parameters()),'kwargs':kwargs,'cases':{}}
            model.policy.set_training_mode(False)
            for size in (1,8):
                sample={key:value[:size] for key,value in observations.items()}
                def infer():
                    with torch.inference_mode():model.predict(sample,deterministic=True)
                record['cases'][f'predict_batch{size}']=timing(infer,device,args.inference_repeats)
            model.policy.set_training_mode(True)
            batch={key:torch.as_tensor(value,device=device) for key,value in observations.items()}
            target=torch.as_tensor(targets,device=device)
            optimizer=torch.optim.Adam(model.actor.parameters(),lr=3e-4)
            losses=[]
            def train():
                prediction=model.actor(batch,deterministic=True)
                loss=(prediction-target).square().mean()
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(),10.)
                optimizer.step()
                losses.append(float(loss.detach()))
            repeats=args.cpu_train_repeats if device=='cpu' else args.cuda_train_repeats
            record['cases']['bc_step_batch256']=timing(train,device,repeats)
            record['disposable_bc_updates']=len(losses)
            record['finite_losses']=bool(np.isfinite(losses).all())
            assert record['finite_losses']
            for case,measurement in record['cases'].items():
                size=256 if case.startswith('bc_') else int(case.rsplit('batch',1)[1])
                measurement['examples_per_second']=size*1000./measurement['median_ms']
            report['results'].append(record);save()
            print(json.dumps({'model':name,'device':device,'encoder_parameters':record['encoder_parameters'],
                'actor_parameters':record['actor_parameters'],'median_ms':{key:value['median_ms'] for key,value in record['cases'].items()},
                'peak_cuda_mib':record['cases']['bc_step_batch256']['cuda_total_peak_allocated_mib']}),flush=True)
            model.env.close();del model,batch,target,optimizer;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
    report['wall_seconds']=time.monotonic()-started
    report['total_disposable_bc_updates']=sum(row['disposable_bc_updates'] for row in report['results'])
    report['status']='complete';save()
    print(json.dumps({'status':'complete','wall_seconds':report['wall_seconds'],'disposable_bc_updates':report['total_disposable_bc_updates'],'physics_ticks':0}),flush=True)


if __name__=='__main__':main()
