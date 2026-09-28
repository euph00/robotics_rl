"""Immutable phase-2 SAC checkpoint bundles and conservative resume validation."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from pnr.telemetry import atomic_json

VERSION = 'phase2-checkpoint-v1'
BASE_METHOD_FIELDS = ('workers','layouts','seed','horizon','delta_scale','batch_size','learning_rate',
                 'gamma','shaping_weight','target_entropy','n_sampled_goal',
                 'demonstrations_sha256','scene_bank_sha256')
METHOD_FIELDS = BASE_METHOD_FIELDS + ('bc_weight','demo_batch_size')
SEMANTIC_SOURCES = ('pnr/env.py','pnr/carry.py','pnr/general_carry.py','pnr/multi_carry.py',
                    'pnr/spatial_features.py','pnr/reference.py','pnr/planning.py',
                    'pnr/local_control.py','pnr/scenes.py')


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def _resource_fingerprint(path):
    tree=ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=='MultiSceneCarry':
            node.body=[method for method in node.body if not (isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and method.name=='prime_cache')]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def _normalized_method(method):
    """Absent regularization fields in an old checkpoint mean ordinary SAC."""
    result=dict(method)
    weight=float(result.get('bc_weight',0.))
    if not np.isfinite(weight) or weight<0:raise ValueError('Invalid checkpoint bc_weight')
    result['bc_weight']=weight
    # This setting is inactive for plain SAC, including pre-regularization runs.
    size=result.get('demo_batch_size',256) if weight>0 else None
    if weight>0 and (isinstance(size,bool) or not isinstance(size,(int,np.integer)) or size<1):
        raise ValueError('Invalid checkpoint demo_batch_size')
    result['demo_batch_size']=size
    return result


def _method(config):
    result={key:config[key] for key in BASE_METHOD_FIELDS}
    result.update({key:config[key] for key in ('bc_weight','demo_batch_size') if key in config})
    return _normalized_method(result)


def _semantic_sources(method):
    return SEMANTIC_SOURCES+(('pnr/regularized_sac.py',) if method['bc_weight']>0 else ())


def signature(config,source_root):
    root=Path(source_root);method=_method(config)
    return {'version':VERSION,'method':method,
            'semantic_sources':{key:digest(root/key) for key in _semantic_sources(method)},
            'resource_normalizations':{'pnr/multi_carry.py':_resource_fingerprint(root/'pnr/multi_carry.py')}}


def check_signature(expected,actual):
    if expected.get('version')!=VERSION or actual.get('version')!=VERSION:
        raise ValueError('Unsupported phase-2 checkpoint protocol')
    for group in ('method','semantic_sources'):
        before_group=_normalized_method(expected[group]) if group=='method' else expected[group]
        after_group=_normalized_method(actual[group]) if group=='method' else actual[group]
        if set(before_group)!=set(after_group):raise ValueError('Resume signature fields differ: '+group)
        for key,value in before_group.items():
            if after_group[key]!=value:
                if group=='semantic_sources' and key=='pnr/multi_carry.py':
                    before=expected.get('resource_normalizations',{}).get(key)
                    after=actual.get('resource_normalizations',{}).get(key)
                    if before is not None and before==after:continue
                raise ValueError(f'Resume changes {group} {key}')


def capture_rng():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def _sync(path):
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def publish_checkpoint(model,run_directory,state,resume_signature,label):
    """Publish latest only after a complete, hashed and flushed bundle exists."""
    root=Path(run_directory)/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    if getattr(model.env,'waiting',False):raise RuntimeError('Checkpoint cannot read pending worker replies')
    generation=f'step_{model.num_timesteps:09d}_{time.time_ns()}'
    temporary=root/('.'+generation+'.pending');target=root/generation
    temporary.mkdir()
    try:
        model.save(temporary/'model.zip');model.save_replay_buffer(temporary/'replay.pkl')
        torch.save(capture_rng(),temporary/'rng.pt')
        payload=dict(state,checkpoint_version=VERSION,label=label,resume_signature=resume_signature,
                     online_decisions=int(model.num_timesteps),online_gradient_updates=int(model._n_updates),
                     replay_position=int(model.replay_buffer.pos),replay_workers=int(model.replay_buffer.n_envs),
                     resume_note='Restore learner/RNG/costs; truncate replay tails and restart physical episodes.')
        atomic_json(temporary/'state.json',payload)
        files=('model.zip','replay.pkl','rng.pt','state.json')
        for name in files:_sync(temporary/name)
        manifest={'version':VERSION,'generation':generation,'files':{name:digest(temporary/name) for name in files}}
        atomic_json(temporary/'manifest.json',manifest);_sync_directory(temporary)
        os.replace(temporary,target);_sync_directory(root)
        atomic_json(root/'latest.json',{'version':VERSION,'generation':generation,
                                      'manifest_sha256':digest(target/'manifest.json')})
        _sync_directory(root)
        return {'generation':generation,'directory':str(target),'manifest_sha256':digest(target/'manifest.json')}
    except BaseException:
        if temporary.exists():shutil.rmtree(temporary)
        raise


def load_checkpoint(run_directory,current_signature):
    """Read/validate without mutating the parent; supports stopped legacy runs."""
    directory=Path(run_directory).resolve();root=directory/'checkpoints'
    pointer=json.loads((root/'latest.json').read_text())
    if pointer.get('version')==VERSION:
        name=pointer['generation']
        if Path(name).name!=name:raise ValueError('Invalid checkpoint generation path')
        folder=root/name;manifest_path=folder/'manifest.json'
        if digest(manifest_path)!=pointer['manifest_sha256']:raise ValueError('Checkpoint manifest hash mismatch')
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('version')!=VERSION or manifest.get('generation')!=name:raise ValueError('Checkpoint manifest identity mismatch')
        if set(manifest['files'])!={'model.zip','replay.pkl','rng.pt','state.json'}:raise ValueError('Incomplete checkpoint bundle')
        for filename,expected in manifest['files'].items():
            if digest(folder/filename)!=expected:raise ValueError('Checkpoint file hash mismatch: '+filename)
        state=json.loads((folder/'state.json').read_text());check_signature(state['resume_signature'],current_signature)
        return {'run_directory':directory,'generation':name,'model':folder/'model.zip',
                'replay':folder/'replay.pkl','rng':folder/'rng.pt','state':state,
                'manifest_sha256':pointer['manifest_sha256'],'legacy':False}
    # Old phase2 runner used mutable files. Refuse reading a running writer.
    status=json.loads((directory/'status.json').read_text())
    if status.get('status')=='running':raise ValueError('Stop the legacy run before reading its mutable checkpoint')
    config=json.loads((directory/'config.json').read_text())
    sources=json.loads((directory/'source_manifest.json').read_text())
    old_method=_method(config)
    old_signature={'version':VERSION,'method':old_method,
                   'semantic_sources':{key:sources[key] for key in _semantic_sources(old_method)}}
    archived=directory/'sources/pnr/multi_carry.py'
    if archived.is_file():
        if digest(archived)!=sources['pnr/multi_carry.py']:raise ValueError('Archived scene-rotation source hash mismatch')
        old_signature['resource_normalizations']={'pnr/multi_carry.py':_resource_fingerprint(archived)}
    check_signature(old_signature,current_signature)
    name=pointer.get('model')
    if not isinstance(name,str) or Path(name).name!=name:raise ValueError('Invalid legacy model pointer')
    paths={'model':root/name,'replay':root/'latest_replay.pkl','rng':root/'rng.pt'}
    for path in paths.values():
        if not path.is_file():raise ValueError('Incomplete legacy checkpoint: '+str(path))
    if pointer.get('online_decisions',0)>0 and not pointer.get('training_states'):
        raise ValueError('Legacy online checkpoint has no worker states')
    state=dict(pointer,resume_signature=old_signature,
               legacy_missing=['recent_episode_window','exact_callback_last_eval','callback_best_reference'])
    best_path=root/'best_online_evaluation.json'
    if best_path.is_file():
        best=json.loads(best_path.read_text())
        state['callback_state']={'best':[best['success_rate'],-best['goal_distance']],
                                 'best_checkpoint':str(root/'best_online.zip')}
    return dict(run_directory=directory,generation='legacy:'+name,state=state,legacy=True,
                manifest_sha256=None,legacy_file_hashes={key:digest(path) for key,path in paths.items()},**paths)


def validate_loaded_model(model,checkpoint,workers):
    state=checkpoint['state'];replay=model.replay_buffer
    if model.n_envs!=workers or replay.n_envs!=workers:raise ValueError('Resume replay/model worker count differs')
    if model.num_timesteps!=state['online_decisions'] or model._n_updates!=state['online_gradient_updates']:
        raise ValueError('Resume model counters differ from checkpoint state')
    if not getattr(replay,'copy_info_dict',False):raise ValueError('HER physical reward information missing')
    method=_normalized_method(state['resume_signature']['method'])
    declared_weight=method['bc_weight']
    if float(getattr(model,'bc_weight',0.))!=declared_weight:
        raise ValueError('Saved SAC bc_weight differs from declared method')
    if declared_weight>0:
        from pnr.regularized_sac import RegularizedSAC
        if not isinstance(model,RegularizedSAC):raise ValueError('Regularized checkpoint requires RegularizedSAC.load')
        if (model.demo_batch_size or model.batch_size)!=method['demo_batch_size']:
            raise ValueError('Saved SAC demo_batch_size differs from declared method')
        if model.demo_manifest is None:raise ValueError('Regularized checkpoint lacks demonstration manifest')
    if model.gamma!=method['gamma'] or model.target_entropy!=method['target_entropy']:
        raise ValueError('Saved SAC discount/entropy differs from declared method')
    if replay.n_sampled_goal!=method['n_sampled_goal'] or model.batch_size!=method['batch_size']:
        raise ValueError('Saved SAC replay/batch settings differ from declared method')
    states=state.get('training_states',[])
    if len(states)!=workers and not (model.num_timesteps==0 and not states):
        raise ValueError('Resume worker-state count differs')
    if np.any(replay._current_ep_start!=replay.pos):raise ValueError('Replay tails were not truncated before physical restart')
    if not checkpoint['legacy'] and replay.pos!=state['replay_position']:
        raise ValueError('Saved replay position differs from checkpoint state')
    if checkpoint['legacy']:
        seeded=state.get('initialization',{}).get('demonstrations',{}).get('transitions_imported')
        if seeded is not None:
            expected=(int(seeded)//workers+model.num_timesteps//workers)%replay.buffer_size
            omitted_vectors=(expected-int(replay.pos))%replay.buffer_size
            if omitted_vectors>1:
                raise ValueError('Legacy replay position differs by more than one interrupted vector step')
            if omitted_vectors:
                state.setdefault('legacy_missing',[]).append('one executed vector transition absent from legacy replay')
                state['legacy_omitted_online_decisions']=workers

