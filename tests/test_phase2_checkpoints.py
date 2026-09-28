"""Actual SAC/HER persistence, interrupted writes and validation accounting."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC,HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from pnr.phase2_checkpoints import (VERSION,METHOD_FIELDS,SEMANTIC_SOURCES,signature,
    publish_checkpoint,load_checkpoint,restore_rng,validate_loaded_model,capture_rng)
from pnr.telemetry import RunLogger,atomic_json
from scripts.phase2_train_carry import Progress


class SyntheticCarry(gym.Env):
    observation_space=spaces.Dict({'observation':spaces.Box(-10,10,(3,),np.float32),
        'achieved_goal':spaces.Box(-10,10,(2,),np.float32),'desired_goal':spaces.Box(-10,10,(2,),np.float32)})
    action_space=spaces.Box(-1,1,(2,),np.float32)
    setup_action_calls=2;setup_physics_ticks=10;initialization_physics_ticks=3
    def __init__(self):self.steps=0;self.reset_count=0
    def reset(self,*,seed=None,options=None):
        self.steps=0;self.reset_count+=1
        return self.observation(),{}
    def observation(self):
        return {'observation':np.array([self.steps,0,0],np.float32),
            'achieved_goal':np.array([self.steps*.1,0],np.float32),'desired_goal':np.array([.5,0],np.float32)}
    def compute_reward(self,achieved_goal,desired_goal,info):
        return -(np.linalg.norm(achieved_goal-desired_goal,axis=-1)>.1).astype(np.float32)
    def step(self,action):
        self.steps+=1;obs=self.observation();success=self.steps==5
        return obs,float(self.compute_reward(obs['achieved_goal'],obs['desired_goal'],{})),False,self.steps==6,{
            'action_calls_delta':5,'physics_ticks_delta':25,'is_success':success,'episode_success':success}
    def set_curriculum_level(self,level):pass
    def get_training_state(self):return {'reset_count':self.reset_count,'synthetic_test_only':True}


def method():
    result={key:0 for key in METHOD_FIELDS}
    result.update(workers=2,layouts=2,seed=37,horizon=6,delta_scale=.075,batch_size=8,
        learning_rate=3e-4,gamma=.995,shaping_weight=2.,target_entropy=-4.,n_sampled_goal=1,
        demonstrations_sha256='d'*64,scene_bank_sha256='b'*64,buffer_size=128)
    return result


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(1)
    source=tmp_path/'source';(source/'pnr').mkdir(parents=True)
    for name in SEMANTIC_SOURCES:(source/name).write_text('# semantic v1\n')
    config=method();sig=signature(config,source)
    vector=DummyVecEnv([SyntheticCarry,SyntheticCarry])
    model=SAC('MultiInputPolicy',vector,buffer_size=128,batch_size=8,learning_starts=12,gamma=.995,target_entropy=-4.,
        policy_kwargs={'net_arch':[8,8]},replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'copy_info_dict':True,'n_sampled_goal':1},seed=37,device='cpu')
    model.learn(28)
    logger=RunLogger('parent',root=tmp_path)
    args=SimpleNamespace(horizon=6,delta_scale=.075,eval_every=10000,checkpoint_every=10000)
    cb=Progress(logger,[],args,SimpleNamespace(requested=False));cb.model=model;cb.resume_signature=sig
    cb.train_calls=140;cb.train_ticks=700;cb.episodes=4;cb.recent.extend([0.,1.]);cb.initialization={'bc':{'updates':10}}
    cb.checkpoint('first')
    yield model,cb,logger,source,config,sig
    vector.close()


def test_real_sac_her_resume_closes_tail_preserves_rng_and_adds_steps(bundle,tmp_path):
    model,cb,logger,source,config,sig=bundle
    checkpoint=load_checkpoint(logger.directory,sig)
    replacement=DummyVecEnv([SyntheticCarry,SyntheticCarry])
    try:
        loaded=SAC.load(checkpoint['model'],env=replacement,device='cpu')
        with pytest.warns(UserWarning,match='last trajectory'):
            loaded.load_replay_buffer(checkpoint['replay'],truncate_last_traj=True)
        validate_loaded_model(loaded,checkpoint,2)
        assert loaded.num_timesteps==28 and loaded._n_updates==model._n_updates
        assert np.all(loaded.replay_buffer._current_ep_start==loaded.replay_buffer.pos)
        assert np.all(loaded.replay_buffer.timeouts[loaded.replay_buffer.pos-1]==1)
        restore_rng(checkpoint['rng']);expected=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng(checkpoint['rng']);actual=(random.random(),np.random.rand(),torch.rand(3))
        assert actual[:2]==expected[:2];torch.testing.assert_close(actual[2],expected[2])
        child=RunLogger('child',root=tmp_path)
        resumed=Progress(child,[],cb.args,cb.stop);resumed.model=loaded;resumed.resume_signature=sig;resumed.restore(checkpoint)
        assert resumed.episodes==4 and list(resumed.recent)==[0.,1.]
        assert resumed.accounting()['training_setup_source_action_calls']==8  # 4 inherited +4 new
        loaded.learn(total_timesteps=8,callback=resumed,reset_num_timesteps=False)
        assert loaded.num_timesteps==36 and resumed.train_calls==180
        assert resumed.accounting()['incremental_online_decisions']==8
        assert resumed.accounting()['incremental_accounting']['training_source_action_calls']==40
        resumed.checkpoint('continued');state=load_checkpoint(child.directory,sig)['state']
        assert state['online_decisions']==36 and state['ancestry'][0]['run']=='parent'
        assert state['initialization']==cb.initialization
    finally:replacement.close()


def test_failed_publication_cannot_replace_complete_latest(bundle,monkeypatch):
    model,cb,logger,_,_,sig=bundle
    previous=(logger.directory/'checkpoints/latest.json').read_bytes()
    def interrupted(path):Path(path).write_bytes(b'partial');raise OSError('simulated disk interruption')
    monkeypatch.setattr(model,'save_replay_buffer',interrupted)
    with pytest.raises(OSError,match='disk interruption'):cb.checkpoint('interrupted')
    assert (logger.directory/'checkpoints/latest.json').read_bytes()==previous
    assert load_checkpoint(logger.directory,sig)['state']['online_decisions']==28
    assert not list((logger.directory/'checkpoints').glob('.*.pending'))


def test_hash_and_semantic_or_worker_changes_are_rejected_but_logging_allowed(bundle):
    model,cb,logger,source,config,sig=bundle
    (source/'pnr/telemetry.py').write_text('# logging-only change')
    load_checkpoint(logger.directory,signature(config,source))
    changed=dict(config,workers=3)
    with pytest.raises(ValueError,match='workers'):load_checkpoint(logger.directory,signature(changed,source))
    (source/'pnr/general_carry.py').write_text('# changed reward')
    with pytest.raises(ValueError,match='general_carry'):load_checkpoint(logger.directory,signature(config,source))
    checkpoint=load_checkpoint(logger.directory,sig)
    with checkpoint['replay'].open('ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):load_checkpoint(logger.directory,sig)


def test_legacy_checkpoint_is_compatible_only_with_stopped_matching_run(bundle,tmp_path):
    model,cb,_,source,config,sig=bundle
    root=tmp_path/'legacy';out=root/'checkpoints';out.mkdir(parents=True)
    model.save(out/'online_0000028.zip');model.save_replay_buffer(out/'latest_replay.pkl');torch.save(capture_rng(),out/'rng.pt')
    atomic_json(root/'config.json',config);atomic_json(root/'source_manifest.json',sig['semantic_sources'])
    atomic_json(root/'status.json',{'status':'running'})
    atomic_json(out/'latest.json',{'model':'online_0000028.zip','online_decisions':28,'online_gradient_updates':model._n_updates,
        'accounting':cb.accounting(),'initialization':cb.initialization,'training_states':model.env.env_method('get_training_state')})
    with pytest.raises(ValueError,match='Stop the legacy'):load_checkpoint(root,sig)
    atomic_json(root/'status.json',{'status':'complete'})
    checkpoint=load_checkpoint(root,sig)
    assert checkpoint['legacy'] and checkpoint['state']['online_decisions']==28
    assert set(checkpoint['legacy_file_hashes'])=={'model','replay','rng'}


def test_stop_pending_transition_is_saved_once(bundle):
    model,old,logger,_,_,sig=bundle
    cb=Progress(logger,[],old.args,SimpleNamespace(requested=True));cb.model=model;cb.resume_signature=sig
    model.learn(8,callback=cb,reset_num_timesteps=False)
    before=model.replay_buffer.pos
    cb.checkpoint('stop');assert model.replay_buffer.pos==(before+1)%model.replay_buffer.buffer_size
    cb.checkpoint('stop_again');assert model.replay_buffer.pos==(before+1)%model.replay_buffer.buffer_size
    assert load_checkpoint(logger.directory,sig)['state']['online_decisions']==30


def test_setup_failures_count_and_partial_validation_cannot_rank(bundle,monkeypatch):
    model,old,logger,_,_,sig=bundle
    layouts=[{'layout_id':f'validation_{i:03d}','maze_seed':i,'maze_hash':str(i)} for i in range(4)]
    cb=Progress(logger,layouts,old.args,SimpleNamespace(requested=False));cb.model=model
    def failed(**kwargs):
        exc=RuntimeError('physical grasp failed');exc.setup_accounting={'source_action_calls':7,'physics_ticks':35,'initialization_physics_ticks':3};raise exc
    monkeypatch.setattr('scripts.phase2_train_carry.GeneralCarryEnv',failed)
    records=cb.evaluate('all_setup_failed')
    assert len(records)==4 and all(r['goal_distance'] is None for r in records)
    assert cb.best==(0.,-30.) and cb.eval_setup_calls==28
    result=json.loads((logger.directory/'artifacts/evaluation_all_setup_failed.json').read_text())
    assert result['validation_complete'] and result['mean_measured_goal_distance'] is None
    assert result['selection_distance_penalty']==30.
    cb.best=(-1.,-float('inf'));calls=[]
    def interrupted(**kwargs):
        calls.append(1)
        if len(calls)==2:cb.stop.requested=True
        return failed(**kwargs)
    monkeypatch.setattr('scripts.phase2_train_carry.GeneralCarryEnv',interrupted)
    cb.evaluate('partial')
    assert cb.best==(-1.,-float('inf'))
    assert not json.loads((logger.directory/'artifacts/evaluation_partial.json').read_text())['validation_complete']


def test_resume_cli_skips_initialization_and_treats_steps_as_additional(bundle,tmp_path,monkeypatch):
    import shutil
    import scripts.phase2_train_carry as runner
    model,cb,logger,source,config,_=bundle
    data=tmp_path/'data.pkl'
    import pickle
    data.write_bytes(pickle.dumps({'episodes':[{'maze_hash':'trainhash','transitions':[]}]}))
    from pnr.phase2_checkpoints import digest
    config=dict(config,demonstrations=str(data),bank='unused',demonstrations_sha256=digest(data))
    sig=signature(config,source);cb.resume_signature=sig;cb.checkpoint('cli_parent')
    atomic_json(logger.directory/'config.json',config)
    (source/'runs').mkdir();shutil.copytree(logger.directory,source/'runs/parent')
    layouts=[{'layout_id':f'train_{i:03d}','maze_seed':i,'maze_hash':'trainhash'} for i in range(2)]
    class Physical(SyntheticCarry):
        def __init__(self,*args,**kwargs):super().__init__()
        def set_training_state(self,state):self.reset_count=state['reset_count']
        def prime_cache(self):return {'scenes':1,'setup_source_action_calls':0,'setup_physics_ticks':0}
    def forbidden(*args,**kwargs):raise AssertionError('Resume repeated BC, replay seeding or critic warmup')
    monkeypatch.setattr(runner,'ROOT',source)
    monkeypatch.setattr(runner,'RunLogger',lambda run_id,**kwargs:RunLogger(run_id,root=source/'runs',**kwargs))
    monkeypatch.setattr(runner,'load_scene_bank',lambda path:{'manifest_sha256':'b'*64})
    monkeypatch.setattr(runner,'scene_records',lambda bank,partition:layouts if partition=='train' else [dict(layouts[0],layout_id=f'validation_{i:03d}') for i in range(4)])
    monkeypatch.setattr(runner,'CarrySchema',SyntheticCarry)
    monkeypatch.setattr(runner,'MultiSceneCarry',Physical)
    monkeypatch.setattr(runner,'SubprocVecEnv',lambda factories,**kwargs:DummyVecEnv(factories))
    monkeypatch.setattr(runner,'ignore_worker_stop_signals',lambda _:None)
    monkeypatch.setattr(runner,'archive_sources',lambda _:None)
    monkeypatch.setattr(runner.Progress,'evaluate',lambda self,label:[])
    monkeypatch.setattr(runner,'seed_replay',forbidden)
    monkeypatch.setattr(runner,'critic_only_warmup',forbidden)
    monkeypatch.setattr(runner,'calibrate_actor_std',forbidden)
    monkeypatch.setattr('sys.argv',['phase2_train_carry.py','--run-id','cli_child','--resume-run','parent','--steps','8','--device','cpu'])
    with pytest.warns(UserWarning,match='last trajectory'):runner.main()
    child=source/'runs/cli_child'
    result=json.loads((child/'artifacts/result.json').read_text())
    assert result['online_decisions']==36 and result['accounting']['incremental_online_decisions']==8
    assert result['initialization']==cb.initialization
    assert result['ancestry'][0]['run']=='parent'
    assert result['accounting']['training_source_action_calls']==180
    assert load_checkpoint(child,sig)['state']['online_decisions']==36


def test_only_additive_cache_priming_is_exempt_from_scene_source_hash(tmp_path):
    from pnr.phase2_checkpoints import check_signature
    (tmp_path/'pnr').mkdir()
    for name in SEMANTIC_SOURCES:(tmp_path/name).write_text('# unchanged\n')
    path=tmp_path/'pnr/multi_carry.py'
    original='class MultiSceneCarry:\n    def reset(self):\n        return 1\n'
    path.write_text(original);before=signature(method(),tmp_path)
    path.write_text(original+'    def prime_cache(self):\n        self.cache = {}\n')
    check_signature(before,signature(method(),tmp_path))
    path.write_text(path.read_text().replace('return 1','return 2'))
    with pytest.raises(ValueError,match='multi_carry'):check_signature(before,signature(method(),tmp_path))


def test_legacy_missing_one_interrupted_vector_is_explicitly_flagged(bundle):
    model,cb,logger,_,_,sig=bundle
    # One vector was physically executed/counted but old SB3 callback stopped
    # before replay insertion. The model's interaction count remains truthful.
    replay=model.replay_buffer
    model.num_timesteps+=model.n_envs
    replay.truncate_last_trajectory()
    checkpoint={'legacy':True,'state':{'online_decisions':model.num_timesteps,
        'online_gradient_updates':model._n_updates,'resume_signature':sig,
        'training_states':model.env.env_method('get_training_state'),
        'initialization':{'demonstrations':{'transitions_imported':0}}}}
    validate_loaded_model(model,checkpoint,2)
    assert checkpoint['state']['legacy_omitted_online_decisions']==2
    assert 'one executed vector transition absent from legacy replay' in checkpoint['state']['legacy_missing']
    model.num_timesteps+=2
    checkpoint['state']['online_decisions']=model.num_timesteps
    with pytest.raises(ValueError,match='more than one'):validate_loaded_model(model,checkpoint,2)


def test_regularization_signature_is_explicit_but_plain_legacy_compatible(bundle):
    from pnr.phase2_checkpoints import check_signature
    _,_,logger,source,config,sig=bundle
    old=copy.deepcopy(sig)
    old['method'].pop('bc_weight');old['method'].pop('demo_batch_size')
    check_signature(old,signature(dict(config,bc_weight=0.,demo_batch_size=256),source))
    (source/'pnr/regularized_sac.py').write_text('# auxiliary objective v1\n')
    regular=signature(dict(config,bc_weight=10.,demo_batch_size=256),source)
    assert 'pnr/regularized_sac.py' in regular['semantic_sources']
    assert 'pnr/regularized_sac.py' not in sig['semantic_sources']
    for before,after in [(old,regular),(regular,old),
                         (regular,signature(dict(config,bc_weight=100.,demo_batch_size=256),source))]:
        with pytest.raises(ValueError,match='bc_weight'):check_signature(before,after)
    with pytest.raises(ValueError,match='demo_batch_size'):
        check_signature(regular,signature(dict(config,bc_weight=10.,demo_batch_size=128),source))
    (source/'pnr/regularized_sac.py').write_text('# different auxiliary objective\n')
    with pytest.raises(ValueError,match='regularized_sac'):
        check_signature(regular,signature(dict(config,bc_weight=10.,demo_batch_size=256),source))
    load_checkpoint(logger.directory,signature(dict(config,bc_weight=0.,demo_batch_size=256),source))


def test_loaded_regularized_identity_cannot_be_claimed_for_plain_sac(bundle):
    model,cb,logger,_,_,sig=bundle
    model.replay_buffer.truncate_last_trajectory()
    checkpoint=load_checkpoint(logger.directory,sig)
    checkpoint['state']['resume_signature']['method'].update(bc_weight=10.,demo_batch_size=256)
    with pytest.raises(ValueError,match='bc_weight'):validate_loaded_model(model,checkpoint,2)
    # Even extra ZIP metadata does not turn SAC.load into the modified algorithm.
    model.bc_weight=10.
    with pytest.raises(ValueError,match='RegularizedSAC.load'):validate_loaded_model(model,checkpoint,2)
