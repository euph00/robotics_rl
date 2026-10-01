"""Compute-only checkpoint/HER tests; these are not robotics task evidence."""
from pathlib import Path
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from synthetic_env import SyntheticProfileEnv
from pnr.telemetry import RunLogger, atomic_json
from pnr.train import ProgressCallback, make_resume_signature, resume_preflight, validate_replay_layout


class SyntheticTrainingEnv(SyntheticProfileEnv):
    """Deterministic finite episodes exercise replay only, without a simulator."""
    horizon=6
    setup_action_calls=0
    curriculum_level=0

    def __init__(self, first_goal=False):
        super().__init__()
        self.first_goal=first_goal
        self.total_steps=0
        self.steps=0
        self.reset_count=0

    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        self.steps=0;self.reset_count+=1
        return observation,info

    def step(self,action):
        self.steps+=1;self.total_steps+=1
        observation={key:np.zeros(space.shape,dtype=np.float32) for key,space in self.observation_space.spaces.items()}
        observation['observation'][0]=self.steps
        observation['achieved_goal'][0]=self.steps*.01
        retained=self.steps%2==0
        success=self.first_goal and self.steps==2
        info={'bilateral_contact':retained,'is_success':success,'episode_success':success,
              'goal_distance':0. if success else 1.,'action_calls_delta':3,'physics_ticks_delta':15,
              'drop_events':int(self.first_goal and self.steps>2),'physical_failure':False}
        reward=float(self.compute_reward(observation['achieved_goal'],observation['desired_goal'],info))
        return observation,reward,False,self.steps>=self.horizon,info

    def get_training_state(self):
        return {'curriculum_level':0,'reset_count':self.reset_count,'synthetic_test_only':True}


def tiny_model(vector):
    torch.set_num_threads(1)
    return SAC('MultiInputPolicy',vector,buffer_size=128,learning_starts=12,batch_size=8,
        train_freq=1,gradient_steps=1,policy_kwargs={'net_arch':[16,16]},seed=13,device='cpu',
        replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs={'copy_info_dict':True,'n_sampled_goal':4})


def configuration(num_envs=2):
    return dict(skill='grasp',scene_seed=42,object_idx=14,seed=13,num_envs=num_envs,horizon=6,goal_features=False)


def source_tree(tmp_path):
    root=tmp_path/'source'
    (root/'pnr').mkdir(parents=True)
    for name in ('env','planning','reference','skills','local_control','features','carry','train','telemetry'):
        (root/'pnr'/f'{name}.py').write_text(f'# synthetic {name} semantics version 1\n')
    return root


@pytest.fixture
def trained_checkpoint(tmp_path):
    vector=DummyVecEnv([SyntheticTrainingEnv,SyntheticTrainingEnv])
    evaluation=SyntheticTrainingEnv(first_goal=True)
    model=tiny_model(vector)
    logger=RunLogger('synthetic_resume',root=tmp_path,kind='test',summary='Synthetic compute test only')
    callback=ProgressCallback(logger,evaluation,eval_every=100000,checkpoint_every=8)
    source=source_tree(tmp_path)
    callback.resume_signature=make_resume_signature(configuration(),source)
    callback.training_setup_actions=11;callback.training_setup_ticks=55
    callback.evaluation_setup_actions=7;callback.evaluation_setup_ticks=35
    callback.invocation_name='synthetic_test'
    model.learn(40,callback=callback)
    callback.evaluate()
    callback.checkpoint()
    preflight=resume_preflight(logger.directory,configuration(),source)
    yield model,callback,logger,source,preflight
    vector.close();evaluation.close()


def test_real_model_and_her_replay_resume_preserve_obligations_and_counts(trained_checkpoint):
    model,callback,logger,source,preflight=trained_checkpoint
    checkpoint=preflight['checkpoint'];counts=preflight['counts']
    assert counts['transitions']==model.num_timesteps==40
    assert counts['source_action_calls']==120
    assert counts['physics_ticks']==600
    assert counts['replay_position']==20
    assert counts['accounting']['evaluation_source_action_calls']==30  # 5 episodes × 2 decisions × 3 calls
    assert counts['accounting']['total_source_action_calls_known']==168
    replacement=DummyVecEnv([SyntheticTrainingEnv,SyntheticTrainingEnv])
    try:
        loaded=SAC.load(checkpoint/'model.zip',env=replacement,device='cpu')
        with pytest.warns(UserWarning,match='last trajectory'):
            loaded.load_replay_buffer(checkpoint/'replay.pkl',truncate_last_traj=True)
        validate_replay_layout(loaded.replay_buffer,2,'grasp')
        obs=replacement.reset()
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
        assert loaded.num_timesteps==40
        assert loaded._n_updates==model._n_updates
        replay=loaded.replay_buffer
        np.testing.assert_array_equal(replay.ep_length[18:20],2)
        np.testing.assert_array_equal(replay.timeouts[19],1)
        indices=np.arange(20);workers=np.zeros(20,dtype=int)
        samples=replay._get_virtual_samples(indices,workers)
        expected=replacement.env_method('compute_reward',samples.next_observations['achieved_goal'].numpy(),
            samples.observations['desired_goal'].numpy(),replay.infos[indices,workers],indices=[0])[0]
        np.testing.assert_array_equal(samples.rewards.numpy().ravel(),expected)
        assert not samples.dones.numpy().any()
        restored=ProgressCallback(logger,SyntheticTrainingEnv())
        restored.restore_progress(counts)
        assert restored.accounting()==callback.accounting()
        restored.training_setup_actions+=5;restored.training_setup_ticks+=25
        assert restored.accounting()['training_setup_source_action_calls']==16
        loaded.gradient_steps=2
        loaded.learn(8,reset_num_timesteps=False)
        assert loaded.num_timesteps==48
        assert loaded._n_updates==model._n_updates+8
    finally:
        replacement.close()


def test_resume_rejects_changed_workers_reward_source_and_feature_config_before_writes(trained_checkpoint):
    _,_,logger,source,preflight=trained_checkpoint
    original_status=(logger.directory/'status.json').read_bytes()
    changed=configuration();changed['num_envs']=3
    with pytest.raises(ValueError,match='num_envs'):
        resume_preflight(logger.directory,changed,source)
    changed=configuration();changed['goal_features']=True
    with pytest.raises(ValueError,match='goal_features'):
        resume_preflight(logger.directory,changed,source)
    (source/'pnr/skills.py').write_text('# reward semantics changed to terminal failure penalty -50\n')
    with pytest.raises(ValueError,match='source pnr/skills.py'):
        resume_preflight(logger.directory,configuration(),source)
    assert (logger.directory/'status.json').read_bytes()==original_status


def test_resume_allows_logging_only_edits_and_rejects_replay_worker_broadcast(trained_checkpoint):
    model,_,logger,source,_=trained_checkpoint
    (source/'pnr/train.py').write_text('# changed logging/checkpoint implementation only\n')
    (source/'pnr/telemetry.py').write_text('# new dashboard chart\n')
    assert resume_preflight(logger.directory,configuration(),source)['counts']['transitions']==40
    with pytest.raises(ValueError,match='worker count'):
        validate_replay_layout(model.replay_buffer,3,'grasp')


def test_periodic_checkpoint_is_after_replay_storage(trained_checkpoint):
    _,_,logger,_,_=trained_checkpoint
    for path in (logger.directory/'checkpoints').glob('step_*/progress.json'):
        counts=json.loads(path.read_text())
        assert counts['replay_position']*counts['replay_n_envs']==counts['transitions']


def test_evaluation_stops_at_first_current_retained_goal(trained_checkpoint):
    _,callback,_,_,preflight=trained_checkpoint
    assert callback.eval_env.total_steps==10  # It never executes the later synthetic drop.
    metadata=json.loads((preflight['checkpoint'].parent/'best/evaluation.json').read_text())
    assert metadata['success_rate']==1.
    assert metadata['evaluation_protocol']=='stop_on_first_retained_goal'


def test_interrupt_inside_callback_flushes_pending_transition_once(tmp_path):
    vector=DummyVecEnv([SyntheticTrainingEnv,SyntheticTrainingEnv])
    model=tiny_model(vector)
    logger=RunLogger('synthetic_interrupt',root=tmp_path,kind='test')
    class Interrupt(ProgressCallback):
        def _on_step(self):
            value=super()._on_step()
            if self.num_timesteps==8:
                raise KeyboardInterrupt()
            return value
    callback=Interrupt(logger,SyntheticTrainingEnv(),eval_every=100000)
    try:
        with pytest.raises(KeyboardInterrupt):
            model.learn(12,callback=callback)
        assert model.num_timesteps==8
        assert model.replay_buffer.pos==3  # The fourth vector step has happened but is not stored yet.
        callback.checkpoint()
        assert model.replay_buffer.pos==4
        callback.checkpoint()
        assert model.replay_buffer.pos==4
        assert callback.native_actions==24
    finally:
        vector.close()


def test_legacy_unknown_accounting_is_explicit(tmp_path):
    logger=RunLogger('synthetic_legacy',root=tmp_path,kind='test')
    callback=ProgressCallback(logger,SyntheticTrainingEnv())
    callback.restore_progress({'source_action_calls':90,'physics_ticks':450},
        {'setup_action_calls':12,'evaluation_setup_action_calls':8})
    account=callback.accounting()
    assert not account['complete']
    assert 'historical_evaluation_interactions' in account['legacy_missing']
    assert account['total_source_action_calls_known']==110


def test_inference_loader_preserves_actual_her_actor_without_simulator(trained_checkpoint):
    model,_,_,_,preflight=trained_checkpoint
    from pnr.policies import load_policy_checkpoint
    from stable_baselines3.common.buffers import DictReplayBuffer
    loaded=load_policy_checkpoint(preflight['checkpoint']/'model.zip')
    assert loaded.env is None
    assert type(loaded.replay_buffer) is DictReplayBuffer
    assert loaded.replay_buffer.buffer_size==1
    observation,_=SyntheticTrainingEnv().reset()
    np.testing.assert_array_equal(model.predict(observation,deterministic=True)[0],
                                  loaded.predict(observation,deterministic=True)[0])


def test_demonstrations_account_all_collection_cost_once_not_just_imported_subset(tmp_path):
    logger=RunLogger('synthetic_demos',root=tmp_path,kind='test')
    callback=ProgressCallback(logger,SyntheticTrainingEnv())
    report={'transitions_imported':20,'imported_source_action_calls':60,'imported_physics_ticks':300,
            'dataset_metadata':{'demonstration_source_action_calls':90,'demonstration_physics_ticks':450,
                                'setup_source_action_calls':10,'setup_physics_ticks':50,'initialization_physics_ticks':7}}
    callback.add_demonstrations(report)
    account=callback.accounting()
    assert account['total_source_action_calls_known']==100
    assert account['total_physics_ticks_known']==507
    assert account['demonstration_transitions_imported']==20
    assert account['complete']
    with pytest.raises(ValueError,match='twice'):
        callback.add_demonstrations(report)
    restored=ProgressCallback(logger,SyntheticTrainingEnv())
    restored.restore_progress({'source_action_calls':0,'physics_ticks':0,'accounting':account,'demonstrations':report})
    assert restored.accounting()==account


def test_seeded_dataset_path_and_hash_are_immutable_resume_configuration(trained_checkpoint):
    _,_,logger,source,_=trained_checkpoint
    (source/'pnr/demonstrations.py').write_text('# synthetic importer contract\n')
    changed=configuration();changed.update(demonstrations='/tmp/new-demonstrations.pkl',demonstrations_sha256='f'*64)
    with pytest.raises(ValueError,match='demonstrations'):
        resume_preflight(logger.directory,changed,source)


class SignalDuringWorkerStep(SyntheticTrainingEnv):
    """Send a real signal while the parent waits for subprocess step replies."""
    def __init__(self, parent_pid, number, send_signal):
        super().__init__()
        from pnr.train import ignore_worker_stop_signals
        ignore_worker_stop_signals(parent_pid)
        self.parent_pid, self.number, self.send_signal=parent_pid,number,send_signal

    def step(self, action):
        import os
        import time
        time.sleep(.08)
        if self.send_signal and self.total_steps==0:
            os.kill(self.parent_pid,self.number)
        time.sleep(.08)
        return super().step(action)


@pytest.mark.parametrize('signal_name',['SIGINT','SIGTERM'])
def test_signal_during_subproc_step_drains_replies_and_checkpoints_pending_transition(tmp_path,signal_name):
    import os
    import signal
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from pnr.train import GracefulStop
    number=getattr(signal,signal_name)
    previous=signal.getsignal(number)
    parent_pid=os.getpid()
    with GracefulStop() as stop_request:
        vector=SubprocVecEnv([
            lambda:SignalDuringWorkerStep(parent_pid,number,True),
            lambda:SignalDuringWorkerStep(parent_pid,number,False)],start_method='spawn')
        try:
            model=tiny_model(vector)
            logger=RunLogger('synthetic_'+signal_name,root=tmp_path,kind='test')
            callback=ProgressCallback(logger,SyntheticTrainingEnv(),eval_every=100000,stop_request=stop_request)
            model.learn(12,callback=callback)
            assert stop_request.requested and stop_request.signal_name==signal_name
            assert not vector.waiting
            assert model.num_timesteps==2
            assert callback.native_actions==6 and callback.physics_ticks==30
            assert model.replay_buffer.pos==0  # SB3 returns before storing when callback returns False.
            callback.checkpoint()
            assert model.replay_buffer.pos==1
            callback.checkpoint()
            assert model.replay_buffer.pos==1
            pointer=json.loads((logger.directory/'checkpoints/latest.json').read_text())
            saved=json.loads((logger.directory/'checkpoints'/pointer['generation']/'progress.json').read_text())
            assert saved['transitions']==2 and saved['replay_position']==1
            assert saved['stop_requested'] and saved['stop_signal']==signal_name
            assert len(saved['training_states'])==2
            assert all(state['synthetic_test_only'] for state in saved['training_states'])
        finally:
            vector.close()
    assert signal.getsignal(number)==previous


def test_signal_during_evaluation_counts_cost_and_does_not_publish_partial_score(tmp_path):
    import os
    import signal
    from pnr.train import GracefulStop
    class InterruptedEvaluation(SyntheticTrainingEnv):
        def step(self,action):
            result=super().step(action)
            os.kill(os.getpid(),signal.SIGTERM)
            return result
    vector=DummyVecEnv([SyntheticTrainingEnv])
    try:
        model=tiny_model(vector)
        logger=RunLogger('synthetic_partial_eval',root=tmp_path,kind='test')
        with GracefulStop() as stop_request:
            callback=ProgressCallback(logger,InterruptedEvaluation(),stop_request=stop_request)
            callback.model=model
            assert callback.evaluate() is False
            assert stop_request.signal_name=='SIGTERM'
            assert callback.evaluation_actions==3 and callback.evaluation_ticks==15
            assert not (logger.directory/'checkpoints/best').exists()
            assert not (logger.directory/'metrics.jsonl').exists()
            assert callback.evaluate() is False  # A request during setup skips evaluation entirely.
            assert callback.evaluation_actions==3
            os.kill(os.getpid(),signal.SIGINT)
            assert stop_request.signal_name=='SIGTERM'  # Retain the original reason on repeated signals.
    finally:
        vector.close()


def test_checkpoint_refuses_pending_vector_reply_without_worker_rpc(tmp_path):
    from types import SimpleNamespace
    logger=RunLogger('synthetic_unsafe_boundary',root=tmp_path,kind='test')
    callback=ProgressCallback(logger,SyntheticTrainingEnv())
    class PendingVector:
        waiting=True
        def env_method(self,*_):
            pytest.fail('A checkpoint must not query a worker before draining its step reply')
    callback.model=SimpleNamespace(env=PendingVector())
    with pytest.raises(RuntimeError,match='waiting for worker replies'):
        callback.checkpoint()
    assert not (logger.directory/'checkpoints').exists()
