from types import SimpleNamespace
import json
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from pnr.general_manipulation import GraspSchema,GeneralGraspFeatures
from pnr.grasp_actor_ablation import CompactGraspFeatures,CompactGraspActorPolicy
from pnr.telemetry import RunLogger
from scripts.phase2_train_grasp import Progress


def observations():
    raw=torch.zeros((2,54),dtype=torch.float32)
    raw[:,34]=1.;raw[:,41]=1.
    raw[:,28:31]=torch.tensor([.7,.1,.8]);raw[:,35:38]=torch.tensor([.75,.12,.67])
    raw[:,50:53]=torch.tensor([.04,.01,-.02]);raw[:,53]=.07
    return {'observation':raw,'achieved_goal':raw[:,35:38].clone(),
            'desired_goal':torch.tensor([[.75,.12,.91],[.75,.12,.91]])}


def test_compact_geometry_uses_fresh_goals_and_excludes_absolute_pose():
    extractor=CompactGraspFeatures(GraspSchema().observation_space);obs=observations();features=extractor(obs)
    assert features.shape==(2,28)
    shifted={k:v.clone() for k,v in obs.items()};shift=torch.tensor([.3,-.2,.1])
    shifted['observation'][:,28:31]+=shift;shifted['observation'][:,35:38]+=shift
    shifted['achieved_goal']+=shift;shifted['desired_goal']+=shift
    torch.testing.assert_close(extractor(shifted),features)
    shifted={k:v.clone() for k,v in obs.items()};shifted['desired_goal'][:,0]+=.1
    changed=extractor(shifted)
    torch.testing.assert_close(changed[:,6],features[:,6]+.5)
    assert not torch.equal(changed[:,-1],features[:,-1])
    flipped={k:v.clone() for k,v in obs.items()};flipped['observation'][:,38:42]*=-1
    torch.testing.assert_close(extractor(flipped),features)


def test_compact_actor_retains_full_critic_and_roundtrips(tmp_path):
    torch.set_num_threads(1);env=DummyVecEnv([GraspSchema])
    model=SAC(CompactGraspActorPolicy,env,buffer_size=1,
              policy_kwargs={'net_arch':[128,128],'features_extractor_class':GeneralGraspFeatures},device='cpu',seed=8101)
    assert model.actor.features_extractor.features_dim==28
    assert model.critic.features_extractor.features_dim==70
    obs=observations();expected=model.actor(obs,deterministic=True)
    model.save(tmp_path/'compact');loaded=SAC.load(tmp_path/'compact',env=env,device='cpu')
    torch.testing.assert_close(loaded.actor(obs,deterministic=True),expected)
    assert loaded.critic.features_extractor.features_dim==70
    assert loaded.action_space.shape==(4,)
    env.close()


def test_interrupted_empty_evaluation_never_reports_zero_success(tmp_path):
    logger=RunLogger('interrupted',root=tmp_path)
    progress=Progress(logger,[{'object_idx':2,'scene_seed':43001,'partition':'validation'}],
                      SimpleNamespace(),SimpleNamespace(requested=True))
    progress.model=SimpleNamespace(num_timesteps=0,_n_updates=0,env=SimpleNamespace())
    score,distance=progress.evaluate('stopped')
    assert np.isneginf(score) and np.isposinf(distance)
    report=json.loads((logger.directory/'artifacts/evaluation_stopped_interrupted.json').read_text())
    assert report['status']=='incomplete' and report['validation_success_rate'] is None
    assert report['completed_cases']==0
    assert 'success_rate' not in logger.state['metrics']
    assert not (logger.directory/'artifacts/evaluation_stopped.json').exists()
