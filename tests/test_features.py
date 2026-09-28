"""Ensure replay goal relabeling and saved policies use the same feature math."""
import numpy as np
import torch
from stable_baselines3 import SAC
from pnr.features import RelativeGoalExtractor
from scripts.profile_learner import SyntheticProfileEnv


def test_relabelled_goal_changes_relative_features_and_survives_checkpoint(tmp_path):
    env=SyntheticProfileEnv()
    model=SAC('MultiInputPolicy',env,policy_kwargs={'net_arch':[16,16],
        'features_extractor_class':RelativeGoalExtractor,'features_extractor_kwargs':{'goal_scale':5.}},device='cpu')
    observation,_=env.reset()
    tensors={key:torch.tensor(value[None]) for key,value in observation.items()}
    extractor=model.actor.features_extractor
    original=extractor(tensors)
    tensors['desired_goal']+=.2
    relabeled=extractor(tensors)
    torch.testing.assert_close(original[:,-3:],torch.zeros((1,3)))
    torch.testing.assert_close(relabeled[:,-3:],torch.ones((1,3)))
    # Simulator feature entries cannot contain a stale relative-goal copy.
    torch.testing.assert_close(original[:,6:-3],relabeled[:,6:-3])
    expected=model.predict(observation,deterministic=True)[0]
    model.save(tmp_path/'policy')
    loaded=SAC.load(tmp_path/'policy.zip',device='cpu')
    np.testing.assert_array_equal(loaded.predict(observation,deterministic=True)[0],expected)


def test_bounded_goals_relabel_in_replay_without_singularities():
    from pnr.features import BoundedGoalExtractor
    env=SyntheticProfileEnv()
    extractor=BoundedGoalExtractor(env.observation_space,goal_scale=20.)
    observation={key:torch.zeros((3,*space.shape)) for key,space in env.observation_space.spaces.items()}
    goals=torch.tensor([[0.,0.,0.],[1.,.5,0.],[-1.,-.5,0.]],requires_grad=True)
    observation['desired_goal']=goals
    features=extractor(observation)
    relative=features[:,-3:]
    assert torch.isfinite(features).all()
    assert torch.all(torch.linalg.vector_norm(relative,dim=1)<1)
    torch.testing.assert_close(relative[0],torch.zeros(3))
    torch.testing.assert_close(relative[1],-relative[2])
    torch.testing.assert_close(relative[1,0],2*relative[1,1])
    relative.sum().backward()
    assert torch.isfinite(goals.grad).all()
