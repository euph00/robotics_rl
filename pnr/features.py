"""Goal arithmetic computed after HER relabeling, never cached in raw state."""
from __future__ import annotations
import torch
from stable_baselines3.common.torch_layers import CombinedExtractor


class RelativeGoalExtractor(CombinedExtractor):
    """Keep all measured observations and expose desired-minus-achieved explicitly.

    Scale converts the stored goals into useful physical units. Because this
    operation runs on each replay batch, relabeled goals stay fully consistent.
    """
    def __init__(self, observation_space, goal_scale=1.):
        super().__init__(observation_space)
        self.goal_scale=float(goal_scale)
        self._features_dim += int(observation_space['desired_goal'].shape[0])

    def forward(self, observations):
        return torch.cat([super().forward(observations),
            (observations['desired_goal']-observations['achieved_goal'])*self.goal_scale], dim=1)


class BoundedGoalExtractor(RelativeGoalExtractor):
    """Preserve goal direction while avoiding unbounded long-route feature values.

    All absolute goals/state remain present. The additional displacement is
    delta/(0.3m + norm(delta)), so near and distant goals use a common range.
    Older checkpoints still use the unchanged RelativeGoalExtractor.
    """
    def forward(self, observations):
        features=super().forward(observations)
        dimensions=observations['desired_goal'].shape[-1]
        delta=features[:,-dimensions:]
        bounded=delta/(.3+torch.linalg.vector_norm(delta,dim=1,keepdim=True))
        return torch.cat([features[:,:-dimensions],bounded],dim=1)
