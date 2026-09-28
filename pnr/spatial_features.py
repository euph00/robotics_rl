"""Learned spatial carry encoder for the original measured 580-state interface.

A CNN receives only measured occupancy and goal-relative geometry. No route,
shortest-path distance, teacher action, future state, or classical planner enters
inference. Goal planes and samples are rebuilt on every forward pass, including
HER relabeled replay batches. Coordinates retain the original world/grid frame.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, values):
        return F.silu(values + self.layers(values))


class CarrySpatialExtractor(BaseFeaturesExtractor):
    """Small global-receptive-field CNN plus local measured grip/control state.

    SB3 normally constructs independent actor and critic extractors. Keep the
    default full physical state for critics, then call the actor extractor's
    set_carry_pose_mask(True). The actor still receives measured finger widths,
    hand-to-payload displacement, planar velocity, and finger contacts: it can
    adapt control to a weak/shifted grip without depending on arbitrary arm pose.
    All mask values are registered buffers and survive ordinary SAC save/load.

    Seven 3x3 convolution layers with dilations 1,(1,1),(2,2),(4,4) have a 31-cell
    receptive field, covering the complete original 12x22 maze. This is learned
    image processing, not a hard-coded graph search or route encoder.
    """
    def __init__(self, observation_space, features_dim=128, channels=32, goal_scale=20.,
                 retain_physical_state=True, goal_sigma=.65, local_spacing=.5):
        if (observation_space['observation'].shape != (580,)
                or observation_space['achieved_goal'].shape != (2,)
                or observation_space['desired_goal'].shape != (2,)):
            raise ValueError('CarrySpatialExtractor requires the580-state two-goal carry adapter')
        if features_dim < 1 or channels < 1 or min(goal_scale, goal_sigma, local_spacing) <= 0:
            raise ValueError('Positive feature/channel counts and geometry scales required')
        super().__init__(observation_space, int(features_dim))
        self.goal_scale = float(goal_scale)
        self.goal_sigma = float(goal_sigma)
        self.local_spacing = float(local_spacing)
        self.channels = int(channels)
        # Raw carry layout: maps[0:528], base[528:530], physical[530:578], contacts[578:580].
        self.register_buffer('state_mask', torch.ones(48) if retain_physical_state else torch.zeros(48))
        rows, cols = torch.meshgrid(torch.arange(12.), torch.arange(22.), indexing='ij')
        # Wall centers are x=column-3.5, y=5.5-row; +world y points UP the tensor.
        self.register_buffer('cell_world', torch.stack((cols - 3.5, 5.5 - rows), dim=0).unsqueeze(0))
        offsets = torch.arange(-2., 3.) * self.local_spacing
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        self.register_buffer('local_offsets', torch.stack((dx, dy), dim=-1))
        self.spatial = nn.Sequential(
            nn.Conv2d(5, self.channels, 3, padding=1), nn.SiLU(),
            *[_DilatedResidualBlock(self.channels, dilation) for dilation in (1, 2, 4)],
        )
        # Base sample, goal sample, global mean/max; 2x5x5 occupancy patch;
        # bounded goal direction2 + normalized length1 + fractional cell2;
        # compact physical summary9 + optional raw48-state.
        fusion_dim = 4 * self.channels + 50 + 5 + 9 + 48
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, self.features_dim),
                                    nn.LayerNorm(self.features_dim), nn.SiLU())

    def set_carry_pose_mask(self, enabled=True):
        """Mask extra raw physical state, preserving the compact grip summary."""
        self.state_mask.fill_(0. if enabled else 1.)

    @staticmethod
    def world_to_grid(world_xy):
        """World XY to grid_sample XY with align_corners=True for a 12x22 map."""
        x = 2. * (world_xy[..., 0] + 3.5) / 21. - 1.
        y = 2. * (5.5 - world_xy[..., 1]) / 11. - 1.
        return torch.stack((x, y), dim=-1)

    @classmethod
    def sample_world(cls, spatial, world_xy):
        """Bilinear sampling at Bx...x2 world points, with border-value extension."""
        if world_xy.ndim == 2:
            grid = cls.world_to_grid(world_xy).unsqueeze(1).unsqueeze(1)
            return F.grid_sample(spatial, grid, mode='bilinear', padding_mode='border',
                                 align_corners=True).flatten(1)
        return F.grid_sample(spatial, cls.world_to_grid(world_xy), mode='bilinear',
                             padding_mode='border', align_corners=True)

    def spatial_inputs(self, observations):
        """Expose measured/goal planes for diagnostics; recompute after relabeling."""
        raw = observations['observation']
        maps = raw[:, :528].reshape(-1, 2, 12, 22)
        goal = observations['desired_goal'] * self.goal_scale
        displacement = goal[:, :, None, None] - self.cell_world
        heatmap = torch.exp(-displacement.square().sum(dim=1, keepdim=True)
                            / (2. * self.goal_sigma**2))
        return torch.cat((maps, heatmap, displacement / self.goal_scale), dim=1)

    @staticmethod
    def grip_summary(raw):
        """Measured fingers, payload relative to hand, planar velocity, contacts."""
        fingers = raw[:, 541:543] / .05
        relative_payload = torch.tanh((raw[:, 565:568] - raw[:, 558:561]) / .1)
        planar_velocity = torch.tanh(raw[:, 543:545] / 2.)
        return torch.cat((fingers, relative_payload, planar_velocity, raw[:, 578:580]), dim=1)

    def forward(self, observations):
        raw = observations['observation']
        base = observations['achieved_goal'] * self.goal_scale
        goal = observations['desired_goal'] * self.goal_scale
        spatial_input = self.spatial_inputs(observations)
        learned = self.spatial(spatial_input)
        sampled = self.sample_world(learned, base)
        goal_sample = self.sample_world(learned, goal)
        pooled_mean, pooled_max = learned.mean(dim=(2, 3)), learned.amax(dim=(2, 3))
        patch_points = base[:, None, None, :] + self.local_offsets
        local_patch = self.sample_world(spatial_input[:, :2], patch_points).flatten(1)
        delta = goal - base
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        goal_vector = torch.cat((delta / (.3 + distance), distance / self.goal_scale), dim=1)
        cell_coordinates = torch.stack((base[:, 0] + 3.5, 5.5 - base[:, 1]), dim=1)
        fractional_cell = cell_coordinates - cell_coordinates.round()
        physical = torch.tanh(raw[:, 530:578]) * self.state_mask
        features = torch.cat((sampled, goal_sample, pooled_mean, pooled_max, local_patch,
                              goal_vector, fractional_cell, self.grip_summary(raw), physical), dim=1)
        return self.fusion(features)
