"""Opt-in gated carry-map encoder inspired by GPPN; no geometric route oracle.

Reference: Lee et al., Gated Path Planning Networks (ICML2018), section3.
https://proceedings.mlr.press/v80/lee18c/lee18c.pdf

This is an adaptation, not a reproduction: five measured/goal planes and a
rectangular map feed SiLU-initialized spatial states; exactly ``iterations``
gated updates produce continuous-control features rather than discrete action
logits. Hidden/cell states are freshly constructed inside every forward call.
"""
from __future__ import annotations

import torch
from torch import nn

from .navigation_features import NavigationSpatialExtractor


class GatedMapCore(nn.Module):
    """Shared spatial messages and per-cell LSTM updates within one observation.

    Two3x3 initialization convolutions give a5-cell receptive field. Each gated
    update extends its theoretical side length by ``kernel_size - 1``. This is
    an occupancy dependency bound before pooling, not a guarantee that a route
    of that length is solved. No transition model or value function is supplied.
    """
    def __init__(self, input_channels=5, channels=32, iterations=20, kernel_size=5):
        super().__init__()
        for name,value in (('input_channels',input_channels),('channels',channels),
                           ('iterations',iterations),('kernel_size',kernel_size)):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if kernel_size%2!=1:
            raise ValueError('kernel_size must be odd to preserve spatial coordinates')
        self.channels=channels;self.iterations=iterations;self.kernel_size=kernel_size
        self.stem=nn.Sequential(nn.Conv2d(input_channels,channels,3,padding=1),nn.SiLU())
        self.initial_state=nn.Conv2d(channels,2*channels,3,padding=1)
        self.message=nn.Conv2d(channels,1,kernel_size,padding=kernel_size//2)
        self.cell=nn.LSTMCell(1,channels)

    @property
    def receptive_field_cells(self):
        return 5+self.iterations*(self.kernel_size-1)

    def forward(self, planes):
        batch,_,height,width=planes.shape
        initial=self.initial_state(self.stem(planes))
        hidden_map,cell_map=initial.chunk(2,dim=1)
        # Flatten only the batch/spatial axes; every cell has independent LSTM
        # state. The message convolution is the learned inter-cell interaction.
        hidden=hidden_map.permute(0,2,3,1).reshape(-1,self.channels)
        memory=cell_map.permute(0,2,3,1).reshape(-1,self.channels)
        for _ in range(self.iterations):
            hidden_map=hidden.reshape(batch,height,width,self.channels).permute(0,3,1,2)
            message=self.message(hidden_map).permute(0,2,3,1).reshape(-1,1)
            hidden,memory=self.cell(message,(hidden,memory))
        return hidden.reshape(batch,height,width,self.channels).permute(0,3,1,2)


class GatedCarrySpatialExtractor(NavigationSpatialExtractor):
    """Preserve the580-state carry interface and replace only its map core.

    Default physical/grip inputs match CarrySpatialExtractor. Its existing raw
    pose mask is retained; optional set_navigation_only() is also available but
    starts disabled. Actor/critic encoders remain separate under ordinary SB3.
    Inherited geometry rebuilding ensures HER uses each supplied desired_goal.
    Internal recurrence is stateless across calls, so no sequence replay is used.
    """
    def __init__(self,observation_space,features_dim=128,channels=32,iterations=20,
                 kernel_size=5,**kwargs):
        super().__init__(observation_space,features_dim=features_dim,channels=channels,**kwargs)
        self.spatial=GatedMapCore(5,channels,iterations,kernel_size)
