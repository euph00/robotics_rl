"""Optional object-invariant route encoding; critics keep full measured state."""
import torch
from .spatial_features import CarrySpatialExtractor


class NavigationSpatialExtractor(CarrySpatialExtractor):
    """Keep measured map, goal, base position and velocity for route decisions.

    Grip dimensions are masked only in the actor when explicitly requested.
    This ablates a potential shortcut from foam-only route demonstrations to
    object-dependent turn choices. It adds no route planner or teacher input.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('navigation_grip_mask', torch.ones(9))

    def set_navigation_only(self, enabled=True):
        self.set_carry_pose_mask(enabled)
        values=[0.,0.,0.,0.,0.,1.,1.,0.,0.] if enabled else [1.]*9
        self.navigation_grip_mask.copy_(torch.tensor(values,device=self.navigation_grip_mask.device))

    def grip_summary(self, raw):
        return super().grip_summary(raw)*self.navigation_grip_mask
