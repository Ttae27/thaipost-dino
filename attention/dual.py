import torch
import torch.nn as nn

from .position import PositionAttentionModule
from .channel import ChannelAttentionModule


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule()

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        out = torch.cat([cam_out, pam_out], dim=1)
        return out
