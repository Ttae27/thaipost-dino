import torch
import torch.nn as nn


class PositionAttentionModule(nn.Module):
    """ self-attention """

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """
        N, C, H, W = x.shape
        query = self.query_conv(x).view(N, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(N, -1, H * W)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H * W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma * out + x
        return out
