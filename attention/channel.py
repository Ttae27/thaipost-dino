import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """
        N, C, H, W = x.shape
        query = x.view(N, C, -1)
        key = x.view(N, C, -1).permute(0, 2, 1)

        energy = torch.bmm(query, key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma * out + x
        return out
