import torch.nn as nn
from transformers import AutoModel

from attention import DualAttentionModule


DEFAULT_BACKBONE = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


class Network(nn.Module):
    def __init__(self, backbone, attention, class_dim=6, device_map="cuda", model_name=DEFAULT_BACKBONE):
        super().__init__()
        self.attention = attention
        self.bb = backbone

        model = AutoModel.from_pretrained(model_name, device_map=device_map)
        self.backbone = model
        num_features = model.config.hidden_size

        self.dam = DualAttentionModule(in_channels=num_features)

        if attention == 'dam':
            in_chan = num_features * 2
        else:
            in_chan = num_features

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_chan, class_dim, bias=False)
        self.ln = nn.LayerNorm(class_dim)

    def extract_backbone(self, x):
        out = self.backbone(x)
        feat = out.last_hidden_state

        total_non_cls_tokens = feat.shape[1] - 1
        grid_size = int(total_non_cls_tokens ** 0.5)
        num_spatial_tokens = grid_size * grid_size

        feat = feat[:, -num_spatial_tokens:, :]

        B, N, C = feat.shape
        feat = feat.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        return feat

    def embed(self, x):
        feat = self.extract_backbone(x)

        if self.attention == 'dam':
            fused = self.dam(feat)
        else:
            fused = feat

        fused = self.gap(fused)
        fused = fused.view(fused.size(0), -1)

        x = self.fc(fused)
        x = self.ln(x)
        return x

    def forward(self, img):
        return self.embed(img)
