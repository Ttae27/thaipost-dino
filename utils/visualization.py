import numpy as np
import torch


def denormalize(tensor):
    """Reverses the ImageNet normalization so the tensor can be plotted as an image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)

    tensor = tensor * std + mean
    img_numpy = tensor.permute(1, 2, 0).cpu().numpy()

    return np.clip(img_numpy, 0, 1)
