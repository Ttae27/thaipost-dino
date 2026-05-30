import random

import torch
import torchvision.transforms.functional as F_vision


class SimulateNightVision:
    """
    Transforms a standard daytime RGB image into a gritty,
    high-contrast infrared night vision image.
    """

    def __init__(self, p=1.0, noise_level=0.08):
        self.p = p
        self.noise_level = noise_level

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img = F_vision.to_grayscale(img, num_output_channels=3)

        img = F_vision.adjust_contrast(img, contrast_factor=random.uniform(1.5, 2.5))
        img = F_vision.adjust_brightness(img, brightness_factor=random.uniform(0.7, 1.2))

        tensor_img = F_vision.to_tensor(img)
        noise = torch.randn_like(tensor_img) * self.noise_level
        noisy_tensor = tensor_img + noise

        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)

        return F_vision.to_pil_image(noisy_tensor)
