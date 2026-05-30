import os

from PIL import Image
from torch.utils.data import Dataset


class UnlabeledCargoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        valid_exts = ('.jpg', '.jpeg', '.png')

        print(f"Scanning directory: {root_dir}")
        for root, dirs, files in os.walk(root_dir):
            path_parts = root.replace('\\', '/').split('/')

            if any(part.endswith('.3') for part in path_parts):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        full_path = os.path.join(root, file)
                        self.samples.append(full_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path
