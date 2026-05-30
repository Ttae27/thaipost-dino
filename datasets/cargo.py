import os
import glob
import random

from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["0 _", "1-20 _", "21-40 _", "41-60 _", "61-80 _", "81-100 _"]
PLOT_CLASS_NAMES = ["0 (6)", "1-20 (5)", "21-40 (4)", "41-60 (3)", "61-80 (2)", "81-100 (1)"]


class CargoSpaceDataset(Dataset):
    def __init__(self, root_dir, dataset_type, mode='train', transform=None, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.class_names = CLASS_NAMES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        valid_exts = ('.jpg', '.jpeg', '.png')

        for cls_name in self.class_names:
            back_dir = os.path.join(root_dir, cls_name, "กล้องหน้า")
            front_dir = os.path.join(root_dir, cls_name, "กล้องหลัง")

            front_imgs = sorted([f for f in glob.glob(os.path.join(front_dir, "*.*")) if f.lower().endswith(valid_exts)])
            back_imgs = sorted([f for f in glob.glob(os.path.join(back_dir, "*.*")) if f.lower().endswith(valid_exts)])

            random.seed(seed)
            random.shuffle(front_imgs)
            random.shuffle(back_imgs)

            train_front, test_front = [], []
            train_back, test_back = [], []

            if dataset_type == 0:
                split_idx = int(0.8 * len(front_imgs))
                train_front = front_imgs[:split_idx]
                test_front = front_imgs[split_idx:]
            elif dataset_type == 1:
                f_split = int(0.8 * len(front_imgs))
                b_split = int(0.8 * len(back_imgs))
                train_front = front_imgs[:f_split]
                test_front = front_imgs[f_split:]
                train_back = back_imgs[:b_split]
                test_back = back_imgs[b_split:]

            if mode == 'train':
                selected_imgs = train_front + train_back
            elif mode == 'test':
                selected_imgs = test_front + test_back
            elif mode == 'test aug':
                if dataset_type == 0:
                    selected_imgs = front_imgs
                elif dataset_type == 1:
                    selected_imgs = front_imgs + back_imgs
                else:
                    selected_imgs = []
            else:
                selected_imgs = []

            label = self.class_to_idx[cls_name]
            for img_path in selected_imgs:
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
