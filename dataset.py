import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np

class HelmetDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416, num_classes=2, transform=None):
        self.img_files = sorted(glob.glob(f"{img_dir}/*.jpg"))
        self.label_files = sorted(glob.glob(f"{label_dir}/*.txt"))
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load and preprocess image
        img = Image.open(self.img_files[index]).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Initialize target tensor (13x13 grid, one box per cell, 5+num_classes values)
        targets = torch.zeros(13*13, 5 + self.num_classes)

        # Load labels (YOLO format: class x y w h)
        with open(self.label_files[index], "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                cls = int(cls)

                # Scale center coords to 13x13 grid
                gi, gj = int(x * 13), int(y * 13)
                idx = gj * 13 + gi

                # Store box + objectness + class one-hot
                targets[idx, 0:4] = torch.tensor([x, y, w, h])  # normalized box
                targets[idx, 4] = 1.0                           # objectness
                targets[idx, 5 + cls] = 1.0                     # class one-hot

        return img, targets
