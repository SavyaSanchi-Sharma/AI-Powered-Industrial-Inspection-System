# src/aircraft/aircraft_dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from ..preprocess import preprocess_image

class AircraftDamageDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = preprocess_image(img_path)

        # All images are damaged → label = 1
        label = 1

        img = torch.tensor(img).permute(2, 0, 1).float()
        return img, label
