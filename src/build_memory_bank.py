# src/build_memory_bank.py
import torch
import numpy as np
from backbone import ResNetBackbone
from feature_extractor import extract_patch_features
from utils import load_images_from_folder
from config import TRAIN_GOOD_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using device: {DEVICE}")

# Load images
images = load_images_from_folder(TRAIN_GOOD_DIR)
print(f"Loaded {len(images)} images")

# Convert to tensor
images = torch.tensor(images).permute(0, 3, 1, 2).float().to(DEVICE)

# Load backbone
model = ResNetBackbone().to(DEVICE)
model.eval()

memory_bank = []

with torch.no_grad():
    for i in range(images.shape[0]):
        img = images[i].unsqueeze(0)
        features = model(img)
        patches = extract_patch_features(features)
        memory_bank.append(patches)

memory_bank = np.concatenate(memory_bank, axis=0)

# Save memory bank
np.save("memory_bank.npy", memory_bank)

print("✅ MEMORY BANK BUILT")
print("Memory shape:", memory_bank.shape)
