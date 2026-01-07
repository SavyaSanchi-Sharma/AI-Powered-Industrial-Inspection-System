# src/augment.py
import cv2
import numpy as np
import os
import random
from config import TRAIN_GOOD_DIR

def augment(img):
    alpha = random.uniform(0.85, 1.15)
    img = img * alpha

    noise = np.random.normal(0, 0.02, img.shape)
    img = img + noise

    return np.clip(img, 0, 1)

files = os.listdir(TRAIN_GOOD_DIR)
num_aug = int(len(files) * 0.4)

for f in files[:num_aug]:
    img_path = os.path.join(TRAIN_GOOD_DIR, f)
    img = cv2.imread(img_path).astype(np.float32) / 255.0

    aug_img = augment(img)
    save_path = os.path.join(TRAIN_GOOD_DIR, f"aug_{f}")

    cv2.imwrite(save_path, (aug_img * 255).astype("uint8"))

print("✅ Augmentation completed")
