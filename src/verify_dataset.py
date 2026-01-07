# src/verify_dataset.py
import cv2
import os
import numpy as np
from config import TRAIN_GOOD_DIR

files = os.listdir(TRAIN_GOOD_DIR)
print("Total images:", len(files))

sample = cv2.imread(os.path.join(TRAIN_GOOD_DIR, files[0]))

print("Shape:", sample.shape)
print("Min pixel:", np.min(sample))
print("Max pixel:", np.max(sample))
