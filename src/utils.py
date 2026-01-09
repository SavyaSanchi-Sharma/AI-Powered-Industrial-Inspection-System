# src/utils.py
import os
import cv2
import torch
import numpy as np
from preprocess import preprocess_image

def load_images_from_folder(folder):
    images = []

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            img = preprocess_image(img_path)
            if img is not None:
                images.append(img)

    return np.array(images)
