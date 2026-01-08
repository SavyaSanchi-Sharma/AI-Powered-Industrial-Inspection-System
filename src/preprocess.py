# src/preprocess.py
import cv2
import numpy as np
from config import IMG_SIZE, COLOR_MODE


def preprocess_frame(frame):
    """Preprocess an in-memory image frame (H, W, C) and return an
    HxWx3 float32 image normalized to [0, 1]. This keeps a 3-channel
    output even if COLOR_MODE is set to grayscale so downstream code
    can safely permute channels.
    """
    if frame is None:
        return None

    img = frame.copy()

    # Ensure color ordering and 3 channels
    if COLOR_MODE == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.astype(np.float32) / 255.0

    return img


def preprocess_image(img_path):
    """Preprocess an image from disk. Delegates to preprocess_frame."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    return preprocess_frame(img)
