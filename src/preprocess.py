# src/preprocess.py
import cv2
import numpy as np
from config import IMG_SIZE, COLOR_MODE

# ------------------------------
# Used for dataset ingestion (Phase 1–3)
# ------------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    if COLOR_MODE == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.astype(np.float32) / 255.0

    return img


# ------------------------------
# Used for LIVE CAMERA (Phase 4–5)
# ------------------------------
def preprocess_frame(frame):
    """
    frame: BGR image from OpenCV camera
    returns: normalized RGB image (224x224)
    """
    img = frame.copy()

    if COLOR_MODE == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.astype(np.float32) / 255.0

    return img
