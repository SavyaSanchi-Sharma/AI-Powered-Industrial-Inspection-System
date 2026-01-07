# src/config.py
import os

IMG_SIZE = 224
COLOR_MODE = "rgb"

BASE_PATH = r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon"

DATASETS = {
    "cable": os.path.join(BASE_PATH, "cable", "train", "good"),
    "metal_nut": os.path.join(BASE_PATH, "metal_nut", "train", "good"),
    "screw": os.path.join(BASE_PATH, "screw", "train", "good"),
    "transistor": os.path.join(BASE_PATH, "transistor", "train", "good"),
    "zipper": os.path.join(BASE_PATH, "zipper", "train", "good"),
}

UNIFIED_DATASET = os.path.join(BASE_PATH, "dataset_unified")
TRAIN_GOOD_DIR = os.path.join(UNIFIED_DATASET, "train", "good")

os.makedirs(TRAIN_GOOD_DIR, exist_ok=True)
