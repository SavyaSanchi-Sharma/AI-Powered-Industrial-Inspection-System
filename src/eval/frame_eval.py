# src/eval/frame_eval.py

import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from backbone import ResNetBackbone
from feature_extractor import extract_patch_features
from anomaly_scoring import AnomalyScorer
from preprocess import preprocess_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and memory bank
model = ResNetBackbone().to(DEVICE)
model.eval()

memory_bank_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory_bank.npy'))
if not os.path.exists(memory_bank_path):
    raise FileNotFoundError(f"Missing memory bank file: {memory_bank_path}")
memory_bank = np.load(memory_bank_path)
scorer = AnomalyScorer(memory_bank, k=5)


def image_score(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return None

    img_tensor = (
        torch.tensor(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    with torch.no_grad():
        features = model(img_tensor)

    patch_features = extract_patch_features(features)
    scores = scorer.score(patch_features)

    # Frame-level anomaly score
    return float(scores.max())


def evaluate_dataset(dataset_root):
    """
    dataset_root example:
    C:/.../testing/cable.tar/cable
    """

    y_true = []
    y_scores = []

    # ---------- NORMAL IMAGES ----------
    normal_dir = os.path.join(dataset_root, "train", "good")
    if not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Missing folder: {normal_dir}")

    for fname in os.listdir(normal_dir):
        img_path = os.path.join(normal_dir, fname)
        score = image_score(img_path)
        if score is not None:
            y_true.append(0)   # normal
            y_scores.append(score)

    # ---------- DEFECT IMAGES (ALL SUBFOLDERS) ----------
    defect_root = os.path.join(dataset_root, "test")
    if not os.path.exists(defect_root):
        raise FileNotFoundError(f"Missing folder: {defect_root}")

    for defect_type in os.listdir(defect_root):
        defect_dir = os.path.join(defect_root, defect_type)

        if not os.path.isdir(defect_dir):
            continue

        for fname in os.listdir(defect_dir):
            img_path = os.path.join(defect_dir, fname)
            score = image_score(img_path)
            if score is not None:
                y_true.append(1)   # defect
                y_scores.append(score)

    # ---------- METRICS ----------
    auroc = roc_auc_score(y_true, y_scores)

    # Operating point for accuracy (for reporting)
    threshold = np.percentile(y_scores, 95)
    preds = [1 if s > threshold else 0 for s in y_scores]
    acc = accuracy_score(y_true, preds)

    return auroc, acc
