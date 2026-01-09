# src/eval/pixel_eval.py

import os
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from backbone import ResNetBackbone
from feature_extractor import extract_patch_features
from anomaly_scoring import AnomalyScorer
from heatmap import build_heatmap
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


def pixel_scores(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return None, None

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

    H, W = features.shape[2], features.shape[3]
    heatmap = build_heatmap(scores, (H, W))

    # Resize heatmap to image resolution
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    return heatmap.flatten(), img.shape[:2]


def evaluate_pixel_level(dataset_root):
    """
    dataset_root example:
    C:/.../testing/cable.tar/cable
    """

    all_scores = []
    all_labels = []

    test_root = os.path.join(dataset_root, "test")
    gt_root = os.path.join(dataset_root, "ground_truth")

    for defect_type in os.listdir(test_root):
        test_dir = os.path.join(test_root, defect_type)
        gt_dir = os.path.join(gt_root, defect_type)

        if not os.path.isdir(test_dir):
            continue

        for fname in os.listdir(test_dir):
            if not fname.endswith(".png"):
                continue

            img_path = os.path.join(test_dir, fname)
            mask_name = fname.replace(".png", "_mask.png")
            mask_path = os.path.join(gt_dir, mask_name)

            if not os.path.exists(mask_path):
                continue

            heatmap_flat, img_shape = pixel_scores(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if heatmap_flat is None or mask is None:
                continue

            # Resize mask to match heatmap dimensions if needed
            mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
            mask = (mask > 0).astype(np.uint8).flatten()

            all_scores.extend(heatmap_flat.tolist())
            all_labels.extend(mask.tolist())

    pixel_auroc = roc_auc_score(all_labels, all_scores)
    return pixel_auroc
