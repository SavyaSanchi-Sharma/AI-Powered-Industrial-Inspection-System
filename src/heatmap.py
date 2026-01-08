# src/heatmap.py
import numpy as np
import cv2

def build_heatmap(patch_scores, feature_map_shape, image_size=224):
    """
    patch_scores: (H*W,)
    feature_map_shape: (H, W)
    """
    h, w = feature_map_shape
    heatmap = patch_scores.reshape(h, w)

    heatmap = cv2.resize(heatmap, (image_size, image_size))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

    return heatmap
