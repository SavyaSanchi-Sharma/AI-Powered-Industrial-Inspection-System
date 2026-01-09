# src/feature_extractor.py
import torch
import numpy as np

def extract_patch_features(feature_map):
    """
    feature_map: (B, C, H, W)
    returns: (num_patches, C)
    """
    B, C, H, W = feature_map.shape

    patches = feature_map.permute(0, 2, 3, 1)  # B, H, W, C
    patches = patches.reshape(-1, C)

    return patches.detach().cpu().numpy()
