# src/anomaly_scoring.py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

class AnomalyScorer:
    def __init__(self, memory_bank, k=5):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        self.nn.fit(memory_bank)

    def score(self, patch_features):
        """
        patch_features: (num_patches, feature_dim)
        returns: anomaly score per patch
        """
        distances, _ = self.nn.kneighbors(patch_features)
        scores = distances.mean(axis=1)
        return scores
