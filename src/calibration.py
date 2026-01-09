# src/calibration.py
import numpy as np

class AutoThreshold:
    def __init__(self, method="percentile", value=99.5):
        """
        method: 'percentile' or 'std'
        value:
          - percentile → 99.5 means top 0.5% anomalies
          - std → k value (e.g., 3 = mean + 3*std)
        """
        self.method = method
        self.value = value
        self.scores = []

    def update(self, patch_scores):
        """
        patch_scores: 1D array of patch anomaly scores
        """
        self.scores.extend(patch_scores.tolist())

    def compute(self):
        scores = np.array(self.scores)

        if self.method == "percentile":
            return np.percentile(scores, self.value)

        elif self.method == "std":
            return scores.mean() + self.value * scores.std()

        else:
            raise ValueError("Unknown threshold method")
