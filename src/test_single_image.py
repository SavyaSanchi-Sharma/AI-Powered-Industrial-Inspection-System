# src/test_single_image.py
import torch
import cv2
import numpy as np

from backbone import ResNetBackbone
from feature_extractor import extract_patch_features
from anomaly_scoring import AnomalyScorer
from heatmap import build_heatmap
from preprocess import preprocess_image

# Load memory bank
memory_bank = np.load("memory_bank.npy")

# Initialize scorer
scorer = AnomalyScorer(memory_bank, k=5)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetBackbone().to(device)
model.eval()

# Load test image (GOOD or DEFECT)
img_path = "C:\\Users\\mdkai\\OneDrive\\Desktop\\AIML PROJECTS\\Aerothon\\Aerothon\\metal_nut\\test\\003.png"   # change this
img = preprocess_image(img_path)

img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

with torch.no_grad():
    features = model(img_tensor)

patch_features = extract_patch_features(features)
scores = scorer.score(patch_features)

# Build heatmap
H, W = features.shape[2], features.shape[3]
heatmap = build_heatmap(scores, (H, W))

# Visualization
heatmap_color = cv2.applyColorMap((heatmap * 255).astype("uint8"),
                                  cv2.COLORMAP_JET)

original = cv2.resize(cv2.imread(img_path), (224, 224))
output = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

cv2.imshow("Anomaly Heatmap", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
