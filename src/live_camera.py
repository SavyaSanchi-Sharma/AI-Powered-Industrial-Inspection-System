# src/live_camera.py
import cv2
import torch
import numpy as np

from backbone import ResNetBackbone
from feature_extractor import extract_patch_features
from anomaly_scoring import AnomalyScorer
from heatmap import build_heatmap
from preprocess import preprocess_image
from bbox import extract_bboxes


# -------------------- SETUP --------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

memory_bank = np.load("memory_bank.npy")
scorer = AnomalyScorer(memory_bank, k=5)

model = ResNetBackbone().to(DEVICE)
model.eval()

cap = cv2.VideoCapture(0)  # 0 = webcam, or RTSP URL

# -------------------- LOOP --------------------
cv2.namedWindow("Live Anomaly Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Live Anomaly Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (224, 224))
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame_resized)

    img = preprocess_image(temp_path)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        features = model(img_tensor)

    patch_features = extract_patch_features(features)
    scores = scorer.score(patch_features)

    H, W = features.shape[2], features.shape[3]
    heatmap = build_heatmap(scores, (H, W))

    # Bounding boxes
    boxes = extract_bboxes(heatmap)

    # Overlay heatmap
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype("uint8"), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(frame_resized, 0.6, heatmap_color, 0.4, 0)

    # Draw boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Status text
    status = "DEFECT" if len(boxes) > 0 else "NORMAL"
    color = (0, 0, 255) if status == "DEFECT" else (0, 255, 0)

    cv2.putText(
        overlay, status, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )

    cv2.imshow("Live Anomaly Detection", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
