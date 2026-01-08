# src/live_camera_async.py

import cv2
import torch
import numpy as np
import threading
from collections import deque

from backbone import ResNetBackbone
from feature_extractor import extract_patch_features
from anomaly_scoring import AnomalyScorer
from heatmap import build_heatmap
from preprocess import preprocess_frame
from bbox import extract_bboxes
from fps import FPSCounter
from roi import get_center_roi
from temporal import TemporalDefectFilter
from box_tracker import BoxTracker

# ---------------- SETUP ----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

memory_bank = np.load("memory_bank.npy")
scorer = AnomalyScorer(memory_bank, k=5)

model = ResNetBackbone().to(DEVICE)
model.eval()

cap = cv2.VideoCapture(0)

frame_queue = deque(maxlen=1)
result_frame = None

fps_counter = FPSCounter()

# ---- PHASE 7: Temporal stability ----
temporal_filter = TemporalDefectFilter(
    window=10,
    min_defect_frames=4
)

box_tracker = BoxTracker(ttl=5)

# Fullscreen window
cv2.namedWindow("Live Anomaly Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Live Anomaly Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ---------------- INFERENCE THREAD ----------------

def inference_loop():
    global result_frame

    while True:
        if not frame_queue:
            continue

        frame = frame_queue[-1]

        # ROI extraction
        roi, (x, y, w, h) = get_center_roi(frame)

        # Preprocess
        img = preprocess_frame(roi)
        img_tensor = (
            torch.tensor(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        # CNN forward
        with torch.no_grad():
            features = model(img_tensor)

        # Patch scoring
        patch_features = extract_patch_features(features)
        scores = scorer.score(patch_features)

        # Heatmap
        H, W = features.shape[2], features.shape[3]
        heatmap = build_heatmap(scores, (H, W))

        # Bounding boxes (ROI space)
        boxes = extract_bboxes(
            heatmap,
            roi_shape=(h, w),
            threshold=0.7,
            min_area=300
        )

        # ---- PHASE 7: Temporal filtering ----
        defect_now = len(boxes) > 0
        stable_defect = temporal_filter.update(defect_now)

        if stable_defect:
            boxes = box_tracker.update(boxes)
        else:
            boxes = box_tracker.update([])

        # Heatmap overlay
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype("uint8"),
            cv2.COLORMAP_JET
        )
        heatmap_color = cv2.resize(heatmap_color, (w, h))

        overlay = frame.copy()
        overlay[y:y+h, x:x+w] = cv2.addWeighted(
            overlay[y:y+h, x:x+w], 0.6, heatmap_color, 0.4, 0
        )

        # Draw bounding boxes (correctly offset)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(
                overlay,
                (x + bx, y + by),
                (x + bx + bw, y + by + bh),
                (0, 0, 255),
                2
            )

        result_frame = overlay

# ---------------- START THREAD ----------------

threading.Thread(target=inference_loop, daemon=True).start()

# ---------------- DISPLAY LOOP ----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_queue.append(frame)

    if result_frame is not None:
        fps = fps_counter.update()

        cv2.putText(
            result_frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Live Anomaly Detection",
            cv2.resize(result_frame, (1920, 1080))
        )

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
