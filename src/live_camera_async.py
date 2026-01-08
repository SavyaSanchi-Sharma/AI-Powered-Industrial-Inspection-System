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

# ---------------- SETUP ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

memory_bank = np.load("memory_bank.npy")
scorer = AnomalyScorer(memory_bank, k=5)

model = ResNetBackbone().to(DEVICE)
model.eval()

cap = cv2.VideoCapture(0)

frame_queue = deque(maxlen=1)
result_frame = None

fps_counter = FPSCounter()

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

        roi, (x, y, w, h) = get_center_roi(frame)
        img = preprocess_frame(roi)

        img_tensor = torch.tensor(img).permute(2, 0, 1)\
            .unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            features = model(img_tensor)

        patch_features = extract_patch_features(features)
        scores = scorer.score(patch_features)

        H, W = features.shape[2], features.shape[3]
        heatmap = build_heatmap(scores, (H, W))

        boxes = extract_bboxes(
            heatmap,
            roi_shape=(h, w),
            threshold=0.7,
            min_area=300
        )
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype("uint8"),
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.resize(heatmap_color, (w, h))
        overlay = frame.copy()
        overlay[y:y+h, x:x+w] = cv2.addWeighted(
            overlay[y:y+h, x:x+w], 0.6, heatmap_color, 0.4, 0
        )

        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(
                overlay,
                (x+bx, y+by),
                (x+bx+bw, y+by+bh),
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
