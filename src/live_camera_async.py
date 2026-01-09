# src/live_camera_async.py

import cv2
import torch
import numpy as np
import threading
import os
from collections import deque
import csv
import time

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

# FPS
fps_counter = FPSCounter()

# ---- DEFECT COORDINATE LOGGER ----
csv_file = open("defect_coordinates.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "frame_id",
    "defect_id",
    "x",
    "y",
    "w",
    "h",
    "cx",
    "cy",
    "score"
])
csv_file.flush()
frame_id = 0

# ---- MODEL + MEMORY BANK ----
memory_bank_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "memory_bank.npy")
)
if not os.path.exists(memory_bank_path):
    raise FileNotFoundError(f"Missing memory bank file: {memory_bank_path}")

memory_bank = np.load(memory_bank_path)
scorer = AnomalyScorer(memory_bank, k=5)

model = ResNetBackbone().to(DEVICE)
model.eval()

# ---- CAMERA ----
cap = cv2.VideoCapture(0)

frame_queue = deque(maxlen=1)
result_frame = None

# ---- TEMPORAL STABILITY ----
temporal_filter = TemporalDefectFilter(
    window=10,
    min_defect_frames=4
)

box_tracker = BoxTracker(ttl=5)

# ---- FULLSCREEN WINDOW ----
cv2.namedWindow("Live Anomaly Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Live Anomaly Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ---------------- INFERENCE THREAD ----------------

def inference_loop():
    global result_frame, frame_id

    while True:
        if not frame_queue:
            continue

        frame = frame_queue[-1]
        frame_id += 1

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
        raw_boxes = boxes.copy()
        # ---- TEMPORAL FILTERING ----
        defect_now = len(raw_boxes) > 0
        stable_defect = temporal_filter.update(defect_now)

        if stable_defect:
            display_boxes = box_tracker.update(raw_boxes)
        else:
            display_boxes = box_tracker.update([])

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

        # ---- LIVE DEFECT COORDINATES OUTPUT ----

        # Clear terminal (dynamic refresh)
        print("\033c", end="")
        print(f"Frame {frame_id} | Detected defects: {len(boxes)}")

        for i, (bx, by, bw, bh) in enumerate(raw_boxes, start=1):

            # Full-frame coordinates
            fx = x + bx
            fy = y + by
            cx = fx + bw // 2
            cy = fy + bh // 2

            # Confidence score (mean heatmap inside box)
            box_heatmap = heatmap[by:by+bh, bx:bx+bw]
            score = float(box_heatmap.mean()) if box_heatmap.size > 0 else 0.0

            # ---- TERMINAL OUTPUT ----
            print(
                f"Defect {i}: "
                f"x={fx}, y={fy}, w={bw}, h={bh}, "
                f"cx={cx}, cy={cy}, score={score:.3f}"
            )

            # ---- CSV LOGGING ----
            csv_writer.writerow([
                time.time(),
                frame_id,
                i,
                fx,
                fy,
                bw,
                bh,
                cx,
                cy,
                round(score, 4)
            ])

            # ---- ON-SCREEN DRAWING ----
            for (bx, by, bw, bh) in display_boxes:
                cv2.rectangle(
                    overlay,
                    (x + bx, y + by),
                    (x + bx + bw, y + by + bh),
                    (0, 0, 255),
                    2
            )


            cv2.putText(
                overlay,
                f"ID:{i} ({fx},{fy})",
                (fx, fy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
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

# ---------------- CLEANUP ----------------

cap.release()
cv2.destroyAllWindows()
csv_file.close()
