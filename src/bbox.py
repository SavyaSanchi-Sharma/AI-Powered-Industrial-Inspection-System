# src/bbox.py
import cv2
import numpy as np

def extract_bboxes(heatmap, threshold=0.6, min_area=100):
    binary = (heatmap > threshold).astype("uint8") * 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return boxes
