# src/bbox.py
import cv2
import numpy as np

def extract_bboxes(
    heatmap,
    roi_shape,
    threshold=0.7,
    min_area=300
):
    """
    heatmap: (224, 224) normalized
    roi_shape: (H, W) of ROI in original frame
    """

    # Resize heatmap to ROI size
    heatmap_resized = cv2.resize(
        heatmap,
        (roi_shape[1], roi_shape[0])
    )

    # Threshold
    binary = (heatmap_resized > threshold).astype("uint8") * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return boxes
