# src/roi.py
def get_center_roi(frame, scale=0.6):
    h, w, _ = frame.shape
    new_w, new_h = int(w * scale), int(h * scale)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2

    roi = frame[y1:y1+new_h, x1:x1+new_w]
    return roi, (x1, y1, new_w, new_h)
