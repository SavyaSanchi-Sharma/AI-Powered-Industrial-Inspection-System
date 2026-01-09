# src/temporal.py
from collections import deque

class TemporalDefectFilter:
    def __init__(self, window=10, min_defect_frames=4):
        self.window = window
        self.min_defect_frames = min_defect_frames
        self.history = deque(maxlen=window)

    def update(self, defect_present: bool):
        self.history.append(defect_present)

        count = sum(self.history)
        return count >= self.min_defect_frames
