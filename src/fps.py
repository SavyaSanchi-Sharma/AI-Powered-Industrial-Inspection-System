# src/fps.py
import time

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self):
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        return int(self.fps)
