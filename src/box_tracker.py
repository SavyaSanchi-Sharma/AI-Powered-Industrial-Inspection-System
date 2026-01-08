# src/box_tracker.py
class BoxTracker:
    def __init__(self, ttl=5):
        self.ttl = ttl
        self.boxes = []

    def update(self, new_boxes):
        updated = []

        for box in new_boxes:
            updated.append({"box": box, "life": self.ttl})

        for b in self.boxes:
            b["life"] -= 1
            if b["life"] > 0:
                updated.append(b)

        self.boxes = updated
        return [b["box"] for b in self.boxes]
