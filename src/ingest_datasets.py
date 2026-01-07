# src/ingest_datasets.py
import os
import cv2
from preprocess import preprocess_image
from config import DATASETS, TRAIN_GOOD_DIR

img_counter = 0

for name, dataset_path in DATASETS.items():
    print(f"📂 Processing {name} dataset...")

    if not os.path.exists(dataset_path):
        print(f"⚠️ Path not found: {dataset_path}")
        continue

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                img = preprocess_image(img_path)

                if img is None:
                    continue

                save_path = os.path.join(
                    TRAIN_GOOD_DIR,
                    f"img_{img_counter:06d}.jpg"
                )

                cv2.imwrite(save_path, (img * 255).astype("uint8"))
                img_counter += 1

print(f"\n✅ PHASE 1 COMPLETE — Total normal images: {img_counter}")
