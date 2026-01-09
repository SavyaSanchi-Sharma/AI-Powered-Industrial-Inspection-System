# src/eval/run_frame_eval.py

import sys
import os

# Ensure `src/` is on sys.path so imports like `backbone` resolve when
# running this script from the repository root.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from frame_eval import evaluate_dataset

DATASETS = {
    "CABLE": r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon\testing\cable.tar\cable",
    "SCREW": r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon\testing\screw.tar\screw",
    "METAL_NUT": r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon\testing\metal_nut.tar\metal_nut",
    "ZIPPER": r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon\testing\zipper.tar\zipper",
    "TRANSISTOR": r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon\testing\transistor.tar\transistor",
}

for name, path in DATASETS.items():
    auroc, acc = evaluate_dataset(path)
    print(f"{name}: AUROC = {auroc:.3f}, Accuracy = {acc:.3f}")
