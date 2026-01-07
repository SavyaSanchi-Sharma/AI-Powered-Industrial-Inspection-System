# #!/usr/bin/env python3
# """
# flatten_dataset.py

# Flatten nested files in 'test' and 'ground_truth' for specified dataset folders.

# Usage:
#   python flatten_dataset.py --root "C:\path\to\dataset" --dry-run
# """

# import argparse
# from pathlib import Path
# import shutil
# import sys

# def get_unique_path(dest: Path) -> Path:
#     if not dest.exists():
#         return dest
#     stem = dest.stem
#     suffix = dest.suffix
#     parent = dest.parent
#     i = 1
#     while True:
#         candidate = parent / f"{stem}_{i}{suffix}"
#         if not candidate.exists():
#             return candidate
#         i += 1

# def flatten_one_section(target_dir: Path, dry_run: bool = True):
#     if not target_dir.exists():
#         return

#     # Collect files not directly in target_dir
#     files = [p for p in target_dir.rglob('*') if p.is_file() and p.parent != target_dir]
#     for src in files:
#         dest = target_dir / src.name
#         dest_unique = get_unique_path(dest)
#         if dry_run:
#             print(f"[DRY-RUN] Would move: '{src}' -> '{dest_unique}'")
#         else:
#             dest.parent.mkdir(parents=True, exist_ok=True)
#             shutil.move(str(src), str(dest_unique))
#             print(f"Moved: '{src}' -> '{dest_unique}'")

#     # Remove empty directories (deepest first)
#     for d in sorted([p for p in target_dir.rglob('*') if p.is_dir()], key=lambda p: -len(str(p))):
#         try:
#             if not any(d.iterdir()):
#                 if dry_run:
#                     print(f"[DRY-RUN] Would remove empty directory: '{d}'")
#                 else:
#                     d.rmdir()
#                     print(f"Removed empty directory: '{d}'")
#         except Exception as e:
#             print(f"Warning: could not remove '{d}': {e}", file=sys.stderr)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root', required=False,
#                         default=r"C:\Users\mdkai\OneDrive\Desktop\AIML PROJECTS\Aerothon\Aerothon",
#                         help="Root dataset folder")
#     parser.add_argument('--folders', nargs='*',
#                         default=['cable','metal_nut','screw','transistor','zipper'],
#                         help="Dataset folders to process")
#     parser.add_argument('--sections', nargs='*', default=['test','ground_truth'],
#                         help="Subfolders to flatten")
#     parser.add_argument('--dry-run', action='store_true', help="Show actions without making changes")
#     args = parser.parse_args()

#     root = Path(args.root)
#     for f in args.folders:
#         for s in args.sections:
#             target = root / f / s
#             print(f"Processing: {f}/{s} -> {target}")
#             flatten_one_section(target, dry_run=args.dry_run)
#     print("Done.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
flatten_aircraft.py

Flatten nested files in:
  aircraft_damage_dataset_v1/train/{crack,dent}
  aircraft_damage_dataset_v1/test/{crack,dent}
  aircraft_damage_dataset_v1/valid/{crack,dent}

Default behavior: dry-run. Use --apply to perform moves.
"""

import argparse
from pathlib import Path
import shutil
import sys

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suf}"
        if not candidate.exists():
            return candidate
        i += 1

def flatten_dir(target: Path, apply: bool):
    if not target.exists():
        print(f"Skipping missing: {target}")
        return
    # find files not directly in target
    files = [p for p in target.rglob('*') if p.is_file() and p.parent != target]
    for src in files:
        dest = target / src.name
        final = unique_path(dest)
        if apply:
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(final))
            print(f"MOVED: '{src}' -> '{final}'")
        else:
            print(f"[DRY] Would move: '{src}' -> '{final}'")
    # remove empty directories (deepest first)
    dirs = sorted([d for d in target.rglob('*') if d.is_dir()], key=lambda p: -len(str(p)))
    for d in dirs:
        try:
            if not any(d.iterdir()):
                if apply:
                    d.rmdir()
                    print(f"REMOVED EMPTY: '{d}'")
                else:
                    print(f"[DRY] Would remove empty dir: '{d}'")
        except Exception as e:
            print(f"Warning removing '{d}': {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='Repo root (default: current dir)')
    parser.add_argument('--apply', action='store_true', help='Actually move files (default is dry-run)')
    args = parser.parse_args()

    root = Path(args.root)
    base = root / 'aircraft_damage_dataset_v1'
    sections = []
    for s in ('train', 'test', 'valid'):
        for sub in ('crack','dent'):
            sections.append(base / s / sub)

    print("Dry-run mode (no changes). Use --apply to perform moves." if not args.apply else "Applying changes.")
    for sec in sections:
        print(f"\nProcessing: {sec}")
        flatten_dir(sec, apply=args.apply)
    print("\nDone.")

if __name__ == '__main__':
    main()