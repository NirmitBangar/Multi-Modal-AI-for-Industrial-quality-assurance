"""
Week 5 — Exploratory Data Analysis for the NEU-YOLO Surface Defect Dataset
============================================================================
Run this BEFORE training in Week 6. It answers: is the data actually clean
and balanced enough to expect a good model, or are there landmines?

Expected dataset layout (standard YOLO / Ultralytics format, matches the
Kaggle "neu-yolo" dataset):

    NEU-DET/
        train/
            images/*.jpg
            labels/*.txt
        valid/
            images/*.jpg
            labels/*.txt
        test/
            images/*.jpg
            labels/*.txt
        data.yaml

Each label .txt line: "class_id x_center y_center width height" (normalized 0-1)

Usage:
    python eda_neu_dataset.py --data_dir /path/to/NEU-DET
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

try:
    import imagehash
    HASHING_AVAILABLE = True
except ImportError:
    HASHING_AVAILABLE = False
    print("[warn] `imagehash` not installed — near-duplicate detection will be skipped. "
          "Install with: pip install imagehash")


CLASS_NAMES_FALLBACK = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def load_class_names(data_dir: Path):
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", CLASS_NAMES_FALLBACK)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        return names
    print("[warn] data.yaml not found, using fallback NEU-DET class names.")
    return CLASS_NAMES_FALLBACK


def find_split_dirs(data_dir: Path):
    splits = {}
    for split in ["train", "valid", "val", "test"]:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if img_dir.exists():
            splits[split] = (img_dir, lbl_dir)
    return splits


def check_images(img_dir: Path):
    """Return (valid_paths, corrupt_paths, sizes)."""
    valid, corrupt, sizes = [], [], []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes.append(im.size)
            valid.append(p)
        except Exception as e:
            corrupt.append((p, str(e)))
    return valid, corrupt, sizes


def parse_labels(lbl_dir: Path, num_classes: int):
    """Return instance counts per class, box (w,h) list, and a list of malformed-label issues."""
    class_counts = Counter()
    box_dims = []  # (width, height) normalized
    issues = []
    empty_label_files = []

    if not lbl_dir.exists():
        return class_counts, box_dims, issues, empty_label_files

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        lines = lbl_path.read_text().strip().splitlines()
        if len(lines) == 0:
            empty_label_files.append(lbl_path)
            continue
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"{lbl_path.name} line {i}: expected 5 fields, got {len(parts)}")
                continue
            try:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                issues.append(f"{lbl_path.name} line {i}: non-numeric value")
                continue

            if not (0 <= cls_id < num_classes):
                issues.append(f"{lbl_path.name} line {i}: class_id {cls_id} out of range [0,{num_classes-1}]")
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append(f"{lbl_path.name} line {i}: center ({x},{y}) outside [0,1]")
            if w <= 0 or h <= 0:
                issues.append(f"{lbl_path.name} line {i}: non-positive box size ({w},{h})")
                continue

            class_counts[cls_id] += 1
            box_dims.append((w, h))

    return class_counts, box_dims, issues, empty_label_files


def find_near_duplicates(img_dir: Path, threshold: int = 4):
    """Perceptual-hash based near-duplicate detection within a split."""
    if not HASHING_AVAILABLE:
        return []
    hashes = {}
    dupes = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        try:
            h = imagehash.phash(Image.open(p))
        except Exception:
            continue
        for other_p, other_h in hashes.items():
            if h - other_h <= threshold:
                dupes.append((p.name, other_p.name, int(h - other_h)))
        hashes[p.name] = h
    return dupes


def plot_class_distribution(counts_by_split, class_names, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    splits = list(counts_by_split.keys())
    x = np.arange(len(class_names))
    width = 0.8 / max(len(splits), 1)
    for i, split in enumerate(splits):
        counts = [counts_by_split[split].get(c, 0) for c in range(len(class_names))]
        ax.bar(x + i * width, counts, width, label=split)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Instance count")
    ax.set_title("Class distribution by instance count (not image count)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_box_geometry(box_dims, out_path):
    if not box_dims:
        print("[warn] no boxes found, skipping geometry plot")
        return
    widths = [w for w, h in box_dims]
    heights = [h for w, h in box_dims]
    aspect = [w / h for w, h in box_dims if h > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(widths, bins=30, color="steelblue")
    axes[0].set_title("Box width (normalized)")
    axes[1].hist(heights, bins=30, color="salmon")
    axes[1].set_title("Box height (normalized)")
    axes[2].hist(aspect, bins=30, color="seagreen")
    axes[2].set_title("Box aspect ratio (w/h)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_sample_grid(img_dir: Path, lbl_dir: Path, class_names, out_path, n=9):
    img_paths = sorted(img_dir.glob("*"))[:n]
    cols = 3
    rows = int(np.ceil(len(img_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, img_paths):
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        ax.imshow(im)
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                x0 = (xc - bw / 2) * w
                y0 = (yc - bh / 2) * h
                rect = plt.Rectangle((x0, y0), bw * w, bh * h,
                                      fill=False, edgecolor="lime", linewidth=2)
                ax.add_patch(rect)
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                ax.text(x0, max(y0 - 4, 0), label, color="lime", fontsize=8,
                        backgroundcolor="black")
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    for ax in axes[len(img_paths):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the NEU-DET dataset root (train/valid/test)")
    parser.add_argument("--out_dir", type=str, default="eda_output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(data_dir)
    print(f"Classes ({len(class_names)}): {class_names}\n")

    splits = find_split_dirs(data_dir)
    if not splits:
        raise SystemExit(f"No train/valid/test image folders found under {data_dir}")

    counts_by_split = {}
    all_box_dims = []
    total_issues = []

    for split, (img_dir, lbl_dir) in splits.items():
        print(f"=== {split.upper()} ===")
        valid, corrupt, sizes = check_images(img_dir)
        print(f"  Images: {len(valid)} valid, {len(corrupt)} corrupt")
        for p, err in corrupt:
            print(f"    [CORRUPT] {p.name}: {err}")

        if sizes:
            ws, hs = zip(*sizes)
            print(f"  Resolution range: {min(ws)}x{min(hs)} to {max(ws)}x{max(hs)}")
            if len(set(sizes)) > 1:
                print(f"  [note] {len(set(sizes))} distinct resolutions present — "
                      f"YOLO will letterbox-resize automatically, but worth knowing.")

        class_counts, box_dims, issues, empty_lbls = parse_labels(lbl_dir, len(class_names))
        counts_by_split[split] = class_counts
        all_box_dims.extend(box_dims)
        total_issues.extend(issues)

        print(f"  Label issues found: {len(issues)}")
        for issue in issues[:10]:
            print(f"    [ISSUE] {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
        if empty_lbls:
            print(f"  [warn] {len(empty_lbls)} empty label files (images with no annotated defects)")

        for cls_id, count in sorted(class_counts.items()):
            print(f"    {class_names[cls_id]:20s}: {count} instances")

        dupes = find_near_duplicates(img_dir)
        if dupes:
            print(f"  [warn] {len(dupes)} near-duplicate image pairs detected within {split}")
            for a, b, dist in dupes[:5]:
                print(f"    {a} ~ {b} (hash distance {dist})")
        print()

    plot_class_distribution(counts_by_split, class_names, out_dir / "class_distribution.png")
    plot_box_geometry(all_box_dims, out_dir / "box_geometry.png")

    first_split, (img_dir, lbl_dir) = next(iter(splits.items()))
    plot_sample_grid(img_dir, lbl_dir, class_names, out_dir / "sample_grid.png")

    print("=" * 60)
    print(f"SUMMARY: {len(total_issues)} total label issues across all splits.")
    if total_issues:
        print("Fix these before training — a bad box silently caps your mAP.")
    else:
        print("No structural label issues found. Dataset looks clean to train on.")
    print(f"All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
