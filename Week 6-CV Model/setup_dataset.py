"""
Week 6 — Dataset setup for NEU-YOLO surface defect detection.

Downloads the dataset from Kaggle and validates it's in the shape
Ultralytics expects before you spend GPU time training on it.

Requires a Kaggle API token (kaggle.json) — see:
https://www.kaggle.com/docs/api#authentication

Usage:
    python setup_dataset.py --out_dir ./NEU-DET
"""

import argparse
import shutil
from pathlib import Path

import yaml

EXPECTED_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def download_dataset(out_dir: Path):
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub not installed. Run: pip install kagglehub\n"
            "Then re-run this script. Alternatively, download manually from "
            "https://www.kaggle.com/datasets/zymzym/neu-yolo and unzip into "
            f"{out_dir}"
        )

    print("Downloading zymzym/neu-yolo from Kaggle...")
    path = kagglehub.dataset_download("zymzym/neu-yolo")
    print(f"Downloaded to cache at: {path}")

    src = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = out_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"Copied dataset into: {out_dir.resolve()}")


def ensure_data_yaml(out_dir: Path):
    """Make sure a valid data.yaml exists with correct relative paths + class names."""
    yaml_path = out_dir / "data.yaml"

    # CORRECTED PATHS
    train_dir = out_dir / "train" / "train" / "images"
    val_dir = out_dir / "valid" / "valid" / "images"
    test_dir = out_dir / "test" / "images" # Assuming test is not nested, or doesn't exist yet

    cfg = {
        "path": str(out_dir.resolve()),
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()) if val_dir.exists() else str(train_dir.resolve()),
    }
    if test_dir.exists():
        cfg["test"] = str(test_dir.resolve())

    if "names" not in cfg or not cfg["names"]:
        cfg["names"] = EXPECTED_CLASSES
    cfg["nc"] = len(cfg["names"])

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"data.yaml written to {yaml_path}")
    print(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def validate(out_dir: Path, cfg: dict):
    ok = True
    for split_key in ["train", "val", "test"]:
        if split_key not in cfg:
            continue
        img_dir = Path(cfg[split_key])
        if not img_dir.exists():
            print(f"[FAIL] {split_key} image dir does not exist: {img_dir}")
            ok = False
            continue
        n_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        lbl_dir = img_dir.parent / "labels"
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"[{split_key}] images: {n_images}, label files: {n_labels}")
        if n_images == 0:
            print(f"[FAIL] No images found in {img_dir}")
            ok = False
        if n_labels == 0:
            print(f"[WARN] No label files found in {lbl_dir} — check dataset structure")

    if len(cfg.get("names", [])) != 6:
        print(f"[WARN] Expected 6 classes, found {len(cfg.get('names', []))}: {cfg.get('names')}")

    print("\nDataset check:", "PASSED" if ok else "FAILED — fix issues above before training")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./NEU-DET")
    parser.add_argument("--skip_download", action="store_true",
                         help="Use if you already downloaded the dataset manually")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if not args.skip_download:
        download_dataset(out_dir)

    cfg = ensure_data_yaml(out_dir)
    validate(out_dir, cfg)


if __name__ == "__main__":
    main()
