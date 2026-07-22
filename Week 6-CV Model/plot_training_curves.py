"""
Week 6 — Plot training curves from full_training_history.json
(produced by extract_checkpoint_metadata.py).

Usage:
    pip install matplotlib
    python plot_training_curves.py --history results/full_training_history.json --out results/results_reconstructed.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="results/full_training_history.json")
    parser.add_argument("--out", type=str, default="results/results_reconstructed.png")
    args = parser.parse_args()

    tr = json.load(open(args.history))
    epochs = tr["epoch"]
    n_epochs = len(epochs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(epochs, tr["train/box_loss"], label="train")
    axes[0, 0].plot(epochs, tr["val/box_loss"], label="val")
    axes[0, 0].set_title("Box Loss")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("epoch")

    axes[0, 1].plot(epochs, tr["train/cls_loss"], label="train")
    axes[0, 1].plot(epochs, tr["val/cls_loss"], label="val")
    axes[0, 1].set_title("Class Loss")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("epoch")

    axes[0, 2].plot(epochs, tr["train/dfl_loss"], label="train")
    axes[0, 2].plot(epochs, tr["val/dfl_loss"], label="val")
    axes[0, 2].set_title("DFL Loss")
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("epoch")

    axes[1, 0].plot(epochs, tr["metrics/precision(B)"], color="tab:orange")
    axes[1, 0].set_title("Precision (B)")
    axes[1, 0].set_xlabel("epoch")

    axes[1, 1].plot(epochs, tr["metrics/recall(B)"], color="tab:green")
    axes[1, 1].set_title("Recall (B)")
    axes[1, 1].set_xlabel("epoch")

    axes[1, 2].plot(epochs, tr["metrics/mAP50(B)"], label="mAP50")
    axes[1, 2].plot(epochs, tr["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[1, 2].set_title("mAP (B)")
    axes[1, 2].legend()
    axes[1, 2].set_xlabel("epoch")

    fig.suptitle(f"YOLOv8s — NEU-DET Training Curves ({n_epochs} epochs, reconstructed from checkpoint history)")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
