"""
Week 6 — Train YOLOv8 on the NEU-DET surface defect dataset.

Usage:
    python train_yolo.py --data ./NEU-DET/data.yaml --model yolov8s.pt --epochs 100

Notes on hyperparameter choices for THIS dataset specifically (1,800 images,
6 classes, grayscale industrial surface images — not natural COCO-style photos):

- Starting from COCO-pretrained weights (transfer learning) instead of random
  init is essential at this dataset size. Training from scratch on 1,800
  images will heavily underfit.
- Grayscale surface textures don't benefit much from color-based augmentation
  (hue/saturation jitter) since there's no color signal to begin with —
  we tone those down and lean on geometric augmentation (flips, rotation,
  scale) instead, which reflects real variation in how the camera/part
  is positioned on the inspection line.
- A moderate patience for early stopping avoids wasting epochs once the
  model plateaus, but isn't so aggressive that it stops during normal
  noisy validation fluctuation on a small validation set.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./NEU-DET/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                         choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25,
                         help="Early stopping patience (epochs with no val improvement)")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default=None,
                         help="Run name; defaults to the model size, e.g. 'yolov8s'")
    args = parser.parse_args()

    run_name = args.name or Path(args.model).stem

    model = YOLO(args.model)  # loads COCO-pretrained weights, transfer learning

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=run_name,

        # Optimizer / schedule
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,             # final LR = lr0 * lrf, cosine decay
        warmup_epochs=3,
        cos_lr=True,

        # Augmentation — geometric-heavy, color-light (grayscale surface images)
        hsv_h=0.0,             # no hue jitter — grayscale-derived images
        hsv_s=0.0,             # no saturation jitter
        hsv_v=0.3,             # brightness jitter is still useful (lighting variation)
        degrees=10.0,          # small rotations — camera isn't perfectly aligned in production
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        flipud=0.2,            # vertical flip is fine here (no fixed "up" for surface texture)
        mosaic=1.0,            # mosaic augmentation helps a lot on small datasets
        mixup=0.1,

        # Regularization
        weight_decay=0.0005,

        # Reproducibility
        seed=42,
        deterministic=True,

        plots=True,             # saves results.png, confusion_matrix.png automatically
        save=True,
        val=True,
    )

    print(f"\nTraining complete. Best weights: {args.project}/{run_name}/weights/best.pt")
    print(f"Training curves + confusion matrix saved under: {args.project}/{run_name}/")


if __name__ == "__main__":
    main()
