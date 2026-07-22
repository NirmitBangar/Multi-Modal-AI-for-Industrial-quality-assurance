"""
Week 6 — Train and compare YOLOv8 nano / small / medium on the same dataset,
to empirically justify (not assume) which model size is the right pick
for this assignment.

Usage:
    python compare_models.py --data ./NEU-DET/data.yaml --epochs 100
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./NEU-DET/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    results_table = []

    for model_name in MODELS:
        run_name = Path(model_name).stem
        print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

        model = YOLO(model_name)
        t0 = time.time()
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=25,
            project="runs/detect",
            name=run_name,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
            degrees=10.0, translate=0.1, scale=0.4,
            fliplr=0.5, flipud=0.2, mosaic=1.0, mixup=0.1,
            seed=42, deterministic=True, plots=True,
        )
        train_time = time.time() - t0

        best_weights = f"runs/detect/{run_name}/weights/best.pt"
        eval_model = YOLO(best_weights)
        metrics = eval_model.val(data=args.data, imgsz=args.imgsz, split="test")

        # Rough inference speed check
        t0 = time.time()
        eval_model.predict(source=metrics.save_dir, imgsz=args.imgsz, verbose=False)
        infer_time_per_run = time.time() - t0

        results_table.append({
            "model": model_name,
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "train_time_min": round(train_time / 60, 1),
        })

    print("\n\n" + "=" * 80)
    print(f"{'Model':<12}{'Precision':<12}{'Recall':<10}{'mAP@50':<10}{'mAP@50-95':<12}{'Train (min)':<12}")
    print("=" * 80)
    for r in results_table:
        print(f"{r['model']:<12}{r['precision']:<12}{r['recall']:<10}{r['map50']:<10}"
              f"{r['map50_95']:<12}{r['train_time_min']:<12}")

    print("\nUse this table + your training-time budget to justify your final model "
          "choice in RESULTS_TEMPLATE.md — 'highest mAP' alone isn't a full justification "
          "if it comes with a training/inference cost that doesn't fit a real deployment.")


if __name__ == "__main__":
    main()
