"""
Week 6 — Evaluate a trained YOLOv8 model and print the metrics required
by the assignment: Precision, Recall, mAP@50, mAP@50-95.

Also saves sample prediction images (ground truth vs. predicted boxes)
for the report's screenshot requirement.

Usage:
    python evaluate.py --weights runs/detect/yolov8s/weights/best.pt --data ./NEU-DET/data.yaml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="./NEU-DET/data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out_dir", type=str, default="eval_output")
    args = parser.parse_args()

    model = YOLO(args.weights)

    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, plots=True)

    precision = metrics.box.mp        # mean precision across classes
    recall = metrics.box.mr           # mean recall across classes
    map50 = metrics.box.map50         # mAP at IoU 0.5
    map50_95 = metrics.box.map        # mAP at IoU 0.5:0.95

    print("\n" + "=" * 50)
    print(f"RESULTS ON '{args.split}' SPLIT — {Path(args.weights).parent.parent.name}")
    print("=" * 50)
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"mAP@50      : {map50:.4f}")
    print(f"mAP@50-95   : {map50_95:.4f}")
    print("=" * 50)

    print("\nPer-class breakdown:")
    class_names = metrics.names
    for i, name in class_names.items():
        try:
            p = metrics.box.p[i]
            r = metrics.box.r[i]
            ap50 = metrics.box.ap50[i]
            print(f"  {name:20s}  P={p:.3f}  R={r:.3f}  AP50={ap50:.3f}")
        except (IndexError, KeyError):
            continue

    # Run predictions on a handful of test images for the report screenshots
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.predict(
        source=str(Path(args.data).parent / args.split / "images"),
        imgsz=args.imgsz,
        conf=0.25,
        save=True,
        project=str(out_dir),
        name="sample_predictions",
        exist_ok=True,
    )
    print(f"\nSample prediction images saved to: {out_dir / 'sample_predictions'}")
    print("Training curves + confusion matrix are in the same folder as your weights "
          "(runs/detect/<run_name>/results.png and confusion_matrix.png)")


if __name__ == "__main__":
    main()
