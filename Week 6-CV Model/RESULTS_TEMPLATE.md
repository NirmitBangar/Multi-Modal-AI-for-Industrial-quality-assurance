# Week 6 Results — YOLOv8 Surface Defect Detection

> Fill this in with your actual numbers and screenshots after running `train_yolo.py` +
> `evaluate.py`. Don't submit this file with the placeholders still in it.

## Setup
- **Model used**: `yolov8_.pt` (nano / small / medium — state which, and why)
- **Epochs**: ___
- **Image size**: ___
- **Batch size**: ___
- **Dataset**: NEU-DET, 1,800 images, 6 classes, split ___/___/___ (train/val/test)

## Final Metrics (test split)

| Metric | Value | Assignment Target |
|---|---|---|
| Precision | ___ | ≥0.80 (good) / ≥0.90 (excellent) |
| Recall | ___ | ≥0.60 (good) / ≥0.75 (excellent) |
| mAP@50 | ___ | ≥0.75 (good) / ≥0.85 (excellent) |
| mAP@50-95 | ___ | ≥0.40 (good) / ≥0.50 (excellent) |

## Per-Class Breakdown

| Class | Precision | Recall | AP@50 |
|---|---|---|---|
| crazing | | | |
| inclusion | | | |
| patches | | | |
| pitted_surface | | | |
| rolled-in_scale | | | |
| scratches | | | |

*Connect this back to Week 5's EDA — if a class underperforms, is it the class with fewer instances or more box-geometry variance you flagged earlier?*

## Screenshots to Attach
- [ ] Training curves (`runs/detect/<run_name>/results.png`)
- [ ] Confusion matrix (`runs/detect/<run_name>/confusion_matrix.png`)
- [ ] Sample predictions (`eval_output/sample_predictions/`)

## Model Size Comparison (if you ran `compare_models.py`)

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Train time |
|---|---|---|---|---|---|
| yolov8n | | | | | |
| yolov8s | | | | | |
| yolov8m | | | | | |

**Chosen model and why**: ___ (justify with both accuracy *and* the training/inference time tradeoff — this is the kind of reasoning that reads well in an interview, not just "it had the highest number")

## Notes / Issues Encountered
- (e.g., any classes with weak recall, training instability, anything EDA predicted that showed up here)
