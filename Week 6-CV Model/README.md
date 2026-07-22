# Week 6 — CV Model: YOLOv8 Surface Defect Detection

##  Goals for This Week

- Train an object detection model (YOLOv8) on the NEU surface-defect dataset
- Understand *why* YOLO's architecture (single-stage, grid-based) suits industrial defect detection
- Learn the object detection metrics that matter — Precision, Recall, mAP@50, mAP@50-95 — and what each one actually tells you
- Experiment across model sizes (nano/small/medium) to understand the accuracy-vs-speed tradeoff, which is exactly the kind of decision a QA engineering team has to make in production

---

##  Dataset

**NEU-DET** (via Kaggle `zymzym/neu-yolo`): 1,800 grayscale steel-surface images across 6 defect classes:

| Class | Description |
|---|---|
| `crazing` | Fine network of surface cracks |
| `inclusion` | Foreign material embedded in the surface |
| `patches` | Irregular surface patches |
| `pitted_surface` | Small pits/craters |
| `rolled-in_scale` | Scale rolled into the surface during manufacturing |
| `scratches` | Linear surface scratches |

300 images per class, already in YOLO format (`images/` + `labels/` with normalized `x_center y_center w h`).

---

##  Why YOLOv8 (and Why Model Size Matters)

Unlike a two-stage detector (Faster R-CNN: propose regions, then classify each), YOLO treats detection as **one regression problem**: divide the image into a grid, predict boxes + class probabilities for every cell in a single forward pass. This is what makes it fast enough for real-time inspection lines — a genuinely relevant property for industrial QA, where the model runs on every unit passing a camera on a production line, not once on a static photo.

`ultralytics` YOLOv8 ships four relevant sizes for this task:

| Model | Params | Relative Speed | Typical Use Case |
|---|---|---|---|
| YOLOv8n (nano) | ~3.2M | Fastest | Edge devices, real-time line speed |
| YOLOv8s (small) | ~11.2M | Fast | Good accuracy/speed balance |
| YOLOv8m (medium) | ~25.9M | Moderate | Higher accuracy, still deployable |
| YOLOv8l (large) | ~43.7M | Slowest | Best accuracy, offline/batch inspection |

On a dataset this small (1,800 images), a larger model isn't automatically better — it has more capacity to overfit. Part of this week's task is *empirically* finding the size that fits this dataset best rather than assuming bigger = better.

---

##  Metrics — What They Actually Mean Here

| Metric | Definition | What it tells you about your QA system |
|---|---|---|
| **Precision** | Of all defects the model *flagged*, what fraction were real defects | Low precision = too many false alarms → operators start ignoring the system |
| **Recall** | Of all real defects, what fraction did the model *catch* | Low recall = missed defects → the actual failure mode you can't afford in QA |
| **mAP@50** | Mean Average Precision at IoU threshold 0.5 (a "loose" localization requirement) | Overall detection quality when boxes only need to roughly overlap the ground truth |
| **mAP@50-95** | mAP averaged over IoU thresholds 0.5 to 0.95 in steps of 0.05 | Stricter, more holistic metric — rewards *precisely* localized boxes, not just "somewhere near" the defect |

In a real QA deployment, recall usually matters more than precision — a missed defect (false negative) that ships to a customer is typically far more costly than a false alarm that gets a human to double-check a good part. Worth calling out explicitly in your report.

---

##  Pipeline (What `train_yolo.py` and `evaluate.py` Do)

1. **Setup**: install `ultralytics`, download the dataset from Kaggle via `kagglehub`/Kaggle API
2. **Verify `data.yaml`**: confirm paths and the 6 class names match what Week 5's EDA found
3. **Train**: fine-tune from COCO-pretrained weights (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) rather than training from scratch — transfer learning matters enormously on a dataset this size
4. **Evaluate**: run `model.val()` on the test split to get final Precision/Recall/mAP@50/mAP@50-95
5. **Visualize**: Ultralytics automatically saves `results.png` (training curves), `confusion_matrix.png`, and prediction images to `runs/detect/train*/` — these are exactly the screenshots the assignment asks for
6. **Compare sizes**: repeat training with n/s/m variants, log metrics side by side, and pick the winner with a stated justification (not just "highest mAP" — factor in training time and inference speed too)

---

##  Files in This Folder

| File | Purpose |
|---|---|
| `setup_dataset.py` | Downloads the NEU-YOLO dataset from Kaggle and validates `data.yaml` |
| `train_yolo.py` | Trains a YOLOv8 model (size selectable via `--model`), with hyperparameters tuned for a small dataset |
| `evaluate.py` | Runs validation on the trained `best.pt`, prints the four required metrics, and saves prediction samples |
| `compare_models.py` | Runs nano/small/medium back-to-back and produces a comparison table |
| `RESULTS_TEMPLATE.md` | Fill this in with your actual numbers + screenshots after running training — this becomes your submission |

---

##  How to Run (Google Colab, free T4 GPU)

```bash
pip install ultralytics kagglehub

python setup_dataset.py
python train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
python evaluate.py --weights runs/detect/train/weights/best.pt
```

Each size (n/s/m) takes roughly 10-25 minutes on a T4 for 100 epochs at this dataset size. Run all three if time allows — `compare_models.py` automates it.

---

##  Realistic Metric Expectations (Not Fabricated Results — Guardrails for Sanity-Checking Your Own Run)

Published results on NEU-DET with YOLOv8 fine-tuning typically land in these ranges. Use this to sanity-check your own numbers — if you're wildly outside these ranges, something in the pipeline (usually the label format or an overfit/undertrained run) needs a second look, not a claim that your model is unusually good or bad:

| Metric | Typical range (yolov8s/m, 100 epochs) |
|---|---|
| Precision | 0.80 – 0.90 |
| Recall | 0.65 – 0.80 |
| mAP@50 | 0.78 – 0.88 |
| mAP@50-95 | 0.42 – 0.52 |

These line up with the assignment's "Good/Excellent" bands. Getting there is very achievable on this dataset with a pretrained `yolov8s.pt` or `yolov8m.pt` backbone and ~100 epochs — the numbers just have to come from an actual training run, since that's the entire point of the exercise (and the metrics + screenshots need to be reproducible if anyone asks to see the run again).

---

##  Resources Used

| Resource | Link |
|---|---|
| Dataset | Kaggle — `zymzym/neu-yolo` |
| Framework | Ultralytics YOLOv8 documentation |

---

##  Key Takeaway

> A single mAP number means nothing without knowing what it's being measured against. The real skill this week isn't running `.train()` — it's connecting Week 5's EDA (which classes are underrepresented, how consistent the boxes are) to *why* the resulting per-class metrics look the way they do, and being able to explain that connection in a report. That's the difference between "I ran a script" and "I understand my model," and it's exactly what shows in an interview.
