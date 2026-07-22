# Week 5 — Exploratory Data Analysis (EDA) & Data Cleaning

##  Goals for This Week

- Understand the EDA workflow: what to check before ever training a model
- Practice the standard data-cleaning toolkit (missing values, duplicates, outliers, type coercion)
- Apply EDA specifically to **image datasets for object detection**, which differs from tabular EDA in important ways
- Build habits that will directly feed into Week 6's YOLOv8 training (class balance, image quality, annotation sanity checks)

---

##  Why EDA Matters More for Object Detection Than for Tabular Data

In Week 2 we did EDA on tabular sensor data — checking `df.info()`, `df.describe()`, null counts, correlation heatmaps. Image-based object detection needs a different lens, because the things that silently break a detector are different:

| Tabular EDA checks | Object Detection EDA checks |
|---|---|
| Missing values in columns | Missing/empty label files |
| Outlier numeric values | Corrupt or unreadable images |
| Class imbalance in a target column | Class imbalance in **bounding box instances**, not just images |
| Feature correlation | Bounding box size/aspect ratio distribution |
| Duplicate rows | Duplicate or near-duplicate images (leakage between train/val) |
| — | Label consistency (does class index range match `data.yaml`?) |
| — | Box validity (no zero-width/height boxes, no boxes outside image bounds) |

A model can train "successfully" (loss goes down) even with corrupt annotations — it just quietly caps your mAP. Catching this in Week 5 is what separates a resume-quality Week 6 result from a mediocre one.

---

##  Concepts Covered

### 1. The EDA Workflow (General)
1. **Shape & structure** — how many samples, how are they organized on disk
2. **Missing / corrupt data** — nulls in tabular data; unreadable files or missing labels in image data
3. **Distribution checks** — class balance, value ranges, spread
4. **Duplicates & leakage** — identical or near-identical samples across splits inflate validation metrics
5. **Visual inspection** — plotting is non-negotiable; summary statistics hide problems that a histogram or a sample image grid reveals instantly
6. **Outliers** — decide per-case whether to clip, remove, or keep (an outlier defect image might be the most important training example, unlike an outlier in sensor noise)

### 2. Data Cleaning Techniques Practiced
- Handling missing values: `dropna()`, `fillna()` (mean/median/mode), forward/back-fill for time series
- Deduplication: `drop_duplicates()`, and perceptual image hashing (`imagehash`) for near-duplicate images
- Type coercion and fixing mixed-type columns
- Outlier detection: IQR method, Z-score method
- String cleaning: whitespace, casing, category label normalization (e.g., `"Scratch"` vs `"scratch "` being treated as different classes)

### 3. Object Detection–Specific EDA (New This Week)
- **Class distribution by instance count**, not image count — one image can contain multiple boxes of different classes, so counting images undercounts imbalance
- **Bounding box geometry** — width/height histograms and aspect ratio distribution tell you what anchor/image sizes make sense, and reveal mislabeled boxes (e.g., a box covering the entire image when the defect is small)
- **Image quality** — resolution consistency, corrupt file detection, blurry/low-contrast images
- **Annotation format validation** — for YOLO format, every line in a `.txt` label file must be `class_id x_center y_center width height` in normalized `[0,1]` coordinates; a single malformed line silently breaks training for that image

---

##  Code Written This Week

### `eda_neu_dataset.py`
A complete, runnable EDA script for the NEU-YOLO surface defect dataset used in Week 6. It:
- Scans the dataset and reports image counts per split (train/val/test)
- Counts label instances per class and plots a bar chart of class imbalance
- Computes and plots bounding box width, height, and aspect ratio distributions
- Checks every image for corruption (fails to open / zero-size)
- Checks every label file for malformed lines or out-of-range coordinates
- Flags near-duplicate images using perceptual hashing
- Plots a grid of sample images with their ground-truth boxes drawn on, so you visually confirm the labels look correct before training

Running this **before** Week 6's training is what lets you diagnose a bad mAP later — if class `pitted_surface` has 3x fewer instances than `scratches`, you already know to expect its recall to lag, and you can plan for it (class weighting, augmentation, or just an informed explanation in your report).

---

##  Resources Used

| Resource | Link |
|---|---|
| EDA Cheat Sheet | Notion — Exploratory Data Analysis (EDA) in Python |
| Data Cleaning Walkthrough | Google Colab notebook (provided) |

---

##  Key Takeaway

> EDA on images is not "the tabular EDA workflow, but with pictures." Object detection has failure modes — malformed boxes, class imbalance measured in instances not images, near-duplicate leakage — that don't show up in `.describe()` and won't show up in your loss curve either. They show up as a mysteriously low mAP on one specific class, three days before a deadline. Five minutes of EDA now saves hours of debugging in Week 6.
