"""
Week 7-8 — YOLOv8 inference wrapper for the Streamlit app.

Kept separate from app.py on purpose: the detection logic has nothing to
do with Streamlit, and keeping it framework-agnostic means it can be unit
tested (see test_detector.py) without spinning up a Streamlit session, and
reused elsewhere (a CLI tool, a batch script, an API) without dragging the
UI code along with it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# NOTE: `ultralytics` (and the torch it pulls in) is imported lazily inside
# DefectDetector.__init__ rather than at module scope. This keeps the
# Defect / DetectionResult dataclasses importable — and therefore unit
# testable (see test_report_logic.py) — in environments without a GPU or
# a multi-GB torch install, e.g. a CI runner that's only checking report
# logic, not running inference.

# Class names must match Week 5/6's data.yaml exactly — if these drift out
# of sync with the weights the model was trained on, class *indices* still
# resolve correctly (Ultralytics stores names inside the .pt checkpoint),
# but this fallback list is kept here for reference / offline tooling.
NEU_DET_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


@dataclass
class Defect:
    """A single detected defect instance."""
    class_name: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) in pixel coordinates

    @property
    def confidence_pct(self) -> int:
        return round(self.confidence * 100)

    @property
    def display_name(self) -> str:
        # "pitted_surface" -> "Pitted Surface", "rolled-in_scale" -> "Rolled-in Scale"
        return self.class_name.replace("_", " ").replace("-", "-").title()


@dataclass
class DetectionResult:
    annotated_image: Image.Image
    defects: List[Defect] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def is_clean(self) -> bool:
        return len(self.defects) == 0


class DefectDetector:
    """Thin wrapper around a trained YOLOv8 checkpoint (best.pt from Week 6)."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        weights_path = str(weights_path)
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path}'. Point this at the "
                f"best.pt produced by Week 6 training (runs/detect/<run>/weights/best.pt), "
                f"or the copy checked into this folder's weights/ directory."
            )
        from ultralytics import YOLO  # lazy import — see module docstring/NOTE above

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image: Image.Image) -> DetectionResult:
        """Run inference on a single PIL image and return structured results."""
        results = self.model.predict(
            source=np.array(image.convert("RGB")),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        result = results[0]

        defects: List[Defect] = []
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = tuple(float(v) for v in box.xyxy[0])
            defects.append(Defect(class_name=names[cls_id], confidence=conf, box_xyxy=xyxy))

        # Ultralytics' own .plot() draws boxes + labels + confidences using
        # the same colors/thickness convention as training-time validation
        # plots — reusing it keeps the app's visuals consistent with the
        # screenshots already in Week 6's RESULTS_TEMPLATE.md instead of
        # inventing a second, different-looking drawing style.
        annotated_bgr = result.plot()
        annotated_image = Image.fromarray(annotated_bgr[:, :, ::-1])  # BGR -> RGB

        inference_ms = float(result.speed.get("inference", 0.0))

        # Sort by confidence, most confident defect first — that's the
        # order the report should reason about them in too.
        defects.sort(key=lambda d: d.confidence, reverse=True)

        return DetectionResult(annotated_image=annotated_image, defects=defects, inference_ms=inference_ms)
