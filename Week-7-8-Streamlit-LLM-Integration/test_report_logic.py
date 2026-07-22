"""
Week 7-8 — Unit tests for severity scoring, fallback report generation, and
markdown formatting. Deliberately does NOT import detector.py's YOLO
dependency at module scope where avoidable, since these should run fast
in CI without a GPU or torch installed just to check the report logic.

Usage:
    python -m pytest test_report_logic.py -v
    (or just: python test_report_logic.py)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detector import Defect  # dataclass only, no torch import triggered by this class itself
from llm_report import generate_fallback_report
from report_utils import Severity, compute_severity


def make_defect(name, conf):
    return Defect(class_name=name, confidence=conf, box_xyxy=(0, 0, 10, 10))


class TestSeverity(unittest.TestCase):
    def test_no_defects_is_none(self):
        self.assertEqual(compute_severity([]), Severity.NONE)

    def test_single_low_confidence_scratch_is_low(self):
        defects = [make_defect("scratches", 0.3)]
        self.assertEqual(compute_severity(defects), Severity.LOW)

    def test_single_high_confidence_crazing_is_higher(self):
        defects = [make_defect("crazing", 0.95)]
        severity = compute_severity(defects)
        self.assertIn(severity, (Severity.MEDIUM, Severity.HIGH))

    def test_many_structural_defects_is_critical(self):
        defects = [
            make_defect("crazing", 0.9),
            make_defect("inclusion", 0.85),
            make_defect("rolled-in_scale", 0.8),
        ]
        self.assertEqual(compute_severity(defects), Severity.CRITICAL)

    def test_display_name_formatting(self):
        self.assertEqual(make_defect("pitted_surface", 0.9).display_name, "Pitted Surface")
        self.assertEqual(make_defect("rolled-in_scale", 0.9).display_name, "Rolled-In Scale")


class TestFallbackReport(unittest.TestCase):
    def test_clean_part_report(self):
        report = generate_fallback_report([])
        self.assertEqual(report.severity, Severity.NONE)
        self.assertIn("No surface defects", report.summary)

    def test_two_defect_report_names_both(self):
        defects = [make_defect("scratches", 0.96), make_defect("pitted_surface", 0.91)]
        report = generate_fallback_report(defects)
        self.assertIn("Scratches", report.summary)
        self.assertIn("Pitted Surface", report.summary)

    def test_markdown_contains_required_sections(self):
        defects = [make_defect("scratches", 0.96)]
        report = generate_fallback_report(defects)
        md = report.to_markdown()
        for section in ("Inspection Date", "Detected Defects", "Summary", "Severity", "Recommended Action"):
            self.assertIn(section, md)


if __name__ == "__main__":
    unittest.main()
