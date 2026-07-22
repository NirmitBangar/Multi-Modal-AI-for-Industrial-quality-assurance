"""
Week 7-8 — Severity scoring and shared report data structures.

Why this exists as its own module: severity has to be *consistent* whether
the final paragraph text comes from the LLM or from the offline fallback
template (see llm_report.py). If severity were left entirely to the LLM's
judgement, two runs on the identical detection could disagree with each
other, and a "Medium" one run / "Critical" the next run is a bad look for
a QA tool a human is meant to trust. So severity is computed by a fixed,
inspectable rule here, and the LLM is only ever asked to *explain* a
severity it's given — not invent one from scratch.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List

from detector import Defect


class Severity(str, Enum):
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# Base severity weight per defect class. These are a judgement call, not a
# measured quantity — reasoning documented so it can be argued with:
#   - rolled-in_scale / crazing: sub-surface / structural in origin, more
#     likely to propagate under load -> weighted higher
#   - inclusion: foreign material embedded in the surface, also structural
#   - patches / pitted_surface / scratches: predominantly cosmetic /
#     surface-level in most cases -> weighted lower, unless present in
#     large numbers or at high confidence
CLASS_SEVERITY_WEIGHT = {
    "crazing": 3,
    "inclusion": 3,
    "rolled-in_scale": 3,
    "patches": 2,
    "pitted_surface": 2,
    "scratches": 1,
}


def compute_severity(defects: List[Defect]) -> Severity:
    """Deterministic severity from defect classes, confidences, and count."""
    if not defects:
        return Severity.NONE

    score = 0.0
    for d in defects:
        weight = CLASS_SEVERITY_WEIGHT.get(d.class_name, 2)
        # A low-confidence detection contributes less to overall severity
        # than a high-confidence one of the same class.
        score += weight * d.confidence

    # More independent defects on one part is worse than one defect
    # repeated, even at matched average weight/confidence.
    score += 0.5 * (len(defects) - 1)

    if score >= 6:
        return Severity.CRITICAL
    if score >= 4:
        return Severity.HIGH
    if score >= 2:
        return Severity.MEDIUM
    return Severity.LOW


# Base recommended actions per severity tier. The LLM is prompted to select
# and phrase from this pool rather than invent unrelated actions, so the
# "Recommended Action" section stays operationally sane even if the LLM
# output is otherwise creative.
RECOMMENDED_ACTIONS = {
    Severity.NONE: [
        "No action required — part meets surface quality standards.",
    ],
    Severity.LOW: [
        "Log the detection for trend monitoring.",
        "No immediate rework required; re-inspect at next scheduled check.",
    ],
    Severity.MEDIUM: [
        "Inspect the affected region manually.",
        "Remove or repair the damaged section if required.",
        "Monitor the production line for recurring defects.",
        "Perform quality verification before shipment.",
    ],
    Severity.HIGH: [
        "Quarantine the part pending manual inspection.",
        "Escalate to the shift quality engineer.",
        "Check upstream process parameters (rolling temperature, roller condition) for a root cause.",
        "Do not release for shipment until re-inspected.",
    ],
    Severity.CRITICAL: [
        "Immediately quarantine the part — do not forward to the next process step.",
        "Halt the line if the same defect recurs on the next 2-3 units.",
        "Trigger a root-cause investigation with the process engineering team.",
        "Notify the shift supervisor before resuming normal throughput.",
    ],
}


@dataclass
class InspectionReport:
    inspection_date: str
    defects: List[Defect]
    severity: Severity
    summary: str
    recommended_actions: List[str]
    source: str  # "llm" or "fallback" — shown in the UI so the user knows which path generated it

    def to_markdown(self) -> str:
        lines = ["# Inspection Report", ""]
        lines.append(f"**Inspection Date:** {self.inspection_date}")
        lines.append("")
        lines.append("**Detected Defects:**")
        if self.defects:
            for d in self.defects:
                lines.append(f"- {d.display_name} ({d.confidence_pct}%)")
        else:
            lines.append("- None detected")
        lines.append("")
        lines.append("**Summary:**")
        lines.append(self.summary)
        lines.append("")
        lines.append(f"**Severity:** {self.severity.value}")
        lines.append("")
        lines.append("**Recommended Action:**")
        for a in self.recommended_actions:
            lines.append(f"- {a}")
        return "\n".join(lines)


def today_str() -> str:
    return date.today().strftime("%d %B %Y")
