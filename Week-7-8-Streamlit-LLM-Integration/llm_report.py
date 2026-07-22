"""
Week 7-8 — LLM report generation via a local Ollama server running Llama 3.2.

Design choices worth calling out (these are the kind of thing an
interviewer will ask about):

1. Severity is NOT decided by the LLM (see report_utils.compute_severity).
   The LLM is only asked to write the natural-language Summary and to
   phrase 3-4 Recommended Actions drawn from a fixed pool for the given
   severity tier. This bounds the LLM to what it's actually good at
   (language) and keeps it away from what it's unreliable at (consistent,
   repeatable risk judgements) — the report shouldn't change conclusions
   between two runs on the identical input just because sampling
   temperature nudged it.

2. The prompt requires strict JSON output rather than free text. Parsing
   free-form LLM prose reliably is fragile; asking for
   {"summary": "...", "recommended_actions": ["...", "..."]} and validating
   it is far more robust for a tool that has to run unattended in a demo.

3. If Ollama isn't running, or the model isn't pulled, or the request
   times out, the app must not just crash — a QA tool that goes down
   because a background LLM daemon isn't running is a bad tool. See
   generate_fallback_report() for the deterministic, template-based
   report used when the LLM path is unavailable. This IS one of the
   things flagged as not fully verified end-to-end — see this folder's
   README "What Wasn't Verified" section.
"""

import json
import re
from typing import List, Optional

import requests

from detector import Defect
from report_utils import (
    RECOMMENDED_ACTIONS,
    InspectionReport,
    Severity,
    compute_severity,
    today_str,
)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
REQUEST_TIMEOUT_S = 30


def is_ollama_reachable(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _build_prompt(defects: List[Defect], severity: Severity) -> str:
    if defects:
        defect_lines = "\n".join(
            f"- {d.display_name}: {d.confidence_pct}% confidence" for d in defects
        )
    else:
        defect_lines = "- No defects detected."

    action_pool = "\n".join(f"- {a}" for a in RECOMMENDED_ACTIONS[severity])

    return f"""You are a quality-assurance assistant at a steel manufacturing plant. \
You are writing the "Summary" and "Recommended Action" sections of an automated \
inspection report for a steel sheet that just passed under a defect-detection camera.

Detected defects (from a YOLOv8 model, class name and confidence):
{defect_lines}

The severity for this inspection has already been assessed as: {severity.value}.
Do not change or restate a different severity — write your summary consistently with it.

Candidate recommended actions for this severity tier (choose and lightly \
rephrase 3-4 of the most relevant ones for these specific defects — do not \
invent actions outside this list):
{action_pool}

Respond with ONLY a valid JSON object, no markdown fences, no extra text, in \
exactly this shape:
{{
  "summary": "2-4 sentences, factual and professional, written the way a QA \
engineer would write it for a shift report. Reference the specific defects \
detected by name.",
  "recommended_actions": ["action 1", "action 2", "action 3"]
}}
"""


def _extract_json(text: str) -> Optional[dict]:
    """Ollama models sometimes wrap JSON in ```json fences or add stray text
    despite instructions. Strip fences, then grab the outermost {...} block."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_llm_report(
    defects: List[Defect],
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> InspectionReport:
    """Call a local Ollama server to generate the Summary + Recommended
    Action text. Raises RuntimeError on any failure — callers should catch
    this and fall back to generate_fallback_report()."""
    severity = compute_severity(defects)
    prompt = _build_prompt(defects, severity)

    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host} with model '{model}'. "
            f"Is `ollama serve` running and has `ollama pull {model}` been run? "
            f"Underlying error: {e}"
        ) from e

    raw_text = resp.json().get("response", "")
    parsed = _extract_json(raw_text)
    if not parsed or "summary" not in parsed or "recommended_actions" not in parsed:
        raise RuntimeError(
            f"Ollama returned a response that wasn't valid JSON in the expected "
            f"shape. Raw response (first 300 chars): {raw_text[:300]!r}"
        )

    actions = parsed["recommended_actions"]
    if not isinstance(actions, list) or not actions:
        actions = RECOMMENDED_ACTIONS[severity][:3]

    return InspectionReport(
        inspection_date=today_str(),
        defects=defects,
        severity=severity,
        summary=str(parsed["summary"]).strip(),
        recommended_actions=[str(a) for a in actions],
        source="llm",
    )


def generate_fallback_report(defects: List[Defect]) -> InspectionReport:
    """Deterministic, template-based report used when Ollama is unreachable.
    No LLM call — pure rule-based text, so the app always produces
    *something* usable even with the LLM daemon down."""
    severity = compute_severity(defects)

    if not defects:
        summary = (
            "No surface defects were detected on the inspected steel sheet. "
            "The part meets the visual quality threshold configured for this "
            "inspection station."
        )
    elif len(defects) == 1:
        d = defects[0]
        summary = (
            f"One surface defect was detected on the steel sheet: "
            f"{d.display_name} at {d.confidence_pct}% confidence. "
            f"This defect may affect surface quality and should be reviewed "
            f"before the part proceeds further down the line."
        )
    else:
        names = ", ".join(d.display_name for d in defects[:-1]) + f" and {defects[-1].display_name}"
        summary = (
            f"{len(defects)} surface defects were detected on the steel sheet "
            f"({names}). These defects may affect surface quality and "
            f"structural consistency and should be inspected before further "
            f"processing."
        )

    actions = RECOMMENDED_ACTIONS[severity][:4]

    return InspectionReport(
        inspection_date=today_str(),
        defects=defects,
        severity=severity,
        summary=summary,
        recommended_actions=actions,
        source="fallback",
    )


def generate_report(
    defects: List[Defect],
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
) -> InspectionReport:
    """Preferred entry point: try the LLM, transparently fall back if it fails."""
    try:
        return generate_llm_report(defects, host=host, model=model)
    except RuntimeError:
        return generate_fallback_report(defects)
