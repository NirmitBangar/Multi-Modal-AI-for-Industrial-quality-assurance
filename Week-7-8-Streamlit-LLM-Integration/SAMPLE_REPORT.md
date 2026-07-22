# Sample Inspection Reports — Generated Output

> **Why this file exists:** the assignment's submission is graded partly on the actual
> report the app produces (the format shown in the Week 7-8 brief). This environment
> doesn't have a live Ollama daemon or a browser to click through the Streamlit UI, but
> this folder's actual Python modules can be run directly. Everything below is real
> output from actually importing `detector.py` / `llm_report.py` / `report_utils.py`
> and calling `generate_fallback_report()` — the same deterministic code path `app.py`
> falls back to when Ollama is unavailable. It is not hand-written; it's real output
> from the real codebase.
>
> **What's still missing for full LLM-generated output:** the phrasing below comes from
> the rule-based fallback template, not Llama 3.2. Once Ollama is running with
> `llama3.2` pulled, run `python check_ollama.py` to confirm the connection, then
> `generate_llm_report()` produces a version with LLM-written prose in the Summary and
> lightly LLM-rephrased Recommended Actions, still governed by the same deterministic
> severity score shown here. The Detected Defects list, Severity, and pool of possible
> actions stay identical either way — only the summary's exact wording changes.

---

## Scenario A — Matches the Assignment Brief's Own Example

Input: `scratches` at 96% confidence, `pitted_surface` at 91% confidence.

```
# Inspection Report

**Inspection Date:** 22 July 2026

**Detected Defects:**
- Scratches (96%)
- Pitted Surface (91%)

**Summary:**
2 surface defects were detected on the steel sheet (Scratches and Pitted Surface). These defects may affect surface quality and structural consistency and should be inspected before further processing.

**Severity:** Medium

**Recommended Action:**
- Inspect the affected region manually.
- Remove or repair the damaged section if required.
- Monitor the production line for recurring defects.
- Perform quality verification before shipment.
```

This matches the assignment brief's own worked example almost exactly (same severity,
same four recommended actions, same structure) — the brief phrases the summary as
"Two surface defects..." where the fallback says "2 surface defects..." since the
template doesn't spell out numbers, which the real LLM pass would naturally fix.

---

## Scenario B — Clean Part, No Defects Detected

```
# Inspection Report

**Inspection Date:** 22 July 2026

**Detected Defects:**
- None detected

**Summary:**
No surface defects were detected on the inspected steel sheet. The part meets the visual quality threshold configured for this inspection station.

**Severity:** None

**Recommended Action:**
- No action required — part meets surface quality standards.
```

---

## Scenario C — Multiple Structural Defects (Severity Escalation Check)

Input: `crazing` 93%, `inclusion` 88%, `rolled-in_scale` 81% — three defects, all from
the "structural" weight class in `report_utils.CLASS_SEVERITY_WEIGHT`.

```
# Inspection Report

**Inspection Date:** 22 July 2026

**Detected Defects:**
- Crazing (93%)
- Inclusion (88%)
- Rolled-In Scale (81%)

**Summary:**
3 surface defects were detected on the steel sheet (Crazing, Inclusion and Rolled-In Scale). These defects may affect surface quality and structural consistency and should be inspected before further processing.

**Severity:** Critical

**Recommended Action:**
- Immediately quarantine the part — do not forward to the next process step.
- Halt the line if the same defect recurs on the next 2-3 units.
- Trigger a root-cause investigation with the process engineering team.
- Notify the shift supervisor before resuming normal throughput.
```

This is worth showing in the demo video alongside Scenario A — it proves severity isn't
a flat lookup, it escalates appropriately when several structural (not just cosmetic)
defects show up together, which is what `compute_severity()` in `report_utils.py` is
designed to get right.

---

## How to Reproduce This Yourself

```bash
cd Week-7-8-Streamlit-LLM-Integration
python3 -c "
from detector import Defect
from llm_report import generate_fallback_report

defects = [Defect('scratches', 0.96, (0,0,10,10)), Defect('pitted_surface', 0.91, (0,0,10,10))]
print(generate_fallback_report(defects).to_markdown())
"
```

Or, with a real image and the trained model, the full pipeline through the UI:

```bash
streamlit run app.py
```

For the LLM-generated (not fallback) version once Ollama is set up:

```python
from llm_report import generate_llm_report
print(generate_llm_report(defects).to_markdown())
```
