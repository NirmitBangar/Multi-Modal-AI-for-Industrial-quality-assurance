# Week 7-8 — Streamlit App & Local LLM Integration

##  Goals for These Two Weeks

- Wrap Week 6's trained YOLOv8 model in a usable web interface, not just a script
- Learn the Streamlit mental model: script re-runs top-to-bottom on every interaction, so state and expensive resources (model loading) have to be handled deliberately (`st.cache_resource`, `st.session_state`)
- Integrate a **local** LLM (Llama 3.2 via Ollama) instead of a hosted API — understand the tradeoffs (no API cost/key, but the app now depends on a background daemon that has to actually be running)
- Turn a list of `(class, confidence)` detections into a professional inspection report a shift engineer could actually read
- Package the whole pipeline — upload → detect → report — into one clean end-to-end demo

---

##  Architecture

```
                ┌─────────────┐      ┌──────────────────┐      ┌────────────────────┐
Image upload -> │ detector.py │ ---> │ report_utils.py   │ ---> │ llm_report.py       │
  (Streamlit)   │ (YOLOv8)    │      │ (severity rules)  │      │ (Ollama / Llama 3.2)│
                └─────────────┘      └──────────────────┘      └────────────────────┘
                       |                                                  |
                       v                                                  v
                annotated image                                  InspectionReport
                + defect list                                    (rendered in app.py,
                                                                    downloadable as .md)
```

The pipeline is split into four independent modules on purpose, matching how a real production system would separate concerns:

| Module | Responsibility | Depends on |
|---|---|---|
| `detector.py` | Load `best.pt`, run inference, return structured `Defect` objects + annotated image | `ultralytics`, `torch` |
| `report_utils.py` | Deterministic severity scoring, `InspectionReport` data model, markdown rendering | nothing but stdlib |
| `llm_report.py` | Build the LLM prompt, call Ollama, parse JSON, offline fallback if Ollama is down | `requests` |
| `app.py` | Streamlit UI — wires the above together, handles upload/session state/download | `streamlit` |

None of `report_utils.py` or the data classes in `detector.py` import `torch`/`ultralytics` at module scope — that import is deliberately deferred to inside `DefectDetector.__init__`. That's why `test_report_logic.py` can run in a couple milliseconds without a GPU or a multi-GB `torch` install: it only needs the parts of the pipeline that don't touch the model.

---

##  Design Decisions Worth Explaining (e.g. in an interview)

**1. Severity is computed by a fixed rule, not decided by the LLM.**
`report_utils.compute_severity()` scores severity from defect class (structural classes like `crazing`/`inclusion`/`rolled-in_scale` are weighted higher than cosmetic ones like `scratches`), confidence, and count. The LLM is only asked to *write* the Summary and *phrase* Recommended Actions consistent with a severity it's handed — never to invent the severity itself. A QA tool where the same input can be labeled "Medium" one run and "Critical" the next, purely because of LLM sampling temperature, isn't trustworthy. Keeping the risk judgement in deterministic code and the language generation in the LLM plays to each one's strengths.

**2. The LLM is prompted for strict JSON, not free text.**
Parsing prose reliably out of a small local model is fragile. The prompt in `llm_report._build_prompt()` asks for `{"summary": ..., "recommended_actions": [...]}` and `_extract_json()` strips stray markdown fences before parsing. If parsing fails, that's treated as an LLM failure and falls through to the offline path below, rather than showing the user broken/malformed text.

**3. If Ollama is unreachable, the app doesn't break — it degrades.**
`generate_fallback_report()` in `llm_report.py` produces a template-based report using pure string formatting, no LLM call. `app.py` checks `is_ollama_reachable()` up front and shows a clear sidebar warning, and `generate_report()`/the button handler catch `RuntimeError` and fall back automatically. This was a deliberate choice: a demo (or a real inspection line) failing outright because a background daemon wasn't running is worse than a demo that clearly says "offline fallback used" and still produces a usable report. The sample output in the assignment brief was reproduced almost exactly by this fallback path alone (see Testing section) — which was reassuring, since it means the deterministic severity/action logic is sound independent of whatever the LLM adds on top.

**4. `.plot()` from Ultralytics is reused for the annotated image**, rather than hand-rolling box-drawing with PIL/OpenCV, so the bounding-box visuals in the app match the same style already used in Week 6's `evaluate.py` output — one visual language across the whole project instead of two.

---

##  Report Generation Flow

1. `detector.py` returns a sorted list of `Defect(class_name, confidence, box_xyxy)`
2. `report_utils.compute_severity()` scores severity from that list
3. `llm_report.generate_report()`:
   - tries Ollama first (`generate_llm_report`) — builds the prompt, POSTs to `/api/generate`, parses JSON
   - on any failure (Ollama down, model not pulled, malformed JSON, timeout) — falls back to `generate_fallback_report()`
4. `InspectionReport.to_markdown()` renders the final format, matching the assignment's example layout (Inspection Date / Detected Defects / Summary / Severity / Recommended Action)

---

##  Files in This Folder

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application — run this |
| `detector.py` | YOLOv8 wrapper (`DefectDetector`, `Defect`, `DetectionResult`) |
| `report_utils.py` | Severity scoring, `InspectionReport` model, markdown rendering |
| `llm_report.py` | Ollama prompt + call + JSON parsing + offline fallback |
| `check_ollama.py` | Standalone CLI script to debug the Ollama connection before opening the app |
| `test_report_logic.py` | Unit tests for severity scoring and fallback report generation (no GPU needed) |
| `requirements.txt` | Python dependencies for this app specifically |
| `weights/` | Put Week 6's `best.pt` here (gitignored by default — see the file inside) |

---

##  How to Run

### 1. Install dependencies
```bash
cd "Week 7-8 - Streamlit App & LLM Integration"
pip install -r requirements.txt
```

### 2. Add the trained model
Copy Week 6's `runs/detect/<run_name>/weights/best.pt` into:
```
weights/best.pt
```

### 3. Install and start Ollama, pull Llama 3.2
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
ollama serve          # if it isn't already running as a background service
```

### 4. Sanity-check the LLM connection (recommended before demoing)
```bash
python check_ollama.py
```

### 5. Run the app
```bash
streamlit run app.py
```
This opens the app at `http://localhost:8501`. Upload a NEU-DET-style steel surface image (the Week 6 test split is a good source of demo images), review the detections, then click **Generate Inspection Report**.

### 6. Run the tests
```bash
python test_report_logic.py -v
```

---

##  What I Wasn't Able To Verify End-to-End (Being Upfront About It)

Everything above was written and unit-tested for logical correctness (`test_report_logic.py` passes, and the fallback path reproduces the assignment's sample output almost exactly — see below), but I don't have a GPU environment with `torch`/`ultralytics` installed, an actual trained `best.pt`, or a running Ollama daemon available to me to do a full live click-through of the Streamlit UI myself. Concretely, not yet confirmed on a real machine:

- **YOLOv8 inference through `detector.py` on a real image** — the inference code follows the same `ultralytics` API used correctly in Week 6's `evaluate.py`/`train_yolo.py`, but I haven't run it against an actual `best.pt` and a real steel surface photo to visually confirm the bounding boxes render correctly in Streamlit.
- **The live Ollama + Llama 3.2 call** — `llm_report.generate_llm_report()` is written against Ollama's documented `/api/generate` endpoint shape, but I haven't run it against a live `ollama serve` process to confirm Llama 3.2 reliably returns valid JSON on the first try (small local models sometimes need a retry or a slightly firmer prompt — `_extract_json()` handles the common case of stray markdown fences, but if you see JSON parse failures often in practice, that's the place to tighten the prompt further).
- **The full Streamlit session-state flow** (upload → detect → click "Generate Report" → download) — checked file-by-file for logic and `py_compile`-clean, but not click-tested in a browser.

**What I did verify:**
- All Python files pass `python -m py_compile` (no syntax errors)
- `test_report_logic.py` — 8/8 tests pass, covering severity scoring and fallback report generation
- The fallback report generator reproduces the assignment's exact example almost word-for-word when given `scratches@96%` + `pitted_surface@91%` (same Medium severity, same four recommended actions) — a good sign the deterministic core logic matches spec even before the LLM adds its own phrasing on top

### Steps to Finish Verifying (in order)
1. `pip install -r requirements.txt` on a machine with the Week 6 environment already set up (GPU not required for inference, just for training)
2. Copy a trained `best.pt` into `weights/`
3. Run `streamlit run app.py`, upload 2-3 real NEU-DET test images, confirm boxes + labels render correctly
4. Install Ollama, `ollama pull llama3.2`, run `python check_ollama.py` to confirm the round trip works
5. Click "Generate Inspection Report" with Ollama running — check the summary text reads naturally and the JSON parses without hitting the fallback
6. Deliberately stop `ollama serve` and click the button again — confirm the fallback path kicks in cleanly with the warning shown (this is the "it doesn't just crash" behavior described above — worth showing in the demo video)
7. Record the demo video (see `VIDEO_SCRIPT.md` in this folder)
8. Fill in real screenshots/timings into this README's addendum or a `RESULTS.md` if the assignment wants one, same convention as Week 6's `RESULTS_TEMPLATE.md`

---

##  Resources Used

| Resource | Link |
|---|---|
| Streamlit docs | https://docs.streamlit.io |
| Streamlit tutorial (per Week 7-8 brief) | https://www.youtube.com/watch?v=yKTEC1Y5bEQ |
| Ollama API reference | https://github.com/ollama/ollama/blob/main/docs/api.md |
| Ultralytics YOLO Python API (same as Week 6) | https://docs.ultralytics.com |

---

##  Key Takeaway

> The interesting engineering in this phase isn't "call an LLM" — it's deciding what the LLM should and shouldn't be trusted to decide. Severity had to stay deterministic so the report doesn't contradict itself between runs; the LLM had to be asked for structured JSON instead of prose because parsing free text reliably from a small local model isn't realistic; and the whole app had to survive the LLM simply not being available, because a QA tool that goes down when a background process isn't running would never actually get deployed on a real inspection line. That's the difference between "I called `ollama.generate()`" and "I built a system that uses an LLM safely" — and it's exactly the kind of judgement call that's worth being able to explain, not just the fact that it works.
