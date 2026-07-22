"""
Week 7-8 — Streamlit app: upload a steel surface image, detect defects with
the Week 6 YOLOv8 model, generate an AI inspection report with a local
Llama 3.2 via Ollama.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from PIL import Image

from detector import DefectDetector
from llm_report import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    generate_fallback_report,
    generate_llm_report,
    is_ollama_reachable,
)

st.set_page_config(
    page_title="Industrial Quality Assurance — AI Inspection",
    page_icon="🔍",
    layout="wide",
)

DEFAULT_WEIGHTS = str(Path(__file__).parent / "weights" / "best.pt")


# --- Cached resources ---------------------------------------------------
# st.cache_resource so the model is loaded from disk once per session, not
# re-loaded on every rerun (Streamlit reruns the whole script on every
# widget interaction — re-loading a YOLO checkpoint each time would make
# the UI feel broken).
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_detector(weights_path: str, conf: float):
    return DefectDetector(weights_path=weights_path, conf_threshold=conf)


# --- Sidebar: configuration ---------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    weights_path = st.text_input(
        "YOLOv8 weights path",
        value=DEFAULT_WEIGHTS,
        help="Path to best.pt from Week 6 training. Place it in "
             "weights/best.pt in this folder, or point elsewhere.",
    )
    conf_threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)

    st.divider()
    st.subheader("🤖 LLM (Ollama)")
    ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
    ollama_model = st.text_input("Ollama model", value=DEFAULT_MODEL)

    ollama_up = is_ollama_reachable(ollama_host)
    if ollama_up:
        st.success(f"Ollama reachable at {ollama_host}")
    else:
        st.warning(
            "Ollama not reachable — reports will use the offline "
            "rule-based fallback instead of the LLM. Run `ollama serve` "
            f"and `ollama pull {ollama_model}` to enable AI-generated summaries."
        )

    st.divider()
    st.caption(
        "Built for Summer of Code — Multi-Modal AI for Industrial Quality "
        "Assurance, Weeks 7-8."
    )

# --- Header ---------------------------------------------------------------
st.title("🔍 AI-Powered Industrial Quality Assurance")
st.caption(
    "Upload a steel surface image → detect defects with YOLOv8 → "
    "generate a professional inspection report with a local LLM."
)

# --- Model loading ---------------------------------------------------------
weights_file_exists = Path(weights_path).exists()
if not weights_file_exists:
    st.error(
        f"No weights found at `{weights_path}`. Copy Week 6's `best.pt` "
        f"into `weights/best.pt` in this folder (see this folder's README), "
        f"then reload the page."
    )
    st.stop()

detector = load_detector(weights_path, conf_threshold)

# --- Upload + inference ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a steel surface image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is None:
    st.info("Upload an image to begin inspection.")
    st.stop()

image = Image.open(uploaded_file)

with st.spinner("Running defect detection..."):
    result = detector.predict(image)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Original")
    st.image(image, use_container_width=True)

with col_right:
    st.subheader(f"Detected Defects ({result.defect_count})")
    st.image(result.annotated_image, use_container_width=True)

st.caption(f"Inference time: {result.inference_ms:.1f} ms")

if result.defects:
    st.subheader("📋 Detections")
    st.table(
        [
            {"Defect": d.display_name, "Confidence": f"{d.confidence_pct}%"}
            for d in result.defects
        ]
    )
else:
    st.success("No defects detected above the confidence threshold.")

st.divider()

# --- Report generation ------------------------------------------------
st.subheader("📄 AI Inspection Report")

if "report" not in st.session_state:
    st.session_state.report = None

generate_clicked = st.button("Generate Inspection Report", type="primary")

if generate_clicked:
    if ollama_up:
        try:
            with st.spinner(f"Generating report with {ollama_model} via Ollama..."):
                st.session_state.report = generate_llm_report(
                    result.defects, host=ollama_host, model=ollama_model
                )
        except RuntimeError as e:
            st.warning(f"LLM generation failed, using offline fallback report. Details: {e}")
            st.session_state.report = generate_fallback_report(result.defects)
    else:
        st.session_state.report = generate_fallback_report(result.defects)

report = st.session_state.report
if report is not None:
    if report.source == "fallback":
        st.caption("⚠️ Generated with the offline rule-based fallback (Ollama unavailable).")
    else:
        st.caption(f"✅ Generated by {ollama_model} via Ollama.")

    st.markdown(report.to_markdown())

    st.download_button(
        "⬇️ Download report (.md)",
        data=report.to_markdown(),
        file_name=f"inspection_report_{report.inspection_date.replace(' ', '_')}.md",
        mime="text/markdown",
    )
