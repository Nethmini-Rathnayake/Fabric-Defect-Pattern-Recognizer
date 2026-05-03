"""Streamlit dashboard for real-time fabric defect detection."""

import os
import time
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

from src.detection.model import load_model, DEFECT_CLASSES
from src.pipeline.run_inference import PREPROCESS, CONF_THRESHOLD, COLORS, annotate_frame, predict_frame

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/efficientnet/best.pt")

st.set_page_config(
    page_title="Fabric Defect Recognizer",
    page_icon="🧵",
    layout="wide",
)

st.title("🧵 Fabric Defect Pattern Recognizer")
st.caption("Real-time AI detection for Sri Lankan apparel factories")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    video_source = st.text_input("Video Source", value="0", help="0 for webcam, or path to video file")
    conf_threshold = st.slider("Confidence Threshold", 0.5, 1.0, CONF_THRESHOLD, 0.01)
    show_rag = st.checkbox("Show Root-Cause Explanation", value=False)
    run_btn = st.button("Start Detection", type="primary")
    stop_btn = st.button("Stop")
    st.divider()
    st.header("Session Summary")
    summary_placeholder = st.empty()


@st.cache_resource
def get_model():
    if not Path(MODEL_PATH).exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(MODEL_PATH, device), device


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

with col2:
    st.subheader("Defect Distribution")
    chart_placeholder = st.empty()
    st.subheader("Latest Alert")
    alert_placeholder = st.empty()


def update_summary(counts: dict):
    df = pd.DataFrame(
        [(k.replace("_", " ").title(), v) for k, v in counts.items() if v > 0],
        columns=["Defect", "Count"],
    )
    summary_placeholder.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        fig = px.bar(df, x="Count", y="Defect", orientation="h", color="Count", color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
        chart_placeholder.plotly_chart(fig, use_container_width=True)


if run_btn:
    result = get_model()
    if result is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Train the model first or update MODEL_PATH in .env.")
        st.stop()

    model, device = result
    source = int(video_source) if video_source.strip().isdigit() else video_source.strip()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error(f"Cannot open video source: {source}")
        st.stop()

    defect_counts = defaultdict(int)
    fps_counter, t_start = 0, time.time()
    stop_flag = st.session_state.get("stop", False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break

        defect_class, confidence = predict_frame(model, frame, device)

        if confidence >= conf_threshold:
            defect_counts[defect_class] += 1
            frame = annotate_frame(frame.copy(), defect_class, confidence)
            alert_placeholder.warning(
                f"**{defect_class.replace('_', ' ').title()}** detected — confidence: {confidence:.1%}"
            )

        fps_counter += 1
        fps = fps_counter / max(time.time() - t_start, 1e-6)
        fps_placeholder.caption(f"FPS: {fps:.1f}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

        if fps_counter % 30 == 0:
            update_summary(defect_counts)

    cap.release()
    update_summary(defect_counts)
    st.success("Detection session ended.")
