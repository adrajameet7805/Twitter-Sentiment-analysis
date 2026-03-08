import streamlit as st
import pandas as pd
import plotly.express as px
from backend.video_processor import process_video
from utils.emotion_config import (
    EMOTION_MAP,
    emotion_label_with_emoji,
    emotion_color,
    emotion_style_class,
)

_SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv"]

def render_video_analysis(engine):
    """
    Video Sentiment Analysis UI following user's specific requirements.
    """
    st.markdown("<h2 class='section-header'>🎬 Video Sentiment Analysis</h2>", unsafe_allow_html=True)

    if not engine:
        st.error("⚠️ Emotion model not loaded.")
        st.stop()

    # ── Upload Section ────────────────────────────────────────────────────────
    uploaded_video = st.file_uploader("Upload your video file", type=_SUPPORTED_FORMATS)

    if uploaded_video is None:
        st.info("📂 Please upload a video file to begin analysis.")
        return

    # ── Video Preview ─────────────────────────────────────────────────────────
    st.markdown("### 🎥 Video Preview")
    st.video(uploaded_video)

    if not st.button("🚀 Analyse Video", type="primary", width="stretch"):
        return

    # ── Processing ────────────────────────────────────────────────────────────
    uploaded_video.seek(0)
    with st.spinner("🧠 Analyzing Face Emotions..."):
        result = process_video(uploaded_video, engine)

    if result.get("error"):
        st.error(f"❌ Error: {result['error']}")
        return

    # ── Results Layout ────────────────────────────────────────────────────────
    st.markdown("---")
    
    # 1. Face Emotion Analysis
    st.markdown("### 🎭 Face Emotion Analysis")
    face_emotion = result.get("face_emotion", "Unknown")
    face_conf = result.get("face_confidence", 0.0)
    
    if face_emotion != "No Face Detected":
        st.info(f"Dominant Face Emotion: **{face_emotion}** ({face_conf:.1%})")
        
        # Display preview frames with bounding boxes
        preview_frames = result.get("face_preview_frames", [])
        if preview_frames:
            st.markdown("#### Detected Faces & Bounding Boxes")
            cols = st.columns(min(len(preview_frames), 5))
            for i, frame in enumerate(preview_frames):
                if i < 5:
                    cols[i].image(frame, use_column_width=True)
                    
        # 2. Emotion Timeline
        st.markdown("### 📊 Emotion Timeline")
        timeline = result.get("emotion_timeline", [])
        if timeline:
            df_timeline = pd.DataFrame(timeline)
            fig_timeline = px.line(df_timeline, x="time", y="emotion", markers=True, title="Emotion Changes Over Time")
            fig_timeline.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E5E7EB"})
            st.plotly_chart(fig_timeline, width="stretch")

        # 3. Emotion Distribution
        st.markdown("### 📈 Emotion Distribution")
        dist = result.get("emotion_distribution", {})
        if dist:
            df_dist = pd.DataFrame(list(dist.items()), columns=["Emotion", "Percentage"])
            fig_dist = px.bar(df_dist, x="Emotion", y="Percentage", color="Emotion", title="Emotion Frequency (%)")
            fig_dist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E5E7EB"})
            st.plotly_chart(fig_dist, width="stretch")
    else:
        st.warning("No face detected in video analysis.")

    # ── Final Stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
        <div style="font-size: 0.8rem; color: #9CA3AF;">
            Frames Analyzed: {result.get('frames_analyzed', 0)} | 
            Faces Detected: {result.get('faces_detected', 0)} | 
            Processing Time: {result.get('processing_time', '0s')}
        </div>
    """, unsafe_allow_html=True)
