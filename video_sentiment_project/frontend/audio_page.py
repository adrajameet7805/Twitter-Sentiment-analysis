"""
frontend/audio_page.py
Presentation layer for the Audio Sentiment Analysis page.
Calls backend/audio_processor.py for transcription and
the existing backend/predictor.py for emotion prediction.
UI style is consistent with the existing dark glassmorphism theme.
"""

import streamlit as st

from backend.audio_processor import transcribe_audio
from backend.predictor import predict_emotion_v4
from utils.emotion_config import (
    emotion_label_with_emoji,
    emotion_color,
    emotion_style_class,
)

# Supported upload formats
_SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "m4a"]


def render_audio_analysis(engine):
    """
    Full Audio Sentiment Analysis page.
    Renders upload widget, runs transcription + prediction, displays results.
    """

    # ── Page Header ────────────────────────────────────────────────────────
    st.markdown("<h2 class='section-header'>🎙️ Audio Sentiment Analysis</h2>",
                unsafe_allow_html=True)

    if not engine:
        st.error("⚠️ Emotion model not loaded. Cannot run analysis.")
        st.stop()

    # ── Info Banner ────────────────────────────────────────────────────────
    st.markdown(f"""
        <div style="
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            color: #93C5FD;
            font-size: 0.9rem;
        ">
            🎤 &nbsp;Upload a voice/audio clip and the AI will
            <strong>transcribe</strong> it and detect the <strong>emotional tone</strong>.<br>
            ⚡ &nbsp;Powered by <strong>faster-whisper tiny (int8)</strong> — full audio, maximum speed.<br>
            Supported formats: <code>{", ".join(_SUPPORTED_FORMATS).upper()}</code>
        </div>
    """, unsafe_allow_html=True)

    # ── Upload Widget ──────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload your audio file",
        type=_SUPPORTED_FORMATS,
        label_visibility="collapsed",
        key="audio_uploader",
    )

    if uploaded is None:
        st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem 1rem;
                background: rgba(15, 23, 42, 0.4);
                border: 2px dashed rgba(255,255,255,0.08);
                border-radius: 16px;
                color: #6B7280;
                font-size: 0.95rem;
            ">
                📂 &nbsp;Drag &amp; drop or click above to upload an audio file
            </div>
        """, unsafe_allow_html=True)
        return

    # ── Audio Player ───────────────────────────────────────────────────────
    st.markdown("#### 🔊 Uploaded Audio")
    st.audio(uploaded, format=f"audio/{uploaded.name.rsplit('.',1)[-1]}")

    # ── Analyse Button ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    if not st.button("🚀 Analyse Audio", type="primary", width="stretch",
                     key="audio_analyse_btn"):
        return

    # ── Processing ─────────────────────────────────────────────────────────
    try:
        with st.spinner("🎙️ AI Engine transcribing audio..."):
            audio_bytes = uploaded.read()
            ext = "." + uploaded.name.rsplit(".", 1)[-1].lower()
            result = transcribe_audio(audio_bytes, file_ext=ext)

        # ── Error Guard ────────────────────────────────────────────────────────
        if result.get("error"):
            st.error(f"❌ Transcription failed: {result['error']}. Please try another file.")
            return

        transcription = result.get("transcription", "")
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}. Please try a different audio clip.")
        return

    # (no trimming — full audio is processed)

    # ── Empty Transcription Guard ──────────────────────────────────────────
    if not transcription:
        st.info("ℹ️ No speech detected in the audio. "
                "Try a clearer recording or a different clip.")
        return

    # ── Predict Emotion ────────────────────────────────────────────────────
    pred_results = predict_emotion_v4([transcription], engine)
    emotion, conf, probs = pred_results[0]

    border_color = emotion_color(emotion)
    style_class  = emotion_style_class(emotion)

    # ── Results Layout ─────────────────────────────────────────────────────
    st.markdown("---")
    col_left, col_right = st.columns([1, 1.4])

    # Left: Transcription card
    with col_left:
        st.markdown("#### 📝 Transcribed Text")
        char_count = len(transcription.split())
        st.markdown(f"""
            <div style="
                background: rgba(15, 23, 42, 0.55);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 1.25rem 1.5rem;
                color: #E5E7EB;
                font-size: 1rem;
                line-height: 1.7;
                min-height: 120px;
            ">
                {transcription}
            </div>
            <p style="color:#6B7280; font-size:0.78rem; margin-top:0.5rem;">
                {char_count} words &nbsp;·&nbsp;
                {result['duration_used']:.1f} s processed
            </p>
        """, unsafe_allow_html=True)

    # Right: Emotion result card
    with col_right:
        st.markdown("#### 🧠 Detected Emotion")
        st.markdown(f"""
            <div style="
                background: rgba(15, 23, 42, 0.6);
                border: 2px solid {border_color};
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                box-shadow: 0 0 24px {border_color}33;
                animation: slideInUp 0.4s ease-out;
            ">
                <p style="
                    color: #9CA3AF;
                    text-transform: uppercase;
                    font-size: 0.8rem;
                    letter-spacing: 0.1em;
                    margin-bottom: 0.5rem;
                ">Audio Emotion</p>
                <div class="{style_class}" style="margin-bottom: 1rem;">
                    {emotion_label_with_emoji(emotion)}
                </div>
                <div style="font-size: 2.75rem; font-weight: 700; color: #E5E7EB;">
                    {conf:.1%}
                </div>
                <p style="color: #9CA3AF; margin: 0;">Confidence Score</p>
            </div>
        """, unsafe_allow_html=True)

    # ── Probability Bar Chart ──────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("#### 📊 Probability Distribution")

    import pandas as pd
    import plotly.express as px
    from utils.emotion_config import EMOTION_MAP

    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    df_probs = pd.DataFrame({
        "Emotion":     [emotion_label_with_emoji(e) for e in sorted_probs],
        "Probability": list(sorted_probs.values()),
    })
    color_map = {
        emotion_label_with_emoji(name): emotion_color(name)
        for name in EMOTION_MAP.keys()
    }
    fig = px.bar(
        df_probs, x="Probability", y="Emotion",
        orientation="h", text_auto=".1%",
        title="Model Certainty by Emotion Class",
    )
    fig.update_traces(
        marker_color=[color_map.get(e, "#888") for e in df_probs["Emotion"]],
        textfont_size=12,
        textfont_color="white",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="Inter"),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        height=360,
        margin=dict(l=0, r=0, t=35, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Re-analyse prompt ──────────────────────────────────────────────────
    st.markdown("""
        <p style="color:#6B7280; font-size:0.82rem; text-align:center; margin-top:0.5rem;">
            Upload a different audio file above and click Analyse Audio again to re-run.
        </p>
    """, unsafe_allow_html=True)
