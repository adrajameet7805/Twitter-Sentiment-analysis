"""
app.py  –  Twitter Sentiment Analysis
Entry point for: streamlit run app.py

This file is now a slim orchestrator.
All business logic lives in backend/
All UI rendering lives in frontend/
Shared constants live in utils/
The inference engine lives in models/
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import logging
from datetime import datetime
import hashlib
from functools import wraps

def check_dependencies():
    """Check optional video dependencies and warn in sidebar — never block the app."""
    missing = []
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        missing.append("mediapipe")
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")

    if missing:
        import streamlit as _st
        _st.sidebar.warning(
            f"⚠️ Optional video-analysis deps missing: **{', '.join(missing)}**\n\n"
            "Run `pip install mediapipe opencv-python` to enable Face Emotion detection."
        )

check_dependencies()

# from auth.database import init_db
# from frontend.login_page import render_login_page
# from frontend.register_page import render_register_page

# Initialize user database
# init_db()

# ── Structured imports ────────────────────────────────────────────────────────
from models.inference_engine import EmotionInferenceV4
from backend.predictor import predict_emotion_v4, build_results_dataframe
from frontend.ui_components import (
    load_css,
    show_ai_batch_processing,
    render_analytics_dashboard,
    render_single_analysis,
)
from frontend.audio_page import render_audio_analysis
from frontend.video_page import render_video_analysis
from utils.emotion_config import (
    EMOTION_ORDER, EMOTION_MAP,
    emotion_label_with_emoji, emotion_color, emotion_style_class,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.getLogger('streamlit').setLevel(logging.ERROR)


# ── Custom error handler ──────────────────────────────────────────────────────
def safe_render_chart(chart_func, fallback_message="Unable to render visualization. Data is being processed..."):
    """Safely render charts with professional error handling."""
    try:
        return chart_func()
    except Exception as e:
        logging.error(f"Chart rendering error: {str(e)}")
        st.info(f"ℹ️ {fallback_message}")
        return None


# ── CSS ───────────────────────────────────────────────────────────────────────
load_css()

# Additional inline CSS (emotion colours etc.) – unchanged from original
st.markdown("""
    <style>
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 3rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #9CA3AF;
        max-width: 600px;
        margin: 0 auto;
    }
    .section-header {
        font-size: 1.875rem;
        font-weight: 600;
        color: #E5E7EB;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .feature-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .feature-card:hover { border-color: rgba(255,255,255,0.16); }
    .feature-icon { font-size: 2rem; margin-bottom: 0.75rem; }
    /* Emotion label colours */
    .emotion-happy   { color: #22C55E; font-weight: bold; font-size: 2.5rem; text-shadow: 0 0 20px rgba(34,197,94,0.3); }
    .emotion-sad     { color: #3B82F6; font-weight: bold; font-size: 2.5rem; text-shadow: 0 0 20px rgba(59,130,246,0.3); }
    .emotion-angry   { color: #EF4444; font-weight: bold; font-size: 2.5rem; text-shadow: 0 0 20px rgba(239,68,68,0.3); }
    .emotion-fear    { color: #A855F7; font-weight: bold; font-size: 2.5rem; text-shadow: 0 0 20px rgba(168,85,247,0.3); }
    .emotion-surprise{ color: #FACC15; font-weight: bold; font-size: 2.5rem; text-shadow: 0 0 20px rgba(250,204,21,0.3); }
    .emotion-disgust { color: #92400E; font-weight: bold; font-size: 2.5rem; }
    .emotion-neutral { color: #9CA3AF; font-weight: bold; font-size: 2.5rem; }
    /* Stat cards */
    .stat-container {
        background: linear-gradient(135deg, rgba(59,130,246,0.05) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .stat-number { font-size: 2.5rem; font-weight: 700; color: #E5E7EB; margin-bottom: 0.5rem; }
    .stat-label  { font-size: 0.875rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; }
    /* AI Processing */
    .ai-processing-overlay {
        padding: 2rem; text-align: center;
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; margin: 1rem 0;
    }
    .ai-message     { color: #60A5FA; font-size: 1.125rem; font-weight: 500; margin-top: 1rem; }
    .ai-message-sub { color: #9CA3AF; font-size: 0.875rem; margin-top: 0.5rem; }
    /* Animations */
    @keyframes slideInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn    { from { opacity: 0; } to   { opacity: 1; } }
    .animate-fadeIn   { animation: fadeIn 300ms ease-out; }
    .animate-slideInUp{ animation: slideInUp 400ms ease-out; }
    .delay-100 { animation-delay: 100ms; }
    .delay-200 { animation-delay: 200ms; }
    .delay-300 { animation-delay: 300ms; }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_engine():
    """Load the advanced V4 inference engine (cached across reruns)."""
    try:
        return EmotionInferenceV4()
    except Exception as e:
        st.error(f"⚠️ Error loading inference engine: {str(e)}")
        return None


engine = load_emotion_engine()

def hash_password(password):
    """Securely hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

# Note: In a real app, these would be in environment variables
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hash_password("admin123") # SHA256 of 'admin123'

def render_login_page():
    """Renders a clean, glassmorphism-styled login page."""
    st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 class="hero-title" style="font-size: 2.5rem;">🔒 Admin Access</h1>
            <p class="hero-subtitle">Sentiment Analysis Enterprise Edition</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        with st.container():
            st.markdown("""
                <div style="
                    background: rgba(15, 23, 42, 0.6);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 16px;
                    padding: 2rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                ">
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                u = st.text_input("Username", placeholder="admin")
                p = st.text_input("Password", type="password", placeholder="••••••••")
                submit = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit:
                    if u == ADMIN_USERNAME and hash_password(p) == ADMIN_PASSWORD_HASH:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = "Admin"
                        st.session_state["role"] = "admin"
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
            
            st.markdown("</div>", unsafe_allow_html=True)

# ── Authentication Handling ────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    render_login_page()
    st.stop()



# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
    <div class="hero-section animate-fadeIn">
        <div class="hero-title">🧠 Twitter Sentiment Analysis</div>
        <div class="hero-subtitle">Advanced AI that detects 10 distinct emotional states in text</div>
    </div>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Twitter Sentiment Analysis")
    st.markdown("_v2.0 Enterprise Edition_")
    st.markdown("---")
    st.markdown("**Navigation**")

    page = st.radio(
        "nav",
        [
            "Home",
            "Single Sentiment Analysis",
            "Batch Sentiment Analysis",
            "Sentiment Dashboard",
            "Audio Sentiment Analysis",
            "Video Sentiment Analysis",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin: 2rem 0; height: 1px; background: rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)

    model_status = "✅ Online" if engine is not None else "❌ Offline"
    status_color = "#22C55E" if engine is not None else "#EF4444"

    st.markdown(f"""
        <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <p style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase; margin-bottom:0.5rem;">System Status</p>
            <p style="font-size: 1.125rem; font-weight: 600; color: {status_color}; margin: 0;">{model_status}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background: rgba(15,23,42,0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem;">
            <p style="font-size: 0.75rem; color: #9CA3AF; margin: 0 0 0.75rem 0; text-transform: uppercase;">Model Specs</p>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #6B7280; font-size: 0.875rem;">Classes:</span>
                <span style="color: #E5E7EB; font-size: 0.875rem;">10 Emotions</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #6B7280; font-size: 0.875rem;">Accuracy:</span>
                <span style="color: #E5E7EB; font-size: 0.875rem;">>95% (Hybrid)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"👤 **{st.session_state.get('user', 'Guest')}** ({st.session_state.get('role', 'user')})")
    
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("<h2 class='section-header'>Welcome to Sentiment Intelligence</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="stat-container animate-slideInUp delay-100">
                <div class="stat-number">10</div>
                <div class="stat-label">Emotion Classes</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="stat-container animate-slideInUp delay-200">
                <div class="stat-number">416K</div>
                <div class="stat-label">Training Samples</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="stat-container animate-slideInUp delay-300">
                <div class="stat-number">>95%</div>
                <div class="stat-label">Target Accuracy</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="feature-card animate-slideInUp delay-100">
                <div class="feature-icon">🎯</div>
                <h3 style="color: #E5E7EB; margin-bottom: 0.75rem;">Sentiment Detection</h3>
                <p style="color: #9CA3AF; margin-bottom: 0;">
                    Goes beyond simple positive/negative sentiment to detect nuances:
                    <br>• <span style="color:#22C55E">Happy/Joy</span> • <span style="color:#3B82F6">Sad</span> • <span style="color:#EF4444">Angry</span>
                    <br>• <span style="color:#A855F7">Fear</span> • <span style="color:#FACC15">Surprise</span> • <span style="color:#94A3B8">Neutral</span>
                </p>
            </div>
            <div class="feature-card animate-slideInUp delay-300">
                <div class="feature-icon">📈</div>
                <h3 style="color: #E5E7EB; margin-bottom: 0.75rem;">Enterprise Analytics</h3>
                <p style="color: #9CA3AF; margin-bottom: 0;">Batch process thousands of records and visualize emotional trends instantly.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card animate-slideInUp delay-200">
                <div class="feature-icon">🚀</div>
                <h3 style="color: #E5E7EB; margin-bottom: 0.75rem;">Quick Test</h3>
        """, unsafe_allow_html=True)

        if engine:
            quick_test = st.text_input(
                "Try it out:",
                placeholder="e.g. I am so excited about this new feature!",
                key="home_input"
            )
            if st.button("Analyze Now", type="primary", use_container_width=True):
                if quick_test:
                    results = predict_emotion_v4([quick_test], engine)
                    emotion, conf, _ = results[0]
                    style_class = emotion_style_class(emotion)
                    st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
                            <div class="{style_class}">{emotion_label_with_emoji(emotion)}</div>
                            <div style="color: #9CA3AF; margin-top: 0.5rem;">Confidence: {conf:.1%}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter some text.")
        else:
            st.error("Model offline.")

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Single Sentiment Analysis":
    render_single_analysis(engine)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Sentiment Analysis":
    st.markdown("<h2 class='section-header'>Batch Sentiment Analysis</h2>", unsafe_allow_html=True)

    if not engine:
        st.stop()

    input_method = st.radio(
        "Select Input Method:",
        ["Upload CSV", "Paste Text"],
        horizontal=True
    )

    data_to_process = None

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV or TXT file", type=['csv', 'txt'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    cols = [c for c in df.columns if c.lower() in ['text', 'tweet', 'content']]
                    if cols:
                        data_to_process = df[cols[0]].tolist()
                        st.success(f"✅ Loaded {len(data_to_process)} records from column '{cols[0]}'")
                    else:
                        st.error("❌ Could not find a 'text' or 'tweet' column in CSV.")
                else: # .txt file
                    content = uploaded_file.read().decode('utf-8')
                    data_to_process = [line.strip() for line in content.split('\n') if line.strip()]
                    st.success(f"✅ Loaded {len(data_to_process)} records from TXT file")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    elif input_method == "Paste Text":
        text_input = st.text_area(
            "Paste multiple lines (one item per line)", height=250,
            placeholder="Example:\nI feel great today!\nThis is disappointing.\nWhat a surprise!"
        )
        if text_input:
            data_to_process = [x.strip() for x in text_input.split('\n') if x.strip()]
            if data_to_process:
                st.info(f"ℹ️ Ready to process {len(data_to_process)} lines")

    if data_to_process and st.button("🚀 Process Batch", type="primary"):
        start_time = datetime.now()
        
        progress_bar = st.progress(0, text="Initializing batch analysis...")
        
        def update_progress(pct):
            progress_bar.progress(pct, text=f"Analyzing data... {pct:.0%}")

        try:
            with st.spinner("🤖 AI Engine processing batch..."):
                results_df = build_results_dataframe(data_to_process, engine, progress_callback=update_progress)

            progress_bar.empty()
            processing_time = (datetime.now() - start_time).total_seconds()

            st.session_state['batch_results'] = results_df
            st.session_state['results_df']    = results_df

            st.success(
                f"✅ Successfully processed {len(results_df)} items in "
                f"{processing_time:.2f} seconds ({len(results_df)/processing_time:.0f} items/sec)"
            )
        except Exception as e:
            st.error(f"❌ Batch analysis failed. Please try again or check your file format. Error: {e}")

    if 'batch_results' in st.session_state:
        results_df = st.session_state['batch_results']

        st.markdown("### 📊 Batch Results Summary")

        counts = results_df['predicted_emotion'].value_counts()
        total  = len(results_df)
        counts = counts.reindex(EMOTION_ORDER, fill_value=0)

        cols = st.columns(4)
        for i, (emotion, count) in enumerate(counts.items()):
            with cols[i % 4]:
                st.metric(emotion_label_with_emoji(emotion), count, f"{count/total:.1%}")

        c1, c2 = st.columns(2)
        _colors = {emotion_label_with_emoji(name): emotion_color(name) for name in EMOTION_MAP.keys()}

        with c1:
            agg_df = counts.reset_index()
            agg_df.columns = ['predicted_emotion', 'Count']
            agg_df['Label'] = agg_df['predicted_emotion'].apply(emotion_label_with_emoji)
            fig = px.pie(
                agg_df, names='Label', values='Count',
                title='Sentiment Distribution', hole=0.4,
                color='Label', color_discrete_map=_colors
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E5E7EB'))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            avg_conf = results_df.groupby('predicted_emotion')['confidence'].mean()
            avg_conf = avg_conf.reindex(EMOTION_ORDER, fill_value=0).reset_index()
            avg_conf['Emotion_Label'] = avg_conf['predicted_emotion'].apply(emotion_label_with_emoji)
            fig = px.bar(
                avg_conf, x='Emotion_Label', y='confidence',
                title='Average Confidence by Sentiment',
                color='Emotion_Label', color_discrete_map=_colors
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E5E7EB'))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Data Table")
        st.dataframe(results_df[['text', 'Emotion_Display', 'confidence']], use_container_width=True)

        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results (CSV)", csv, "emotion_results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment Dashboard":
    st.markdown("<h2 class='section-header'>Sentiment Dashboard</h2>", unsafe_allow_html=True)

    if 'batch_results' not in st.session_state:
        st.info("ℹ️ Please run a Batch Analysis first to generate data for this dashboard.")
    else:
        df = st.session_state['batch_results']
        render_analytics_dashboard(df)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO SENTIMENT ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Audio Sentiment Analysis":
    render_audio_analysis(engine)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO SENTIMENT ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Video Sentiment Analysis":
    render_video_analysis(engine)
