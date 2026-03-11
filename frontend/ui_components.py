
"""
frontend/ui_components.py
Presentation layer: all st.markdown / plotly rendering functions.
Logic is identical to the original app.py — only location has changed.
"""

import time
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.emotion_config import (
    EMOTION_ORDER, EMOTION_MAP,
    emotion_label_with_emoji, emotion_color, emotion_style_class,
)
from backend.predictor import predict_emotion_v4, get_analytics_metrics


# ── CSS Loader ────────────────────────────────────────────────────────────────

def load_css():
    """Load styles.css from the project root."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(root, "styles.css")
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # use default if not found


# ── Processing Animations ─────────────────────────────────────────────────────

def show_ai_processing(placeholder, duration=4.0):
    """Display futuristic AI processing animation with progressive messages."""
    messages = [
        "🔍 Reading text data...",
        "🧠 analyzing emotional patterns...",
        "💭 Detecting semantic nuances...",
        "⚖️ Evaluating confidence...",
        "✨ Identifying dominant emotion...",
    ]
    time_per_message = duration / len(messages)
    for i, message in enumerate(messages):
        placeholder.markdown(f"""
            <div class='ai-processing-overlay'>
                <div class='ai-core'>
                    <div class='ai-core-center'></div>
                    <div class='ai-core-ring'></div>
                    <div class='ai-core-ring'></div>
                </div>
                <div class='ai-message'>{message}</div>
            </div>
        """, unsafe_allow_html=True)
        if i < len(messages) - 1:
            time.sleep(time_per_message)
    time.sleep(0.3)


def show_ai_batch_processing(placeholder, total_items, duration=4.5):
    """Display futuristic AI processing animation for batch analysis."""
    messages = [
        "📂 Loading batch data...",
        "🧠 Running emotion inference engine...",
        "📊 Aggregating emotional metrics...",
        "✨ Generating insights...",
    ]
    time_per_message = duration / len(messages)
    for i, message in enumerate(messages):
        placeholder.markdown(f"""
            <div class='ai-processing-overlay'>
                <div class='ai-core'>
                    <div class='ai-core-center'></div>
                    <div class='ai-core-ring'></div>
                    <div class='ai-core-ring'></div>
                </div>
                <div class='ai-message'>{message}</div>
                <div class='ai-message-sub'>Processing {total_items} items...</div>
            </div>
        """, unsafe_allow_html=True)
        if i < len(messages) - 1:
            time.sleep(time_per_message)
    time.sleep(0.3)


# ── Analytics Dashboard ───────────────────────────────────────────────────────

def render_analytics_dashboard(df: pd.DataFrame):
    """
    Render professional 10-class Sentiment Analytics Dashboard.
    Identical to the original render_analytics_dashboard() in app.py.
    """
    emotion_colors = {
        emotion_label_with_emoji(name): emotion_color(name)
        for name in EMOTION_MAP.keys()
    }

    metrics, top_samples, dominance = get_analytics_metrics(df)

    # --- 1. Summary Cards ---
    st.markdown("### 📊 Sentiment Summary")
    cols = st.columns(4)
    total_records = len(df)

    with cols[0]:
        st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">{total_records:,}</div>
                <div class="stat-label">Total Records</div>
            </div>
        """, unsafe_allow_html=True)

    best_emotion = metrics['Avg_Conf'].idxmax()
    best_conf = metrics['Avg_Conf'].max()
    with cols[1]:
        st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number" style="color: {emotion_colors.get(emotion_label_with_emoji(best_emotion), '#fff')}">{best_conf:.1%}</div>
                <div class="stat-label">Highest Avg Conf ({emotion_label_with_emoji(best_emotion)})</div>
            </div>
        """, unsafe_allow_html=True)

    dom_emotion = metrics['Count'].idxmax()
    dom_count = metrics['Count'].max()
    with cols[2]:
        st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">{dom_count}</div>
                <div class="stat-label">Most Frequent ({emotion_label_with_emoji(dom_emotion)})</div>
            </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        score = dominance.max()
        st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">{score:.0f}</div>
                <div class="stat-label">Impact Score</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- 2. Sentiment Distribution ---
    st.markdown("#### 📊 Sentiment Distribution")

    if not df.empty:
        # Distribution Bar Chart
        counts = df['predicted_emotion'].value_counts().reindex(EMOTION_ORDER, fill_value=0)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pie = px.pie(
                values=counts.values, 
                names=[emotion_label_with_emoji(e) for e in counts.index],
                title="Overall Emotion Distribution",
                hole=0.4,
                color=[emotion_label_with_emoji(e) for e in counts.index],
                color_discrete_map=emotion_colors
            )
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={"color": '#E5E7EB'})
            st.plotly_chart(fig_pie, width="stretch")

        with c2:
            fig_bar = px.bar(
                x=[emotion_label_with_emoji(e) for e in counts.index],
                y=counts.values,
                title="Emotion Frequency",
                labels={"x": "Emotion", "y": "Count"},
                color=[emotion_label_with_emoji(e) for e in counts.index],
                color_discrete_map=emotion_colors
            )
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={"color": '#E5E7EB'})
            st.plotly_chart(fig_bar, width="stretch")

        # --- 3. Top Emotion Samples ---
        st.markdown("---")
        st.markdown("## 🔥 Top Emotion Samples")

        for i in range(0, len(EMOTION_ORDER), 2):
            cols = st.columns(2)
            for idx in range(2):
                if i + idx < len(EMOTION_ORDER):
                    emotion = EMOTION_ORDER[i + idx]
                    emoji  = EMOTION_MAP[emotion]["emoji"]
                    color  = emotion_color(emotion)
                    
                    with cols[idx]:
                        st.markdown(f"### {emoji} {emotion}")
                        top = df[df["predicted_emotion"] == emotion].sort_values("confidence", ascending=False).head(3)
                        
                        if top.empty:
                            st.info(f"No samples for {emotion}")
                        else:
                            for _, row in top.iterrows():
                                conf_val = f"({row['confidence']:.1%})"
                                st.markdown(f"""
                                    <div style="
                                        background: rgba(15, 23, 42, 0.4);
                                        padding: 12px;
                                        border-radius: 10px;
                                        margin-bottom: 10px;
                                        border-left: 4px solid {color};
                                        font-size: 14px;
                                        color: #E5E7EB;
                                    ">
                                        <span style="color: {color}; font-weight: bold;">{conf_val}</span> {row['text']}
                                    </div>
                                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data available to display.")

    # --- 4. Advanced Charts ---
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### 🕸️ Emotional Fingerprint (Radar)")
        radar_df = pd.DataFrame({
            'Emotion': [emotion_label_with_emoji(e) for e in metrics.index],
            'Value':   metrics['Percentage'],
        })
        fig_radar = px.line_polar(
            radar_df, r='Value', theta='Emotion',
            line_close=True, color_discrete_sequence=['#00FF88']
        )
        fig_radar.update_traces(fill='toself')
        fig_radar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font={"color": '#E5E7EB'},
            polar={"bgcolor": 'rgba(0,0,0,0)', "radialaxis": {"showticklabels": False, "ticks": ''}},
            margin={"t": 20, "b": 40, "l": 40, "r": 40},
        )
        st.plotly_chart(fig_radar, width="stretch")

    with c4:
        st.markdown("#### 📉 Confidence Histogram")
        _df = df.copy()
        _df['Emotion_Label'] = _df['predicted_emotion'].apply(emotion_label_with_emoji)
        fig_hist = px.histogram(
            _df, x="confidence", nbins=30,
            color="Emotion_Label", color_discrete_map=emotion_colors, marginal="box"
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font={"color": '#E5E7EB'}, barmode='overlay',
            margin={"t": 20, "b": 20, "l": 20, "r": 20},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig_hist.update_traces(opacity=0.75)
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("---")


# ── Single Analysis Page ──────────────────────────────────────────────────────

def render_single_analysis(engine):
    """
    Render professional Single Emotion Analysis page with input selection.
    Identical to the original render_single_analysis() in app.py.
    """
    st.markdown("<h2 class='section-header'>Single Sentiment Analysis</h2>", unsafe_allow_html=True)

    if not engine:
        st.error("Model not loaded.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📥 Input Source")
        input_type = st.radio(
            "Select Input Type:",
            ["Type Manual", "Select Sample"],
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("### 📝 Enter Text")
        input_text = ""

        if input_type == "Type Manual":
            input_text = st.text_area(
                "Input Text", height=150,
                placeholder="Enter your text here...",
                label_visibility="collapsed",
                key="manual_input"
            )
        else:
            samples = [
                "I absolutely love this product! Best purchase ever.",
                "This is the worst experience of my life.",
                "I feel so lonely and sad.",
                "Wow, I didn't expect that!",
                "The meeting is at 2 PM.",
                "I am terrified of the dark.",
                "This makes me so angry!",
                "I am extremely grateful for this opportunity.",
            ]
            input_text = st.selectbox(
                "Select a sample:", samples,
                label_visibility="collapsed", key="sample_input"
            )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Emotion", type="primary", width="stretch"):
        if input_text:
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                with st.spinner("Analyzing..."):
                    results = predict_emotion_v4([input_text], engine)
                    emotion, conf, probs = results[0]

                    style_class  = emotion_style_class(emotion)
                    border_color = emotion_color(emotion)

                    st.markdown(f"""
                        <div style="
                            background: rgba(15, 23, 42, 0.6);
                            border: 2px solid {border_color};
                            border-radius: 16px;
                            padding: 2rem;
                            text-align: center;
                            box-shadow: 0 0 20px {border_color}33;
                            animation: slideInUp 0.5s ease-out;
                        ">
                            <p style="color: #9CA3AF; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Detected Emotion</p>
                            <div class="{style_class}" style="margin-bottom: 1rem;">{emotion_label_with_emoji(emotion)}</div>
                            <div style="font-size: 2.5rem; font-weight: 700; color: #E5E7EB;">{conf:.1%}</div>
                            <p style="color: #9CA3AF; margin: 0;">Confidence Score</p>
                        </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("### 📊 Probability Distribution")
                sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
                df_probs = pd.DataFrame({
                    'Emotion':     [emotion_label_with_emoji(e) for e in sorted_probs],
                    'Probability': list(sorted_probs.values()),
                })
                color_map = {
                    emotion_label_with_emoji(name): emotion_color(name)
                    for name in EMOTION_MAP.keys()
                }
                fig = px.bar(
                    df_probs, x='Probability', y='Emotion',
                    orientation='h', text_auto='.1%',
                    title="Model Certainty by Class"
                )
                fig.update_traces(
                    marker_color=[color_map.get(e, '#888') for e in df_probs['Emotion']],
                    textfont_size=12, textfont_color='white'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font={"color": '#E5E7EB', "family": 'Inter'},
                    yaxis={"autorange": "reversed"},
                    xaxis={"showgrid": True, "gridcolor": 'rgba(255,255,255,0.1)'},
                    height=350, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning("⚠️ Please provide input text.")
