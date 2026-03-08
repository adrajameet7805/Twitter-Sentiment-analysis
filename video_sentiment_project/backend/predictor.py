
"""
backend/predictor.py
Business logic layer: prediction routing and analytics metric computation.
All logic is identical to the original app.py — only the location has changed.
"""

import pandas as pd
from utils.emotion_config import EMOTION_ORDER, emotion_label_with_emoji, emotion_color


# ── Inference Wrapper ────────────────────────────────────────────────────────

def predict_emotion_v4(texts, engine):
    """
    Batch inference using the loaded EmotionInferenceV4 engine.
    Returns a list of tuples: (emotion, confidence, probabilities_dict)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Use the new optimized predict_batch method
    if hasattr(engine, 'predict_batch'):
        return engine.predict_batch(texts)
    
    # Fallback to loop if needed
    results = []
    for t in texts:
        results.append(engine.predict(t))
    return results


# ── Analytics Metric Computation ─────────────────────────────────────────────

def get_analytics_metrics(df: pd.DataFrame):
    """
    Calculate all analytics metrics.
    Returns: (emotion_metrics DataFrame, top_samples DataFrame, dominance_score Series)
    """
    emotion_metrics = df.groupby("predicted_emotion").agg(
        {"confidence": ["count", "mean", "max"]}
    )
    emotion_metrics.columns = ["Count", "Avg_Conf", "Max_Conf"]
    emotion_metrics = emotion_metrics.reindex(EMOTION_ORDER, fill_value=0)

    total_len = len(df)
    emotion_metrics["Percentage"] = (
        (emotion_metrics["Count"] / total_len * 100) if total_len > 0 else 0
    )

    top_samples = (
        df.sort_values("confidence", ascending=False)
          .groupby("predicted_emotion")
          .head(5)
    )

    dominance_score = (
        emotion_metrics["Count"] * emotion_metrics["Avg_Conf"]
    ).sort_values(ascending=False)

    return emotion_metrics, top_samples, dominance_score


# ── Batch Post-Processing ─────────────────────────────────────────────────────

def build_results_dataframe(data_to_process, engine, progress_callback=None):
    """
    Run batch inference and return a fully annotated results DataFrame.
    Includes a progress_callback for real-time UI updates.
    """
    CHUNK_SIZE = 100 # Increased chunk size for better tensor manipulation utilization
    total_items = len(data_to_process)
    results = []

    for i in range(0, total_items, CHUNK_SIZE):
        chunk = data_to_process[i : i + CHUNK_SIZE]
        batch_predictions = predict_emotion_v4(chunk, engine)
        
        for text, (emotion, conf, probs) in zip(chunk, batch_predictions):
            results.append({
                'text': text,
                'predicted_emotion': emotion,
                'confidence': conf,
            })
        
        if progress_callback:
            progress_callback(min(1.0, (i + len(chunk)) / total_items))

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['Emotion_Label']   = results_df['predicted_emotion']
        results_df['Emotion_Display'] = results_df.apply(
            lambda r: f"{emotion_label_with_emoji(r['predicted_emotion'])} ({r['confidence']:.1%})",
            axis=1,
        )
        results_df['Confidence'] = results_df['confidence']
        results_df['Text']       = results_df['text']
    
    return results_df
