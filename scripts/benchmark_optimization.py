import pickle
import pandas as pd
import numpy as np
import time
from app import predict_emotion, preprocess_text, stemmer
import json
import sys
from unittest.mock import MagicMock

# Mock streamlit
sys.modules['streamlit'] = MagicMock()

print("Loading optimized model...")
start_load = time.time()
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    emotion_map = pickle.load(open('emotion_mapping.pkl', 'rb'))
    print(f"Model loaded in {time.time() - start_load:.2f}s")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    sys.exit(1)

# 1. Target Validation (Neutral/Surprise > 90%)
print("\n" + "="*50)
print("TARGET VALIDATION (Neutral/Surprise > 90%)")
print("="*50)

targets = [
    ("Wow! This is unbelievable!", "Surprise"),
    ("What? I didn't expect that.", "Surprise"),
    ("The meeting is at 2 PM.", "Neutral"),
    ("Let's schedule a call for tomorrow.", "Neutral"),
    ("I am so happy today!", "Happy / Joy"), # Regression check
]

results = predict_emotion([t[0] for t in targets], model, vectorizer, emotion_map)

all_passed = True
for (text, expected), (pred_emotion, conf, _) in zip(targets, results):
    # Check match
    match = False
    if expected in pred_emotion:
        match = True
    
    # Check confidence constraint for Neutral/Surprise
    conf_ok = True
    if expected in ["Neutral", "Surprise"]:
        if conf < 0.90:
            conf_ok = False
            
    status = "PASS" if match and conf_ok else "FAIL"
    print(f"{status} '{text}'")
    print(f"   -> Pred: {pred_emotion} ({conf:.1%})")
    
    if not match:
        print(f"   Warning: Wrong Emotion (Exp: {expected})")
        all_passed = False
    if not conf_ok:
        print(f"   Warning: Confidence < 90% for {expected}")
        all_passed = False

# 2. Performance Benchmark
print("\n" + "="*50)
print("PERFORMANCE BENCHMARK (Batch < 5s)")
print("="*50)

# Generate dummy dataset of 5000 rows
N_ROWS = 5000
print(f"Generating {N_ROWS} dummy records...")
dummy_texts = [
    "I am quite happy today", 
    "This is terrible and sad", 
    "I'm furious about this", 
    "What a surprise!", 
    "Just a normal day"
] * (N_ROWS // 5)

print(f"Benchmarking batch inference on {len(dummy_texts)} items...")
start_time = time.time()

# Run prediction
batch_results = predict_emotion(dummy_texts, model, vectorizer, emotion_map)

end_time = time.time()
duration = end_time - start_time
speed = len(dummy_texts) / duration

print(f"\nTime taken: {duration:.4f} seconds")
print(f"Speed: {speed:.0f} texts/second")

if duration < 5.0:
    print("\nPASS: Inference time < 5.0 seconds")
else:
    print("\nFAIL: Inference time > 5.0 seconds")
    
if all_passed and duration < 5.0:
    print("\nALL GOALS MET!")
else:
    print("\nSOME GOALS MISSED.")
