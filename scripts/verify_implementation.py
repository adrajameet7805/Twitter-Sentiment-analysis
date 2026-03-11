import pickle
import pandas as pd
import numpy as np
from app import predict_emotion, preprocess_text, stemmer
import json

# Mock streamlit for the import
import sys
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

print("Loading model for verification...")
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    emotion_map = pickle.load(open('emotion_mapping.pkl', 'rb'))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Test Cases
test_cases = [
    ("I am so happy and excited about this!", "Happy / Joy"),
    ("I feel so lonely and sad today.", "Sad"),
    ("This makes me so angry and furious!", "Angry"),
    ("I am terrified of the dark.", "Fear"),
    ("Wow, I didn't expect that at all!", "Surprise"),
    ("The meeting is at 2 PM.", "Neutral") # Expecting Neutral/Low Confidence
]

print("\n" + "="*50)
print("VERIFYING PREDICTIONS AND JSON FORMAT")
print("="*50)

passed = 0
for text, expected in test_cases:
    print(f"\nInput: \"{text}\"")
    emotion, confidence, probs = predict_emotion(text, model, vectorizer, emotion_map)
    
    # Verify JSON structure compliance
    result_json = {
        "emotion": emotion,
        "confidence": round(confidence * 100, 1)
    }
    
    json_output = json.dumps(result_json)
    print(f"Output: {json_output}")
    
    # Check if emotion matches expected (loosely for Neutral which depends on threshold)
    match = False
    if expected == "Neutral":
        if emotion == "Neutral" or confidence < 0.6:
            match = True
    elif expected in emotion:
        match = True
        
    if match:
        print(f"✅ Prediction Verified ({emotion})")
        passed += 1
    else:
        print(f"❌ Prediction Mismatch (Expected {expected}, Got {emotion})")

print("\n" + "="*50)
print(f"Verification Complete: {passed}/{len(test_cases)} Passed")
print("="*50)
