
"""
Refined Inference Script (v3)
Includes Rule-Based Reinforcement for Frustrated, Disgust, Excited.
"""
import pickle
import re
import numpy as np
import time
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Reinforcement Rules
RULE_KEYWORDS = {
    "Frustrated": ["frustrated", "fed up", "mentally exhausted", "aggravated", "annoyed", "irritated", "stuck"],
    "Disgust": ["disgusting", "gross", "nauseating", "revolting", "vile", "yuck", "sickening"],
    "Excited": ["thrilled", "can't wait", "so excited", "cant wait", "looking forward", "hyped", "pumped"]
}

def preprocess_text(text):
    # Must match train_optimized_v3.py exactly
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    cleaned = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(cleaned)

class EmotionPredictorV3:
    def __init__(self):
        print("Loading v3 models...")
        with open('model_v3.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('vectorizer_v3.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Consistent class mapping
        self.classes = self.model.classes_

    def predict(self, text):
        start = time.time()
        
        # 1. Preprocess
        processed = preprocess_text(text)
        
        # 2. Vectorize
        vec = self.vectorizer.transform([processed])
        
        # 3. Model Prediction (Probabilities)
        probs = self.model.predict_proba(vec)[0]
        
        # 4. Rule-Based Reinforcement
        # If keywords exist, boost the probability of that class significantly
        text_lower = text.lower()
        
        # Map class names to indices
        class_indices = {name: i for i, name in enumerate(self.classes)}
        
        boost_applied = False
        boost_msg = ""
        
        for emotion, keywords in RULE_KEYWORDS.items():
            if emotion in class_indices:
                idx = class_indices[emotion]
                # Check keywords
                for kw in keywords:
                    if kw in text_lower:
                        # Apply Boost: e.g., add 0.3 to probability or set min threshold
                        # Simple logic: If keyword present, ensure it's at least a strong candidate
                        # We multiply existing prob by 3.0 or add 0.4, then re-normalize?
                        # Simpler: Add 0.5 to the score directly to likely force a win, 
                        # essentially "Prioritize" as requested.
                        probs[idx] += 0.5
                        boost_applied = True
                        boost_msg = f"(Boosted {emotion} via '{kw}')"
                        break 
        
        # 5. Final Decision
        final_idx = np.argmax(probs)
        final_label = self.classes[final_idx]
        
        # Prob may be > 1.0 due to boost, so just take max. 
        # For confidence display, we can clip or re-normalize, but raw is fine for argmax.
        # Let's normalize for clean output
        if boost_applied:
            probs = probs / probs.sum()
            
        final_conf = probs[final_idx]
        
        return {
            "label": final_label,
            "confidence": float(final_conf),
            "time_ms": (time.time() - start) * 1000,
            "note": boost_msg
        }

if __name__ == "__main__":
    predictor = EmotionPredictorV3()
    tests = [
        "I am so fed up with this situation!",
        "This food is disgusting.",
        "I can't wait for the concert!",
        "I am very angry at you."
    ]
    
    print("-" * 50)
    for t in tests:
        res = predictor.predict(t)
        print(f"Input: {t}")
        print(f"Pred : {res['label']} ({res['confidence']:.2%}) {res['note']}")
        print("-" * 50)
