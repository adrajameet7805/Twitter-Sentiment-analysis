
"""
Optimized Inference for Classical ML (v2)
Must match preprocessing of train_optimized_v2.py
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

BOOST_KEYWORDS = {
    "Disgust": ["gross", "yuck", "disgusting", "revolting", "nasty", "vile", "repulsive", "sickening"],
    "Excited": ["excited", "thrilled", "cant wait", "can't wait", "pumped", "hyped", "energetic", "looking forward"],
    "Frustrated": ["frustrated", "frustrating", "annoyed", "irritated", "stuck", "fed up", "aggravated"]
}

def preprocess_text(text):
    text = str(text).lower()
    for emotion, keywords in BOOST_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                text += f" {kw} {kw}"
    
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s!?\U00010000-\U0010ffff]', '', text) 
    
    tokens = text.split()
    cleaned = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(cleaned)

class OptimizedPredictor:
    def __init__(self):
        print("Loading optimized models...")
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open('emotion_mapping.pkl', 'rb') as f:
            self.mapping = pickle.load(f)

    def predict(self, text):
        start = time.time()
        processed = preprocess_text(text)
        vec = self.vectorizer.transform([processed])
        probs = self.model.predict_proba(vec)[0]
        idx = np.argmax(probs)
        label = self.model.classes_[idx]
        confidence = probs[idx]
        
        return {
            "label": self.mapping.get(label, label),
            "confidence": float(confidence),
            "time_ms": (time.time() - start) * 1000
        }

if __name__ == "__main__":
    predictor = OptimizedPredictor()
    tests = [
        "I am so excited about this!",
        "This is absolutely disgusting.",
        "I am extremely frustrated with this error."
    ]
    for t in tests:
        print(f"'{t}' -> {predictor.predict(t)}")
