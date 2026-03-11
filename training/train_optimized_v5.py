import os
import sys
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import preprocess_texts

DATASET_PATH = '../data/emotions.csv'
MODEL_FILE = '../models/model_v5_light.pkl'
VEC_FILE = '../models/vectorizer_v5_light.pkl'
MAP_FILE = '../models/emotion_mapping_v5.pkl'

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust",
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

# Base Keywords for refinement
REFINE_KEYWORDS = {
    "Disgust": ["disgust", "disgusting", "gross", "yuck", "revolting", "nasty", "vile", "repulsive"],
    "Excited": ["excited", "thrilled", "cant wait", "can't wait", "pumped", "hyped", "ecstatic"],
    "Frustrated": ["frustrated", "frustrating", "annoyed", "irritated", "stuck", "fed up", "aggravated"],
    "Love": ["love", "adore", "cherish", "affection", "romance", "romantic", "sweetheart"],
    "Neutral": ["meeting", "schedule", "time", "date", "call", "agenda", "calendar"]
}

def normalize_label(value):
    if isinstance(value, (int, np.integer)):
        temp_map = {0: "Sad", 1: "Happy / Joy", 2: "Love", 3: "Angry", 4: "Fear", 5: "Surprise"}
        return temp_map.get(int(value), "Neutral")
    return "Neutral"

def refine_label(label, text):
    text_lower = str(text).lower()
    for k in REFINE_KEYWORDS["Disgust"]:
        if k in text_lower: return "Disgust"
    for k in REFINE_KEYWORDS["Excited"]:
        if k in text_lower: return "Excited"
    for k in REFINE_KEYWORDS["Frustrated"]:
        if k in text_lower: return "Frustrated"
    for k in REFINE_KEYWORDS["Love"]:
        if k in text_lower: return "Love"
    for k in REFINE_KEYWORDS["Neutral"]:
        if k in text_lower: return "Neutral"
    return label

def main():
    start_time = time.time()
    print("🚀 Running Optimized V5 Training Pipeline (LinearSVC + spaCy + n-grams)")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH)
    
    if len(df) > 250000:
        df = df.sample(n=250000, random_state=42)
    
    print(f"Dataset size for training: {len(df)} rows")
    
    df['label'] = df['label'].apply(normalize_label)
    df['final_label'] = df.apply(lambda r: refine_label(r['label'], r['text']), axis=1)
    df = df[df['final_label'].isin(EMOTION_LABELS)]
    
    print("Class Distribution:")
    print(df['final_label'].value_counts())
    
    # 2. Preprocess
    print("\nPreprocessing text using spaCy (batch optimized)...")
    prep_start = time.time()
    # Fill NAs
    texts = df['text'].fillna("").tolist()
    processed_texts = preprocess_texts(texts)
    df['processed'] = processed_texts
    print(f"Preprocessing completed in {time.time() - prep_start:.1f}s")
    
    # Drop empty strings after processing
    df = df[df['processed'].str.len() > 2]
    
    # 3. Vectorize
    print("\nVectorizing (TF-IDF 1-2 grams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # Bigrams capture context ("not happy")
        max_features=120000,       # Higher vocabulary
        min_df=2,
        sublinear_tf=True,         # Log scaling
        strip_accents='unicode'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['final_label'], test_size=0.1, random_state=42, stratify=df['final_label']
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # 4. Model Training
    print("\nTraining LinearSVC...")
    base_clf = LinearSVC(
        class_weight='balanced',
        random_state=42,
        max_iter=5000,
        dual=False,
        C=0.35 # Optimized for 250k samples
    )
    
    model = CalibratedClassifierCV(base_clf, cv=3)
    
    train_start = time.time()
    model.fit(X_train_vec, y_train)
    print(f"Training completed in {time.time() - train_start:.1f}s")
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save Models
    print("\nSaving model files for deployment...")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f: pickle.dump(model, f)
    with open(VEC_FILE, 'wb') as f: pickle.dump(vectorizer, f)
    with open(MAP_FILE, 'wb') as f: pickle.dump({l: l for l in EMOTION_LABELS}, f)
    
    print(f"\n✅ V5 Training completely finished in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
