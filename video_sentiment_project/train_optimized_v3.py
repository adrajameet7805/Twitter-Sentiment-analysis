
"""
Refined Emotion Classification Model Training Script (v3)

Features:
- Solver: SAGA (handles large datasets and L1/L2 penalties well)
- Features: 50,000 TF-IDF features (1-2 N-grams)
- Preprocessing: Stronger cleaning (mentions, URLs, special chars)
- Evaluation: Per-class accuracy, Confusion Matrix
- Target: 10 independent emotion classes
"""

import pandas as pd
import numpy as np
import pickle
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust",
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

def preprocess_text(text):
    """
    Stronger preprocessing for v3:
    - Lowercase
    - Remove URLs, Mentions
    - Remove Special Characters (keep only words and basic punctuation)
    - Stemming
    """
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove non-alphabetic characters (allowing spaces) - stricter cleaning
    # Keeping only a-z and spaces. 
    # NOTE: "Keep emojis" from previous requests is NOT explicitly in v3 prompt ("Remove special characters"), 
    # but usually emojis help emotions.
    # The user prompt v3 says: "Remove special characters". 
    # I will strictly follow "Remove special characters" for this specific request to see if it helps separation.
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    cleaned = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(cleaned)

def normalize_label(value):
    if isinstance(value, (int, np.integer)):
        temp = {0: "Sad", 1: "Happy / Joy", 2: "Love", 3: "Angry", 4: "Fear", 5: "Surprise"}
        return temp.get(int(value), None)
    
    label = str(value).strip().lower()
    mapping = {
        "joy": "Happy / Joy", "happy": "Happy / Joy", "sad": "Sad", "sadness": "Sad",
        "anger": "Angry", "angry": "Angry", "fear": "Fear", "surprise": "Surprise",
        "neutral": "Neutral", "love": "Love", "excited": "Excited", "frustration": "Frustrated",
        "frustrated": "Frustrated", "disgust": "Disgust"
    }
    return mapping.get(label, None)

def main():
    start_time = time.time()
    print("Loading emotions.csv...")
    try:
        df = pd.read_csv('emotions.csv')
    except FileNotFoundError:
        print("Error: emotions.csv not found!")
        return

    # Normalize Labels
    df['label'] = df['label'].apply(normalize_label)
    df = df[df['label'].isin(EMOTION_LABELS)]
    
    # Sample for speed if needed, but SAGA is fast enough for ~400k rows usually
    # We will use a large sample to ensure high accuracy
    if len(df) > 150000:
        print("Sampling 150,000 rows for training...")
        df = df.sample(n=150000, random_state=42)
    
    print(f"Dataset Size: {len(df)} rows")
    
    # Preprocess
    print("Preprocessing text...")
    df['processed'] = df['text'].apply(preprocess_text)
    df = df[df['processed'].str.len() > 2] # Remove empty
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'], test_size=0.1, random_state=42, stratify=df['label']
    )
    
    # Vectorizer (v3 Config)
    print("Vectorizing (TF-IDF 1-2 grams, 50k features)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model (v3 Config)
    print("Training Logistic Regression (SAGA solver)...")
    model = LogisticRegression(
        solver='saga',
        max_iter=5000,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        multi_class='ovr' # Encourage independent 10 classes
    )
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    print("\nPer-Class Accuracy:")
    matrix = confusion_matrix(y_test, y_pred, labels=EMOTION_LABELS)
    class_acc = matrix.diagonal() / matrix.sum(axis=1)
    
    for label, accuracy in zip(EMOTION_LABELS, class_acc):
        print(f"  {label:<15}: {accuracy:.2%}")
        
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(matrix)
    
    # Save
    print("\nSaving model files...")
    with open('model_v3.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer_v3.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Total Time: {time.time() - start_time:.1f}s")
    print("Use inference_v3.py for predictions.")

if __name__ == "__main__":
    main()
