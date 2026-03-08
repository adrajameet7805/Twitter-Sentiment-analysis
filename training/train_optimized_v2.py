
"""
Optimized Emotion Classification Model Training Script (v2 - SMOTE + Boosting)

Features:
- SMOTE Oversampling for class balance
- Keyword Boosting for weak classes (Disgust, Excited, Frustrated)
- Emoji preservation
- 1-3 gram TF-IDF
- GridSearchCV for C parameter
"""

import pandas as pd
import numpy as np
import pickle
import re
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Config
TARGET_ACCURACY = 0.90
RANDOM_STATE = 42
SAMPLE_SIZE = 100000  # Large sample for better results

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust", 
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

# Keywords for boosting (Weighted x3 during preprocessing)
BOOST_KEYWORDS = {
    "Disgust": ["gross", "yuck", "disgusting", "revolting", "nasty", "vile", "repulsive", "sickening"],
    "Excited": ["excited", "thrilled", "cant wait", "can't wait", "pumped", "hyped", "energetic", "looking forward"],
    "Frustrated": ["frustrated", "frustrating", "annoyed", "irritated", "stuck", "fed up", "aggravated"]
}

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Enhanced preprocessing:
    - Keeps emojis (Ranges for emoticons)
    - Boosts keywords by repeating them
    - 1-3 grams captured by Vectorizer later
    """
    text = str(text).lower()
    
    # Keyword Boosting: Repeat specific strong words to increase TF-IDF weight
    for emotion, keywords in BOOST_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                # Add the keyword 2 more times (total 3x weight approx)
                text += f" {kw} {kw}"
    
    # Remove URLs and Mentions
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    
    # Keep A-Z, a-z, spaces, and Emoji ranges
    # Basic Emoji range: \U0001F600-\U0001F64F
    # We will remove anything that is NOT a letter, space, or emoji-like character
    # For simplicity in regex, we can strip mostly everything but keep some punctuation relevant to emotion (! ?)
    # OR we can just remove specific "junk" and keep the rest.
    
    # Strategy: Remove non-alphanumeric but KEEP specific punctuation and emojis
    # Instead of complex regex for all emojis, let's strictly remove what we KNOW is bad (special chars)
    # but keep standard emojis.
    
    # Clean non-standard chars but keep standard punctuation for improved context if we used BERT, 
    # but for TF-IDF usually punctuation is stripped. 
    # However, user specifically asked to Keep Emojis.
    
    # Replace standard punctuation with space (except ! ?)
    text = re.sub(r'[^\w\s!?\U00010000-\U0010ffff]', '', text) 
    
    tokens = text.split()
    # Stemming & Stopwords
    cleaned = [stemmer.stem(t) for t in tokens if t not in stop_words]
    
    return ' '.join(cleaned)

def normalize_label(value):
    # Same normalization logic as before
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

def refine_label(label, text):
    text = str(text).lower()
    if label == "Happy / Joy":
        if any(k in text for k in ["love", "romance", "adore"]): return "Love"
        if any(k in text for k in BOOST_KEYWORDS["Excited"]): return "Excited"
    if label == "Angry":
        if any(k in text for k in BOOST_KEYWORDS["Frustrated"]): return "Frustrated"
        if any(k in text for k in BOOST_KEYWORDS["Disgust"]): return "Disgust"
    return label

def main():
    start_time = time.time()
    
    print("Loading dataset...")
    try:
        df = pd.read_csv('emotions.csv')
    except FileNotFoundError:
        print("Error: emotions.csv not found")
        return

    # Normalize
    df['label'] = df['label'].apply(normalize_label)
    df = df[df['label'].notna()]
    df['label'] = df.apply(lambda r: refine_label(r['label'], r['text']), axis=1)
    df = df[df['label'].isin(EMOTION_LABELS)]
    
    # Sample if too large (SMOTE is expensive on CPU)
    if len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} rows...")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    print(f"Data Loaded: {len(df)} rows")
    print("Pre-SMOTE Distribution:")
    print(df['label'].value_counts())

    print("Preprocessing text...")
    df['processed'] = df['text'].apply(preprocess_text)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['label']
    )
    
    # Pipeline Construction
    # ImbPipeline allows SMOTE to be part of the CV fold (prevents data leakage)
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3), 
            min_df=3, 
            max_df=0.9, 
            sublinear_tf=True
        )),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', LogisticRegression(
            solver='lbfgs', 
            max_iter=2000, 
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Grid Search
    param_grid = {
        'clf__C': [1.0, 5.0, 10.0, 20.0]  # Higher C = Less regularization (better for fitting complex boundaries for weak classes)
    }
    
    print("Starting Grid Search with SMOTE (this looks slow but works hard)...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"Best Params: {grid.best_params_}")
    best_model = grid.best_estimator_
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = best_model.predict(X_test)
    
    print("Optimization Results:")
    print("-" * 30)
    print(classification_report(y_test, y_pred))
    
    # Check specific improvements
    report = classification_report(y_test, y_pred, output_dict=True)
    for class_name in ["Disgust", "Excited", "Frustrated"]:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name} F1-Score: {metrics['f1-score']:.2f}")
    
    # Save
    print("\nSaving optimized models...")
    with open('model.pkl', 'wb') as f:
        # Save the CLASSIFIER step specifically if needed, or the whole pipeline
        # Usually app.py expects a model with a .predict() method
        # If app.py expects separate vectorizer, we might need to split it
        # CURRENT app.py loads 'model.pkl' and 'vectorizer.pkl'. 
        # So we must split the pipeline components.
        pickle.dump(best_model.named_steps['clf'], f)
        
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(best_model.named_steps['tfidf'], f)
        
    with open('emotion_mapping.pkl', 'wb') as f:
        emotion_map = {label: label for label in EMOTION_LABELS}
        pickle.dump(emotion_map, f)
        
    print(f"Completed in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
