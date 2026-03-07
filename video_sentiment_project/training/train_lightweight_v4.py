
import pandas as pd
import numpy as np
import pickle
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Config
DATASET_PATH = 'emotions.csv'
MODEL_FILE = 'model_v4_light.pkl'
VEC_FILE = 'vectorizer_v4_light.pkl'
MAP_FILE = 'emotion_mapping_v4.pkl'

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust",
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

# Keywords for refinement (consistent with V4 strategy)
REFINE_KEYWORDS = {
    "Disgust": ["disgust", "disgusting", "gross", "yuck", "revolting", "nasty", "vile", "repulsive", "sickening", "nauseating"],
    "Excited": ["excited", "thrilled", "cant wait", "can't wait", "pumped", "hyped", "ecstatic", "thrilling", "enthusiastic"],
    "Frustrated": ["frustrated", "frustrating", "annoyed", "irritated", "stuck", "fed up", "aggravated", "mentally exhausted"],
    "Love": ["love", "adore", "cherish", "affection", "romance", "romantic", "sweetheart"],
    "Neutral": ["meeting", "schedule", "time", "date", "call", "agenda", "calendar"]
}

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    # Remove URLs and Mentions
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    # Keep only letters and spaces (User requirements #1: Basic cleaning)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(cleaned)

def normalize_label(value):
    if isinstance(value, (int, np.integer)):
        temp_map = {0: "Sad", 1: "Happy / Joy", 2: "Love", 3: "Angry", 4: "Fear", 5: "Surprise"}
        return temp_map.get(int(value), "Neutral")
    return "Neutral"

def refine_label(label, text):
    text_lower = str(text).lower()
    # 1. Priority Keywords for rare/important classes
    if any(k in text_lower for k in REFINE_KEYWORDS["Disgust"]):
        return "Disgust"
    if any(k in text_lower for k in REFINE_KEYWORDS["Excited"]):
        return "Excited"
    if any(k in text_lower for k in REFINE_KEYWORDS["Frustrated"]):
        return "Frustrated"
    if any(k in text_lower for k in REFINE_KEYWORDS["Love"]):
        return "Love"
    
    # 2. Neutral check (Lowest priority)
    if any(k in text_lower for k in REFINE_KEYWORDS["Neutral"]):
        return "Neutral"
        
    return label

def main():
    start_time = time.time()
    print("🚀 Running Lightweight 10-Class Training Pipeline (LinearSVC)")
    
    # 1. Load and Refine
    df = pd.read_csv(DATASET_PATH)
    df['label'] = df['label'].apply(normalize_label)
    df['final_label'] = df.apply(lambda r: refine_label(r['label'], r['text']), axis=1)
    df = df[df['final_label'].isin(EMOTION_LABELS)]
    
    print("Class Distribution:")
    print(df['final_label'].value_counts())
    
    # 2. Preprocess
    print("Preprocessing text...")
    df['processed'] = df['text'].apply(preprocess_text)
    df = df[df['processed'].str.len() > 2]
    
    # 3. Vectorize (User Requirement #1)
    print("Vectorizing (TF-IDF 1-2 grams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=60000,
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['final_label'], test_size=0.1, random_state=42, stratify=df['final_label']
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Model (Option A)
    print("Training LinearSVC with Calibration (for probabilities)...")
    base_clf = LinearSVC(
        class_weight='balanced',
        random_state=42,
        max_iter=5000,
        dual=False
    )
    model = CalibratedClassifierCV(base_clf, cv=3)
    model.fit(X_train_vec, y_train)
    
    # 5. Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save
    print("Saving model files...")
    with open(MODEL_FILE, 'wb') as f: pickle.dump(model, f)
    with open(VEC_FILE, 'wb') as f: pickle.dump(vectorizer, f)
    with open(MAP_FILE, 'wb') as f: pickle.dump({l: l for l in EMOTION_LABELS}, f)
    
    print(f"✅ Training complete in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
