"""
Optimized Emotion Classification Model Training Script (v3)

Improvements:
- Feature Extraction: TF-IDF (1-3 ngrams), sublinear_tf=True, min_df=2
- Punctuation Features: EXCLAMATION, QUESTION tokens
- Model: CalibratedClassifierCV(LogisticRegression(class_weight='balanced'))
- Data: Stratified sample of 80,000 rows
"""

import pandas as pd
import numpy as np
import pickle
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Enhanced preprocessing pipeline:
    1. Detects and preserves punctuation features (EXCLAMATION, QUESTION)
    2. Lowercases and cleans text
    3. Stems and removes stopwords
    """
    text = str(text)
    
    # 1. Punctuation Feature Engineering
    # Count specific punctuation before cleaning
    exclam_count = text.count('!')
    question_count = text.count('?')
    
    # Append special tokens based on counts
    punctuation_tokens = []
    if exclam_count > 0:
        punctuation_tokens.extend(['EXCLAMATION'] * min(exclam_count, 3)) # Cap at 3
    if question_count > 0:
        punctuation_tokens.extend(['QUESTION'] * min(question_count, 3)) # Cap at 3
        
    # 2. Lowercase and remove non-letters (keep spaces)
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    
    # 3. Tokenize
    words = text.split()
    
    # 4. Stem and remove stopwords
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    # 5. Combine words and punctuation tokens
    final_tokens = words + punctuation_tokens
    
    return ' '.join(final_tokens)

# Start timer
start_time = time.time()

# Load dataset
print("Loading emotions.csv dataset...")
try:
    df = pd.read_csv('emotions.csv')
    print(f"Full dataset: {len(df):,} rows")
except FileNotFoundError:
    print("❌ Error: emotions.csv not found!")
    exit(1)

# Map Love (label 2) to Happy/Joy (label 1)
df['label'] = df['label'].replace(2, 1)

# Stratified sampling (80k rows for speed/performance balance)
SAMPLE_SIZE = 80000
print(f"\nUsing stratified sample of {SAMPLE_SIZE:,} rows...")
if len(df) > SAMPLE_SIZE:
    df_sample = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(SAMPLE_SIZE/5)), random_state=42)
    ).sample(frac=1, random_state=42).reset_index(drop=True)
else:
    df_sample = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Actual sample size: {len(df_sample):,} rows")

# Preprocess texts
print("\nPreprocessing texts (with punctuation features)...")
df_sample['processed_text'] = [preprocess_text(t) for t in df_sample['text']]

# Remove empty processed texts
df_sample = df_sample[df_sample['processed_text'].str.len() > 0]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df_sample['processed_text'], 
    df_sample['label'], 
    test_size=0.1, 
    random_state=42, 
    stratify=df_sample['label']
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# Create Optimized TF-IDF vectorizer
print("\nCreating TF-IDF vectorizer (1-3 ngrams, sublinear_tf)...")
vectorizer = TfidfVectorizer(
    max_features=15000,      # Increased features to capture more ngrams
    ngram_range=(1, 3),      # Unigrams, Bigrams, Trigrams
    min_df=2,                # Ignore very rare terms
    sublinear_tf=True,       # Apply logarithmic scaling
    strip_accents='unicode',
    stop_words='english'     # Built-in stop words for vectorizer as well
)

# Fit vectorizer
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

# Train Model
print("\nTraining Calibrated Logistic Regression model...")

# Base estimator
base_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced', # Handle class imbalance
    solver='lbfgs',
    n_jobs=-1
)

# Calibrated Classifier
model = CalibratedClassifierCV(
    estimator=base_model,
    method='sigmoid',
    cv=3,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)
print("Model training complete!")

# Evaluate
print("\n" + "="*40)
print("MODEL EVALUATION")
print("="*40)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Emotion mapping
emotion_map = {
    0: 'Sad',
    1: 'Happy / Joy',
    3: 'Angry',
    4: 'Fear',
    5: 'Surprise'
    # Neutral is handled by threshold logic, not a training class usually, 
    # unless dataset has specific Neutral label (which this dataset seems not to have explicitly in 0-5 mapping usually)
    # However, if dataset HAS neutral, we should map it. 
    # Standard diff: 0=sad, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise. 
    # We mapped love->joy. So we have 0,1,3,4,5.
}

print(classification_report(
    y_test, 
    y_pred,
    target_names=[emotion_map.get(i, f'Class {i}') for i in sorted(y_test.unique())],
    digits=4
))

# Test Specific Cases
print("\n" + "="*40)
print("QUICK VERIFICATION")
print("="*40)
test_sentences = [
    "Wow! That is amazing!",
    "I am so angry right now.",
    "The meeting is at 2pm."
]

for text in test_sentences:
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    probs = model.predict_proba(vec)[0]
    pred_idx = np.argmax(probs)
    conf = probs[pred_idx]
    emotion = emotion_map.get(model.classes_[pred_idx], "Unknown")
    
    print(f"Text: '{text}'")
    print(f"  -> Tokens: {processed}")
    print(f"  -> Pred: {emotion} ({conf:.1%})")

# Save files
print("\nSaving model files...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('emotion_mapping.pkl', 'wb') as f:
    pickle.dump(emotion_map, f)

total_time = time.time() - start_time
print("\nOptimization complete in {total_time:.1f} seconds!")
