"""
Optimized Emotion Classification Model Training Script

Uses a stratified sample of the dataset for faster training while maintaining class balance.
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

def preprocess_text(text):
    """Preprocess text using the same steps as the current system"""
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Load dataset
print("Loading emotions.csv dataset...")
df = pd.read_csv('emotions.csv')
print(f"Full dataset: {len(df):,} rows")

# Map Love (label 2) to Happy/Joy (label 1)
print("\nMapping Love (label 2) → Happy / Joy (label 1)...")
df['label'] = df['label'].replace(2, 1)

# Use stratified sampling for faster training (50K samples)
print("\nUsing stratified sample of 50,000 rows for faster training...")
df_sample = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 10000), random_state=42)
).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sample size: {len(df_sample):,} rows")
print("\nLabel distribution in sample:")
print(df_sample['label'].value_counts().sort_index())

# Create emotion mapping
emotion_map = {
    0: 'Sad',
    1: 'Happy / Joy',
    3: 'Angry',
    4: 'Fear',
    5: 'Surprise'
}

# Preprocess texts
print("\nPreprocessing texts...")
df_sample['processed_text'] = df_sample['text'].progress_apply(preprocess_text) if hasattr(df_sample, 'progress_apply') else df_sample['text'].apply(preprocess_text)

# Remove empty processed texts
df_sample = df_sample[df_sample['processed_text'].str.len() > 0]
print(f"After removing empty texts: {len(df_sample):,} rows")

# Prepare features and labels
X = df_sample['processed_text']
y = df_sample['label']

# Split dataset
print("\nSplitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# Create TF-IDF vectorizer
print("\nCreating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs'
)

model.fit(X_train_vec, y_train)
print("✅ Model training complete!")

# Evaluate model
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=[emotion_map.get(i, f'Class {i}') for i in sorted(y_test.unique())],
    digits=4
))

# Save model and vectorizer
print("\n" + "="*60)
print("SAVING MODEL FILES")
print("="*60)

print("\nSaving model.pkl...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Saving vectorizer.pkl...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Saving emotion_mapping.pkl...")
with open('emotion_mapping.pkl', 'wb') as f:
    pickle.dump(emotion_map, f)

print("\n✅ Files saved successfully!")

# Test predictions
print("\n" + "="*60)
print("TESTING PREDICTIONS")
print("="*60)

test_samples = [
    "I'm so happy and excited! This is amazing!",
    "I feel so sad and lonely right now.",
    "This makes me so angry and frustrated!",
    "I'm terrified and scared about what might happen.",
    "Wow! I can't believe this happened!",
    "The meeting is at 3pm tomorrow."
]

print("\nSample predictions:")
for text in test_samples:
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred_label = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)
    
    emotion = emotion_map.get(pred_label, f'Unknown ({pred_label})')
    
    # Apply neutral logic
    if max_prob < 0.60:
        emotion = "Neutral"
        confidence = max_prob
    else:
        confidence = max_prob
    
    print(f"\n  Text: \"{text}\"")
    print(f"  Emotion: {emotion}")
    print(f"  Confidence: {confidence:.1%}")

print("\n" + "="*60)
print("✅ Training completed successfully!")
print("="*60)
