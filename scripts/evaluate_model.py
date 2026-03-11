import os
import sys
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import preprocess_texts

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust",
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

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
    print("Loading data...")
    df = pd.read_csv('../data/emotions.csv')
    
    df = df.sample(n=min(50000, len(df)), random_state=42)
    print(f"Sampled rows for evaluation: {len(df)}")
    
    df['label'] = df['label'].apply(normalize_label)
    df['final_label'] = df.apply(lambda r: refine_label(r['label'], r['text']), axis=1)
    df = df[df['final_label'].isin(EMOTION_LABELS)]
    
    model_path = '../models/model_v5_light.pkl'
    vec_path = '../models/vectorizer_v5_light.pkl'
    
    print("Loading models...")
    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vec_path, 'rb'))
    print(f"Model classes: {model.classes_}")
    
    print("Preprocessing text...")
    X_processed = preprocess_texts(df['text'].fillna("").tolist())
    
    print("Vectorizing...")
    X_vec = vectorizer.transform(X_processed)
    
    print("Predicting...")
    y_pred = model.predict(X_vec)
    y_true = df['final_label'].values
    
    print("\nMetrics:")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")

if __name__ == "__main__":
    main()
