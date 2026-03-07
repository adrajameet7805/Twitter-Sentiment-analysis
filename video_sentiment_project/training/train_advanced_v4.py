
import os
import torch
import pandas as pd
import numpy as np
import re
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import nltk
from datasets import Dataset

# ==================================================================================
# CONFIGURATION
# ==================================================================================
MODEL_NAME = "distilbert-base-uncased"
DATASET_PATH = "emotions.csv"
OUTPUT_DIR = "./results_v4_distilbert"
MAX_LENGTH = 128
BATCH_SIZE = 32 # Increased for faster training if GPU supports
LEARNING_RATE = 3e-5
EPOCHS = 5
RANDOM_SEED = 42

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust",
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

# Create mappings
label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ==================================================================================
# KEYWORDS FOR 10-CLASS REFINEMENT
# ==================================================================================
REFINE_KEYWORDS = {
    "Disgust": ["disgust", "disgusting", "gross", "yuck", "revolting", "nasty", "vile", "repulsive", "sickening", "nauseating"],
    "Excited": ["excited", "thrilled", "cant wait", "can't wait", "pumped", "hyped", "ecstatic", "thrilling", "enthusiastic"],
    "Frustrated": ["frustrated", "frustrating", "annoyed", "irritated", "stuck", "fed up", "aggravated", "mentally exhausted"],
    "Love": ["love", "adore", "cherish", "affection", "romance", "romantic", "sweetheart"],
    "Neutral": ["meeting", "schedule", "time", "date", "call", "agenda", "calendar", "factual", "information"]
}

def normalize_label(value):
    if isinstance(value, (int, np.integer)):
        temp_map = {0: "Sad", 1: "Happy / Joy", 2: "Love", 3: "Angry", 4: "Fear", 5: "Surprise"}
        return temp_map.get(int(value), "Neutral")
    return "Neutral"

def refine_label(label, text):
    text_lower = str(text).lower()
    
    # Priority Keywords for rare classes
    if any(k in text_lower for k in REFINE_KEYWORDS["Disgust"]):
        return "Disgust"
    if any(k in text_lower for k in REFINE_KEYWORDS["Excited"]):
        return "Excited"
    if any(k in text_lower for k in REFINE_KEYWORDS["Frustrated"]):
        return "Frustrated"
    
    # Existing refinements
    if label == "Happy / Joy" and any(k in text_lower for k in REFINE_KEYWORDS["Love"]):
        return "Love"
    
    # Basic Neutral check: very short or strictly factual (simple heuristic)
    if label == "Neutral" or len(text_lower.split()) < 5:
        if any(k in text_lower for k in REFINE_KEYWORDS["Neutral"]):
            return "Neutral"
            
    return label

# ==================================================================================
# CUSTOM TRAINER FOR CLASS IMBALANCE
# ==================================================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if hasattr(self, "class_weights") and self.class_weights is not None:
            weight = self.class_weights.to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================
def main():
    print("🚀 Starting Advanced 10-Class Training Pipeline (v4)")
    
    # 1. Load Data
    print("Loading emotions.csv...")
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Refine Labels
    print("Processing 10-class refinement...")
    df['base_label'] = df['label'].apply(normalize_label)
    df['final_label'] = df.apply(lambda r: refine_label(r['base_label'], r['text']), axis=1)
    
    # 3. Clean and Filter
    print("Cleaning data...")
    df = df[df['final_label'].isin(EMOTION_LABELS)]
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    
    print("Final Class Distribution:")
    counts = df['final_label'].value_counts()
    print(counts)
    
    # Ensure all classes have some data (heuristic: duplicate very rare ones if needed, 
    # but SMOTE/weights are better)
    
    # 4. Prepare for Transformers
    df['label'] = df['final_label'].map(label2id)
    
    # Adaptive Sampling based on hardware
    if torch.cuda.is_available():
        sample_size = 50000
        batch_size = 16
    else:
        print("⚠️ No GPU detected. Reducing sample size to 10,000 for CPU training.")
        sample_size = 10000
        batch_size = 4
        
    rare_classes = ["Disgust", "Excited", "Frustrated"]
    df_rare = df[df['final_label'].isin(rare_classes)]
    # Keep all rare samples, then sample common to reach target
    common_target = max(100, sample_size - len(df_rare))
    df_common = df[~df['final_label'].isin(rare_classes)].sample(min(len(df), common_target), random_state=RANDOM_SEED)
    df_train_full = pd.concat([df_rare, df_common]).sample(frac=1, random_state=RANDOM_SEED)
    
    X_train, X_val, y_train, y_val = train_test_split(
        df_train_full['text'].tolist(), 
        df_train_full['label'].tolist(), 
        test_size=0.1, 
        stratify=df_train_full['label'], 
        random_state=RANDOM_SEED
    )
    
    # 5. Tokenization
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
    val_dataset = Dataset.from_dict({'text': X_val, 'label': y_val})
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # 6. Weights
    print("Computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.arange(len(EMOTION_LABELS)), 
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # 7. Model
    print("Initializing DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(EMOTION_LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # 8. Training
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        push_to_hub=False,
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.class_weights = class_weights
    
    print("Training started...")
    trainer.train()
    
    # 9. Final Evaluation
    print("\nFinal Evaluation:")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # 10. Save and Export
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save a simple pickle for app.py compatibility if needed, 
    # but app.py should be updated for transformer inference.
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
