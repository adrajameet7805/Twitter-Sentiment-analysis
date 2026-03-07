
import os
import torch
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# ==================================================================================
# CONFIGURATION
# ==================================================================================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
DATASET_PATH = "emotions.csv"
OUTPUT_DIR = "./results_roberta_emotion"
MAX_LENGTH = 128  # Efficient length for most tweets/sentences
batch_size = 16
learning_rate = 2e-5
epochs = 4
random_seed = 42

EMOTION_LABELS = [
    "Happy / Joy",
    "Sad",
    "Angry",
    "Fear",
    "Disgust",
    "Surprise",
    "Neutral",
    "Love",
    "Excited",
    "Frustrated"
]

# Create mappings
label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ==================================================================================
# PREPROCESSING FUNCTIONS
# ==================================================================================

# Define additional keywords for label refinement (copied from original logic)
LOVE_KEYWORDS = {"love", "adore", "cherish", "affection", "romance", "sweetheart"}
FRUSTRATION_KEYWORDS = {"frustrated", "frustration", "irritated", "annoyed", "fed up", "exasperated", "aggravated"}
EXCITED_KEYWORDS = {"excited", "excitement", "thrilled", "eager", "pumped", "cant wait", "can't wait"}

def normalize_label(value):
    """Normalize inconsistent labels from the CSV."""
    if isinstance(value, (int, np.integer)):
        temp_map = {
            0: "Sad", 1: "Happy / Joy", 2: "Love",
            3: "Angry", 4: "Fear", 5: "Surprise"
        }
        return temp_map.get(int(value), None)
    
    label = str(value).strip().lower()
    mapping = {
        "joy": "Happy / Joy", "happy": "Happy / Joy",
        "sad": "Sad", "sadness": "Sad",
        "anger": "Angry", "angry": "Angry",
        "fear": "Fear",
        "surprise": "Surprise",
        "neutral": "Neutral",
        "love": "Love",
        "excited": "Excited",
        "frustration": "Frustrated"
    }
    return mapping.get(label, None)

def refine_label(label, text):
    """Refine labels based on text keywords."""
    text_lower = str(text).lower()
    if label == "Happy / Joy":
        if any(k in text_lower for k in LOVE_KEYWORDS):
            return "Love"
        if any(k in text_lower for k in EXCITED_KEYWORDS):
            return "Excited"
    if label == "Angry" and any(k in text_lower for k in FRUSTRATION_KEYWORDS):
        return "Frustrated"
    return label

def clean_text(text):
    """
    Clean text for Transformer model.
    - Remove URLs
    - Remove user mentions (@user)
    - Lowercase
    - Keep emojis (important for emotion)
    - Remove excessive whitespace
    """
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()  # RoBERTa is case-sensitive usually, but this version might benefit from lowercasing or we can keep case.
                         # CHECK: cardiffnlp/twitter-roberta-base-emotion IS case-sensitive usually (RoBERTa).
                         # However, user Requirements #4 says "Lowercase". So we will lowercase.

# ==================================================================================
# CUSTOM TRAINER FOR CLASS IMBALANCE
# ==================================================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        probabilities = torch.nn.functional.softmax(logits, dim=-1) # optional linear layer usually handles this
        
        # We need the weights on the same device as the model
        if self.class_weights is not None:
            weight = self.class_weights.to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==================================================================================
# METRICS COMPUTATION
# ==================================================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    # Calculate confusion matrix (optional to print, but here we just return standard metrics)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================
def main():
    print(f"Loading and processing {DATASET_PATH}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: {DATASET_PATH} not found.")
        return

    # 2. Preprocess Labels using existing logic
    print("Normalizing labels...")
    df['label'] = df['label'].apply(normalize_label)
    df = df[df['label'].notna()]
    df['label'] = df.apply(lambda r: refine_label(r['label'], r['text']), axis=1)
    
    # Filter to only the 10 classes we care about
    print(f"Filtering for {len(EMOTION_LABELS)} target emotions...")
    df = df[df['label'].isin(EMOTION_LABELS)]
    
    # 3. Clean Text
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove duplicates and empty strings
    df.drop_duplicates(subset=['text'], inplace=True)
    df = df[df['text'].str.len() > 3] # minimal length
    
    print(f"Final dataset size: {len(df)}")
    print("Class distribution:")
    print(df['label'].value_counts())

    # Map labels to IDs
    df['label_id'] = df['label'].map(label2id)

    # 4. Stratified Split
    print("Splitting dataset...")
    # Using a smaller subset for demonstration if dataset is huge, otherwise use full
    # For speed in this example, if dataset > 20k, we might want to sample. 
    # But user asked for >95% accuracy so we should use as much as possible.
    # We'll use the full dataset for high accuracy. 
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label_id'].tolist(), 
        test_size=0.2, 
        stratify=df['label_id'], 
        random_state=random_seed
    )

    # 5. Tokenization
    print(f"Tokenizing with {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    class EmotionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)

    # 6. Compute Class Weights
    print("Computing class weights for imbalance handling...")
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 7. Model Initialization
    # Note: ignoring mismatched sizes is necessary because we have different number of labels than pre-trained
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(EMOTION_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )

    # 8. Training Setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=learning_rate,
        save_total_limit=2,
        fp16=torch.cuda.is_available()  # Use mixed precision if GPU available
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Inject class weights into trainer
    trainer.class_weights = class_weights

    # 9. Train
    print("Starting training...")
    trainer.train()

    # 10. Evaluation & Save
    print("Evaluating model...")
    results = trainer.evaluate()
    print("Evaluation Results:", results)

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! You can now use inference.py to run predictions.")

if __name__ == "__main__":
    main()
