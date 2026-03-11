
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from nltk.corpus import stopwords

# ==================================================================================
# CONFIGURATION
# ==================================================================================
# Paths are resolved relative to the project root (where app.py lives)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH     = os.path.join(_ROOT, "results_v4_distilbert")
MODEL_FILE_LIGHT = os.path.join(_ROOT, "models", "model_v4_light.pkl")
VEC_FILE_LIGHT   = os.path.join(_ROOT, "models", "vectorizer_v4_light.pkl")

EMOTION_LABELS = [
    "Happy / Joy", "Sad", "Angry", "Fear", "Disgust",
    "Surprise", "Neutral", "Love", "Excited", "Frustrated"
]

# Rule Triggers
RULE_TRIGGERS = {
    "Frustrated": ["frustrated", "fed up", "mentally exhausted", "irritated", "aggravated"],
    "Disgust":    ["disgusting", "gross", "nauseating", "revolting", "yuck", "vile"],
    "Excited":    ["thrilled", "so excited", "can't wait", "pumped", "hyped", "thrilling"],
    "Love":       ["love", "adore", "cherish", "lovey", "sweetheart"],
    "Surprise":   ["wow", "unexpected", "unbelievable", "omg", "really?", "wasn't expecting"]
}

class EmotionInferenceV4:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_transformer = False
        self.stop_words = stopwords.words('english')

        # 1. Try Loading Transformer Model
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            print(f"Loading DistilBERT model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.use_transformer = True
        # 2. Try Loading Lightweight ML Model
        elif os.path.exists(MODEL_FILE_LIGHT):
            import pickle
            print("Loading Lightweight LinearSVC model...")
            self.model = pickle.load(open(MODEL_FILE_LIGHT, 'rb'))
            self.vectorizer = pickle.load(open(VEC_FILE_LIGHT, 'rb'))
        else:
            print("Warning: No models found. System will fail on prediction.")

    def apply_hybrid_rules(self, text, probs):
        """
        Boost probabilities based on keyword triggers.
        probs is a dict mapping label name -> probability
        """
        text_lower = text.lower()
        boosted_probs = probs.copy()

        triggered = False
        for emotion, keywords in RULE_TRIGGERS.items():
            if any(k in text_lower for k in keywords):
                boosted_probs[emotion] += 0.5
                triggered = True

        if triggered:
            total = sum(boosted_probs.values())
            boosted_probs = {k: v / total for k, v in boosted_probs.items()}

        return boosted_probs

    def predict(self, text):
        """
        Predict emotion for a single text.
        Returns: (best_emotion, confidence, all_probs)
        """
        if self.use_transformer:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=128, padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            prob_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
        else:
            from utils.preprocessor import preprocess_text
            processed = preprocess_text(text)

            vec = self.vectorizer.transform([processed])
            probs = self.model.predict_proba(vec)[0]
            classes = self.model.classes_
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
            for lbl in EMOTION_LABELS:
                if lbl not in prob_dict:
                    prob_dict[lbl] = 0.0

        final_probs = self.apply_hybrid_rules(text, prob_dict)
        sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        best_emotion, confidence = sorted_probs[0]

        return best_emotion, confidence, final_probs

    def predict_batch(self, texts):
        """
        Batched prediction for multiple texts.
        Returns a list of tuples: (best_emotion, confidence, all_probs)
        """
        if not texts:
            return []

        results = []
        if self.use_transformer:
            # For transformer, we can batch everything if memory allows, but chunking is safer.
            # Here we just process in chunks. The predictor.py already chunks by 50, so this list is <= 50.
            inputs = self.tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=128, padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs_batch = F.softmax(logits, dim=-1).cpu().numpy()
            
            for i, text in enumerate(texts):
                probs = probs_batch[i]
                prob_dict = {EMOTION_LABELS[j]: float(probs[j]) for j in range(len(EMOTION_LABELS))}
                final_probs = self.apply_hybrid_rules(text, prob_dict)
                sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
                best_emotion, confidence = sorted_probs[0]
                results.append((best_emotion, confidence, final_probs))
        else:
            from utils.preprocessor import preprocess_text
            processed_texts = [preprocess_text(text) for text in texts]

            vecs = self.vectorizer.transform(processed_texts)
            probs_batch = self.model.predict_proba(vecs)
            classes = self.model.classes_

            for i, text in enumerate(texts):
                probs = probs_batch[i]
                prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
                for lbl in EMOTION_LABELS:
                    if lbl not in prob_dict:
                        prob_dict[lbl] = 0.0
                
                final_probs = self.apply_hybrid_rules(text, prob_dict)
                sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
                best_emotion, confidence = sorted_probs[0]
                results.append((best_emotion, confidence, final_probs))

        return results


# Example Test
if __name__ == "__main__":
    engine = EmotionInferenceV4()
    examples = [
        "I am so frustrated with this service!",
        "That's absolutely disgusting and gross.",
        "I'm so excited for the concert tonight!",
        "The meeting is at 5 PM.",
        "I love you so much!"
    ]
    for ex in examples:
        emo, conf, _ = engine.predict(ex)
        print(f"Text: {ex}\nPred: {emo} ({conf:.1%})\n")
