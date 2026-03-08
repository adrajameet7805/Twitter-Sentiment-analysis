
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import time
import numpy as np

class EmotionClassifier:
    def __init__(self, model_path="./results_roberta_emotion"):
        """
        Initialize the inference pipeline.
        Allowed to fallback to default model if local path doesn't exist (for testing).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except OSError:
            print(f"Warning: Could not find model at {model_path}. Using base model for specific test (will not have correct 10 classes trained).")
            # This fallback is just to prevent crash if running without training first, 
            # effectively it should fail or user should train first. 
            # forcing exit if not found is safer for production.
            raise FileNotFoundError(f"Model not found at {model_path}. Please run train_transformer.py first.")

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load labels from model config
        self.id2label = self.model.config.id2label

    def preprocess(self, text):
        """Clean text similar to training."""
        text = str(text)
        text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def predict(self, texts, batch_size=32):
        """
        Optimized batch prediction.
        Args:
            texts (str or list): Single string or list of strings.
            batch_size (int): Batch size for processing.
        Returns:
            list of dicts: [{'label': 'Happy', 'score': 0.98}, ...]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess all texts
        cleaned_texts = [self.preprocess(t) for t in texts]
        
        results = []
        total_time = 0
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(cleaned_texts), batch_size):
                batch_texts = cleaned_texts[i:i+batch_size]
                
                start_time = time.time()
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Inference
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get top prediction
                scores, indices = torch.max(probs, dim=1)
                
                batch_time = time.time() - start_time
                total_time += batch_time
                
                # Format results
                for score, idx in zip(scores, indices):
                    results.append({
                        "label": self.id2label[idx.item()],
                        "score": score.item()
                    })

        avg_time_ms = (total_time / len(texts)) * 1000 if texts else 0
        # print(f"Inference speed: {avg_time_ms:.2f} ms/sample")
        
        return results

# Example Usage Block
if __name__ == "__main__":
    # Test samples
    samples = [
        "I am so happy that we finished this project on time!",
        "This is absolutely disgusting, I can't believe it.",
        "I'm really worried about the results tomorrow.",
        "The event was okay, nothing special.",
        "I love this new feature, it's amazing!",
        "Why is this not working? It's so frustrating!",
        "I tried to contact support but no one answered.",
        "Winning the lottery would be a dream come true.",
        "I'm shocked by the news!",
        "I feel so energetic and ready to go!"
    ] * 100  # valid stress test with 1000 samples

    print("Initializing classifier...")
    # NOTE: This will fail if model is not trained. 
    # For demonstration, we assume model exists at default path.
    # If simply testing code structure without model, comment out the load or handle exception.
    
    try:
        classifier = EmotionClassifier()
        
        print(f"Running inference on {len(samples)} samples...")
        start = time.time()
        predictions = classifier.predict(samples, batch_size=64)
        end = time.time()
        
        print(f"Total time: {end - start:.4f} seconds")
        print(f"Throughput: {len(samples) / (end - start):.2f} samples/sec")
        
        # Show first 5 results
        for text, res in zip(samples[:5], predictions[:5]):
            print(f"Text: {text[:50]}... -> {res['label']} ({res['score']:.4f})")
            
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Please run 'python train_transformer.py' first to generate the model.")
