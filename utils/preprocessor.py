"""
utils/preprocessor.py
Shared text-preprocessing utilities used by both the Streamlit app and training scripts.
Optimized version (v5) replacing NLTK PorterStemmer with spaCy Lemmatization, 
adding robust URL/mention cleaning, and providing a batch processing method.
"""

import re
import spacy

# Load spaCy English model (make sure to run `python -m spacy download en_core_web_sm` if not installed)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Disable pipeline components we don't need for speed
except OSError:
    import subprocess
    import sys
    # Try fully automated download if it fails on Render
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text: str) -> str:
    """
    Enhanced preprocessing pipeline (v5):
    1. Detects and preserves punctuation features (EXCLAMATION, QUESTION)
    2. Lowercases and aggressively cleans text (removes URLs, mentions)
    3. Lemmatizes using spaCy and removes stopwords
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Punctuation Feature Engineering
    exclam_count  = text.count('!')
    question_count = text.count('?')

    punctuation_tokens = []
    if exclam_count > 0:
        punctuation_tokens.extend(['EXCLAMATION'] * min(exclam_count, 3))
    if question_count > 0:
        punctuation_tokens.extend(['QUESTION'] * min(question_count, 3))

    # 2. Text Cleaning: Lowercase, remove URLs, mentions, and non-letters
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)  # URLs
    text = re.sub(r'@\w+', '', text)                                           # Mentions
    text = re.sub(r'[^a-z\s]', ' ', text)                                      # Keep only letters

    # 3. Tokenize & Lemmatize using spaCy (removes stop words & punctuations)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_space and len(token.lemma_) > 1]

    # Combine
    final_tokens = lemmas + punctuation_tokens
    return ' '.join(final_tokens)

def preprocess_texts(texts: list) -> list:
    """
    Hyper-optimized batch processing method for lists of texts.
    Leverages `nlp.pipe` for dramatic speed improvements in large batches.
    """
    # Pre-clean strings in pure Python to save spaCy pipeline work
    cleaned_texts = []
    punct_feats = []
    
    for t in texts:
        t_str = str(t).lower()
        exc = t_str.count('!')
        que = t_str.count('?')
        
        feats = []
        if exc > 0: feats.extend(['EXCLAMATION'] * min(exc, 3))
        if que > 0: feats.extend(['QUESTION'] * min(que, 3))
        punct_feats.append(feats)
        
        t_str = re.sub(r'http\S+|www\.\S+|https\S+', '', t_str, flags=re.MULTILINE)
        t_str = re.sub(r'@\w+', '', t_str)
        t_str = re.sub(r'[^a-z\s]', ' ', t_str)
        cleaned_texts.append(t_str)

    # Fast batch lemmatization
    processed_results = []
    for doc, feats in zip(nlp.pipe(cleaned_texts, batch_size=2000, n_process=1), punct_feats):
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_space and len(token.lemma_) > 1]
        processed_results.append(' '.join(lemmas + feats))

    return processed_results
