
"""
utils/preprocessor.py
Shared text-preprocessing utilities used by both the Streamlit app and training scripts.
No logic has been changed from the original app.py / training scripts.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
    Enhanced preprocessing pipeline (must match training pipeline):
    1. Detects and preserves punctuation features (EXCLAMATION, QUESTION)
    2. Lowercases and cleans text
    3. Stems and removes stopwords
    """
    text = str(text)

    # 1. Punctuation Feature Engineering
    exclam_count  = text.count('!')
    question_count = text.count('?')

    punctuation_tokens = []
    if exclam_count > 0:
        punctuation_tokens.extend(['EXCLAMATION'] * min(exclam_count, 3))
    if question_count > 0:
        punctuation_tokens.extend(['QUESTION'] * min(question_count, 3))

    # 2. Lowercase and remove non-letters
    text = re.sub('[^a-zA-Z]', ' ', text.lower())

    # 3. Tokenize
    words = text.split()

    # 4. Stem and remove stopwords
    words = [stemmer.stem(w) for w in words if w not in stopwords.words('english')]

    # 5. Combine
    final_tokens = words + punctuation_tokens

    return ' '.join(final_tokens)
