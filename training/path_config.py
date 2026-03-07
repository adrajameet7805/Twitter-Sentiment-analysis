"""
training/path_config.py
Resolves project-root-relative paths for all training scripts.
Import this at the top of any training script inside training/ to get the
correct absolute paths regardless of which directory the script is run from.

Usage inside any training/*.py:
    from training.path_config import ROOT, DATASET_PATH, MODELS_DIR
"""

import os

# The project root is one level above this file (training/)
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

# Canonical data file location
DATASET_PATH = os.path.join(DATA_DIR, "emotions.csv")

# Canonical model output paths
MODEL_V4_LIGHT   = os.path.join(MODELS_DIR, "model_v4_light.pkl")
VEC_V4_LIGHT     = os.path.join(MODELS_DIR, "vectorizer_v4_light.pkl")
MAP_V4           = os.path.join(MODELS_DIR, "emotion_mapping_v4.pkl")
MODEL_PKL        = os.path.join(MODELS_DIR, "model.pkl")
VEC_PKL          = os.path.join(MODELS_DIR, "vectorizer.pkl")
MAP_PKL          = os.path.join(MODELS_DIR, "emotion_mapping.pkl")
DISTILBERT_DIR   = os.path.join(ROOT, "results_v4_distilbert")
