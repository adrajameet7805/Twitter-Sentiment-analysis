import io
import os
import logging
import librosa
from faster_whisper import WhisperModel
import re

# ── Singleton Whisper model (loaded only once) ─────────────────────
_whisper_model = None

_model_cache = None

def _get_model():
    """Load and cache the Faster-Whisper model using a module-level singleton."""
    global _model_cache
    if _model_cache is None:
        try:
            # "tiny.en" is extremely fast and sufficient for basic English sentiment.
            _model_cache = WhisperModel(
                "tiny.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1
            )
            logging.info("WhisperModel loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading WhisperModel: {e}")
            raise
    return _model_cache

def clean_transcription(text: str) -> str:
    """Simple text cleaning before sentiment analysis."""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def transcribe_audio(audio_bytes: bytes, file_ext=None):
    """
    Fast audio transcription using faster-whisper.
    Works without FFmpeg if librosa uses alternatives, and cleans signal.
    """

    result = {
        "transcription": "",
        "error": None,
        "duration_used": 0.0
    }

    try:
        # ── Convert uploaded bytes → waveform ───────────────────────
        audio_file = io.BytesIO(audio_bytes)

        # 1. Load with native sample rate first
        waveform, sr = librosa.load(
            audio_file,
            sr=None
        )

        import numpy as np

        if waveform is None or len(waveform) == 0:
            raise ValueError("Empty audio file")

        # 2. Trim silence segments to reduce processing time
        waveform, _ = librosa.effects.trim(waveform)
        
        # 3. Normalize amplitude to reduce background noise discrepancies
        waveform = librosa.util.normalize(waveform)
        
        # 4. Resample to 16000Hz for Whisper
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

        # ── Load Whisper model ──────────────────────────────────────
        model = _get_model()

        segments, info = model.transcribe(
            waveform,
            beam_size=1,
            best_of=1,
            temperature=0,
            vad_filter=True,
            word_timestamps=False
        )

        # ── Collect transcription lines properly ────────────────────
        lines = []

        for seg in segments:
            text = seg.text.strip()
            if text:
                lines.append(text)

        transcription = " ".join(lines)
        
        # Clean the final text
        transcription = clean_transcription(transcription)

        if not transcription:
            result["transcription"] = "Audio could not be transcribed"
        else:
            result["transcription"] = transcription

        result["duration_used"] = getattr(info, "duration", 0)

    except Exception as e:
        logging.error(f"Transcription error: {e}")
        result["error"] = str(e)
        result["transcription"] = "Audio could not be transcribed"

    return result