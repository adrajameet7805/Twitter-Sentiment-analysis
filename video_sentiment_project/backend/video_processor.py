import cv2
import numpy as np
import os
import tempfile
import time
import logging
from collections import Counter

try:
    from deepface import DeepFace
except ImportError:
    import subprocess
    import sys
    try:
        print("DEBUG: DeepFace missing. Attempting automatic installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deepface"])
        from deepface import DeepFace
    except Exception:
        raise RuntimeError("DeepFace dependency missing. Please run: pip install deepface")

# Removed: from backend.audio_processor import transcribe_audio, extract_audio_ffmpeg
# Removed: from backend.predictor import predict_emotion_v4

# Canonical mapping to keep UI consistent
EMOTION_LABEL_MAP = {
    "angry":    "Angry",
    "disgust":  "Disgust",
    "fear":     "Fear",
    "happy":    "Happy",
    "neutral":  "Neutral",
    "sad":      "Sad",
    "surprise": "Surprise",
}

def process_video(uploaded_file, engine) -> dict:
    """
    Video sentiment analysis pipeline (Face-Only):
    1. Face emotion detection at 6 FPS
    2. Draw bounding boxes with emotion labels
    3. Generate timeline and distribution
    """
    start_time = time.time()
    result = {
        "face_emotion": "Unknown",
        "face_confidence": 0.0,
        "emotion_distribution": {},
        "emotion_timeline": [],
        "face_preview_frames": [], # For UI display
        "faces_detected": 0,
        "frames_analyzed": 0,
        "processing_time": "0.0 sec",
        "error": None,
    }

    video_tmp = None

    try:
        # ── 1. Save video to temp ─────────────────────────────────────────────
        suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as vf:
            vf.write(uploaded_file.read())
            video_tmp = vf.name

        # ── 2. Face Emotion Processing ────────────────────────────────────────
        cap = cv2.VideoCapture(video_tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Sample at 6 FPS
        frame_interval = max(1, int(fps / 3))
        
        emotion_results = []
        timeline = []
        preview_frames = []
        total_faces = 0
        f_idx = 0
        analyzed_count = 0

        while True:
            success, frame = cap.read()
            if not success or analyzed_count >= 500:
                break
                
            if f_idx % frame_interval == 0:
                timestamp = f"{f_idx / fps:.2f}s"
                try:
                    # DeepFace Analysis
                    objs = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    if not isinstance(objs, list): objs = [objs]
                    
                    frame_emotions = []
                    draw_frame = frame.copy()
                    face_in_frame = False

                    for obj in objs:
                        region = obj.get("region", {})
                        if not region or region.get("w", 0) == 0: continue
                        
                        face_in_frame = True
                        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                        emotion = obj.get("dominant_emotion", "Unknown")
                        conf = obj.get("emotion", {}).get(emotion, 0) / 100.0
                        
                        frame_emotions.append(emotion)
                        total_faces += 1

                        # Bounding Box and Label
                        cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{emotion.capitalize()} ({conf:.0%})"
                        cv2.putText(draw_frame, label, (x, max(y-10, 20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if face_in_frame:
                        dom_emotion = Counter(frame_emotions).most_common(1)[0][0]
                        emotion_results.append(dom_emotion)
                        timeline.append({"time": timestamp, "emotion": dom_emotion})
                        
                        if len(preview_frames) < 10:
                            rgb_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                            preview_frames.append(rgb_frame)
                    else:
                        timeline.append({"time": timestamp, "emotion": "No Face"})

                    analyzed_count += 1
                except Exception as e:
                    logging.error(f"DeepFace error at {timestamp}: {e}")

            f_idx += 1

        cap.release()

        # ── 3. Aggregation ────────────────────────────────────────────────────
        if emotion_results:
            counts = Counter(emotion_results)
            majority = counts.most_common(1)[0][0]
            dist = {k: round(v / len(emotion_results), 2) for k, v in counts.items()}
            
            result["face_emotion"] = majority
            result["face_confidence"] = counts[majority] / len(emotion_results)
            result["emotion_distribution"] = dist
            result["emotion_timeline"] = timeline
            result["face_preview_frames"] = preview_frames
            result["faces_detected"] = total_faces
            result["frames_analyzed"] = analyzed_count
        else:
            result["face_emotion"] = "No Face Detected"
            result["emotion_timeline"] = timeline

    except Exception as e:
        result["error"] = str(e)
        logging.error(f"Global processing error: {e}")

    finally:
        if video_tmp and os.path.exists(video_tmp):
            try: os.unlink(video_tmp)
            except: pass

    result["processing_time"] = f"{round(time.time() - start_time, 2)} sec"
    return result
