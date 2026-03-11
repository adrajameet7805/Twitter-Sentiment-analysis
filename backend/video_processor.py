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
    raise ImportError(
        "DeepFace is not installed. Run: pip install deepface\n"
        "Or install all dependencies: pip install -r requirements.txt"
    )

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
        frame_interval = max(1, int(fps / 6))
        
        emotion_results = []
        timeline = []
        preview_frames = []
        total_faces = 0
        f_idx = 0
        analyzed_count = 0

        from deepface.modules import modeling
        emotion_client = modeling.build_model('facial_attribute', 'Emotion')
        emotion_model = emotion_client.model if hasattr(emotion_client, 'model') else emotion_client
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        batch_faces = []
        batch_meta = [] # Store (timestamp, orig_frame, (x, y, w, h))

        def process_batch(faces, meta):
            if not faces: return
            
            batch_arr = np.array(faces)
            preds = emotion_model.predict(batch_arr, verbose=0)
            
            frame_emotions_map = {}
            frame_image_map = {}
            
            for i, pred in enumerate(preds):
                emotion_idx = np.argmax(pred)
                emotion = emotion_labels[emotion_idx]
                conf = pred[emotion_idx]
                
                timestamp, draw_frame, (x, y, w, h) = meta[i]
                
                if timestamp not in frame_emotions_map:
                    frame_emotions_map[timestamp] = []
                    frame_image_map[timestamp] = draw_frame
                
                frame_emotions_map[timestamp].append(emotion)
                
                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion.capitalize()} ({conf:.0%})"
                cv2.putText(draw_frame, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for ts, ems in frame_emotions_map.items():
                dom_emotion = Counter(ems).most_common(1)[0][0]
                emotion_results.append(dom_emotion)
                timeline.append({"time": ts, "emotion": dom_emotion})
                
                if len(preview_frames) < 10:
                    try:
                        rgb_frame = cv2.cvtColor(frame_image_map[ts], cv2.COLOR_BGR2RGB)
                        preview_frames.append(rgb_frame)
                    except:
                        pass
        
            return len(preds)

        while True:
            success = cap.grab()
            if not success or analyzed_count >= 500:
                break
                
            if f_idx % frame_interval == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break

                timestamp = f"{f_idx / fps:.2f}s"
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) == 0:
                        timeline.append({"time": timestamp, "emotion": "No Face"})
                    else:
                        draw_frame = frame.copy()
                        for (x, y, w, h) in faces:
                            face_roi = gray[y:y+h, x:x+w]
                            face_resized = cv2.resize(face_roi, (48, 48))
                            face_normalized = face_resized / 255.0
                            face_expanded = np.expand_dims(face_normalized, axis=-1)
                            
                            batch_faces.append(face_expanded)
                            batch_meta.append((timestamp, draw_frame, (x, y, w, h)))
                    
                    if len(batch_faces) >= 32:
                        processed_faces_in_batch = process_batch(batch_faces, batch_meta)
                        if processed_faces_in_batch:
                            total_faces += processed_faces_in_batch
                        batch_faces = []
                        batch_meta = []

                    analyzed_count += 1
                except Exception as e:
                    logging.error(f"Error at {timestamp}: {e}")

            f_idx += 1

        processed_faces_in_batch = process_batch(batch_faces, batch_meta)
        if processed_faces_in_batch:
            total_faces += processed_faces_in_batch
        batch_faces = []
        batch_meta = []
        
        timeline.sort(key=lambda x: float(str(x['time']).replace('s', '')))
        
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
