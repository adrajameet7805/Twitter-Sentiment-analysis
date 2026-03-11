import cv2
import numpy as np
import os
import tempfile
import time
import logging
import gc
from collections import Counter

try:
    from deepface import DeepFace
except ImportError:
    raise ImportError(
        "DeepFace is not installed. Run: pip install deepface\n"
        "Or install all dependencies: pip install -r requirements.txt"
    )

import mediapipe as mp
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    Optimized with MediaPipe, 2 FPS sampling, weighted tracking, and GC.
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
        
        # Sample at 2 FPS to reduce computation load significantly length
        frame_interval = max(1, int(fps / 2))
        
        timeline = []
        preview_frames = []
        total_faces = 0
        f_idx = 0
        analyzed_count = 0

        from deepface.modules import modeling
        emotion_client = modeling.build_model('facial_attribute', 'Emotion')
        emotion_model = emotion_client.model if hasattr(emotion_client, 'model') else emotion_client
        
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_scores = {label: 0.0 for label in emotion_labels}
        
        model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
        if not os.path.exists(model_path):
            logging.info("Downloading MediaPipe Face Detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        face_detection = vision.FaceDetector.create_from_options(options)

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
                conf = float(pred[emotion_idx])
                
                # Weighted averaging instead of strict counting
                emotion_scores[emotion] += conf
                
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
                timeline.append({"time": ts, "emotion": dom_emotion})
                
                if len(preview_frames) < 20: # hard limit UI payload
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
                draw_frame = None
                rgb_frame = None
                
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    results = face_detection.detect(mp_image)
                    
                    if not results.detections:
                        timeline.append({"time": timestamp, "emotion": "No Face"})
                    else:
                        draw_frame = frame.copy()
                        ih, iw, _ = frame.shape
                        for detection in results.detections:
                            bbox = detection.bounding_box
                            
                            x = int(bbox.origin_x)
                            y = int(bbox.origin_y)
                            w = int(bbox.width)
                            h = int(bbox.height)
                            
                            # Skip frames where faces are too small
                            if w < 30 or h < 30:
                                continue
                                
                            x = max(0, x)
                            y = max(0, y)
                            w = min(iw - x, w)
                            h = min(ih - y, h)
                            
                            if w == 0 or h == 0:
                                continue

                            face_roi = rgb_frame[y:y+h, x:x+w]
                            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                            face_resized = cv2.resize(face_gray, (48, 48))
                            face_normalized = face_resized / 255.0
                            face_expanded = np.expand_dims(face_normalized, axis=-1)
                            
                            batch_faces.append(face_expanded)
                            batch_meta.append((timestamp, draw_frame, (x, y, w, h)))
                            
                            del face_roi, face_gray, face_resized, face_normalized, face_expanded
                    
                    if len(batch_faces) >= 32:
                        processed_faces_in_batch = process_batch(batch_faces, batch_meta)
                        if processed_faces_in_batch:
                            total_faces += processed_faces_in_batch
                        batch_faces = []
                        batch_meta = []

                    analyzed_count += 1
                    
                    # Prevent memory spiralling
                    if analyzed_count % 30 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logging.error(f"Error at {timestamp}: {e}")
                    
                del frame, rgb_frame
                if draw_frame is not None:
                    del draw_frame

            f_idx += 1

        processed_faces_in_batch = process_batch(batch_faces, batch_meta)
        if processed_faces_in_batch:
            total_faces += processed_faces_in_batch
        batch_faces = []
        batch_meta = []
        
        timeline.sort(key=lambda x: float(str(x['time']).replace('s', '')))
        
        cap.release()
        face_detection.close()

        # ── 3. Aggregation ────────────────────────────────────────────────────
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            majority = max(emotion_scores, key=emotion_scores.get)
            dist = {k: round(v / total_score, 2) for k, v in emotion_scores.items()}
            
            result["face_emotion"] = EMOTION_LABEL_MAP.get(majority, majority.capitalize())
            result["face_confidence"] = emotion_scores[majority] / total_score
            
            # Use mapped labels for UI
            result["emotion_distribution"] = {EMOTION_LABEL_MAP.get(k, k.capitalize()): v for k, v in dist.items() if v > 0}
            result["emotion_timeline"] = timeline
            result["face_preview_frames"] = preview_frames[:20]
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
        gc.collect()

    result["processing_time"] = f"{round(time.time() - start_time, 2)} sec"
    return result
