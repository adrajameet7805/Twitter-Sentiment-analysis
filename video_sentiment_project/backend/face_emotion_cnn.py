import os
import json
import cv2
import numpy as np
import logging

class FaceEmotionPredictor:
    """Predicts facial emotions from video frames using MediaPipe (Long Range) + CNN."""

    def __init__(self):
        self.model = None
        self.emotion_labels = {}
        self.is_ready = False
        self.detector = None
        
        self._initialize_models()

    def _initialize_models(self):
        try:
            from tensorflow.keras.models import load_model
            import mediapipe as mp
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_emotion_cnn.h5")
            labels_path = os.path.join(base_dir, "models", "emotion_labels.json")

            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                logging.warning(f"Face emotion CNN model or labels not found at {model_path}.")
                return

            self.model = load_model(model_path)

            with open(labels_path, 'r') as f:
                labels_dict = json.load(f)
                self.emotion_labels = {int(k): v for k, v in labels_dict.items()}

            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4)

            self.is_ready = True
            logging.info("Face Emotion Predictor successfully initialized with MediaPipe Long-Range.")

        except Exception as e:
            logging.error(f"Error initializing Face Emotion Predictor: {e}")


    def predict_frame(self, frame) -> list:
        """
        Detects all faces in the given frame using MediaPipe and returns predictions.
        Returns a list of dicts: {"emotion": str, "confidence": float, "box": tuple, "probabilities": list}
        """
        if not self.is_ready or self.detector is None:
            return []

        try:
            # Resize frame to width 640 while keeping aspect ratio
            h, w = frame.shape[:2]
            if w != 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply brightness normalization (dummy but effective approach)
            alpha = 1.1 # slight contrast
            beta = 20   # slight brightness
            enhanced_rgb = cv2.convertScaleAbs(rgb_frame, alpha=alpha, beta=beta)
            
            # Detect faces
            results = self.detector.process(enhanced_rgb)
            
            out_results = []
            
            if not results.detections:
                return []
                
            print(f"DEBUG: MediaPipe detected {len(results.detections)} faces.")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_height, image_width = gray.shape

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * image_width)
                y = int(bboxC.ymin * image_height)
                w_box = int(bboxC.width * image_width)
                h_box = int(bboxC.height * image_height)
                
                # Expand bounding box by 15% to capture full face
                expand_w = int(w_box * 0.15)
                expand_h = int(h_box * 0.15)
                
                new_x = max(0, x - expand_w // 2)
                new_y = max(0, y - expand_h // 2)
                new_w = min(image_width - new_x, w_box + expand_w)
                new_h = min(image_height - new_y, h_box + expand_h)
                
                if new_w <= 0 or new_h <= 0:
                    continue

                # Crop from original frame in grayscale for emotion prediction
                face_roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]
                
                if face_roi.size == 0:
                    continue
                
                # Preprocess for CNN (48x48, convert to grayscale - already is, normalize)
                face_resized = cv2.resize(face_roi, (48, 48))
                face_normalized = face_resized / 255.0
                face_ready = np.reshape(face_normalized, (1, 48, 48, 1))

                # Predict Emotion
                predictions = self.model.predict(face_ready, verbose=0)[0]
                emotion_idx = np.argmax(predictions)
                confidence = float(np.max(predictions))
                predicted_emotion = self.emotion_labels.get(emotion_idx, "Unknown")
                
                out_results.append({
                    "emotion": predicted_emotion,
                    "confidence": confidence,
                    "box": (new_x, new_y, new_w, new_h),
                    "probabilities": predictions.tolist(),
                    "resized_frame": frame # Return the 640px frame for drawing
                })
            
            return out_results

        except Exception as e:
            logging.error(f"Error predicting frame emotion: {e}")
            return []
