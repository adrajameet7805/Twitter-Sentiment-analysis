from deepface import DeepFace
import numpy as np

emotion_model = DeepFace.build_model("Emotion")
print("Input shape:", emotion_model.input_shape)
print("Output shape:", emotion_model.output_shape)
