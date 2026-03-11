from deepface import DeepFace
# test what build_model("Emotion") actually returns and predict

emotion_model = DeepFace.build_model("Emotion")
print("Emotion model loaded.")
