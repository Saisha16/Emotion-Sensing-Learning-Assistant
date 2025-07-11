import os
import numpy as np
import cv2
from PIL import Image
from openvino.runtime import Core
from deepface import DeepFace

# Emotion labels as per OpenVINO model
openvino_emotion_labels = ["neutral", "happy", "sad", "surprise", "anger"]

# Load OpenVINO model
model_path = os.path.join(
    os.path.dirname(__file__),
    "intel", "emotions-recognition-retail-0003", "FP16", "emotions-recognition-retail-0003.xml"
)
core = Core()
compiled_model = core.compile_model(model_path, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def fallback_deepface_emotion(frame_np):
    """Fallback to DeepFace when OpenVINO output is 'neutral' or uncertain."""
    try:
        result = DeepFace.analyze(frame_np, actions=["emotion"], enforce_detection=False)
        emotion_scores = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"].lower()
        confidence = round(emotion_scores[dominant], 2)
        return dominant, confidence, emotion_scores
    except Exception as e:
        print("❌ DeepFace failed:", e)
        return "unknown", 0.0, {}

def predict_emotion_openvino(frame_np):
    """Predict emotion from image using OpenVINO, fallback to DeepFace if low confidence."""
    debug_path = os.path.join(os.path.dirname(__file__), "debug_face.jpg")
    Image.fromarray(frame_np).save(debug_path)

    # Preprocess image
    img = cv2.resize(frame_np, (64, 64))
    img = img.transpose((2, 0, 1)).reshape(1, 3, 64, 64).astype(np.float32)
    img /= 255.0

    # Inference
    output = compiled_model([img])[output_layer]
    probabilities = output[0]
    idx = int(np.argmax(probabilities))
    confidence = round(float(probabilities[idx]) * 100, 2)
    predicted_emotion = openvino_emotion_labels[idx]

    print(f"🧠 OpenVINO: {predicted_emotion} ({confidence}%)")

    # Fallback if neutral or low confidence
    if predicted_emotion == "neutral" or confidence < 40:
        print("⚠️ Using DeepFace fallback...")
        fallback_emotion, fallback_conf, fallback_scores = fallback_deepface_emotion(frame_np)
        print(f"🔁 DeepFace: {fallback_emotion} ({fallback_conf}%)")
        return fallback_emotion, fallback_conf, fallback_scores

    # Convert OpenVINO scores to % dict
    score_dict = {
        openvino_emotion_labels[i]: round(probabilities[i] * 100, 2)
        for i in range(len(openvino_emotion_labels))
    }

    return predicted_emotion, confidence, score_dict
