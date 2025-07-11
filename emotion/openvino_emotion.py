import os
from openvino.runtime import Core
import numpy as np
import cv2

# List of emotions this model supports
emotion_labels = ["neutral", "happy", "sad", "surprise", "anger"]

# Dynamically construct the absolute path to the .xml model
model_path = os.path.join(
    os.path.dirname(__file__),
    "intel", "emotions-recognition-retail-0003", "FP16", "emotions-recognition-retail-0003.xml"
)

# Initialize OpenVINO Core and load the model
core = Core()
compiled_model = core.compile_model(model_path, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def predict_emotion_openvino(frame):
    """
    Predict the emotion from a given image frame using OpenVINO.

    Returns:
        emotion_label (str): Predicted emotion label like 'happy'
        confidence (float): Confidence percentage (0–100)
    """
    # Resize and normalize the input frame
    img = cv2.resize(frame, (64, 64))
    img = img.transpose((2, 0, 1))  # Convert HWC to CHW
    img = img.reshape(1, 3, 64, 64).astype(np.float32)
    img /= 255.0

    # Run inference
    output = compiled_model([img])[output_layer]
    top_idx = int(np.argmax(output))
    confidence = round(float(output[0][top_idx]) * 100, 2)

    # ✅ Return label instead of index
    return emotion_labels[top_idx], confidence
