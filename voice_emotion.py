from openvino.runtime import Core
import numpy as np
import librosa
import os

# Load the OpenVINO model
core = Core()
model_path = os.path.join(os.path.dirname(__file__), "voice_emotion_model.onnx")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"ONNX model not found at: {model_path}")

model_ov = core.read_model(model_path)
compiled_model = core.compile_model(model_ov, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Define emotion classes (same order as your training dataset)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def predict_voice_emotion(file_path="temp/live.wav"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load audio and extract MFCC features
    y, sr = librosa.load(file_path, sr=None, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1).astype(np.float32)

    # Perform inference
    result = compiled_model([features])[output_layer]
    label_index = int(np.argmax(result))

    return emotion_labels[label_index]
