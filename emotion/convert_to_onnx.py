import keras
import tf2onnx
import onnx
import numpy as np
import tensorflow as tf

# Load your Keras model
model = keras.models.load_model("emotion_recognition/voice_model_keras.h5")

# Create dummy input with the correct shape
dummy_input = tf.TensorSpec([None, 13], tf.float32, name="input")

# Convert using tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[dummy_input],
    opset=13,
    output_path="emotion_recognition/voice_emotion_model.onnx"
)

print("✅ Keras model successfully converted to ONNX.")
