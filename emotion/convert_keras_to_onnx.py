import keras
import tf2onnx
import tensorflow as tf

# Load the trained Sequential model
sequential_model = keras.models.load_model("emotion_recognition/voice_model_keras.h5")

# Wrap into Functional model to avoid output_names error
inputs = keras.Input(shape=(13,), name="input")
outputs = sequential_model(inputs)
model = keras.Model(inputs, outputs)

# Define input signature
input_signature = [tf.TensorSpec([None, 13], tf.float32, name="input")]

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path="emotion_recognition/voice_emotion_model.onnx"
)

print("✅ Sequential model wrapped and converted to ONNX successfully.")
