'''from emotion_recognition.facial_emotion import get_face_emotion
from emotion_recognition.voice_emotion import predict_voice_emotion
from emotion_recognition.fuse import fuse_emotions
from utils.features import decide_action
import pandas as pd
from datetime import datetime



# Get emotions
face_emotion = get_face_emotion()
voice_emotion = predict_voice_emotion()

# Fuse and decide
fused_emotion = fuse_emotions(face_emotion, voice_emotion)
action = decide_action(fused_emotion)

# Log result
log = {
    "timestamp": [datetime.now()],
    "face_emotion": [face_emotion],
    "voice_emotion": [voice_emotion],
    "fused_emotion": [fused_emotion],
    "action": [action]
}
df = pd.DataFrame(log)
df.to_csv("logs/logs.csv", mode='a', header=False, index=False)

print(f"Face: {face_emotion}, Voice: {voice_emotion}, Fused: {fused_emotion}")
print(f"Suggested Action: {action}")
'''
'''import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from deepface import DeepFace
import librosa
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os

# ---------------------------
# VOICE EMOTION RECOGNITION
# ---------------------------
def record_voice(file_path="audio.wav", duration=5, fs=44100):
    print("Recording voice...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write(file_path, fs, audio)
    print("Voice recorded and saved.")
    return file_path

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def predict_voice_emotion(file_path, model_path="voice_emotion_model.pkl"):
    features = extract_features(file_path).reshape(1, -1)
    model = joblib.load(model_path)
    return model.predict(features)[0]

# ---------------------------
# FACIAL EMOTION RECOGNITION
# ---------------------------
def capture_face_emotion():
    cap = cv2.VideoCapture(0)
    emotion = "unknown"
    print("Capturing facial emotion...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Emotion detection error:", e)

        cv2.imshow('Facial Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion

# ---------------------------
# FUSION AND LOGIC
# ---------------------------
def fuse_emotions(face_emotion, voice_emotion):
    if face_emotion == voice_emotion:
        return face_emotion
    elif 'bored' in [face_emotion, voice_emotion]:
        return 'bored'
    else:
        return face_emotion

def decide_action(emotion):
    if emotion in ['bored', 'neutral']:
        return "Add interactive quiz"
    elif emotion == 'confused':
        return "Repeat topic with examples"
    elif emotion == 'happy':
        return "Advance to next topic"
    elif emotion == 'angry':
        return "Pause and address issues"
    else:
        return "Continue teaching"

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    # Step 1: Facial Emotion
    face_emotion = capture_face_emotion()

    # Step 2: Voice Emotion
    audio_path = record_voice()
    voice_emotion = predict_voice_emotion(audio_path)

    # Step 3: Fuse Emotions
    fused_emotion = fuse_emotions(face_emotion, voice_emotion)
    action = decide_action(fused_emotion)

    # Step 4: Print Results
    print(f"\nFace: {face_emotion}, Voice: {voice_emotion}, Fused: {fused_emotion}")
    print(f"Suggested Action: {action}")

    # Step 5: Log Results
    log = {
        "timestamp": [datetime.now()],
        "face_emotion": [face_emotion],
        "voice_emotion": [voice_emotion],
        "fused_emotion": [fused_emotion],
        "action": [action]
    }
    df = pd.DataFrame(log)
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/logs.csv", mode='a', header=not os.path.exists("logs/logs.csv"), index=False)

if __name__ == "__main__":
    main()
'''
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from deepface import DeepFace
import librosa
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os
import time



# ---------------------------
# VOICE EMOTION RECOGNITION
# ---------------------------
def record_voice(file_path="audio.wav", duration=5, fs=44100):
    print("Recording voice...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write(file_path, fs, audio)
    print("Voice recorded and saved.")
    return file_path

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def predict_voice_emotion(file_path, model_path="voice_emotion_model.pkl"):
    if not os.path.exists(model_path):
        from emotion_recognition.voice_emotion import train_model
        print("Model not found. Training voice emotion model...")
        train_model()
    features = extract_features(file_path).reshape(1, -1)
    model = joblib.load(model_path)
    return model.predict(features)[0]

# ---------------------------
# FACIAL EMOTION RECOGNITION
# ---------------------------
def capture_face_emotion():
    cap = cv2.VideoCapture(0)
    emotion = "unknown"
    print("Capturing facial emotion...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Emotion detection error:", e)

        cv2.imshow('Facial Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion

# ---------------------------
# FUSION AND LOGIC
# ---------------------------
def fuse_emotions(face_emotion, voice_emotion):
    if face_emotion == voice_emotion:
        return face_emotion
    elif 'bored' in [face_emotion, voice_emotion]:
        return 'bored'
    else:
        return face_emotion

def decide_action(emotion):
    if emotion in ['bored', 'neutral']:
        return "Add interactive quiz"
    elif emotion == 'confused':
        return "Repeat topic with examples"
    elif emotion == 'happy':
        return "Advance to next topic"
    elif emotion == 'angry':
        return "Pause and address issues"
    else:
        return "Continue teaching"
    
def train_model():
    import os
    import numpy as np
    import librosa
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    data_dir = "D:/emotion_assistant/data/archive"
    X, y = [], []

    print("Looking for files in:", data_dir)
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                try:
                    emotion = int(file.split("-")[2])  # RAVDESS emotion code
                    emotion_label = {
                        1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry",
                        6: "fearful", 7: "disgust", 8: "surprised"
                    }.get(emotion, "unknown")
                    path = os.path.join(root, file)
                    y_, sr = librosa.load(path)
                    mfcc = librosa.feature.mfcc(y=y_, sr=sr, n_mfcc=40)
                    X.append(np.mean(mfcc.T, axis=0))
                    y.append(emotion_label)
                except Exception as e:
                    print("Error processing file", file, e)

    print(f"Loaded {len(X)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Voice Emotion Model trained with accuracy: {acc * 100:.2f}%")

    # 🔥 Ensure model is saved where main.py expects it
    joblib.dump(model, "voice_emotion_model.pkl")


# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    # Step 1: Facial Emotion
    face_emotion = capture_face_emotion()

    # Step 2: Voice Emotion
    audio_path = record_voice()
    voice_emotion = predict_voice_emotion(audio_path)

    # Step 3: Fuse Emotions
    fused_emotion = fuse_emotions(face_emotion, voice_emotion)
    action = decide_action(fused_emotion)

    # Step 4: Print Results
    print(f"\nFace: {face_emotion}, Voice: {voice_emotion}, Fused: {fused_emotion}")
    print(f"Suggested Action: {action}")

    # Step 5: Log Results
    log = {
        "timestamp": [datetime.now()],
        "face_emotion": [face_emotion],
        "voice_emotion": [voice_emotion],
        "fused_emotion": [fused_emotion],
        "action": [action]
    }
    df = pd.DataFrame(log)
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/logs.csv", mode='a', header=not os.path.exists("logs/logs.csv"), index=False)

if __name__ == "__main__":
    main()
start = time.time()
emotion = predict_voice_emotion_openvino("temp/live.wav")
end = time.time()
print(f"Inference Time: {end - start:.4f} seconds")