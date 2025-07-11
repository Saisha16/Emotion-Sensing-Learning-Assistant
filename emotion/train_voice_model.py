'''
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import soundfile as sf

# Path to RAVDESS
DATA_PATH = "D:/emotion_assistant/data/archive/"

# Extract features (MFCCs)
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error:", file_path, e)
        return None

# Load data
features = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion = int(file.split("-")[2])
            emotions_map = {
                1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
            }
            label = emotions_map.get(emotion)
            feature = extract_features(os.path.join(root, file))
            if feature is not None:
                features.append(feature)
                labels.append(label)

X = np.array(features)
y = to_categorical(LabelEncoder().fit_transform(labels))

X = X.reshape(X.shape[0], -1)  # Flatten

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Dense(256, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("voice_emotion_model.h5")
print("✅ Voice Emotion Model trained and saved as voice_emotion_model.h5")
'''
import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR = 'data/archive'
OUTPUT_DIR = 'emotion_recognition'
EMOTIONS = 8  # RAVDESS has 8 emotions (0 to 7)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def load_data():
    X, y = [], []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".wav"):
                try:
                    emotion = int(file.split("-")[2]) - 1
                    feat = extract_features(os.path.join(root, file))
                    if feat is not None:
                        X.append(feat)
                        y.append(emotion)
                except:
                    pass
    return np.array(X), np.array(y)

def build_model(input_dim=13, output_dim=8):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ Accuracy: {acc*100:.2f}%")

    model.save(os.path.join(OUTPUT_DIR, "voice_model_keras.h5"))
    print("✅ Keras model saved.")

if __name__ == "__main__":
    main()
