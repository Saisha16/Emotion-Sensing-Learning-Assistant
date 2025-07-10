# detection.py – Emotion Detection with Corrected Fusion and Expanded Emotions

import streamlit as st
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from utils.genai import get_genai_suggestion
from openvino_face import predict_emotion_openvino
from openvino_voice import predict_voice_emotion
from fuse import fuse_emotions  # Corrected fusion logic

st.set_page_config(page_title="Live Emotion Detection", layout="wide")
AUDIO_PATH = "temp/audio.wav"

# Init session state
for k in ["face_emotion", "voice_emotion", "fused_emotion", "confidence", "suggestion"]:
    st.session_state.setdefault(k, "Unknown")
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# Record Audio
@st.cache_data(show_spinner=False)
def get_ai_suggestion(emotion):
    return get_genai_suggestion(emotion)

def record_audio(duration=3, fs=44100):
    st.info("🎤 Recording 3 seconds of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)
    write(AUDIO_PATH, fs, audio)
    st.success("✅ Audio recorded")

# Emoji mapping
emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
    "neutral": "😐", "bored": "🥱", "tired": "😴", "curious": "🤔",
    "engaged": "🧠", "surprised": "😲", "calm": "🌿", "disgust": "🤢",
    "optimistic": "😄", "reflective": "🧐", "joyful": "😁"
}

# UI
st.title("🎯 Live Emotion Detection")


col1, col2 = st.columns([3, 2])
with col1:
    image_input = st.camera_input("📸 Take a photo")

    if st.button("▶️ Start Detection"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        face_confidence = 0

        # Face detection
        if image_input:
            try:
                img = Image.open(image_input)
                img_np = np.array(img.convert("RGB"))
                face_emotion, face_confidence, _ = predict_emotion_openvino(img_np)
                st.session_state.face_emotion = face_emotion
            except Exception as e:
                st.session_state.face_emotion = "unknown"
                st.error(f"Face detection failed: {e}")
        else:
            st.warning("📸 Please take a photo first.")

        # Voice detection
        try:
            record_audio()
            voice_emotion, voice_conf = predict_voice_emotion(AUDIO_PATH)
            if voice_emotion == "calm":  # Optional: ignore calm
                voice_emotion = "neutral"
            st.session_state.voice_emotion = voice_emotion
        except Exception as e:
            st.session_state.voice_emotion = "unknown"
            st.error(f"Voice detection failed: {e}")

        # Fuse logic
        fused = fuse_emotions(st.session_state.face_emotion, st.session_state.voice_emotion)
        st.session_state.fused_emotion = fused
        st.session_state.confidence = f"{face_confidence:.0f}%"
        st.session_state.suggestion = get_ai_suggestion(fused)

        st.session_state.emotion_history.append({
            "Timestamp": timestamp,
            "Face Emotion": st.session_state.face_emotion,
            "Voice Emotion": st.session_state.voice_emotion,
            "Fused Emotion": fused,
            "Confidence": st.session_state.confidence,
            "Suggestion": st.session_state.suggestion
        })

        # Logs
        print("🧠 Face:", st.session_state.face_emotion)
        print("🎤 Voice:", st.session_state.voice_emotion)
        print("🌀 Fused:", st.session_state.fused_emotion)

with col2:
    emotion = st.session_state.fused_emotion
    emoji = emoji_map.get(emotion.lower(), "💬")

    st.markdown(f"""
    <div style="background:#1F2937;padding:20px;border-radius:12px;color:white;text-align:center;">
        <div style="font-size:14px;text-transform:uppercase;">Detected Emotion</div>
        <div style="font-size:42px;font-weight:bold;margin-top:10px;color:#EF4444;">{emotion.capitalize()} {emoji}</div>
        <div style="margin-top:10px;"><b>Confidence:</b> {st.session_state.confidence}</div>
        <hr>
        <div><b>AI Suggestion:</b><br>{st.session_state.suggestion}</div>
    </div>
    """, unsafe_allow_html=True)

# Emotion History
st.markdown("## 📊 Emotion History")
if st.session_state.emotion_history:
    df_history = pd.DataFrame(st.session_state.emotion_history)
    st.dataframe(df_history, use_container_width=True)

    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download History as CSV", data=csv, file_name="emotion_history.csv", mime="text/csv")

    fig = go.Figure()
    for emotion in df_history["Fused Emotion"].unique():
        emo_df = df_history[df_history["Fused Emotion"] == emotion]
        fig.add_trace(go.Scatter(x=emo_df["Timestamp"], y=[emotion]*len(emo_df), mode="markers", name=emotion))
    fig.update_layout(title="Detected Emotions Timeline", yaxis_title="Fused Emotion")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No emotion data yet. Run detection to generate history.")

st.markdown("---")
col3, col4 = st.columns([3, 1])
with col3:
    st.markdown("### 📨 Stay up to date with EmotionSense")
    email = st.text_input("Enter your email")
    if st.button("Subscribe"):
        st.success("Subscribed successfully!")
with col4:
    st.image("https://img.icons8.com/color/96/brain.png", width=64)
    st.markdown("2025 © EmotionSense")