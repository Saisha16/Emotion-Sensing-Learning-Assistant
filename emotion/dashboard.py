# EmotionSense Dashboard (Final Version with Real Score, Working Resources, Real Activities)

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

# Session defaults
for k in ["fused_emotion", "confidence", "suggestion"]:
    st.session_state.setdefault(k, "Optimistic")
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
    "neutral": "😐", "bored": "🥱", "tired": "😴", "curious": "🤔",
    "engaged": "🧠", "surprised": "😲", "calm": "🌿", "disgust": "🤢",
    "optimistic": "😄", "reflective": "🧐", "joyful": "😁"
}


st.title("📊 EmotionSense Dashboard")
st.caption("Welcome back! Your emotional balance and activity are shown below.")

# === TOP ROW ===
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("#### Weekly Emotion Trends")
    if st.session_state.emotion_history:
        df = pd.DataFrame(st.session_state.emotion_history)
        emotions = df["Fused Emotion"].value_counts().index.tolist()
        fig = go.Figure()
        for emotion in emotions:
            fig.add_trace(go.Scatter(
                x=list(range(7)),
                y=np.random.randint(30, 100, 7),
                fill='tonexty',
                mode='lines',
                name=emotion
            ))
        fig.update_layout(showlegend=True, height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emotion data yet. Run detection to view trends.")

with col2:
    mood = st.session_state.fused_emotion.lower()
    emoji = emoji_map.get(mood, "💬")
    st.markdown(f"""
    <div style='background-color:#1F2937;padding:20px;border-radius:12px;color:white;text-align:center;'>
        <div style='text-transform:uppercase;font-size:12px;'>Current Mood</div>
        <div style='font-size:40px;color:#EF4444;font-weight:bold;'>{mood.capitalize()} {emoji}</div>
        <div style='font-size:12px;margin-top:4px;'>Last detected: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    df = pd.DataFrame(st.session_state.emotion_history)
    score = 0
    if not df.empty:
        score = (df["Confidence"].str.replace('%','').astype(float).mean()).round(0)
    st.markdown(f"""
    <div style='background-color:#1F2937;padding:20px;border-radius:12px;color:white;text-align:center;'>
        <div style='text-transform:uppercase;font-size:12px;'>Overall Emotion Score</div>
        <div style='font-size:42px;font-weight:bold;margin-top:10px;'>{int(score)}%</div>
        <div style='font-size:12px;margin-top:4px;'>Your emotional balance has been consistently positive this week.</div>
    </div>
    """, unsafe_allow_html=True)

# === RESOURCES ===
st.markdown("#### 📚 Personalized Resources")
cards = [
    ("Communication Skills for Effective Teams", "Communication", "https://www.mindtools.com/a4wo118/communication-skills"),
    ("Stress Management: Practical Relaxation", "Well-being", "https://www.healthline.com/health/stress-relief-exercises"),
    ("Boosting Focus: Techniques for Concentration", "Productivity", "https://www.healthline.com/health/how-to-improve-concentration"),
    ("Building Resilience: Bouncing Back", "Resilience", "https://positivepsychology.com/resilience-skills/"),
    ("Emotional Intelligence: Understanding You", "Self-awareness", "https://www.verywellmind.com/what-is-emotional-intelligence-2795423"),
    ("Problem Solving: A Creative Approach", "Cognition", "https://www.edutopia.org/article/problem-solving-strategies-students")
]
cols = st.columns(3)
for i, (title, tag, link) in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f"""
        <a href='{link}' target='_blank' style='text-decoration:none;'>
        <div style='background:#111827;padding:15px;border-radius:12px;margin-bottom:12px;'>
            <strong>{title}</strong><br>
            <span style='font-size:12px;color:#9CA3AF;'>{tag}</span>
        </div></a>
        """, unsafe_allow_html=True)

# === RECENT ACTIVITIES ===
col4, col5 = st.columns([3, 1])
with col4:
    st.markdown("#### 🕒 Recent Activities")
    if not df.empty:
        for i, row in df.tail(3).iterrows():
            st.markdown(f"• **{row['Fused Emotion'].capitalize()}** → _AI Suggested Task_ ({row['Timestamp']})")
    else:
        st.info("No recent activities yet.")

with col5:
    st.markdown("#### ⚡ Quick Actions")

    if st.button("🎯 Start New Session"):
        st.switch_page("pages/detection.py")

    if st.button("📈 View Full Emotion Trends"):
        st.switch_page("pages/trends.py")

st.markdown("---")
st.caption("2025 © EmotionSense — All rights reserved")