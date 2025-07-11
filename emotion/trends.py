# EmotionSense Trends Page (Fully Functional & Visily-style)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Emotion Trends", layout="wide")
st.title("📈 Emotion Trends Over Time")

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# Convert to DataFrame
if st.session_state.emotion_history:
    df = pd.DataFrame(st.session_state.emotion_history)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
else:
    st.warning("No emotion history to show trends.")
    st.stop()

# --- Time Filter ---
st.sidebar.header("🕒 Time Filter")
filter_option = st.sidebar.selectbox("Select Time Range", ["Last 7 Days", "Last 30 Days", "All"])

if filter_option == "Last 7 Days":
    cutoff = datetime.now() - timedelta(days=7)
    df = df[df["Timestamp"] >= cutoff]
elif filter_option == "Last 30 Days":
    cutoff = datetime.now() - timedelta(days=30)
    df = df[df["Timestamp"] >= cutoff]

# --- Bar Chart ---
st.markdown("### 📊 Emotion Trends Over Time")
trend_fig = px.histogram(
    df,
    x=df["Timestamp"].dt.date,
    color="Fused Emotion",
    title="Distribution of Detected Emotions",
    labels={"x": "Date"},
    nbins=len(df["Timestamp"].dt.date.unique()),
    barmode="group"
)
trend_fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Count")
st.plotly_chart(trend_fig, use_container_width=True)

# --- Detailed Emotion Logs ---
st.markdown("### 🗂️ Detailed Emotion Log")
log_df = df[["Timestamp", "Face Emotion", "Voice Emotion", "Fused Emotion", "Confidence"]].sort_values(by="Timestamp", ascending=False)
st.dataframe(log_df, height=300, use_container_width=True)

# --- Trend Overview ---
st.markdown("### 📌 Trend Overview")
col1, col2, col3 = st.columns(3)

# Score
with col1:
    score = df["Confidence"].str.replace('%', '').astype(float).mean().round(0)
    st.metric("Emotion Score", f"{int(score)}/100", "Positive")

# Top Emotion
with col2:
    top_emotion = df["Fused Emotion"].mode()[0]
    st.metric("Most Frequent Emotion", top_emotion.capitalize())

# Insight
with col3:
    insight = "Positive emotional pattern observed." if score >= 65 else "Neutral or mixed emotion trend."
    st.metric("Emotional Insight", insight)

# --- Pagination Controls (optional future) ---
# st.button("⬅ Previous")
# st.button("Next ➡")

st.markdown("---")
st.caption("2025 © EmotionSense • Emotion trend visualizer")