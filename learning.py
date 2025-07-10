import streamlit as st
import plotly.express as px
from utils.genai import get_genai_suggestion

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Learning - EmotionSense")



# --- Get Detected Emotion ---
fused_emotion = st.session_state.get("fused_emotion", "Neutral").capitalize()

# --- Resource Data by Emotion ---
resources = {
    "Angry": {
        "type": "video",
        "title": "How to Control Anger – Relaxation Techniques",
        "desc": "Learn quick tips to calm down and manage anger effectively.",
        "link": "https://youtu.be/wkse4PPxkk4",
        "video": "https://youtu.be/wkse4PPxkk4"
    },
    "Happy": {
        "type": "video",
        "title": "Mood-Boosting Exercises",
        "desc": "Try these simple exercises to maintain your joy.",
        "link": "https://youtu.be/naPiQIImGzk",
        "video": "https://youtu.be/naPiQIImGzk"
    },
    "Sad": {
        "type": "video",
        "title": "Simple Techniques to Feel Better When You're Sad",
        "desc": "Watch this comforting guide to improve your mood gently.",
        "link": "https://youtu.be/PEgnPB01LPY",
        "video": "https://youtu.be/PEgnPB01LPY"
    },
    "Neutral": {
        "type": "video",
        "title": "The Power of Routine",
        "desc": "Establish structure and balance for a neutral mindset.",
        "link": "https://youtu.be/CqgmozFr_GM",
        "video": "https://youtu.be/CqgmozFr_GM"
    },
    "Surprised": {
        "type": "video",
        "title": "Why Surprise Is Good for Your Brain",
        "desc": "Understand the science of surprise and how it helps learning.",
        "link": "https://youtu.be/Hm0y-SuqvA0",
        "video": "https://youtu.be/Hm0y-SuqvA0"
    },
    "Fearful": {
        "type": "video",
        "title": "How to Face Your Fears",
        "desc": "Practical methods to confront fear and reduce anxiety.",
        "link": "https://youtu.be/ukNC17sA3ME",
        "video": "https://youtu.be/ukNC17sA3ME"
    },
    "Disgusted": {
        "type": "video",
        "title": "Why Do We Feel Disgust?",
        "desc": "A psychological look at the emotion of disgust and its role in survival.",
        "link": "https://youtu.be/vbjlDuDtjUg",
        "video": "https://youtu.be/vbjlDuDtjUg"
    },
    "Curious": {
        "type": "video",
        "title": "How Curiosity Changes Your Brain",
        "desc": "Explore how being curious boosts your learning and creativity.",
        "link": "https://youtu.be/wl0ElFvPyiw",
        "video": "https://youtu.be/wl0ElFvPyiw"
    },
    "Bored": {
        "type": "quiz",
        "title": "Fun Quiz to Break Boredom",
        "desc": "Challenge yourself with trivia to beat boredom!",
        "link": "https://www.proprofs.com/quiz-school/story.php?title=bored-lets-do-this",
        "img": "https://source.unsplash.com/800x600/?boredom,fun"
    }
}

# --- Get Matching Resource ---
resource = resources.get(fused_emotion, resources["Neutral"])

# --- Display Resource ---
st.markdown("## 🧠 Personalized Resource Based on Your Mood")
st.success(f"Detected Mood: **{fused_emotion}**")

if resource["type"] == "video" and "video" in resource:
    st.video(resource["video"])
else:
    st.image(resource.get("img", "https://source.unsplash.com/800x600/?learning"), use_container_width=True)

st.markdown(f"**{resource['title']}**  \n{resource['desc']}")
st.link_button("Open Resource", url=resource["link"])

# --- Simulate Mood Dropdown ---
st.markdown("### 🌀 Simulate Mood")
mood_options = list(resources.keys())
selected_index = mood_options.index(fused_emotion) if fused_emotion in mood_options else mood_options.index("Neutral")
st.selectbox("Change Mood", mood_options, index=selected_index)

# --- Emotion Trend Chart ---
st.markdown("### 📊 Recent Emotion Trends")
trend_data = {
    "Emotion": ["Happy", "Neutral", "Angry", "Curious", "Sad", "Bored", "Fearful"],
    "Score": [78, 62, 40, 85, 52, 45, 50]
}
fig = px.bar(trend_data, x="Emotion", y="Score", color="Emotion", title="Recent Emotion Trends")
st.plotly_chart(fig, use_container_width=True)

# --- AI Suggestion Block ---
st.markdown("---")
st.markdown("### 💡 AI-Based Activity Suggestion")
suggestion = get_genai_suggestion(fused_emotion)
st.markdown(f"""
<div style='background-color:#1F2937; color:white; padding:20px; border-radius:12px; margin-top:10px;'>
    <h4 style='color:#FBBF24;'>Activity Suggestion</h4>
    <p style='font-size:16px; color:#D1D5DB;'>{suggestion}</p>
</div>
""", unsafe_allow_html=True)

# --- Tips Section ---
st.markdown("---")
st.markdown("### 💡 Tips for You")

with st.expander("😵 Feeling Overwhelmed?"):
    st.write("Take a short, guided breathing exercise to calm your mind.")

with st.expander("🧠 Boost Your Brainpower"):
    st.write("Engage in quick mental exercises to sharpen your cognitive skills.")

with st.expander("⏱ Time Management Tip"):
    st.write("Use the Pomodoro Technique for focused work sessions and regular breaks.")
