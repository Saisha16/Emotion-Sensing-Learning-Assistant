# 🎓 Emotion-Sensing Learning Assistant (ESLA)

An AI-powered web application that detects a student's **emotion** using real-time **facial expressions** and **voice input**, then recommends **engaging activities** and **YouTube videos** to help teachers tailor the learning experience.

---

##  Project Overview

**ESLA** is designed to make digital classrooms more emotionally aware. It:
- Detects student emotions in real time using webcam and microphone.
- Suggests activities through OpenAI based on emotion (e.g., games, questions, discussions).
- Recommends matching videos from YouTube to boost engagement.
- Logs and visualizes emotion trends for teachers using dashboards.

This project was developed as part of **Intel Unnati Training 2025** to showcase the impact of **AI in education**.

---

##  Features
- 🎥 **Facial Emotion Detection** (via webcam)
- 🎙️ **Voice Emotion Recognition** (via mic input)
- 🔗 **Activity Suggestions** (via OpenAI API)
- 📺 **YouTube Video Recommender**
- 📊 **Live Dashboard** with emotion logs and trend graphs
- 🖥️ **Streamlit Frontend** with custom CSS

---

## 🛠️ Tech Stack
| Layer             | Tools & Libraries                      |
|------------------|----------------------------------------|
| Programming      | Python                                 |
| Emotion Detection| DeepFace, OpenVINO, TensorFlow          |
| Audio Processing | SpeechRecognition, PyAudio, librosa     |
| ML Conversion    | ONNX, Keras                             |
| Frontend         | Streamlit, HTML, CSS                    |
| Visualization    | Matplotlib, Streamlit Charts            |
| AI Services      | OpenAI GPT API, YouTube Search API      |

---

## 📂 Folder Structure
├── main.py                 # Main app logic and routing
├── facial_emotion.py       # Facial emotion detection
├── voice_emotion.py        # Voice emotion recognition
├── fuse.py                 # Combine face + voice predictions
├── webcam_emotion_live.py  # Webcam input and live emotion detection
├── mic_record.py           # Voice input and audio processing
├── learning.py             # OpenAI-based adaptive learning suggestions
├── dashboard.py            # Emotion graphs and trend visualization
├── requirements.txt        # Python dependencies
├── models/                 # Pre-trained and converted models
├── screenshots/            # App screenshots (optional)
└── assets/                 # Icons, stylesheets, and static assets


---

## 🚀 How to Run
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Saisha16/Emotion-Sensing-Learning-Assistant.git
cd Emotion-Sensing-Learning-Assistant
2. Install Requirements
Make sure Python 3.8+ is installed.

bash
Copy
Edit
pip install -r requirements.txt
3. Run the App
bash
Copy
Edit
streamlit run main.py
✅ Make sure your microphone and webcam are enabled when prompted.
🔍 Emotion Categories
Happy 😊

Sad 😢

Angry 😠

Neutral 😐

Fearful 😨

Surprised 😲

📈 Dashboard Preview
Pie chart of emotion distribution

Line graph showing emotion trends

Logs of past detections with timestamp

Note: Video is in EmotionSensingVideo Folder
🔮 Future Enhancements
Group emotion analysis

Multilingual voice support

LMS integration (Google Classroom, Moodle)

Deploy as Dockerized web app

Fine-tuned GPT prompts for subject-specific activities

📜 License
MIT License © 2025 I²nsane Coders

👥 Contributors
Saisha16
Ishta13
I²nsane Coders

