
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emotion_recognition.facial_emotion import get_face_emotion_continuous

import threading
import pandas as pd
from datetime import datetime
from emotion_recognition.facial_emotion import get_face_emotion_continuous  # Modified function, see note
from emotion_recognition.voice_emotion import predict_voice_emotion_continuous  # Modified function, see note
from emotion_recognition.fuse import fuse_emotions
from utils.features import decide_action

import queue
import time

# Queues to hold the latest emotions from each source
face_queue = queue.Queue(maxsize=1)
voice_queue = queue.Queue(maxsize=1)

def run_facial_emotion():
    for face_emotion in get_face_emotion_continuous():
        # Put latest face emotion in queue (overwrite old if present)
        if face_queue.full():
            try:
                face_queue.get_nowait()
            except queue.Empty:
                pass
        face_queue.put(face_emotion)

def run_voice_emotion():
    for voice_emotion in predict_voice_emotion_continuous():
        # Put latest voice emotion in queue (overwrite old if present)
        if voice_queue.full():
            try:
                voice_queue.get_nowait()
            except queue.Empty:
                pass
        voice_queue.put(voice_emotion)

def main_loop():
    while True:
        # Try to get latest emotions, or fallback to None
        try:
            face_emotion = face_queue.get(timeout=1)
        except queue.Empty:
            face_emotion = None

        try:
            voice_emotion = voice_queue.get(timeout=1)
        except queue.Empty:
            voice_emotion = None

        if face_emotion and voice_emotion:
            fused_emotion = fuse_emotions(face_emotion, voice_emotion)
            action = decide_action(fused_emotion)

            # Log to CSV
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

        time.sleep(1)  # adjust as needed

if __name__ == "__main__":
    # Run face and voice emotion detection in background threads
    face_thread = threading.Thread(target=run_facial_emotion, daemon=True)
    voice_thread = threading.Thread(target=run_voice_emotion, daemon=True)
    face_thread.start()
    voice_thread.start()

    # Run main fusion and logging loop in main thread
    main_loop()
