def fuse_emotions(face_emotion, voice_emotion):
    # Priority order or fusion rules
    priority = {
        "happy": 1,
        "engaged": 2,
        "curious": 3,
        "neutral": 4,
        "sad": 5,
        "bored": 6,
        "tired": 7,
        "angry": 8,
        "fearful": 9
    }

    # If same, return it
    if face_emotion == voice_emotion:
        return face_emotion

    # If either is neutral, prefer the non-neutral one
    if face_emotion == "neutral":
        return voice_emotion
    if voice_emotion == "neutral":
        return face_emotion

    # Else, pick the one with higher priority (lower number is higher priority)
    face_score = priority.get(face_emotion, 10)
    voice_score = priority.get(voice_emotion, 10)

    return face_emotion if face_score < voice_score else voice_emotion


