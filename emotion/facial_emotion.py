from deepface import DeepFace
import cv2

def get_face_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "unknown"

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(e)
        return "error"
    
'''from deepface import DeepFace
import cv2

def get_face_emotion_continuous():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            yield "unknown"
            continue

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            print(f"Face detection error: {e}")
            emotion = "error"

        yield emotion

        # OPTIONAL: Show frame (can comment this out if not needed)
        # cv2.putText(frame, f"Emotion: {emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow("Facial Emotion Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    '''

