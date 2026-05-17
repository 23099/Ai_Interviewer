# ==========================================
# AI INTERVIEWER (DEEPFACE + SVM FULL SYSTEM)
# ==========================================

import cv2
import pyttsx3
import speech_recognition as sr
import time
import joblib
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from deepface import DeepFace

# ================================
# TRAIN SVM MODEL (if not exists)
# ================================
if not os.path.exists("svm_model.pkl"):
    print("Training SVM model...")

    data = [
        ("I am hardworking and passionate", 1),
        ("I have strong problem solving skills", 1),
        ("I am very motivated and disciplined", 1),
        ("I am a quick learner and team player", 1),
        ("I don't know", 0),
        ("nothing", 0),
        ("no idea", 0),
        ("I just need a job", 0)
    ]

    texts = [x[0] for x in data]
    labels = [x[1] for x in data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = SVC(kernel='linear', probability=True)
    model.fit(X, labels)

    joblib.dump(model, "svm_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================================
# TEXT TO SPEECH
# ================================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

# ================================
# SPEECH TO TEXT
# ================================
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print("User:", text)
            return text
        except:
            return ""

# ================================
# SVM EVALUATION
# ================================
def evaluate_answer(answer):
    X = vectorizer.transform([answer])
    pred = model.predict(X)[0]
    return (1, "Good answer") if pred == 1 else (0, "Weak answer")

# ================================
# STRESS CALCULATION
# ================================
def compute_stress(emotion, eye_contact, silence_time):
    stress = 0

    if emotion in ["angry", "fear", "sad"]:
        stress += 40
    elif emotion == "neutral":
        stress += 20
    else:
        stress += 10

    if eye_contact < 0.5:
        stress += 30
    else:
        stress += 10

    if silence_time > 3:
        stress += 30

    return min(stress, 100)

# ================================
# CAMERA (DEEPFACE)
# ================================
def start_camera(duration=5):

    cap = cv2.VideoCapture(0)
    start = time.time()

    eye_contact_score = 0.7
    emotion_label = "neutral"

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # =========================
        # EMOTION DETECTION (DeepFace)
        # =========================
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )
            emotion_label = result[0]['dominant_emotion']
        except:
            emotion_label = "neutral"

        # =========================
        # FAKE EYE CONTACT (CENTER)
        # =========================
        h, w, _ = frame.shape
        center_x = w / 2

        face_center = w / 2  # simple assumption

        if abs(face_center - center_x) < 100:
            eye_contact_score += 0.01
        else:
            eye_contact_score -= 0.01

        eye_contact_score = np.clip(eye_contact_score, 0, 1)

        # =========================
        # STRESS
        # =========================
        stress = compute_stress(emotion_label, eye_contact_score, 2)

        # =========================
        # DISPLAY
        # =========================
        cv2.putText(frame, f"Emotion: {emotion_label}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Stress: {stress}%", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Eye Contact: {eye_contact_score:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("AI Interviewer - DeepFace", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# QUESTIONS
# ================================
questions = [
    "Tell me about yourself",
    "What are your strengths",
    "What are your weaknesses",
    "Why should we hire you",
    "Where do you see yourself in five years"
]

# ================================
# MAIN INTERVIEW
# ================================
def run_interview():

    speak("Welcome to AI Interviewer")
    speak("Your interview is starting now")

    scores = []

    for q in questions:
        speak(q)

        answer = listen()

        if answer == "":
            speak("No response detected")
            scores.append(0)
            continue

        score, feedback = evaluate_answer(answer)
        speak(feedback)
        scores.append(score)

    final_score = sum(scores) / len(scores)

    speak(f"Your final score is {final_score * 100} percent")

    if final_score > 0.6:
        speak("You performed well")
    else:
        speak("You need improvement")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    start_camera()
    run_interview()