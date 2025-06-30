import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
import dlib
# import pygame  # Temporarily disabled
import joblib
from scipy.spatial import distance
from deepface import DeepFace

# === Register mixed precision Policy for deserialization ===
from tensorflow.keras.mixed_precision import Policy

@register_keras_serializable()
class MixedPolicy(Policy):
    pass

tf.keras.utils.get_custom_objects()['Policy'] = MixedPolicy
tf.keras.utils.get_custom_objects()['Adam'] = Adam

# === Load Models ===
cnn_model = load_model("final_drowsiness_model_elu_balanced.h5")
rf_model = joblib.load("drowsiness_ml_model.pkl")

# === Setup Dlib for face detection and landmarks ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === Alarm Sound (Temporarily Disabled) ===
alarm_sound = None
sound_enabled = False
# import pygame
# pygame.mixer.init()
# alarm_sound = "alarm.wav"
# sound_enabled = True

# === EAR/MAR thresholds ===
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
drowsy_frames = 0
yawning_frames = 0

# === Landmark Indexes ===
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(60, 68))

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    if len(mouth) < 7:
        return 0
    A = distance.euclidean(mouth[2], mouth[6])  # vertical
    B = distance.euclidean(mouth[3], mouth[5])  # vertical
    C = distance.euclidean(mouth[0], mouth[4])  # horizontal
    return (A + B) / (2.0 * C)

# === Start camera ===
cap = cv2.VideoCapture(0)

print("📷 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y = max(0, x), max(0, y)
        if (y + h) >= gray.shape[0] or (x + w) >= gray.shape[1]:
            continue

        face_img = gray[y:y+h, x:x+w]
        face_rgb = frame[y:y+h, x:x+w]
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH]

        avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # CNN Prediction
        cnn_label, cnn_confidence = 0, 1.0
        try:
            cnn_input = cv2.resize(face_img, (64, 64)) / 255.0
            cnn_input = cnn_input.reshape(1, 64, 64, 1)
            cnn_pred = cnn_model.predict(cnn_input, verbose=0)
            cnn_label = np.argmax(cnn_pred)
            cnn_confidence = float(np.max(cnn_pred))
        except:
            pass

        # RF Prediction
        rf_input = np.array([[avg_EAR, 0.3, 0, 2.5, 2.4, 2.6]])
        rf_raw_label = rf_model.predict(rf_input)[0]
        label_map = {'drowsy': 1, 'alert': 0}
        rf_label = label_map.get(str(rf_raw_label).lower(), 0)

        # Final decision
        final_label = "DROWSY" if cnn_label == 1 or rf_label == 1 or avg_EAR < EAR_THRESHOLD else "ALERT"
        color = (0, 0, 255) if final_label == "DROWSY" else (0, 255, 0)

        # Emotion detection
        try:
            results = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display on Frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{final_label} (CNN: {cnn_confidence:.2f})", (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Yawning: {'Yes' if mar > MAR_THRESHOLD else 'No'}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}  MAR: {mar:.2f}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        # ✅ Terminal log
        print(f"Status: {final_label}, EAR: {avg_EAR:.2f}, MAR: {mar:.2f}")
        print(f"Yawning Detected: {'Yes' if mar > MAR_THRESHOLD else 'No'}")
        print(f"Emotion: {emotion}")
        print("-" * 50)

        # Alarm logic (disabled)
        if final_label == "DROWSY":
            drowsy_frames += 1
        else:
            drowsy_frames = 0

        if mar > MAR_THRESHOLD:
            yawning_frames += 1
        else:
            yawning_frames = 0

        if (drowsy_frames >= 3 or yawning_frames >= 2) and sound_enabled:
            pygame.mixer.music.load(alarm_sound)
            pygame.mixer.music.play()
        elif sound_enabled:
            pygame.mixer.music.stop()

    cv2.imshow("Drowsiness Detection - CNN + RF + Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# pygame.mixer.quit()
