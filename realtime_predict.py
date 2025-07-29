import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import joblib
from collections import deque, Counter

# Load trained model and label encoder
model = tf.keras.models.load_model('gesture_classifier.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Webcam
cap = cv2.VideoCapture(0)

# Track predictions
prediction_history = deque(maxlen=15)  # Store last 15 predictions
last_spoken_label = None
last_spoken_time = 0
cooldown_time = 1.5  # seconds

print("Real-time gesture recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_label = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get 63 landmark values
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)
                confidence = np.max(prediction)
                current_label = label_encoder.inverse_transform([predicted_index])[0]

                # Add current label to prediction history
                if confidence > 0.7:
                    prediction_history.append(current_label)

    else:
        prediction_history.append("None")

    # Get smoothed (most frequent) prediction
    if prediction_history:
        most_common_label, count = Counter(prediction_history).most_common(1)[0]
        smoothed_prediction = most_common_label
    else:
        smoothed_prediction = "None"

    # Display label and confidence
    display_text = f"{smoothed_prediction}"
    if smoothed_prediction != "None":
        display_text += f": {confidence:.2f}"

    cv2.putText(frame, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    current_time = time.time()

    # Speak if new label and cooldown passed
    if (
        smoothed_prediction != last_spoken_label
        and smoothed_prediction != "None"
        and (current_time - last_spoken_time > cooldown_time)
    ):
        engine.stop()
        engine.say(smoothed_prediction)
        time.sleep(1.0)
        engine.runAndWait()
        last_spoken_label = smoothed_prediction
        last_spoken_time = current_time

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
