import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1) Paths
MODEL_PATH = "emotion_model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# 2) Load & verify files
assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
assert os.path.exists(CASCADE_PATH), f"Cascade not found at {CASCADE_PATH}"

emotion_model = load_model(MODEL_PATH)
face_cascade  = cv2.CascadeClassifier(CASCADE_PATH)
labels        = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# 3) Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48)).astype('float32')/255
        face = np.expand_dims(face, axis=(0,-1))  # shape=(1,48,48,1)

        preds   = emotion_model.predict(face)[0]
        idx     = np.argmax(preds)
        emotion = labels[idx]
        conf    = preds[idx]

        # Draw
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.putText(frame,f"{emotion} {conf*100:.0f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
