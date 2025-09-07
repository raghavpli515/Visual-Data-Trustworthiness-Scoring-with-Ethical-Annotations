import cv2
import numpy as np

def blur_faces(frame):
    # Simple Haar cascade face blurring as a placeholder.
    # In production, use more robust detectors.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (51, 51), 30)
        frame[y:y+h, x:x+w] = roi
    return frame
