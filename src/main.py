import cv2
import os
from argparse import ArgumentParser
from facedetect import FaceDetector

assets_dir = os.environ["ASSETS"]
casc = os.path.join(assets_dir, "haarcascade_frontalface_default.xml")
detector = FaceDetector(casc)

cap = cv2.VideoCapture(0)

process_each = 5
current = 0

if not cap.isOpened():
    print("Camera open failure")
    exit(-1)

while True:
    err, frame = cap.read()

    if not err:
        print(f"Did not receive frame, err: {err}")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if current >= process_each:
        faces = detector.detect_faces(grayscale)

        for (x, y, w, h) in faces:
            cv2.rectangle(grayscale, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        current = 0

    cv2.imshow('Cascade Classifier', grayscale)

    if cv2.waitKey(1) == ord('q'):
        break

    current += 1

cap.release()
cv2.destroyAllWindows()