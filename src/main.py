import cv2
import os
import numpy as np
from argparse import ArgumentParser
from facedetect import FaceDetector

if __name__ == '__main__':
    assets_dir = os.environ["ASSETS"]
    casc = os.path.join(assets_dir, "haarcascade_frontalface_default.xml")
    detector = FaceDetector(casc)

    cap = cv2.VideoCapture(0)

    process_each = 5
    current = 0

    if not cap.isOpened():
        print("Camera open failure")
        exit(-1)

    detector.start()

    while True:
        err, frame = cap.read()

        if not err:
            print(f"Did not receive frame, err: {err}")
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current >= process_each:
            faces = detector.put_image(np.copy(grayscale))
            current = 0

        faces = detector.get_faces()

        for (x, y, w, h) in faces:
            cv2.rectangle(grayscale, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Cascade Classifier', grayscale)

        if cv2.waitKey(1) == ord('q'):
            break

        current += 1

    cap.release()
    cv2.destroyAllWindows()