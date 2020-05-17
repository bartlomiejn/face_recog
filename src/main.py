import cv2
import os
import numpy as np
from argparse import ArgumentParser
from facedetect import FaceDetector

if __name__ == '__main__':
    assets_dir = os.environ["ASSETS"]
    casc = os.path.join(assets_dir, "haarcascade_frontalface_default.xml")
    detector = FaceDetector(casc)

    openface_dir = os.environ["OPENFACE_DIR"]
    if not openface_dir:
        print("No OPENFACE_DIR added")
        exit(-1)

    cap = cv2.VideoCapture(0)

    process_each = 5
    current = 0

    if not cap.isOpened():
        print("Camera open failure")
        exit(-1)

    detector.size_thresh = 200
    detector.start()

    while True:
        err, frame = cap.read()

        if not err:
            print(f"Did not receive frame, err: {err}")
            break

        if current >= process_each:
            faces = detector.put_image(frame)
            current = 0
        faces = detector.get_faces()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(
                frame, 
                f'{x}, {y}, {w}, {h}', 
                (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                1)
        current += 1

        cv2.imshow('Cascade Classifier', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()