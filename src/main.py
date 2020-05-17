import os
from argparse import ArgumentParser
import cv2
import numpy as np
from openface import AlignDlib
from openface import TorchNeuralNet
from detection import FaceDetector
from recognition import FaceRecognizer

if __name__ == '__main__':
    img_dim = 96
    process_each = 5
    frame_no = 0

    assets_dir = os.environ["ASSETS_DIR"]
    openface_dir = os.environ["OPENFACE_DIR"]
    if not openface_dir:
        raise Exception(f"Missing OPENFACE_DIR environment variable")
    
    align = os.path.join(
        openface_dir, 
        "models/dlib/shape_predictor_68_face_landmarks.dat")
    model = os.path.join(openface_dir, "models/openface/nn4.small2.v1.t7")
    if not os.path.exists(align) or not os.path.exists(model):
        raise Exception("Missing align predictor or pytorch model")
    align_dlib = AlignDlib(align)
    net = TorchNeuralNet(model, img_dim)
    recognizer = FaceRecognizer(align_dlib, net, img_dim)

    casc = os.path.join(assets_dir, "haarcascade_frontalface_default.xml")
    casc_clsf = cv2.CascadeClassifier(casc)
    if casc_clsf.empty():
        raise Exception(f"Couldn't load casc_clsf: {casc_clsf}")
    detector = FaceDetector(casc_clsf, recognizer)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Camera open failure")

    detector.start()
    recognizer.start()

    while True:
        err, frame = cap.read() # Reads BGR frame

        if not err:
            print(f"Did not receive frame, err: {err}")
            break

        if frame_no >= process_each:
            faces = detector.enqueue(frame)
            frame_no = 0
        faces = detector.get_faces() # x, y, w, h
        
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
        
        cv2.imshow('Cascade classifier', frame)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()
