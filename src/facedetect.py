import cv2


class FaceDetector:
    def __init__(self, classifier):
        self.casc = cv2.CascadeClassifier(classifier)
        if self.casc.empty():
            raise Exception(f"Couldn't load classifier: {classifier}")

    def detect_faces(self, image):
        faces = self.casc.detectMultiScale(image, 1.1, 6)
        return faces
