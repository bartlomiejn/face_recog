import cv2
from threading import Thread
from threading import Lock
from queue import LifoQueue


class FaceDetector(Thread):

    def __init__(self, classifier):
        Thread.__init__(self)

        self.casc = cv2.CascadeClassifier(classifier)

        if self.casc.empty():
            raise Exception(f"Couldn't load classifier: {classifier}")

        self.daemon = True
        self.queue = LifoQueue()
        self.lock = Lock()
        self.faces = []

    def run(self):
        while (True):
            image = self.queue.get()
            self.lock.acquire()
            self.faces = self.casc.detectMultiScale(image, 1.1, 6)
            self.lock.release()

    def get_faces(self):
        self.lock.acquire()
        faces = self.faces
        self.lock.release()
        return faces

    def put_image(self, image):
        self.queue.put(image)
