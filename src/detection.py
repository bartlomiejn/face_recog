import cv2
import dlib
from threading import Thread
from threading import Lock
from queue import LifoQueue

class FaceDetector(Thread):
    def __init__(self, casc, recognizer):
        Thread.__init__(self)

        self.casc = casc
        self.recognizer = recognizer
        self.daemon = True
        self.size_thresh = 200
        self.queue = LifoQueue()
        self.lock = Lock()
        self.faces = []

    def run(self):
        while True:
            image = self.queue.get()
            
            faces = self.casc.detectMultiScale(image, 1.1, 4)
            bboxes = []

            self.lock.acquire()
            self.faces = []
            for (x, y, w, h) in faces:
                if w > self.size_thresh and h > self.size_thresh:
                    self.faces.append((x, y, w, h))
                    bboxes.append(
                        dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h))
            self.lock.release()

            self.recognizer.enqueue(image, bboxes)

    def get_faces(self):
        self.lock.acquire()
        faces = self.faces
        self.lock.release()
        return faces

    def enqueue(self, image):
        self.queue.put(image)
