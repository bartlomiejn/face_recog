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
        self.size_thresh = 200
        self.queue = LifoQueue()
        self.lock = Lock()
        self.faces = []

    def run(self):
        while True:
            image = self.queue.get()
            
            faces = self.casc.detectMultiScale(image, 1.1, 4)
            
            self.lock.acquire()
            self.faces = []
            for (x, y, w, h) in faces:
                if w > self.size_thresh and h > self.size_thresh:
                    self.faces.append((x, y, w, h))
            self.lock.release()



    def get_faces(self):
        self.lock.acquire()
        faces = self.faces
        self.lock.release()
        return faces

    def put_image(self, image):
        self.queue.put(image)
