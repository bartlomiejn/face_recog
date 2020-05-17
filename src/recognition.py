import cv2
import openface
from threading import Thread
from queue import LifoQueue

class FaceRecognizer(Thread):
    def __init__(self, predictor, model, img_dim):
        Thread.__init__(self)

        self.daemon = True
        self.predictor = predictor
        self.model = model
        self.img_dim = img_dim
        self.queue = LifoQueue()

    def run(self):
        while True:
            img, bboxes = self.queue.get()
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for bbox in bboxes:
                aligned_face = self.predictor.align(self.img_dim, rgb_img, bbox, 
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                
                if aligned_face is None:
                    raise Exception("Unable to align provided image.")

                emb = self.model.forward(aligned_face)

                print("Face embeddings:")
                print(emb)


    def enqueue(self, image, bboxes):
        self.queue.put((image, bboxes))
