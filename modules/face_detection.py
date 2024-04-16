from typing import List
import logging
import os
import numpy as np

import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN


class FaceDetector:

    logger: logging.Logger = logging.getLogger("FaceDetectorModule")
    detector: MTCNN
    detected_faces: List

    def __init__(self):
        self.detector = MTCNN()
        self.detected_faces = []

    def detect_faces(self, filepath):
        img = cv2.imread(filepath)
        if img is None:
            self.logger.error(f"File with path {filepath} doesn't exist")
            exit(1)
        faces = self.detector.detect_faces(img)

        for result in faces:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            self.detected_faces.append(img[y: y1, x: x1])

    def save_detected_faces(self):
        if not self.detected_faces:
            self.logger.error("There where no detected faces")
            return

        counter: int = 0
        for face in self.detected_faces:
            cv2.imwrite(f"data/face_{counter}", face)
            counter += 1

    def bulk_save_and_detect_faces(self, paths: List):
        for path in paths:
            self.detect_faces(path)

        filenames: List = list(map(lambda x: x.split("/")[-1], paths))  # Remove parents from filename
        if len(self.detected_faces) != len(filenames):
            self.logger.error("Detected faces num != filenames")
        else:
            counter = 0
            for name in filenames:
                self.logger.info(f"Saving {name} image into data directory")
                cv2.imwrite(f"../data/{name}.jpg", self.detected_faces[counter])
                self.logger.info(f"Successfully saved {name} image into data directory")
                counter += 1

            self.detected_faces = []


class ImageTransformation:
    crop_face = True
    black_and_white = True
    width = 256
    height = 256

    def set_crop_face(self, crop_face):
        self.crop_face = crop_face

    def set_black_and_white(self, black_and_white):
        self.black_and_white = black_and_white

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    @staticmethod
    def detect_face(image):
        detector = MTCNN()
        # transforms PIL image to opencv format
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = detector.detect_faces(cv2_image) # does not make sense to have more images
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            x1, y1 = x + w, y + h

            # transforms opencv format to PIL image again
            return Image.fromarray(cv2.cvtColor(cv2_image[y: y1, x: x1], cv2.COLOR_BGR2RGB))

    # Receives a PIL Image
    def apply(self, image):
        if self.crop_face:
            image = self.detect_face(image)
            # case where face is not detected
            if not image: return None

        if self.black_and_white:
            image = image.convert('L')

        image = image.resize((self.width, self.height), Image.LANCZOS)
        return image

    def apply_batch(self, directory):
        for file in os.listdir(directory):
            if file.split(".")[-1] == 'jpg':
                image = Image.open(f'{directory}/{file}')
                image = self.apply(image)
                image.save(f'{directory}/{file}')
                print(f'Finished {file}')
