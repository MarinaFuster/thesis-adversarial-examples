import cv2
import numpy as np
from glob import glob
import logging
import json


class Config:
    logger = logging.getLogger("ConfigModule")

    def __init__(self, json_path):
        json_file = Config.read_configuration(json_path)
        self.image_prefix = json_file["image_prefix"]
        self.image_prefix = None if self.image_prefix == "" else self.image_prefix
        self.train_folder = json_file["training_folder"]

        self.mode = json_file["mode"]

    @staticmethod
    def read_configuration(json_path):
        with open(json_path, "r") as handler:
            config = json.load(handler)
        return config

    def load_data(self):
        if self.image_prefix is None:
            # The `or` is needed because while testing the context isn't root
            filenames = glob(f"{self.train_folder}/*")
        else:
            filenames = glob(f"{self.train_folder}/{self.image_prefix}*")

        if not filenames:
            self.logger.error("Couldn't find any suitable images...")
            raise FileNotFoundError("Invalid prefix")

        train_set = []
        test_set = []

        for filename in filenames:
            im = cv2.imread(filename)  # default is color
            train_set.append(im)
            test_set.append(im)

        return np.array(train_set), np.array(test_set)
