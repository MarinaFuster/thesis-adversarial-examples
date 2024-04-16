import os
import unittest

from tensorflow.keras.models import Model
from modules.model_loader import ModelLoader


class ModelLoaderTest(unittest.TestCase):
    def test_invalid_modelpath(self):
        with self.assertRaises(FileNotFoundError):
            ModelLoader.load_model("thisisntreal", "./models/autoencoder_weights.h5")

    def test_invalid_weightspath(self):
        with self.assertRaises(FileNotFoundError):
            ModelLoader.load_model("./models/autoencoder.json", "thisisntreal")

    def test_load_model(self):
        model = ModelLoader.load_model("./models/autoencoder.json", "./models/autoencoder_weights.h5")
        self.assertIsInstance(model, Model)

    def test_save_model(self):
        model = ModelLoader.load_model("./models/autoencoder.json", "./models/autoencoder_weights.h5")
        ModelLoader.save_model(model, 'testautoencoder.json', 'testautoencoder_weights.h5', './models')
    
        self.assertTrue(os.path.isfile("./models/testautoencoder.json"))
        self.assertTrue(os.path.isfile("./models/testautoencoder_weights.h5"))

        os.remove("./models/testautoencoder.json")
        os.remove("./models/testautoencoder_weights.h5")
