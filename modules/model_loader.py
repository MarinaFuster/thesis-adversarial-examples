import logging
import os

from tensorflow.keras.models import model_from_json

logger = logging.getLogger("ModelLoaderModule")


class ModelLoader:

    @staticmethod
    def save_model(model, model_file, weights_file, folder='../models'):
        """
        This method stores a keras model on models/ folder.
        Make sure you have a models/ folder on your root project.
        It will serialize model to a JSON file and model weights to HDF5 file.
        """

        model_path = f"{folder}/{model_file}"
        weights_path = f"{folder}/{weights_file}"

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        logger.info(f"Saved model in {model_path}")
        
        # serialize weights to HDF5
        model.save_weights(weights_path)
        logger.info(f"Saved weights in {weights_path}")

    @staticmethod
    def load_model(model_path, weights_path):
        """
        This method will recover a keras model from models/ folder.
        Make sure you have both a json model and hdf5 weights to use it properly.
        """

        if not os.path.isfile(model_path):
            logger.error(f"{model_path} is not a valid model_path")
            raise FileNotFoundError("Invalid model_path")

        if not os.path.isfile(weights_path):
            logger.error(f"{weights_path} is not a valid weights_path")
            raise FileNotFoundError("Invalid weights_path")
            
        with open(model_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        # loads model
        loaded_model = model_from_json(loaded_model_json)
        logger.info(f"Loaded model from {model_path}")

        # loads weights
        loaded_model.load_weights(weights_path)
        logger.info(f"Loaded weights from {weights_path}")

        return loaded_model
