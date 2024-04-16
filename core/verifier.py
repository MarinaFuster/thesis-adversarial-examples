import os
import cv2
import numpy as np
from constants import *
from modules.data_loader import load_data
from modules.model_loader import ModelLoader


def miss(path):
    return not os.path.isfile(path)


class Verifier:
    """
    This program intends to verify that models work the way they are supposed to. You can run the program by using this
    file or, since it is encapsulated, you can check all of our models anywhere you want.
    Keep in mind that this will add overhead to whatever process you are running.
    """
    def verify_all(self):
        self.verify_encoder()
        self.verify_decoder()

    @staticmethod
    def verify_encoder():
        """
        This method will verify that, given the same images to the encoder over
        and over, the model will always return the same latent vector.
        """
        if miss(ENCODER_PATH) or miss(ENC_WEIGHTS_PATH):
            print("You cannot verify encoder since you do not have appropriate files for it.")
            return 1

        print("Starting encoder verification.")
        _, test_dataset, _, _ = load_data()
        print(f"Will verify encoder model with {len(test_dataset)} registers.")

        encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
        predictions = encoder.predict(test_dataset)
        for i in range(5):
            reproduce = encoder.predict(test_dataset)
            assert all(
                list(map(lambda pair: np.array_equal(pair[0], pair[1]), zip(predictions, reproduce)))
            )
        print("Finished encoder verification successfully")

    @staticmethod
    def verify_decoder():
        """
        This method will verify that, given the same latent vector to the decoder over
        and over, the model will always return the same image.
        This verification assumes that verify_encoder works properly
        """
        if miss(DECODER_PATH) or miss(DEC_WEIGHTS_PATH) or miss(ENCODER_PATH) or miss(ENC_WEIGHTS_PATH):
            print("You cannot verify decoder since you do not have appropriate files for it.")
            return 1

        print("Starting decoder verification.")
        _, test_dataset, _, _ = load_data()
        print(f"Will verify decoder model with {len(test_dataset)} registers.")

        encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
        latents = encoder.predict(test_dataset)
        decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
        predictions = decoder.predict(latents)

        for i in range(5):
            reproduce = decoder.predict(latents)
            assert all(
                list(map(lambda pair: np.array_equal(pair[0], pair[1]), zip(predictions, reproduce)))
            )
        print("Finished decoder verification successfully")

    @staticmethod
    def reconstruct_images(mode='train'):
        """
        This method will populate results folder with reconstruction
        and its respective original image. This is done for training
        or test dataset. You can use mode 'train' or 'test'
        """
        train_dataset, test_dataset, _, _ = load_data(mode='gray')
        dataset = train_dataset if mode == 'train' else test_dataset
        dataset = dataset.astype("float32") / 255.0
        print("Finished loading training data and scaling pixels.")

        encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
        latents = encoder.predict(dataset)

        decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
        reconstructed = decoder.predict(latents)

        for i in range(len(reconstructed)):
            # grab the original image and reconstructed image
            original = (dataset[i] * 255).astype("uint8")
            recon = (reconstructed[i] * 255).astype("uint8")
            output = np.hstack([original, recon])
            cv2.imwrite(f'../results/{mode}_reconstructed_{i}.png', output)


if __name__ == '__main__':
    verifier = Verifier()
    # verifier.verify_encoder()
    # verifier.verify_decoder()
    verifier.reconstruct_images()
