import logging
import numpy as np
import os

logger = logging.getLogger("LatentLoader")


class LatentLoader:
    
    @staticmethod
    def load_latent(filepath, delimiter=None):
        if not os.path.isfile(filepath):
            logger.error(f"{filepath} is not a valid filepath")
            raise FileNotFoundError("Invalid filepath")
        
        return np.genfromtxt(filepath, delimiter=delimiter)
    
    @staticmethod
    def save_latent(latent, filename, folder='latents', delimiter=None):
        filepath = f"{folder}/{filename}"
        np.savetxt(filepath, latent, delimiter=delimiter)
        logger.info(f"Latent space saved in {filepath}")
