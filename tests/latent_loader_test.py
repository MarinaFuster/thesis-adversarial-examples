import os
import numpy as np
import unittest

from modules.latent_loader import LatentLoader


class LatentLoaderTest(unittest.TestCase):
    def test_load_invalid_filepath(self):
        with self.assertRaises(FileNotFoundError):
            LatentLoader.load_latent("thisisntreal")
    
    def test_load_latent(self):
        latent = LatentLoader.load_latent("latents/testexample")
        
        self.assertEqual((6,),latent.shape)
    
    def test_save_latent(self):
        latent = np.array([0,2,4,6,8])
        LatentLoader.save_latent(latent, "testsave", "./latents")
        
        self.assertTrue(os.path.isfile("./latents/testsave"))
        os.remove("./latents/testsave")