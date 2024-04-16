# Hypothesis: when changing the overall look of the person (what makes it
# marina or nachito to our human eyes), there are still some things left
# that a facial recognition system will recognize as the original person
# even though we do not visually recognize it. This will cause the FRS
# to misidentify a person, according to the "true label" (which is the one given by humans)

# For each original image (of nachito and marina) we will generate 5 new images
# by changing the first principal component projection for a value that corresponds
# to the other person of the picture

# Example: I start with marina64 and generate principal components' projections
# I know that first principal component range for nachito is ~-4 to ~-2, so I choose
# X values between that range and generate X new images

# Date: 02/08/2021

import cv2
import numpy as np
from joblib import load
from glob import glob

from modules.data_loader import load_data
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_experiment_directory, create_subexperiment_directory
from infra_aws.comparisons_control import get_adversarial_example_metrics

def decode_principal_components_projections(pc_projections, pca, scaler, model):
    scaled_latent = pca.inverse_transform(pc_projections)
    latent = scaler.inverse_transform(scaled_latent)
    return (model.predict(latent[None, :]).astype("float32") * 255.0)[0]


def generate_images_with_different_first_component():
    images, full_labels = load_data(mode='gray', split_test=False)
    images = images.astype("float32") / 255.0

    encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
    decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
    latents_sc = load(LATENTS_SCALER_PATH)
    pca_module = load(PCA_PATH)

    pca_processed = pca_module.transform(latents_sc.transform(encoder.predict(images)))

    parent_dir = create_experiment_directory("experiment13")
    nachito_dir = create_subexperiment_directory(parent_dir, "from_nachito_originals")
    marina_dir = create_subexperiment_directory(parent_dir, "from_marina_originals")

    n_first_component_values = np.arange(-5, -1.5, 0.2)
    m_first_component_values = np.arange(1, 6.5, 0.2)

    for projections, label in zip(pca_processed, full_labels):

        if "marina" in label:
            folder = marina_dir
            values = n_first_component_values
        elif "nachito" in label:
            folder = nachito_dir
            values = m_first_component_values
        else:
            raise ValueError("Target is neither marina nor nachito.")

        for value in values:
            current_projections = projections.copy()
            current_projections[0] = value
            current_decoded = decode_principal_components_projections(
                current_projections,
                pca_module,
                latents_sc,
                decoder
            )
            cv2.imwrite(
                f'{folder}/{label}_{value}.jpg',
                current_decoded
            )

    return parent_dir


def generate_ae_metrics_for_images(directory, experiment_directory):
    filenames = glob(f'{directory}/*')
    for filename in filenames:
        get_adversarial_example_metrics(filename, root_dir=experiment_directory)


# Example for calling this experiment.
# parent_directory = generate_images_with_different_first_component()

# first parameter is directory where images are.
# second parameter is directory where you want to save AE_{filename} folders with results.
# generate_ae_metrics_for_images('../results/test', '../results/test')
