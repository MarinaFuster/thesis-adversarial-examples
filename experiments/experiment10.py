# This is a preliminary experiment of a more smart way to take advantage
# of the fact that the first component seems to "control" the overall
# visual identity of the person

# We will try to use a face with no extravagant expression from nachito (nachito40)
# and use its 63 principal components over 4 marina's images, ranging from those with
# extravagant expression, to normal expression.

# format is 'original_keeping_first_component' -> 'reference_for_last_components'

# (0) marina0 -> nachito40
# (1) marina80 -> nachito40
# (2) marina69 -> nachito40
# (3) marina41 -> nachito40

# Then, we'll do the same with nachito. The objective is to send this to Rekognition to check
# if just by keeping the first component we manage to keep a visual aspect of the person but
# achieving dodging when trying to recognize the person.

# (4) nachito16 -> marina69
# (5) nachito37 -> marina69
# (6) test-nachito9 -> marina69
# (7) nachito50 -> marina69

# (8) nachito40 -> marina 64 (dummy test, want to check if adding a smile is possible).

# Date: 28/07/2021
# Important! make sure you have created a transformations folder inside results

import matplotlib.pyplot as plt
import numpy as np
from joblib import load

from modules.data_loader import read_image
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_parent_experiment_directory, create_subexperiment_directory


# This method saves in /transformations folder the process of keeping the first principal
# component projection of original image with 63 principal components projections of reference image
def transform_original_with_reference(original_filename, reference_filename, dir="../results"):
    original = read_image(f'../data/{original_filename}.jpg').astype("float32") / 255.0
    reference = read_image(f'../data/{reference_filename}.jpg').astype("float32") / 255.0

    images = np.array([original, reference])
    pca_processed = pca_module.transform(latents_sc.transform(encoder.predict(images)))
    transformed = np.copy(pca_processed[1])  # all components from second image
    transformed[0] = pca_processed[0][0]  # first component of first image

    projections = np.array([pca_processed[0], pca_processed[1], transformed])
    latents = latents_sc.inverse_transform(pca_module.inverse_transform(projections))
    decoded = (decoder.predict(latents).astype("float32") * 255.0)

    # this should have a plot with original, reference and transformed.
    figures = [decoded[0], decoded[1], decoded[2]]
    labels = [f"Original {original_filename.replace('nachito', 'ignacio')}", f"Reference ({reference_filename.replace('nachito', 'ignacio')})", "Result"]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(figures[i], cmap='gray')
        plt.title(labels[i])
        plt.tight_layout()

    #plt.suptitle('First Component Projection from Original, while 63 left from Reference (all images after decoder)')
    plt.savefig(f'{dir}/{original_filename}_with_63_components_of_{reference_filename}.jpg')
    plt.close()


# Step Zero: Setup experiment directory
parent_dir = create_parent_experiment_directory("Experiment10")

transform_dir = create_subexperiment_directory(parent_dir, "transformations")
encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
latents_sc = load(LATENTS_SCALER_PATH)
pca_module = load(PCA_PATH)

transform_original_with_reference('marina0', 'nachito40', dir=transform_dir)
#transform_original_with_reference('marina80', 'nachito40', dir=transform_dir)
transform_original_with_reference('marina69', 'nachito40', dir=transform_dir)
transform_original_with_reference('marina41', 'nachito40', dir=transform_dir)
#transform_original_with_reference('nachito16', 'marina69', dir=transform_dir)
transform_original_with_reference('nachito37', 'marina69', dir=transform_dir)
transform_original_with_reference('test-nachito9', 'marina69', dir=transform_dir)
transform_original_with_reference('nachito50', 'marina69', dir=transform_dir)
transform_original_with_reference('nachito40', 'marina64', dir=transform_dir)
transform_original_with_reference('marina69', 'nachito22', dir=transform_dir)


