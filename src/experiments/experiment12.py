
# We want to see how each principal components modifies the image and if the changes
# have a significant effect on said image.
#
# To test this, we first calculate the average face of all images and modify each component
# moving from the minimum to the maximium value of said component in nachito and in marina.
#

# Date: 02/08/2021

import numpy as np
from joblib import load
import matplotlib.pyplot as plt

from modules.data_loader import load_data
from enum import Enum
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_parent_experiment_directory


class RunMode(Enum):
    COMPONENTS = "PCA MODE",
    LATENT_SPACE = "LATENT SPACE MODE"


def get_full_data_components(images):

    # Second step : get principal components of those images
    encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)

    latents_sc = load(LATENTS_SCALER_PATH)
    pca_module = load(PCA_PATH)

    pca_processed = pca_module.transform(latents_sc.transform(encoder.predict(images)))

    return pca_processed


def get_limits_for_ith_component(i, ns):
    ith_component_values = np.array([x[i] for x in ns])
    min_index = np.where(
        ith_component_values == ith_component_values.min()
    )[0][0]

    max_index = np.where(
        ith_component_values == ith_component_values.max()
    )[0][0]

    print(f"Component {i} label_min: {full_labels[min_index]} | label_max: {full_labels[max_index]}")

    return ith_component_values.min(), ith_component_values.max()


def get_full_data_latents(full_images):

    encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)

    latents = encoder.predict(full_images)

    return latents


def face_from_components(face_components):
    # Second step : get principal components of those images
    decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
    latents_sc = load(LATENTS_SCALER_PATH)
    pca_module = load(PCA_PATH)

    avg_latent = latents_sc.inverse_transform(
        pca_module.inverse_transform(face_components)
    )
    return (decoder.predict(avg_latent[None, :]).astype("float32") * 255.0)[0]


def face_from_latent(latent):
    decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
    latent = latent[None, :]
    reconstructed = decoder.predict(latent)
    reconstructed = reconstructed.astype("float32") * 255.0
    return reconstructed[0]


def generate_changes(avg_face, ns, title_suffix, start_comp=0, end_comp=4, mode=None):
    comps_to_plot = 4
    iterations_from_comp = 8
    _, axs = plt.subplots(comps_to_plot, iterations_from_comp, figsize=(64, 32))

    for i in range(start_comp, end_comp):
        min_val, max_val = get_limits_for_ith_component(i, ns)
        step = abs(max_val - min_val)/7

        for modifier in range(0, 8):
            new_face = np.copy(avg_face)
            new_face[i] = min_val + step*modifier
            dimension_index = i % 5  # We want the index to be a number from 0 to 4
            plt.sca(axs[dimension_index][modifier])
            if mode == RunMode.LATENT_SPACE:
                plt.imshow(face_from_latent(new_face), cmap="gray")
            elif mode == RunMode.COMPONENTS:
                plt.imshow(face_from_components(new_face), cmap="gray")
            else:
                raise ValueError("Invalid Mode!")

            plt.title(f"{title_suffix}{i+1}, value: {round(float(new_face[i]), 4)}", fontsize=20)
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout(pad=8.0)
    plt.savefig(f'{parent_dir}/from_{start_comp}_to_{end_comp-1}_component.jpg')
    plt.close()
    print(f"Finished components from {start_comp} component to {end_comp} component")


# Step Zero: Setup experiment directory
parent_dir = create_parent_experiment_directory("Experiment12", timestamp=False)

# First step : load data
images, full_labels = load_data(mode='gray', split_test=False)
images = images.astype("float32") / 255.0

full_components = get_full_data_components(images)
avg_face_components = full_components.mean(axis=0)

full_latents = get_full_data_latents(images)
avg_latents = full_latents.mean(axis=0)

components_ranges = [(5 * x, 5*(x+1)) for x in range(13)]

# This breaks the computer
# for (start, end) in components_ranges:
#     generate_changes(
#         avg_face_components, ns,
#         "Nachito Component",
#         start_comp=start, end_comp=end
#     )

# This was used to run it manually
start = 0 # starts at component zero
end = 4 # does not include fourth component


latent: bool = True

if latent:
    generate_changes(
        avg_latents, np.array(full_latents),
        "Latent Dimension #",
        start_comp=start, end_comp=end,
        mode=RunMode.LATENT_SPACE
    )
else:
    generate_changes(
        avg_face_components, np.array(full_components),
        "Principal Component #",
        start_comp=start, end_comp=end,
        mode=RunMode.COMPONENTS
    )
