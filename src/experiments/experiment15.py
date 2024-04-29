
#  The objective of this experiment is to use the images obtained in experiment12
#  to try and check for adversarial examples.
#  This is more of a "spray attack". Checking manually images that might be ambiguous enough
#  to be difficult to recognize by rekognition.
#
#  Date: 13/08/2021


import numpy as np


from modules.data_loader import load_data
from experiments.utils import (
    create_parent_experiment_directory,
    get_latents_from_dataset,
    reconstruct_face_from_latent
)
import cv2

# Keys: Latent dimension; Values: Value to change it to
changes = {
    3: [133.24],
    4: [8.338],
    21: [122.153999],
    41: [229.666, 337.17499]

}


def generate_adversarial_examples(avg_face):
    for dimension, values in changes.items():
        new_face = np.copy(avg_face)
        for i, value in enumerate(values):
            new_face[dimension] = value
            cv2.imwrite(f"{parent_dir}/adversarial_{dimension}_{i}.png", reconstruct_face_from_latent(new_face))


# Step 0: Setup experiment directory
parent_dir = create_parent_experiment_directory("Experiment15", timestamp=False)

# Step 1 : load data
images, full_labels = load_data(mode='gray', split_test=False)
images = images.astype("float32") / 255.0

full_latents = get_latents_from_dataset(images)
avg_latents = full_latents.mean(axis=0)


generate_adversarial_examples(
    avg_latents
)

