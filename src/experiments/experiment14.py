# This experiment intends to "zoom in" on the best result obtained
# in experiment13 -> marina33_-1.59 modified gave 69% for nachito and 64% for marina
# so we could argue that i cannot distinguish one from another, even though
# the general appearance is nachito.

# This experiment takes that image and modifies both second and third principal components
# to analyze if recognition shifts.

# Date: 03/08/2021

import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from glob import glob

from modules.data_loader import read_image
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_parent_experiment_directory


def decode_principal_components_projections(pc_projections, pca, scaler, model):
    scaled_latent = pca.inverse_transform(pc_projections)
    latent = scaler.inverse_transform(scaled_latent)
    return (model.predict(latent[None, :]).astype("float32") * 255.0)[0]


# After calling this method, I selected 42 pictures and run get_adversarial_example_metrics
# This gave 42 folders with ae_metrics which will be analyzed in the method below this one
def generate_marina33_modifying_first_three_pc():
    original = read_image(f'../data/marina33.jpg').astype("float32") / 255.0

    encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
    decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
    latents_sc = load(LATENTS_SCALER_PATH)
    pca_module = load(PCA_PATH)

    images = np.array([original])
    pca_processed = pca_module.transform(latents_sc.transform(encoder.predict(images)))

    parent_dir = create_parent_experiment_directory("Experiment14")

    marina33_projections = pca_processed[0]

    for i in np.arange(-1.7, -1.4, 0.1):  # first principal component projection
        for j in np.arange(-6, 8, 0.5):  # second principal component projection
            for k in np.arange(-6, 6, 0.5):  # third principal component projection
                current_pca = marina33_projections.copy()
                current_pca[0] = i
                current_pca[1] = j
                current_pca[2] = k
                current_decoded = decode_principal_components_projections(
                    current_pca,
                    pca_module,
                    latents_sc,
                    decoder
                )
                cv2.imwrite(
                    f'{parent_dir}/marina33|{round(i,2)}|{round(j,2)}|{round(k,2)}.jpg',
                    current_decoded
                )


def gather_metrics_from_ae_results(parent_directory, results_filename, suptitle, title):
    directories = glob(f"../results/{parent_directory}/AE_*")

    m_similarities = []
    n_similarities = []
    similarities_labels = []

    for directory in directories:

        with open(f'{directory}/AE_metrics.csv', 'r') as f:
            f.readline()
            marina_metrics = f.readline().split(",")
            nachito_metrics = f.readline().split(",")

            m_similarities.append(float(marina_metrics[4]))
            n_similarities.append(float(nachito_metrics[4]))

        dir_name = directory.split("/")[-1].split("_")[1].split("-Aug")[0]
        similarities_labels.append(dir_name)


    # plotting similarities
    xs = np.arange(0, len(m_similarities))

    zipped_lists = zip(m_similarities, n_similarities)
    sorted_lists = sorted(zipped_lists, reverse=True)
    m_sorted = [e for e, _ in sorted_lists]
    n_sorted = [e for _, e in sorted_lists]

    label_zipped_lists = zip(m_similarities, similarities_labels)
    label_sorted_lists = sorted(label_zipped_lists, reverse=True)
    sim_labels = [e for _, e in label_sorted_lists]

    plt.plot(xs, m_sorted, color='magenta', label="Marina")
    plt.plot(xs, n_sorted, color='paleturquoise', label="Nachito")
    plt.xticks(xs, labels=sim_labels, rotation=90, size=8)
    plt.ylabel("Similarity Percentage")
    plt.title(title)
    plt.suptitle(suptitle)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../results/{results_filename}.jpg")
    plt.close()


# Important! check comments on folders before running these
if __name__ == '__main__':
    gather_metrics_from_ae_results(
        # This folder was created from a selection after generating all images
        # (first part of the experiment).
        # It was kind of a random selection, with 42 images that have first three
        # principal components projections modified. It was intended that these
        # images looked kind of real, but not all of them look real (some are awful)
        'experiment_14_subset',
        'similarity_for_ae_marina33',
        "Similarity for Adversarial Examples",
        "True Label: Nachito"
    )
    gather_metrics_from_ae_results(
        # This folder was created from a selection from experiment13.
        # The intention was to select images that looked real and behave as expected:
        # after modifying its first principal component projection, its overall look changed
        # to the other person in the dataset (marina -> nachito or nachito -> marina)
        'best',
        'similarity_for_best_from_e13',
        "Similarity for Adversarial Examples",
        None
    )
    gather_metrics_from_ae_results(
        # This folder was created from a selection from experiment13.
        # The intention was to select images that looked real and DID NOT behave as expected:
        # after modifying its first principal component projection, its overall look stayed the same
        # This behaviour happened mostly with marina's pictures with nachito's first principal component projection.
        'still_marina',
        'similarity_for_still_marina_e13',
        "(even after changing first component)",
        "Similarity for examples that retain their look"
    )
