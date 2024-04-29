# This experiment is a consequence of experiment 07. We will try to analyse
# class separation (c1:Nacho, c2:Marina) for:
#       - Principal components projections
#       - Latent representation of images
#       - Scaled latent representation of images

# Date: 27/07/2021
# Modified: 29/07/2021

import matplotlib.pyplot as plt
import numpy as np
from joblib import load

from modules.data_loader import load_data
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_parent_experiment_directory

def plot_classes(c1, c2, labels, n, axis, title_prefix):

    axis.set_ylim([-5, 5])
    axis.set_yticks([])
    axis.scatter(c2, np.repeat(1, len(c2)), color='magenta')
    axis.scatter(c1, np.repeat(-1, len(c1)), color='paleturquoise')
    axis.legend(labels)

    axis.set_title(f"{title_prefix} {n+1}")

# This function will separate, for a set of vectors (which can be latent vectors
# or pca vectors), those corresponding to nachito and those corresponding to marina.
# Then, an image with clusters per dimension (latent dimension or principal component
# projection) and a plot of distances between m_avg or n_avg for those set of vectors.
# Files will be store in results folder.
def representation_analysis(
        vectors,
        labels,
        dimensions,
        target_dir,
        clusters_per_dimension_filename,
        clusters_per_dimension_title_prefix,
        distances_filename,
        distances_xlabel
):
    ns = []
    ms = []
    for label, components in zip(labels, vectors):
        if "nachito" in label:
            ns.append(components)
        else:
            ms.append(components)

    ns = np.array(ns)
    ms = np.array(ms)

    classes = ["Marina", "Ignacio"]  # they are inverted in the method

    _, axs = plt.subplots(8, 8, figsize=(64, 64))
    axs = axs.flatten()
    for component, ax in zip(np.arange(dimensions), axs):
        nachito = ns[:, component]
        marina = ms[:, component]
        plot_classes(
            nachito,
            marina,
            classes,
            component,
            ax,
            clusters_per_dimension_title_prefix
        )

    plt.savefig(f"{target_dir}/{clusters_per_dimension_filename}.jpg")
    plt.close()

    n_avg = ns.mean(axis=0)
    m_avg = ms.mean(axis=0)

    distances = [np.linalg.norm(x1 - x2) for (x1, x2) in zip(n_avg, m_avg)]
    plt.plot(distances, marker='o')

    # x ticks starting at 1 instead of 0
    ticks = np.arange(start=1, stop=65, step=1)
    labels = [str(x) if x % 5 == 0 else "" for x in ticks]
    labels[0] = "1"
    labels[-1] = "64"
    plt.xticks(ticks=ticks, labels=labels)

    plt.yscale("log")
    plt.xlabel(distances_xlabel)
    plt.ylabel("Euclidean Distance")
    plt.title("Distance between avg member of class, per component")

    plt.savefig(f"{target_dir}/{distances_filename}.jpg")
    plt.close()


# Step Zero: Setup experiment directory
parent_dir = create_parent_experiment_directory("Experiment08")

# First step : load data
images, full_labels = load_data(mode='gray', split_test=False)
images = images.astype("float32") / 255.0


# Second step : get principal components of those images
encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
latents_sc = load(LATENTS_SCALER_PATH)
pca_module = load(PCA_PATH)

latents = encoder.predict(images)
scaled_latents = latents_sc.transform(latents)
pca_processed = pca_module.transform(scaled_latents)

representation_analysis(
    pca_processed,
    full_labels,
    PCA_COMPONENTS,
    parent_dir,
    "clusters_per_principal_component",
    "Principal Component",
    "distance_between_averages_pca",
    "Component"
)
representation_analysis(
    latents,
    full_labels,
    LATENT_DIMENSION,
    parent_dir,
    "clusters_per_latent",
    "Latent Dimension",
    "distance_between_averages_latent",
    "Dimension"
)
representation_analysis(
    scaled_latents,
    full_labels,
    PCA_COMPONENTS,
    parent_dir,
    "clusters_per_scaled_latent",
    "Scaled Latent Dimension",
    "distance_between_averages_scaled_latent",
    "Dimension"
)
