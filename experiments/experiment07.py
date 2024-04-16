# We have 64 principal components. We have plotted two classes and we can
# clearly distinguish two clusters when plotting first two components.
# In this experiment, we are computing n_avg and m_avg, where n_avg and m_avg
# are the average principal components of the images of Nachito and Marina respectively.
# We intend to go through clusters coordinates corresponding to the other person to see what happens.

# Comments: when using n_avg with first two components of m cluster, we don't really see any m resemblance.
# Instead, when using m_avg with first two components of n cluster, we see SOME resemblance to n in a few cases.
# Could this be because m cluster is more sparse than n cluster ?

# Remaining question... Why if first two components represent only ~26.5% of explained variance,
# the image seems to be the person that we are using those first two components? E.g: if I use 62
# principal components from nacho with 2 (1st and 2nd) components from marina, the result will
# seem to be marina, even though we have ~74.5% information of nacho.

# Hypothesis: even though principal components can show two very clear clusters, that doesn't happen
# anymore when analysing the same situation on other principal components, meaning, clusters will
# be merged. This will be checked on experiment 08.

# Date: 26/07/2021
# Important! Images will be stored in results folder

import cv2
import numpy as np
from joblib import load

from modules.data_loader import load_data
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_experiment_directory, create_subexperiment_directory

# First step : load data
images, full_labels = load_data(mode='gray', split_test=False)
images = images.astype("float32") / 255.0

# Second step : get principal components of those images
encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
latents_sc = load(LATENTS_SCALER_PATH)
pca_module = load(PCA_PATH)

pca_processed = pca_module.transform(latents_sc.transform(encoder.predict(images)))

# Third step : get n_avg and m_avg
ns = []
ms = []
for label, components in zip(full_labels, pca_processed):
    if "nachito" in label:
        ns.append(components)
    else:
        ms.append(components)
ns = np.array(ns)
ms = np.array(ms)
n_avg = ns.mean(axis=0)
m_avg = ms.mean(axis=0)

# Fourth step : run experiment for n_avg
# (result will be "mostly" nacho in terms of explained variance ratio with first two comp from marina)
parent_dir = create_experiment_directory("Experiment07")

target_dir = create_subexperiment_directory(parent_dir, "Original_Marina")
for first in np.arange(0, 6, 0.2):
    for second in np.arange(-6, 6, 0.2):
        current_n = np.copy(n_avg)
        current_n[0] = first
        current_n[1] = second
        current_n_pca = pca_module.inverse_transform(current_n)
        current_latent = latents_sc.inverse_transform(current_n_pca)
        current_decoded = (decoder.predict(current_latent[None, :]).astype("float32") * 255.0)[0]
        cv2.imwrite(
            f'{target_dir}/marina_c1_{round(first, 3)}_c2_{round(second, 3)}.jpg',
            current_decoded
        )

# Fifth step : run experiment for m_avg
# (result will be "mostly" marina in terms of explained variance ratio with first two comp from nacho)
target_dir = create_subexperiment_directory(parent_dir, "Original_Nachito")
for first in np.arange(-4, -1, 0.2):
    for second in np.arange(-4, 4, 0.2):
        current_m = np.copy(m_avg)
        current_m[0] = first
        current_m[1] = second
        current_pca = pca_module.inverse_transform(current_m)
        current_latent = latents_sc.inverse_transform(current_pca)
        current_decoded = (decoder.predict(current_latent[None, :]).astype("float32") * 255.0)[0]
        cv2.imwrite(
            f'{target_dir}/nacho_c1_{round(first, 3)}_c2_{round(second, 3)}.jpg',
            current_decoded
        )
