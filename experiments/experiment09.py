# After studying principal components projections distribution, this experiment
# intends to see what happens when taking an avg of all the projections onto
# the last 63 principal components and changing the first component across its range

# Hypothesis: since avg of nachito and marina for 63 components are very close, sliding
# over the first component might switch to one person and the other VISUALLY.

# This might give us an idea on how to KEEP the image looking visually like
# the person we want. E.g: we can play with all the other 63 components as long
# as we don't touch the first one. That might ensure us that we are still looking
# like the person we intend.

# We need to test this in another experiment

# Date: 28/07/2021

import cv2
import numpy as np
from joblib import load

from modules.data_loader import load_data
from modules.model_loader import ModelLoader
from core.constants import *
from experiments.utils import create_parent_experiment_directory, create_subexperiment_directory

# Step Zero: Setup experiment directory
parent_dir = create_parent_experiment_directory("Experiment09")

# First step : load data
images, full_labels = load_data(mode='gray', split_test=False)
images = images.astype("float32") / 255.0

# Second step : get principal components of those images
encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
latents_sc = load(LATENTS_SCALER_PATH)
pca_module = load(PCA_PATH)

print("Eigenvalue for first component")
print(pca_module.explained_variance_[0])
print("Eigenvalue ratio for first component")
print(pca_module.explained_variance_ratio_[0])
print("Eigenvector for first component")
print(pca_module.components_[0])

pca_processed = pca_module.transform(latents_sc.transform(encoder.predict(images)))

avg_pca = pca_processed.mean(axis=0)

# We are saving avg image on results
avg_latent = latents_sc.inverse_transform(
    pca_module.inverse_transform(avg_pca)
)
avg_decoded = (decoder.predict(avg_latent[None, :]).astype("float32") * 255.0)[0]
cv2.imwrite(f'{parent_dir}/average_image.jpg', avg_decoded)


avg_subdir = create_subexperiment_directory(parent_dir, "averages")

# We know from experiment08's plot that first component ranges from ~-4.5 to ~6.5
for i, value in enumerate(np.arange(-4.5, 6.5, 0.2)):
    current = np.copy(avg_pca)
    current[0] = value
    current_latent = latents_sc.inverse_transform(pca_module.inverse_transform(current))
    current_decoded = (decoder.predict(current_latent[None, :]).astype("float32") * 255.0)[0]
    cv2.imwrite(f'{avg_subdir}/{i}_average_63_projections_with_first_{round(value, 2)}.jpg', current_decoded)
