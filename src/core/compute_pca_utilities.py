import os.path
import matplotlib.pyplot as plt
import numpy as np

from joblib import dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modules.data_loader import load_data
from modules.model_loader import ModelLoader
from constants import *


# function that plots principal components projections
# corresponding to two different classes
def plot_classes(c1, c2, labels, filename=None, show=False):
    c1_xs = c1[:, 0]
    c1_ys = c1[:, 1]
    plt.scatter(c1_xs, c1_ys, color='paleturquoise')

    c2_xs = c2[:, 0]
    c2_ys = c2[:, 1]
    plt.scatter(c2_xs, c2_ys, color='magenta')

    plt.legend(labels)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Principal Components Projection of Dataset")
    if show:
        plt.show()
    else:
        plt.savefig(f"../results/{filename}.jpg")


if not os.path.isfile(ENCODER_PATH) or not os.path.isfile(ENC_WEIGHTS_PATH):
    print(
        f"Your encoder model at {ENCODER_PATH} or your weights file at {ENC_WEIGHTS_PATH} do not exist.\n"
        f"Please make sure you trained your dataset before running this program."
    )
    exit(1)

print("[INFO] Loading personal dataset...")
images, full_labels = load_data(mode='gray', split_test=False)

# scale the pixel intensities to the range [0, 1]
full_data = images.astype("float32") / 255.0
print("[INFO] Finished loading and scaling pixels.")

# We want to encoder all test images into their latent spaces
encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
print("[INFO] Finish loading encoder.")

latents = encoder.predict(full_data)
print("[INFO] Predictions made. Latents ready.")

# Once we have latent spaces, before applying PCA, we need to
# center and scale latents features. This is why we use a StandardScaler,
# which will be saved and used for further transformations.
sc_latents = StandardScaler()
processed_latents = sc_latents.fit_transform(latents)

dump(sc_latents, LATENTS_SCALER_PATH)
print("[INFO] Latents Scaler ready.")

# Once our latents have been preprocessed properly, we instantiate our PCA module
# and fit and transform our latents. We save this module for further use and after tihs...
pca_module = PCA(n_components=PCA_COMPONENTS)
pca_processed = pca_module.fit_transform(processed_latents)
dump(pca_module, PCA_PATH)
print("[INFO] PCA module ready.")

ns = []
ms = []
for label, components in zip(full_labels, pca_processed):
    if "nachito" in label:
        ns.append(components)
    else:
        ms.append(components)
plot_classes(
    c1=np.array(ns),
    c2=np.array(ms),
    labels=["Ignacio", "Marina"],
    show=True
)
