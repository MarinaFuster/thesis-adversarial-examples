import cv2
import numpy as np
import os.path
import sys
from joblib import load
from glob import glob

from core.constants import *
from modules.data_loader import read_image
from modules.model_loader import ModelLoader

output_folder = "../components"


"""
Given a filename, it returns the label associated with it
For example
./data/something/marina0.jpg -> marina0
./data/nachito10.jpg -> nachito10
"""
def get_label(filename):
    return filename.split("/")[-1].split(".")[0]


def image_to_principal_components(filename):
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist.")
        return 1
    im = read_image(filename)  # default is mode='gray'
    print(im.shape)
    im = im.astype("float32") / 255.0
    im = im[None, :]

    # Needed utilities to obtain components from image
    encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)
    latents_sc = load(LATENTS_SCALER_PATH)
    pca_module = load(PCA_PATH)

    # *** Flow to transform an image to its principal components' projections ***

    # We use our encoder to get the image's latent space
    latent = encoder.predict(im)
    # We center and scale the latent space
    processed_latent = latents_sc.transform(latent)
    # We use our pca module to transform the coordinates
    principal_components = pca_module.transform(processed_latent)

    # this is in order to get rid of folders and extension
    csv_name = get_label(filename)
    # Saves image's components in components/ folder
    np.savetxt(f"{output_folder}/{csv_name}.csv", principal_components, fmt="%.4f", delimiter='\n')

    print(f"[INFO] Principal Components of {filename} saved in {output_folder} folder")


def components_to_image(filename):
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist.")
        exit(1)

    principal_components = np.genfromtxt(filename)

    # Needed utilities to recover image from components
    decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
    latents_sc = load(LATENTS_SCALER_PATH)
    pca_module = load(PCA_PATH)

    # *** Flow to recover image from principal components ***

    # We recover original coordinates of processed latent space
    processed_latent = pca_module.inverse_transform(principal_components)
    # We undo centering and scaling of latent components
    latent = latents_sc.inverse_transform(processed_latent)
    latent = latent[None, :]
    # We predict with our decoder which image will be product of that latent space
    reconstructed = decoder.predict(latent)
    reconstructed = reconstructed.astype("float32") * 255.0

    jpg_name = get_label(filename)
    # Saves recovered image in components/ folder
    cv2.imwrite(f"{output_folder}/{jpg_name}.jpg", reconstructed[0])

    print(f"[INFO] Image reconstructed from {filename} saved in {output_folder} folder")


"""
Main program development
"""

im_to_comp = '-i'
comp_to_im = '-c'

option = 1
file = 2


# TODO: Use a fancier method for CLI arguments
def main():
    args = sys.argv
    if len(args) <= 2:
        print(f"Usage: {args[0]} (-c|-i) filename [--batch]")
        exit(1)

    if args[option] == im_to_comp:
        if len(args) > 3 and args[3] == "--batch":
            for filename in glob(args[file]):
                print(f"Running Image to Components for {filename}")
                image_to_principal_components(filename)
        else:
            image_to_principal_components(args[file])
    elif args[option] == comp_to_im:
        if len(args) > 3 and args[3] == "--batch":
            for filename in glob(args[file]):
                print(f"Running Image to Components for {filename}")
                components_to_image(filename)
        else:
            components_to_image(args[file])


def swap_components_variables(filename1: str, filename2: str, component_index: int):
    with open(filename1, "r") as f:
        first_components = f.readlines()

    with open(filename2, "r") as f:
        second_components = f.readlines()

    first = float(first_components[component_index])
    second = float(second_components[component_index])

    with open(filename1, "w") as f:
        first_components[component_index] = f"{(first + second)/2}\n"
        f.writelines(first_components)


if __name__ == '__main__':
    main()
