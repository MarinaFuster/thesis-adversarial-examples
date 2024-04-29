import os
import json
from datetime import datetime
from modules.model_loader import ModelLoader
from core.constants import *
from joblib import load

results_dir: str = "../results"


def save_rekognition_results(result, filename):
    json_object = json.dumps(result, indent=2)
    with open(filename, 'w') as f:
        f.write(json_object)


def create_experiment_directory(dir_name: str) -> str:
    time_suffix: str = datetime.now().strftime("%d-%h|%H:%M")
    target_dir: str = f"{results_dir}/{dir_name}_{time_suffix}"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print(f"Created directory {dir_name}")
    else:
        print(f"Directory {dir_name} already exists")

    return target_dir


def create_subexperiment_directory(parent_path: str, dir_name: str) -> str:
    """
    Create a subexperiment (not timestamped)
    """
    target_dir: str = f"{parent_path}/{dir_name}"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print(f"Created directory {dir_name}")
    else:
        print(f"Directory {dir_name} already exists")

    return target_dir


def create_parent_experiment_directory(dir_name: str, timestamp=True) -> str:
    """
    This method is the parent directory of the experiment
    """
    if timestamp:
        time_suffix: str = datetime.now().strftime("(%d-%h|%H:%M)")
        target_dir: str = f"{results_dir}/{dir_name}_{time_suffix}"
    else:
        target_dir: str = f"{results_dir}/{dir_name}"

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print(f"Created directory {dir_name}")
    else:
        print(f"Directory {dir_name} already exists")

    return target_dir


def get_latents_from_dataset(data):

    encoder = ModelLoader.load_model(ENCODER_PATH, ENC_WEIGHTS_PATH)

    latents = encoder.predict(data)

    return latents


def reconstruct_face_from_latent(latent):
    decoder = ModelLoader.load_model(DECODER_PATH, DEC_WEIGHTS_PATH)
    latent = latent[None, :]
    reconstructed = decoder.predict(latent)
    reconstructed = reconstructed.astype("float32") * 255.0
    return reconstructed[0]
