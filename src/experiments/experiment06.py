# This experiment will use experiment04bis results as a baseline
# We will make a "zoom" for transition from nachito9_5 to nachito9_6
# Date: 01/03/2021

# IMPORTANT
# CHECK OUT CSV_FROM, CSV_TO, IMAGE_FROM, IMAGE_TO
# AND MAKE SURE YOU HAVE THEM ON INTENDED FOLDERS

import numpy as np
from core.transformation_flow import image_to_principal_components, components_to_image
import logging

CSV_FROM = "../components/nachito9_5.csv"
CSV_TO = "../components/nachito9_6.csv"

IMAGE_FROM = "../components/nachito9_5.jpg"
IMAGE_TO = "../components/nachito9_6.jpg"


def generate_latent_vector(filenames):
    for filename in filenames:
        image_to_principal_components(filename)


if __name__ == '__main__':
    logger = logging.getLogger("EXP01")

    logger.info("Generating the .csv files for the latent vectors")
    #generate_latent_vector([IMAGE_TO, IMAGE_FROM])

    with open(CSV_FROM) as f:
        components_from = f.readlines()

    with open(CSV_TO) as f:
        # We are only interested in Maru's first component
        components_to = f.readlines()

    N = len(components_from)

    NUMBER_OF_STEPS = 21
    current = 0

    # We need to calculate step for every component
    steps = []
    for i in range(N):
        to = float(components_to[i])
        origin = float(components_from[i])
        step = abs(to-origin)/(NUMBER_OF_STEPS-1)
        if to < origin:
            step = step * -1
        steps.append(step)

    while current != NUMBER_OF_STEPS:
        filename = f"../components/nachito9_zoom0_{current}.csv"
        new_components_from = []
        with open(filename, "w") as file:
            for i in range(N):
                to = float(components_to[i])
                origin = float(components_from[i])
                step = steps[i]
                new_components_from.append(f"{origin + step * current}\n")

            file.writelines(new_components_from)

        components_to_image(filename)
        current += 1