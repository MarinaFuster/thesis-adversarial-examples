# In this experiment we will manipulate all componets
# Date: 01/03/2021

import numpy as np
from core.transformation_flow import image_to_principal_components, components_to_image
import logging

CSV_FROM = "../components/nachito9.csv"
CSV_TO   = "../components/marina2.csv"

IMAGE_FROM = "../data/nachito9.jpg"
IMAGE_TO   = "../data/marina2.jpg"


def generate_latent_vector(filenames):
    for filename in filenames:
        image_to_principal_components(filename)


if __name__ == '__main__':
    logger = logging.getLogger("EXP01")

    logger.info("Generating the .csv files for the latent vectors")
    generate_latent_vector([IMAGE_TO, IMAGE_FROM])

    with open(CSV_FROM) as f:
        components_from = f.readlines()

    with open(CSV_TO) as f:
        # We are only interested in Maru's first component
        components_to = f.readlines()

    logger.info("Modifying first component of nachito")
    N = len(components_from)


    NUMBER_OF_STEPS = 11
    current = 0

    while current != NUMBER_OF_STEPS:
        filename = f"../components/nachito9_{current}.csv"
        with open(filename, "w") as file:
            for i in range(N):
                to = float(components_to[i])
                origin = float(components_from[i])
                step = abs(to - origin)/10
                if to < origin:
                    step = step * -1
                components_from[i] = f"{origin + step * current}\n"

            file.writelines(components_from)

        components_to_image(filename)
        current += 1
