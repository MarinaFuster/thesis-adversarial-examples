# In this experiment we will manipulate the first  two components since
# it represents 68% of the image information
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
    first_from, second_from, third_from = float(components_from[0]), float(components_from[1]),float(components_from[2])
    first_to, second_to, third_to   = float(components_to[0]), float(components_to[1]), float(components_to[2])
    step_first = abs((first_to - first_from))/10
    step_second = abs((second_to - second_from)) / 10
    step_third = abs((third_to - third_from)) / 10
    if first_to < first_from:
        step_first = step_first * -1

    if second_to < second_from:
        step_second = step_second * -1

    if third_to < third_from:
        step_third = step_third * -1

    NUMBER_OF_STEPS = 11
    current = 0

    while current != NUMBER_OF_STEPS:
        filename = f"../components/nachito9_{current}.csv"
        with open(filename, "w") as file:
            components_from[0] = f"{first_from + step_first * current}\n"
            components_from[1] = f"{second_from + step_second * current}\n"
            components_from[2] = f"{third_from + step_third * current}\n"

            file.writelines(components_from)

        components_to_image(filename)
        current += 1
