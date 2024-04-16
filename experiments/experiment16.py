# This "experiment", which is actually NOT an experiment, intends to run
# quality analysis on all images relevant to our thesis.

# First we run this quality analysis on after_autoencoder/ directory, in order
# to get mean and std of confidence of recognition, brightness and sharpness.

# On the other hand. images from experiments 13, 14 and 15 were put in different
# folders and a quality analysis was run for each one of them.

# For now, no further action is being taken, but this is relevant information we need
# in order to analyze if modified images have the same quality as non modified images,
# which is something to take into consideration, given that Rekognition might lower its
# recognition of a person if the image does not have a good quality.

# All utilities to get quality analysis of an image are in infra_aws.quality_control

# Date: 18/08/2021

import matplotlib.pyplot as plt
from infra_aws.quality_control import measure_batch_quality, get_statistics_from_csv
from utils import create_parent_experiment_directory


if __name__ == '__main__':
    print("In order to run this experiment, please read comments carefully.")
    print("If you just need results, ask another team member if they already exist, since AWS will charge you"
          "every time you run this experiment.")

    # Do not run this twice, since it will append results
    # after_autoencoder_results_directory = create_parent_experiment_directory(
    #     "Experiment16_after_autoencoder",
    #     timestamp=False
    # )
    # measure_batch_quality(
    #     '../after_autoencoder',
    #     f'{after_autoencoder_results_directory}/after_autoencoder_quality_detail.csv',
    #     after_autoencoder_results_directory
    # )

    # In order to run this part, you need to create a two_faces folder in results/
    # It happened that Rekognition detected more than one face in some images, which is an odd
    # result since the equivalent on data/ (before autoencoder) recognizes only one face
    # (which is expected result).
    # Create the folder and then add, from data/ (NOT after_autoencoder/) the following images:
    #    - marina8.jpg
    #    - marina11.jpg
    #    - test-marina2.jpg
    #    - nachito17.jpg
    #    - nachito26.jpg
    # two_faces_results_directory = create_parent_experiment_directory(
    #     "Experiment16_two_faces_data_originals",
    #     timestamp=False
    # )
    # measure_batch_quality(
    #     '../results/two_faces',
    #     f'{two_faces_results_directory}/two_faces_quality_detail.csv',
    #     two_faces_results_directory
    # )

    # In order to run this, you'll need to create a folder with AE from experiment 13
    # still_marina_results_directory = create_parent_experiment_directory(
    #     "Experiment16_AE_E13_still_marina",
    #     timestamp=False
    # )
    # measure_batch_quality(
    #     '../results/AE_Experiment13_still_marina',
    #     f'{still_marina_results_directory}/still_marina_quality_detail.csv',
    #     still_marina_results_directory
    # )

    # In order to run this, you'll need to create a folder with AE from experiment 13
    # best_results_directory = create_parent_experiment_directory(
    #     "Experiment16_AE_E13_best",
    #     timestamp=False
    # )
    # measure_batch_quality(
    #     '../results/AE_Experiment13_best',
    #     f'{best_results_directory}/best_quality_detail.csv',
    #     best_results_directory
    # )

    # In order to run this, you'll need to create a folder with AE from experiment 14
    # experiment14_results_directory = create_parent_experiment_directory(
    #     "Experiment16_AE_Experiment14",
    #     timestamp=False
    # )
    # measure_batch_quality(
    #     '../results/AE_Experiment14',
    #     f'{experiment14_results_directory}/experiment14_quality_detail.csv',
    #     experiment14_results_directory
    # )

    # In order to run this, you'll need to create a folder with AE from experiment 15
    # experiment15_results_directory = create_parent_experiment_directory(
    # "Experiment16_AE_Experiment15",
    # timestamp=False
    # )
    # measure_batch_quality(
    #     '../results/AE_Experiment15',
    #     f'{experiment15_results_directory}/experiment15_quality_detail.csv',
    #     experiment15_results_directory
    # )

    # get_statistics_from_csv(
    #     '../results/Experiment16_after_autoencoder/after_autoencoder_quality_detail.csv',
    #     '../results/Experiment16_after_autoencoder/after_autoencoder_statistics.csv'
    # )
