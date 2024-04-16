# This experiment intends to gather benchmark metrics
# that will be useful to analyze later adversarial examples.

# Date 02/08/2021
# Updated Date 12/08/2021 (fix metrics method)

# Important! in order to run this experiment, you must have
# AE_directories_threshold_0 and AE_directories_threshold_50
# folders inside results directory.
# You can find a compressed version of these on:
# Tesis -> Ejemplos Adversarios Experimentos -> 02.08 - E11 Ground Truth (Metrics of Non AE examples)

import copy
import json
import statistics
from glob import glob
from infra_aws.comparisons_control import compute_adversarial_example_metrics


# DO NOT UNCOMMENT: these will generate ae_metrics for every image, consuming aws resources
# from infra_aws.comparisons_control import get_adversarial_example_metrics
# filenames = glob('../after_autoencoder/*')
# for filename in filenames:
#     get_adversarial_example_metrics(filename)


def add_metrics(metrics_dict, marina_data, nachito_data):
    # target,score,mean_confidence,std_confidence,mean_similarity,std_similarity
    marina_values = marina_data.split(",")
    nachito_values = nachito_data.split(",")
    metrics_dict["m_scores"].append(int(marina_values[1]))
    metrics_dict["m_confidences"].append(float(marina_values[2]))
    metrics_dict["m_similarities"].append(float(marina_values[4]))
    metrics_dict["n_scores"].append(float(nachito_values[1]))
    metrics_dict["n_confidences"].append(float(nachito_values[2]))
    metrics_dict["n_similarities"].append(float(nachito_values[4]))


# folder must be in results
def generate_benchmarks(folder, benchmark_filename):
    directories = glob(f'../results/{folder}/*')

    template = {
        "m_confidences": [],  # values inside list are averages for each image
        "m_similarities": [],  # values inside list are averages for each image
        "m_scores": [],
        "n_confidences": [],  # values inside list are averages for each image
        "n_similarities": [],  # values inside list are averages for each image
        "n_scores": []
    }

    targets = ["marina", "nachito"]
    metrics = {
        targets[0]: copy.deepcopy(template),
        targets[1]: copy.deepcopy(template)
    }

    for directory in directories:
        target = directory.split("/")[-1].split("_")[1]
        with open(f'{directory}/AE_metrics_fixed.csv', 'r') as f:
            lines = f.readlines()
            if "marina" in target:
                add_metrics(metrics["marina"], lines[1], lines[2])
            elif "nachito" in target:
                add_metrics(metrics["nachito"], lines[1], lines[2])
            else:
                raise ValueError("Target is neither marina nor nachito.")

    with open(f'../results/{benchmark_filename}.csv', 'w') as f:

        f.write(
            "target,"
            "m_mean_confidence,m_mean_similarity,m_mean_score,"
            "n_mean_confidence,n_mean_similarity,n_mean_score\n"
        )

        # marina metrics for marina images
        m_mean_confidence = statistics.mean(metrics["marina"]["m_confidences"])
        m_mean_similarity = statistics.mean(metrics["marina"]["m_similarities"])
        m_mean_score = statistics.mean(metrics["marina"]["m_scores"])

        # nachito metrics for marina images
        n_mean_confidence = statistics.mean(metrics["marina"]["n_confidences"])
        n_mean_similarity = statistics.mean(metrics["marina"]["n_similarities"])
        n_mean_score = statistics.mean(metrics["marina"]["n_scores"])

        f.write(
            f"marina,"
            f"{m_mean_confidence},{m_mean_similarity},{m_mean_score},"
            f"{n_mean_confidence},{n_mean_similarity},{n_mean_score}\n"
        )

        # marina metrics for nachito images
        m_mean_confidence = statistics.mean(metrics["nachito"]["m_confidences"])
        m_mean_similarity = statistics.mean(metrics["nachito"]["m_similarities"])
        m_mean_score = statistics.mean(metrics["nachito"]["m_scores"])

        # nachito metrics for nachito images
        n_mean_confidence = statistics.mean(metrics["nachito"]["n_confidences"])
        n_mean_similarity = statistics.mean(metrics["nachito"]["n_similarities"])
        n_mean_score = statistics.mean(metrics["nachito"]["n_scores"])

        f.write(
            f"nachito,"
            f"{m_mean_confidence},{m_mean_similarity},{m_mean_score},"
            f"{n_mean_confidence},{n_mean_similarity},{n_mean_score}\n"
        )


generate_benchmarks("AE_threshold_0", "benchmarks_threshold_0_fixed")
generate_benchmarks("AE_threshold_50", "benchmarks_threshold_50_fixed")


# I should run here a method to fix ae_metrics.csv for each folder
# I should update method for gathering ae_metrics.csv ???? in case
# we are running metrics with images again.
def fix_benchmark_metrics(folder):
    directories = glob(f'../results/{folder}/*')

    for directory in directories:
        target = directory.split("/")[-1].split("_")[1]
        print(f'Working on target {target}')
        with open(f'{directory}/{target}_detail.json', 'r') as json_file:
            json_string = str(json_file.read()).replace("'", '"')
            json_object = json.loads(json_string)
            result = compute_adversarial_example_metrics(target, json_object["FaceMatches"])

            with open(f"{directory}/AE_metrics_fixed.csv", "w") as csv_file:
                csv_file.write(f"target,score,mean_confidence,std_confidence,mean_similarity,std_similarity\n")

                for target in ["marina", "nachito"]:
                    confidences = result[target]["confidences"]
                    mean_confidence = statistics.mean(confidences) if len(confidences) > 0 else 0
                    std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0

                    similarities = result[target]["similarities"]
                    mean_similarity = statistics.mean(similarities) if len(similarities) > 0 else 0
                    std_similarity = statistics.stdev(similarities) if len(similarities) > 1 else 0

                    csv_file.write(
                        f"{target},{result[target]['score']},{mean_confidence},\
                        {std_confidence},{mean_similarity},{std_similarity}\n"
                    )


#fix_benchmark_metrics("AE_threshold_0")
#fix_benchmark_metrics("AE_threshold_50")
