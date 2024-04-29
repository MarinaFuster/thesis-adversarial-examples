"""
This module intends to add functionality to measure quality
brightness and quality sharpness of an image, according to
Amazon Rekognition.

This can be done for a single image or a batch of images.
"""

import os
import json
import statistics
from glob import glob
from infra_aws.rekognition import Rekognition
from experiments.utils import save_rekognition_results

rek = Rekognition()


def measure_image_quality(filepath, results_filename, results_file):

    result = rek.detect_face(filepath)
    save_rekognition_results(result, results_file)

    filename = filepath.split("/")[-1]

    details = result['FaceDetails']

    if len(details) == 0 or len(details) > 1:
        print(f'[ERROR]: Rekognition detected {len(details)} faces for {filename}')
        return 1

    # we already checked that we found only one face
    confidence = details[0]["Confidence"]
    brightness = details[0]["Quality"]['Brightness']
    sharpness = details[0]["Quality"]['Sharpness']

    line = f'{filename},{confidence},{brightness},{sharpness}\n'
    header = 'Filename,Confidence,Brightness,Sharpness\n'
    text = line if os.path.isfile(results_filename) else header+line

    with open(results_filename, 'a') as f:
        f.write(text)

    print(f'Quality measured for {filename}')


def measure_batch_quality(directory, results_filename, results_directory):
    files = glob(f"{directory}/*")
    for filepath in files:
        filename, _ = os.path.splitext(filepath)
        measure_image_quality(
            filepath, results_filename, f'{results_directory}/{filename.split("/")[-1]}.json')


def get_information_from_csv(filepath):
    labels = []
    confidences = []
    brightness = []
    sharpness = []

    with open(filepath, 'r') as f:
        f.readline()  # skips header
        for line in f.readlines():
            elements = line.split(",")
            labels.append(str(elements[0]))
            confidences.append(float(elements[1]))
            brightness.append(float(elements[2]))
            sharpness.append(float(elements[3]))

    return labels, confidences, brightness, sharpness


def get_statistics_from_csv(filepath, statistics_filename):
    _, confidences, brightness, sharpness = get_information_from_csv(filepath)
    with open(statistics_filename, 'w') as f:
        f.write("Measure,Mean,StandardDeviation\n")

        mean_confidence = statistics.mean(confidences) if len(confidences) > 0 else 0
        std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
        f.write(f"Confidence,{mean_confidence},{std_confidence}\n")

        mean_brightness = statistics.mean(brightness) if len(brightness) > 0 else 0
        std_brightness = statistics.stdev(brightness) if len(brightness) > 1 else 0
        f.write(f"Brightness,{mean_brightness},{std_brightness}\n")

        mean_sharpness = statistics.mean(sharpness) if len(sharpness) > 0 else 0
        std_sharpness = statistics.stdev(sharpness) if len(sharpness) > 1 else 0
        f.write(f"Sharpness,{mean_sharpness},{std_sharpness}\n")
