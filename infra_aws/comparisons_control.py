
# We want to send an image (Adversarial Example)  to Amazon Rekognition and get the metrics of the
# metrics of said adversarial example.

from infra_aws.rekognition import Rekognition
from experiments.utils import create_experiment_directory, create_subexperiment_directory
import os
import statistics
from glob import glob


def compute_adversarial_example_metrics(original, matches, detail=None):
    """
    Computes results for face matches from rekognition.
    Original is the name of the image. We need it since we do not want
    to compare an image against itself.
    Matches is a dictionary obtained from Rekognition.
    """

    result = {
        "marina": {
            "score": 0,
            "confidences": [],
            "similarities": [],
        },
        "nachito": {
            "score": 0,
            "confidences": [],
            "similarities": [],
        },
        "quality": {
            "sharpness": 0,
            "brightness": 0,
        }
    }
    for match in matches:

        if "_processed" in match["Face"]["ExternalImageId"]:
            match_name = match["Face"]["ExternalImageId"].split("_processed")[0]
        else:
            match_name = match["Face"]["ExternalImageId"].split(".")[0]

        # This case is added since we do not want to compare the image to itself. This will give
        # a similarity of 100% which will "inflate" a little bit the results we are trying to gather.
        if original == match_name:
            print(f"Bypassing {original} with {match_name} since it's the same image.")
            target = None
        elif "marina" in match_name:
            target = "marina"
        elif "nachito" in match_name:
            target = "nachito"
        else:
            raise ValueError(f'{match_name} is neither marina or nachito')

        # Only adds a new result when target is marina or nachito
        if target:
            result[target]["confidences"].append(float(match["Face"]["Confidence"]))
            result[target]["similarities"].append(float(match["Similarity"]))
            result[target]["score"] += 1

    print(detail)
    result["quality"]["sharpness"] = detail["Quality"]["Sharpness"]
    result["quality"]["brightness"] = detail["Quality"]["Brightness"]
    return result


def get_adversarial_example_metrics(filepath: str, face_threshold=0.0, root_dir=None):
    """
    Receives the filepath to the adversarial image and saves the information of the adversarial
    examples in a special experiment file.
    """
    rek = Rekognition()

    res = rek.compare_face_to_rek_collection_from_local(filepath, face_threshold=face_threshold)
    matches = res["FaceMatches"]

    filename = filepath.split('/')[-1].split(".")[0]
    result = compute_adversarial_example_metrics(filename, matches)
    if not root_dir:
        target_dir = create_experiment_directory(f"AE_{filename}")
    else:
        target_dir = create_subexperiment_directory(root_dir, f"AE_{filename}")

    with open(f"{target_dir}/AE_metrics.csv", "w") as csv_file:
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

    output_file = open(f"{target_dir}/{filename}_detail.json", "w")
    output_file.write(str(res))
    os.system(f"cp {filepath} '{target_dir}'")

def get_adversarial_example_metrics2(filepath: str, face_threshold=0.0, root_dir=None):
    """
    Receives the filepath to the adversarial image and saves the information of the adversarial
    examples in a special experiment file.
    """
    rek = Rekognition()

    res = rek.compare_face_to_rek_collection_from_local(filepath, face_threshold=face_threshold)
    detail = rek.detect_face(filepath)["FaceDetails"]
    matches = res["FaceMatches"]

    filename = filepath.split('/')[-1].replace(".png", "")

    if len(detail) != 1:
        print(f"Ignoring {filepath}... too many faSceDetects")
        return None

    result = compute_adversarial_example_metrics(filename, matches, detail[0])

    if not root_dir:
        target_dir = create_experiment_directory(f"AE_{filename}")
    else:
        component = filename.split("_")[1]
        target_dir = create_subexperiment_directory(root_dir, f"comp_{component}")

    with open(f"{target_dir}/AE_metrics_{filename}.csv", "w") as csv_file:
        csv_file.write(f"target,score,mean_confidence,std_confidence,mean_similarity,std_similarity,quality_sharpness,quality_brightness\n")

        for target in ["marina", "nachito"]:

            confidences = result[target]["confidences"]
            mean_confidence = statistics.mean(confidences) if len(confidences) > 0 else 0
            std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0

            similarities = result[target]["similarities"]
            mean_similarity = statistics.mean(similarities) if len(similarities) > 0 else 0
            std_similarity = statistics.stdev(similarities) if len(similarities) > 1 else 0

            csv_file.write(
                f"{target},{result[target]['score']},{mean_confidence},\
                {std_confidence},{mean_similarity},{std_similarity},\
                {result['quality']['sharpness']},{result['quality']['brightness']}\n"
            )

    output_file = open(f"{target_dir}/{filename}_detail.json", "w")
    output_file.write(str(res))
    os.system(f"cp {filepath} '{target_dir}'")


if __name__ == '__main__':
    # filenames = glob("../results/Experiment15/*.png")
    # for images in filenames:
    #     get_adversarial_example_metrics(images)

    # for image in filenames:
    #     get_adversarial_example_metrics(image, root_dir="../results/Experiment15")

    filenames = glob("../results/Experiment17/*.png")

    for image in filenames:
        get_adversarial_example_metrics2(image, root_dir="../results/Experiment17")
    # get_adversarial_example_metrics2(filenames[0], root_dir="../results/Experiment17")