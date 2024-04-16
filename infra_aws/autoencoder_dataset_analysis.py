import csv
import sys
from os import path
from glob import glob
from infra_aws.rekognition import Rekognition
import numpy as np


def _get_datasets():
    # return [f"ID_{x}_{y}" for x in range(1, 5) for y in range(0, 5)]
    return [f"ID_1_{x}" for x in range(0, 2)]


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print("You should at least specify 1 dataset")
        print(f"Usage: python {args[0].split('/')[-1]} (Dataset_1, Dataset_2, ..., Dataset_n)]")
        exit(1)

    rekognition = Rekognition()
    datasets = _get_datasets()

    root_folder = "cross_validation"
    with open(f"../datasets_summaryC.csv", "w") as summary_file:
        summary_writer = csv.writer(summary_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        summary_writer.writerow(["Dataset", "Mean Confidence", "STD Confidence", "Mean Comparison",
                                 "STD Comparison", "Mean Brightness", "STD Brightness",
                                 "Mean Sharpness", "STD Sharpness"])

        fields_dict = {"confidence": [], "comparison": [], "brightness": [], "sharpness": []}
        for ds in datasets:
            comparisons = []
            file_path = f"{root_folder}/{ds}"
            if not path.isdir(file_path):
                print(f"ERROR: {file_path} is not a valid directory")
                exit(1)

            with open(f"{file_path}/result.csv", "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(["Filename", "Confidence", "Comparison", "Quality Brightness", "Quality Sharpness"])
                filenames = glob(f"{file_path}/data/*.jpg")
                for fn in filenames:
                    relative_name=fn.split('/')[-1]
                    target_image = f"{file_path}/recon/{relative_name}"
                    if not path.exists(target_image):
                        print(f"ERROR: Couldn't find target image {target_image}")
                        exit(1)

                    face_matches = rekognition.compare_face(fn, target_image)
                    for face in face_matches:
                        quality = face["Face"]["Quality"]
                        confidence = face["Face"]["Confidence"]
                        fields_dict["comparison"].append(face["Similarity"])
                        fields_dict["confidence"].append(confidence)
                        fields_dict["brightness"].append(quality["Brightness"])
                        fields_dict["sharpness"].append(quality["Sharpness"])
                        writer.writerow([relative_name, confidence, face["Similarity"], quality["Brightness"], quality["Sharpness"]])

            comparisons = np.array(fields_dict["comparison"])
            mean_comp, std_comp = comparisons.mean(), comparisons.std()

            confidences = np.array(fields_dict["confidence"])
            mean_conf, std_conf = confidences.mean(), confidences.std()

            brightnesses = np.array(fields_dict["brightness"])
            mean_br, std_br = brightnesses.mean(), brightnesses.std()

            sharpnesses = np.array(fields_dict["sharpness"])
            mean_sh, std_sh = sharpnesses.mean(), sharpnesses.std()

            summary_writer.writerow([ds, mean_conf, std_conf, mean_comp, std_comp, mean_br,
                                     std_br, mean_sh, std_sh])
