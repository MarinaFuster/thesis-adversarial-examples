import glob
import json
import re
import sys

FACE_MATCHES = "FaceMatches"
FACE = "Face"
EXTERNAL_IMAGE_ID = "ExternalImageId"


def is_face_matches(element):
    return FACE_MATCHES == element


def is_external_image_id(element):
    return EXTERNAL_IMAGE_ID == element


def get_label(external_image_id):
    regex = re.compile("([a-zA-Z]+)([0-9]+)")
    return regex.match(external_image_id.split(".")[0]).groups()[0]


def process_comparison(filename):
    with open(filename, 'r') as data_file:
        data = json.load(data_file)
        data = {key: value for key, value in data.items() if is_face_matches(key)}
        for face in data[FACE_MATCHES]:
            face[FACE] = \
                get_label(
                    {key: value for key, value in face[FACE].items() if is_external_image_id(key)}[EXTERNAL_IMAGE_ID]
                )
    with open(filename, 'w') as data_file:
        json.dump(data, data_file, indent=2)


def process_comparison_batch(folder):
    files = glob.glob(f"{folder}*")
    for file in files: process_comparison(file)


SINGLE = '-s'  # for single
BATCH = '-b'  # for batch

OPTION = 1
PATH = 2


def main():
    args = sys.argv
    if len(args) == 1:
        print("You should use -c or -i option with a filename. Check README for examples.")
        exit(1)
    if args[OPTION] == SINGLE:
        process_comparison(args[PATH])
    elif args[OPTION] == BATCH:
        process_comparison_batch(args[PATH])
    else:
        print("Invalid option.")
        exit(1)


if __name__ == '__main__':
    main()
