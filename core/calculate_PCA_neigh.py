import sys
from glob import glob
import numpy as np


def process_csv(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
        # Remove the trailing \n before returning
        return [np.float(x.rstrip()) for x in lines]


# Given TWO patterns, we calculate the two images whose PCA components
# are closest to each other (they are neighbors ! ).
#
# For this to work pattern1 and pattern2 should match to a series of
# csv files generated in --batch mode by `transformation_flow.py`.
# Said csv files are the principal components of the images
if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print(f"Usage: {args[0]} pattern1 pattern2")
        exit(0)

    is_first = True
    min_distance = None
    pattern1 = args[1]
    pattern2 = args[2]
    files1 = glob(pattern1)
    files2 = glob(pattern2)
    for f1 in files1:
        c1 = np.array(process_csv(f1))
        for f2 in files2:
            c2 = np.array(process_csv(f2))
            distance = np.linalg.norm(c1 - c2)
            if is_first or distance < min_distance:
                names = (f1, f2) # For reference
                min_distance = distance

    print(f"MIN DISTANCE: {min_distance}. {names} are neighbors")
