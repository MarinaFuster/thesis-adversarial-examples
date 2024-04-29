import cv2
import numpy as np
from glob import glob
import logging


logger = logging.getLogger("DATA_LOADER")

test_prefix = "test-"

def read_image(filename, mode='gray'):
    if mode == 'color':
        im = cv2.imread(filename)
    else:
        im = cv2.imread(filename, 0)
        im = np.expand_dims(im, 2)
    return im


def load_data(data_prefix=None, train_folder='data', mode='color', split_test=True):
    # The `or` is needed because while testing the context isn't root
    filenames = \
        glob(f"{train_folder}/*") or glob(f"../{train_folder}/*") if data_prefix is None else \
        glob(f"{train_folder}/{data_prefix}*") or glob(f"../{train_folder}/{data_prefix}*")

    if not filenames:
        logger.error("Couldn't find any suitable images...")
        raise FileNotFoundError("Invalid prefix")

    train_set = []
    test_set = []
    train_labels = []
    test_labels = []

    for filename in filenames:
        im = read_image(filename, mode)
        fn = filename.split('/')[-1]

        if not split_test or not filename.split('/')[-1].startswith(test_prefix):
            train_set.append(im)
            train_labels.append(fn)
        else:
            test_set.append(im)
            test_labels.append(fn)

    train_set = np.array(train_set)
    test_set = np.array(test_set)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    if split_test:
        return train_set, test_set, train_labels, test_labels
    else:
        return train_set, train_labels
