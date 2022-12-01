import cv2
import numpy as np


def block_mean(arr: np.array, block_size=1):
    """
    Assume that arr in HSV color space
    :param arr:
    :param block_size:
    :return:
    """
    if np.size(arr.shape) != 2:
        print(f"Array shape in block_mean must be 2 dimensions, not {arr.shape}")
        assert (np.size(arr.shape) == 2)

    filtered = cv2.blur(arr, (block_size, block_size))

    return filtered
