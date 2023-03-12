import cv2
import numpy as np
import scipy
import scipy.fftpack as fftpack
import sklearn.metrics as metrics


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


def normalize(image):
    def normalize_channel(channel, max_across_channels):

        channel /= max_across_channels  # channel.max()

        min = channel.min()
        max = channel.max()
        if min < 0 or max > 1:
            print(f"min = {min}, max = {max}")

        assert min >= 0 and max <= 1

        return channel

    if len(image.shape) == 1 or len(image.shape) == 2:
        return normalize_channel(image)

    assert (len(image.shape) == 3)
    num_channels = image.shape[2]
    global_max = None
    for ch in range(num_channels):
        min = image[:, :, ch].min()
        max = image[:, :, ch].max()

        if min >= 0 and max <= 1:
            continue

        if np.isclose(image[:, :, ch], 0).all():
            continue

        image[:, :, ch] -= min

        if global_max is None or image[:, :, ch].max() > global_max:
            global_max = image[:, :, ch].max()

    for ch in range(num_channels):
        min = image[:, :, ch].min()
        max = image[:, :, ch].max()

        if min >= 0 and max <= 1:
            continue

        if np.isclose(image[:, :, ch], 0).all():
            continue

        image[:, :, ch] = normalize_channel(image[:, :, ch], global_max)

    return image