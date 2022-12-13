import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def normalize(old_min, old_max, new_min, new_max, val):
    return (val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
