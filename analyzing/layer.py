import numpy as np


class Layer:
    def __init__(self, feature_map: np.array):
        assert (type(feature_map).__module__ == np.__name__)
        self.feature_map = feature_map
        self.shape = self.feature_map.shape
