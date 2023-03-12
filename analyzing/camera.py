import numpy as np


class Camera:
    def __init__(self, intrinsic, extrinsic):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

    def to_image_plane(self, points):
        pass
