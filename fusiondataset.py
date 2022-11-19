import os.path

import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
import numpy as np

import utils.cloud_utils
import utils.image_utils


class FusionDataset(BaseDataset):
    def __init__(self, cloud_path, image_path, cloud_ext='.bin', img_ext='.png', transform_mat=np.array([]), num_digits=10):
        """
        :param cloud_path:
        :param image_path:
        :param cloud_ext:
        :param img_ext:
        :param transform_mat: matrix applied to point cloud in order to project it on image
        :param num_digits:
        """
        super(FusionDataset, self).__init__()
        self.cloud_path = cloud_path
        self.image_path = image_path
        self.cloud_ext = cloud_ext
        self.img_ext = img_ext
        self.num_digits = num_digits
        self.transform_mat = transform_mat

    def __getitem__(self, index):
        """
        Returns pair (cloud, image)
        Returns tuple (front_image, )
        :param index: str
        :return: tuple
        """
        assert (len(str(index)) <= self.num_digits)
        cloud_item_path = os.path.join(self.cloud_path, f"{str(index).zfill(self.num_digits)}{self.cloud_ext}")
        if not os.path.exists(cloud_item_path):
            raise ValueError(f"path {cloud_item_path} doesn't exists")
        img_item_path = os.path.join(self.image_path, f"{str(index).zfill(self.num_digits)}{self.img_ext}")
        if not os.path.exists(img_item_path):
            raise ValueError(f"path {img_item_path} doesn't exists")

        cloud = utils.cloud_utils.load_cloud(cloud_item_path)
        image = utils.image_utils.load_image(img_item_path)

        return cloud, image
