import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils.fusion import FusionLayer
import numpy as np


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.fusion1 = FusionLayer(params=self.params,
                                   projection='front')
        self.fusion2 = FusionLayer(params=self.params,
                                   projection='bev')
        # self.deeplabv3 = smp.DeepLabV3Plus()

    def forward(self, cloud, img):
        front = self.fusion1(cloud, img)
        bev, bev_color = self.fusion2(cloud, img)
        # res = self.deeplabv3(img)
        return front, bev, bev_color
