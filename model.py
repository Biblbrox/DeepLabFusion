import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils.fusion import FusionLayer
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.fusion1 = FusionLayer(params=self.params,
                                   projection='front')
        self.fusion2 = FusionLayer(params=self.params,
                                   projection='bev')

    def forward(self, cloud: np.array, img: np.array):
        fused_front = self.fusion1(cloud.copy(), img.copy())
        fused_bev = self.fusion2(cloud.copy(), img.copy())
        return fused_front, fused_bev
