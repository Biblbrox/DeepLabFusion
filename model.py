import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.deeplabv3 = smp.DeepLabV3Plus()

    def forward(self, img):
        res = self.deeplabv3(img)
        return res
