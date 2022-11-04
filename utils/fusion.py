import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Fusion layer for image and point cloud
    Projection argument can be 'bev' or 'front'
    """
    def __init__(self, cloud, projection='bev'):
        super(FusionLayer, self).__init__()
        assert(projection == 'bev' or projection == 'front')
        

    def forward(self):
        pass
