import torch.nn as nn
import numpy as np
import open3d as o3d
from utils.cloud_utils import make_bev_color, make_bev_height_intensity, make_front_proj

"""
Expected type of point cloud is [X, Y, Z (,I)]
"""

"""
Build feature map from cloud and image
Try 1 RGBD
"""


class BevImageFusion(nn.Module):
    def __init__(self, params):
        super(BevImageFusion, self).__init__()
        self.bev_width = params['bev_width']
        self.bev_height = params['bev_height']
        self.width = params['width']
        self.height = params['height']
        self.resolution = params['resolution']
        self.roi = params['roi']
        self.extrinsic = params['extrinsic']
        self.intrinsic = params['intrinsic']

    def forward(self, cloud: o3d.t.geometry.PointCloud, image: np.ndarray):
        rgbd_image = cloud.project_to_rgbd_image(self.width, self.height, self.intrinsic,
                                                 self.extrinsic,
                                                 depth_max=10000, depth_scale=1)
        rgbd_image.color = o3d.t.geometry.Image(image)
        color_cloud = o3d.t.geometry.PointCloud()
        color_cloud = color_cloud.create_from_rgbd_image(rgbd_image, self.intrinsic, self.extrinsic, depth_max=10000,
                                                         depth_scale=1)
        bev_color = make_bev_color(color_cloud, self.resolution, self.bev_height, self.bev_width,
                                   self.roi)
        bev = make_bev_height_intensity(cloud, self.resolution, self.bev_height, self.bev_width,
                                        self.roi)
        return bev, bev_color


class FrontImageFusion(nn.Module):
    def __init__(self, params):
        super(FrontImageFusion, self).__init__()
        self.width = params['width']
        self.height = params['height']
        self.intrinsic = params['intrinsic']
        self.extrinsic = params['extrinsic']

    def forward(self, cloud: np.ndarray, image: np.ndarray):
        front = make_front_proj(cloud, self.width, self.height, self.intrinsic, self.extrinsic)
        return front


class FusionLayer(nn.Module):
    """
    Fusion layer for image and point cloud
    Projection argument can be 'bev' or 'front'
    """

    def __init__(self, params: dict, projection='bev'):
        super(FusionLayer, self).__init__()
        assert (projection == 'bev' or projection == 'front')
        assert ('bev_width' in params)
        assert ('bev_height' in params)
        assert ('resolution' in params)
        assert ('roi' in params)
        assert ('width' in params)
        assert ('height' in params)
        assert ('intrinsic' in params)
        assert ('extrinsic' in params)

        if projection == 'bev':
            self.fusion = BevImageFusion(params)
        else:
            self.fusion = FrontImageFusion(params)

    def forward(self, cloud, image):
        return self.fusion(cloud, image)
