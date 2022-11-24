import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms
from scipy import signal, ndimage
import open3d as o3d
import numpy.linalg as lin
import scipy.optimize as opt
from utils.cloud_utils import make_bev_color, make_bev_height_intensity, make_front_proj
from utils.cloud_utils import np_to_o3d_cloud
import segmentation_models_pytorch as smp
import torchvision.models as models


def to01(image):
    image += np.abs(np.min(image))
    image *= 1.0 / np.max(image)
    return image


class Decomposer:
    def __init__(self, lam=5):
        self.lam = lam

    def find_detail(self, image):
        gx = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ])
        gy = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])

        convx = signal.convolve2d(image, gx, boundary='symm', mode='same')
        convy = signal.convolve2d(image, gy, boundary='symm', mode='same')
        conv = image + self.lam * (convx ** 2 + convy ** 2)
        conv = to01(conv)
        return conv

    def __call__(self, image: np.array):
        # Obtain base part
        channels = image.shape[2]
        if np.max(image) > 1.0:
            image *= 1.0 / np.max(image)

        base = np.zeros((image.shape[0], image.shape[1], channels), dtype=image.dtype)
        detail = np.zeros((image.shape[0], image.shape[1], channels), dtype=image.dtype)
        for ch in range(channels):
            I_d = self.find_detail(image[:, :, ch])
            I_b = np.maximum(image[:, :, ch] - I_d, 0)
            base[:, :, ch] = I_b
            detail[:, :, ch] = I_d

        return base, detail


def add_channel(cloud_):
    cloud = np.zeros((cloud_.shape[0], cloud_.shape[1], 3), dtype=np.float32)
    cloud[:, :, 0] = cloud_[:, :, 0]
    cloud[:, :, 1] = cloud_[:, :, 1]
    cloud[:, :, 2] = np.maximum(cloud_[:, :, 0], cloud_[:, :, 1])
    cloud[:, :, 2] = to01(cloud[:, :, 2])
    return cloud


def fuse_base(cloud: np.array, image: np.array, alpha1=0.5, alpha2=0.5):
    fused = np.zeros((cloud.shape[0], cloud.shape[1], cloud.shape[2]))
    fused[:, :, 0] = alpha1 * np.maximum(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]) + alpha2 * image[:, :, 0]
    fused[:, :, 1] = alpha1 * np.maximum(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]) + alpha2 * image[:, :, 1]
    fused[:, :, 2] = alpha1 * np.maximum(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]) + alpha2 * image[:, :, 2]

    return fused


class DetailFusion(nn.Module):

    def __init__(self, img_det1: np.array, img_det2: np.array):
        super(DetailFusion, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.encoder = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        tensor_shape = (img_det1.shape[1], img_det1.shape[0], img_det1.shape[2])
        self.img_det1, self.img_det2 = np.reshape(img_det1, tensor_shape), np.reshape(img_det2, tensor_shape)
        self.img_det1, self.img_det2 = self.transform(self.img_det1), self.transform(self.img_det2)

    def forward(self):
        input1 = self.img_det1[None]
        input2 = self.img_det2[None]

        map1 = self.encoder(input1)
        map2 = self.encoder(input2)

        return map1, map2


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

    def forward(self, cloud: np.array, image: np.array):
        rgbd_image = np_to_o3d_cloud(cloud).project_to_rgbd_image(self.width, self.height, self.intrinsic,
                                                                  self.extrinsic,
                                                                  depth_max=10000, depth_scale=1)
        rgbd_image.color = o3d.t.geometry.Image(image)
        color_cloud = o3d.t.geometry.PointCloud()
        color_cloud = color_cloud.create_from_rgbd_image(rgbd_image, self.intrinsic, self.extrinsic, depth_max=10000,
                                                         depth_scale=1)
        color_cloud = np.c_[color_cloud.point.positions.numpy(), color_cloud.point.colors.numpy()]
        bev_color = make_bev_color(color_cloud, self.resolution, self.bev_height, self.bev_width,
                                   self.roi)
        bev = make_bev_height_intensity(cloud, self.resolution, self.bev_height, self.bev_width,
                                        self.roi)

        base_cloud, detail_cloud = Decomposer()(bev)
        base_img, detail_img = Decomposer()(bev_color)

        base_cloud = add_channel(base_cloud)
        detail_cloud = add_channel(detail_cloud)
        fused_base = fuse_base(base_cloud, base_img)

        return {
            'bev': bev,
            'bev_color': bev_color,
            'base_cloud': base_cloud,
            'detail_cloud': detail_cloud,
            'fused_base': fused_base
        }


class FrontImageFusion(nn.Module):
    def __init__(self, params):
        super(FrontImageFusion, self).__init__()
        self.width = params['width']
        self.height = params['height']
        self.intrinsic = params['intrinsic']
        self.extrinsic = params['extrinsic']

    def forward(self, cloud, image):
        front, colors = make_front_proj(cloud, self.width, self.height, self.intrinsic, self.extrinsic)
        base_cloud, detail_cloud = Decomposer()(front)
        # base_img, detail_img = Decomposer()(image)
        base_img, detail_img = Decomposer()(colors)
        base_cloud = add_channel(base_cloud)
        detail_cloud = add_channel(detail_cloud)
        fused_base = fuse_base(base_cloud, base_img)

        detail_fusion = DetailFusion(detail_img, detail_cloud)
        res = detail_fusion()
        print(res)

        return {
            'front': front,
            'image': image,
            'base_image': base_img,
            'detail_image': detail_img,
            'base_cloud': base_cloud,
            'detail_cloud': detail_cloud,
            'fused_base': fused_base
        }


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

    def forward(self, cloud: np.array, image: np.array):
        return self.fusion(cloud, image)
