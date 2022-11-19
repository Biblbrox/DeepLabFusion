import open3d as o3d
from matplotlib import cm

from camera import Camera
from model import Model
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as fn
import matplotlib
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import cv2
from utils.visualization import show_front_proj
from utils.cloud_utils import load_cloud
import fusiondataset
import json
import utils.cloud_utils as cloud_utils
import utils.visualization as vis


def cam2image(cloud, K):
    ndim = cloud.ndim
    if ndim == 2:
        cloud = np.expand_dims(cloud, 0)
    points_proj = np.matmul(K[:3, :3].reshape([1, 3, 3]), cloud)
    depth = points_proj[:, 2, :]
    depth[depth == 0] = -1e-6
    u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(np.int)
    v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(np.int)

    if ndim == 2:
        u = u[0]
        v = v[0]
        depth = depth[0]
    return u, v, depth


def test():
    with open("./config.json") as config_file:
        config = json.load(config_file)
        cloud_path = config['cloud_path']
        img_path = config['image_path']
        extrinsic = np.array([
            [0.04307104361, - 0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, - 0.9960641394, - 0.08786966659, -0.1770225824],
            [0, 0, 0, 1]
        ])

        R = np.eye(4)
        R[:3, :3] = np.array(config['rectified']).reshape(3, 3)

        intrinsic = np.array(config['intrinsic']).reshape(3, 4)
        intrinsic = intrinsic[:, 0:3]
        extrinsic = np.linalg.inv(extrinsic)
        extrinsic = np.matmul(R, extrinsic)
        camera = Camera(intrinsic, extrinsic)
        width = 1408
        height = 376
        bev_width = 608
        bev_height = 608

    boundary = {
       "minX": 0,
       "maxX": 50,
       "minY": -25,
       "maxY": 25,
       "minZ": -2.73,
       "maxZ": 1.27
    }
    resolution = (boundary["maxX"] - boundary["minX"]) / bev_height
    params = {
        'width': width,
        'height': height,
        'bev_width': bev_width,
        'bev_height': bev_height,
        'intrinsic': intrinsic,
        'extrinsic': extrinsic,
        'resolution': resolution,
        'roi': boundary
    }

    model = Model(params)
    model.eval()

    dataset = fusiondataset.FusionDataset(cloud_path, img_path)

    cloud_orig, img = dataset[0]

    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32

    intensity = cloud_orig[:, 3].flatten()
    points = cloud_orig[:, :3]
    cloud = o3d.t.geometry.PointCloud(device)
    cloud.point.positions = o3d.core.Tensor(np.asarray(points), dtype, device)
    cloud.point.intensities = o3d.core.Tensor(np.asarray(intensity), dtype, device)
    cloud.point.colors = cloud.point.intensities

    intensity = cloud.point.intensities.numpy()
    max_tot = np.max(intensity)
    source_attribute = intensity / max_tot
    colors_map = cm.get_cmap('jet', 256)
    source_colors = colors_map(source_attribute)
    cloud.point.colors = o3d.core.Tensor(np.asarray(source_colors[:, :3]), dtype)

    # vis.draw_cloud3d(cloud)

    fig, ax = plt.subplots(4, 3, figsize=(13, 13), facecolor='white')
    sub = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    sub.set_title("Original")
    sub.imshow(img)

    with torch.no_grad():
        front, bev, bev_color = model(cloud, img)

    sub = plt.subplot2grid((4, 3), (1, 0), colspan=3)
    sub.imshow(front.color.as_tensor().numpy(), cmap='gray')
    sub.set_title("Front view (Intensity channel)")

    sub = plt.subplot2grid((4, 3), (2, 0), colspan=3)
    sub.imshow(front.depth.as_tensor().numpy(), cmap='gray')
    sub.set_title("Front view (Depth channel)")

    sub = plt.subplot2grid((4, 3), (3, 0), colspan=1)
    sub.imshow(bev[:, :, 0], cmap='gray')
    sub.set_title("Bird's eye view (Intensity channel)")

    sub = plt.subplot2grid((4, 3), (3, 1), colspan=1)

    sub.imshow(bev_color, cmap='gray')
    sub.set_title("Bird's eye view (With colors)")

    sub = plt.subplot2grid((4, 3), (3, 2), colspan=1)
    sub.imshow(bev[:, :, 1], cmap='gray')
    sub.set_title("Bird's eye view (Height channel)")

    plt.tight_layout()
    plt.show()
