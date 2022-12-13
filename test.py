import math

import open3d as o3d
from matplotlib import cm
from scipy.signal import convolve2d

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


def estimate_noise(image):
    total_noise = 0
    for ch in range(image.shape[2]):
        I = image[:, :, ch]
        I = np.reshape(I, (I.shape[1], I.shape[0]))

        H, W = I.shape[0], I.shape[1]

        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
        total_noise += sigma

    return total_noise


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

    cloud, img = dataset[0]

    shape = (9, 3)
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(13, 13), facecolor='white')
    sub = plt.subplot2grid(shape, (0, 0), colspan=3)
    sub.set_title("Original")
    sub.imshow(img)

    with torch.no_grad():
        fused_front, fused_bev = model(cloud, img)

    front = fused_front['front']
    base_img = fused_front['base_image']
    detail_img = fused_front['detail_image']
    front_base_cloud = fused_front['base_cloud']
    front_detail_cloud = fused_front['detail_cloud']
    front_fused_base = fused_front['fused_base']
    front_fused_detail = fused_front['fused_detail']

    print(f"Original image noise: {estimate_noise(img)}")
    print(f"Original front cloud noise: {estimate_noise(front)}")
    print(f"Fused base noise: {estimate_noise(front_fused_base)}")

    bev_base_cloud = fused_bev['base_cloud']
    bev_detail_cloud = fused_bev['detail_cloud']
    bev_fused_base = fused_bev['fused_base']
    bev_fused_detail = fused_bev['fused_detail']

    bev = fused_bev['bev']
    bev_color = fused_bev['bev_color']

    path = "content/"

    cv2.imwrite(f"{path}/kitti_image.png", cv2.cvtColor(255 * img, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (1, 0), colspan=3)
    sub.imshow(front[:, :, 0], cmap='gray')
    sub.set_title("Front view (Intensity channel)")
    cv2.imwrite(f"{path}/front_intensity.png", front[:, :, 0] * 255)

    sub = plt.subplot2grid(shape, (2, 0), colspan=3)
    sub.imshow(front[:, :, 1], cmap='gray')
    sub.set_title("Front view (Depth channel)")
    cv2.imwrite(f"{path}/front_depth.png", 255 * front[:, :, 1])

    sub = plt.subplot2grid(shape, (3, 0), colspan=1)
    sub.imshow(bev[:, :, 0], cmap='gray')
    sub.set_title("Bird's eye view (Intensity channel)")
    cv2.imwrite(f"{path}/bev_intensity.png", 255 * bev[:, :, 0])

    sub = plt.subplot2grid(shape, (3, 1), colspan=1)
    sub.imshow(bev_color)
    sub.set_title("Bird's eye view (With colors)")
    cv2.imwrite(f"{path}/bev_colors.png", cv2.cvtColor(255 * bev_color, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (3, 2), colspan=1)
    sub.imshow(bev[:, :, 1], cmap='gray')
    sub.set_title("Bird's eye view (Height channel)")
    cv2.imwrite(f"{path}/bev_height.png", 255 * bev[:, :, 1])

    sub = plt.subplot2grid(shape, (4, 0), colspan=1)
    sub.imshow(front_base_cloud, cmap='gray')
    sub.set_title("Front cloud base")
    cv2.imwrite(f"{path}/front_base.png", cv2.cvtColor(front_base_cloud * 255, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (4, 1), colspan=1)
    sub.imshow(front_detail_cloud, cmap='gray')
    sub.set_title("Front cloud detail")
    cv2.imwrite(f"{path}/front_detail.png", cv2.cvtColor(255 * front_detail_cloud, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (4, 2), colspan=1)
    sub.imshow(front_fused_base)
    sub.set_title("Front cloud fused with image (base)")
    cv2.imwrite(f"{path}/front_fused_with_image_base.png",
                255 * cv2.cvtColor(front_fused_base.astype('float32'), cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (5, 0), colspan=1)
    sub.imshow(base_img, cmap='gray')
    sub.set_title("Front image base")
    cv2.imwrite(f"{path}/image_base.png", cv2.cvtColor(255 * base_img, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (5, 1), colspan=1)
    sub.imshow(detail_img)
    sub.set_title("Front image detail")
    cv2.imwrite(f"{path}/image_detail.png", cv2.cvtColor(255 * detail_img, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (6, 0), colspan=1)
    sub.imshow(bev_base_cloud, cmap='gray')
    sub.set_title("Bev cloud base")
    cv2.imwrite(f"{path}/bev_base.png", cv2.cvtColor(bev_base_cloud * 255, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (6, 1), colspan=1)
    sub.imshow(bev_detail_cloud, cmap='gray')
    sub.set_title("Bev cloud detail")
    cv2.imwrite(f"{path}/bev_detail.png", cv2.cvtColor(255 * bev_detail_cloud, cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (6, 2), colspan=1)
    sub.imshow(bev_fused_base)
    sub.set_title("Bev cloud fused with image (base)")
    cv2.imwrite(f"{path}/bev_fused_with_image_base.png",
                255 * cv2.cvtColor(bev_fused_base.astype('float32'), cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (7, 0), colspan=3)
    print(front_fused_detail.shape)
    front_fused_detail = np.reshape(front_fused_detail, (
    front_fused_detail.shape[2], front_fused_detail.shape[1], front_fused_detail.shape[0]))
    sub.imshow(front_fused_detail)
    sub.set_title("Front fused detail")
    cv2.imwrite(f"{path}/front_fused_with_image_detail.png",
                255 * cv2.cvtColor(front_fused_detail.astype('float32'), cv2.COLOR_RGB2BGR))

    sub = plt.subplot2grid(shape, (8, 0), colspan=3)
    bev_fused_detail = np.reshape(bev_fused_detail, (
        bev_fused_detail.shape[2], bev_fused_detail.shape[1], bev_fused_detail.shape[0]))
    sub.imshow(bev_fused_detail)
    sub.set_title("Bev cloud fused with image (detail)")
    cv2.imwrite(f"{path}/bev_fused_with_image_detail.png",
                255 * cv2.cvtColor(bev_fused_detail.astype('float32'), cv2.COLOR_RGB2BGR))

    # sub = plt.subplot2grid(shape, (7, 1), colspan=1)
    # sub.imshow(map2)
    # sub.set_title("Feature map2")

    plt.tight_layout()
    plt.show()
