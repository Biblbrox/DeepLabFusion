import os.path

import numpy as np
import open3d as o3d

from utils import image_utils

CLOUD_TYPES = [
    "xyz",
    "xyzd",
    "xyzdi",
    "xyzid",
    "xyzrgb",
    "xyzrgbd",
    "xyzrgbdi"
]


def make_front_proj(cloud, width, height, intrinsic, extrinsic):
    projection = cloud.project_to_rgbd_image(width, height, intrinsic, extrinsic,
                                             depth_max=10000)
    return projection


def transform_cloud(cloud, transform):
    assert transform.shape == (3, 4)
    cutted = cloud[:, 0:3]
    cutted = np.matmul(transform.T, cutted.T)
    cutted = np.c_[cutted, cloud[:, 3]]
    return cutted


def cam2image(points, intrinsic):
    intrinsic = np.array([[788.629315, 0.000000, 687.158398],
                          [0.000000, 786.382230, 317.752196], [0.000000, 0.000000, 0.000000]])
    ndim = points.ndim
    if ndim == 2:
        points = np.expand_dims(points, 0)
    points_proj = np.matmul(intrinsic[:3, :3].reshape([1, 3, 3]), points)
    depth = points_proj[:, 2, :]
    depth[depth == 0] = -1e-6
    u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(np.int)
    v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(np.int)

    if ndim == 2:
        u = u[0]
        v = v[0]
        depth = depth[0]
    return u, v, depth


def project_to_image(image: np.array, cloud: np.array, projection_mat: np.array, empty_color=(0, 0, 0)):
    """
    Project point cloud to image, make 3d cloud with colors from the image
    :param projection_mat:
    :param cloud:
    :param image:
    :param empty_color:
    :return:
    """
    #    assert projection_mat.shape == (4, 4)
    #    assert 3 <= cloud.shape[0] <= 8
    cloud = np.reshape(cloud, (cloud.shape[1], cloud.shape[0]))
    cloud = np.delete(cloud, 1, 0)
    print(cloud.shape)
    projection = np.zeros(image.shape)
    u, v, depth = cam2image(cloud, [])
    print(u.shape)
    print(v.shape)
    print(depth.shape)

    return projection


def cut_fov_front(front_proj, fov):
    """
    Cut field of view front view cilindric projection of point cloud
    :return:
    """
    pass


def filter_roi_cloud(cloud: o3d.t.geometry.PointCloud, roi):
    # Boundary condition
    minX = roi["minX"]
    maxX = roi["maxX"]
    minY = roi["minY"]
    maxY = roi["maxY"]
    minZ = roi["minZ"]
    maxZ = roi["maxZ"]

    # Remove the point out of range x,y,z
    mask = np.where(
        (cloud[:, 0] >= minX)
        & (cloud[:, 0] <= maxX)
        & (cloud[:, 1] >= minY)
        & (cloud[:, 1] <= maxY)
        & (cloud[:, 2] >= minZ)
        & (cloud[:, 2] <= maxZ)
    )
    cloud = cloud[mask]

    cloud[:, 2] = cloud[:, 2] - minZ

    return cloud


def make_bev_color(cloud_: o3d.t.geometry.PointCloud, resolution, height_, width_, roi):
    height = height_ + 1
    width = width_ + 1

    # Discretize Feature Map
    cloud: o3d.t.geometry.PointCloud = cloud_.clone()

    points = cloud.point.positions.numpy()
    points = np.c_[points, cloud.point.colors.numpy()]
    points = filter_roi_cloud(points, roi)
    points[:, 0] = np.int_(np.floor(points[:, 0] / resolution))
    points[:, 1] = np.int_(np.floor(points[:, 1] / resolution) + width / 2)

    # Get indices of unique values
    _, indices = np.unique(points[:, 0:2], axis=0, return_index=True)

    # Intensity Map & DensityMap
    r_map = np.zeros((height, width))
    g_map = np.zeros((height, width))
    b_map = np.zeros((height, width))

    _, indices, counts = np.unique(
        points[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    cloud_top = points[indices]

    r_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 3]
    g_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 4]
    b_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 5]

    # Fill channels
    bev = np.zeros((height - 1, width - 1, 3))
    bev[:, :, 0] = r_map[:height_, :width_]
    bev[:, :, 1] = g_map[:height_, :width_]
    bev[:, :, 2] = b_map[:height_, :width_]
    bev = np.flip(bev, axis=0)
    bev = np.flip(bev, axis=1)

    return bev


def make_bev_height_intensity(cloud_: o3d.t.geometry.PointCloud, resolution, height_, width_, roi):
    height = height_ + 1
    width = width_ + 1

    # Discretize Feature Map
    cloud: o3d.t.geometry.PointCloud = cloud_.clone()

    points = cloud.point.positions.numpy()
    points = np.c_[points, cloud.point.intensities.numpy()]
    points = filter_roi_cloud(points, roi)
    points[:, 0] = np.int_(np.floor(points[:, 0] / resolution))
    points[:, 1] = np.int_(np.floor(points[:, 1] / resolution) + width / 2)

    # Get indices of unique values
    _, indices = np.unique(points[:, 0:2], axis=0, return_index=True)

    # Intensity Map & DensityMap
    intensity_map = np.zeros((height, width))
    height_map = np.zeros((height, width))

    _, indices, counts = np.unique(
        points[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    cloud_top = points[indices]

    intensity_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 3]
    height_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 2]

    # Fill channels
    bev = np.zeros((height - 1, width - 1, 2))
    bev[:, :, 0] = intensity_map[:height_, :width_]  # Intensity channel
    bev[:, :, 1] = height_map[:height_, :width_]  # Height channel
    bev = np.flip(bev, axis=0)
    bev = np.flip(bev, axis=1)

    return bev


def load_cloud(path: str):
    assert (os.path.exists(path))

    cloud = []
    # Check file extension
    if path.endswith(".bin"):
        print(f"Load .bin file {path}")
        cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif path.endswith(".pcd"):
        print(f"Load .pcd file {path}")
        cloud = np.genfromtxt(
            path, dtype=np.float32, skip_header=11, usecols=(0, 1, 2, 3), comments="%"
        ).reshape(-1, 4)

    print(f"Loaded point cloud with {np.size(cloud)} points")

    return cloud
