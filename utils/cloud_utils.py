import os.path

import cv2
import numpy as np
import open3d as o3d
from matplotlib import cm

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


def np_to_o3d_cloud(np_cloud: np.array) -> o3d.t.geometry.PointCloud:
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32

    intensity = np.c_[np_cloud[:, 3], np_cloud[:, 3], np_cloud[:, 3]]
    points = np_cloud[:, :3]
    cloud = o3d.t.geometry.PointCloud(device)
    cloud.point.positions = o3d.core.Tensor(np.asarray(points), dtype, device)
    cloud.point.intensities = o3d.core.Tensor(np.asarray(intensity), dtype, device)

    cloud.point.colors = cloud.point.intensities

    return cloud


def o3d_cloud_to_np(cloud: o3d.t.geometry.PointCloud):
    pass


def o3d_rgbd_to_np(cloud: o3d.t.geometry.PointCloud):
    pass


def make_front_proj(cloud: np.array, width, height, intrinsic, extrinsic) -> np.array:
    projection: o3d.t.geometry.RGBDImage = np_to_o3d_cloud(cloud).project_to_rgbd_image(width, height, intrinsic,
                                                                                        extrinsic,
                                                                                        depth_max=10000, depth_scale=1)
    intensity = projection.color.as_tensor().numpy()
    depth = np.log(np.maximum(0.0000001, projection.depth.as_tensor().numpy()))
    front = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.float32)
    front[:, :, 0] = intensity[:, :, 0]
    front[:, :, 1] = depth[:, :, 0]

    return front, projection.color.as_tensor().numpy()


def _make_front_proj(cloud, width, height, intrinsic, extrinsic):
    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]
    intensity = cloud[:, 3]
    ranges = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    fov_down = np.deg2rad(17.8)
    fov_up = np.deg2rad(-2)

    ## For every point in cloud
    # Get Euler angles
    pitch_values = np.arcsin(z / ranges)
    yaw_values = np.arctan2(y, x)
    # Normalizing and scaling
    fov = fov_up + np.abs(fov_down)
    normalized_pitch = height * (1 - (pitch_values + np.abs(fov_down)) / fov)
    normalized_yaw = width * 0.5 * (yaw_values / np.pi + 1)

    # Round and clamp for use as index
    u_values = np.floor(normalized_pitch)
    v_values = np.floor(normalized_yaw)

    v_values = np.minimum(width - 1, v_values)
    v_values = np.maximum(0.0, v_values)

    u_values = np.minimum(height - 1, u_values)
    u_values = np.maximum(0.0, u_values)

    # Get image coordinates
    u_values = np.floor(u_values)
    v_values = np.floor(v_values)
    u_values = u_values.astype(np.int)
    v_values = v_values.astype(np.int)

    ## Build image from u, v
    image = np.zeros([height, width, 1], dtype=np.uint8)
    old_max = np.max(x)
    old_min = np.min(x)
    for i in range(np.size(u_values)):
        image[u_values[i], v_values[i]] = image_utils.normalize(old_min, old_max, 0, 255, x[i])  # * 255
        image[u_values[i], v_values[i]] = intensity[i] * 255


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
    projection = np.zeros(image.shape)
    u, v, depth = cam2image(cloud, [])

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


def make_bev_color(cloud_: np.array, resolution, height_, width_, roi):
    height = height_ + 1
    width = width_ + 1

    # Discretize Feature Map
    cloud = cloud_.copy()

    cloud = filter_roi_cloud(cloud, roi)
    cloud[:, 0] = np.int_(np.floor(cloud[:, 0] / resolution))
    cloud[:, 1] = np.int_(np.floor(cloud[:, 1] / resolution) + width / 2)

    # Get indices of unique values
    _, indices = np.unique(cloud[:, 0:2], axis=0, return_index=True)

    # Intensity Map & DensityMap
    r_map = np.zeros((height, width), dtype=np.float32)
    g_map = np.zeros((height, width), dtype=np.float32)
    b_map = np.zeros((height, width), dtype=np.float32)

    _, indices, counts = np.unique(
        cloud[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    cloud_top = cloud[indices]

    r_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 3]
    g_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 4]
    b_map[np.int_(cloud_top[:, 0]), np.int_(cloud_top[:, 1])] = cloud_top[:, 5]

    # Fill channels
    bev = np.zeros((height - 1, width - 1, 3), dtype=np.float32)
    bev[:, :, 0] = r_map[:height_, :width_]
    bev[:, :, 1] = g_map[:height_, :width_]
    bev[:, :, 2] = b_map[:height_, :width_]
    bev = np.flip(bev, axis=0)
    bev = np.flip(bev, axis=1)

    return bev


def make_bev_height_intensity(cloud_: np.array, resolution, height_, width_, roi):
    height = height_ + 1
    width = width_ + 1

    # Discretize Feature Map
    cloud = cloud_.copy()

    cloud = filter_roi_cloud(cloud, roi)
    cloud[:, 0] = np.int_(np.floor(cloud[:, 0] / resolution))
    cloud[:, 1] = np.int_(np.floor(cloud[:, 1] / resolution) + width / 2)

    # Get indices of unique values
    _, indices = np.unique(cloud[:, 0:2], axis=0, return_index=True)

    # Intensity Map & DensityMap
    intensity_map = np.zeros((height, width))
    height_map = np.zeros((height, width))

    _, indices, counts = np.unique(
        cloud[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    cloud_top = cloud[indices]

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
