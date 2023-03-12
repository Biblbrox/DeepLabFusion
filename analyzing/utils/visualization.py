import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import open3d as o3d


def draw_cloud3d(cloud):
    intensity = cloud.point.intensities.numpy()
    max_tot = np.max(intensity)
    source_attribute = intensity / max_tot
    colors_map = cm.get_cmap('jet', 256)
    source_colors = colors_map(source_attribute)
    cloud.point.colors = o3d.core.Tensor(np.asarray(source_colors[:, :3]))

    o3d.visualization.draw([{'name': 'cloud', 'geometry': cloud}], bg_color=(0.0, 0.0, 0.0, 1.0), raw_mode=True,
                           show_skybox=False)


def show_front_proj(cloud_proj):
    fig, ax = plt.subplots(1, 1, facecolor="white", figsize=(14, 16))
    ax.imshow(cloud_proj, cmap="gray")
    plt.tight_layout()


def show_bev_plot(cloud_proj):
    pass
