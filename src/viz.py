import open3d as o3d
import numpy as np
import scipy.spatial as sps
from open3d.visualization import draw_geometries
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
s27d = "s32_downsample_0.04.pcd"


def highlight_inliers(pcd, inlier_idxs, color=[1.0, 0, 0], draw=False):
    inlier_cloud = pcd.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color(color)
    if draw:
        outlier_cloud = pcd.select_by_index(inlier_idxs, invert=True)
        draw([inlier_cloud, outlier_cloud])
    return inlier_cloud


def iter_draw(idxs_list, pcd):
    pcds = []
    for idxs in idxs_list:
        if len(idxs) > 0:
            pcds.append(pcd.select_by_index(idxs))
    print(f"iterdraw: drawing {len(pcds)} pcds")
    pcd.paint_uniform_color([0, 0, 0.2])
    colors = [
        mcolors.to_rgb(plt.cm.Spectral(each)) for each in np.linspace(0, 1, len(pcds))
    ]

    for idx, sub_pcd in enumerate(pcds):
        sub_pcd.paint_uniform_color(colors[idx])

    # o3d.visualization.draw_geometries([stem_cloud]+pcds)
    o3d.visualization.draw_geometries(pcds)
    return pcd


def draw(pcds, raw=True, **kwargs):
    if (not(isinstance(pcds, list))
        and not(isinstance(pcds, np.ndarray))):
        pcds = [pcds]
    if raw:
        draw_geometries(pcds)
    else:
        draw_geometries(
            pcds,
            mesh_show_wireframe=True,
            zoom=0.7,
            front=[0, 2, 0],
            lookat=[3, -3, 4],
            up=[0, -1, 1],
            **kwargs,
        )
