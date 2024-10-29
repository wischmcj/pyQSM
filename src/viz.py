from copy import deepcopy
import open3d as o3d
import numpy as np
import scipy.spatial as sps
from open3d.visualization import draw_geometries
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

s27d = "s32_downsample_0.04.pcd"


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


def draw(pcds, raw=False, side_by_side=False, **kwargs):
    if (not(isinstance(pcds, list))
        and not(isinstance(pcds, np.ndarray))):
        pcds = [pcds]
    if raw:
        draw_geometries(pcds)
    if side_by_side:
        trans = 0
        pcds_to_draw = []
        for pcd in pcds:
            to_draw = deepcopy(pcd)
            to_draw.translate([trans,0,0])
            pcds_to_draw.append(to_draw)
            min_bound = to_draw.get_axis_aligned_bounding_box().get_min_bound()
            max_bound = to_draw.get_axis_aligned_bounding_box().get_max_bound()
            bounds = max_bound - min_bound
            trans+=bounds[0]
    else:
        pcds_to_draw = pcds
    #below config used for main dev
    # tree, Secrest27
    draw_geometries(
            pcds_to_draw,
            # mesh_show_wireframe=True,
            zoom=0.7,
            front=[0, 2, 0],
            lookat=[3, -3, 4],
            up=[0, -1, 1],
            **kwargs,
        )
