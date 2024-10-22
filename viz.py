import open3d as o3d
import numpy as np
import scipy.spatial as sps 
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

s27d = "s32_downsample_0.04.pcd"


def iter_draw(idxs_list, pcd):
    pcds = []
    for idxs in idxs_list: 
        if len(idxs)>0:
            pcds.append(pcd.select_by_index(idxs))

    pcd.paint_uniform_color([0,0,0.2])
    colors = [mcolors.to_rgb(plt.cm.Spectral(each)) for each in np.linspace(0, 1, len(pcds))]

    for idx, sub_pcd  in enumerate(pcds): 
        sub_pcd.paint_uniform_color(colors[idx])

    # o3d.visualization.draw_geometries([stem_cloud]+pcds)
    o3d.visualization.draw_geometries(pcds)
    return pcds

    