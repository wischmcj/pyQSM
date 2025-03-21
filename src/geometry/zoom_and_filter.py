
from numpy import array as arr
import numpy as np 
import open3d as o3d
from set_config import config, log

def zoom_pcd(zoom_region,
            pcd, 
            reverse=False):
    pts = arr(pcd.points)
    colors = arr(pcd.colors)
    in_pts,in_colors = zoom(zoom_region, pts, colors,reverse)
    in_pcd = o3d.geometry.PointCloud()
    in_pcd.points = o3d.utility.Vector3dVector(in_pts)
    in_pcd.colors = o3d.utility.Vector3dVector(in_colors)
    return in_pcd

def zoom(zoom_region, #=[(x_min,y_min), (x_max,y_max)],
        pts,
        colors = None,
        reverse=False):
    "Returns points in pts that fall within the given region"
    low_bnd = arr(zoom_region)[0,:]
    up_bnd =arr(zoom_region)[1,:]
    if isinstance(pts, list): pts = arr(pts)
    # breakpoint()
    if len(up_bnd)==2:
        up_bnd = np.append(up_bnd, max(pts[:,2]))
        low_bnd = np.append(low_bnd, min(pts[:,2]))

    inidx = np.all(np.logical_and(low_bnd <= pts, pts <= up_bnd), axis=1)   
    # print(f'{max(pts)},{min(pts)}')
    in_pts = pts[inidx]
    in_colors= None
    if colors is not None : in_colors = colors[inidx]   
    
    if reverse:
        out_pts = pts[np.logical_not(inidx)]
        in_pts = out_pts
        if colors is not None: in_colors = colors[np.logical_not(inidx)]

    return in_pts, in_colors

def filter_list_to_region(ids_and_pts,
                        zoom_region):
    in_ids = []
    for id_w_pts in ids_and_pts:
        idc, pts = id_w_pts
        new_pts, _ = zoom(zoom_region,pts,reverse = False )
        if len(new_pts) == len(pts): 
            in_ids.append(idc)
    return  in_ids

def filter_to_region_pcds(clusters,
                        zoom_region):
    """
        for each cluster in the list, remove points falling outside of the given region
    """
    pts = [tuple((idc,np.asarray(cluster.points))) for idc, cluster in clusters]
    new_idcs = filter_list_to_region(pts,zoom_region)
    new_clusters = [(idc, cluster) for idc, cluster in clusters if idc in new_idcs]
    return  new_clusters


def filter_pcd_list(pcds,
                    max_pctile=85,
                    min_pctile = 30):
    pts = [arr(pcd.points) for pcd in pcds]
    colors = [arr(pcd.colors) for pcd in pcds]

    cluster_sizes = np.array([len(x) for x in pts])
    large_cutoff = np.percentile(cluster_sizes,max_pctile)
    small_cutoff = np.percentile(cluster_sizes,min_pctile)

    log.info(f'isolating clusters between {small_cutoff} and {large_cutoff} points')
    to_keep_cluster_ids  = np.where(
                            np.logical_and(cluster_sizes< large_cutoff,
                                                cluster_sizes> small_cutoff)
                            )[0]
    return  pcds[to_keep_cluster_ids]
    # for idc in small_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])