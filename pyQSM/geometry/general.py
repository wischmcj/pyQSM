
from numpy import array as arr
import numpy as np 
import open3d as o3d
from set_config import config, log


def center_and_rotate(pcd, center=None):
    center = pcd.get_center() if center is None else center
    rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    pcd.translate(np.array([-x for x in center ]))
    pcd.rotate(rot_90_x)
    pcd.rotate(rot_90_x)
    pcd.rotate(rot_90_x)
    return center

def zoom_pcd(zoom_region,
            pcd, 
            reverse=False):
    """
    Extract points from a point cloud that fall within or outside a specified region.
    
    Args:
        zoom_region (list): Region bounds in format [(x_min, y_min), (x_max, y_max)] 
                           or [(x_min, y_min, z_min), (x_max, y_max, z_max)]
        pcd (o3d.geometry.PointCloud): Input point cloud with points and colors
        reverse (bool, optional): If True, return points outside the region instead 
                                 of inside. Defaults to False.
    
    Returns:
        o3d.geometry.PointCloud: New point cloud containing filtered points and colors
    """
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
    """
    Filter points and colors based on a spatial region.
    
    Returns points in pts that fall within the given region. If the region is 2D,
    automatically extends it to 3D using the min/max Z values from the points.
    
    Args:
        zoom_region (list): Region bounds in format [(x_min, y_min), (x_max, y_max)]
                           or [(x_min, y_min, z_min), (x_max, y_max, z_max)]
        pts (numpy.ndarray or list): Array of 3D points with shape (N, 3)
        colors (numpy.ndarray, optional): Array of colors corresponding to points.
                                         Defaults to None.
        reverse (bool, optional): If True, return points outside the region instead
                                 of inside. Defaults to False.
    
    Returns:
        tuple: (filtered_points, filtered_colors) where:
            - filtered_points (numpy.ndarray): Points within/outside the region
            - filtered_colors (numpy.ndarray or None): Corresponding colors, or None
    """
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
    """
    Filter a list of ID-point pairs to find which IDs have all points within a region.
    
    Args:
        ids_and_pts (list): List of tuples (id, points) where:
                           - id: identifier for the point set
                           - points: numpy array of 3D points
        zoom_region (list): Region bounds in format [(x_min, y_min), (x_max, y_max)]
    
    Returns:
        list: List of IDs whose corresponding point sets are entirely within the region
    """
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
    Filter clusters to keep only those that are entirely within a specified region.
    
    For each cluster in the list, checks if all points fall within the given region
    and returns only the clusters that satisfy this condition.
    
    Args:
        clusters (list): List of tuples (id, cluster) where:
                        - id: cluster identifier
                        - cluster: point cloud object with .points attribute
        zoom_region (list): Region bounds in format [(x_min, y_min), (x_max, y_max)]
    
    Returns:
        list: List of (id, cluster) tuples for clusters entirely within the region
    """
    pts = [tuple((idc,np.asarray(cluster.points))) for idc, cluster in clusters]
    new_idcs = filter_list_to_region(pts,zoom_region)
    new_clusters = [(idc, cluster) for idc, cluster in clusters if idc in new_idcs]
    return  new_clusters


def filter_pcd_list(pcds,
                    max_pctile=85,
                    min_pctile = 30):
    """
    Filter a list of point clouds based on cluster size percentiles.
    
    Removes point clouds that are too large or too small based on the number of
    points they contain, keeping only those within the specified percentile range.
    
    Args:
        pcds (numpy.ndarray): Array of point cloud objects, each with .points and .colors
        max_pctile (int, optional): Maximum percentile cutoff for cluster size.
                                   Clusters larger than this percentile are removed.
                                   Defaults to 85.
        min_pctile (int, optional): Minimum percentile cutoff for cluster size.
                                   Clusters smaller than this percentile are removed.
                                   Defaults to 30.
    
    Returns:
        numpy.ndarray: Filtered array of point clouds within the size range
    """
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