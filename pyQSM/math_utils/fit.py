# import open3d as o3d
import copy
# import scipy.spatial as sps
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.cluster as spc
from sklearn.metrics import silhouette_score
# from sklearn.metrics import calinski_harabasz_score
# from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
from matplotlib import pyplot as plt

from geometry.point_cloud_processing import get_shape
from set_config import config, log
from math_utils.general import (get_radius, 
                    get_center, 
                    rotation_matrix_from_arr,
                    unit_vector)

def z_align_and_fit(pcd, axis_guess, **kwargs):
    """
        Attempts to fit a cylinder to a point cloud.
        Reduces dimensionality (3d->2d) by estimating the 
            axis of said cylinder, rotating the point cloud and
            fitting its 2D projection with a circle.
    """
    R_to_z = rotation_matrix_from_arr(unit_vector(axis_guess), [0, 0, 1])
    R_from_z = rotation_matrix_from_arr([0, 0, 1], unit_vector(axis_guess))
    # align with z-axis
    pcd_r = copy.deepcopy(pcd)
    pcd_r.rotate(R_to_z)
    # approx via circle
    mesh, _, inliers, fit_radius, _ = fit_shape_RANSAC(pcd=pcd_r, **kwargs)
    # Rotate mesh and pcd back to original orientation'
    # pcd.rotate(R_from_z)
    if mesh is None:
        log.warning('No mesh found')
        return mesh, _, inliers, fit_radius, _
    mesh_pts = mesh.sample_points_uniformly(1000)
    mesh_pts.paint_uniform_color([0, 1.0, 0])
    mesh_pts.rotate(R_from_z)
    return mesh, _, inliers, fit_radius, _

def choose_and_cluster(new_neighbors, main_pts, cluster_type, debug=False):
    """
    Determines the appropriate clustering algorithm to use
    and returns the result of said algorithm
    """
    returned_clusters = []
    nn_points = main_pts[new_neighbors]
    if cluster_type == "kmeans":
        # in these cases we expect the previous branch
        #     has split into several new branches. Kmeans is
        #     better at characterizing this structure
        log.info("clustering via kmeans")
        labels, returned_clusters = kmeans(nn_points, 1)
        if debug:
            labels = [idx for idx,_ in enumerate(returned_clusters)]
            ax = plt.figure().add_subplot(projection='3d')
            for cluster in returned_clusters: 
                ax.scatter(nn_points[cluster][:,0], nn_points[cluster][:,1], nn_points[cluster][:,2], 'r')
            plt.show()
    if cluster_type != "kmeans" or len(returned_clusters) < 2:
        log.info("clustering via DBSCAN")
        labels, returned_clusters, noise = cluster_DBSCAN(
            new_neighbors,
            nn_points,
            eps=config['dbscan']["epsilon"],
            min_pts=config['dbscan']["min_neighbors"],
        )
    return labels, returned_clusters

def evaluate_orientation(pcd):
    """
    Determine the apporoximate orientation
       of a bounding cylinder for a point cloud
    """
    # Get normals and align to Z axis
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20)
    )
    pcd.orient_normals_consistent_tangent_plane(100)
    pcd.normalize_normals()
    norms = np.array(pcd.normals)
    axis_guess = orientation_from_norms(norms, samples=100, max_iter=1000)
    return axis_guess

def z_align_and_fit(pcd, axis_guess, **kwargs):
    """
        Attempts to fit a cylinder to a point cloud.
        Reduces dimensionality (3d->2d) by estimating the 
            axis of said cylinder, rotating the point cloud and
            fitting its 2D projection with a circle.
    """
    R_to_z = rotation_matrix_from_arr(unit_vector(axis_guess), [0, 0, 1])
    R_from_z = rotation_matrix_from_arr([0, 0, 1], unit_vector(axis_guess))
    # align with z-axis
    pcd_r = copy.deepcopy(pcd)
    pcd_r.rotate(R_to_z)
    # approx via circle
    mesh, _, inliers, fit_radius, _ = fit_shape_RANSAC(pcd=pcd_r, **kwargs)
    # Rotate mesh and pcd back to original orientation'
    # pcd.rotate(R_from_z)
    if mesh is None:
        # draw([pcd])
        return mesh, _, inliers, fit_radius, _
    mesh_pts = mesh.sample_points_uniformly(1000)
    mesh_pts.paint_uniform_color([0, 1.0, 0])
    mesh_pts.rotate(R_from_z)
    draw([mesh_pts, pcd])
    return mesh, _, inliers, fit_radius, _

## I beleive that the below function needs some work
##  and may eventually be removed entirely
# def choose_and_cluster(new_neighbors, main_pts, cluster_type):
#     """
#     Determines the appropriate clustering algorithm to use
#     and returns the result of said algorithm
#     """
#     returned_clusters = []
#     try:
#         nn_points = main_pts[new_neighbors]
#     except Exception as e:
#         breakpoint()
#         print(f"error in choose_and_cluster {e}")
#     if cluster_type == "kmeans":
#         # in these cases we expect the previous branch
#         #     has split into several new branches. Kmeans is
#         #     better at characterizing this structure
#         print("clustering via kmeans")
#         labels, returned_clusters = kmeans(nn_points, 1)
#         # labels = [idx for idx,_ in enumerate(returned_clusters)]
#         # ax = plt.figure().add_subplot(projection='3d')
#         # for cluster in returned_clusters: ax.scatter(nn_points[cluster][:,0], nn_points[cluster][:,1], nn_points[cluster][:,2], 'r')
#         # plt.show()
#     if cluster_type != "kmeans" or len(returned_clusters) < 2:
#         print("clustering via DBSCAN")
#         labels, returned_clusters, noise = cluster_DBSCAN(
#             new_neighbors,
#             nn_points,
#             eps=config['dbscan']["epsilon"],
#             min_pts=config['dbscan']["min_neighbors"],
#         )
#     return labels, returned_clusters

def kmeans(points, min_clusters):
    """
    https://www.comet.com/site/blog/how-to-evaluate-clustering-models-in-python/
    """
    # ch_max = 50
    clusters_to_try = [
        min_clusters,
        min_clusters + 1,
        min_clusters + 2,
        min_clusters + 3,
    ]
    codes, book = None, None
    pts_2d = points[:, :2]
    best_score = 0.4
    best = None
    for num in clusters_to_try:
        log.info(f"trying {num} clusters")
        if num > 0:
            codes, book = spc.vq.kmeans2(pts_2d, num)
            cluster_sizes = np.bincount(book)
            if num == 1:
                best = book
            else:
                try:
                    sh_score = silhouette_score(points, book)
                except ValueError as err:
                    log.info(f"Error in silhouette_score {err}")
                    sh_score = 0
                # ch_score = calinski_harabasz_score(points,book)
                # db_score = davies_bouldin_score(points,book)
                # log.info(f'''num clusters: {num}, sizes: {cluster_sizes},
                #              sh_score: {sh_score}, ch_score: {ch_score}, db_score: {db_score}''')
                # results.append(((codes,book,num),[sh_score,ch_score,db_score]))
                if sh_score > best_score:
                    best = book
    cluster_idxs = []
    labels = []
    try:
        plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=best)
        plt.show()
    except Exception as err:
        log.info(f"Error in plotting {err}")
    for i in range(max(best)):
        ids_group_i = [idx for idx, val in enumerate(best) if val == i]
        cluster_idxs.append(ids_group_i)
        labels.append(i)
    return labels, cluster_idxs


def cluster_DBSCAN(pts_idxs, points, eps, min_pts):
    """
    Attepts to cluster by finding a minimal set of points
       s.t. all points in the set are within a distance,
       epsilon (eps), of at least on point
    """
    clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(points)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    idxs = []
    noise = []
    # converting ids for 'points' to ids
    #   in main_pcd stored in pts_idxs
    for k in unique_labels:
        class_member_mask = labels == k
        if k == -1:
            idx_bool = class_member_mask & (core_samples_mask == False)
            c_pt_idxs = np.where((np.array(idx_bool) == True))
            noise = pts_idxs[c_pt_idxs]
        else:
            idx_bool = class_member_mask & core_samples_mask
            c_pt_idxs = np.where((np.array(idx_bool) == True))
            neighbor_idxs = pts_idxs[c_pt_idxs]
            idxs.append(neighbor_idxs)

    log.info(f"Estimated number of clusters: {num_clusters}")
    log.info("Estimated number of noise points: %d" % num_noise)
    return unique_labels, idxs, noise


def fit_shape_RANSAC(
    pcd=None,
    pts=None,
    threshold=0.1,
    lower_bound=None,
    max_radius=None,
    align_to_z=False,
    shape="circle",
    **kwargs,
):
    if pts is None:
        pts = np.asarray(pcd.points)
    if lower_bound:
        for pt in pts:
            if pt[2] < lower_bound:
                pt[2] = lower_bound

    lowest, heighest = min(pts[:, 2]), max(pts[:, 2])

    radius = get_radius(pts)
    fit_pts = np.asarray(pts.copy())
    if shape == "circle":
        for pt in fit_pts:
            pt[2] = 0
        shape_cls = pyrsc.Circle()
    if shape == "cylinder":
        shape_cls = pyrsc.Cylinder()

    center, axis, fit_radius, inliers = shape_cls.fit(
        pts=np.asarray(fit_pts), thresh=threshold
    )
    log.info(f"fit_cyl = center: {center}, axis: {axis}, radius: {fit_radius}")

    if max_radius is not None:
        if fit_radius> max_radius:
            log.info(f'{shape} had radius {fit_radius} but max_radius is {max_radius}')
            return None, None, None, None, None

    if len(center) == 0:
        log.info(f"no no fit {shape} found")
        return None, None, None, None, None

    in_pts = pts[inliers]
    lowest, heighest = min(in_pts[:, 2]), max(in_pts[:, 2])
    height = heighest - lowest
    center_height = (height / 2) + lowest
    test_center = [center[0], center[1], center_height]

    # test_center = center
    if height <= 0:
        breakpoint()
        return None, None, None, None, None

    if align_to_z:
        if ((axis[0] == 0 and axis[1] == 0 and axis[2] == 1) or
            (axis[0] == 0 and axis[1] == 0 and axis[2] == -1)):
            log.info(f'No Rotation Needed')
            cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,
                                center=tuple(test_center),
                                radius=fit_radius*1.2,
                                height=height)
        else:
            log.info(f'Rotation Needed')
            cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,
                                center=tuple(test_center),
                                radius=fit_radius*1.2,
                                height=height,
                                axis=axis)
        
    shape_radius = fit_radius * 1.05
    cyl_mesh = get_shape(
        pts,
        shape="cylinder",
        as_pts=False,
        center=tuple(test_center),
        radius=shape_radius,
        height=height,
        axis=axis,
        **kwargs,
    )
    in_pcd = None
    if pcd is not None:
        in_pcd = pcd.select_by_index(inliers)
        pcd.paint_uniform_color([1.0, 0, 0])
        in_pcd.paint_uniform_color([0, 1.0, 0])

    return cyl_mesh, in_pcd, inliers, fit_radius, axis
