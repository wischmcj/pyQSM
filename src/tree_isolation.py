# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np
import copy
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from open3d.visualization import draw_geometries
from open3d.io import read_point_cloud, write_point_cloud
from open3d.geometry import TriangleMesh

from itertools import chain
from collections import defaultdict

from fit import cluster_DBSCAN, fit_shape_RANSAC, kmeans
from fit import choose_and_cluster, cluster_DBSCAN, fit_shape_RANSAC, kmeans
from lib_integration import find_neighbors_in_ball

from config import config
from mesh_processing import define_conn_comps, get_surface_clusters, map_density
from point_cloud_processing import ( filter_by_norm,
    clean_cloud,
    crop,
    orientation_from_norms,
    filter_by_norm,
    get_ball_mesh
)
from viz import iter_draw, draw
from utils import (
    get_center,
    get_radius,
    rotation_matrix_from_arr,
    unit_vector,
    get_percentile,
)
from octree import color_node_pts, draw_leaves, cloud_to_octree, nodes_from_point_idxs, nodes_to_pcd

skeletor = "/code/code/Research/lidar/converted_pcs/skeletor.pts"
s27 = "/code/code/Research/lidar/converted_pcs/Secrest27_05.pts"
s32 = "/code/code/Research/lidar/converted_pcs/Secrest32_06.pts"

s27d = "data/input/s27_downsample_0.04.pcd"
s32d = "data/input/s32_downsample_0.04.pcd"


def get_stem_pcd(pcd=None, source_file=None
                ,normals_radius   = config['stem']["normals_radius"]
                ,normals_nn       = config['stem']["normals_nn"]    
                ,nb_neighbors   = config['stem']["stem_neighbors"]
                ,std_ratio      = config['stem']["stem_ratio"]
                ,voxel_size     = config['stem']["stem_voxel_size"]
                ,post_id_stat_down = config['stem']["post_id_stat_down"]
                ,):
    """
        filters the point cloud to only those 
        points with normals implying an approximately
        vertical orientation 
    """
    if source_file:
    # print("IDing stem_cloud")
        pcd = read_point_cloud(source_file)
    print("cleaned cloud")
    # cropping out ground points
    pcd_pts = np.asarray(pcd.points)
    pcd_cropped_idxs = crop(pcd_pts, minz=np.min(pcd_pts[:, 2]) + 0.5)
    pcd = pcd.select_by_index(pcd_cropped_idxs)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn)
    )

    stem_cloud = filter_by_norm(pcd, config['stem']["angle_cutoff"])
    if voxel_size:
        stem_cloud = stem_cloud.voxel_down_sample(voxel_size=voxel_size)
    if post_id_stat_down:
        _, ind = stem_cloud.remove_statistical_outlier(nb_neighbors= nb_neighbors,
                                                       std_ratio=std_ratio)
        stem_cloud = stem_cloud.select_by_index(ind)
    return stem_cloud

def sphere_step(
    curr_pts,
    last_radius,
    main_pcd,
    cluster_idxs,
    branch_order,
    branch_num,
    total_found,
    run=0,
    branches=[[]],
    id_to_num=defaultdict(int),
    cyls=[],
    cyl_details=[],
    spheres=[],
    draw_every=10,
    debug=False,
):
    """
    1. Takes in points from a neighbor cluster found in previous run
         set found
    2. Approximates a center and radius for a cylinder bounding those points
        - stored this to a list representing the QSM
    3. Approimates a circle to surround the neighbors of the cluster points
        -

    Returns:
        _description_
    """
    if branches == [[]]:
        branches[0].append(total_found)

    main_pts = np.asarray(main_pcd.points)

    cyl_mesh = None
    fit_radius = 0
    # try to fit a cylinder to previous neighbors via a 2D circle
    #    only keep cyl_mesh if fit radius is reasonable
    prev_neighbor_height = np.min(curr_pts[:, 2])
    cyl_mesh, fit_pcd, inliers, fit_radius, axis = fit_shape_RANSAC(
        pts=curr_pts,
        shape="circle",
        threshold=0.04,
        lower_bound=prev_neighbor_height,
        max_radius=last_radius * config['sphere']["radius_multiplier"],
    )

    good_fit_found = (
        cyl_mesh is not None
        and fit_radius < config['sphere']["bad_fit_radius_factor"] * last_radius
    )

    if good_fit_found:
        cyls.append(cyl_mesh.sample_points_uniformly(500))
        center = get_center(curr_pts)
        cyl_details.append(
            {
                "center": center,
                "axis": axis,
                "height": prev_neighbor_height,
                "radius": fit_radius,
            }
        )

    # get all points within radius of the former neighbor
    #    group's center point.
    # Exclude any that have already been found
    sphere, new_neighbors = find_neighbors_in_ball(curr_pts, main_pts, total_found)
    spheres.append(sphere)

    clusters = ()
    if len(new_neighbors) > 0:
        cluster_type = "kmeans" if not good_fit_found else "DBSCAN"
        # Cluster new neighbors to finpd potential branches
        labels, clusters = choose_and_cluster(np.asarray(new_neighbors), main_pts, cluster_type)

    if clusters == [] or len(new_neighbors) < config['sphere']["min_contained_points"]:
        return []
    else:
        if (len(clusters[0]) > 2
            or debug):
            try:
                test = iter_draw(clusters[1], main_pcd)
            except Exception as e:
                print(f"error in iterdraw {e}")

    for cluster_idxs in clusters:
        # done here so all idxs are added to total_found
        #  prior to doing any more fitting, finding
        total_found.extend(cluster_idxs)

    returned_idxs = []
    returned_pts = []
    for label, cluster_idxs in zip(labels, clusters):
        cluster_branch = branch_order
        if label != 0:
            cluster_branch += 1
            branches.append([])

        branch_id = branch_num + cluster_branch
        print(f"{branch_id=}, {cluster_branch=}")

        cluster_dict = {cluster_idx: branch_id for cluster_idx in cluster_idxs}
        id_to_num.update(cluster_dict)
        branches[cluster_branch].extend(cluster_idxs)

        # curr_plus_cluster = np.concatenate([curr_pts_idxs, cluster_idxs])
        cluster_pcd_pts = np.asarray(main_pcd.points)[cluster_idxs]
        # center = get_center(sub_pcd_pts)
        cluster_radius = get_radius(cluster_pcd_pts)
        print(f"{cluster_radius=}")
        if cluster_radius < config['sphere']["min_sphere_radius"]:
            cluster_radius = config['sphere']["min_sphere_radius"]
        if cluster_radius > config['sphere']["max_radius"]:
            cluster_radius = config['sphere']["max_radius"]
        if cluster_radius < last_radius / 2:
            cluster_radius = last_radius / 2
 
        if (len(cyls) % draw_every == 0 
            or debug):
            try:
                test = main_pcd.select_by_index(total_found)
                draw([test])
                for cyl_pts in cyls:
                    cyl_pts.paint_uniform_color([0, 1, 0])
                draw([test] + cyls)
                sphere_pts = [sph.sample_points_uniformly(500) for sph in spheres]
                for sphere_pt in sphere_pts:
                    sphere_pt.paint_uniform_color([0, 0, 1])
                draw([test] + sphere_pts)

                not_found = main_pcd.select_by_index(returned_idxs)
                test.paint_uniform_color([0, 1, 0])
                not_found.paint_uniform_color([1, 0, 0])
                draw([test, not_found])
            except Exception as e:
                print(f"error in checkpoint_draw {e}")
            breakpoint()
        fit_remaining_branch = sphere_step(
            cluster_pcd_pts,
            cluster_radius,
            main_pcd,
            cluster_idxs.
            cluster_branch,
            branch_num,
            total_found,
            run + 1,
            branches,
            id_to_num,
            cyls,
            cyl_details,
            spheres,
        )
        if fit_remaining_branch == []:
            print(f"returned ids {cluster_idxs}")
            returned_idxs.extend(cluster_idxs)
            returned_pts.extend(cluster_idxs)
        branch_num += 1
    print("reached end of function, returning")
    return branches, id_to_num, cyls, cyl_details

def find_low_order_branches():
    # ***********************
    # IDEAS FOR CLEANING RESULTS
    # - Mutliple iterations of statistical outlier removal
    #    start with latge std ratio, and low num neighbors(4?)
    # - Larger Voxel size for Stem Filtering - more representitive normals
    #      - also look into k_nearest_neighbor smoothing
    # - Fit plane to ground, remove ground before finiding lowest
    #       - ended up removing the bottom .5 meters
    # - Using density calculations to inform radius for fit
    #       - more dense areas -> smaller radius
    # ***********************

    # Reading in cloud and smooth
    pcd = read_point_cloud('27_pt02.pcd',print_progress=True)
    write_point_cloud("27_pt02.pcd",pcd)
    print('read in cloud')
    stat_down=pcd
    stat_down = clean_cloud(pcd,
                            voxels=config['initial_clean']['voxel_size'],
                            neighbors=config['initial_clean']['neighbors'],
                            ratio=config['initial_clean']['ratio'],
                            iters = config['initial_clean']['iters'])
    # "data/results/saves/27_vox_pt02_sta_6-4-3.pcd" # < ---  post-clean pre-stem
    stem_cloud = get_stem_pcd(stat_down)


    algo_pcd_pts = np.asarray(algo_source_pcd.points)
    not_so_low_idxs, _ = get_percentile(algo_pcd_pts, 0, 1)
    low_cloud = algo_source_pcd.select_by_index(not_so_low_idxs)
    low_cloud_pts = np.asarray(low_cloud.points)

    algo_source_pcd = stem_cloud
    print("Identifying trunk ...")
    print("Identifying based layer for search ...")
    sphere, neighbors = find_neighbors_in_ball(low_cloud_pts, algo_pcd_pts, not_so_low_idxs)
    new_neighbors = np.setdiff1d(neighbors, not_so_low_idxs)
    total_found = np.concatenate([not_so_low_idxs, new_neighbors])

    print("Fitting cyl to trunk ...")
    nn_pcd = algo_source_pcd.select_by_index(new_neighbors)
    nn_pts = algo_pcd_pts[new_neighbors]
    mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape_RANSAC(pcd=nn_pcd, shape="circle")

    print("Running sphere algo")
    res, id_to_num, cyls, cyl_details = sphere_step(
        nn_pts,
        fit_radius,
        algo_source_pcd,
        inliers,
        branch_order=0,
        branch_num=0,
        total_found=list(total_found),
    )
    breakpoint()

    # Coloring and drawing the results of 
    #  sphere step
    iter_draw(res[0], algo_source_pcd)

    branch_index = defaultdict(list)
    for idx, branch_num in id_to_num.items():
        branch_index[branch_num].append(idx)
    try:
        iter_draw(list(branch_index.values()), algo_pcd_pts)
    except Exception as e:
        print("error" + e)


if __name__ == "__main__":
    find_low_order_branches()

    # pcd =  read_point_cloud("27_vox_pt02_sta_6-4-3.pcd")
    # pcd =  read_point_cloud("stem_cloud.pcd")
    # labels = np.array( pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # draw([pcd])
