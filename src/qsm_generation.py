# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

from collections import defaultdict
import logging
from itertools import chain

import open3d as o3d
import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt

from matplotlib import patches

from open3d.io import read_point_cloud, write_point_cloud

from geometry.skeletonize import extract_skeleton, extract_topology
from utils.fit import cluster_DBSCAN, fit_shape_RANSAC, kmeans
from utils.fit import choose_and_cluster, cluster_DBSCAN, fit_shape_RANSAC, kmeans
from utils.lib_integration import find_neighbors_in_ball
from utils.math_utils import (
    get_angles,
    get_center,
    get_radius,
    rotation_matrix_from_arr,
    unit_vector,
    get_percentile,
)

from set_config import config
from geometry.mesh_processing import define_conn_comps, get_surface_clusters, map_density
from geometry.point_cloud_processing import ( filter_by_norm,
    clean_cloud,
    crop, get_shape,
    orientation_from_norms,
    filter_by_norm,
    get_ball_mesh
)
from viz.viz_utils import iter_draw, draw


log = logging.getLogger('calc')

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
                ,angle_cutoff      = config['stem']["angle_cutoff"]
                ,voxel_size     = config['stem']["stem_voxel_size"]
                ,post_id_stat_down = config['stem']["post_id_stat_down"]
                ,):
    """
        filters the point cloud to only those 
        points with normals implying an approximately
        vertical orientation 
    """
    if source_file:
    # log.info("IDing stem_cloud")
        pcd = read_point_cloud(source_file)
    log.info("cleaned cloud")
    # cropping out ground points
    pcd_pts = arr(pcd.points)
    pcd_cropped_idxs = crop(pcd_pts, minz=np.min(pcd_pts[:, 2]) + 0.5)
    pcd = pcd.select_by_index(pcd_cropped_idxs)
    pcd.estimate_normals(    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))
map_density
    stem_cloud = filter_by_norm(pcd, angle_cutoff)
    if voxel_size:
        stem_cloud = stem_cloud.voxel_down_sample(voxel_size=voxel_size)
    if post_id_stat_down:
        _, ind = stem_cloud.remove_statistical_outlier(nb_neighbors= nb_neighbors,
                                                       std_ratio=std_ratio)
        stem_cloud = stem_cloud.select_by_index(ind)
    return stem_cloud

def compare_normals_approaches(stat_down):
    from viz.viz_utils import color_continuous_map
    stat_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=20))
    norms = arr(stat_down.normals)

    def angle_func(*args): get_angles(*args, reference='XZ')
    angles = np.apply_along_axis(angle_func, .1, norms)
    angles = np.degrees(angles)
    print("Mean:", np.mean(angles))
    print("Median:", np.median(angles))
    print("Standard Deviation:", np.std(angles))
    color_continuous_map(stat_down,angles)
    draw(stat_down)
    stem_cloud =  filter_by_norm(stat_down,10)
    draw(stem_cloud)

def get_low_cloud(pcd, 
                  start = config['trunk']['lower_pctile'],
                  end = config['trunk']['upper_pctile']):
    algo_source_pcd = pcd  
    algo_pcd_pts = arr(algo_source_pcd.points)
    log.info(f"Getting points between the {start} and {end} percentiles")
    not_too_low_idxs, _ = get_percentile(algo_pcd_pts,start,end)
    low_cloud = algo_source_pcd.select_by_index(not_too_low_idxs)
    return low_cloud, not_too_low_idxs

def sphere_step(
    curr_pts,
    last_radius,
    main_pcd,
    cluster_idxs,
    branch_order=0,
    branch_num=0,
    total_found=[],
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

    main_pts = arr(main_pcd.points)

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
    
    if debug:
        curr_cluster_pcd = main_pcd.select_by_index(cluster_idxs)
        draw([cyl_mesh.sample_points_uniformly(), curr_cluster_pcd])
        breakpoint()

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
    sphere, new_neighbors, center, radius = find_neighbors_in_ball(curr_pts, main_pts, total_found)
    spheres.append(sphere)

    clusters = ()
    if len(new_neighbors) > 0:
        cluster_type = "kmeans" if not good_fit_found else "DBSCAN"
        # Cluster new neighbors to finpd potential branches
        labels, clusters = choose_and_cluster(arr(new_neighbors), main_pts, cluster_type, debug)

    if clusters == [] or len(new_neighbors) < config['sphere']["min_contained_points"]:
        return []
    else:
        if (len(clusters[0]) > 2
            or debug):
            try:
                test = iter_draw(clusters[1], main_pcd)
            except Exception as e:
                log.info(f"error in iterdraw {e}")

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
        log.info(f"{branch_id=}, {cluster_branch=}")

        cluster_dict = {cluster_idx: branch_id for cluster_idx in cluster_idxs}
        id_to_num.update(cluster_dict)
        branches[cluster_branch].extend(cluster_idxs)

        # curr_plus_cluster = np.concatenate([curr_pts_idxs, cluster_idxs])
        cluster_pcd_pts = arr(main_pcd.points)[cluster_idxs]
        # center = get_center(sub_pcd_pts)
        cluster_radius = get_radius(cluster_pcd_pts)
        log.info(f"{cluster_radius=}")
        if cluster_radius < config['sphere']["min_radius"]:
            cluster_radius = config['sphere']["min_radius"]
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
                log.info(f"error in checkpoint_draw {e}")
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
            debug
        )
        if fit_remaining_branch == []:
            log.info(f"returned ids {cluster_idxs}")
            returned_idxs.extend(cluster_idxs)
            returned_pts.extend(cluster_idxs)
        branch_num += 1
    log.info("reached end of function, returning")
    return branches, id_to_num, cyls, cyl_details


def find_low_order_branches(start = 'initial_clean', 
                             file = '27_pt02.pcd',
                             extract_skeleton = False):
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
    started = False
    pcd = None
    stem_cloud = None
    stat_down = None
    if start == 'initial_clean':
        started = True
        # Reading in cloud and smooth
        pcd = read_point_cloud(file, 'xyz', print_progress=True)
        # breakpoint()
        if len(np.array(pcd.points)) < 1:
            log.info("No points found in point cloud")
            pcd = read_point_cloud(file, print_progress=True)

        # write_point_cloud(file,pcd)
        log.info('read in cloud')
        # stat_down=pcd
        stat_down = clean_cloud(pcd,
                                voxels=config['initial_clean']['voxel_size'],
                                neighbors=config['initial_clean']['neighbors'],
                                ratio=config['initial_clean']['ratio'],
                                iters = config['initial_clean']['iters'])
        draw(stat_down)
    
    if start == 'stem_id' or started:
        # breakpoint()
        if not started:
            stat_down = read_point_cloud(file, print_progress=True)
            started = True
        # "data/results/saves/27_vox_pt02_sta_6-4-3.pcd" # < ---  post-clean pre-stem
        stem_cloud = get_stem_pcd(stat_down)
        draw(stem_cloud)
    
    if extract_skeleton:
        contracted, total_point_shift = extract_skeleton(pcd, 
                                                            trunk_points=None,  
                                                            debug =False)
                                                            # moll = moll,
                                                            # max_iter= iter_num,
                                                            # contraction_factor = contract)
        topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph = extract_topology(contracted)
        breakpoint()

    if start == 'trunk_id' or started:
        # breakpoint()
        if not started:
            stem_cloud = read_point_cloud(file, print_progress=True)
            started = True
        log.info("Identifying low percentile by height...")
        low_cloud, idxs = get_low_cloud(stem_cloud)
        low_cloud_pts = arr(low_cloud.points)
        draw(low_cloud)

        log.info(f"Clustering low %ile points to identify trunk")
        labels = np.array(low_cloud.cluster_dbscan(eps=         config['trunk']['cluster_eps'],        
                                                   min_points=  config['trunk']['cluster_nn'], 
                                                   print_progress=True))
        max_label = labels.max()
        log.info(f"point cloud has {max_label + 1} clusters")
        
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        low_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        unique_vals, counts = np.unique(labels, return_counts=True)
        draw(low_cloud)
        
        log.info("Identifying trunk ...")

        unique_vals, counts = np.unique(labels, return_counts=True)
        largest = unique_vals[np.argmax(counts)]
        trunk_idxs = np.where(labels == largest)[0]
        trunk = low_cloud.select_by_index(trunk_idxs)
        trunk_pts = arr(trunk.points)
        # draw(trunk)

        # obb = trunk.get_oriented_bounding_box()
        # obb.color = (1, 0, 0)
        # draw([trunk,obb])

        obb = trunk.get_minimal_oriented_bounding_box()
        obb.color = (1, 0, 0)
        draw([trunk,obb])
        fit_radius = obb.extent[0] / 2

        # breakpoint()
        # log.info("Fitting cylinder to trunk ...")
        # cyl_mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape_RANSAC(pcd=low_cloud, shape="circle")
        # o3d.visualization.draw([cyl_mesh, trunk])

        total_found= trunk_idxs
        # breakpoint()
        algo_source_pcd = stem_cloud
        algo_pcd_pts = arr(algo_source_pcd.points)

        log.info("Identifying first set of points above trunk")
        sphere, neighbors, center, radius = find_neighbors_in_ball(trunk_pts, algo_pcd_pts, trunk_idxs)
        new_neighbors = np.setdiff1d(neighbors, trunk_idxs)
        total_found = np.concatenate([trunk_idxs, new_neighbors])
        nn_pcd = algo_source_pcd.select_by_index(new_neighbors)
        nn_pts = algo_pcd_pts[new_neighbors]
        sphere=get_shape(pts=None, shape='sphere', as_pts=True,   center=tuple(center), radius=radius)
        o3d.visualization.draw_geometries([nn_pcd,trunk,sphere])
        breakpoint()
        # cyl_mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape_RANSAC(pcd=low_cloud, shape="circle")
        # mesh_pts = cyl_mesh.sample_points_uniformly(1000)
        # mesh_pts.paint_uniform_color([0, 1.0, 0])
        # height = np.max(nn_pts[:, 2]) - np.min(nn_pts[:, 2])
        # cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,
        #                 center=tuple(test_center),
        #                 radius=fit_radius,
        #                 height=height,
        #                 axis=axis)
        # o3d.visualization.draw_geometries([nn_pcd,mesh_pts])
    

    if start == 'sphere_step' or started:
        started = True
        log.info("Running sphere algo")
        res, id_to_num, cyls, cyl_details = sphere_step(
            curr_pts = nn_pts, # The points in the branch currently being traversed 
            last_radius = fit_radius, # The radius of the last cyliner fit
            main_pcd = algo_source_pcd, # The parent point cloud being traversed
            cluster_idxs = new_neighbors, # The indexes of the points in the last cluster
            branch_order = 0,
            branch_num = 0,
            total_found = list(total_found),
            debug = True
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
            log.info("error" + e)

def color_and_draw(pcd , labels):
    max_label = labels.max()
    # visualize the labels
    log.info(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcd.colors = colors
    draw([pcd])
    return colors

    # global_x = min(extents[:,0,0])

if __name__ == "__main__":
    import pickle 
    # extents, contains_region, pcds = find_extents()
    # with open(f'part_file_extent_dict.pkl','wb') as f: pickle.dump(extents,f)
    # with open(f'part_file_extent_dict.pkl','rb') as f: 
    #     extents = pickle.load(f)
    # breakpoint()
    import scipy.spatial as sps
    import itertools

    # dividing part files into
    breakpoint()
    # o3d.geometry.get_oriented_bounding_box


    # collective_min = [ 34.05799866, 286.28399658, -21.90399933]
    # collective_max = [189.47099304, 458.29800415,  40.64099884]

    # # extent = collective.get_max_bound() - collective.get_min_bound()
    
    # print('getting low cloud...')
    
    
   
   

    import scipy.spatial as sps
    import itertools
    from copy import deepcopy


    # Playing with octrees
    # octree = build_octree(collective,ot_depth=6)
    # nb_leaves = [octree.locate_leaf_node(pt) for pt in arr(clusters[19].points)]
    # nb_centers =  np.array([leaf_info.origin for _,leaf_info in nb_leaves])
    # ancestors = get_ancestors(octree, find_node=nb_leaves[0][0])
    
    # The idea is to use the leaf tree to identify nearest surrounding 
    #    octleaves. This allows for selecting the localized area around the cluster
    #  Probably want to ignore the Z bounds and select all w/in x,y bounds
    # all_leaves = []
    # octree.traverse(lambda node,node_info: all_leaves.append((node,node_info)) if isinstance(node, o3d.geometry.OctreeLeafNode) else None)
    # leaf_points = [x[1].origin for x in all_leaves]
    # leaf_tree =  sps.KDTree(leaf_points)

    # containing_path = []
    # curr_node = octree.root_node
    # for index in ancestors:
    #     curr_node = curr_node.children[index]
    #     containing_path.append(curr_node)
        
    # parent_inds = parent.indices
    # nbhood = collective.select_by_index(parent_inds)

    # unique_nodes, node_occurences = np.unique(centers, return_counts=True)
    # inds = [node.indices for node,node_info in leaves]
    # combined_inds = chain.from_iterable(inds)

    # if len(centers) ==1:
    #     nb_inds = leaves[0][0].indices
    #     nbhood = collective.select_by_index(nb_inds)
    #     nb_labels = np.array( nbhood.cluster_dbscan(eps=.5, min_points=20, print_progress=True))
    #     nb_colors = color_and_draw_clusters(nbhood, nb_labels)
        
    breakpoint()        
