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
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn)
    )

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

        breakpoint()

# def build_trees_from_clusters(pcd,
#                                 cluster_pts):
#     """
#         Takes a parent pcd, a list of seed clusters 
#             and optionally, a neighborhood distance limit 
#         assignes each point in the parent pcd to one of the seed clusters 
#         by iterively calculating its nearest neighbor
#     """   
#     ptree = o3d.geometry.KDTreeFlann(pcd)
#     for cluster in clusters:





# bounds = { 'pcd': {'bounds': (pcd.get_max_bound(),pcd.get_min_bound()), 'fragment': 60}
    #         ,'pcd2':  {'bounds': (pcd2.get_max_bound(),pcd2.get_min_bound()), 'fragment': 80}
    #         ,'pcd4':  {'bounds': (pcd4.get_max_bound(),pcd4.get_min_bound()), 'fragment': 200}
    #         ,'pcd5':  {'bounds': (pcd5.get_max_bound(),pcd5.get_min_bound()), 'fragment': 370}
    #         ,'pcd6':  {'bounds': (pcd6.get_max_bound(),pcd6.get_min_bound()), 'fragment': 420}
    #         ,'pcd7':  {'bounds': (pcd7.get_max_bound(),pcd7.get_min_bound()), 'fragment': 500}
    #         ,'pcd8':  {'bounds': (pcd8.get_max_bound(),pcd8.get_min_bound()), 'fragment': 540}
    #         ,'pcd9':  {'bounds': (pcd9.get_max_bound(),pcd9.get_min_bound()), 'fragment': 580}
    #         ,'pcd10': {'bounds': (pcd10.get_max_bound(),pcd10.get_min_bound()), 'fragment': 620}
    #         ,'pcd11': {'bounds': (pcd11.get_max_bound(),pcd11.get_min_bound()), 'fragment': 660}
    #         ,'pcd12': {'bounds': (pcd12.get_max_bound(),pcd12.get_min_bound()), 'fragment': 700}
    #         ,'pcd13': {'bounds': (pcd13.get_max_bound(),pcd13.get_min_bound()), 'fragment': 760}
    #         ,'pcd14': {'bounds': (pcd14.get_max_bound(),pcd14.get_min_bound()), 'fragment': 780}
    #         }
    #     {'pcd': {'bounds': (array(
    # [131.06900024, 392.91799927,   0.        ]), array([ 46.77000046, 286.962677  ,  -6.60300016])), 'fragment': 60}, 'pcd2': {'bounds': (array(
    # [131.07200623, 360.44699097,  16.38299942]), array([ 98.30500031, 327.67999268,   0.        ])), 'fragment': 80}, 'pcd4': {'bounds': (array(
    # [131.07200623, 393.21600342,   0.        ]), array([ 46.77000046, 286.962677  ,  -8.04100037])), 'fragment': 200}, 'pcd5': {'bounds': (array(
    ## [180.00500488, 454.43301392,  24.36133385]), array([ 34.05799866, 286.28399658, -21.90399933])), 'fragment': 370}, 'pcd6': {'bounds': (array(
    # [ 98.3030014 , 360.44699097,  16.38299942]), array([ 65.53700256, 327.67999268,   0.        ])), 'fragment': 420}, 'pcd7': {'bounds': (array(
    ### [131.07200623, 360.44699097,  16.38299942]), array([ 98.30500031, 327.67999268,   0.        ])), 'fragment': 500}, 'pcd8': {'bounds': (array(
    ### [131.07200623, 393.21600342,  23.        ]), array([ 65.53700256, 327.67999268,   0.        ])), 'fragment': 540}, 'pcd9': {'bounds': (array(
    # [131.07200623, 393.21600342,  16.38299942]), array([ 98.30500031, 360.44900513,   0.        ])), 'fragment': 580}, 'pcd10': {'bounds': (array(
    # [165.99499512, 393.21600342,  40.64099884]), array([ 98.30500031, 296.33099365,   0.        ])), 'fragment': 620}, 'pcd11': {'bounds': (array(
    # [185.56399536, 393.21600342,  31.97999954]), array([131.07200623, 327.67999268,   0.        ])), 'fragment': 660}, 'pcd12': {'bounds': (array(
    # [189.47099304, 438.6539917 ,  28.50699997]), array([ 40.65166855, 360.44900513,   0.        ])), 'fragment': 700}, 'pcd13': {'bounds': (array(
    # [131.07200623, 425.98300171,  20.59600067]), array([ 98.30500031, 393.21600342,   0.        ])), 'fragment': 760}, 'pcd14': {'bounds': (array(
    # [163.83900452, 458.29800415,  37.79150009]), array([ 65.54799652, 393.21600342,   0.        ])), 'fragment': 780}}

def build_octree(pcd,
                   ot_depth=4,
                   ot_expand_factor = 0.01):
    print('octree division')
    octree = o3d.geometry.Octree(max_depth=ot_depth)
    octree.convert_from_point_cloud(pcd, size_expand=ot_expand_factor)
    # o3d.visualization.draw_geometries([octree])
    return octree

def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            n = len([x for x in node.children if x])
            # for child in node.children:
            #     if child is not None:
            #         n += 1

            num_pts = len(node.indices)
            spaces_by_depth = '    ' * node_info.depth
            print(
                f"""{spaces_by_depth}{node_info.child_index}: Internal node at depth {node_info.depth} 
                    has {n} children and {num_pts} points ({node_info.origin})""")

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        log.info(f'Reached a unrecognized node type: {type(node)}')
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop

# def find_local_octree_nodes(oct, point):
#     pt_node = oct.locate_leaf_node(point)
#     pt_node
#     traversed = []
#     traversed_info = []
#     def find(node, node_info, depth = 4):
#     octree.locate_leaf_node(pcd.points[0])

def get_ancestors(octree, 
                  find_node):
    ret_tree = []
    def is_ancestor(node, ancestor_tree, find_idx):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            for idc, child in enumerate(node.children):
                if child is not None:
                    if find_idx in node.indices:
                        ancestor_tree.append(idc)
                        print(f'found find idx {node=} {idc=}')  
                        return is_ancestor(child, ancestor_tree, find_idx)
                elif child is None:
                    print(f'child {idc} is none')
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
                return True
    find_idx = find_node.indices[0]
    is_ancestor(octree.root_node, ret_tree, find_idx)
    return ret_tree

# def traverse_get_parent(node, node_info):
#     if found_node: return True
#     early_stop = found_node or False
#     if isinstance(node, o3d.geometry.OctreeInternalPointNode):
#         for child in node_info.children:
#             if child is not None:
#                 if find_idx in node.indicies:
#                     curr_tree.append((node,node_info))
#                     log.info(f'found find idx in node {node} {node_info}')   
#                 else:
#                     early_stop = True
#     elif isinstance(node, o3d.geometry.OctreeLeafNode):
#         if node_info.origin == find_origin:
#             log.info(f'found find node, ending with {curr_tree}')
#             found_node = True
#             early_stop = True
#     return early_stop

def color_and_draw_clusters(pcd , labels):
    max_label = labels.max()
    # visualize the labels
    log.info(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcd.colors = colors
    draw([pcd])
    return colors

def node_to_pcd(parent_pcd,node):
    nb_inds = node.indices
    nbhood = parent_pcd.select_by_index(nb_inds)
    nb_labels = np.array( nbhood.cluster_dbscan(eps=.5, min_points=20, print_progress=True))
    # nb_colors = color_and_draw_clusters(nbhood, nb_labels)
    return nbhood, nb_labels


def get_leaves(node, leaves):
    if not isinstance(node, o3d.geometry.OctreeLeafNode): 
        for c_node in [x for x in node.children if x]:
            get_leaves(c_node, leaves)
    else:
        leaves.append(node)
        return leaves

def get_containing_tree(octree, 
                  find_node):
    ret_tree = []
    def is_ancestor(node, ancestor_tree, find_idx):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            for idc, child in enumerate(node.children):
                if child is not None:
                    if find_idx in node.indices:
                        ancestor_tree.append(idc)
                        print(f'found find idx {node=} {idc=}')  
                        return is_ancestor(child, ancestor_tree, find_idx)
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
                return True
    find_idx = find_node.indices[0]
    is_ancestor(octree.root_node, ret_tree, find_idx)
    return ret_tree

    
def load_clusters_pickle(pfile = 'cluster126_fact10_0to50.pkl',
                        single_pcd = False):
    import pickle
    contain_region = []    

    # For loading results of KNN loop
    with open(pfile,'rb') as f:
        tree_clusters = dict(pickle.load(f))
    pcds = []
    tree_pts= []
    tree_color=[]
    labels = np.asarray([x for x in tree_clusters.keys()])
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # for pcd, color in zip(tree_pcds, colors): pcd.paint_uniform_color(color[:3])
    for pts, color in zip(tree_clusters.values(), colors): 
        if single_pcd:
            tree_pts.extend(pts)
            tree_color.extend(color[:3]*len(pts))
        else:
            cols = [color[:3]]*len(pts)
            print(f'creating pcd with {len(pts)} points')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            pcds.append(pcd)
    if single_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tree_pts)
        pcd.colors = o3d.utility.Vector3dVector([x for x in tree_color])
        pcds.append(pcd)
    return pcds


def plot_squares(extents):
    fig, ax = plt.subplots()
    bounds = [((x_min,y_min),x_max-x_min, y_max-y_min ) for ((x_min, y_min,_), ( x_max, y_max,_)) in extents.values() ]
    g_x_min = min(arr([x[0][0] for x in extents.values()]))
    g_x_max = max(arr([x[1][0] for x in extents.values()]))
    g_y_min = min(arr([x[0][1] for x in extents.values()]))
    g_y_max = max(arr([x[1][1] for x in extents.values()]))
    plt.xlim(g_x_min - 1, g_x_max + 2)
    plt.ylim(g_y_min - 1, g_y_max + 2)
    for pt,w,h in bounds: 
        ax.add_patch(patches.Rectangle(pt, w, h, linewidth=1, edgecolor='black', facecolor='none'))
    plt.show()
    # global_x = min(extents[:,0,0])
    

def zoom(pcd,
        zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):

    collective = pcd
    low_pts = arr(collective.points)
    low_colors = arr(collective.colors)
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       and pt[1]>zoom_region[1][0] and  pt[1]<zoom_region[1][1])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in region]))
    pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in region]))
    draw(pcd)

def bin_colors(zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):
    # binning colors in pcd, finding most common
    # round(2) reduces 350k to ~47k colors
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) 
                if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       
                    and pt[1]>zoom_region[1][0] 
                    and  pt[1]<zoom_region[1][1])]

    cols = [','.join([f'{round(y,1)}' for y in x[1]]) for x in region]
    d_cols, cnts = np.unique(cols, return_counts=True)
    print(len(d_cols))
    print((max(cnts),min(cnts)))
    # removing points of any color other than the most common
    most_common = d_cols[np.where(cnts)]
    most_common_rgb = [tuple((float(num) for num in col.split(','))) for col in most_common]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) in most_common_rgb ]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) != tuple((1.0,1.0,1.0)) ]
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in color_limited]))
    limited_pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in color_limited]))
    draw(limited_pcd)


def find_extents(file_prefix = 'data/input/SKIO/part_skio_raffai',
                    return_pcd_list =False,
                    return_single_pcd = False):
    extents = {}
    pcds = []
    pts = []
    colors = []
    contains_region= []
    base = 20000000

    factor=5
    x_min, x_max= 85.13-factor, 139.06+factor
    y_min, y_max= 338.26-factor, 379.921+factor

    for i in range(40):
        num = base*(i+1) 
        file = f'{file_prefix}_{num}.pcd'
        pcd = read_point_cloud(file)
        # extent = (pcd.get_min_bound(),pcd.get_max_bound())
        # extents[file] = extent
        # colors.extend(list(arr(pcd.colors)))
        # if (( (extent[0][0]>x_min and extent[0][0]<x_max)
        #         or (extent[1][0]>x_min and extent[1][0]<x_max)
        #     or (extent[0][1]>y_min and extent[0][1]<y_max)
        #     or   (extent[1][1]>y_min and extent[1][1]<y_max))
        #     # and extent[1][2]>9.25
        # ):
        #     contains_region.append(i)
        print(f"{num}: ({extent[0]}, {extent[1]})")
        if return_pcd_list:
            pcds.append(pcd)
        elif return_single_pcd:
            pts.extend(list(arr(pcd.points)))   
            colors.extend(list(arr(pcd.colors)))
    if return_single_pcd:
        collective = o3d.geometry.PointCloud()
        collective.points = o3d.utility.Vector3dVector(pts)
        collective.colors = o3d.utility.Vector3dVector(colors)   
        pcds.append(pcd)     
    return extents, contains_region, pcds

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
    pcds = load_clusters_pickle()
    bnd_boxes = [pcd.get_oriented_bounding_box() for pcd in pcds]

    base = 20000000
    file_prefix = 'data/input/SKIO/part_skio_raffai'
    for idb, bnd_box in enumerate(bnd_boxes):
        if idb>0:
            contained_pts = []
            all_colors = []
            for i in range(39):
                num = base*(i+1) 
                file = f'{file_prefix}_{num}.pcd'
                print(f'checking file {file}')
                pcd = read_point_cloud(file)
                pts = arr(pcd.points)
                if len(pts)>0:
                    cols = arr(pcd.colors)
                    pts_vect = o3d.utility.Vector3dVector(pts)
                    pt_ids = bnd_box.get_point_indices_within_bounding_box(pts_vect) 
                    if len(pt_ids)>0:
                        pt_values = pts[pt_ids]
                        colors = cols[pt_ids]
                        print(f'adding {len(pt_ids)} out of {len(pts)}')
                        contained_pts.extend(pt_values)
                        all_colors.extend(colors)
            try:
                file = f'whole_clus/cluster_{idb}_all_points.pcd'
                print(f'writing pcd file {file}')
                
                # Reversing the initial voxelization done
                #   to make the dataset manageable 
                # KNN search each cluster against nearby pts in
                #   the original scan. Drastically increase detail
                cluster_tree = sps.KDTree(arr(pcds[idb].points))
                whole_tree = sps.KDTree(contained_pts)
                # dists,nbrs = whole_tree.query(query_pts, k=750, distance_upper_bound= .2) 
                nbrs = cluster_tree.query_ball_tree(whole_tree, r= .3) 
                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x!= len(contained_pts)]
                print(f'{len(nbrs)} nbrs found for cluster {idx}')
                # for nbr_pt in nbr_pts:
                #     high_c_pt_assns[tuple(nbr_pt)] = idx


                # final_pts = np.append(arr(contained_pts)[nbrs])#,pcds[idb].points)
                # final_colors = np.append(arr(all_colors)[nbrs])#,pcds[idb].colors)
                final_pts =    arr(contained_pts)[nbrs]
                final_colors = arr(all_colors)[nbrs]
                wpcd = o3d.geometry.PointCloud()
                wpcd.points = o3d.utility.Vector3dVector(final_pts)
                wpcd.colors = o3d.utility.Vector3dVector(final_colors)  
                o3d.io.write_point_cloud(file, wpcd)

                draw(wpcd)
                out_pts = np.delete(contained_pts,nbrs ,axis =0)
                out_pcd = o3d.geometry.PointCloud()
                out_pcd.points = o3d.utility.Vector3dVector(out_pts)
                wpcd.paint_uniform_color([0,0,1])                    
                out_pcd.paint_uniform_color([1,0,0])                
                draw([wpcd,out_pcd])

                breakpoint()

            except Exception as e:
                breakpoint()
                print(f'error {e} getting clouds')
    breakpoint()
    # o3d.geometry.get_oriented_bounding_box


    # collective_min = [ 34.05799866, 286.28399658, -21.90399933]
    # collective_max = [189.47099304, 458.29800415,  40.64099884]


    # result = np.zeros(factor * factor, pix_num, pix_num)
    # n = 0
    # for r in range(0, img.shape[0], pix_num):
    #     for c in range(0, img.shape[1], pix_num):
    #         result[n, :, :] = img[r:r + pix_num, c:c + pix_num]    

    # # extent = collective.get_max_bound() - collective.get_min_bound()
    
    # o3d.io.write_point_cloud('collective.pcd',collective)
    collective = o3d.io.read_point_cloud('collective.pcd')
    
    zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]
    low_pts = arr(collective.points)
    low_colors = arr(collective.colors)
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       and pt[1]>zoom_region[1][0] and  pt[1]<zoom_region[1][1])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in region]))
    pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in region]))
    draw(pcd)

    # pcd_cols = arr([tuple((round(y,1)for y in x[1])) for x in region])
    # pcd.colors = o3d.utility.Vector3dVector(pcd_cols)
        # [float(num) for num in col.split(',')] for col in cols]))


    # print('getting low cloud...')
    # lowc, lowc_ids_from_col = get_low_cloud(collective, 16,18)
    # lowc = clean_cloud(lowc)
    # highc, highc_ids_from_col = get_low_cloud(collective, 18,100)
    # o3d.io.write_point_cloud('low_cloud_all_16-18pct.pcd',lowc)
    # o3d.io.write_point_cloud('collective_highc_18plus.pcd',highc)
    # draw(lowc)

    lowc= o3d.io.read_point_cloud('low_cloud_all_16-18pct.pcd')
    highc= o3d.io.read_point_cloud('collective_highc_18plus.pcd')
    # print('low_c')
    # print(lowc.get_max_bound())
    # print(lowc.get_min_bound())
    # print('high_c')
    # print(highc.get_max_bound())
    # print(highc.get_min_bound())

    # breakpoint()
    # draw(lowc)
    # low_stem = get_stem_pcd(lowc)
    # draw(low_stem)
    # breakpoint()

    print('clustering')
    # Current settings, close trees being combined, need less neighbors, smaller eps
    # labels = np.array( lowc.cluster_dbscan(eps=.5, min_points=20, print_progress=True))   
    # with open('skio_labels_low_16-18_cluster_pt5-20.pkl','wb') as f:
    #     pickle.dump(labels,f)
    
    with open('skio_labels_low_16-18_cluster_pt5-20.pkl','rb') as f:
        labels = pickle.load(f)
    breakpoint()

    # min_x = 97
    # max_x = 118
    # min_y = 342
    # max_y = 362
    # if (( (pcd.get_min_bound()[0]>97 and pcd.get_min_bound()[0]<118.44)
    #             or (pcd.get_max_bound()[0]>97 and pcd.get_max_bound()[0]<118.44)
    #         or (pcd.get_min_bound()[1]>342.10 and pcd.get_min_bound()[1]<362.86)
    #         or   (pcd.get_max_bound()[1]>342.10 and pcd.get_max_bound()[1]<362.86))
    #         and pcd.get_max_bound()[2]>9.25
    #     ):
   
    max_label = labels.max()
    # visualize the labels
    # log.info(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # lowc.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # draw([lowc])

    # Define subpcds implied by labels
    unique_vals, counts = np.unique(labels, return_counts=True)
    label_idls = [ np.where(labels ==val)[0] for val in unique_vals]
    clusters = [lowc.select_by_index(idls) for idls in label_idls]
    cluster_centers = [get_center(arr(x.points)) for x in clusters]
    # breakpoint()

    import scipy.spatial as sps
    import itertools
    from copy import deepcopy


    # Filtering down the cluster list to reduce computation needed 
    
    ## Filtering out unlikley trunk canidates 
    # cluster_sizes = np.array([len(x) for x in label_idls])
    # large_cutoff = np.percentile(clu0ster_sizes,85)
    # large_clusters  = np.where(cluster_sizes> large_cutoff)[0]
    # draw(clusters)

    # for idc in large_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])
    # draw(clusters)
    # small_cutoff = np.percentile(cluster_sizes,30)
    # small_clusters  = np.where(cluster_sizes< large_cutoff)[0]
    # draw(clusters)
    # for idc in small_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])
    # draw(clusters)

    # Limiting to just the clusters near cluster[19]
    factor = 10 # @40 reduces clusters 50%, @20 we get a solid 20 or so clusters in the
    center_id = 126
    zoom_region = [(clusters[center_id].get_max_bound()[0]+factor,    
                    clusters[center_id].get_min_bound()[0]-factor),     
                    (clusters[center_id].get_max_bound()[1]+factor,    
                    clusters[center_id].get_min_bound()[1]-factor)] # y max/min
    new_clusters = [cluster for cluster in clusters  if (cluster.get_max_bound()[0]<zoom_region[0][0] and              cluster.get_max_bound()[1]<zoom_region[1][0] and              cluster.get_min_bound()[0]>zoom_region[0][1] and              cluster.get_min_bound()[1]>zoom_region[1][1])]
    new_cluster_pts = [arr(cluster.points) for cluster in new_clusters]
    ## color and draw these local clusters
    clustered_pts = list(chain.from_iterable(new_cluster_pts))
    labels = arr(range(len(clustered_pts)))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    for color,cluster in zip(colors,new_clusters): cluster.paint_uniform_color(color[:3])
    draw(new_clusters)

    # Reclustering the local points
    #  for manual runs, reducing multi tree clusters
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr(clustered_pts))
    labels = np.array( pcd.cluster_dbscan(eps=.25, min_points=20, print_progress=True))   
    max_label = labels.max()
    # visualize the labels
    log.info(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    first = colors[0]
    colors[0] = colors[-1]
    colors[-1]=first
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # draw([pcd])
    breakpoint()

    unique_vals, counts = np.unique(labels, return_counts=True)
    label_idls= [ np.where(labels ==val)[0] for val in unique_vals]
    clusters = [pcd.select_by_index(idls) for idls in label_idls if len(idls)>200]
    # cluster_ids_in_col = [arr(lowc_ids_from_col)[idls] for idls in label_idls]
    cluster_extents = [cluster.get_max_bound()-cluster.get_min_bound() 
                                for cluster in clusters]
    cluster_pts = [arr(cluster.points) for cluster in clusters]
    # centers = [get_center(arr(x.points)) for x in clusters]


    print('preparing KDtree')
    highc_pts = arr(highc.points)
    # highc_tree = sps.KDTree(highc_pts)
    # with open('cluster126_fact100_iter50.pkl','wb') as f:  pickle.dump(tree_pts,f)
    # with open('entire_50iter_assns.pkl','rb') as f: high_c_pt_assns = pickle.load(f)
    with open('highc_KDTree.pkl','rb') as f:
        highc_tree = pickle.load(f)
    highc_pts = highc_tree.data
    
    minz = min(arr(lowc.points)[:,2])
    

    # clustered_ids = list(chain.from_iterable(cluster_ids_in_col))
    # mask = np.ones(col_pts.shape[0], dtype=bool)
    # mask[clustered_ids] = False
    # not_yet_traversed = np.where(mask)[0] # ids of non-clustered
    # not_yet_traversed = [idx for idx,_ in enumerate(highc_pts)]
        
    traversed = []
    tree_pts = [list(arr(x.points)) for x in clusters]
    curr_pts = cluster_pts
    high_c_pt_assns = defaultdict(lambda:-1) 

    ########## Notes for continued Runs #############
    # Run more that 100 iters, there are looong trees to be built
    # 

    #####################################
    iters = 50
    recreate = False
    for i in range(100):
        print('start iter')
        if iters<=0:
            iters =50
            tree_pts = defaultdict(list)
            for k,v in high_c_pt_assns.items(): tree_pts[v].append(k)
            with open(f'cluster126_fact10_0to50.pkl','wb') as f:
                to_write = list([tuple((k,pt_list)) for k,pt_list in tree_pts.items()])
                pickle.dump(to_write,f)

            # for j in [(0,50),(50,100),(100,150),(150,200), (200,281)]:
            #     try:
            #         with open(f'cluster126_fact10_{j[0]}to{j[1]}.pkl','wb') as f:  
            #             to_write = list([tuple((k,pt_list)) for k,pt_list in tree_pts.items()])
            #             pickle.dump(to_write,f)
            #     except Exception as e:
            #         breakpoint()
            #         print(f'error {e}')
            if i %30==0:
                tree_pcds= []
                labels = arr(range(len(tree_pts)))
                max_label = labels.max()
                colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                # for pcd, color in zip(tree_pcds, colors): pcd.paint_uniform_color(color[:3])
                for pts, color in zip(tree_pts.values(), colors):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.paint_uniform_color(color[:3])
                    tree_pcds.append(o3d.geometry.PointCloud())
                draw(tree_pcds)
                test = deepcopy(tree_pcds)
                test.extend(clusters)
                for c in clusters:c.paint_uniform_color([1,0,0])
                draw(test)
                breakpoint()
        [idx for idx,_ in enumerate(curr_pts) if len(curr_pts) ==0]

        iters=iters-1
        for idx, (cluster, cluster_extent) in enumerate(zip(clusters, 
                                                                    cluster_extents)):
            if len(curr_pts[idx])>0:
                print(f'querying {i}')
                dists,nbrs = highc_tree.query(curr_pts[idx],k=750,distance_upper_bound= .25) #max(cluster_extent))
                print(f'reducting nbrs {idx}')

                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=40807492]
                print(f'{len(nbrs)} nbrs found for cluster {idx}')
                # Note: distances are all rather similar -> uniform distribution of collective
                nbr_pts = [nbr_pt for nbr_pt in highc_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                for nbr_pt in nbr_pts:
                    high_c_pt_assns[tuple(nbr_pt)] = idx

                curr_pts[idx] = nbr_pts # note how these are overwritten each cycle
                # tree_pts[idx].extend(curr_pts[idx])
                # tree_idxs[idx].extend(nbrs)
                traversed.extend(nbrs)
            else:
                print(f'no more new neighbors for cluster {idx}')
                curr_pts[idx] = []
            # if idx == 19:
            #     print('draw nbrs')
            #     tree_pcd = highc.select_by_index(d_nbrs)
            #     draw([tree_pcd, cluster])
            #     breakpoint()
    # breakpoint()
    print('finish!')
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
