# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

from collections import defaultdict
import logging

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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
    pcd_pts = np.asarray(pcd.points)
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
    norms = np.asarray(stat_down.normals)

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
    algo_pcd_pts = np.asarray(algo_source_pcd.points)
    log.info(f"Getting points between the {start} and {end} percentiles")
    not_so_low_idxs, _ = get_percentile(algo_pcd_pts,start,end)
    low_cloud = algo_source_pcd.select_by_index(not_so_low_idxs)
    return low_cloud,not_so_low_idxs

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
        labels, clusters = choose_and_cluster(np.asarray(new_neighbors), main_pts, cluster_type, debug)

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
        cluster_pcd_pts = np.asarray(main_pcd.points)[cluster_idxs]
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
        low_cloud_pts = np.asarray(low_cloud.points)
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
        trunk_pts = np.asarray(trunk.points)
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
        algo_pcd_pts = np.asarray(algo_source_pcd.points)

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

# def find_scan_nums(points, scans):
#     for 
#     
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
    # [180.00500488, 454.43301392,  24.36133385]), array([ 34.05799866, 286.28399658, -21.90399933])), 'fragment': 370}, 'pcd6': {'bounds': (array(
    # [ 98.3030014 , 360.44699097,  16.38299942]), array([ 65.53700256, 327.67999268,   0.        ])), 'fragment': 420}, 'pcd7': {'bounds': (array(
    # [131.07200623, 360.44699097,  16.38299942]), array([ 98.30500031, 327.67999268,   0.        ])), 'fragment': 500}, 'pcd8': {'bounds': (array(
    # [131.07200623, 393.21600342,  23.        ]), array([ 65.53700256, 327.67999268,   0.        ])), 'fragment': 540}, 'pcd9': {'bounds': (array(
    # [131.07200623, 393.21600342,  16.38299942]), array([ 98.30500031, 360.44900513,   0.        ])), 'fragment': 580}, 'pcd10': {'bounds': (array(
    # [165.99499512, 393.21600342,  40.64099884]), array([ 98.30500031, 296.33099365,   0.        ])), 'fragment': 620}, 'pcd11': {'bounds': (array(
    # [185.56399536, 393.21600342,  31.97999954]), array([131.07200623, 327.67999268,   0.        ])), 'fragment': 660}, 'pcd12': {'bounds': (array(
    # [189.47099304, 438.6539917 ,  28.50699997]), array([ 40.65166855, 360.44900513,   0.        ])), 'fragment': 700}, 'pcd13': {'bounds': (array(
    # [131.07200623, 425.98300171,  20.59600067]), array([ 98.30500031, 393.21600342,   0.        ])), 'fragment': 760}, 'pcd14': {'bounds': (array(
    # [163.83900452, 458.29800415,  37.79150009]), array([ 65.54799652, 393.21600342,   0.        ])), 'fragment': 780}}


if __name__ == "__main__":
    import pickle
    # find_low_order_branches(file = 'data/input/27_vox_pt02_sta_6-4-3.pcd'
    #                         ,start = 'stem_id')
    # 
    # 
    # find_low_order_branches(file=skeletor)
    # find_low_order_branches(file='skeletor_super_clean.pcd',start = 'trunk_id')

    # pcd =  read_point_cloud("/code/code/pyQSM/compiled_vox_down_skio_raffai_60000000.pcd")
    # draw([pcd])
    
    pcd =   read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_60000000.pcd")
    pcd2 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_80000000.pcd")
    pcd4 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_200000000.pcd")
    pcd5 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_370000000.pcd")
    pcd6 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_420000000.pcd")

    pcd7 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_500000000.pcd")
    pcd8 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_540000000.pcd")
    pcd9 =  read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_580000000.pcd")
    pcd10 = read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_620000000.pcd")

    pcd11 = read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_660000000.pcd")
    pcd12 = read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_700000000.pcd")
    pcd13 = read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_760000000.pcd")
    pcd14 = read_point_cloud("data/input/SKIO/compiled_vox_down_skio_raffai_780000000.pcd")
    
    pcds = [pcd,pcd2,pcd4,pcd5,pcd6, pcd7,pcd8,pcd9,pcd10, pcd11,pcd12,pcd13,pcd14]
    
    pts = []
    colors = []
    print('aggregating...')
    for pcdi in pcds: 
      pts.extend(list(np.asarray(pcdi.points)))
      colors.extend(list(np.asarray(pcdi.colors)))
      del pcdi
    print('creating collective')
    collective = o3d.geometry.PointCloud()
    collective.points = o3d.utility.Vector3dVector(pts)
    collective.colors = o3d.utility.Vector3dVector(colors)
    # print('drawing collective')
    # draw(collective)
    # breakpoint()

    # Using newer 8 scans
    # (Pdb) collective.get_max_bound()
    # array([189.47099304, 458.29800415,  40.64099884])
    # (Pdb) collective.get_min_bound()
    # array([ 40.65166855, 296.33099365,   0.        ])
    # using 0 thru 5th percentile 
    # (Pdb) lowc.get_max_bound()
    # array([174.31300354, 456.79199219,   1.56596673])
    #  (if using 2.5 - array([172.69900513, 455.75299072,   0.74398184]))
    # (Pdb) lowc.get_min_bound()
    # array([44.4210014, 323.834015 3.33333330e-04])


    # Thru 6
    # lowc = get_low_cloud(collective, 35,45)[0]

    # 7, 8, 9, 10
    # no ground included in scans, easier to find waist height
    # lowc = get_low_cloud(collective, 0,5)[0]

    print('getting low cloud...')
    lowc = get_low_cloud(collective, 15,17)[0]
    draw(lowc)
    low_stem = get_stem_pcd(lowc)
    draw(low_stem)
    breakpoint()

    print('clustering')
    # Current settings, close trees being combined, need less neighbors, smaller eps
    labels = np.array( lowc.cluster_dbscan(eps=.5, min_points=20, print_progress=True))
    labels_and_pts = dict((('labels', labels),('points', np.asarray(lowc.points))))
    # with open('skio_clusters_low_pt5-20.pkl','wb') as f:
    #     pickle.load( f)

    # with open('skio_clusters.pkl','wb') as f:
    #     labels = pickle.dump(f)

    max_label = labels.max()
    # visualize the labels
    log.info(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    lowc.colors = o3d.utility.Vector3dVector(colors[:, :3])
    draw([lowc])
    breakpoint()

    # Examine/filter the clusters
    unique_vals, counts = np.unique(labels, return_counts=True)
    label_idls= [ np.where(labels ==val)[0] for val in unique_vals]
    clusters = [lowc.select_by_index(idls) for idls in label_idls]
    
    cluster_sizes = np.array([len(x) for x in label_idls])
    large_cutoff = np.percentile(cluster_sizes,85)
    large_clusters  = np.where(cluster_sizes> large_cutoff)[0]
    draw(clusters)
    for idc in large_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])
    draw(clusters)
    small_cutoff = np.percentile(cluster_sizes,30)
    small_clusters  = np.where(cluster_sizes< large_cutoff)[0]
    draw(clusters)
    for idc in small_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])
    draw(clusters)

    final_clusters = clusters

    

    cluster_centers = []
    clusters = []
    lowc_pts = np.asarray(lowc.points)
    for cluster in clusters in unique_vals: 
        bounds = (cluster.get_max_bound(), lowc.get_min_bound())
        coordinate_ranges = zip(*bounds)
        # bounds = (lowc.get_max_bound(), lowc.get_min_bound())


        # cluster_pts = lowc_pts[label_idxs]
        # cluster_center = get_center(cluster_pts)
        # cluster_centers.append(cluster_center)
        # cluster_pts.append(cluster_pts)
        # clusters.append(lowc.select_by_index(label_idxs))
        
    breakpoint()
        


