# with open("/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

from collections import defaultdict
import logging
from itertools import chain
from copy import deepcopy

import scipy.spatial as sps
import open3d as o3d
import numpy as np
from numpy import asarray as arr
import matplotlib.pyplot as plt
from matplotlib import patches
from open3d.io import read_point_cloud, write_point_cloud

from geometry.skeletonize import extract_skeleton, extract_topology
from tree_isolation import extend_seed_clusters
from utils.fit import cluster_DBSCAN, fit_shape_RANSAC, kmeans
from utils.fit import choose_and_cluster, cluster_DBSCAN, fit_shape_RANSAC, kmeans
from utils.io import save, load, save_line_set
from utils.lib_integration import find_neighbors_in_ball
from viz.color import color_on_percentile
from viz.viz_utils import color_continuous_map
from utils.math_utils import (
    get_angles,
    get_center,
    get_radius,
    rotation_matrix_from_arr,
    unit_vector,
    get_percentile,
)

from set_config import config
from geometry.point_cloud_processing import ( 
    cluster_plus,
    crop_by_percentile,
    filter_by_norm,
    clean_cloud,
    crop, get_shape,
    create_one_or_many_pcds,
    orientation_from_norms,
    filter_by_norm,
    get_ball_mesh
)
from viz.viz_utils import iter_draw, draw
from geometry.point_cloud_processing import join_pcds
from geometry.reconstruction import get_neighbors_kdtree
from tree_isolation import pcds_from_extend_seed_file
import pickle 
from geometry.mesh_processing import map_density
from utils.plotting import plot_dist_dist

log = logging.getLogger('calc')

skeletor = "/code/Research/lidar/converted_pcs/skeletor.pts"
s27 = "/code/Research/lidar/converted_pcs/Secrest27_05.pts"
s32 = "/code/Research/lidar/converted_pcs/Secrest32_06.pts"

s27d = "data/input/s27_downsample_0.04.pcd"
s32d = "data/input/s32_downsample_0.04.pcd"

normals_radius   = config['stem']["normals_radius"]
normals_nn       = config['stem']["normals_nn"]    
nb_neighbors   = config['stem']["stem_neighbors"]
std_ratio      = config['stem']["stem_ratio"]
angle_cutoff      = config['stem']["angle_cutoff"]
voxel_size     = config['stem']["stem_voxel_size"]
post_id_stat_down = config['stem']["post_id_stat_down"]

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
    log.info("Estimating and orienting normals")
    # cropping out ground points
    pcd_pts = arr(pcd.points)
    pcd_cropped_idxs = crop(pcd_pts, minz=np.min(pcd_pts[:, 2]) + 0.5)
    pcd = pcd.select_by_index(pcd_cropped_idxs)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))
    pcd.orient_normals_consistent_tangent_plane(100)
    log.info("Filtering on normal angles")
    stem_cloud = filter_by_norm(pcd, angle_cutoff)
    log.info("cleaning result (if requested)")
    if voxel_size:
        stem_cloud = stem_cloud.voxel_down_sample(voxel_size=voxel_size)
    if post_id_stat_down:
        _, ind = test.remove_statistical_outlier(nb_neighbors= nb_neighbors, std_ratio=std_ratio)
        test = test.select_by_index(ind)
    # recluster and remove all branch starts. Then fill in w detail from pcd
    # clustering result to isolate trunk
    # works well at eps .35,.45 and min pts 80-110
    # clusters = cluster_plus(stem_cloud,eps=.05*7, min_points=80,return_pcds=True,from_points=False)
    # for i in range(1,10): 
    #     cluster_plus(test,eps=.05*7, min_points=90,return_pcds=True,from_points=False)
    # write_point_cloud(f'{inputs}/skeletor_dtrunk.pcd',dt4)

    #  pcd.detect_planar_patches(
    # normal_variance_threshold_deg=60,
    # coplanarity_deg=75,
    # outlier_ratio=0.75,
    # min_plane_edge_length=0,
    # min_num_points=0,
    # search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))


    try:
        o3d.visualization.draw_geometries(stem_cloud, point_show_normals=True)
        o3d.visualization.draw([stem_cloud])
        breakpoint()
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=normals_radius)
        sp.translate(pcd.points[50000])
        pcd_tree = o3d.geometry.KDTreeFlann(stem_cloud)
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[50000], normals_radius)
        np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
        o3d.visualization.draw([stem_cloud])
        o3d.visualization.draw([pcd,sp])
        breakpoint()
    except Exception as e:
        breakpoint()
        print('')
    return stem_cloud

def compare_normals_approaches(stat_down):
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


def fit_cyl_to_cluster(main_pcd,
                        curr_pts,
                        last_radius,
                        cluster_idxs,   
                        cyls=[],
                        cyl_details=[],debug=False):
    cyl_mesh = None
    fit_radius = 0
    log.info(f"Attempting to fit a 2D circle to projection of points")
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
        log.info(f"good fit found, adding cyl to list")
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
    return good_fit_found
    

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
    3. Approimates a circle to surround the neighbors of the choose_and_clustercluster points
        -

    Returns:
        _description_
    """
    if branches == [[]]:
        branches[0].append(total_found)
    main_pts = arr(main_pcd.points)

    good_fit_found = fit_cyl_to_cluster(main_pcd, curr_pts, last_radius, cluster_idxs,cyls=cyls, cyl_details=cyl_details,debug=debug)
    
    log.info(f"getting new neighbors ")
    # get all points within radius of the former neighbor
    #    group's center point.
    # Exclude any that have already been found
    sphere, new_neighbors, center, radius = find_neighbors_in_ball(curr_pts, main_pts, total_found)
    spheres.append(sphere)
    # full_tree = sps.KDTree(main_pts)
    # new_neighbors = get_neighbors_in_tree(curr_pts,full_tree, last_radius)

    clusters = ()
    if len(new_neighbors) > 0:
        log.info(f"Clustering neighbor points found in ball")
        cluster_type = "kmeans" if not good_fit_found else "DBSCAN"
        # Cluster new neighbors to finpd potential branches
        labels, clusters = choose_and_cluster(arr(new_neighbors), main_pts, cluster_type, debug= debug)

    if clusters == [] or len(new_neighbors) < config['sphere']["min_contained_points"]:
        return []
    else:
        if (len(labels) > 2
            or debug):
            try:
                test = iter_draw(clusters, main_pcd)
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
                test = main_pcd.select_by_index(new_neighbors)
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

        fit_remaining_branch = sphere_step(
            cluster_pcd_pts,
            cluster_radius,
            main_pcd,
            cluster_idxs,
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
                            #  file = '27_pt02.pcd',
                            file = 'inputs/skeletor_clean.pcd',
                             extract_skeleton = False,
                             use_knn=True,
                             debug = False):
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
    pcd = read_point_cloud(file, print_progress=True)
    if start == 'initial_clean':
        started = True
        # Reading in cloud and smooth
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
        if debug:
            draw(stat_down)
        breakpoint()
    
    if start == 'stem_id' or started:
        if not started:
            stat_down = read_point_cloud(file, print_progress=True)
            started = True
        # "data/results/saves/27_vox_pt02_sta_6-4-3.pcd" # < ---  post-clean pre-stem
        stem_cloud = get_stem_pcd(stat_down)
        if debug:
            draw(stem_cloud)
        breakpoint()

    if start == 'trunk_id' or started:
        if not started:
            stem_cloud = read_point_cloud(file, print_progress=True)
            started = True
        log.info("Identifying low percentile by height...")
        low_cloud, low_idxs = crop_by_percentile(stem_cloud)
        low_cloud_pts = arr(low_cloud.points)
        # draw(low_cloud)

        trunk, trunk_idxs= cluster_plus(low_cloud, top=1)
        trunk_pts = arr(trunk.points)
        trunk_stem_idxs = low_idxs[trunk_idxs]
        if debug:
            draw(trunk)
        
        # next_slice_full, slice_idxs = get_low_cloud(stem_cloud, 
        #                             config['trunk']['upper_pctile'], 
        #                             config['trunk']['upper_pctile']+2)
        
        # slice_pcd, slice_idxs = cluster_plus(next_slice_full,top=1)
        # draw(next_slice_trunk)

    if start == 'clustering' or started:
        started=True
        ## Finding the inititial set of new neighbors
        # stem_cloud =read_point_cloud(f'{inputs}/skeletor_stem_ds2_ac10.pcd')
        # trunk =read_point_cloud(f'{inputs}/skeletor_trunk_ds2_ac10_epspt35_mp80.pcd')
        dtrunk =read_point_cloud(f'{inputs}/skeletor_dtrunk.pcd')
        # stem_cloud_clusters = load('inputs/skeletor_stem_clust_ds2_ac10_epspt35_mp80.pkl')
        # clusters = [x[1] for x in stem_cloud_clusters]
        # cluster_pcds = create_one_or_many_pcds(pts = clusters, lables = [x[0] for x in test])
        # breakpoint()
        
        from copy import deepcopy
        # stem_cloud_pts = arr(stem_cloud.points)
        # obb = dtrunk.get_minimal_oriented_bounding_box()
        # obb.color = (1, 0, 0)
        
        # up_shifted_obb = deepcopy(obb).translate((0,0,1),relative = True)
        # up_shifted_obb.scale(1.1,center = up_shifted_obb.center)
        # draw([trunk,obb,up_shifted_obb])
        # new_pt_ids = up_shifted_obb.get_point_indices_within_bounding_box( 
        #                             o3d.utility.Vector3dVector(stem_cloud_pts) )
        # old_pt_ids = obb.get_point_indices_within_bounding_box( 
        #                             o3d.utility.Vector3dVector(stem_cloud_pts) )

        # next_slice =stem_cloud.select_by_index(new_pt_ids)
        # draw([next_slice,trunk,obb,up_shifted_obb])
        
        # new_neighbors = np.setdiff1d(new_pt_ids, old_pt_ids)
        # nn_pts = stem_cloud_pts[new_neighbors]
        # total_found = new_pt_ids
        # breakpoint()

        ### Might make sense to fill out the lower trunk
        #### more with the below code 
        # lowc, idxs = crop_by_percentile(trunk_proper,0,80)
        # draw(lowc)
        # dt6, nbrs, chained_nbrs = get_neighbors_kdtree(pcd, dt5, dist=.3)
        # test = source_pcd.select_by_index(chained_nbrs)
        # test.paint_uniform_color([1,0,1])
        # draw([test,trunk_proper])
        # breakpoint()
    
        source_pcd = pcd 
        source_pcd_pts = arr(source_pcd.points)
        trunk_proper, nbrs, chained_nbrs = get_neighbors_kdtree(source_pcd, dtrunk, dist=.3)
        trunk_nbr_pcd, new_nbrs, new_chained_nbrs = get_neighbors_kdtree(source_pcd, trunk_proper, dist=.3)
        seed_cluster_idxs = np.setdiff1d(new_chained_nbrs, chained_nbrs)
        total_found = np.concatenate([chained_nbrs, new_chained_nbrs])

        

        new_nbr_pcd = source_pcd.select_by_index(seed_cluster_idxs)
        branch_seeds_pcd, branch_seed_idxs = crop_by_percentile(new_nbr_pcd,50,100)
        clean_branch_seeds_pcd, clean_branch_seed_idxs= branch_seeds_pcd.remove_statistical_outlier(nb_neighbors= nb_neighbors, std_ratio=std_ratio)
        branch_seed_idxs_in_src = seed_cluster_idxs[branch_seed_idxs[clean_branch_seed_idxs]]
        # testing that correct ids were pulled
        if debug:
            branch_seeds_pcd = source_pcd.select_by_index(branch_seed_idxs_in_src)
            branch_seeds_pcd.paint_uniform_color([1,0,1])
            draw([branch_seeds_pcd,trunk_proper])



        branch_seed_pts = source_pcd_pts[branch_seed_idxs_in_src]
        seed_clusters = cluster_plus(clean_branch_seeds_pcd, eps=.16, min_points=20,return_pcds=True,from_points=False)
        seed_clusters = seed_clusters[1:] # seed_clusters[0] is uncategorized points
        if debug:
            draw(seed_clusters)

        clusters_and_pts = [(idx,arr(cluster.points))for idc, cluster in enumerate(seed_clusters)]
        save('skeletor_trunk_total_found.pkl',total_found)
        save('skeletor_final_branch_seeds.pkl',clusters_and_pts)
        # query_pts, query_pt_idxs = top_trunk_pts, top_trunk_pts_stem_idxs
        # sphere, neighbors, center, radius = find_neighbors_in_ball(query_pts, stem_cloud_pts,
        #                             branch_seed_pts                                query_pt_idxs,
        #                                                             radius=1.8, draw_results=True)

        # nn_pcd = stem_cloud.select_by_index(query_pt_idxs)
        # nn_pts = source_pcd_pts[new_neighbors]
        # sphere=get_shape(pts=None, shape='sphere', as_pts=True,   center=tuple(center), radius=radius)
        # o3d.visualization.draw_geometries([nn_pcd,trunk,sphere])
        # labels, clusters = choose_and_cluster(arr(new_neighbors), nn_pts, cluster_type, debug= debug)

    if start == 'knn' or started:
        source_pcd = pcd 

        total_found = load('skeletor_trunk_total_found.pkl')
        clusters_and_pts = load('skeletor_final_branch_seeds.pkl')
        final_seed_clusters = create_one_or_many_pcds(pts = [x[1] for x in clusters_and_pts], labels = [x[0] for x in clusters_and_pts])
        clusters_and_idxs = [(idc,cluster) for idc, cluster in enumerate(final_seed_clusters)]
        not_yet_classfied = source_pcd.select_by_index(total_found, invert=True )
        tree_pcds = extend_seed_clusters(clusters_and_idxs, not_yet_classfied, 'skeletor4', k=50, max_distance=.1, cycles= 200, save_every = 50, draw_every = 50)
                
        
        # clusters_and_idxs2 = [(idc,cluster) for idc, cluster in enumerate(seed_clusters2)]
        # not_yet_classfied = source_pcd.select_by_index(total_found, invert=True )
        # tree_pcds = extend_seed_clusters(clusters_and_idxs2, not_yet_classfied, 'skeletor', k=200, max_distance=.1, cycles= 150, save_every = 50, draw_every = 50)
        
        # clusters_and_idxs = [(idc,cluster) for idc, cluster in enumerate(seed_clusters)]
        # tree_pcds2 = extend_seed_clusters(clusters_and_idxs, not_yet_classfied, 'skeletor2', k=200, max_distance=.1, cycles= 150, save_every = 50, draw_every = 50)
        breakpoint()
    else:
        fit_radius = 1.8# seed_clusters[0] is uncategorized points
        started = True
        log.info("Running sphere algo")        
        res, id_to_num, cyls, cyl_details = sphere_step(
            curr_pts = branch_seed_pts,
            last_radius = fit_radius, # The radius of the last cyliner fit
            main_pcd = source_pcd, # The parent point cloud being traversed
            cluster_idxs = branch_seed_idxs_in_src, # The# res, id_to_num, cyls, cyl_details = sphere_step(    curr_pts = branch_seed_pts,    last_radius = fit_radius,     main_pcd = source_pcd,     cluster_idxs = branch_seed_idxs_in_src,     branch_order = 0,    branch_num = 0,    total_found = list(total_found),    debug = True) indexes of the points in the last cluster
            branch_order = 0,
            branch_num = 0,
            total_found = list(total_found)
            # debug = True
        )
        breakpoint()

        # Coloring and drawing the results of 
        #  sphere step
        # iter_draw(res[0], algo_source_pcd)

        # branch_index = defaultdict(list)
        # for idx, branch_num in id_to_num.items():
        #     branch_index[branch_num].append(idx)
        # try:
        #     iter_draw(list(branch_index.values()), algo_pcd_pts)
        # except Exception as e:
        #     log.info("error" + e)

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

def filter_line_set(ls,idxs):
    points = arr(ls.points)[idxs]
    lines = arr([(u,v) for u,v in arr(ls.lines) if u in points or v in points])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def exclude_dense_areas(pcd, radius=.05):
    import rustworkx as rx
    radius=.1
    degrees,cnts,line_set= get_pairs(query_pts=arr(pcd.points),radius=radius, return_line_set=True)
    draw(line_set)
    # making graph, finding connected components
    lines =  arr(line_set.lines)
    pts = arr(line_set.points)
    g = rx.PyGraph()
    _=g.add_nodes_from([pt for idp,pt in enumerate(arr(line_set.points))])
    _=g.add_edges_from_no_data([tuple(pair) for pair in arr(line_set.lines)])
    conn_comps = arr([x for x in sorted(rx.connected_components(g), key=len, reverse=True)])
    print(len(conn_comps))
    print([len(x) for x in conn_comps])
    print( len(np.where(arr([len(x) for x in conn_comps])>1)[0]))
    # nodes = [arr(g.nodes())[list(conn_comp)] for conn_comp in conn_comps]
    # components to subgraphs
    large_comps= [x for x in conn_comps[0:500] if len(x)>150 and len(x)<200]
    large_comps= [x for x in conn_comps[0:500] if len(x)<100]
    subgraphs = list(chain.from_iterable([g.subgraph(list(conn_comp)) for conn_comp in large_comps]))
    # subgraphs to line sets
    pts = list([subgraph.nodes() for subgraph in subgraphs])
    lines = list([subgraph.edge_list() for subgraph in subgraphs])
    line_sets = [o3d.geometry.LineSet() for _ in range(len(subgraphs))]
    for idp,sub_line_set in enumerate(line_sets): sub_line_set.points = o3d.utility.Vector3dVector(pts[idp])
    for idp,sub_line_set in enumerate(line_sets): sub_line_set.lines = o3d.utility.Vector2iVector(lines[idp])
    draw(line_sets)
    

    branches, _, analouge_idxs = get_neighbors_kdtree(trim, query_pts=new_pts,k=200, dist=.01)
    draw(branches)


if __name__ == "__main__":

    root_dir = '/media/penguaman/code/ActualCode/Research/pyQSM/'
    inputs = 'data/skeletor/inputs'
    from ray_casting import sparse_cast_w_intersections, project_to_image,mri,cast_rays, project_pcd
    meshfix

    trim = read_point_cloud(f'{inputs}/trim/skeletor_full_ds2.pcd')
    pcd = trim.uniform_down_sample(15)
    # pcd = read_point_cloud('finpcdal_branch_cc_iso_pt9_top100_mdpt_k50.pcd')
    from ray_casting import sparse_cast_w_intersections, project_to_image,mri,cast_rays, project_pcd
    mesh = project_pcd(pcd)
    
    # breakpoint()
    # file = 'data/skeletor/skel_clean_ext_full.pkl'   
    # pcds= pcds_from_extend_seed_file(file)   
    
    # file = 'data/skeletor/skel_clean_ext_cutoff10.pkl'   
    # branch_pcds= pcds_from_extend_seed_file(file)
    # draw(pcds)
    # draw(branch_pcds)
    # to_project = branch_pcds
    # ds_branches = [pcd.uniform_down_sample(15).translate([0,0,0]) for pcd in branch_pcds]
    # pts = arr([x for x in chain.from_iterable([arr(pcd.points) for pcd in to_project])])

    # breakpoint()

    # breakpoint()
    from viz.color import color_distribution
    from utils.lib_integration import get_pairs
    # source_pcd = read_point_cloud(f'{inputs}/skeletor_clean.pcd')
    trim = read_point_cloud(f'{inputs}/trim/skeletor_full_ds2.pcd')
    # source_pcd = read_point_cloud(f'{inputs}/skeletor_clean_ds2.pcd') 
    # file = 'data/skeletor/skel_clean_ext_full.pkl'   
    # pcds= pcds_from_extend_seed_file(file)
    # # branches = []
    # # leaves = []
    # # low_idx_lists= []
    # # # degree_lists = [] 
    # for idp, pcd in enumerate(pcds):

        
    #     # plot_dist_dist(pcd)
    #     # distances = pcd.compute_nearest_neighbor_distance()
    #     pts = arr(pcd.points)
    #     detailed_branch, _, analouge_idxs = get_neighbors_kdtree(trim, query_pts=pts,k=200, dist=.1)
    #     print(f'Orig Size {len(pts)}')
    #     print(f'Detailed Size {len(detailed_branch.points)}')
    #     if idp<2:
    #         draw(detailed_branch)
    #     write_point_cloud(f'data/skeletor/branch_and_leaves/detailed_branch{idp}.pcd',pcd)
        # # local_density = []  
        # degrees,cnts,line_set= get_pairs(query_pts=arr(pcd.points),radius=.3)
        # # breakpoint()
        # # density_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr(pcd.points)))
        # # color_continuous_map(density_pcd,arr(degrees))
        # # draw(density_pcd)
        # lowd_idxs = np.where(degrees<np.percentile(degrees,30))[0]
        # branch = pcd.select_by_index(lowd_idxs)
        # # leafs = pcd.select_by_index(lowd_idxs,invert=True)
        # branches.append(branch)
        # # leaves.append(leafs)
        # low_idx_lists.append(lowd_idxs)
        # # draw(branch)
        # # draw(leafs)
        # print(f'finished {idp}')
        # draw(branch)
        # draw(leaves)

        # breakpoint()
        # # avg_dist = np.mean(distances)
        # colored_jleaves, _, analouge_idxs = get_neighbors_kdtree(trim, jleaves, dist=.01)
        # colored_leaves, _, analouge_idxs = get_neighbors_kdtree(colored_pcd, leaves, dist=.01)
        # colored_branch, _, analouge_idxs = get_neighbors_kdtree(colored_pcd, branch, dist=.01)
        # # colored_branch = colored_pcd.select_by_index(lowd_idxs,)
        
        # draw(colored_leaves)
        # draw(colored_branch)
        # # breakpoint()
        # leaf_colors = arr(colored_leaves.colors)
        # all_colors = arr(colored_branch.colors)
        # try:
        #     _, hsv_fulls = color_distribution(leaf_colors,all_colors,cutoff=.1,space='',sc_func =lambda sc: sc)
        # except Exception as e:
        #     breakpoint()
        #     print('err')
        # breakpoint()
        # # orig_detail = source_pcd.select_by_index(analouge_idxs)
        # # draw(orig_detail)
        # ds_pcd = pcd.uniform_down_sample(3)
        # draw(ds_pcd)
        # res = get_mesh(pcd,ds_pcd,pcd)
        # breakpoint()
    # jleaves = join_pcds(leaves)[0]
    # jbranches= join_pcds(branches)[0]
    # clusters = cluster_plus(jbranches,eps=.2, min_points=50,return_pcds=True,from_points=False)
    # test = read_point_cloud('branches_first5_pt3_25pct.pcd')
    # breakpoint()
    # write_point_cloud('leaves_first5_pt3_25pct.pcd',jleaves)
    breakpoint()
    dtrunk= read_point_cloud(f'{inputs}/skeletor_dtrunk.pcd')
    joined = read_point_cloud('branches_pt3_30pct.pcd')
    
    import rustworkx as rx
    conn_comps_list = []
    final_list = []
    for radius in [.06,.07,.08,.09]:
        degrees,cnts,line_set= get_pairs(query_pts=arr(joined.points),radius=radius, return_line_set=True)
        # draw(line_set)
        g = rx.PyGraph()
        g.add_nodes_from([idp for idp,_ in enumerate(arr(line_set.points))])
        g.add_edges_from_no_data([tuple(pair) for pair in arr(line_set.lines)])
        conn_comps = arr([x for x in sorted(rx.connected_components(g), key=len, reverse=True)])
        print(len(conn_comps))
        print([len(x) for x in conn_comps])

        large_comp_pt_ids = list(chain.from_iterable([x for x in conn_comps[0:500] if len(x)>100]))
        candidate_comps = list(chain.from_iterable([x for x in conn_comps[0:100]]))
        new_pts = arr(line_set.points)
        new_lines = g.edges(candidate_comps)
        new_line_set = o3d.geometry.LineSet()
        new_line_set.points = o3d.utility.Vector3dVector(new_pts)
        new_line_set.lines = o3d.utility.Vector2iVector(new_lines)
        draw(new_line_set)
        branches, _, analouge_idxs = get_neighbors_kdtree(trim, query_pts=new_pts,k=200, dist=.01)
        draw(branches)
        # from viz.color import segment_hues
        # hue_pcds,no_hue_pcds =segment_hues(branches,'test',draw_gif=False, save_gif=False)
        # res = plt.hist(arr(hsv_fulls[0])[:,0],nbins,facecolor=rgb)

        branch_nbrs, _, analouge_idxs = get_neighbors_kdtree(branches,dtrunk,k=500, dist=2)
        draw(branch_nbrs)
        obb = dtrunk.get_minimal_oriented_bounding_box()
        obb.color = (1, 0, 0)
        obb.scale(1.2,center=obb.get_center())
        obb.translate((0,0,-6),relative = True)
        draw([obb,branches])
        test = obb.get_point_indices_within_bounding_box( o3d.utility.Vector3dVector(arr(branches.points)) )
        nbrs = branches.select_by_index(test)
        draw(nbrs)
        test = cluster_plus(arr(nbrs.points),eps=.1)
        clusters_and_idxs = [(idc, t) for idc, t in enumerate(test)]
        tree_pcds = extend_seed_clusters(clusters_and_idxs, branches, 'skeletor4', k=50, max_distance=.1, cycles= 200, save_every = 50, draw_every = 50)
        draw(branch_nbrs)
        
        
        conn_comps_list.append(conn_comps)
        final_list.append(branches)
        
    # draw(new_line_set)
    # draw(branches)

    breakpoint()

   
    # draw(pcd)
    ds10_pcd =write_point_cloud(f'{inputs}/skeletor_clean_ds10.pcd')
    draw(ds10_pcd)
    print(f'down sampled_cloud: {ds10_pcd}')


    skel_res = extract_skeleton(ds10_pcd,debug=True)
    from canopy_metrics import draw_shift
    lowc,highc,_ = draw_shift(ds10_pcd, 'ds10', skel_res[1], draw_results = False)
    write_point_cloud(f'{inputs}/ds10_contracted.pcd',skel_res[0])
    save('ds10_sbs.pkl',skel_res[2])
    breakpoint()
    topo_res=extract_topology(skel_res[0])
    topology, topology_graph, skeleton, skeleton_points, skeleton_graph,rx_graph, mapping = topo_res

    plot_dist_dist(pcd)
    plot_dist_dist(ds10_pcd)

    ## Find and color/draw neighborhood
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=normals_radius*5)
    sp.translate(ds10_pcd.points[50000])
    sp.paint_uniform_color([1,0,0])
    pcd_tree = o3d.geometry.KDTreeFlann(ds10_pcd)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(ds10_pcd.points[90000], normals_radius*5)
    np.asarray(ds10_pcd.colors)[idx[1:], :] = [0, 1, 0]
    # draw(test)
    o3d.visualization.draw([ds10_pcd])
    o3d.visualization.draw([ds10_pcd,sp])
