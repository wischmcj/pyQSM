from collections import defaultdict
from copy import deepcopy
from glob import glob
from itertools import product
import multiprocessing
import re
import time
from typing import Any
from joblib import Parallel, delayed


from tree_isolation import extend_seed_clusters, pcds_from_extend_seed_file
from utils.io import np_to_o3d, save
            
import open3d as o3d
import numpy as np
from numpy import asarray as arr
import pyvista as pv
import pc_skeletor as pcs

from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from math_utils.fit import cluster_DBSCAN
from matplotlib import pyplot as plt
from glob import glob
import os
from scipy import spatial as sps

from geometry.surf_recon import get_mesh
from set_config import config, log
from geometry.general import center_and_rotate
from geometry.reconstruction import get_neighbors_kdtree
from geometry.skeletonize import extract_skeleton, extract_topology
from geometry.point_cloud_processing import (
    clean_cloud,
    join_pcds,
    join_pcd_files
)
from utils.lib_integration import get_pairs
from utils.io import load, load_line_set,save_line_set, create_table
from viz.ray_casting import project_pcd
from viz.viz_utils import color_continuous_map, draw, draw_view, rotating_compare_gif
from viz.color import (
    remove_color_pts, 
    get_green_surfaces,
    split_on_percentile,
    segment_hues,
    saturate_colors
)
import pyvista as pv
import pc_skeletor as pcs

import tensorflow as tf
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

from set_config import config, log
from geometry.general import center_and_rotate
from geometry.reconstruction import get_neighbors_kdtree
from geometry.skeletonize import extract_skeleton
from geometry.point_cloud_processing import (
    join_pcd_files
)
from utils.io import load
from viz.ray_casting import project_pcd
from viz.viz_utils import color_continuous_map, draw, rotating_compare_gif
from viz.plotting import plot_3d, histogram
from viz.color import (
    split_on_percentile,
    segment_hues,
    saturate_colors
)
from utils.algo import smooth_feature
from utils.io import convert_las
from geometry.surf_recon import meshfix
from sklearn.cluster import KMeans
from geometry.point_cloud_processing import cluster_plus
from cluster_joining import user_cluster
from general import list_if
from math_utils.interpolation import smooth_feature

from reconstruction import get_nbrs_voxel_grid, overlap_voxel_grid

color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,
               'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,
               'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,
               'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,
               'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,
               'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,
               'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])

def identify_epiphytes(file_content, save_gif=False, out_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/'):
    log.info('identifying epiphytes')
    # user_input = 65
    # while user_input is not None:
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    assert shift_one is not None # No shifts for this seed; Ensure you pass get_shifts=True to read_pcds_and_feats
    
    orig_colors = deepcopy(arr(clean_pcd.colors))
    c_mag = np.array([np.linalg.norm(x) for x in shift_one])
    
    highc_idxs, highc,lowc = split_on_percentile(clean_pcd,c_mag,65, color_on_percentile=True)
    clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
    lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
    highc = clean_pcd.select_by_index(highc_idxs, invert=False)

    # draw([clean_pcd])
    # draw([highc])
    # draw([lowc])
    high_shift = shift_one[highc_idxs]
    z_mag = np.array([x[2] for x in high_shift])
    leaves_idxs, leaves, epis = split_on_percentile(highc,z_mag,60, color_on_percentile=True)
    epis_colored  = highc.select_by_index(leaves_idxs, invert=True)
    # draw([lowc, leaves,epis])
    # draw([epis])
    project_components_in_clusters(pcd, clean_pcd, epis, leaves, lowc, seed)
    
    # The below is an attempt to make a good mesh for the epiphytes
    # id_mesh =False
    # if id_mesh:
    #     for pcd in [epis]:
    #         normals_radius = 0.005
    #         normals_nn = 10

    #         temp_pcd = deepcopy(pcd)
    #         temp_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))
    #         o3d.visualization.draw_geometries([temp_pcd], point_show_normal=True)
    #         tmesh = get_ball_mesh(temp_pcd,radii= [.15,.2,.25])
    #         draw([tmesh])
    #         del temp_pcd
    #         # from geometry.surf_recon1 import meshfix as meshfix1

    #         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))
    #         o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #         pcd.orient_normals_consistent_tangent_plane(100)
    #         bmesh = get_ball_mesh(pcd,radii= [.15,.2,.25])
    #         draw([bmesh])
    #         # breakpoint()
    #         new_mesh = tmesh + bmesh
    #         fmesh = meshfix(new_mesh) 
    #         mesh_file_dir = f'{out_path}/ray_casting/epi_mesh/'
    #         o3d.io.write_triangle_mesh(f'{mesh_file_dir}/{seed}_epis_mesh.ply', new_mesh)
    #         # breakpoint()    


def get_shift(file_content,
              initial_shift = True, contraction=3, attraction=.8, iters=1, 
              debug=False, vox=None, ds=None, use_scs = True):
    """
        Orig. run with contraction_factor 3, attraction .6, max contraction 2080 max attraction 
        Determines what files (e.g. information) is missing for the case passed and 
            calculates what is needed 
    """
    seed, src_file, clean_pcd, shift_one = file_content['seed'], file_content['src_file'], file_content['clean_pcd'], file_content['shift_one']
    trunk = None
    pcd_branch = None
    src_dir = os.path.dirname(src_file)
    target_dir = os.path.join(src_dir, 'shifts')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_base = os.path.join(target_dir, f'{seed}_')
    # skel_{str(contraction).replace('.','pt')}_{str(attraction).replace('.','pt')}_seed{seed}_vox{vox or 0}_ds{ds or 0}'
    log.info(f'getting shift for {seed}')
    if shift_one is None:
        log.warning(f'no shift found for {seed}')
        return None
    skel_res = extract_skeleton(clean_pcd, max_iter = iters, debug=True, cmag_save_file=file_base, contraction_factor=contraction, attraction_factor=attraction)
    breakpoint()
    cmag = np.array([np.linalg.norm(x) for x in skel_res[1]])
    color_continuous_map(clean_pcd, cmag)
    draw([clean_pcd])
    return skel_res


def get_skeleton(file_content,
              initial_shift = True, contraction=5, attraction=.5, iters=1, 
              debug=False, vox=None, ds=None, use_scs = True):
    """
        Orig. run with contraction_factor 3, attraction .6, max contraction 2080 max attraction 
        Determines what files (e.g. information) is missing for the case passed and 
            calculates what is needed 
    """
    seed, src_file, clean_pcd, shift_one = file_content['seed'], file_content['src_file'], file_content['clean_pcd'], file_content['shift_one']
    trunk = None
    pcd_branch = None
    src_dir = os.path.dirname(src_file)
    target_dir = os.path.join(src_dir, 'shifts')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_base = os.path.join(target_dir, f'{seed}_')
    # skel_{str(contraction).replace('.','pt')}_{str(attraction).replace('.','pt')}_seed{seed}_vox{vox or 0}_ds{ds or 0}'
    log.info(f'getting shift for {seed}')
    if shift_one is None:
        log.warning(f'no shift found for {seed}')
        return None
    skel_res = extract_skeleton(clean_pcd, max_iter = iters, debug=True, cmag_save_file=file_base, contraction_factor=contraction, attraction_factor=attraction)
    breakpoint()
    return skel_res


def contract(in_pcd,shift, invert=False):
    "Translates the points by the magnitude and direction indicated by the shift vector"
    pts=arr(in_pcd.points)
    if not invert:
        shifted=[(pt[0]-shift[0],pt[1]-shift[1],pt[2]-shift[2]) for pt, shift in zip(pts,shift)]
    else:
        shifted=[(pt[0]+shift[0],pt[1]+shift[1],pt[2]+shift[2]) for pt, shift in zip(pts,shift)]
    contracted = o3d.geometry.PointCloud()
    contracted.colors = o3d.utility.Vector3dVector(arr(in_pcd.colors))
    contracted.points = o3d.utility.Vector3dVector(shifted)
    return contracted

def get_downsample(file = None, pcd = None):
    if file: pcd = read_pcd(file)
    log.info('Down sampling pcd')
    from open3d.visualization import draw_geometries_with_editing as edit
    edit([pcd])
    breakpoint()
    voxed_down = pcd.voxel_down_sample(voxel_size=.15)
    clean_pcd = voxed_down.uniform_down_sample(5)
    breakpoint()
    print(f'clean version has {len(clean_pcd.points)} points')
    return clean_pcd

def contraction_analysis(file, pcd, shift):

    green = get_green_surfaces(pcd)
    not_green = get_green_surfaces(pcd,True)
    # draw(lowc_pcd)
    draw(green)
    draw(not_green)

    c_mag = np.array([np.linalg.norm(x) for x in shift])
    z_mag = np.array([x[2] for x in shift])

    highc_idxs, highc, lowc = color_on_percentile(pcd,c_mag,70)

    z_cutoff = np.percentile(z_mag,80)
    log.info(f'{z_cutoff=}')
    low_idxs = np.where(z_mag<=z_cutoff)[0]
    lowc = clean_pcd.select_by_index(low_idxs)
    ztrimmed_shift = shift[low_idxs]
    ztrimmed_cmag = c_mag[low_idxs]
    draw(lowc)
    highc_idxs, highc, lowc = color_on_percentile(lowc,c_mag,70)

    # color_continuous_map(test,c_mag)
    highc_idxs = np.where(c_mag>np.percentile(c_mag,70))[0]
    highc_pcd = test.select_by_index(highc_idxs)
    lowc_pcd = test.select_by_index(highc_idxs,invert=True)
    draw([lowc_pcd])
    draw([highc_pcd])

    breakpoint()
    
    lowc_detail = get_neighbors_kdtree(pcd,lowc_pcd)
    draw(lowc_detail)
    breakpoint()

def split_on_pct(pcd,pct,cmag=None, shift=None):
    if shift is not None:   
        cmag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs = np.where(cmag>np.percentile(cmag,pct))[0]
    lowc = pcd.select_by_index(highc_idxs, invert=True)
    highc = pcd.select_by_index(highc_idxs)
    return lowc,highc

def expand_features_to_orig(nbr_pcd, orig_pcd, nbr_data):
    # # get neighbors of comp_pcd in the extracted feat pcd
    dists, nbrs = get_neighbors_kdtree(nbr_pcd, orig_pcd, return_pcd=False)

    full_detail_feats = defaultdict(list)
    full_detail_feats['points'] = orig_pcd.points
    # For each list of neighbors, get the average value of each feature in all_data and add it to full_detail_feats
    feat_names = [k for k in nbr_data.keys() if k not in ['points','colors', 'labels']]
    final_data =[]
    nbrs = [np.array([x for x in nbr_list  if x< len(orig_pcd.points)]) for nbr_list in nbrs]
    nbrs = [nbr_list if len(nbr_list) > 0 else np.array([0]) for nbr_list in nbrs]
    for nbr_list in tqdm(nbrs):
        nbr_vals = np.array([np.mean(nbr_data[feat_name][nbr_list]) for feat_name in feat_names])
        final_data.append(nbr_vals)
    final_data = np.array(final_data)
    full_detail_feats['features'] = final_data
    return full_detail_feats

def width_at_height(file_content, save_gif=False, height=1.37, tolerance=0.1, axis=2):
    """
    Calculate the width of a point cloud at a given height above ground.
    """
    log.info('identifying epiphytes')
    # user_input = 65
    # while user_input is not None:
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    logdir = "/media/penguaman/writable/lidar_sync/py_qsm/tensor_board/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    params = [(lambda sc: sc + (1-sc)/3, 1, '33 inc, 1x'), 
                (lambda sc: sc + (1-sc)/2, 1, '50 inc, 1x'),
                (lambda sc: sc + (1-sc)/3, 1.5, '33 inc, 1.5x'), 
                (lambda sc: sc + (1-sc)/2, 1.5, '50 inc, 1.5x'),
                (lambda sc: sc + (1-sc)/3, .5, '33 inc, .5x'), 
                (lambda sc: sc + (1-sc)/2, .5, '50 inc, .5x')
                ]

    with writer.as_default():
        for sat_func, sat_cutoff, case_name in params:
            sat_pcd, sat_orig_colors = saturate_colors(pcd, cutoff=sat_cutoff, sc_func=sat_func)
            step+=1
            summary.add_3d('sat_test', to_dict_batch([sat_pcd]), step=step, logdir=logdir)
    
    import numpy as np
    height = 2.8
    # Get a 'slice' of the pointcloud at the given height
    pts = np.asarray(clean_pcd.points)
    # Find ground level (minimum in z/axis)
    ground = np.min(pts[:, axis])
    print(f'{ground=},{height=},{tolerance=},{axis=}')
    # Calculate slice bounds
    z_min = ground + height - tolerance
    z_max = ground + height + tolerance
    # Get indices of points within slice
    idx = np.where((pts[:, axis] >= z_min) & (pts[:, axis] <= z_max))[0]
    slice_pts = pts[idx]
    plane_pts = slice_pts[:, :2]
    # Viz. the slice against the original pointcloud
    new_pcd = clean_pcd.select_by_index(idx)
    _, ind = new_pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=.95)
    new_pcd = new_pcd.select_by_index(ind)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    center = new_pcd.get_center()
    coord_axis.translate(center, relative=False)
    vis.add_geometry(coord_axis)
    vis.add_geometry(clean_pcd)
    vis.add_geometry(new_pcd)
    vis.run()
    vis.destroy_window()
    draw([new_pcd]) 

    breakpoint()

    # Collect metrics to inform choice of width
    bounds = new_pcd.get_max_bound() - new_pcd.get_min_bound()
    # Calculate all pairwise distances; width is the maximum
    from scipy.spatial.distance import pdist
    plane_pts = arr(new_pcd.points)[:,:2]
    dists = pdist(plane_pts)
    p90 = np.percentile(dists, 90)
    p95 = np.percentile(dists, 95)
    dists = np.sort(dists)
    # histogram([x for x in dists if x >=np.percentile(dists, 70)], 'width_dists')
    max_width = dists.max() if len(dists) > 0 else 0.0
    median = np.median(dists) if len(dists) > 0 else 0.0
    print(f'{max_width=}')
    print(f'{p95=}')
    print(f'{p90=}')
    print(f'{median=}')
    print(f'{bounds=}')
    # print(f'{bb_bounds=}')
    width = p95
    user_input = input(f'width (currently {width})?')
    if user_input.isdigit():
        width = float(user_input)
    return {'seed':seed, 'width':width, 'bounds':bounds}

def project_in_slices(pcd,seed, name='', off_screen = True,alpha=70,target_dir='data/projection'):
    pcd = pcd.uniform_down_sample(5)
    points=arr(pcd.points)
    z_vals = np.array([x[2] for x in points])
    z_vals_sorted = np.sort(z_vals)
    # Break 'points' into chunks by z value percentile (slices)
    slices = {}
    percentiles = [0, 20, 40, 60, 80, 100]
    # percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    z_percentile_edges = np.percentile(z_vals, percentiles)
    for i in range(len(percentiles)-1):
        z_lo = z_percentile_edges[i]
        z_hi = z_percentile_edges[i+1]
        # get indices in this z interval
        in_slice = np.where((z_vals >= z_lo) & (z_vals < z_hi))[0] if i < len(percentiles)-2 else np.where((z_vals >= z_lo) & (z_vals <= z_hi))[0]
        slices[f'slice_{percentiles[i]}_{percentiles[i+1]}'] = points[in_slice]

    metrics = {}
    for slice_name, slice_points in slices.items():
        mesh = project_pcd(pts=slice_points, alpha=alpha, plot=True, seed=seed, name=name, sub_name=slice_name, off_screen=off_screen, screen_shots=[[-10,0,0]], 
        target_dir=target_dir)
        # geo = mesh.extract_geometry()
        metrics[slice_name] ={'mesh': mesh, 'mesh_area': mesh.area }
    metrics['total_area'] = np.sum([x['mesh_area'] for x in metrics.values()])
    log.info(f'{name} total area: {metrics["total_area"]}')
    return metrics

def project_components_in_slices(pcd, clean_pcd, epis, leaves, wood ,seed, name='', off_screen = True, target_dir='data/projection'):
    metrics={}
    # metrics['epis'] = project_in_slices(epis,seed, name='epis', off_screen=off_screen)
    metrics['leaves'] = project_in_slices(leaves,seed, name='leaves', off_screen=off_screen, target_dir=target_dir)
    metrics['wood'] = project_in_slices(wood,seed, name='wood', off_screen=off_screen, target_dir=target_dir)
    
    fin_metrics = {}
    total_area = 0
    for metric_name, metric_dict in metrics.items():
        print(f'{metric_name=}')
        fin_metrics[metric_name] = metric_dict.pop('total_area')
        print(fin_metrics[metric_name])
        total_area += fin_metrics[metric_name]
        fin_metrics[f'{metric_name}_slices'] = [x['mesh_area'] for x in metric_dict.values()]
    fin_metrics['total_area'] = total_area

    mesh = project_pcd(pts=arr(clean_pcd.points), plot=False, seed=seed, name='whole', off_screen=off_screen, target_dir=target_dir)
    fin_metrics['whole'] = mesh.area
    print(f'{fin_metrics["whole"]=}')
    # mesh = project_pcd(pts=arr(wood.points), plot=False, seed=seed, name='wood_singular', off_screen=off_screen, target_dir=target_dir)
    # fin_metrics['wood_singular'] = mesh.area

    import pickle
    with open(f'/media/penguaman/data/kevin_holden/projection/slice_metrics_{seed}.pkl', 'wb') as f: 
        pickle.dump(fin_metrics, f)
    log.info(f'{seed}, {fin_metrics=}')

def project_components_in_clusters(in_pcd, clean_pcd, epis, leaves, wood ,seed, name='', off_screen = True,
                                    voxel_size=25, eps=120, min_points=30, target_dir='data/projection'):
    metrics=defaultdict(dict)
    from geometry.point_cloud_processing import cluster_plus
    import pickle
    for case in [
                #(epis, 'epi_clusters'), 
                (leaves, 'leaf_clusters'), 
                (wood, 'wood_clusters')]:
        case_pcd, case_name = case
        case_pcd = case_pcd.voxel_down_sample(voxel_size)
        # draw([case_pcd])
        # case_pcd,_ = case_pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=.95)
        # draw([case_pcd])
        # breakpoint()
        print(f'clustering {case_name}')
        # label_to_cluster_orig, eps, min_points = user_cluster(case_pcd, return_pcds=True)
        # label_to_cluster =  cluster_plus(np.array(case_pcd.points), eps=eps, min_points=min_points, return_pcds=False)
        features = np.array(case_pcd.points)[:,:3]
        kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(features)
        unique_vals, counts = np.unique(kmeans.labels_, return_counts=True)
        log.info(f'{unique_vals=} {counts=}')
        cluster_idxs = [np.where(kmeans.labels_==val)[0] for val in unique_vals]
        label_to_cluster = [case_pcd.select_by_index(idxs) for idxs in cluster_idxs]
        label_to_cluster_orig = {val: case_pcd.select_by_index(idxs) for val, idxs in zip(unique_vals, cluster_idxs)}
        
        num_good_clusters = len(label_to_cluster_orig)
       
        total_area=0
        label_to_cluster = label_to_cluster_orig
        for cluster_idx, cluster_pcd in tqdm(label_to_cluster.items()):
            # if cluster_idx <= 0:
            #     breakpoint()
            #     continue
            print(f'{len(cluster_pcd.points)}')
            print(f'projecting cluster {cluster_idx}')
            clean_cluster_pcd = cluster_pcd.uniform_down_sample(4)
            print(f'{len(clean_cluster_pcd.points)} after downsampling')
            alpha=50
            mesh = project_pcd(pts=np.array(clean_cluster_pcd.points), alpha=alpha, plot=True, seed=seed, name=case_name, sub_name=f'{cluster_idx}', off_screen=True, screen_shots=[[-10,0,0]], target_dir=target_dir)
            print(f'{alpha=}, {mesh.area=}')
            metrics[case_name][f'{cluster_idx}'] ={'mesh_area': mesh.area }
            total_area += mesh.area
        print(f'summing cluster areas for {case_name}')
        metrics[case_name] = total_area
        print(f'{case_name} total area: {total_area}')
    

    import pickle
    # f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/projected_areas_clusters/all_metrics_split5.pkl'
    with open(f'/media/penguaman/data/kevin_holden/projection/metrics_{seed}.pkl', 'wb') as f: 
        pickle.dump(metrics, f)
    log.info(f'{seed}, {metrics=}')
    return {seed: metrics}

def assemble_nbrs(requested_dirs:list[str]=[],
                    nbr_file_pattern='*.npz'):
    """
        Used to use nbr files from get_nbrs_voxel_grid to determine
           which points in the src pcd tiles files are not assigned to a tree
    """
    if len(requested_dirs)>0:
        nbr_dirs = requested_dirs
    else:
        nbr_dirs = glob('/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/color_int_tree_nbrs/SKIO-RaffaiEtAlcolor_int*')
    nbr_lists = defaultdict(list)
    log.info(f'{nbr_dirs=}')
    for nbr_dir in tqdm(nbr_dirs):
        if '.npz' not in nbr_dir:
            # Construct existing file name
            if nbr_dir[-1] == '/':
                nbr_dir = nbr_dir[:-1]
            existing_nbr_file = f'{nbr_dir}_all_tree_nbrs.npz'
            all_tree_nbrs=[]
            if os.path.exists(existing_nbr_file):
                log.info(f'{existing_nbr_file=} already exists, loading it')
                # all_tree_nbrs = list(np.load(f'{nbr_dir}_all_tree_nbrs.npz')['nbrs'])
            log.info(f'{existing_nbr_file=}')

            nbr_files_path = f'{nbr_dir}/{nbr_file_pattern}'
            log.info(f'getting nbr files form {nbr_files_path=}')
            nbr_files = glob(nbr_files_path)
            log.info(f'{nbr_files=}')

            for nbr_file in nbr_files:
                print(f'processing nbr file {os.path.basename(nbr_file)}')
                # nbrs = np.load(nbr_file)['nbrs']
                # all_tree_nbrs.extend(nbrs)

            save_file = f'{nbr_dir}_all_tree_nbrs_fin.npz'
            log.info(f'saving all tree nbrs to {save_file}')
            # np.savez_compressed(f'{nbr_dir}_all_tree_nbrs_fin.npz', nbrs=all_tree_nbrs)
    return nbr_lists

def get_remaining_pcds():
    from cluster_joining import user_cluster
    nbr_files = glob('/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/color_int_tree_nbrs/SKIO-RaffaiEtAlcolor_int*_all_tree_nbrs_fin.npz')
    for nbr_file in nbr_files:
        nbr_name = nbr_file.split('/')[-1].replace('.npz','')
        print(f'processing nbr file {nbr_name=}')
        nbrs = np.load(nbr_file)['nbrs']
        src_file = f'{nbr_file.replace('/color_int_tree_nbrs','').replace('_all_tree_nbrs_fin','')}'
        src_data= np.load(src_file)

        nbr_mask = np.ones_like(src_data['intensity'], dtype=bool)
        nbr_mask[nbrs] = False
        to_write = {}
        for file_name in src_data.files:
            to_write[file_name] = src_data[file_name][nbr_mask]
        num_pts_remaining = len(to_write['points'])
        print(f'{num_pts_remaining} points remaining')
        # np.savez_compressed(f'{nbr_file.replace('_all_tree_nbrs_fin.npz','')}_remaining.npz', **to_write)
        print(f'creating pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(to_write['points'][::10])
        pcd.colors = o3d.utility.Vector3dVector(to_write['colors'][::10]/255)
        labels, eps, min_points = user_cluster(pcd, src_pcd=None)
        # src_pcd = o3d.geometry.PointCloud()
        # src_pcd.points = o3d.utility.Vector3dVector(src_data['points'][::20])
        # src_pcd.paint_uniform_color([1,0,0])
        draw([pcd])
        # o3d.io.write_point_cloud(f'{nbr_file.replace('_all_tree_nbrs_fin.npz','')}_remaining.pcd', pcd)
        # 
    return

def script_for_extracting_epis_and_labeling_orig_detail():

    detail_ext_dir = f'{base_dir}/ext_detail/'
    shift_dir = f'{base_dir}/pepi_shift/'
    addnl_skel_dir = f'{base_dir}/results/skio/skels2/'
    loop_over_files(identify_epiphytes,
                    kwargs={'save_gif':True},
                    detail_ext_dir=detail_ext_dir,
                    shift_dir=shift_dir,
                    )

    # Joining extracted epiphytes
    base_dir = f'/media/penguaman/backupSpace/lidar_sync/pyqsm/epis/'
    _ =join_pcd_files(base_dir, pattern = '*epis.pcd')

    # Getting orig detail for epis and not epis 
    src_pcd = read_pcd(f'{base_dir}/joined_epis.pcd')
    get_and_label_neighbors(src_pcd, base_dir, 'epi', non_nbr_label='wood')

    # Joining extracted epiphytes
    _ = join_pcd_files(f'{base_dir}/detail/', pattern = '*nbrs.pcd')
    # breakpoint()

    _ = join_pcd_files(f'{base_dir}/not_epis/',
                        pattern = '*non_matched.pcd',
                        voxel_size = .05,
                        write_to_file = False)
    # breakpoint()

def get_smoothed_features(all_data, 
                plot_labels=['intensity', ''], 
                save_file=None,
                step_through = True,
                file_name=''):
    points = all_data['points']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    smoothed_data = defaultdict(list)
    smoothed_data_file = save_file.replace('.npz', '_smoothed.npz')
    detail_data_file = save_file.replace('.npz', '_detail_feats.npz')

    # Check for existing smoothed data
    if os.path.exists(smoothed_data_file):
        log.info(f'{smoothed_data_file=} already exists, loading from file')
        smoothed_data = np.load(smoothed_data_file)

    try:
        datum_name = 'intensity'
        datum = smoothed_data.get(datum_name)

        detail_file_name = save_file.replace('with_all_feats/','').replace('int_color_data', 'full_ext').replace('.npz', f'_orig_detail.pcd')
        detail_pcd = o3d.io.read_point_cloud(detail_file_name)

        detail_data_file = save_file.replace('.npz', f'_{datum_name}_detail_feats.npz')

        if os.path.exists(detail_data_file):
            final_data = np.load(detail_data_file)['intensity']
            new_detail_pcd, _ = color_continuous_map(detail_pcd, final_data)
            draw_view(new_detail_pcd, file_name)
            histogram(final_data, datum_name)
            np.sort(final_data)
            # breakpoint()
            from scipy.stats import mode
            datum_mode = mode(final_data)[0]
            idxs = np.where(np.logical_and(final_data != datum_mode, final_data >= -1750))[0]
            new_pcd = detail_pcd.select_by_index(idxs)
            draw_view(new_pcd, file_name)
        else:
            dists, nbrs = get_neighbors_kdtree(pcd, detail_pcd, return_pcd=False)
            num_pts = len(pcd.points)
            nbrs = [np.array([x for x in nbr_list  if x< num_pts]) for nbr_list in nbrs]
            nbrs = [nbr_list if len(nbr_list) > 0 else np.array([0]) for nbr_list in nbrs]
            final_data = []
            for nbr_list in tqdm(nbrs): final_data.append(np.array(np.mean(datum[nbr_list])))
            final_data = np.array(final_data)
            np.savez_compressed(detail_data_file, intensity = final_data)

    except Exception as e:
        log.info(f'error {e} when getting detail features')

def compare_dirs(dir1, dir2, 
                file_pat1 ='', file_pat2 ='',
                key_pat1 ='', key_pat2 =''):
    if not file_pat2: file_pat2 = file_pat1
    if not key_pat2: key_pat2 = key_pat1
    files1 = glob(file_pat1, root_dir=dir1)
    files2 = glob(file_pat2, root_dir=dir2)
    keys2 = [re.match(re.compile(key_pat2), file2).groups(1)[0] for file2 in files2]
    in_one_not_two_files = []
    in_one_not_two_keys = []
    for file1 in files1:
        key1 = re.match(re.compile(key_pat1), file1).groups(1)[0]
        if key1 not in keys2:
            in_one_not_two_files.append(file1)
            in_one_not_two_keys.append(key1)

    return in_one_not_two_files, in_one_not_two_keys

def crop_and_remove():
    from open3d.visualization import draw_geometries_with_editing as edit
    
    files = glob(f'fave*.ply')
    # pcds = [read_pcd(file) for file in files]
    out_pcd = o3d.geometry.PointCloud()
    for file in files:
        pcd = read_pcd(file)
        o3d.io.write_point_cloud(file.replace('.ply', '.pcd'), pcd)
    
    files = glob('/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/EpiphytusTV4color_int_[0-9]_treeiso.npz')
    all_points = []
    all_colors = []
    all_intensity = []
    for file in files:
        print(f'{file=}')
        to_filter_data = np.load(file).append(to_filter_data['points'])
        all_colors.append(to_filter_data['colors'])
        all_intensity.append(to_filter_data['intensity'][:, np.newaxis])
  
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    all_intensity = np.vstack(all_intensity)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    draw([pcd])
    breakpoint()
    new_data = { 'points': to_filter_data['points'], 'colors': to_filter_data['colors'], 'intensity': to_filter_data['intensity'] }
    np.savez_compressed('/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/EpiphytusTV4color_int_treeiso.npz', **new_data)
    breakpoint()
    # comp_pcd = np_to_o3d('/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/EpiphytusTV4color_int_0.npz')
    # comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mask_pcd,voxel_size=1)
    # queries = np.array(comp_pcd.points)
    # in_occupied_voxel_mask= mask_voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    # breakpoint()



if __name__ == "__main__":
        # files = glob('/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/to_get_detail/*tl_1_custom.npz')
    # for idf, file in enumerate(files):
    #     if idf>1:
    #         pcd_data = np.load(file)
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(pcd_data['points'])
    #         seed = re.match(re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*'), file).groups(1)[0]
    #         print(f'{seed=}')
    #         get_nbrs_voxel_grid(comp_pcd = pcd, 
    #                         comp_file_name = seed,
    #                         tile_dir = '/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/',
    #                         tile_pattern = 'SKIO-RaffaiEtAlcolor_int_*.npz', invert=False,
    #                         out_folder='/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/detail')
                    
    # base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/'
    # remaining_files, remaining_keys = compare_dirs(base_dir + 'detail/', base_dir + 'shifts/',
    #                             file_pat1 = 'skio_*_tl_*.npz', file_pat2 = 'skio_*_tl_*_shift.pkl',
    #                             key_pat1 = '(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*', key_pat2 = '(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
    # remaining_keys = list(set(remaining_keys) - set(skip_seeds))
    # print(f'{remaining_keys=}')
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining'
    ######## 138, 193 partial trees
    # loop_over_files(
    #                 get_shift,
    #                 requested_seeds=requested_seeds,
    #                 parallel = False,
    #                 base_dir=base_dir,
    #                 data_file_config={ 
    #                     'src': {
    #                             'folder': 'detail/',
    #                             'file_pattern': f'*.npz',
    #                             'load_func': read_and_downsample, 
    #                         },
    #                     # 'shift_one': {
    #                     #         'folder': 'shifts',
    #                     #         'file_pattern': 'skio_*_tl_*_shift.pkl',
    #                     #         'load_func': lambda x, root_dir: load(x,root_dir)[0], 
    #                     #         'kwargs': {'root_dir': '/'},
    #                     #     },
    #                 },
    #                 seed_pat = re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
    #                 )
    loop_over_files(
                    identify_epiphytes,
                    requested_seeds=requested_seeds,
                    parallel = False,
                    base_dir=base_dir,
                    data_file_config={ 
                        'src': {
                                'folder': 'detail/',
                                'file_pattern': f'*.npz',
                                'load_func': read_and_downsample, 
                            },
                        'shift_one': {
                                'folder': 'shifts',
                                'file_pattern': 'skio_*_tl_*_shift.pkl',
                                'load_func': lambda x, root_dir: load(x,root_dir)[0], 
                                'kwargs': {'root_dir': '/'},
                            },
                    },
                    seed_pat = re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
                    )
    # breakpoint()
    