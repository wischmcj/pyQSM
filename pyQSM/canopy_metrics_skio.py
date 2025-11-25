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

color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,
               'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,
               'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,
               'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,
               'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,
               'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,
               'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])

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
     

def segment_feature(file_content, 
                    feature_name='intensity',):
    seed, src_file, pcd = file_content['seed'], file_content['src_file'], file_content['src']
    feature = file_content.get(feature_name, None)
    points = arr(pcd.points)

    # if feature_name=='all':
    smoothed_data_file = src_file.replace('detail','detail_all_smoothed')
    # else:
        # smoothed_data_file = src_file.replace('detail','detail_smoothed')
    
    # Check for existing smoothed data
    # if os.path.exists(smoothed_data_file):
    #     log.info(f'{smoothed_data_file=} already exists, loading from file...')
    #     try:
    #         smoothed_data = np.load(smoothed_data_file)
    #         log.info(f'{smoothed_data.files}')
    #         smoothed_features = {feature_name: smoothed_data[feature_name]}
    #     except Exception as e:
    #         log.error(f'error loading smoothed data: {e}')
    #         # breakpoint()
    #         smoothed_data = None
    # else:
    # if 1==1:
    #     if feature_name=='all':
    # smoothed_features = {}
    feats = get_nbrs_voxel_grid(pcd, seed, tile_dir = '/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/',  tile_pattern = 'SKIO_voxpt05_all*.npz')
    # feat_points = feats['points']
    # for feat in feats.keys():
    #     if feat not in ['intensity', 'color', 'points']:
    #         smoothed_features[feat] = smooth_feature(feat_points, feats[feat], query_pts=points)
    #                 # breakpoint()
    #     else:        
    #         smoothed_features = {feature_name: smooth_feature(points, feature, query_pts=points)}
    #     np.savez_compressed(smoothed_data_file, **smoothed_features)


    # for feat, smoothed_feat in smoothed_features.items():
    #     color_continuous_map(pcd, smoothed_feat)
    #     draw([pcd])
    #     histogram(smoothed_feat, feat)

    #     filter_val = -1750
    #     user_input = input(f"Filter value? ({filter_val}): ").strip().lower()
    #     if user_input.isdigit():
    #         user_input = int(user_input)
    #     else:
    #         user_input = filter_val
    #     try:
    #         filter_idxs = np.where(smoothed_feat > user_input)[0]
    #         non_filter_idxs = np.where(smoothed_feat <= user_input)[0]
    #         non_epi_pcd = pcd.select_by_index(filter_idxs)
    #         draw([non_epi_pcd])
    #         epi_pcd = pcd.select_by_index(filter_idxs, invert=True)
    #         draw([epi_pcd])
    #         filtered_feature = smoothed_feat[non_filter_idxs]
    #         histogram(filtered_feature, feature_name)
    #     except Exception as e:
    #         # breakpoint()
    #         log.info(f'error saving smoothed data: {e}')

        # # breakpoint()

    # return smoothed_features


def crop_with_box(pcd, center=None, extent=None):
    bb = pcd.get_oriented_bounding_box()
    print(bb)
    if center is not None:
        bb.center=center
    if extent is not None:
        bb.extent=extent
    print(bb)
    new_pt_ids = bb.get_point_indices_within_bounding_box( o3d.utility.Vector3dVector(arr(pcd.points)))
    new_pcd = pcd.select_by_index(new_pt_ids)
    draw([new_pcd])
    return new_pcd

def width_at_height(file_content, save_gif=False, height=1.37, tolerance=0.1, axis=2):
    """
    Calculate the width of a point cloud at a given height above ground.
    """
    log.info('identifying epiphytes')
    # user_input = 65
    # while user_input is not None:
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    
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

    # crop_pcd = crop_with_box(new_pcd,center=np.array([89.47, 347.5, 2.17695]),extent = np.array([1.300391, 1.7, 1.18837]))
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
        # Recluster points in cluster 0 using a larger eps
        # if -1 in label_to_cluster_orig:
        #     cluster0_pcd = label_to_cluster_orig[-1]
        #     cluster0_points = np.array(cluster0_pcd.points)
        #     if len(cluster0_points) > 0:
        #         # Choose a larger eps (e.g. double the original value)
        #         recluster_eps = eps * 3
        #         min_points = min_points*2
        #         new_label_to_cluster = cluster_plus(cluster0_points, eps=recluster_eps, min_points=min_points, return_pcds=False)
        #         unique_new_labels = np.unique([k for k,v in new_label_to_cluster.items()])
        #         reclustered_clusters = {ulabel: cluster0_pcd.select_by_index(new_label_to_cluster[ulabel]) for ulabel in unique_new_labels}
        #         print(f'Reclustered cluster 0 with eps={recluster_eps}: {len(reclustered_clusters)} clusters')
        #         for ulabel, cluster in reclustered_clusters.items():
        #             if ulabel == -1:
        #                 continue
        #             label_to_cluster_orig[ulabel+num_good_clusters+1] = cluster
        # label_to_cluster_orig.pop(-1)

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


def get_files_by_seed(data_file_config, 
                        base_dir,
                        # key_pattern = re.compile('.*seed([0-9]{1,3}).*')
                        key_pattern = re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
                        ):
    seed_to_files = defaultdict[Any, dict](dict)
    for file_type, file_info in data_file_config.items():
        # Get all matchig files
        folder = f'{base_dir}/{file_info["folder"]}'
        file_pattern = file_info['file_pattern']
        files = glob(file_pattern,root_dir=folder)
        # organize files by seed
        for file in files:
            file_key = re.match(key_pattern,file)
            if file_key is None:
                log.info(f'no seed found in seed_to_content: {file}. Ignoring file...')
                continue
            if len(file_key.groups(1)) > 0:
                file_key = file_key.groups(1)[0]
            else:
                file_key = file_key[0]
            seed_to_files[file_key][file_type] = f'{base_dir}/{file_info["folder"]}/{file}'
    return seed_to_files


def np_feature(feature_name):
    def my_feature_func(npz_file):
        npz_data = np.load(npz_file)
        return npz_data[feature_name]
    return my_feature_func

def read_and_downsample(file_path, **kwargs):
    if file_path.endswith('.npz'):
        pcd = np_to_o3d(file_path)
    elif file_path.endswith('.las'):
        pcd = convert_las(file_path)
    else:
        pcd = read_pcd(file_path, **kwargs)
    clean_pcd = get_downsample(pcd=pcd, **kwargs)
    return pcd, clean_pcd

def get_data_from_config(seed_file_info, data_file_config):
    seed_to_content = defaultdict(dict)
    for file_type, file_path in seed_file_info.items():
        load_func = data_file_config[file_type]['load_func']
        load_kwargs = data_file_config[file_type].get('kwargs',{})
        if load_func == read_and_downsample:
            seed_to_content[file_type], seed_to_content['clean_pcd'] = load_func(file_path, **load_kwargs)
        else:
            seed_to_content[file_type] = load_func(file_path, **load_kwargs)
        seed_to_content[f'{file_type}_file'] = file_path
    return seed_to_content

def get_seed_id_from_file(file, seed_pat = re.compile('.*seed([0-9]{1,3}).*')):
    return re.match(seed_pat,file).groups(1)[0]

def loop_over_files(func,args = [], kwargs =[],
                    requested_seeds=[],
                    skip_seeds = [],
                    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm',
                    detail_ext_folder = 'ext_detail',
                    data_file_config:dict = { 
                        'src': {
                                'folder': 'ext_detail',
                                'file_pattern': '*orig_detail.pcd',
                                'load_func': read_pcd, # custom, for pickles
                            },
                        'shift_one': {
                                'folder': 'pepi_shift',
                                'file_pattern': '*shift*',
                                'load_func': load, # custom, for pickles
                                'kwargs': {'root_dir': '/'},
                            },
                        'smoothed_feat_data': {
                                'folder': 'ext_detail/with_all_feats/',
                                'file_pattern': 'int_color_data_*_smoothed.npz',  
                                'load_func': np.load,
                            },
                        'detail_feat_data': {
                                'folder': 'ext_detail/with_all_feats/',
                                'file_pattern': 'int_color_data_*_detail_feats.npz',  
                                'load_func': np.load,
                            },
                    },
                    seed_pat = re.compile('.*seed([0-9]{1,3}).*'),
                    parallel = True,
                    ):
    # reads in the files from the indicated directories
    files_by_seed = get_files_by_seed(data_file_config, base_dir, key_pattern=seed_pat)
    
    files_by_seed = {seed:finfo for seed, finfo in files_by_seed.items() 
                            if ((requested_seeds==[] or seed in requested_seeds)
                              and seed not in skip_seeds)}

    if args ==[]: args = [None]*len(kwargs)
    inputs = [(arg,kwarg) for arg,kwarg in zip(list_if(args),list_if(kwargs))]
    if inputs == []:
        to_run = product(files_by_seed.items(), [([''],{})])
    else:
        to_run =  product(files_by_seed.items(), inputs) 
    results = []
    errors = []
    if parallel:
        content_list = [get_data_from_config(seed_file_info, data_file_config) for seed, seed_file_info in files_by_seed.items()]
        to_call = product(content_list, inputs)
        results = Parallel(n_jobs=3)(delayed(func(content.update({'seed':seed}),*arg_tup,**kwarg_dict)) for (seed,content), (arg_tup, kwarg_dict) in to_call)
    else: 
        for (seed, seed_file_info), (arg_tup, kwarg_dict) in to_run:
            try:
                print(f'{seed=}')
                content = get_data_from_config(seed_file_info, data_file_config)
                content['seed'] = seed
                print(f'running function for {seed} done')
                result = func(content,*arg_tup,**kwarg_dict)
                results.append(result)
            except Exception as e:
                breakpoint()
                log.info(f'error {e} when processing seed {seed}')
                errors.append(seed)
    print(f'{errors=}')
    print(f'{results=}')


    # results = Parallel(n_jobs=3)(delayed(extract_skeleton(pcd, max_iter = 1, debug=False, cmag_save_file=f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/shifts/{seed}_', contraction_factor=1, attraction_factor=.6))(seed,pcd) for seed,pcd in to_run)
    
    
    # for seed, seed_file_info in files_by_seed.items():
    #     log.info(f'processing seed {seed}')
    #     # try:
    #     if 1==1:
    #         if not requested_pcds:
    #             content = get_data_from_config(seed_file_info, data_file_config)
    #             content['seed'] = seed
    #         else:
    #             content = requested_pcds
    #         if len(inputs) == 0:
    #             results = Parallel(n_jobs=7)(delayed(func)(content) for content in split)
    #             result  = func(content)
    #         for arg_tup, kwarg_dict in inputs:
    #             result  = func(content,*arg_tup,**kwarg_dict)
    #             results.append(result)
        # except Exception as e:
        #     log.info(f'error {e} when processing seed {seed}')
        #     continue

    log.info('loop over files done')


def get_features(file, step_through=True):
    all_nbrs= {}
    # Get files to add features to
    # base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/'
    # glob_pattern = f'{base_dir}/ext_detail/*orig_detail.pcd'
    # files = glob(glob_pattern)
    comp_file_name = file.split('/')[-1].replace('.pcd','').replace('full_ext_','').replace('_orig_detail','')
    log.info(f'{comp_file_name=}')
    comp_pcd = read_pcd(file)
    if step_through:
        draw_view(comp_pcd, comp_file_name)
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/ext_detail/with_all_feats/'
    save_file = f'{base_dir}/int_color_data_{comp_file_name}.npz'
    if os.path.exists(save_file):
        log.info(f'{save_file=} already exists, loading from file')
        all_data = np.load(save_file)
    else:
        get_nbrs_voxel_grid(comp_pcd, comp_file_name, tile_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/ext_detail/with_all_feats/',  tile_pattern = 'SKIO_voxpt05_all*.npz')

def overlap_voxel_grid(src_pts, comp_voxel_grid = None, source_pcd = None, invert = False):
    if comp_voxel_grid is None:
        comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source_pcd,voxel_size=0.2)

    log.info('querying voxel grid')
    queries = src_pts
    in_occupied_voxel_mask= comp_voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    num_in_occupied_voxel = np.sum(in_occupied_voxel_mask)
    log.info(f'{num_in_occupied_voxel} points in occupied voxels')
    if num_in_occupied_voxel == 0:
        return []
    if invert:
        not_in_occupied_voxel_mask = np.ones_like(in_occupied_voxel_mask, dtype=bool)
        not_in_occupied_voxel_mask[in_occupied_voxel_mask] = False
        uniques = np.where(not_in_occupied_voxel_mask)[0]
    # else:
    uniques = np.where(in_occupied_voxel_mask)[0]

    return uniques

def get_nbrs_voxel_grid(comp_pcd, 
                        comp_file_name,
                        tile_dir, 
                        tile_pattern,
                        invert=False,
                        out_folder='detail',
                        out_file_prefix='detail_feats',
                        ):
    log.info('creating voxel grid')
    comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(comp_pcd,voxel_size=0.1)
    log.info('voxel grid created')
    nbr_ids = defaultdict(list)
    all_data =  defaultdict(list)
    
    # dodraw = True
    files = glob(f'{tile_dir}/{tile_pattern}')
    pcd=None
    for file in files:
        if 'SKIO-RaffaiEtAlcolor_int_0' in file or 'SKIO-RaffaiEtAlcolor_int_1' in file or 'SKIO-RaffaiEtAlcolor_int_2' in file:
            log.info(f'skipping {file}')
            continue
        file_name = file.split('/')[-1].replace('.pcd','').replace('.npz','')
        log.info(f'processing {file_name}')
        if '.pcd' in file:
            pcd = read_pcd(file)
            data = {'points': np.array(pcd.points), 'colors': np.array(pcd.colors)}
            data_keys = data.keys()
        else:
            data = np.load(file)
            data_keys = data.files
            log.info(f'{data_keys=}')

        # if dodraw:
        #     pcd = to_o3d(data['points'])
        #     draw([pcd, comp_pcd])

        # Determine if the boundaries of pcd and comp_pcd intersect at all
        # Compute bounding boxes for both point clouds and check for intersection
        pcd_min = np.min(data['points'], axis=0)
        pcd_max = np.max(data['points'], axis=0)
        comp_min = np.min(np.asarray(comp_pcd.points), axis=0)
        comp_max = np.max(np.asarray(comp_pcd.points), axis=0)
        # Intersection exists if, on all axes, the max of the lower bounds <= min of the upper bounds
        intersect = np.all((pcd_max >= comp_min) & (comp_max >= pcd_min))
        log.info(f'Bounding box intersection: {intersect}')
        if not intersect:
            log.info("Bounding boxes do not intersect, skipping file.")
            continue
        else:
            log.info("Bounding boxes intersect, processing file.")
    
        nbr_dir = f'{tile_dir}/color_int_tree_nbrs/{file_name}'
        uniques = overlap_voxel_grid(data['points'], comp_voxel_grid, invert=invert)

        if not os.path.exists(nbr_dir):
            os.makedirs(nbr_dir)
        np.savez_compressed(f'{nbr_dir}/{out_file_prefix}_{comp_file_name}.npz', nbrs=uniques)
        log.info('filtering data to neighbors') 
        # smooth_intensity = smooth_feature(src_pts, data['intensity'])
        # all_data['smooth_intensity'].append(smooth_intensity)
        filtered_data = {zfile:data[zfile][uniques] for zfile in data_keys}
        for datum_name, datum in filtered_data.items():
            if len(datum.shape) == 1:
                all_data[datum_name].append(datum)#[:,np.newaxis])
            else:
                all_data[datum_name].append(datum)

    log.info('Saving all data')
    for datum_name, datum in all_data.items():
        if len(datum[0].shape) == 1:
            all_data[datum_name] = np.hstack(datum)
        else:
            all_data[datum_name] = np.vstack(datum)


    np.savez_compressed(f'{out_folder}/{comp_file_name}.npz', **all_data)
    # # breakpoint()
    # o3d.io.write_point_cloud(f'{base_dir}/int_color_data_{comp_file_name}.pcd', pcd)
    # # breakpoint()
    return all_data
    # log.info('Calculating smoothed features and plotting')
    # get_smoothed_features(all_data, save_file = save_file, step_through=step_through, file_name=comp_file_name)
     
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
    
    # for datum_name, datum in tqdm(all_data.items()):
    #     log.info(f'{datum_name=}')
    #     if datum_name == 'points' or datum_name == 'colors':
    #         continue
    #     log.info(f'{len(points)=} points and {len(datum)=} intensity added to feature pcd')

    #     smoothed_datum = smoothed_data.get(datum_name, None)
    #     if smoothed_datum is None or smoothed_datum.shape[0] != len(points):
    #         # breakpoint()
    #         smoothed_datum = smooth_feature(points, datum, pcd=pcd)
    #         smoothed_data[datum_name] = smoothed_datum

    #     pcd, _ = color_continuous_map(pcd, smoothed_datum)
    #     draw_view(pcd, file_name)
    #     histogram(smoothed_datum, datum_name)

    # if save_file is not None:
    #     np.savez_compressed(smoothed_data_file, **smoothed_data)

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


    # if save_file is not None:
    #     np.savez_compressed(smoothed_data_file, **smoothed_data)
    # if step_through:
    #     # breakpoint()
    #     plot_labels = ['planarity', 'intensity', 'linearity']
    #     try:
    #         plot_3d([all_data[x] for x in plot_labels], plot_labels)
    #     except Exception as e:
    #         log.info(f'error {e} when plotting {plot_labels}')
    # # return idxs

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
    breakpoint()

    # pcd =  read_pcd('/media/penguaman/backupSpace/lidar_sync/pyqsm/epip/ep_branch_mass_connector.ply') + read_pcd('ep_lower_trunk.ply')

    # mask_pcd = read_pcd('/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/EpiphytusTV4_treeiso.pcd')
    # mask_pcd.paint_uniform_color([1,0,0])
    # mask_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mask_pcd,voxel_size=1)
    
    files = glob('/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/EpiphytusTV4color_int_[0-9]_treeiso.npz')
    all_points = []
    all_colors = []
    all_intensity = []
    for file in files:
        print(f'{file=}')
        to_filter_data = np.load(file).append(to_filter_data['points'])
        all_colors.append(to_filter_data['colors'])
        all_intensity.append(to_filter_data['intensity'][:, np.newaxis])
        
        # queries = queries - np.array( [20.39992401, 148.73167031, -18.70106613])
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(queries)

        # in_occupied_voxel_mask= mask_voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
        # new_data = {file_name: to_filter_data[file_name][in_occupied_voxel_mask] for file_name in to_filter_data.files}
        
        # np.savez_compressed(file.replace('.npz', '_treeiso_mask.npz'), mask = in_occupied_voxel_mask)
        # np.savez_compressed(file.replace('.npz', '_treeiso.npz'), **new_data)

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


    # num_in_occupied_voxel = np.sum(in_occupied_voxel_mask)
    # log.info(f'{num_in_occupied_voxel} points in occupied voxels')
    # not_in_occupied_voxel_mask = np.ones_like(in_occupied_voxel_mask, dtype=bool)
    # not_in_occupied_voxel_mask[in_occupied_voxel_mask] = False
    # uniques = np.where(not_in_occupied_voxel_mask)[0]
    # new_pcd = comp_pcd.select_by_index(uniques)
    # draw(new_pcd)
    # breakpoint()
    # print('asdf')


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
    # breakpoint()
    # file_content = defaultdict(list)
    # file_content['seed'] = '33'
    # file_content['clean_pcd'] =pcd
    # # width_at_height(file_content, tolerance=.1)
    # # breakpoint()
    # # assemble_nbrs(nbr_file_pattern = 'detail_*')
    # get_remaining_pcds()
    # # breakpoint()
    # skip_seeds = ['skio_114_tl_0','skio_0_tl_246','skio_0_tl_246','skio_138_tl_0','skio_138_tl_0','skio_0_tl_240','skio_0_tl_240','skio_134_tl_0','skio_134_tl_0','skio_0_tl_223','skio_0_tl_223','skio_115_tl_0','skio_115_tl_0','skio_0_tl_69','skio_0_tl_69','skio_112_tl_0','skio_112_tl_0','skio_111_tl_0','skio_111_tl_0','skio_107_tl_0','skio_107_tl_0','skio_0_tl_188','skio_0_tl_188','skio_135_tl_0','skio_135_tl_0','skio_116_tl_493','skio_33_tl_0','skio_33_tl_0','skio_0_tl_490  ','skio_0_tl_490  ','skio_0_tl_15   ','skio_0_tl_15   ','skio_108_tl_0  ','skio_108_tl_0  ','skio_136_tl_9  ','skio_136_tl_9  ','skio_0_tl_6    ','skio_0_tl_6    ','skio_133_tl_68 ','skio_133_tl_68 ','skio_137_tl_34 ','skio_190_tl_220','skio_190_tl_220','skio_189_tl_236','skio_191_tl_236','skio_191_tl_236','skio_116_tl_493','skio_137_tl_34','skio_189_tl_236', 'skio_0_tl_306','skio_0_tl_157', 'skio_0_tl_377','skio_0_tl_49',
    #                 'skio_0_tl_167', 'skio_0_tl_158', 'skio_0_tl_23', 
    #                 'skio_0_tl_14', 'skio_0_tl_191', 'skio_0_tl_154', 'skio_0_tl_512',
    # #                 # The below may be worth rerunning shift calc without vox
    #                 'skio_33_tl_0',
    #                 'skio_107_tl_0',
    #                 'skio_108_tl_0',
    #                 'skio_111_tl_0',
    #                 'skio_112_tl_0',
    #                 'skio_114_tl_0',
    #                 'skio_116_tl_493',
    #                 'skio_133_tl_68',
    #                 'skio_134_tl_0',
    #                 'skio_135_tl_0',
    #                 'skio_136_tl_9',
    #                 'skio_137_tl_34',
    #                 'skio_189_tl_236',
    #                 'skio_190_tl_220',
    #                 'skio_191_tl_236',
    #                 'skio_0_tl_15',
    #                 'skio_0_tl_188',
    #                 'skio_0_tl_223',
    #                 'skio_0_tl_240',
    #                 'skio_0_tl_246',
    #                 'skio_0_tl_490',
    #                 'skio_0_tl_6',
    #                 'skio_0_tl_69',
    #                 ]
                     
    requested_seeds = [
        #'skio_133_tl_1',
                        'skio_193_tl_241'
                        ]
                    
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
    