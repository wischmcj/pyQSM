from collections import defaultdict
from copy import deepcopy
from glob import glob
from itertools import product
import multiprocessing
import re
import time
from joblib import Parallel, delayed

from tree_isolation import extend_seed_clusters, pcds_from_extend_seed_file
from utils.io import save
            
import open3d as o3d
import numpy as np
from numpy import asarray as arr
import pyvista as pv
import pc_skeletor as pcs

import tensorflow as tf
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
from open3d.io import read_point_cloud as read_pcd
from scipy import spatial as sps

from set_config import config, log
from geometry.general import center_and_rotate
from geometry.reconstruction import get_neighbors_kdtree
from geometry.skeletonize import extract_skeleton, extract_topology
from geometry.point_cloud_processing import (
    clean_cloud,
    get_ball_mesh,
    join_pcds,
    join_pcd_files
)
from utils.lib_integration import get_pairs
from utils.io import load, load_line_set,save_line_set, create_table
from viz.ray_casting import project_pcd
from viz.viz_utils import color_continuous_map, draw, rotating_compare_gif
from viz.plotting import plot_3d, histogram
from viz.color import (
    remove_color_pts, 
    get_green_surfaces,
    split_on_percentile,
    segment_hues,
    saturate_colors
)
from geometry.surf_recon import meshfix
from sklearn.cluster import KMeans

def list_if(x):
    if isinstance(x,list):
        return x
    else:
        return [x]

color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,
               'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,
               'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,
               'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,
               'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,
               'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,
               'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])

# subsample data with voxel_size
def voxelize(data, voxel_size):
    points = data[:, :3]
    points = np.round(points, 2)
    if data.shape[1] >= 4:
        other = data[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    bound = np.max(np.abs(points)) + 100
    min_bound, max_bound = np.array([-bound, -bound, -bound]), np.array([bound, bound, bound])
    downpcd, _, idx = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)

    if data.shape[1] >= 4:
        idx_keep = [item[0] for item in idx]
        other = other[idx_keep]
        data = np.hstack((np.asarray(downpcd.points), other))
    else:
        data = np.asarray(downpcd.points)
    
    return data, idx

def to_o3d(coords=None, colors=None, labels=None, las=None):
    if las is not None:
        las = np.asarray(las)
        coords = las[:, :3]
        if las.shape[1]>3:
            labels = las[:, 3]
        if las.shape[1]>4:
            colors = las[:, 4:7]       
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    elif labels is not None:
        pcd, _ = color_continuous_map(pcd,labels)
    # labels = labels.astype(np.int32)
    return pcd

def kmeans_feature(smoothed_feature, pcd= None):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(smoothed_feature[:,np.newaxis])
    unique_vals, counts = np.unique(kmeans.labels_, return_counts=True)
    log.info(f'{unique_vals=} {counts=}')
    cluster_idxs = [np.where(kmeans.labels_==val)[0] for val in unique_vals]
    cluster_features = [smoothed_feature[idxs] for idxs in cluster_idxs]
    if pcd is not None:
        for idxs in cluster_idxs: draw(pcd.select_by_index(idxs))
    for feats in cluster_features: histogram(feats)
    return cluster_idxs, cluster_features


def propogate_shift(pcd,clean_pcd, cvar):
    """
        extrapolating point shift to the more detailed pcd
    """
    highc_idxs = np.where(cvar>np.percentile(cvar,40))[0]
    highc = clean_pcd.select_by_index(highc_idxs)
    cvar_high = cvar[highc_idxs]
    _, nbrs = get_neighbors_kdtree(highc,pcd, k=50, return_pcd=False)
    nbrs = [[x for x in nbr_list if x<len(cvar_high)] for nbr_list in nbrs]
    proped_cvar = np.zeros(len(nbrs))
    for idnl, nbr_list in enumerate(nbrs): 
        if len(nbr_list)>0: 
            proped_cvar[idnl] = np.mean(arr(cvar_high[nbr_list]))
    # proped_c_mag = [np.mean(arr(cvar_high[nbr_list])) for nbr_list in nbrs]
    # color_continuous_map(pcd,arr(proped_c_mag))
    # draw(pcd)
    np.mean(arr(proped_cvar)[np.where(arr(proped_cvar)>0)[0]])
    highc_proped_idxs = np.where(arr(proped_cvar)>0)[0]
    test = pcd.select_by_index(highc_proped_idxs, invert = True)
    draw(test)
    return proped_cvar
    
def draw_shift(pcd,
                seed,
                shift,
                down_sample=False,draw_results=False, save_gif=False, out_path = None,
                on_frames=25, off_frames=25, addnl_frame_duration=.01, point_size=5,
                pctile_cutoff = 70):
    out_path = out_path or f'data/results/gif/'
    clean_pcd = pcd
    if down_sample: clean_pcd = get_downsample(pcd=clean_pcd, normalize=False)
    c_mag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs, highc,lowc = split_on_percentile(clean_pcd,c_mag,pctile_cutoff, color_on_percentile=True)
    # center_and_rotate(lowc)
    # center_and_rotate(highc)

    log.info('preping contraction/coloring')
    if draw_results:
        log.info('giffing')
        lowc.rotate(rot_90_x,center =  clean_pcd.get_center())
        lowc.rotate(rot_90_x,center =  clean_pcd.get_center())
        lowc.rotate(rot_90_x,center =  clean_pcd.get_center())
        highc.rotate(rot_90_x,center = clean_pcd.get_center())
        highc.rotate(rot_90_x,center = clean_pcd.get_center())
        highc.rotate(rot_90_x,center = clean_pcd.get_center())
        gif_kwargs = {'on_frames': on_frames, 'off_frames': off_frames, 
                        'addnl_frame_duration':addnl_frame_duration, 'point_size':point_size,
                        'save':save_gif, 'out_path':out_path, 'rot_center':clean_pcd.get_center(),
                         'sub_dir':f'{seed}_draw_shift' }
        rotating_compare_gif(highc,constant_pcd_in=lowc,**gif_kwargs)
    return highc,lowc,highc_idxs

    # # nowhite_custom =remove_color_pts(pcd, lambda x: sum(x)>2.3,invert=True)
    # draw(highc_detail

def get_shift(file_content,
              initial_shift = True, contraction=1, attraction=.6, iters=1, 
              debug=False, vox=None, ds=None, use_scs = True):
    """
        Orig. run with contraction_factor 3, attraction .6, max contraction 2080 max attraction 
        Determines what files (e.g. information) is missing for the case passed and 
            calculates what is needed 
    """
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    trunk = None
    pcd_branch = None
    file_base = f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/shifts/{seed}_'
    # skel_{str(contraction).replace('.','pt')}_{str(attraction).replace('.','pt')}_seed{seed}_vox{vox or 0}_ds{ds or 0}'
    log.info(f'getting shift for {seed}')
    # if initial_shift:
    #     cmag = np.array([np.linalg.norm(x) for x in shift_one])
    #     highc_idxs = np.where(cmag>np.percentile(cmag,70))[0]
    #     test = clean_pcd.select_by_index(highc_idxs, invert=True)
    # else:
    #     test = clean_pcd
    # if vox: test = test.voxel_down_sample(voxel_size=vox)
    # if ds: test = test.uniform_down_sample(ds)
    # if not use_scs:
    skel_res = extract_skeleton(clean_pcd, max_iter = iters, debug=False, cmag_save_file=file_base, contraction_factor=contraction, attraction_factor=attraction)
    # else:
    #     try:
            # lbc = pcs.LBC(point_cloud=test, filter_nb_neighbors = config['skeletonize']['n_neighbors'], max_iteration_steps= config['skeletonize']['max_iter'], debug = False, termination_ratio=config['skeletonize']['termination_ratio'], step_wise_contraction_amplification = config['skeletonize']['init_contraction'], max_contraction = config['skeletonize']['max_contraction'], max_attraction = config['skeletonize']['max_attraction'])
            # lbc = pcs.LBC(point_cloud=clean_pcd,
            #         filter_nb_neighbors = config['skeletonize']['n_neighbors'],
            #         max_iteration_steps=20,
            #         debug = False,
            #         down_sample = 0.0001,
            #         termination_ratio=config['skeletonize']['termination_ratio'],
            #         step_wise_contraction_amplification = config['skeletonize']['init_contraction'],
            #         max_contraction = config['skeletonize']['max_contraction'],
            #         max_attraction = config['skeletonize']['max_attraction'])
            # lbc.extract_skeleton()
            # # Debug/Visualization
            # # lbc.visualize()
            # contracted = lbc.contracted_point_cloud
            # lbc_pcd = lbc.pcd
            # total_shift = arr(lbc_pcd.points)-arr(contracted.points)
            # save(f'{file_base}_total_shift.pkl',total_shift)
            # write_pcd(f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/shifts/{file_base}_contracted.pcd',contracted)

            # topo=extract_topology(lbc.contracted_point_cloud)
            # save_line_set(topo[0],file_base)
            # import pickle 
            # try:
            #     with open(f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/shifts/{file_base}_topo_graph.pkl','rb') as f:
            #         pickle.dump(topo[1],f)
            # except Exception as e:
            #     log.info(f'error saving topo {e}')

        # except Exception as e:
        #     log.info(f'error getting lbc {e}')
    # return lbc, topo

def contract(in_pcd,shift, invert=False):
    "Translates the points in the "
    pts=arr(in_pcd.points)
    if not invert:
        shifted=[(pt[0]-shift[0],pt[1]-shift[1],pt[2]-shift[2]) for pt, shift in zip(pts,shift)]
    else:
        shifted=[(pt[0]+shift[0],pt[1]+shift[1],pt[2]+shift[2]) for pt, shift in zip(pts,shift)]
    contracted = o3d.geometry.PointCloud()
    contracted.colors = o3d.utility.Vector3dVector(arr(in_pcd.colors))
    contracted.points = o3d.utility.Vector3dVector(shifted)
    return contracted

def get_downsample(file = None, pcd = None, normalize = False):
    if file: pcd = read_pcd(file)
    log.info('Down sampling pcd')
    voxed_down = pcd.voxel_down_sample(voxel_size=.05)
    clean_pcd = voxed_down.uniform_down_sample(3)
    # clean_pcd = remove_color_pts(uni_down,invert = True)
    if normalize: _ = center_and_rotate(clean_pcd)
    return clean_pcd

def smooth_feature( points, values, query_pts=None,
                    n_nbrs = 25,
                    nbr_func=np.mean):
    log.info(f'fitting nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='auto').fit(points)
    smoothed_feature = []
    query_pts = query_pts if query_pts is not None else points
    split = np.array_split(query_pts, 100000)
    log.info(f'smoothing feature...')
    def get_nbr_summary(idx, pts):
        # Could also cluster nbrs and set equal to avg of largest cluster 
        return nbr_func(values[nbrs.kneighbors(pts)[1]], axis=1)
    results = Parallel(n_jobs=7)(delayed(get_nbr_summary)(idx, pts) for idx, pts in enumerate(split))
    smoothed_feature = np.hstack(results)
    return smoothed_feature

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
    #         breakpoint()
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
    #                 breakpoint()
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
    #         breakpoint()
    #         log.info(f'error saving smoothed data: {e}')

        # breakpoint()

    # return smoothed_features

        
def project_in_slices(pcd,seed, name='', off_screen = True):
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
        mesh = project_pcd(pts=slice_points, alpha=.2, plot=True, seed=seed, name=name, sub_name=slice_name, off_screen=off_screen)
        # geo = mesh.extract_geometry()
        metrics[slice_name] ={'mesh': mesh, 'mesh_area': mesh.area }
    metrics['total_area'] = np.sum([x['mesh_area'] for x in metrics.values()])
    log.info(f'{name} total area: {metrics["total_area"]}')
    return metrics

def cluster_color(pcd,labels):
    import matplotlib.pyplot as plt
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    orig_colors = np.array(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, orig_colors

def crop_with_box():
    bb = new_pcd.get_oriented_bounding_box()
    bb.center=arr([122,395,1.4])
    bb.extent=arr([14, 8, 0.2])
    new_pt_ids = bb.get_point_indices_within_bounding_box( o3d.utility.Vector3dVector(arr(new_pcd.points)))
    new_pcd = new_pcd.select_by_index(new_pt_ids)

def width_at_height(file_content, save_gif=False, height=1.37, tolerance=0.1, axis=2):
    """
    Calculate the width of a point cloud at a given height above ground.

    Args:
        pcd: open3d.geometry.PointCloud, assumed z is vertical (axis=2)
        height: float, target height above ground in meters
        tolerance: float, +/- range around 'height' for points to use (thickness of slice)
        axis: int, axis index for vertical (default is 2 for z)

    Returns:
        width: float, max distance in x/y-plane (other two axes) between points in slice.
    """
    log.info('identifying epiphytes')
    # user_input = 65
    # while user_input is not None:
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    
    import numpy as np
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
    if len(idx) < 2:
        return 0.0  # Not enough points to define width
    slice_pts = pts[idx]
    # Project to the x/y-plane for width calculation
    if axis == 2:
        plane_pts = slice_pts[:, :2]
    else:
        # e.g., axis=1 means drop y, use x and z
        axes = [i for i in range(3) if i != axis]
        plane_pts = slice_pts[:, axes]
    
    # Viz. the slice against the original pointcloud
    new_pcd = clean_pcd.select_by_index(idx)
    _, ind = new_pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=.95)
    new_pcd = new_pcd.select_by_index(ind)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    center = new_pcd.get_center()
    axis.translate(center, relative=False)
    vis.add_geometry(axis)
    vis.add_geometry(clean_pcd)
    vis.add_geometry(new_pcd)
    vis.run()
    vis.destroy_window()
    draw([new_pcd]) 

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


test = [{'seed':'skio_112_tl_0', 'width': '0.698', 'bounds': np.array([0.698, 0.621, 0.   , 1.103, 0.743])}, \
{'seed':'skio_116_tl_493', 'width': '0.621', 'bounds': np.array([0.621, 0.531, 0.17 , 1.103, 0.743])}, \
{'seed':'skio_137_tl_34', 'width': '1.103', 'bounds': np.array([1.02157047, 1.8493    , 0.1911    ])}, \
{'seed':'skio_108_tl_0', 'width': '0.743', 'bounds': np.array([0.6524477 , 0.72928627, 0.158     ])}, \
{'seed': 'skio_114_tl_0', 'width': '0.481', 'bounds': np.array([0.42325   , 0.53070539, 0.16966039])}, \
{'seed': 'skio_111_tl_0', 'width': '0.588', 'bounds': np.array([0.706  , 0.45875, 0.182  ])}, \
{'seed': 'skio_189_tl_236', 'width': '1.268', 'bounds': np.array([1.02157047, 1.8493    , 0.1911    ])}, \
{'seed': 'skio_0_tl_223', 'width': '1.116', 'bounds': np.array([1.35583333, 0.86      , 0.16698718])}, \
{'seed': 'skio_0_tl_69', 'width': '0.695', 'bounds': np.array([0.6524477 , 0.72928627, 0.158     ])}, \
{'seed': 'skio_191_tl_236', 'width': '1.406', 'bounds': np.array([1.02157047, 1.8493    , 0.1911    ])}, \
{'seed': 'skio_107_tl_0', 'width': '0.701', 'bounds': np.array([0.62389246, 0.71      , 0.18244444])},  \
{'seed': 'skio_134_tl_0', 'width': '0.69', 'bounds': np.array([0.63188333, 0.72208108, 0.16166667])}, \
{'seed': 'skio_0_tl_6', 'width': '0.940', 'bounds': np.array([1.115     , 0.82305952, 0.192     ])}, \
{'seed': 'skio_0_tl_188', 'width': '0.753', 'bounds': np.array([0.8845    , 0.63366607, 0.179     ])}, \
{'seed': 'skio_136_tl_9', 'width': '1.155', 'bounds': np.array([0.97840909, 1.58758846, 0.186     ])}, \
{'seed': 'skio_112_tl_0', 'width': '0.697', 'bounds': np.array([0.62353623, 0.59716897, 0.1728474 ])}, \
{'seed': 'skio_116_tl_493', 'width': '0.627', 'bounds': np.array([0.62732651, 0.61358025, 0.16866667])},  \
{'seed': 'skio_0_tl_157', 'width': '.200', 'bounds': np.array([0.29091667, 0.4775    , 0.17116667])},  \
{'seed': 'skio_0_tl_23', 'width': '0.859', 'bounds': np.array([0.75925   , 0.99261111, 0.1814    ])},  \
{'seed': 'skio_137_tl_34', 'width': '1.37', 'bounds': np.array([1.3733    , 1.53746863, 0.18664286])},  \
{'seed': 'skio_108_tl_0', 'width': '0.813', 'bounds': np.array([0.95068421, 0.64367083, 0.197     ])}, \
{'seed': 'skio_0_tl_307', 'width': '0.585', 'bounds': np.array([0.55324138, 0.62558333, 0.18616667])}, \
{'seed': 'skio_0_tl_246', 'width': '0.79', 'bounds': np.array([0.832 , 0.9095, 0.199 ])}, \
{'seed': 'skio_135_tl_0', 'width': '1.51', 'bounds': np.array([1.51782222, 1.42825   , 0.17480952])}, \
{'seed': 'skio_0_tl_490', 'width': '1.592', 'bounds': np.array([2.109875  , 1.63352174, 0.199     ])}
# {'seed': '109', 'width': '1.027', 'bounds': array([1.32123491, 0.74951172, 0.19942859])},
# {'seed': '68', 'width': '.759', 'bounds': array([0.72000122, 0.70999908, 0.16999984])},
# {'seed': '190', 'width': '1.109', 'bounds': array([1.09286679, 1.20758184, 0.16613868])}]

]
### not for use 
# {'seed': 'skio_33_tl_0', 'width': '0', 'bounds': np.array([2.246     , 3.49415461, 0.2       ])}, # has a bush against trunk
# {'seed': 'skio_115_tl_0', 'width': '0', 'bounds': np.array([0.09126471, 0.22861176, 0.16081176])},
# {'seed': 'skio_138_tl_0', 'width': '0', 'bounds': np.array([3.0225, 5.5895, 0.2   ])},
# {'seed': 'skio_0_tl_15', 'width': '0', 'bounds': np.array([1.98865074, 0.89584848, 0.189     ])}, # half a tree
# {'seed': 'skio_133_tl_68', 'width': '0', 'bounds': np.array([12.06221429,  3.83677778,  0.191     ])},   # three trees

def project_components_in_slices(pcd, clean_pcd, epis, leaves, wood ,seed, name='', off_screen = True):
    metrics={}
    breakpoint()
    metrics['epis'] = project_in_slices(epis,seed, name='epis', off_screen=off_screen)
    metrics['leaves'] = project_in_slices(leaves,seed, name='leaves', off_screen=off_screen)
    metrics['wood'] = project_in_slices(wood,seed, name='wood', off_screen=off_screen)
    
    fin_metrics = {}
    total_area = 0
    for metric_name, metric_dict in metrics.items():
        fin_metrics[metric_name] = metric_dict.pop('total_area')
        total_area += fin_metrics[metric_name]
        fin_metrics[f'{metric_name}_slices'] = [x['mesh_area'] for x in metric_dict.values()]
    fin_metrics['total_area'] = total_area

    mesh = project_pcd(pts=arr(clean_pcd.points), plot=True, seed=seed, name='whole', off_screen=off_screen)
    fin_metrics['whole'] = mesh.area
    mesh = project_pcd(pts=arr(wood.points), plot=True, seed=seed, name='wood_singular', off_screen=off_screen)
    fin_metrics['wood_singular'] = mesh.area

    import pickle
    with open(f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/projected_areas/{seed}/all_metrics_split5.pkl', 'wb') as f: 
        pickle.dump(fin_metrics, f)
    log.info(f'{seed}, {fin_metrics=}')

def identify_epiphytes(file_content, save_gif=False, out_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/'):
    log.info('identifying epiphytes')
    # user_input = 65
    # while user_input is not None:
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    assert shift_one is not None # No shifts for this seed; Ensure you pass get_shifts=True to read_pcds_and_feats
    run_name = f'{seed}_id_epi_'
    epi_file_dir = f'/media/penguaman/backupSpace/lidar_sync/pyqsm/ext_detail/epis/'
    epi_file_name = f'seed{seed}_epis.pcd'
    # draw([pcd])
    # breakpoint()
    orig_colors = deepcopy(arr(clean_pcd.colors))
    # try:
    # highc, lowc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif)
    # except Exception as e:
    #     log.info(f'error drawingskio_0_tl_69 shift for {seed}: {e}')
    #     breakpoint()
    #     log.info(f'error drawing shift for {seed}: {e}')
    highc, lowc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif, pctile_cutoff=65)
    clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
    lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
    highc = clean_pcd.select_by_index(highc_idxs, invert=False)

    draw([clean_pcd])
    draw([highc])
    draw([lowc])
    high_shift = shift_one[highc_idxs]
    z_mag = np.array([x[2] for x in high_shift])
    leaves_idxs, leaves, epis = split_on_percentile(highc,z_mag,60, color_on_percentile=True)
    epis_colored  = highc.select_by_index(leaves_idxs, invert=True)
    draw([lowc, leaves,epis])
    draw([epis])
    # user_input=None
    # user_input = input("Try new percentile? (number): ").strip().lower()
    # if user_input.isdigit():
    #     user_input = int(user_input)
    # else:
    #     user_input = None
    project_components_in_slices(pcd, clean_pcd, epis, leaves, lowc, seed)

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
    #         breakpoint()
    #         new_mesh = tmesh + bmesh
    #         fmesh = meshfix(new_mesh) 
    #         mesh_file_dir = f'{out_path}/ray_casting/epi_mesh/'
    #         o3d.io.write_triangle_mesh(f'{mesh_file_dir}/{seed}_epis_mesh.ply', new_mesh)
    #         breakpoint()    

def identify_epiphytes_tb(file_content, save_gif=False, out_path = 'data/results/gif/'):
    logdir = "/media/penguaman/backupSpace/lidar_sync/pyqsm/tensor_board/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    assert shift_one is not None # No shifts for this seed; Ensure you pass get_shifts=True to read_pcds_and_feats
    run_name = f'{seed}_id_epi_'

    epi_file_dir = f'/media/penguaman/backupSpace/lidar_sync/pyqsm/epis/'
    epi_file_name = f'seed{seed}_epis.pcd'
    step=0
    try:
        with writer.as_default():
            log.info('Calculating/drawing contraction')
            step+=1
            summary.add_3d(run_name, to_dict_batch([clean_pcd]), step=step, logdir=logdir)
            orig_colors = deepcopy(arr(clean_pcd.colors))
            try:
                highc, lowc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif)
            except Exception as e:
                log.info(f'error drawing shift for {seed}: {e}')
                breakpoint()
                log.info(f'error drawing shift for {seed}: {e}')
                
            clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
            lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
            highc = clean_pcd.select_by_index(highc_idxs, invert=False)
            step+=1
            summary.add_3d(run_name, to_dict_batch([lowc]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([highc]), step=step, logdir=logdir)
            draw([clean_pcd])
            draw([highc])
            draw([lowc])
            breakpoint()
            high_shift = shift_one[highc_idxs]
            z_mag = np.array([x[2] for x in high_shift])
            leaves_idxs, leaves, epis = split_on_percentile(highc,z_mag,60, color_on_percentile=True)
            epis_colored  = highc.select_by_index(leaves_idxs, invert=True)
            o3d.io.write_point_cloud(f'{epi_file_dir}/{epi_file_name}', epis_colored)
            # epis_idxs, epis, leaves = split_on_percentile(highc,z_mag,60, comp=lambda x,y:x<y, color_on_percentile=True)
            pcd_no_epi = join_pcds([highc,leaves])[0]
            step+=1
            summary.add_3d(run_name, to_dict_batch([epis]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([epis_colored]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([pcd_no_epi]), step=step, logdir=logdir)
            breakpoint()
            cvar = np.array([np.linalg.norm(x) for x in shift_one])
            proped_cmag = propogate_shift(pcd,clean_pcd,cvar)
            color_continuous_map(pcd,proped_cmag)
            draw([pcd])
            breakpoint()
            log.info('Orienting, extracting hues')
            # center_and_rotate(lowc) 
            hue_pcds,no_hue_pcds =segment_hues(lowc,seed,hues=['white','blues','pink'],draw_gif=False, save_gif=save_gif)
            no_hue_pcds = [x for x in no_hue_pcds if x is not None]
            target = no_hue_pcds[len(no_hue_pcds)-1]
            # draw(target)
            # epis_hue_pcds,epis_no_hue_pcds =segment_hues(epis,seed,hues=['white','blues','pink'],draw_gif=False, save_gif=save_gif)
            # epis_no_hue_pcds = [x for x in epis_no_hue_pcds if x is not None]
            # stripped_epis = epis_no_hue_pcds[len(epis_no_hue_pcds)-1]

            step+=1
            # summary.add_3d('epis', to_dict_batch([stripped_epis]), step=step, logdir=logdir)
            summary.add_3d('id_epi_low', to_dict_batch([target]), step=step, logdir=logdir)
            # summary.add_3d('removed', to_dict_batch([hue_pcds[len(hue_pcds)-1]]), step=step, logdir=logdir)
    except Exception as e:
        log.info(f'error getting epiphytes for {seed}: {e}')
    return []

def get_files_by_seed(data_file_config, 
                        base_dir,
                        # key_pattern = re.compile('.*seed([0-9]{1,3}).*')
                        key_pattern = re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
                        ):
    seed_to_files = defaultdict(dict)
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
            file_key = file_key.groups(1)[0]
            seed_to_files[file_key][file_type] = f'{base_dir}/{file_info["folder"]}/{file}'
    
    return seed_to_files

def np_to_o3d(npz_file):
    data = np.load(npz_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['points'])
    log.info('warning: dividing colors by 255')
    if 'colors' in data.files:
        pcd.colors = o3d.utility.Vector3dVector(data['colors']/255)
    return pcd

def np_feature(feature_name):
    def my_feature_func(npz_file):
        npz_data = np.load(npz_file)
        return npz_data[feature_name]
    return my_feature_func

def get_data_from_config(seed_file_info, data_file_config):
    seed_to_content = defaultdict(dict)
    for file_type, file_path in seed_file_info.items():
        load_func = data_file_config[file_type]['load_func']
        load_kwargs = data_file_config[file_type].get('kwargs',{})
        seed_to_content[file_type] = load_func(file_path, **load_kwargs)
        seed_to_content[f'{file_type}_file'] = file_path
        if file_type == 'src':
            seed_to_content['clean_pcd'] = get_downsample(pcd=seed_to_content['src'],
                                                          **load_kwargs)
    return seed_to_content

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

    if args ==[]: args = ['']*len(kwargs)
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
            # try:
            print(f'getting data for {seed}')
            content = get_data_from_config(seed_file_info, data_file_config)
            content['seed'] = seed
            print(f'running function for {seed} done')
            result = func(content,*arg_tup,**kwarg_dict)
            results.append(result)
            # except Exception as e:
            #     log.info(f'error {e} when processing seed {seed}')
            #     errors.append(seed)
    print(f'{errors=}')
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

    log.info(results)
    breakpoint()
    myTable = create_table(results)
    log.info(myTable)
    breakpoint()    
    log.info('dont finish yet')
    return results

def get_and_label_neighbors(comp_pcd, base_dir, nbr_label, non_nbr_label = 'wood'):

    # comp_kd_tree = sps.KDTree(arr(comp_pcd.points))

    glob_pattern = f'{base_dir}/inputs/skio_parts/*'
    files = glob(glob_pattern)
    logdir = "/media/penguaman/backupSpace/lidar_sync/pyqsm/tensor_board/get_epi_details"
    #Tensor Board prep
    names_to_labels  = {'unknown':0,f'orig_{nbr_label}':1, f'found_{nbr_label}':2, non_nbr_label:3}
    writer = tf.summary.create_file_writer(logdir)
    comp_points = np.array(comp_pcd.points)
    comp_labels = [names_to_labels[f'orig_{nbr_label}']]*len(comp_pcd.points) 
    combined_summary = {'vertex_positions':np.vstack(comp_points,np.array(pcd.points)), 
                        }

    def update_tb(case_name, summary, pcd_labels, step):
        if step >0: 
            summary['vertex_positions'] = 0
            summary['vertex_scores'] = 0
            summary['vertex_features'] = 0
        summary['vertex_labels'] = np.vstack(comp_labels,pcd_labels)            
        summary.add_3d(case_name, summary, step=step, logdir=logdir)
        step = step + 1
    
    epi_ids = defaultdict(list)
    finished = []
    # finished = ['660000000.pcd','400000000.pcd','800000000.pcd','560000000.pcd']
    files = [file for file in files if not any(file_name in file for file_name in finished)]
    with writer.as_default(): 
        for file in files:
            #  = {'vertex_positions': np.array(comp_pcd.points), 'vertex_labels': np.zeros(len(comp_pcd.points)), 'vertex_scores': np.zeros((len(comp_pcd.points), 3)), 'vertex_features': np.zeros((len(comp_pcd.points), 3))}
            # summary.add_3d('get_epi_details', summary_dict, step=step, logdir=logdir)
            try:
                #identify neighbors
                file_name = file.split('/')[-1].replace('.pcd','')
                case_name = f'{file_name}_get_{nbr_label}_details'
                pcd = read_pcd(file)
                log.info(f'getting neighbors for {file_name}')
                nbrs_pcd, nbrs, chained_nbrs = get_neighbors_kdtree(pcd, comp_pcd, return_pcd=True)
                if nbrs_pcd is None:
                    log.info(f'no nbrs found for {file_name}')
                    epi_ids[file_name] = []
                    continue
                uniques = np.unique(chained_nbrs)
                o3d.io.write_point_cloud(f'{base_dir}/detail/{file_name}_nbrs.pcd', nbrs_pcd)
                non_matched = pcd.select_by_index(uniques, invert=True)
                o3d.io.write_point_cloud(f'{base_dir}/not_epis/{file_name}_non_matched.pcd', non_matched)

                # Add run to Tensor Board (done at the end in case there are no neighbors)
                ## Initial
                step=0
                vertices = np.vstack(comp_points,np.array(pcd.points))
                combined_summary['vertex_positions'] = vertices
                combined_summary['vertex_scores'] = np.zeros_like(vertices)
                combined_summary['vertex_features'] = np.zeros_like(vertices)

                pcd_labels = np.zeros(len(pcd.points))
                pcd_labels = np.vstack(comp_labels,pcd_labels)
                update_tb(case_name, combined_summary, pcd_labels, step)
                log.info(f'{step=}')
            
                # Add labels for epiphytes
                pcd_labels[uniques] = names_to_labels[f'found_{nbr_label}']
                update_tb(case_name, combined_summary, pcd_labels, step)
                log.info(f'{step=}')

                # Update labels for 'wood' (e.g. whatever is left over)
                pcd_labels = np.full_like(pcd_labels, names_to_labels[non_nbr_label])
                pcd_labels[uniques] = names_to_labels[f'found_{nbr_label}']
                update_tb(case_name, combined_summary, pcd_labels, step)
                log.info(f'{step=}')

                # Save epiphyte nbrs for future use 
                epi_ids[file_name] = uniques
            except Exception as e:
                log.info(f'error {e} when getting neighbors in tile {file_name}')

    np.savez_compressed(f'{base_dir}/epis/epis_ids_by_tile.npz', **epi_ids)
    breakpoint()

def get_features(file, step_through=True):#comp_pcd, base_dir, comp_file_name = ''):
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
    # if invert:
    #     uniques = np.where(~in_occupied_voxel_mask)[0]
    # else:
    uniques = np.where(in_occupied_voxel_mask)[0]

    return uniques

def get_nbrs_voxel_grid(comp_pcd, 
                        comp_file_name,
                        tile_dir, 
                        tile_pattern,
                        invert=False,
                        out_folder='detail',
                        ):
    log.info('creating voxel grid')
    comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(comp_pcd,voxel_size=0.1)
    log.info('voxel grid created')
    nbr_ids = defaultdict(list)
    all_data =  defaultdict(list)
                        
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

        nbr_dir = f'{tile_dir}/color_int_tree_nbrs/{file_name}'
        # existing_nbrs = assemble_nbrs([nbr_dir])[nbr_dir]
        # if len(existing_nbrs) > 0:
        #     log.info(f'existing nbrs found: {len(existing_nbrs)}')

        uniques = overlap_voxel_grid(data['points'], comp_voxel_grid, invert=invert)

        if not os.path.exists(nbr_dir):
            os.makedirs(nbr_dir)
        np.savez_compressed(f'{nbr_dir}/detail_feats_{comp_file_name}.npz', nbrs=uniques)
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

    # some_feats = [all_data[ffile] for ffile in all_data.keys() if ffile not in ['color','points']]
    # feats = np.hstack([x[:,np.newaxis] for x in some_feats])
    # kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(feats)
    # unique_vals, counts = np.unique(kmeans.labels_, return_counts=True)
    # log.info(f'{unique_vals=} {counts=}')
    # cluster_idxs = [np.where(kmeans.labels_==val)[0] for val in unique_vals]

    np.savez_compressed(f'{out_folder}/{comp_file_name}.npz', **all_data)
    # breakpoint()
    # o3d.io.write_point_cloud(f'{base_dir}/int_color_data_{comp_file_name}.pcd', pcd)
    # breakpoint()
    return all_data
    # log.info('Calculating smoothed features and plotting')
    # get_smoothed_features(all_data, save_file = save_file, step_through=step_through, file_name=comp_file_name)
     
def assemble_nbrs(requested_dirs:list[str]=[]):
    """
        Used to use nbr files from get_nbrs_voxel_grid to determine
           which points in the src pcd tiles files are not assigned to a tree
    """
    if len(requested_dirs)>0:
        nbr_dirs = requested_dirs
    else:
        nbr_dirs = glob('/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/color_int_tree_nbrs/*')
    nbr_lists = defaultdict(list)
    for nbr_dir in tqdm(nbr_dirs):
        log.info(f'getting nbr files form {nbr_dir=}')
        all_tree_nbrs = []
        nbr_files = glob(f'{nbr_dir}/*.npz')
        for nbr_file in nbr_files:
            nbr_name = nbr_file.split('/')[-1].replace('.npz','')
            nbrs = np.load(nbr_file)['nbrs']
            all_tree_nbrs.extend(nbrs)
        # nbr_lists[nbr_dir] = all_tree_nbrs
        np.savez_compressed(f'{nbr_dir}_all_tree_nbrs.npz', nbrs=all_tree_nbrs)
    return nbr_lists

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
    breakpoint()

    _ = join_pcd_files(f'{base_dir}/not_epis/',
                        pattern = '*non_matched.pcd',
                        voxel_size = .05,
                        write_to_file = False)
    breakpoint()

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
    breakpoint()    

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
    #         breakpoint()
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
            breakpoint()
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
    #     breakpoint()
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

if __name__ =="__main__":

    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining'
    files = glob(f'{base_dir}/to_get_detail/*_joined_replaced.pcd')
    # seed_pat = '.*seed([0-9]{1,3}).*'
    seed_pat = '.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*'
    # seed_pat = '([0-9]{1,3})_joined_2.*'
    for file in files:
        comp_file_name = re.match(re.compile(seed_pat),file).groups(1)[0]
        # comp_file_name = f'skio_0_tl_{comp_file_name}'
        log.info(f'processing {comp_file_name}')
        tile_dir = '/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/'
        tile_pattern = 'SKIO-RaffaiEtAlcolor_int_*.npz'
        comp_pcd = read_pcd(file)
        # comp_pcd = to_o3d(coords=np.load(file)['points'] + np.array([113.42908253, 369.58605156,   4.60074394]))
        get_nbrs_voxel_grid(comp_pcd,
                        comp_file_name,
                        tile_dir = tile_dir,
                        tile_pattern = tile_pattern,
                        invert=False,
                        out_folder= f'{base_dir}/detail')
    breakpoint()
    # assemble_nbrs()
    # breakpoint()
    # # 246
    # base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/'
    # remaining_files, remaining_keys = compare_dirs(base_dir + 'detail/', base_dir + 'shifts/',
    #                             file_pat1 = 'skio_*_tl_*.npz', file_pat2 = 'skio_*_tl_*_shift.pkl',
    #                             key_pat1 = '(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*', key_pat2 = '(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
    # remaining_keys = list(set(remaining_keys) - set(skip_seeds))
    # print(f'{remaining_keys=}')
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/'
    ######## 138, 193 partial trees
    loop_over_files(
                    # width_at_height,
                    
                    parallel=False,
                    base_dir=base_dir,
                    data_file_config={ 
                        'src': {
                                'folder': 'to_get_detail/',
                                'file_pattern': f'*.pcd',
                                'load_func': read_pcd, 
                            },
                        # 'shift_one': {
                        #         'folder': 'shifts',
                        #         'file_pattern': 'skio_*_tl_*_shift.pkl',
                        #         'load_func': lambda x,root_dir: load(x,root_dir)[0], 
                        #         'kwargs': {'root_dir': '/'},
                        #     },
                        # 'intensity': {
                        #         'folder': 'detail',
                        #         'file_pattern': f'skio_*_tl_*.npz',
                        #         'load_func': np_feature('intensity'), # custom, for pickles
                        #     },
                    },
                    # seed_pat = re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
                    seed_pat = re.compile('.*seed([0-9]{1,3}).*'),
                    
                    )
    breakpoint()
    