import os
import numpy as np
import open3d as o3d

import pickle
from collections import defaultdict
from itertools import chain
import re
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from canopy_metrics import project_components_in_clusters, sps
from open3d.visualization import draw_geometries as draw, draw_geometries_with_editing as edit

from exploration import random_forest_classification

colors_palette = [
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [1, 0, 1],  # magenta
    [0, 1, 1],  # cyan
    [0.5, 0.5, 0.5],  # grey
    [1, 0.5, 0],  # orange
    [0.5, 0, 1],  # purple
    [0, 0, 0]   # black
]

def get_pcds_from_lbls(pts, colors,
            lbl_idxs, grped_lbls,
            label_groups = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
   
    from itertools import chain
    labeled_pcds = []
    unlabeled_mask = np.ones_like(pcd.points, dtype=bool)
    
    for i, idx_list in enumerate(lbl_idxs.values()):
        subset_pcd = pcd.select_by_index(idx_list)
        unlabeled_mask[idx_list] = False
        subset_pcd.paint_uniform_color(colors_palette[i % len(colors_palette)])
        labeled_pcds.append(subset_pcd)

    unlabeled_idxs = np.where(unlabeled_mask)[0]
    unlabeled_pcd = pcd.select_by_index(unlabeled_idxs)

    # Assign integer labels for each group
    group_labels = []
    if label_groups is None:
        label_groups = grped_lbls
    for i, group_name in enumerate(label_groups.keys()):
        group_indices = np.concatenate(grped_lbls[group_name])
        group_labels.append(np.full_like(group_indices, i, dtype=int))
    group_labels = np.concatenate(group_labels)

    return group_labels, unlabeled_idxs, labeled_pcds, unlabeled_pcd

def call_random_forest_classification(feat_names, file_name, pts, colors, all_labeled_idxs, unlabeled_idxs):
    group_labels, unlabeled_idxs, labeled_pcds, unlabeled_pcd = get_pcds_from_lbls(pts, colors)
    smoothed_feats = dict()
    for feat_name in feat_names:
        smoothed_feats[feat_name] = np.load(f'/media/penguaman/data/kevin_holden/orig/features/{file_name}_smoothed_{feat_name}.npz')['feats']
    smoothed_feats['x_coords'] = pts[:, 0]
    smoothed_feats['y_coords'] = pts[:, 1]
    smoothed_feats['z_coords'] = pts[:, 2]
    model_feat_names = [ 
                'x_coords',
                'y_coords',
                'linearity', 
                'planarity',
                'surface_variation',
                'anisotropy',
                'sphericity',
    ]
    stacked_feats = [smoothed_feats[f] for f in model_feat_names]
    # Stack to shape (num_features, N), then transpose after index
    all_feats = np.stack(stacked_feats, axis=1)
    labeled_feats = all_feats[all_labeled_idxs]
    unlabeled_feats = all_feats[unlabeled_idxs]
    random_forest_classification(model_feat_names, smoothed_feats, 
                                   all_labeled_idxs, unlabeled_idxs, group_labels, file_name)

def get_labels():
    from itertools import chain
    import laspy
    labled_idxs_file = np.load('kh_labeled_idxs_fin.npz')
    labeled_idxs = {fname:labled_idxs_file[fname] for fname in labled_idxs_file.files}
    label_groups = {
        'wood': ['branch_bottoms', 'distal_branches', 'high_lin_branches', 'trunk_and_branches','trunk'
        ,'rf_wood'
         ],
        'leaves': ['high_sv_leaves', 'low_v_leaves', 'not_high_ansio_leaves', 'unlabeled_mostly_leaves'
        , 'rf_leaves'],
        'improved_wood': ['improved_wood']
    }
    grped_lbls = {grp_name: [labeled_idxs[lbl] for lbl in grp_lbls] for grp_name,grp_lbls in label_groups.items()}
    lbl_idxs = {grp_name: list(chain.from_iterable(lbl_lists)) for grp_name,lbl_lists in grped_lbls.items()}

    return lbl_idxs, grped_lbls

def get_kevin_holden_data():
    input_dir = '/media/penguaman/data/kevin_holden/orig'
    file_name_pattern = 'fave_cropped_kh_div100.pcd'
    pcd = o3d.io.read_point_cloud(os.path.join(input_dir, file_name_pattern))
    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)*100)

    lbl_idxs, grped_lbls = get_labels()
    group_labels, unlabeled_idxs, labeled_pcds, unlabeled_pcd = get_pcds_from_lbls(pcd.points, pcd.colors, lbl_idxs, grped_lbls)
    breakpoint()  
    clean_pcd = pcd.uniform_down_sample(15)
    res = project_components_in_clusters(pcd, clean_pcd, labeled_pcds[0], labeled_pcds[1], labeled_pcds[-1], seed='kh_holden',)
    # res = project_components_in_slices(pcd, clean_pcd, labeled_pcds[0], labeled_pcds[1], labeled_pcds[0], seed='kh_holden',)
    breakpoint()
    print(res)

def recover_orig_file_details():
    # read in cropped and translated pcd used for work
    input_dir = '/media/penguaman/data/kevin_holden/orig'
    file_name_pattern = 'fave_cropped_kh_div100.pcd'
    pcd = o3d.io.read_point_cloud(os.path.join(input_dir, file_name_pattern))
    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)/10)

    # read in only translated version
    src_pcd = o3d.io.read_point_cloud(os.path.join('/media/penguaman/data/kevin_holden/orig/', 'MyFavPop_Alone_kh.pcd'))
    src_pcd.points = o3d.utility.Vector3dVector(np.array(src_pcd.points)/1000)
    src_center = np.array(src_pcd.get_center())

    # read in original las file
    import laspy
    las = laspy.read('/media/penguaman/data/kevin_holden/orig/MyFavPop_Alone_kh.las')
    las_pcd = o3d.geometry.PointCloud()
    las_pcd.points = o3d.utility.Vector3dVector(np.array(las.xyz))
    las_center = np.array(las_pcd.get_center())
    

    trans =  src_center -las_center # =  np.array([  3.99181472, -12.22140417,  -9.41454787])
    las_pcd.translate(trans)
    
    # draw translated orig and final to ensure translation is correct
    las_pcd.paint_uniform_color([0,1,0])
    pcd.paint_uniform_color([1,0,0])
    draw([las_pcd,pcd])
    breakpoint()
    
    # Match points in pcd to points in las 1 to 1
    src_pts = np.array(las_pcd.points)
    query_pts = np.array(pcd.points)
    src_tree = sps.KDTree(src_pts)
    dists,nbrs = src_tree.query(query_pts, k=1, distance_upper_bound= .01)
    num_pts = len(src_pts)
    good_nbrs = np.array([x for x in nbrs  if x< num_pts])
    len(nbrs)
    len(good_nbrs)
    match_pcd = las_pcd.select_by_index(good_nbrs)
    draw([las_pcd,pcd,match_pcd])
    breakpoint()

    new_las = las[good_nbrs]

    new_file = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_file.xyz = new_las.xyz
    feature_names =  ['verticality', 'linearity', 'planarity', 'surface_variation',  'omnivariance', 'eigenentropy', 'anisotropy', 'PCA1', 'PCA2',  'sphericity', 'eigenvalue_sum']
    file_name = '/media/penguaman/data/kevin_holden/orig/MyFavPop_Alone_kh.pcd'
    smoothed_feats={}
    for feat_name in feature_names: 
        smoothed_feats[feat_name] = np.load(f'/media/penguaman/data/kevin_holden/orig/features/MyFavPop_Alone_kh_smoothed_{feat_name}.npz')['feats']
        new_las.add_extra_dim(laspy.ExtraBytesParams(name=feat_name, type='float32'))
        new_las[feat_name] = smoothed_feats[feat_name]
    
    new_las.write('/media/penguaman/data/kevin_holden/orig/MyFavPop_Alone_kh_cropped.las')
    fin_las = laspy.read('/media/penguaman/data/kevin_holden/orig/orig_cropped_and_trasformed.las')

    lbl_idxs, _ = get_labels()
    label_to_val={
        'leaves': 1,
        'wood': 2,
        'improved_wood': 2
    }
    for grp_name, grp_lbls in lbl_idxs.items():
        fin_las.add_extra_dim(laspy.ExtraBytesParams(name='label', type='int32'))
        fin_las['label'][grp_lbls] = label_to_val[grp_name]

    fin_las = laspy.read('/media/penguaman/data/kevin_holden/orig/orig_cropped_and_trasformed.las')
    fin_las.add_extra_dim(laspy.ExtraBytesParams(name='label', type='int32'))
    new_las['label'] = np.ones_like(new_las['label'], dtype=int) * -1

# def call_loop_over_files():
#     from open3d.visualization import draw_geometries_with_editing as edit
#     from qsm_generation import get_stem_pcd
#     edit([pcd])
#     stem_pcd = get_stem_pcd(clean_pcd)
#     breakpoint()

#     base_dir = '/media/penguaman/data/kevin_holden/'
#     loop_over_files(
#                     get_shift,
#                     # requested_seeds=requested_seeds,
#                     parallel = False,
#                     base_dir=base_dir,
#                     data_file_config={ 
#                         'src': {
#                                 'folder': 'orig/',
#                                 # 'file_pattern': f'*_cropped.pcd
#                                 'file_pattern': f'*.las',
#                                 'load_func': read_and_downsample, 
#                             },
#                     },
#                     seed_pat = re.compile('(.*)_kh.*')
#                     )

if __name__ =="__main__":
    get_kevin_holden_data()
    breakpoint()