import os
import pickle
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm

from collections import defaultdict
from joblib import Parallel, delayed
from jakteristics import compute_features as compute_features_j
from sklearn.neighbors import NearestNeighbors
from set_config import log
from viz.viz_utils import color_continuous_map
from viz.plotting import histogram
from pyQSM.viz.viz_utils import draw
from string import Template

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def voxelize_pcd(file, vox_size=None, ext = '.pcd'):
    if ext == '.npz':
        print(f'loading data from {file}')
        data = np.load(file)
        pts = data['points']
        colors = data['colors']
        all_points=pts
        all_colors =colors
        # pts = np.vstack(all_points)
        # colors = np.vstack(all_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    elif ext == '.pcd':
        print(f'loading data from {file}')
        pcd = o3d.io.read_point_cloud(file)
        pts = np.array(pcd.points)
        colors = np.array(pcd.colors)
        all_points=pts
        all_colors =colors
    # breakpoint()
    print(f'voxelizing {file} with {len(all_points)} points')
    if vox_size is not None:
        vox_pcd = pcd.voxel_down_sample(voxel_size=vox_size)
        pts = np.array(vox_pcd.points)
        colors = np.array(vox_pcd.colors)
        print(f'voxelized {file} with {len(pts)} points')
    return pts, colors

def replace_nanfeatures(features, feature_names, nan_func= np.nanmean):
    # eventually I want to switch this over to use 
    # an NN interpolator: https://stackoverflow.com/questions/68197762/fill-nan-with-nearest-neighbor-in-numpy-array
    for feature_name in feature_names:
        ind_nan = np.isnan(features[feature_name])
        mean_values = nan_func(features[feature_name], axis=0)
        print(f" {ind_nan.sum()} points have a null value for {feature_name}.")
        for i in range(features[feature_name].shape[1]):
            if ind_nan[:, i].sum() > 0:
                features[feature_name][:, i][ind_nan[:, i]] = mean_values[i]
    return features

def compute_features(points, search_radius=0.6, feature_names=['verticality'], num_threads=4):
    #eigenvalue_sum, Omnivariance, Eigenentropy, Anisotropy,Planarity,Linearity, PCA1,PCA2,Surface Variation,Sphericity,Verticality
    print(f'Computing features for {len(points)} points')
    features = compute_features_j(points, search_radius=search_radius, num_threads=num_threads, feature_names=feature_names)
    features = replace_nanfeatures(features, feature_names)
    features = features.astype(np.float32)
    return features

def smooth_feature( points, values, query_pts=None,
                    n_nbrs = 25,
                    smoothing_func=np.mean):
    import time
    start_time = time.time()
    log.info(f'{start_time}: fitting nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='auto').fit(points)
    smoothed_feature = []
    query_pts = query_pts if query_pts is not None else points
    split = np.array_split(query_pts, 100000)
    log.info(f'smoothing feature...')
    def get_nbr_summary(idx, pts):
        # Could also cluster nbrs and set equal to avg of largest cluster 
        return smoothing_func(values[nbrs.kneighbors(pts)[1]], axis=1)
    results = Parallel(n_jobs=7)(delayed(get_nbr_summary)(idx, pts) for idx, pts in enumerate(split))
    smoothed_feature = np.hstack(results)
    
    elapsed = time.time() - start_time
    elapsed_hours = int(elapsed // 3600)
    elapsed_minutes = int((elapsed % 3600) // 60)
    log.info(f'{start_time}: {elapsed_hours} hours, {elapsed_minutes} minutes')
    return smoothed_feature

def check_files_for_feature(feature_names, file_template):
    """
    Checks the existence of precomputed feature files for given feature names within an input directory.
    Determines whether each feature needs to be recalculated or can be loaded from disk.
    For each requested feature:
        - Looks for a master feature file or instance feature file based on the reduction factor and search radius.
        - If a feature file exists, loads the data and adds it to the result. Otherwise, marks the feature for computation.
    Returns:
        feats:        Dictionary of features loaded from file, keyed by feature name.
        to_calc:      List of feature names that need to be computed.
    Args:
        feature_names:    List of feature names to check for.
        input_dir:        Directory to search for feature files in 'features/'.
        feats_file:       The file to check for features.
    """
    use_master_feats = False
    feats = {}
    to_calc = []
    for feature_name in feature_names:
        print(f' looking for  {feature_name}')
        instance_feats_file = file_template.substitute(featureName=feature_name) 
        if os.path.exists(instance_feats_file):
            print(f'{instance_feats_file} exists')
            feats_data = np.load(instance_feats_file)
            if feature_name in feats_data.keys():
                feats[feature_name] = feats_data[feature_name]
            else:
                print(f'{feature_name} not found in {feats_file}')
                to_calc.append(feature_name)
        else:
            print(f'no feats file found for {feature_name}')
            to_calc.append(feature_name)
    return feats, to_calc

def get_file_and_features(input_dir,file_name_pattern = None 
                                ,pcd = None
                                ,vox_size=.01
                                ,reduction_factor=1
                                ,feature_names = ['planarity','linearity','verticality', 'surface_variation']
                                ,search_radius=0.6
                                ,smoothing_func = None
                                ,smoothing_nbrs=50):
    """
    Get the file and features for the given input directory and file name snippet
    Available features (feature_names):
        Eigenvalue sum, Omnidatavariance, Eigenentropy, 
        Anisotropy,Planarity,Linearity, PCA1,PCA2, 
        Surface Variation,Sphericity,Verticality
            
    Args:
        input_dir: The directory to search for files
        file_name_pattern: The pattern to search for in the file names
        pcd: The pcd file to use
        vox_size: The voxel size to use
        reduction_factor: The reduction factor to apply to the features
        feature_names: The features to compute
        search_radius: The search radius to use
        smoothing_func: The function to use for smoothing
        smoothing_nbrs: The number of nearest neighbors to use for smoothing
    """
    files = glob(f'{input_dir}/{file_name_pattern}')
    detail_file_name=''
    for file in files:
        print(f' inspecting {file}')
        detail_file_name = os.path.basename(file).split('.')[0]
        if pcd is None:
            pts, colors = voxelize_pcd(file, vox_size)
        else:
            pts = np.array(pcd.points)
        
        # Define standard for files generated 
        if search_radius is None: 
            file_name= f'{detail_file_name}_reduced_{reduction_factor}_$featureName'
        else:
            file_name = f'{detail_file_name}_reduced_{reduction_factor}_search_{search_radius}_$featureName'
        file_template = Template(f'{input_dir}/features/{file_name}.npz')
        feat_values, to_calc = check_files_for_feature(feature_names, file_template)
        
        # Compute features that are not already computed
        if len(to_calc) > 0:
            print(f'{detail_file_name} needs to calculate {to_calc}')
            all_feats = compute_features(pts, feature_names=to_calc, search_radius=search_radius)
            for feature_name, feats in zip(to_calc, all_feats.T):
                feats_file = file_template.substitute(featureName=feature_name)
                save_dict = {feature_name: feats}
                np.savez_compressed(feats_file, **save_dict)
                print(f'{feats_file} saved')
        
        # Smooth features that are not already smoothed
        smooth_file_template = Template(f'{input_dir}/features/{detail_file_name}_smoothed_$featureName.npz')
        smooth_feat_values, to_calc = check_files_for_feature(feature_names, smooth_file_template)
        smooth_feat_values= dict()
        if smoothing_func is not None:
            for feat_name in to_calc:
                log.info(f'smoothing {feat_name}...')
                smooth = smooth_feature(pts, feats, n_nbrs=smoothing_nbrs, smoothing_func=smoothing_func)
                save_file = smooth_file_template.substitute(featureName=feat_name)
                np.savez_compressed(save_file, feats=smooth)
                smooth_feat_values[feat_name] = smooth
        
        draw_feats(detail_file_name, pts, colors, feat_values, smooth_feat_values)
        # draw_from_lbls(detail_file_name, pts, colors, feat_values)
    return feat_values, smooth_feat_values

# @tensorboard_summary(logdir="src/logs/draw_feats_tb")
def visualize_skio_pointcloud(npz_file_path='/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/SKIO_voxpt05_all.npz', 
                              voxelized_file_path=None,
                              feature_to_visualize='verticality', 
                            #   exclude_boundaries=None,
                              use_colors=True,
                              voxel_size=None):
    """
    Visualize the SKIO point cloud from NPZ file with voxel_sizefeatures and colors.
    
    Args:
        npz_file_path (str): Path to the NPZ file containing point cloud data
        feature_to_visualize (str): Feature to color-code the point cloud by
        use_colors (bool): Whether to use original colors or feature-based coloring
        voxel_size (float): Optional voxel size for downsampling (None for no downsampling)
    """
    print(f"Loading point cloud from {npz_file_path}")
    reduction_factor = 1 if voxelized_file_path is not None else 5
    # # Load the NPZ file
  
    file = voxelized_file_path or npz_file_path
    data = np.load(file)
    points = data['points'][::reduction_factor]
    colors = data['colors'][::reduction_factor] if 'colors' in data else None
    intensity = data['intensity'][::reduction_factor] if 'intensity' in data else None
    print(f"Available keys in NPZ file: {list(data.keys())}")
    
    idx= None
    pcd = None
    # voxelize the data if requested
    # if voxelized_file_path is None:
    #     if voxel_size is not None:
    #         to_join = [points]
    #         if colors is not None:
    #             to_join.append(colors)
    #         if intensity is not None:
    #             to_join.append(intensity[:,np.newaxis])
    #         data = np.hstack(to_join)
    #             # voxelize the data if requested
    #         pcd, data, idx = voxelize(data, voxel_size)
    #         points = data[:, :3]
    #         colors = data[:, 3:5]
    #         intensity = data[:, 6]

    if pcd is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        print(f'coloring...')


    print(f"Loaded {len(points)} points")
    # Set colors
    if use_colors and colors is not None:
        # Normalize colors to [0,1] range if they're in [0,255]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Using original colors")
    elif intensity is not None:
        # Use intensity as grayscale colors
        # intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # intensity_colors = np.column_stack([intensity_normalized, intensity_normalized, intensity_normalized])
        # pcd.colors = o3d.utility.Vector3dVector(intensity_colors)
        # Smooth intensity: for each point, set its intensity to the average intensity of its 100 nearest neighbors (including itself)
        print(f'smoothing intensity with nearest neighbors...')
        if intensity is not None:
            breakpoint()
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = 100
            # Prepare points for neighbor search
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(points)
            smoothed_intensity = []
            split = np.array_split(points, 10000)
            all_nbr_intensities =[]
            for pts in tqdm(split): 
                nbr_intensities =intensity[nbrs.kneighbors(pts)[1]]
                all_nbr_intensities.append(nbr_intensities)
            all_nbr_intensities = np.concatenate(all_nbr_intensities)
            # Smpoothed
            mean_nbr_intensity = np.mean(all_nbr_intensities, axis=1)
            smoothed_intensity.append(mean_nbr_intensity)
            # np.savez_compressed('/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/SKIO_voxpt05_int_smoothed_voxpt05.npz',points = data[:, :3], colors = data[:,3:5], intensity=data[:,6])
            print(f'coloring with smoothed intensity ...')
            color_continuous_map(pcd, smoothed_intensity)
        #     draw([pcd])
        intensity = smoothed_intensity 
        

        print(f'filtering intensity outliers...')
        inlier_idxs = np.where(intensity<-1700)[0] # good for epiphytes
        max_int = max(intensity)
        min_int = min(intensity)
        inlier_idxs = np.where(intensity[0]>=-1800)[0]
        inlier_intensity = intensity[inlier_idxs]

        in_pcd = pcd.select_by_index(inlier_idxs)
        color_continuous_map(pcd, inlier_intensity)
        
        draw([pcd])
        if intensity is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.hist(intensity, bins=100, color='dodgerblue', edgecolor='black', alpha=0.75)
            plt.title('Intensity Histogram')
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        breakpoint()
        color_continuous_map(pcd, intensity)
        print("Using intensity as grayscale colors")
    # Create a histogram of intensity if intensity values are available
    else:
        # Default white color
        pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)))
        print("Using default white colors")
    
    # Optional voxel downsampling
    if voxel_size is not None:
        print(f"Downsampling with voxel size {voxel_size}")
        original_count = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled from {original_count} to {len(pcd.points)} points")
    
    # Visualize the point cloud
    print("Displaying point cloud...")
    draw([pcd])
    
    # Visualize features if available
    available_features = ['verticality', 'linearity', 'planarity', 'surface_variation', 
                         'omnivariance', 'eigenentropy', 'anisotropy', 'PCA1', 'PCA2', 
                         'sphericity', 'eigenvalue_sum']
    
    for feature_name in available_features:
        if feature_name in data:
            feature_values = data[feature_name][::reduction_factor]
            if idx is not None:
                feature_values = feature_values[idx]

            # Create a copy of the point cloud for feature visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Apply voxel downsampling if specified
            if voxel_size is not None:
                pcd_feature = pcd_feature.voxel_down_sample(voxel_size=voxel_size)
                # Downsample feature values to match point cloud
                # This is a simple approach - in practice you might want more sophisticated downsampling
                indices = np.random.choice(len(feature_values), len(pcd_feature.points), replace=False)
                feature_values = feature_values[indices]
            
            # Color the point cloud by the feature
            color_continuous_map(pcd_feature, feature_values)
            draw([pcd_feature])
            
            # If specific feature was requested, break after showing it
            if feature_to_visualize == feature_name:
                break
    
    print("Visualization complete!")

def draw_from_lbls(file_name, pts, colors, feats_dict):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    smoothed_feats = dict()
    for feat_name, feats  in feats_dict.items():
        smooth_file = f'/media/penguaman/data/kevin_holden/orig/features/{file_name}_smoothed_{feat_name}.npz'
        smooth = np.load(smooth_file)['feats']
        smoothed_feats[feat_name] = smooth

    labled_idxs_file = np.load('kh_labeled_idxs_fin.npz')
    labeled_idxs = {fname:labled_idxs_file[fname] for fname in labled_idxs_file.files}
    # Draw each labeled subset of pcd, coloring each one differently
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
    from itertools import chain
    labeled_pcds = []
    label_names = list(labeled_idxs.keys())
    label_groups = {
        'wood': ['branch_bottoms', 'distal_branches', 'high_lin_branches', 'trunk_and_branches','trunk'
        ,'rf_wood'
         ],
        # 'trunk','trunk_and_branches'
        'leaves': ['high_sv_leaves', 'low_v_leaves', 'not_high_ansio_leaves', 'unlabeled_mostly_leaves'
        , 'rf_leaves']
        # 'small_vert_leaves'
    }
    ignore_keys = ['trunk',]
    unlabeled_mask = np.ones_like(pcd.points, dtype=bool)
    grped_idxs = defaultdict(list)
    for i, (group_name, group_label_names) in enumerate(label_groups.items()):
        for label_name in group_label_names:
            print(f'adding {label_name} to {group_name}')
            grped_idxs[group_name].append(labeled_idxs[label_name])
            unlabeled_mask[labeled_idxs[label_name]] = False
            # subset_pcd = pcd.select_by_index(list(chain.from_iterable(grped_idxs[group_name])))
            # subset_pcd.paint_uniform_color(colors_palette[i % len(colors_palette)])
            # draw(subset_pcd)
        subset_pcd = pcd.select_by_index(list(chain.from_iterable(grped_idxs[group_name])))
        subset_pcd.paint_uniform_color(colors_palette[i % len(colors_palette)])
        labeled_pcds.append(subset_pcd)
    breakpoint()
    # for ignore_key in ignore_keys:
    #     unlabeled_mask[labeled_idxs[ignore_key]] = False
    unlabeled_idxs = np.where(unlabeled_mask)[0]
    unlabeled_pcd = pcd.select_by_index(unlabeled_idxs)
    # labeled_pcds.append(pcd.select_by_index(unlabeled_idxs))
    draw(labeled_pcds)
    draw(unlabeled_pcd)
    breakpoint()
    
    from sklearn.cluster import KMeans

    # Prepare the features and labels for kmeans training
    # We'll use all grouped labeled indices (wood + leaves) as training
    all_labeled_idxs = np.concatenate(list(chain.from_iterable(grped_idxs.values())))

    

    # Add x and y coords from pts to the model_feats (i.e., use pts[:,0] and pts[:,1] for all_labeled_idxs)
    # Assume pts is (N,3) xyz point cloud numpy array
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    z_coords = pts[:, 2]
    smoothed_feats['x_coords'] = x_coords
    smoothed_feats['y_coords'] = y_coords
    smoothed_feats['z_coords'] = z_coords
    model_feat_names_options =[ 
        # [
        #         'x_coords',
        #         'y_coords',
        #         'linearity', 
        #         'planarity',
        #         'surface_variation'],
        #        [
                # 'x_coords',
                # 'y_coords',
        #         'z_coords',
        #         'linearity', 
        #         'planarity',
        #         'surface_variation'],
                [
                'x_coords',
                'y_coords',
                'linearity', 
                'planarity',
                'surface_variation',
                'anisotropy',
                'sphericity',]
    ]

def random_forest_classification(model_feat_names_options:list[str], 
                                  smoothed_feats, 
                                  all_labeled_idxs, 
                                  unlabeled_idxs, 
                                  group_labels,
                                  file_name,
                                  num_trees=[200],
                                  train_size=.8):
    for idx, model_feat_names in enumerate(model_feat_names_options):
        print(f'using {model_feat_names} for model {idx}')
        # Stack feature arrays for the labeled points: shape (num_features, num_labeled_points)
        stacked_feats = [smoothed_feats[f] for f in model_feat_names]
        # Stack to shape (num_features, N), then transpose after index
        all_feats = np.stack(stacked_feats, axis=1)
        labeled_feats = all_feats[all_labeled_idxs]
        unlabeled_feats = all_feats[unlabeled_idxs]
        # Use stratify for balanced class splits
        X_train, X_test, y_train, y_test, train_idxs, test_idxs = train_test_split(
            labeled_feats, group_labels, all_labeled_idxs,
            train_size=.8, random_state=42, stratify=group_labels, shuffle=True,
        )

        for num_trees in [201]:
            # Fit KMeans with number of clusters = # groups
            # log.info(f'fitting random forest classifier on {X_train.shape[0]} training samples')
            rf = RandomForestClassifier(n_estimators=num_trees, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            # Validate on labeled non-training (test) data
            # log.info(f'evaluating accuracy on {X_test.shape[0]} held-out labeled points')
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f'using {model_feat_names} for model {idx} with {num_trees} trees')
            log.info(f"RandomForest accuracy on hold-out labeled set: {acc:.3f}")
            log.info("Detailed classification report:\n" + classification_report(y_test, y_pred, target_names=list(label_groups.keys())))
            log.info('feature importances:\n' + str(rf.feature_importances_))
            with open(f'rf_model_{idx}_{num_trees}trees.pkl', 'wb') as f:
                pickle.dump(rf, f)

    for idx, model_feat_names in enumerate(model_feat_names_options):
        for num_trees in [201]:
            try:
                # Stack feature arrays for the labeled points: shape (num_features, num_labeled_points)
                stacked_feats = [smoothed_feats[f] for f in model_feat_names]
                # Stack to shape (num_features, N), then transpose after index
                all_feats = np.stack(stacked_feats, axis=1)
                unlabeled_feats = all_feats[unlabeled_idxs]
                with open(f'rf_model_{idx}_{num_trees}trees.pkl', 'rb') as f:
                        rf = pickle.load(f)
                # Predict groups for each unlabeled point
                log.info(f'predicting groups for {len(unlabeled_idxs)} unlabeled points')
                unlabeled_pred_labels = rf.predict(unlabeled_feats)

                # Organize point indices by predicted group
                predicted_group_idxs = {g:[] for g in label_groups.keys()}
                group_names_list = list(label_groups.keys())
                for idp, pred in zip(unlabeled_idxs, unlabeled_pred_labels):
                    predicted_group_idxs[group_names_list[pred]].append(idp)
                
                predicted_pcds = []
                for i, group_name in enumerate(label_groups.keys()):
                    if predicted_group_idxs[group_name]:
                        subset_pcd = pcd.select_by_index(predicted_group_idxs[group_name])
                        subset_pcd.paint_uniform_color(colors_palette[i % len(colors_palette)])
                        predicted_pcds.append(subset_pcd)
                draw(predicted_pcds)
                breakpoint()
            except Exception as e:
                breakpoint()
                print(f'error loading model {idx}_{num_trees}trees: {e}')
                continue   

    print("Labeled groups:", [f"{g}: {len(idxs)} pts" for g, idxs in grped_idxs.items()])
    print("Predicted unlabeled group sizes:", {g:len(v) for g,v in predicted_group_idxs.items()})

    # Visualize labeled + predicted groups
    draw(labeled_pcds + predicted_pcds)
    draw(predicted_pcds)
    breakpoint()


def draw_feats(file_name, pts, colors, 
                  feats_dict, smoothed_feats,
                   bnds = {'verticality':{'high':0.75, 'low':0.40}}):
    """
    Draw the features and colors in a table
    """
    from geometry.point_cloud_processing import cluster_plus
    print(f'drawing {file_name}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    from geometry.point_cloud_processing  import cluster_plus
    bnds = {}
    bnd_idxs = defaultdict(dict)

    for feat_name, smooth in smoothed_feats.items():
        color_continuous_map(pcd, feats_dict[feat_name])
        histogram(feats_dict[feat_name], feat_name)
        draw([pcd])

        color_continuous_map(pcd, smooth[feat_name])
        histogram(smooth[feat_name], feat_name)
        draw([pcd])
    
        if bnds.get(feat_name, None) is not None:
            upper_bound = bnds.get(feat_name).get('high')
            if upper_bound is not None:
                in_idxs = np.where(smooth>=upper_bound)[0]
                in_pcd = pcd.select_by_index(in_idxs)
                draw([in_pcd])
                bnd_idxs[feat_name]['high'] = in_idxs
            lower_bound = bnds.get(feat_name).get('low')
            if lower_bound is not None:
                in_idxs = np.where(smooth<=lower_bound)[0]
                in_pcd = pcd.select_by_index(in_idxs)
                draw([in_pcd])
                bnd_idxs[feat_name]['low'] = in_idxs

def color_in_slices(pcd, feat):
    points = np.array(pcd.points)
    z_vals = np.array([x[2] for x in points])
    slice_idxs = {}
    percentiles = [0, 20, 40, 60, 80, 100]
    # percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    z_percentile_edges = np.percentile(z_vals, percentiles)
    for i in range(len(percentiles)-1):
        z_lo = z_percentile_edges[i]
        z_hi = z_percentile_edges[i+1]
        # get indices in this z interval
        in_slice = np.where((z_vals >= z_lo) & (z_vals < z_hi))[0] if i < len(percentiles)-2 else np.where((z_vals >= z_lo) & (z_vals <= z_hi))[0]
        slice_idxs[f'slice_{percentiles[i]}_{percentiles[i+1]}'] = in_slice

    for slice_name, slice_idxs in slice_idxs.items():
        slice_pcd = pcd.select_by_index(slice_idxs)
        histogram(feat[slice_idxs], slice_name)
        color_continuous_map(slice_pcd, feat[slice_idxs])
        draw([slice_pcd])


if __name__ == '__main__':
    # Example usage of the new visualization function
    # print("=== SKIO Point Cloud Visualization ===")
    # visualize_skio_pointcloud(
    #     npz_file_path='/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/SKIO_voxpt05_all.npz',
    #     voxelized_file_path='/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/SKIO_voxpt05_int_smoothed_reduced5_voxpt05.npz',
    #     feature_to_visualize='verticality',
    #     use_colors=False,
    #     voxel_size=0.1  # Optional downsampling for better performance
    # )
    from geometry.point_cloud_processing import voxelize_and_trace, recover_from_trace
    # breakpoint()
    # Original functionality
    # input_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/ext_detail/'
    input_dir = '/media/penguaman/data/kevin_holden/orig/MyFavPop_Alone_kh.pcd'
    file_name_pattern = 'cropped_transformed_w_all_attrs_feats_labels.las'
    files = glob(f'{input_dir}/{file_name_pattern}')
    reduction_factor = 1
    feature_names =  ['verticality', 'linearity', 'planarity', 'surface_variation',  'omnivariance', 'eigenentropy', 'anisotropy', 'PCA1', 'PCA2',  'sphericity', 'eigenvalue_sum']
    # pcd = o3d.io.read_point_cloud('/media/penguaman/data/kevin_holden/orig/fave_cropped_kh_div100.pcd')
    import laspy
    fin_las = laspy.read('/media/penguaman/data/kevin_holden/orig/cropped_transformed_w_all_attrs_feats_labels.las')
    vox_pcd = voxelize_and_trace(np.array(fin_las.xyz),.01)
    fin_pcd = o3d.geometry.PointCloud()
    fin_pcd.points = o3d.utility.Vector3dVector(np.array(fin_las.xyz))
    color_in_slices(fin_pcd, fin_las['intensity'])
    breakpoint()
    draw([fin_pcd])

    colors = np.stack([fin_las['red'], fin_las['green'], fin_las['blue']]).T
    colors = colors / 65280.0
    fin_pcd.colors = o3d.utility.Vector3dVector(colors)


    vox_pcd = voxelize_and_trace(fin_pcd.points,.01)
    draw([vox_pcd])
    breakpoint()
    clean_pcd, idxs = vox_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=.15)
    draw([clean_pcd])
    draw([vox_pcd.select_by_index(idxs, invert=True)])

    # o3d.io.write_point_cloud(pcd,'/media/penguaman/data/kevin_holden/orig/fave_cropped_kh_div100.pcd')
    # breakpoint()
    vox_size = None #.02
    get_file_and_features( input_dir, 
                            file_name_pattern,
                            pcd = fin_pcd,
                            reduction_factor=reduction_factor,
                            feature_names=feature_names,
                            vox_size=vox_size)
    # breakpoint()