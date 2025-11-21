import os
import numpy as np
from glob import glob
import open3d as o3d
from jakteristics import compute_features as compute_features_j
from set_config import log
from tqdm import tqdm
from viz.viz_utils import color_continuous_map

# from geometry.point_cloud_processing import ( 
#     cluster_plus,
#     crop_by_percentile,
#     get_percentile,
# )
# from utils.tiles import voxelize
from pyQSM.viz.viz_utils import draw
# from tensor_board_dec import tensorboard_summary

def replace_nanfeatures(features):
    ind_nan = np.isnan(features)
    mean_values = np.nanmean(features, axis=0)
    print(f"There are {ind_nan.sum()} nan features in the whole forest. Replacing them with mean feature values")

    for i in range(features.shape[1]):
        if ind_nan[:, i].sum() > 0:
            features[:, i][ind_nan[:, i]] = mean_values[i]
    print(f"After replacement there are {np.isnan(features).sum()} nan features in the whole forest.")
    return features

def compute_features(points, search_radius=0.6, feature_names=['verticality'], num_threads=4):
    #eigenvalue_sum, Omnivariance, Eigenentropy, Anisotropy,Planarity,Linearity, PCA1,PCA2,Surface Variation,Sphericity,Verticality
    print(f'Computing features for {len(points)} points')
    features = compute_features_j(points, search_radius=search_radius, num_threads=num_threads, feature_names=feature_names)
    features = replace_nanfeatures(features)
    features = features.astype(np.float32)
    return features

def check_files_for_feature(feature_names, input_dir, detail_file_name, reduction_factor):
    use_master_feats = False
    feats = {}
    to_calc = []
    for feature_name in feature_names:
        print(f' looking for  {feature_name}')
        master_feats_file = f'{input_dir}/features/{detail_file_name}_reduced_1_{feature_name}.npz'
        instance_feats_file = f'{input_dir}/features/{detail_file_name}_reduced_{reduction_factor}_{feature_name}.npz'
        feats_file =''
        if os.path.exists(master_feats_file):
            feats_file = master_feats_file
            use_master_feats = True
        elif os.path.exists(instance_feats_file):
            feats_file = instance_feats_file
        
        if feats_file != '':
            print(f'{feats_file} exists')
            feats_data = np.load(feats_file)
            if feature_name in feats_data.keys():
                if use_master_feats:
                    feats[feature_name] = feats_data[feature_name][::reduction_factor]
                else:
                    feats[feature_name] = feats_data[feature_name]
            else:
                print(f'{feature_name} not found in {feats_file}')
                to_calc.append(feature_name)
        else:
            print(f'no feats file found for {feature_name}')
            to_calc.append(feature_name)
    return feats, to_calc

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

def get_file_and_features(input_dir,file_name_pattern = None 
                                ,pcd = None
                                ,reduction_factor=1
                                ,feature_names = ['planarity','linearity','verticality', 'surface_variation']
                                ,vox_size=.01):
    """
    Get the file and features for the given input directory and file name snippet
    Available features:
        Eigenvalue sum, Omnidatavariance, Eigenentropy, 
        Anisotropy,Planarity,Linearity, PCA1,PCA2, 
        Surface Variation,Sphericity,Verticality
            
    Args:
        input_dir: The directory to search for files
        file_name_snip: The snippet to search for in the file names
        reduction_factor: The reduction factor to apply to the features
        feature_names: The features to compute
        vox_size: The voxel size to use
    """
    files = glob(f'{input_dir}/{file_name_pattern}')
    
    file_names = []  # Initialize the missing variable
    detail_file_name=''
    all_pts = []
    all_colors=[]
    all_feat_values = []
    breakpoint()
    for file in files:
        print(f' inspecting {file}')
        detail_file_name = os.path.basename(file).split('.')[0]
        if pcd is None:
            pts, colors = voxelize_pcd(file, vox_size)
        else:
            pts = np.array(pcd.points)
            colors = np.array(pcd.colors)
        feat_values, to_calc = check_files_for_feature(feature_names, input_dir, detail_file_name, reduction_factor)
        
        if len(to_calc) > 0:
            print(f'{detail_file_name} needs to calculate {to_calc}')
            all_feats = compute_features(pts, feature_names=to_calc)
            for feature_name, feats in zip(to_calc, all_feats.T):
                feats_file = f'{input_dir}/features/{detail_file_name}_reduced_{reduction_factor}_{feature_name}.npz'
                save_dict = {feature_name: feats}
                np.savez_compressed(feats_file, **save_dict)
                print(f'{feats_file} saved')
        draw_feats_tb(detail_file_name, pts, colors, feat_values)
        # all_pts.append(pts)
        # all_colors.append(colors)
        # all_feat_values.append(feat_values)
        # file_names.append(detail_file_name
    # all_pts = np.vstack(all_pts)
    # all_colors = np.vstack(all_colors)
    # breakpoint()
    # all_feat_values = np.vstack(all_feat_values)
    draw_feats_tb(detail_file_name, all_pts, all_colors, feat_values)
    # new_data = {'points': all_pts, 'colors': all_colors, 'intensity': feat_vals}
    # np.savez_compressed(f'/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/features/EpiphytusTV4color_int_all_reduced_1.npz', **new_data)
    # return file_names, all_pts, all_colors, all_feats

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

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ex: 
    in_cond = bounding_box(pts, min_x=ll[0], max_x=ur[0], min_y=ll[1], max_y=ur[1], min_z=ll[2], max_z=ur[2])
    """
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter

def filter_by_bb(pcd, exclude_boundaries, reverse_filter):
    filtered_chunk_data= []
    points = np.array(pcd.points)
    new_chunk_data=points
    if exclude_boundaries is not None:
        for boundary in exclude_boundaries:
            (x_min, y_min, z_min), (x_max, y_max, z_max) = boundary
            print('getting mask for boundary')
            mask = bounding_box(points, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max, min_z=z_min, max_z=z_max)
            
            print(f'filtering')
            if len(new_chunk_data) > 0:
                if reverse_filter: # exclude all points within the boundaries
                    new_chunk_data = new_chunk_data[~mask]
                else: # only include points within the boundaries (the union, if multiple)
                    filtered_chunk_data.append(new_chunk_data[mask])

    if not reverse_filter:
        new_chunk_data = np.vstack(filtered_chunk_data)
    
    return new_chunk_data

def draw_feats_tb(file_name, pts, colors, feats_dict):
    """
    Draw the features and colors in a table
    """
    print(f'drawing {file_name}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # draw([pcd])
    for feat_name, feats  in feats_dict.items():
        print(f'drawing {feat_name}')
        color_continuous_map(pcd, feats)
        draw([pcd])
    # prcentile_cutoff = 90
    # # feat = feats['linearity'] 
    # feat = feats['verticality']
    # # feat = feats['planarity']
    
    # split_resuts = split_by_percentile(pts, feat, percentile_cutoff, colors =colors,low=True,high=True)
    # low_pts, low_colors,low_indices, top_pts, top_colors, top_indices = split_resuts
    # top_pcd = to_o3d(coords = top_pts, colors = top_colors)
    # clean_top_pcd, clean_top_idxs = top_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=.25)
    # # clean_top_orig_idxs = top_indices[clean_top_idxs]
    # draw([clean_top_pcd])
    # breakpoint()


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


    # file = '/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/features/EpiphytusTV4color_int_all_reduced_1.npz'
    # data = np.load(file)
    # points = data['points']
    # colors = data['colors']/255.0
    # intensity = data['intensity']
    # files = ['/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/features/EpiphytusTV4color_int_0_reduced_1_verticality.npz', '/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/features/EpiphytusTV4color_int_2_reduced_1_verticality.npz','/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/features/EpiphytusTV4color_int_1_reduced_1_verticality.npz']
    # vert = np.hstack([np.load(file)['verticality'] for file in files])
    # draw_feats_tb(file.split('/')[-1], points, colors, {'verticality': vert})
    from canopy_metrics import histogram, smooth_feature
    pcd = o3d.io.read_point_cloud('/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/EpiphytusTV4_treeiso_statdown.pcd')
    # linearity gives a good separation of branches
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)
    feature_names = ['planarity', 'linearity','surface_variation', 'verticality']
    for feature in feature_names:
        feat = np.load(f'/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/features/EpiphytusTV4_treeiso_statdown_reduced_1_{feature}.npz')[feature]
        feat = smooth_feature(points, feat, query_pts=points, n_nbrs=50)
        draw_feats_tb(feature, points, colors, {feature: feat})
    breakpoint()
    in_idxs = np.where(feat>=.8)[0]
    draw_feats_tb('test', points[in_idxs], colors[in_idxs], {'linearity': feat[in_idxs]})
    breakpoint()

    in_idxs = np.where(feat<=0.4)[0]
    draw_feats_tb('test', points[in_idxs], np.zeros_like(points)[in_idxs], {'verticality': feat[in_idxs]})

    # breakpoint()
    # Original functionality
    # input_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/ext_detail/'
    input_dir = '/media/penguaman/backupSpace/lidar_sync/tls_lidar/MonteVerde/'
    file_name_pattern = 'EpiphytusTV4_treeiso_statdown.pcd'
    reduction_factor = 1
    feature_names = ['planarity', 'linearity','surface_variation', 'verticality']
    vox_size = None #.02
    get_file_and_features( input_dir, 
                            file_name_pattern,
                            reduction_factor=reduction_factor,
                            feature_names=feature_names,
                            vox_size=vox_size)
    # breakpoint()