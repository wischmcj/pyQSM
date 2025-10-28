from collections import defaultdict
import numpy as np
import pickle
from glob import glob
import os
from open3d.ml.vis import o3d

from open3d.visualization import draw_geometries as draw
from scipy.spatial import cKDTree
from viz.viz_utils import color_continuous_map

ROT_STEP = 10
def draw_view(pcds=None,front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024]):
    o3d.visualization.draw_geometries(pcds)

data_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/'

decent_trees = [6, 9, 14, 15, 23, 34, 49, 68, 69, 151, 154, 157, 158, 163, 167, 188, 191, 192, 202,
                    220, 223, 236, 240,  241,
                    246, 283, 297,306,307, 377, 460, 490, 493, 512,
                    556, 669]
less_interesting_from_prev_run = [154, 460, 377,  297, 283, 246, 202, 192,158, 163, 191, 460, 669, 306]
best_clusters_ratings = {  '490': 'g', '512': 's', '283': 's', 
                            '154': 'b', '191': 's', '220': 'g', '15': 'g', 
                            '240': 'b', '223': 'g', '192': 'b', '307': 's',
                            '34': 'g', '241': 'g', '23': 'b', '669': 's', 
                            '556': 'g', '49': 'b', '163': 's', '188': 'g', 
                            '14': 'b', '158': 's', '157': 's', '297': 's', '9': 'g', 
                            '493': 'g', '236': 'g', '6': 'g', '377': 's', '246': 'b', 
                            '167': 's', '306': 's', '68': 'g', '69': 'b', '202': 's', '460': 's'}
best_clusters = [  490, 512, 283, 154, 191, 220, 15, 240, 223, 192, 307, 34, 241, 23, 669, 556, 49, 163, 188, 14, 158, 157, 297, 9, 493, 236, 6, 377, 246, 167, 306, 68, 69, 202, 460]
ok_clusters_ratings = {'157': 's', '163': 's', '191': 's', '307': 's', '377': 's', '669': 's',
                            '69': 'g', '23': 'b', '202': 's', '283': 's', '158': 's', 
                            '512': 'g', '49': 'g', '154': 's', '14': 'b', '246': 's', 
                            '167': 's', '460': 's', '306': 's', '220': 'g', '240': 'g', '297': 'g', '192': 's'}
ok_clusters = [157, 163, 191, 307, 377, 669, 69, 23, 202, 283, 158, 512, 49, 154, 14, 246, 167, 460, 306, 220, 240, 297, 192]
all_good_clusters = best_clusters+ ok_clusters
# small_clusters =[157, 163, 191, 307, 377, 669, 202, 283, 158, 154, 246, 167, 460, 306, 192, 512, 283, 191, 307, 669, 163, 158, 157, 297, 377, 167, 306, 202, 460]

skio_clusters = [109, 138, 132, 107, 111, 115, 193, 151, 135, 136, 190, 133, 191, 134, 189, 108, 116, 137, 112, 114]
approved_matches = dict([ (151, 68), (136, 9), (190, 220), (133, 68), (191, 236), (189, 236), (116, 493), (137, 34)])
replace_with_tl = dict([(132, 68), (193, 241)])

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

def load_kdtrees(label_list:list[int], case_name:str = ''):
    print(f'loading kdtrees')
    kdtrees = []
    if case_name != '':
        label=f'{label}_{case_name}'
    for label in label_list:
        with open(f'{data_dir}/cluster_joining/kdtree/{label}_kd_tree.pkl', 'rb') as f:
             kdtrees.append((label, pickle.load(f)))
    return kdtrees

def load_adjacency_dict(case_name = ''):
    print(f'loading adjacency dict')
    if case_name != '':
        file_name = f'adj_{case_name}.pkl'
    else:
        file_name = f'adj.pkl'
    with open(f'{data_dir}/cluster_joining/adjacency/{file_name}', 'rb') as f:
        adj = pickle.load(f)
    adj = {int(k1):{int(k2):v for k2,v in adj[k1].items()} for k1 in adj.keys()}
    return adj

def create_kdtrees(coords, labels, case_name = ''):
    print(f'creating kdtrees')
    kdtrees = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        print(f'creating kdtree for {label}')
        pt_sample = coords[labels == label][::10]
        kdtree = cKDTree(pt_sample)
        if case_name != '':
            label=f'{label}_{case_name}'
        with open(f'{data_dir}/cluster_joining/kdtree/{label}_kd_tree.pkl', 'wb') as f:
            pickle.dump(kdtree, f)
        kdtrees.append((label, kdtree))
    return kdtrees

def determine_cluster_adjacency(src_file:str, decent_trees:list[int], threshold:float=0.35, case_name = ''):

    try:
        adj = load_adjacency_dict(case_name)
        return adj
    except:
        print(f'no saved adjacency dict found, creating from scratch')

    # load or create kdtrees
    try:
        kdtrees = load_kdtrees(unique_labels, case_name)
    except:
        _, unique_labels, labels, coords = labeled_clusters_from_pw_results(src_file, return_src_arrays=True)
        # create from scratch
        kdtrees = create_kdtrees(coords, labels, case_name)
    
    # load or create adjacency dict
    try:
        adj = load_adjacency_dict(case_name)
    except:
        # create from scratch
        if coords is None or labels is None:
            _, unique_labels, labels, coords = labeled_clusters_from_pw_results(src_file, return_src_arrays=True)
        adj = determine_adjacency(decent_trees, kdtrees, threshold, case_name)
    return adj


def determine_adjacency(label_list, kdtrees=None, threshold=0.35, case_name = '', src_kdtrees=None):
    """
        Given clusters as a list of arrays of points (N x 3), determine adjacency
        Two clusters are adjacent if any points are within a small threshold distance
    """
    if src_kdtrees is None:
        src_kdtrees = kdtrees

    adjacency_dict = {label: {} for label in label_list}

    # pts_and_dists = [[tree_i.sparse_distance_matrix(tree_j, threshold, output_type='ndarray') for label_j, tree_j in kdtrees] for label_i, tree_i in src_kdtrees]
    # pts_and_dists = [[x['v'].min() for x in y if x.shape[0] > 0] for y in pts_and_dists]
    print(f'mapping clusters')
    for label_i, tree_i in src_kdtrees:
        if label_i in label_list:
            print(f'{label_i} in label_list, getting neighbors')
            print(f'mapping cluster {label_i}')
            for label_j, tree_j in kdtrees:
                if label_j== label_i or label_j in label_list or label_j == 0:
                    print(f'{label_j} invalid for adjacency check with {label_i}')
                    continue
                # Query cluster i against cluster j for close neighbors
                # For efficiency, only check shortest pairwise distance
                pts_and_dists = tree_i.sparse_distance_matrix(tree_j, threshold, output_type='ndarray')

                if pts_and_dists.shape[0] > 0:
                    min_dist = pts_and_dists['v'].min()
                    adjacency_dict[int(label_i)][int(label_j)] = min_dist 
                else:
                    print(f'no neighbors found for {label_i} and {label_j}: {pts_and_dists.shape=}')
    
    if case_name != '':
        file_name = f'adj_{case_name}.pkl'
    else:
        file_name = f'adj.pkl'

    with open(f'{data_dir}/cluster_joining/adjacency/{file_name}', 'wb') as f:
        pickle.dump(adjacency_dict, f)
    return adjacency_dict
    
def cluster_color(pcd,labels):
    import matplotlib.pyplot as plt
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    orig_colors = np.array(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, orig_colors

def get_clusters_joined_from_input_files(requested_labels:list[int]=[]):
    adj = load_adjacency_dict()
    path = f'{data_dir}/cluster_joining/inputs/*_inputs*.pkl'
    files = glob(path)
    clusters_joined = defaultdict(list)
    breakpoint()
    for file in files:
        key = os.path.basename(file).split('_')[0]
        if key in requested_labels:
            print(f'loading {file} for {key}')
            closest_list = adj.get(int(key))
            if closest_list is not None:
                inputs = np.load(file, allow_pickle=True)
                num_closest = len(inputs)
                adj_labels, dists = zip(*closest_list.items())
                low_dist_idxs = np.array(dists).argsort()[:num_closest]
                closest_lbls = np.array(adj_labels)[low_dist_idxs]
                for lbl, input_dict in zip(closest_lbls, inputs):
                    if input_dict['user_input'] in ['y', 'r']:
                        clusters_joined[key].append(lbl)
    return clusters_joined

EPS_DEFAULT = 1.2
MIN_POINTS_DEFAULT = 175
def user_cluster(pcd, src_pcd, eps=EPS_DEFAULT, min_points=MIN_POINTS_DEFAULT, filter_non_clusters=True, draw_result=True):
    print('clustering')
    choosing_inputs = True
    eps = EPS_DEFAULT
    min_points = MIN_POINTS_DEFAULT
    while choosing_inputs:
        user_input = input(f"eps? (default={eps})").strip().lower()
        try:
            eps = float(user_input or eps)
            if eps >3:
                print(f'eps too large, setting to {EPS_DEFAULT}')
                eps = EPS_DEFAULT
        except:
            print(f'error parsing min_points input, setting to {EPS_DEFAULT}')
            eps = EPS_DEFAULT
        eps = float(user_input or 1)

        user_input = input(f"min_points? (default={min_points})").strip().lower()
        try:
            min_points = int(user_input or min_points)
        except:
            print(f'error parsing min_points input, setting to {MIN_POINTS_DEFAULT}')
            min_points = MIN_POINTS_DEFAULT

        # Cluster and draw
        print(f'clustering with eps={eps} and min_points={min_points}')
        labels =  np.array( pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
        unique_labels = np.unique(labels)
        print(f'Number of clusters found: {len(unique_labels)}')
        non_cluster_idxs = np.where(labels < 0)[0]
        if len(labels) == 0:
            print(f'no clusters found, trying again')
            continue
        new_pcd, orig_colors = cluster_color(pcd,labels)
        new_pcd = new_pcd.select_by_index(non_cluster_idxs, invert=True)
        draw_view([new_pcd])
        draw_view([new_pcd, src_pcd])
        # Finalize
        user_input = 'draw_again'
        while user_input == 'draw_again':
            user_input = input("Accept? (default=draw_again)").strip().lower()
        if user_input == 'y':
            choosing_inputs = False
    return labels, eps, min_points

def loop_and_ask(src_lbl,src_pts, labeled_clusters, 
                save_file=False, return_chosen_pts=False,
                ignore_labels:list[int]=[],
                case_name:str=''):
    """
        src_pts (np.array[N,3]): points of the source cluster
        src_lbls (np.array[N]): labels of the source cluster
        labeled_clusters (list[tuple[int, np.array[N]]]): list of tuples of labels and points of the labeled clusters
    """
    src_lbls = [src_lbl]*len(src_pts)
    final_pts = [src_pts]
    final_lbls = [str(src_lbl)]
    pt_lens = [len(src_pts)]
    src_pcd = to_o3d(coords=src_pts,colors= [[0,0,1]]*len(src_pts))
    inputs = []
    for lbl, pts in labeled_clusters:
        if int(lbl) in ignore_labels:
            print(f'excluding cluster {lbl} from consideration')
            continue
        if len(pts)>50:
            if return_chosen_pts: 
                print(f'running algo for subcluster {lbl}')
            else:
                print(f'processing cluster {lbl}')
            lbls = [lbl]*len(pts)
            comp_pcd = to_o3d(coords=pts, colors= [[1,0,0]]*len(pts))
            draw_view([comp_pcd, src_pcd])
            user_input = input("Add points to_cluster? (y/n/r): ").strip().lower()
            in_dict = {'user_input': user_input}
            if user_input == 'q':
                print(f'quitting cluster join process')
                break
            if user_input == 's':
                print(f'skipping cluster {lbl}')
                continue
            if user_input == 'y':
                final_pts.append(pts)
                final_lbls.append(str(lbl))
            if user_input == 'r':
                print(f'recursing on cluster {lbl}')
                # split the cluster into subclusters
                sub_labels, eps, min_points = user_cluster(comp_pcd, src_pcd)
                in_dict['eps'] = eps
                in_dict['min_points'] = min_points
                unique_sub_labels = np.unique(sub_labels)
                labeled_subclusters = [(sub_lbl, pts[sub_labels == sub_lbl]) for sub_lbl in unique_sub_labels]
                # run algo to choose which subclusters to add to the final cluster
                chosen_pts_list, chosen_sub_lbls, inputs = loop_and_ask(src_lbl, src_pts, labeled_subclusters, return_chosen_pts=True, ignore_labels=[])
                in_dict['recurse_inputs'] = inputs
                # generate compound labels, add chosen to final
                for chosen_pts, chosen_sub_lbl in zip(chosen_pts_list, chosen_sub_lbls):
                    final_pts.append(chosen_pts)
                    for lbl in chosen_sub_lbl:
                        final_lbls.append(f'{lbl}_{chosen_sub_lbl}')
                    pt_lens.append(len(chosen_pts))
            inputs.append(in_dict)
    if return_chosen_pts:
        return final_pts[1:], final_lbls[1:], inputs
    if save_file:
        # save inputs from user
        try:
            file_name = f'{src_lbl}_inputs_3.pkl'
            if case_name != '':
                file_name = f'{case_name}/{src_lbl}_inputs_3.pkl'
            with open(f'{data_dir}/cluster_joining/inputs/{file_name}', 'wb') as f: pickle.dump(inputs, f)
        except Exception as e:
            breakpoint()
            print(f'failed to save inputs')
        # save the joined cluster
        try:
            pt_lens = np.array(pt_lens)
            final_pts = np.vstack(final_pts)
            final_lbls = np.array(final_lbls)
            pcd = to_o3d(coords=final_pts)
            draw_view([pcd])
            # file_name = '_'.join(final_lbls)'
            file_name = f'{src_lbl}_joined_3.npz'
            if case_name != '':
                file_name = f'{case_name}/{src_lbl}_joined_3.npz'
            np.savez_compressed(f'{data_dir}/cluster_joining/results/{file_name}', points=final_pts, labels=final_lbls, pt_lens=pt_lens)
        except:
            breakpoint()
            print(f'failed to save inputs')

def loop_and_rate():
    joined_not_categorized = [14 ,23 ,49 ,69 ,157 ,154 ,158 ,163 ,167 ,191 ,192 ,202 ,220 ,240 ,246 ,283 ,297 ,306 ,307 ,377 ,460 ,512 ,669 ,241 ,490 ,15 ,223 ,34 ,556 ,188 ,9 ,493 ,236 ,6 ,68]

    skio_base_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/results/'
    res = labeled_clusters_from_files(pattern='*joined.npz', 
                                        base_path=skio_base_path, 
                                        return_src_arrays=True,
                                        seed_pattern='.*/([0-9]{1,3})_joined.npz',
                                        )    
    tl_label_to_points, tl_unique_labels,_,_ = res
    tl_label_to_points = {int(label):points for label, points in tl_label_to_points.items()}
    tl_pcds = [to_o3d(coords=points) for label, points in tl_label_to_points.items() if int(label) in joined_not_categorized]
    
    ratings = {}
    for lbl in tl_unique_labels:
        coords = tl_label_to_points.get(int(lbl))
        pcd = to_o3d(coords=coords)
        draw_view([pcd])
        user_input = None
        while user_input is None:
            user_input = input("Good Cluster? (g/b/s): ").strip().lower()
            if user_input in ['g','b','s']:
                ratings[lbl] = user_input
            else:
                print("Invalid input. Please enter 'g', 'b', or 's'.")
                user_input = None
    print(ratings)
    return ratings

def labeled_clusters_from_pw_results(results_file:str,
                                    coords_file:str='coords',
                                    labels_file:str='instance_preds',
                                    return_src_arrays:bool=False):
    data = np.load(results_file)
    labels = data[labels_file]
    coords = data[coords_file]
    unique_labels = np.unique(labels)
    label_to_points = {label:coords[labels == label] for label in unique_labels}
    if return_src_arrays:
        return label_to_points, unique_labels, np.array(labels), np.array(coords)
    
    return label_to_points, unique_labels, None, None

import re
def labeled_clusters_from_files(pattern:str, 
                                base_path,
                                return_src_arrays:bool=False,
                                requested_labels:list[int]=[],
                                seed_pattern:str = '.*([0-9][0-9][0-9])_voxed_3.npz',
                                down_sample:int=True):
    files = glob(f'{base_path}/{pattern}')
    if len(files) == 0:
        print(f'no files found for {pattern} in {base_path}')
        return None, None, None, None
    unique_labels = []
    labeled_clusters = []
    all_labels = []
    all_coords = []
    for file in files:        
        label = re.match(re.compile(seed_pattern),file)
        print(f'loading {file}')
        if label is None:
            print(f'no seed found in {file}. Ignoring file...')
            continue

        label = label.groups(1)[0]
        unique_labels.append(label)
        if len(requested_labels) > 0 and int(label.groups(1)[0]) not in requested_labels:
            print(f'label {label.groups(1)[0]} not requested, skipping...')
            continue

        if '.pcd' in file:
            pcd = o3d.io.read_point_cloud(file)
            if down_sample:
                print(f'downsampling {file}')
                pcd = pcd.voxel_down_sample(voxel_size=.1)
                pcd = pcd.uniform_down_sample(3)
                coords = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                np.savez_compressed(f'{base_path}/ext_detail_voxed/{label}_voxed_3.npz', points=coords, colors=colors)
            coords = np.asarray(pcd.points)
        else:
            data = np.load(file)
            coords = data['points']


        labeled_clusters.append((label, coords))
        if return_src_arrays:
            all_labels.extend([label]*len(coords))
            all_coords.extend(coords)

    label_to_points = dict(labeled_clusters)
    if return_src_arrays:
        return label_to_points, unique_labels,  np.array(all_labels), np.array(all_coords)
    breakpoint()
    return label_to_points, unique_labels, None, None

def join_clusters(base_clusters:list[int],
                    num_closest:int=15,
                    threshold:float=0.35,
                    exclude_clusters:dict[str|int,list[int|str]]=[],
                    exclude_finished:bool=True,
                    alternative_base_cluster_pts:dict[str|int,np.array]={},
                    src_file:str='/media/penguaman/overflow/pointwise_results.npz'):

    base_clusters = [int(label) for label in base_clusters]
    exclude_clusters = {int(label): [int(lbl or int(label)) for lbl in exclude_labels] 
                            for label, exclude_labels in exclude_clusters.items()}
    # run_name = 'full_collective_diverse'
    # file = f'/media/penguaman/tosh2b/lidar_sync/adjacency_dict/{run_name}/pipeline/results/pointwise_results/pointwise_results.npz'
    label_to_points, _, _, _ = labeled_clusters_from_pw_results(src_file, return_src_arrays=False)
    pcd = to_o3d(coords=list(label_to_points.values())[460]) 

    adj = determine_cluster_adjacency(src_file, base_clusters, threshold)

    print(f'building nbrhood pcds')
    # Now select and draw the clusters for each cluster's five closest (and itself)
    results_files = glob(f'{data_dir}/cluster_joining/results/*.npz')
    finished = [os.path.basename(file).split('_')[0] for file in results_files]

    for label in base_clusters:
        if exclude_finished and str(label) in finished:
            print(f'excluding finished cluster {label}')
            continue

        if alternative_base_cluster_pts.get(label) is not None:
            src_pts = alternative_base_cluster_pts[label]
            print(f'using alternative base cluster points for {label}')
        else:
            src_pts=label_to_points[label]

        # Get the <num_closest> closest other clusters
        adj_labels, dists = zip(*adj[label].items())
        low_dist_idxs = np.array(dists).argsort()[:num_closest]
        closest_lbls = np.array(adj_labels)[low_dist_idxs]

        # Loop over the closest labels and allow the user
        #   to select all/part of each to include in the final cluster
        print(f'Expanding cluster {label}')
        close_pts = [(lbl, label_to_points[lbl]) for lbl in closest_lbls]
        loop_and_ask(label, 
                        src_pts, 
                        close_pts[1:], 
                        save_file=True,
                        ignore_labels=exclude_clusters[label])
        breakpoint()

def get_results(requested_labels:list[int]=[], 
                exclude_labels:list[int]=[],
                draw_results:bool=True):
    points_used = {}
    inputs_used = {}
    clusters_joined = {}
    results_notes = {}
    files = glob(f'{data_dir}/cluster_joining/results/*.npz')
    for file in files:
        base_cluster_label = os.path.basename(file).split('_')[0]

        if len(requested_labels) > 0 and base_cluster_label not in requested_labels:
            print(f'excluding {base_cluster_label} from results')
            continue
        if base_cluster_label in exclude_labels:
            print(f'excluding {base_cluster_label} from results')
            continue
        
        data = np.load(file)
        points = data['points']
        
        labels = data['labels']
        joined_source_clusters = [x.split('_')[0] for x in labels if x.split('_')[0] != '-']
        clusters_joined[int(base_cluster_label)] = joined_source_clusters
        points_used[int(base_cluster_label)] = points

        if draw_results:
            print(f'drawing results for {base_cluster_label}')
            points = data['points']
            pcd = to_o3d(coords=points)
            draw_view([pcd])
            user_input = input("Add notes: ").strip().lower()
            results_notes[base_cluster_label] = user_input

        try:
            with open(f'{data_dir}/cluster_joining/inputs/{base_cluster_label}_inputs.pkl', 'rb') as f: 
                inputs_used[base_cluster_label] = pickle.load(f)
        except:
            print(f'no inputs found for {base_cluster_label}')
            continue

    print(results_notes)
    return clusters_joined, results_notes, inputs_used, points_used
        
def join_collective_results(base_clusters:list[int]=[]):

    num_closest = 20
    threshold = 0.35
    # Get closest clusters to each tree cluster
    best_clusters = [label for label in best_clusters_ratings.keys()
                    #  if results_categories[label] == 'g'
                     if str(label) in ok_clusters_ratings]
    exclude_clusters = [label for label in best_clusters_ratings.keys() if label not in best_clusters]
    clusters_joined, results_notes, inputs_used, points_used = get_results(best_clusters, [], draw_results=False)
    
    join_clusters(list(map(int, best_clusters)), num_closest, 
                   threshold, exclude_clusters=clusters_joined, exclude_finished=False,
                   alternative_base_cluster_pts=points_used,
                   src_file='/media/penguaman/overflow/pointwise_results.npz')

    # draw_results(less_interesting_from_prev_run)

def compare_skio_clusters_to_tl_clusters(new_trees:bool=False):
    num_closest = 2
    best_clusters = [label for label in best_clusters_ratings.keys()]

    
    skio_base_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/ext_detail/ext_detail_voxed/'
    res = labeled_clusters_from_files(pattern='*.npz', 
                                        base_path=skio_base_path, 
                                        return_src_arrays=True,
                                         seed_pattern = '.*([0-9][0-9][0-9])_voxed_3.npz',
                                        )
    skio_label_to_points, skio_unique_labels, skio_labels, skio_coords = res
    skio_label_to_points = {int(label):points for label, points in skio_label_to_points.items()}
    skio_unique_labels = [int(label) for label in skio_unique_labels]

    if new_trees:
        tl_src_file = '/media/penguaman/overflow/pointwise_results.npz'
        tl_label_to_points, tl_unique_labels,tl_labels, tl_coords = labeled_clusters_from_pw_results(tl_src_file, return_src_arrays=True)
        tl_kdtrees = load_kdtrees(decent_trees)
        collective_kdtrees = load_kdtrees(skio_unique_labels)
        adj = determine_adjacency(skio_unique_labels, tl_kdtrees, case_name='skio', src_kdtrees=collective_kdtrees)
        # collective_kdtrees = [(label.replace('_skio', ''), kdtree) for label, kdtree in collective_kdtrees]
    else:
        adj = load_adjacency_dict(case_name='skio')
        tl_base_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/results/'
        res = labeled_clusters_from_files(pattern='*joined.npz', 
                                            base_path=tl_base_path, 
                                            return_src_arrays=True,
                                            seed_pattern='.*/([0-9]{1,3})_joined.npz',
                                            )    
        tl_label_to_points, tl_unique_labels,_,_ = res
        res = labeled_clusters_from_files(pattern='*joined_2.npz', 
                                            base_path=tl_base_path, 
                                            return_src_arrays=True,
                                            seed_pattern='.*/([0-9]{1,3})_joined_2.npz',
                                            )   
        tl_label_to_points_2, tl_unique_labels_2,_,_ = res 
        tl_label_to_points.update(tl_label_to_points_2)
        tl_unique_labels.extend(tl_unique_labels_2)
    
    # for k,v in collective_kdtrees:
    #     if 'skio' in k:
    #         collective_kdtrees.remove((k,v))

    # collective_kdtrees = load_kdtrees(skio_unique_labels)
    covered_labels = []
    # adj = {int(k1):{int(k2):v for k2,v in adj[k1].items()} for k1 in skio_unique_labels if 'skio' not in str(k1)}
    skio_unique_labels = [int(label) for label in skio_unique_labels]
    tl_unique_labels = [int(label) for label in tl_unique_labels]
    tl_label_to_points = {int(label):points for label, points in tl_label_to_points.items()}
    
    good_matches =[]
    replace_with_tl =[] 
    for label in skio_unique_labels:
        try:
            src_pcd = to_o3d(coords=skio_label_to_points[label])
            src_pcd.paint_uniform_color([1,0,0])
            # Get the <num_closest> closest other clusters
            closest_list = adj.get(label)
            if closest_list is None or closest_list == {}:
                print(f'{label} has no close clusters')
                continue
            adj_labels, dists = zip(*closest_list.items())
            low_dist_idxs = np.array(dists).argsort()[:num_closest]
            closest_lbls = np.array(adj_labels)[low_dist_idxs]
            print(f'{label} is closest to {closest_lbls}')

            for lbl in closest_lbls:
                covered_labels.append(lbl)
                coords = tl_label_to_points.get(lbl) 
                if coords is None:
                    coords = tl_label_to_points.get(int(lbl))
                if coords is None:
                    print(f'{lbl} not found in tl_label_to_points')
                    continue
                pcd = to_o3d(coords=coords)
                draw_view([src_pcd, pcd])

                user_input = None
                while user_input is None:
                    user_input = input("Good Cluster? (g/r/n): ").strip().lower()
                    if user_input == 'g':
                        good_matches.append((label, lbl))
                    elif user_input == 'r':
                        replace_with_tl.append((label, lbl))
                    elif user_input == 'n':
                        break
                    else:
                        print("Invalid input. Please enter 'g', 'r', or 'n'.")
                        user_input = None
                if user_input !='n':
                    break
                        
        except Exception as e:
            breakpoint()
            print(f'error drawing {label}: {e}')
            continue
    print(good_matches)
    print(replace_with_tl)
    breakpoint()
    print(f'covered labels: {covered_labels}')
    print(f'best clusters: {best_clusters}')
    not_covered_pcds = []
    not_covered_lbls = [label for label in best_clusters if label not in covered_labels]
    # ['490', '512', '283', '154', '191', '220', '15', '240', '223', '192', '307', '34', '241', '23', '669', '556', '49', '163', '188', '14', '158', '157', '297', '9', '493', '236', '6', '377', '246', '167', '306', '68', '69', '202', '460']
    not_covered_pts = [tl_label_to_points.get(int(label)) for label in not_covered_lbls]
    not_covered_pcds = [ to_o3d(coords=pts) for pts in not_covered_pts if pts is not None]

    no_close_lbls = [114, 115, 112, 138, 109]
    no_close_pcds = [ to_o3d(coords=skio_label_to_points[label]) for label in no_close_lbls]
    breakpoint()

def generate_combined_clusters():
    print('generating combined clusters')
    skio_base_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/ext_detail'
    files = glob(f'{skio_base_path}/full_ext_*')
    skio_unique_labels = []
    skio_label_to_file = {}
    for file in files:        
        label = re.match(re.compile('.*seed([0-9][0-9][0-9])'),file)
        if label is None:
            print(f'no seed found in {file}. Ignoring file...')
            continue
        label = label.groups(1)[0]
        skio_unique_labels.append(label)
        skio_label_to_file[int(label)] = file
    # res = labeled_clusters_from_files(pattern='full_ext_*.pcd', 
    #                                     base_path=skio_base_path, 
    #                                     return_src_arrays=True,
    #                                     down_sample=False,
    #                                     seed_pattern = '.*seed([0-9][0-9][0-9])',
    #                                     )
    # skio_label_to_points, skio_unique_labels, skio_labels, skio_coords = res
    # skio_label_to_points = {int(label):points for label, points in skio_label_to_points.items()}
    skio_unique_labels = [int(label) for label in skio_unique_labels]

    tl_base_path = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/results/'
    res = labeled_clusters_from_files(pattern='*joined.npz', 
                                        base_path=tl_base_path, 
                                        return_src_arrays=True,
                                        seed_pattern='.*/([0-9]{1,3})_joined.npz',
                                        )    
    tl_label_to_points, tl_unique_labels,_,_ = res
    res = labeled_clusters_from_files(pattern='*joined_2.npz', 
                                        base_path=tl_base_path, 
                                        return_src_arrays=True,
                                        seed_pattern='.*/([0-9]{1,3})_joined_2.npz',
                                        )   
    tl_label_to_points_2, tl_unique_labels_2,_,_ = res 
    tl_label_to_points.update(tl_label_to_points_2)
    tl_unique_labels.extend(tl_unique_labels_2)
    tl_label_to_points = {int(label):points for label, points in tl_label_to_points.items()}
    breakpoint()
    covered_tl_labels = []
    rejected_labels = []
    for label in skio_unique_labels:
        tl_lbl = 0
        match_type = 'none'
        print(f'processing {label}')
        skio_file = skio_label_to_file[label]
        print(f'processing {skio_file}')
        skio_coords = np.asarray(o3d.io.read_point_cloud(skio_file).points) - np.array([113.42908253, 369.58605156,   4.60074394])
        to_join_lbl = approved_matches.get(label)
        to_replace_lbl = replace_with_tl.get(label)
        if approved_matches.get(label) is not None:
            covered_tl_labels.append(to_join_lbl)
            all_coords = np.vstack([skio_coords, tl_label_to_points[to_join_lbl]])
            tl_lbl = to_join_lbl
            match_type = 'joined'
        elif replace_with_tl.get(label) is not None:
            covered_tl_labels.append(to_replace_lbl)
            all_coords = tl_label_to_points[to_replace_lbl]
            tl_lbl = to_replace_lbl
            match_type = 'replaced'
        else:
            all_coords = skio_coords

        if tl_lbl !=0:
            src_pcd = to_o3d(coords=skio_coords)
            src_pcd.paint_uniform_color([1,0,0])
            pcd = to_o3d(coords=tl_label_to_points[tl_lbl])
            draw_view([src_pcd, pcd])
            user_input = input(f"Reject {match_type}? (r)").strip().lower()
            if user_input =='r':
                rejected_labels.append((label, match_type))
                continue
        print(f'saving {file_code}')
        file_code = f'skio_{label}_tl_{tl_lbl}_joined_replaced.npz'
        np.savez_compressed(f'{data_dir}/cluster_joining/skio_tl_joins/{file_code}', points=all_coords)

    for lbl, pts in tl_label_to_points.items():
        if lbl not in covered_tl_labels and lbl in all_good_clusters:
            file_code = f'skio_0_tl_{lbl}_not_covered.npz'
            np.savez_compressed(f'{data_dir}/cluster_joining/skio_tl_joins/{file_code}', points=pts)
    print(rejected_labels)


if __name__ == '__main__':

    generate_combined_clusters()
    breakpoint()
    # new_trees = False
    # compare_skio_clusters_to_tl_clusters(new_trees)
    # breakpoint()    