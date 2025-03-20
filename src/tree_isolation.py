
import open3d as o3d
import scipy.spatial as sps
import numpy as np
from numpy import asarray as arr
import pickle
from copy import deepcopy
import itertools


from collections import defaultdict
import logging
from itertools import chain

import open3d as o3d
import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt
\
from open3d.io import read_point_cloud, write_point_cloud

from set_config import config, log
from reconstruction import recover_original_details, get_neighbors_kdtree, get_pcd
from utils.math_utils import (
    get_center,
    generate_grid
)
from utils.fit import kmeans,cluster_DBSCAN
from geometry.skeletonize import extract_skeleton, extract_topology, contraction_mag_processing
from geometry.mesh_processing import define_conn_comps, get_surface_clusters, map_density
from geometry.point_cloud_processing import ( filter_by_norm,
    clean_cloud,
    crop, get_shape,
    orientation_from_norms,
    filter_by_norm,
    get_ball_mesh,
    crop_by_percentile,
    cluster_plus
)
from utils.io import save,load,update
from viz.viz_utils import iter_draw, color_continuous_map, draw
from viz.color import *

def step_through_and_evaluate_clusters():
    # file = "cell3_complete2.pkl"
    file = "completed_cluster_idxs.pkl"

    idxs, pts = load_completed(0, [file])
    # lowc = o3d.io.read_point_cloud('low_cloud_all_16-18pct.pcd')
    # highc= o3d.io.read_point_cloud('collective_highc_18plus.pcd')
    collective= o3d.io.read_point_cloud('collective.pcd')
    clusters = create_one_or_many_pcds(pts)

    breakpoint()
    # draw(clusters)
    good_clusters = [] 
    multi_clusters = [] 
    partial_clusters = []
    bad_clusters = []
    factor = 5
    for idc, cluster in zip(idxs,clusters):
        is_good = 0
        is_multi = 0
        partial = 0
        region = [( cluster.get_min_bound()[0]-factor,
                         cluster.get_min_bound()[1]-factor),     
                        (cluster.get_max_bound()[0]+factor,
                        cluster.get_max_bound()[1]+factor)]
        zoomed_col = zoom_pcd(region, collective)
        # cluster.paint_uniform_color([1,0,0])
        draw([zoomed_col,cluster])

        # tree = sps.KDTree(np.asarray(arr(zoomed_col.points)))
        # dists,nbrs = tree.query(curr_pts[idx],k=750,distance_upper_bound= .3) #max(cluster_extent))
        # nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
        # nbr_pts = [nbr_pt for nbr_pt in highc_tree.data[nbrs]]      
        breakpoint()
        if is_good ==1:
            good_clusters.append((file,idc))
        elif is_multi==1:
            multi_clusters.append((file,idc))
        elif partial==1:
            partial_clusters.append((file,idc))
        else: 
            bad_clusters.append((file,idc))
    try:
        update('good_clusters.pkl',good_clusters)
        update('partial_clusters.pkl',partial_clusters)
        update('bad_clusters.pkl',bad_clusters)
        update('multi_clusters.pkl',multi_clusters)
    except Exception as e:
        breakpoint()
        print('failed writing to files')

def join_pcds(pcds):
    pts = [arr(pcd.points) for pcd in pcds]
    colors = [arr(pcd.colors) for pcd in pcds]
    return create_one_or_many_pcds(pts, colors, single_pcd=True)

def create_one_or_many_pcds( pts,
                        colors = None,
                        labels = None,
                        single_pcd = False):    
    log.info('creating pcds from points')
    pcds = []
    tree_pts= []
    tree_color=[]
    if not isinstance(pts[0],list) and not isinstance(pts[0],np.ndarray):
        pts = [pts]
    if not labels:
        labels = np.asarray([idx for idx,_ in enumerate(pts)])
    if (not colors):
        labels = arr(labels)
        max_label = labels.max()
        try:
            label_colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        except Exception as e:
            print('err')

    # for pcd, color in zip(tree_pcds, colors): pcd.paint_uniform_color(color[:3])
    for pts, color in zip(pts, colors or label_colors): 
        if not colors: 
            # if true, then colors were generated from labels
            color = [color[:3]]*len(pts)
        if single_pcd:
            log.info(f'adding {len(pts)} points to final set')
            tree_pts.extend(pts)
            tree_color.extend(color)
        else:
            cols = [color[:3]]*len(pts)
            log.info(f'creating pcd with {len(pts)} points')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcds.append(pcd)
    if single_pcd:
        log.info(f'combining {len(tree_pts)} points into final pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tree_pts)
        pcd.colors = o3d.utility.Vector3dVector([x for x in tree_color])
        pcds.append(pcd)
    return pcds

def filter_pcd_list(pcds,
                    max_pctile=85,
                    min_pctile = 30):
    pts = [arr(pcd.points) for pcd in pcds]
    colors = [arr(pcd.colors) for pcd in pcds]

    cluster_sizes = np.array([len(x) for x in pts])
    large_cutoff = np.percentile(cluster_sizes,max_pctile)
    small_cutoff = np.percentile(cluster_sizes,min_pctile)

    log.info(f'isolating clusters between {small_cutoff} and {large_cutoff} points')
    to_keep_cluster_ids  = np.where(
                            np.logical_and(cluster_sizes< large_cutoff,
                                                cluster_sizes> small_cutoff)
                            )[0]
    return  pcds[to_keep_cluster_ids]
    # for idc in small_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])

def zoom_pcd(zoom_region,
            pcd, 
            reverse=False):
    pts = arr(pcd.points)
    colors = arr(pcd.colors)
    in_pts,in_colors = zoom(zoom_region, pts, colors,reverse)
    in_pcd = o3d.geometry.PointCloud()
    in_pcd.points = o3d.utility.Vector3dVector(in_pts)
    in_pcd.colors = o3d.utility.Vector3dVector(in_colors)
    return in_pcd

def zoom(zoom_region, #=[(x_min,y_min), (x_max,y_max)],
        pts,
        colors = None,
        reverse=False):
    "Returns points in pts that fall within the given region"
    low_bnd = arr(zoom_region)[0,:]
    up_bnd =arr(zoom_region)[1,:]
    if isinstance(pts, list): pts = arr(pts)
    # breakpoint()
    if len(up_bnd)==2:
        up_bnd = np.append(up_bnd, max(pts[:,2]))
        low_bnd = np.append(low_bnd, min(pts[:,2]))

    inidx = np.all(np.logical_and(low_bnd <= pts, pts <= up_bnd), axis=1)   
    # print(f'{max(pts)},{min(pts)}')
    in_pts = pts[inidx]
    in_colors= None
    if colors is not None : in_colors = colors[inidx]   
    
    if reverse:
        out_pts = pts[np.logical_not(inidx)]
        in_pts = out_pts
        if colors is not None: in_colors = colors[np.logical_not(inidx)]

    return in_pts, in_colors

def filter_list_to_region(ids_and_pts,
                        zoom_region):
    in_ids = []
    for id_w_pts in ids_and_pts:
        idc, pts = id_w_pts
        new_pts, _ = zoom(zoom_region,pts,reverse = False )
        if len(new_pts) == len(pts): 
            in_ids.append(idc)
    return  in_ids

def filter_to_region_pcds(clusters,
                        zoom_region):
    """
        for each cluster in the list, remove points falling outside of the given region
    """
    pts = [tuple((idc,np.asarray(cluster.points))) for idc, cluster in clusters]
    new_idcs = filter_list_to_region(pts,zoom_region)
    new_clusters = [(idc, cluster) for idc, cluster in clusters if idc in new_idcs]
    return  new_clusters

def load_clusters_get_details():
    # For loading results of KNN loop
    pfile = 'data/in_process/cluster126_fact10_0to50.pkl'
    with open(pfile,'rb') as f:
        tree_clusters = dict(pickle.load(f))
    labels = [x for x in  tree_clusters.keys()]
    pts =[x for x in tree_clusters.values()]

    cluster_pcds = create_one_or_many_pcds(pts, labels = labels)
    recover_original_detail(cluster_pcds, file_num_base = 20000000)
    breakpoint()

def labeled_pts_to_lists(labeled_pts,
                            idc_to_label_map={},
                            file='',
                            draw_cycle=False,
                            save_cycle=False
                         ):
    tree_pts = defaultdict(list)
    for pt,idc in labeled_pts.items(): #tree_pts[idc_to_label_map.get(idc,idc)].append(pt)
        label =idc_to_label_map.get(idc,idc)
        tree_pts[label].append(pt)
    tree_pcds = None
    pt_lists = list([tuple((label,pt_list)) for label,pt_list in  tree_pts.items() ])
    if draw_cycle:
        labels = list([x for x in range(len(tree_pts))])
        pts = list([[y for y in x] for x in tree_pts.values()])
        tree_pcds = create_one_or_many_pcds(pts = pts, labels = labels)
    if save_cycle:
        print(f'Pickling {file}...')
        save(file, pt_lists)
    return pt_lists, tree_pcds


def extend_seed_clusters(clusters_and_idxs:list[tuple],
                            src_pcd,
                            file_label,
                            k=200,
                            max_distance=.3,
                            cycles= 150,
                            save_every = 10,
                            draw_every = 10):
    """
        Takes a list of tuples defining clusters (e.g.[(label,cluster_pcd)...])
        and a source pcd
    """
    cluster_pts = [(idc, arr(cluster.points)) for idc,cluster in clusters_and_idxs]


    # create dict to track point labels
    high_c_pt_assns = defaultdict(lambda:-1) 
    curr_pts = [[]]*len(clusters_and_idxs)
    curr_nbrs = [[]]*len(clusters_and_idxs)
    all_nbrs = [[]]*len(clusters_and_idxs)
    idc_to_label =  defaultdict() 
    label_to_idc =  defaultdict() 
    for idc, (label, cluster_pt_list) in enumerate(cluster_pts):
        curr_pts[idc] = cluster_pt_list
        # label pts already in clusters
        for pt in cluster_pt_list: 
            high_c_pt_assns[tuple(pt)] = idc
            # BElow is how this is done if we also send order through
            # high_c_pt_assns[tuple(pt)] = (idc,0)
        idc_to_label[idc] = label
        label_to_idc[label] = idc

    try: 
        src_tree = load(f'{file_label}_kd_search_tree.pkl')
    except Exception as e:
        print('creating KD search tree')
        src_pts = arr(src_pcd.points)
        src_tree = sps.KDTree(src_pts)
        with open(f'{file_label}_kd_search_tree.pkl','wb') as f: pickle.dump(src_tree,f)
    
    src_pts = src_tree.data

    file = f'{file_label}_in_process.pkl'
    num_pts =  len(src_pts)
    complete = []
    max_orders = defaultdict(int)
    save_iters = save_every
    draw_iters = draw_every
    end_early =False
    check_overlap = True
    # all_nbrs= []
    for cycle_num in range(cycles):
        print('start iter')
        draw_cycle,save_cycle = draw_iters<=0, save_iters<=0 
        if draw_cycle or save_cycle:
            pt_lists, tree_pcds = labeled_pts_to_lists(high_c_pt_assns,
                                                        idc_to_label,
                                                        file,
                                                        draw_cycle,
                                                        save_cycle)
            if save_cycle: save_iters = save_every
            if draw_cycle: 
                draw_iters = draw_every
                draw(tree_pcds)
            breakpoint()
            del tree_pts

        if end_early:
            break
        save_iters=save_iters-1
        draw_iters=draw_iters-1
        print(f'querying {cycle_num}')
        for label, cluster in clusters_and_idxs:
            idx = label_to_idc[label]
            if idx not in complete:               
                if len(curr_pts[idx])>0:
                    # get neighbors in the vicinity of the cluster
                    dists,nbrs = src_tree.query(curr_pts[idx],
                                                k=k,
                                                distance_upper_bound= max_distance)
                    nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
                    nbr_pts = [nbr_pt for nbr_pt in src_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                    
                    if check_overlap:
                        nbr_set = set(nbrs)
                        # overlap = chain.from_iterable([nbr_set.intersection(nbr_list) for nbr_list in all_nbrs])
                        overlaps = [(idnl,nbr_set.intersection(nbr_list)) for idnl,nbr_list in enumerate(all_nbrs)]
                        overlaps = [x for x in overlaps if len(x)>0]
                        if len(overlaps)>0: 
                            pt_lists, tree_pcds = labeled_pts_to_lists(high_c_pt_assns,idc_to_label,draw_cycle=True)
                        for overlap in overlaps:
                            overlap_pcd = src_pcd.select_by_index(overlap)
                            overlap_pcd.paint_uniform_color([1,0,1])
                            draw([overlap_pcd]+tree_pcds)
                            breakpoint()
                            print('breakpoint in overlap draw')
                        # label_to_clusters = cluster_plus(nbr_pcd,eps=.3, min_points=5,return_pcds=False)
                    # order = cycle_num
                    if len(nbr_pts)>0:
                        for nbr_pt in nbr_pts: 
                            high_c_pt_assns[tuple(nbr_pt)] = idx
                        # if max_orders[idx]< order: max_orders[idx] = order
                        curr_pts[idx] = nbr_pts
                        curr_nbrs[idx] = nbrs
                        all_nbrs[idx].extend(nbrs)
                        if len(curr_pts[idx])<5:
                            complete.append(idx)
                            print(f'{idx} added to complete')

                if len(curr_pts[idx])==0:
                    complete.append(idx)
                    print(f'{idx} added to complete')
    
    print('finish!')
    try:
        pt_lists, tree_pcds = labeled_pts_to_lists(high_c_pt_assns,
                                                        idc_to_label,
                                                        file,
                                                        draw_cycle=True,
                                                        save_cycle=True)
        breakpoint()
        return tree_pcds
    except Exception as e:
        breakpoint()
        print(f'error saving {e}')
        return None

def id_trunk_bases(pcd =None, 
            exclude_boundaries = None,
            low_percentiles = (16,18),
            high_percentiles = (18,100),
            file_name_base='',
            save_files = False,
            **kwargs
):
    """Splits input cloud into a 'low' and 'high' component.
        Identifies clusters in the low cloud that likely correspond to 
    """
    print('getting low (slightly above ground) cross section')
    low_lb,low_ub = low_percentiles[0],low_percentiles[1]
    hi_lb,hi_ub = high_percentiles[0],high_percentiles[1]
    lowc, lowc_ids_from_col = crop_by_percentile(pcd,low_lb,low_ub)
    lowc = clean_cloud(lowc)

    print('getting "high" portion of cloud')
    highc, highc_ids_from_col = crop_by_percentile(pcd,hi_lb,hi_ub)
    draw(highc)

    print('Removing buildings')
    for region in exclude_boundaries:
        test = zoom_pcd( region, test, reverse=True)
        highc = zoom_pcd(region,  highc,reverse=True)

    # with open('skio_labels_low_16-18_cluster_pt5-20.pkl','rb') as f: 
    #         labels = pickle.load(f)
    print('clustering')
    try:
        label_to_clusters = cluster_plus(lowc,eps=.3, min_points=20,return_pcds=False)
    except Exception as e:
        print(f'error {e} when clustering ')
    if save_files:
        with open(f'{file_name_base}_low_16-18_cluster_pt5-20.pkl','wb') as f: 
            pickle.dump(label_to_clusters,f)
        o3d.io.write_point_cloud(f'{file_name_base}_low_cloud_all_{low_lb}-{low_ub}pct.pcd',lowc)
        o3d.io.write_point_cloud(f'{file_name_base}_collective_highc_{hi_lb}plus.pcd',highc)
    return lowc, highc, label_to_clusters

def build_trees_knn(pcd, exclude_boundaries=[],load_from_file=False,
                        lowc=None, highc=None, label_to_clusters=None):
                
#### Reading in data and preping clusters
    ## Identify and clean up cross section clusters 
    if (lowc == None or
        highc == None or
        label_to_clusters  == None):     
        lowc,highc, label_to_clusters = id_trunk_bases(pcd,  exclude_boundaries, load_from_file)
        breakpoint()

    ## Define clusters based on labels 
    label_idls= label_to_clusters.values()
    clusters = [(idc,lowc.select_by_index(idls)) for idc, idls in enumerate(label_idls)]

    # clusters = filter_pcd_list(clusters)
#### Reading data from previous runs to exclude from run
    rerun_cluster_selection = True 
    completed_cluster_idxs=[]
    completed_cluster_pts=[]
    # if rerun_cluster_selection:
    #     completed_cluster_idxs , completed_cluster_pts = load_completed(5)
    #     save('completed_cluster_idxs.pkl',completed_cluster_idxs)
    #     # save('completed_cluster_pts.pkl',completed_cluster_pts)
    # else:
    #     completed_cluster_idxs = load('completed_cluster_idxs.pkl')

    nk_clusters= [(idc, cluster) for idc, cluster in clusters if idc not in completed_cluster_idxs]
    breakpoint()

####  Dividing clouds into smaller 
    # col_min = collective.get_min_bound()
    # col_max = collective.get_max_bound()
    col_min = arr([ 34.05799866, 286.28399658, -21.90399933])
    col_max = arr([189.47099304, 458.29800415,  40.64099884])
    safe_grid = generate_grid(col_min,col_max)
    ## Using the grid above to divide the cluster pcd list 
    ##  into easier to process parts 
    cell_clusters = []
    for idc, cell in enumerate(safe_grid):
        ids_and_clusters = filter_to_region_pcds(nk_clusters, cell)
        cell_clusters.append(ids_and_clusters)
    
####  KD tree with no prev. complete clusters 
    # creating the kd tree that will act as the search space 
    # when building the trees from seed clusters 
    print('preparing KDtree')   
    if rerun_cluster_selection:
        # KD Tree reduction given completed clusters 
        highc_pts = [tuple(x) for x in highc.points]
        categorized_pts = [tuple(x) for x in itertools.chain.from_iterable(completed_cluster_pts)]
        highc_pts = set(highc_pts)-set(categorized_pts)
        highc_pts = arr([x for x in highc_pts])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(highc_pts))
        draw(pcd)
        o3d.io.write_point_cloud('highc_incomplete.pcd',pcd)
        del pcd
        highc_tree = sps.KDTree(np.asarray(highc_pts))
        
        with open(f'highc_KDTree.pkl','wb') as f: pickle.dump(highc_tree,f)
    else:
        with open('data/in_process/highc_KDTree.pkl','rb') as f:  highc_tree = pickle.load(f)

####  Running KNN algo to build trees

    # cell_to_run_id = 3
    # grid_wo_overlap =  grid[cell_to_run_id]
    for cell_to_run_id in ['all']:
        extend_seed_clusters(clusters,highc, f'cell{cell_to_run_id}')

def build_trees_nogrid(pcd=None, exclude_boundaries=[],
                        cluster_source=None, search_pcd=None, label_to_clusters=None,
                        labeled_cluster_pcds=None, completed_cluster_idxs=[],file_base_name='seed_roots',
                          **kwargs):
                
#### Reading in data and preping clusters

    ## Identify and clean up cross section clusters 
    if labeled_cluster_pcds==None or (cluster_source == None or
                                search_pcd == None or
                                label_to_clusters  == None):     
        cluster_source, search_pcd, label_to_clusters = id_trunk_bases(pcd,  exclude_boundaries, file_name_base=file_base_name, **kwargs)

        label_idls= label_to_clusters.values()
        labeled_cluster_pcds = [(idc,cluster_source.select_by_index(idls)) 
                                    for idc, idls in enumerate(label_idls)
                                        if idc not in completed_cluster_idxs]
    # breakpoint()
    extend_seed_clusters(labeled_cluster_pcds,search_pcd,file_base_name,cycles=10,save_every=2,**kwargs) # extending downward


def pcds_from_extend_seed_file(file,pcd_index=-1):
    with open(file,'rb') as f:
        # dict s.t. {cluster_id: [list_of_pts]}
        cell_completed = dict(pickle.load(f))
    log.info('read in source file, creating pcds...')
    if 'w_order' in file:
        pts_and_orders =[x for x in cell_completed.values()]
        pts =[[x[0] for x  in y] for y in pts_and_orders]
    else:
        pts =[x for x in cell_completed.values()]
        pts =[[x for x  in y] for y in pts_and_orders]
    labels = [x for x in cell_completed.keys()]
    if pcd_index >-1:
        pts=[pts[pcd_index]]
        labels=[labels[pcd_index]]
    cluster_pcds = create_one_or_many_pcds(pts)

    return cluster_pcds
        
def read_point_shift():
    shift=load('')

def contract(in_pcd,shift):
    pts=arr(in_pcd.points)
    shifted=[(pt[0]+shift[0],pt[1]+shift[1],pt[2]+shift[2]) for pt, shift in zip(pts,shift)]
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(arr(in_pcd.colors))
    pcd.points = o3d.utility.Vector3dVector(shifted)
    return  pcd

def identify_epiphytes(pcd, shift):
    # orig = o3d.io.read_point_cloud(f'rf_cluster{i}_orig_detail.pcd')
    # test = orig.voxel_down_sample(voxel_size=.05)
    # test = remove_color_pts(test,invert = True)
    # skel_res = extract_skeleton(lowc_pcd, max_iter = 1, debug=True,cmag_save_file=f'rf_cluster{i}')
    # contracted, total_point_shift, shift_by_step = skel_res

    voxed_down = pcd.voxel_down_sample(voxel_size=.05)
    test = remove_color_pts(voxed_down,invert = True)
    # contracted = contract(test,shift)
    c_mag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs = np.where(c_mag>np.percentile(c_mag,70))[0]
    lowc_pcd = test.select_by_index(highc_idxs,invert=True)
    # voxed_down = pcd.voxel_down_sample(voxel_size=.01)
    lowc_detail, nbrs = get_neighbors_kdtree(pcd,lowc_pcd,k=200)

    nowhite =remove_color_pts(pcd, lambda x: sum(x)>2.3,invert=True)
    draw(lowc_detail)

    green = get_green_surfaces(lowc_detail)
    not_green = get_green_surfaces(lowc_detail,True)
    # draw(lowc_pcd)
    draw(green)
    draw(not_green)

    breakpoint()
    z_mag = np.array([x[2] for x in shift])
    z_cutoff = np.percentile(z_mag,80)
    print(f'{z_cutoff=}')
    high_z_idxs = np.where(z_mag>z_cutoff)[0]
    high_z_pcd = test.select_by_index(high_z_idxs)
    low_z_pcd = test.select_by_index(high_z_idxs,invert=True)
    draw([low_z_pcd])
    draw([high_z_pcd])
  
    breakpoint()

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

def inspect_others():
    
    # # reading in results from extend
    # rf_ext_pcds = pcds_from_extend_seed_file('new_seeds_w_order_in_progress.pkl')
    oth_ext_pcds = pcds_from_extend_seed_file('other_clusters_w_order_in_process.pkl')
    # root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    # Source seed cluster id to ext in 'other_clusters_w_order_in_process.pkl'
    oth_seed_to_ext = {40: 0, 41: 1, 44: 2, 45: 3, 48: 4, 49: 5, 51: 6, 52: 7, 54: 8, 62: 9, 63: 10, 64: 11, 65: 12, 66: 13, 77: 14, 78: 15, 79: 16, 85: 17, 86: 18, 88: 19, 90: 20, 121: 21, 122: 22, 123: 23, 125: 24, 126: 25, 128: 26, 144: 27, 146: 28, 147: 29, 149: 30, 150: 31, 152: 32, 153: 33, 156: 34, 157: 35, 181: 36, 182: 37}

    for pcd,seed_id_tup in zip(oth_ext_pcds,oth_seed_to_ext.items()):
        print(f'{seed_id_tup}')
        seed,idc = seed_id_tup
        draw(pcd)
        breakpoint()
            # draw(oth_ext_pcds[13])
    breakpoint()
    seeds = []
    clusters = []
    
def run_extend():
     # # #       and the clusters fed into extend seed clusters
    lowc  = o3d.io.read_point_cloud('new_low_cloud_all_16-18pct.pcd')
    # highc = o3d.io.read_point_cloud('new_collective_highc_18plus.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl')
    empty_pcd= o3d.geometry.PointCloud() 
    # # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
    c_trunk_clusters = create_one_or_many_pcds([arr(lc.points) for lc in trunk_clusters],labels=[k for k,v in label_to_clusters.items()])

    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
    # rf_colored = arr(c_trunk_clusters)[rf_seeds]  
    # other_colored = arr(c_trunk_clusters)[~arr(refined_clusters)]
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    all_ids = rf_seeds + final_int_ids
    all_ids.sort()
    
    other_clusters = [(idc,cluster) if idc in final_int_ids else (idc,empty_pcd) for idc,cluster in enumerate(c_trunk_clusters) ]
    extend_seed_clusters(clusters_and_idxs = other_clusters,
                            src_pcd=highc,
                            file_label='other_clusters',cycles=250,save_every=20,draw_every=160, max_distance=.1)

if __name__ =="__main__":
    # Prepping IDs and mappngs
    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] #152
    # rf_colored = arr(c_trunk_clusters)[rf_seeds]  
    # other_colored = arr(c_trunk_clusters)[~arr(refined_clusters)]
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    
    # load all seed clusters 
    lowc  = o3d.io.read_point_cloud('new_low_cloud_all_16-18pct.pcd')
    # highc = o3d.io.read_point_cloud('new_collective_highc_18plus.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl')
    empty_pcd= o3d.geometry.PointCloud() 
    # # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
   
   
    all_ids = rf_seeds + final_int_ids
    all_ids.sort()
    root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    seed_to_root_map = {seed_id: root_pcd for seed_id,root_pcd in zip(all_ids,root_pcds)}
    seed_to_root_id = {seed: idc for idc,seed in enumerate(all_ids)}

    seed_to_exts = {152: 0 # untested 51 root
                    ,107: 1# ext 1 is two trees
                    ,108: 1#   and so correlates to seeds 107 and 108
                    ,109: 2
                    ,111: 3
                    ,112: 4                    
                    #(113: 5)
                    ,114: 6
                    ,115: 7 # this is a good seed but the ext only goes up the trunk a couple feet
                    ,116: 8 # this is a good ext, but the seed is spanish moss
                    ,133: 9  # This is a group of 3 trees
                    ,134: 10 
                    ,135: 11
                    ,136: 12 # This seed only goes halfway around the trunk
                    ,137: 13
                    ,138: 14 # seed is vine, cluster contains a branch of some tree
                    ,190: 15
                    ,191: 16 # ***needs to be rerun, is cutoff by the edge of the selected area
                    #(193, 17), # is a mismatch
                    ,194: 18  # ***needs to be rerun, is cutoff by the edge of the selected area
    } 
    exts_to_seed = {v:k for k,v in seed_to_exts.items()}
    unmatched_clusters = [151,110,113,180,189,193,132,148]
    unmatched_exts = [17] # 17 is the bottom of some spanish moss

    #to_pass_back through
    pass_again = [107,108]

    all_seeds = {seed: 
                    (trunk_clusters[seed] ,
                      seed_to_root_map[seed], seed_to_root_id[seed]) for seed in rf_seeds}

    # highc = o3d.io.read_point_cloud('new_collective_highc_18plus.pcd')
    pcd = o3d.io.read_point_cloud('partitioning_search_area_collective.pcd')
    highc, _ = crop_by_percentile(pcd, 17,100)
    rf_ext_pcds = pcds_from_extend_seed_file('new_seeds_w_order_in_progress.pkl')
    # unmatched_ext_pcds = arr(ext_pcds)[unmatched_clusters]
    # unmatched_ext_pcds = [pcds_from_extend_seed_file(ext_result_file, pcd_index=i)[0] for i in unmatched_exts]

    # ext_result_file = 'other_clusters_w_order_in_process.pkl'
    ext_result_file= 'new_seeds_w_order_in_progress.pkl'
    ext_pcds = pcds_from_extend_seed_file(ext_result_file)
    # draw(unmatched_ext_pcds+all_seeds)
    # breakpoint()
    # o3d.io.write_point_cloud(f'data/refined_clusters/k_cluster_{152}.pcd',c_tc)
    good_ids = []
    bad_ids = []
    # test_seeds = [(seed,details) for seed,details in all_seeds if seed_to_exts.get(seed)]
    # for seed, seed_details in test_seeds: draw([seed_details[1],ext_pcds[seed_to_exts.get(seed,-1)],seed_details[0]])
    completed = [107,108,109,111,112]
    for seed, seed_details in all_seeds.items():
        try:
            if seed not in completed:
                seed_pcd, root_pcd, root_pcd_id = seed_details
                prnt = False
                reext = False
                rf_ext_id = seed_to_exts.get(seed,-1)
                if rf_ext_id ==-1:
                    print(f'ext for seed {seed} not found ')
                    bad_ids.append((seed,None,None))
                else:
                    ext_pcd = ext_pcds[rf_ext_id]

                    seed_pcd.paint_uniform_color([1,0,1])
                    # root_pcd = seed_to_root_map[21] and 22 correspond to cluster 1 
                    root_pcd.paint_uniform_color([1,0,0])
                    # draw([root_pcd]+[ext_pcd,seed_pcd])
                    # prnt = True
                    # reext = True
                    detail_pcd_file = f'rf_cluster{rf_ext_id}_orig_detail.pcd'
                    # if reext:
                    #     # Rebuilds cluster from root_pcd to seperate tree from neighbors
                    #     ## get original details near the new ext from the orig detail version of the old ext
                    #     ## changes detail_pcd_file to point to the new, hopefully isolated tree
                    #     clusters_and_idxs=[(seed,root_pcd)]#,(108,seed_to_root_map[108])]
                    #     tree_pcds = extend_seed_clusters(clusters_and_idxs,detail_pcd,f'rf_cluster_{seed}_rebuild',cycles=700,save_every=70,draw_every=70,max_distance=.04,k=50) #
                    #         # cycles=800,save_every=70,draw_every=70,max_distance=.05) #
                    #     draw(tree_pcds)
                    #     breakpoint()
                    #     detail_pcd = o3d.io.read_point_cloud(detail_pcd_file)
                    #     ext_pcd = tree_pcds[0]
                    #     breakpoint()
                    #     # Recovering original detail here, but no details for the root are available
                    #     new_detail_pcd,pts = get_neighbors_kdtree(detail_pcd, ext_pcd)
                    #     draw([root_pcd]+[new_detail_pcd,seed_pcd])
                    #     breakpoint()
                    #     # write_point_cloud(f'data/results/full_skio_iso/rf_cluster{seed}_orig_detail.pcd', new_detail_pcd)
                    #     write_point_cloud(f'rf_cluster_rebuild{seed}_orig_detail.pcd', new_detail_pcd)
                    #     detail_pcd_file = f'rf_cluster_rebuild{seed}_orig_detail.pcd'
                    if not prnt:
                        good_ids.append((seed,rf_ext_id,root_pcd_id))
                        recover_original_details([root_pcd],save_file=f'data/results/full_skio_iso/rf_cluster_root{seed}')
                        root_details_file = f'data/results/full_skio_iso/rf_cluster_root{seed}_orig_detail.pcd'
                        detail_pcd = o3d.io.read_point_cloud(detail_pcd_file)
                        root_details_pcd = o3d.io.read_point_cloud(root_details_file)
                        joined = join_pcds([root_details_pcd, detail_pcd])[0]
                        # draw(joined)
                        # breakpoint()
                        write_point_cloud(f'data/results/full_skio_iso/full_ext_seed{seed}_rf{rf_ext_id}_orig_detail.pcd',joined)
        except Exception as e:
            print(f'error on seed {seed}: {e}')

    good_ids.extend([(seed,seed_to_exts[seed],seed_to_root_id[seed]) for seed in completed])

    save('rf_seed_ext_root_ids.pkl',good_ids)
    breakpoint()
    # good_ids = load('rf_seed_ext_root_ids.pkl')
    # for seed, root_ids, rf_ext_id in good_ids:
    #     ext_pcd = ext_pcds[rf_ext_id]      
    #     seed_pcd.paint_uniform_color([1,0,1])
    #     good_ids.append((seed,rf_ext_id,root_pcd_id))
    #     recover_original_details(root_details,save_file=f'rf_root_cluster_{seed}')
    #     root_details = f'rf_root_cluster_{seed}_orig_detail.pcd'
    #     final = join_pcds([root_details, ext_pcd])   
    #     o3d.io.write_point_cloud(root_details,final)  
    # exclude_regions = {
    #         'building1' : [ (77,350,0),(100, 374,5.7)], 
    #         'building2' : [(0, 350), (70, 374)],
    #         'far_front_yard':[ (0,400),(400, 500)],
    #         'far_rside_brush':[ (140,0),(190, 500)],
    #         'lside_brush':[ (0,0),(77, 50 all_ids = rf_seeds + final_int_ids0)],
    #         'far_back_brush': [ (0,0),(200, 325)]
    # }
    pcd = o3d.io.read_point_cloud('partitioning_search_area_collective.pcd')
    # lowc, lowc_ids_from_col = crop_by_percentile(pcd, 16,18)
    # all_seeds = [o3d.io.read_point_cloud(f'data/refined_clusters/k_cluster_{seed}.pcd') for seed in rf_seeds]
    # build_trees_nogrid(pcd=pcd, exclude_boundaries=list(exclude_regions.values),
    #                     cluster_source=None, search_pcd=None, label_to_clusters=None,
    #                     labeled_cluster_pcds=None, completed_cluster_idxs=[],)
    
    # build_trees_knn(pcd, load_from_file=True, lowc, highc, label_to_clusters)

    # file = "cell3_complete2.pkl"
    file = "completed_cluster_idxs.pkl"
    # load_clusters_get_details()
