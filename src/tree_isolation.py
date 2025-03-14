
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

from matplotlib import patches


from open3d.io import read_point_cloud, write_point_cloud

from utils.math_utils import (
    get_center
)

from set_config import config, log
from reconstruction import recover_original_details, get_neighbors_kdtree, get_pcd
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
from viz.viz_utils import iter_draw, draw
from viz.viz_utils import draw, color_continuous_map

def step_through_and_evaluate_clusters():
    # file = "cell3_complete2.pkl"
    file = "completed_cluster_idxs.pkl"

    # idxs, pts = load_completed(0, [file])
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

def bin_colors(zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):
    # binning colors in pcd, finding most common
    # round(2) reduces 350k to ~47k colors
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) 
                if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       
                    and pt[1]>zoom_region[1][0] 
                    and  pt[1]<zoom_region[1][1])]

    cols = [','.join([f'{round(y,1)}' for y in x[1]]) for x in region]
    d_cols, cnts = np.unique(cols, return_counts=True)
    print(len(d_cols))
    print((max(cnts),min(cnts)))
    # removing points of any color other than the most common
    most_common = d_cols[np.where(cnts)]
    most_common_rgb = [tuple((float(num) for num in col.split(','))) for col in most_common]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) in most_common_rgb ]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) != tuple((1.0,1.0,1.0)) ]
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in color_limited]))
    limited_pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in color_limited]))
    draw(limited_pcd)


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
    pts = [tuple((idc,np.asarray(cluster.points))) for idc, cluster in clusters]
    new_idcs = filter_list_to_region(pts,zoom_region)
    new_clusters = [(idc, cluster) for idc, cluster in clusters if idc in new_idcs]
    return  new_clusters

def load_clusters_get_details():
    # For loading results of KNN loop
    pfile = 'data/in_process/cluster126_fact10_0to50.pkl'
    with open(pfile,'rb') as f:
        tree_clusters = dict(pickle.load(f))
    # breakpoint()
    labels = [x for x in  tree_clusters.keys()]
    pts =[x for x in tree_clusters.values()]

    # pts = pts[:3]
    # labels= labels[:3]

    cluster_pcds = create_one_or_many_pcds(pts, labels = labels)
    recover_original_detail(cluster_pcds, file_num_base = 20000000)
    breakpoint()

def pcds_from_extend_seed_file(file,pcd_index=-1):
    with open(file,'rb') as f:
        # dict s.t. {cluster_id: [list_of_pts]}
        cell_completed = dict(pickle.load(f))
    log.info('read in source file, creating pcds...')
    pts_and_orders =[x for x in cell_completed.values()]
    pts =[[x[0] for x  in y] for y in pts_and_orders]
    labels = [x for x in cell_completed.keys()]
    if pcd_index >-1:
        pts=[pts[pcd_index]]
        labels=[labels[pcd_index]]
    cluster_pcds = create_one_or_many_pcds(pts)

    return cluster_pcds

def load_completed(cell_to_run_id=None,
                    files = []):
    # Loading completed clusters and their 
    #  found nbr points from prev runs

    if files == []:
        for cell_num in range(cell_to_run_id):
            files.append(f'cell{cell_num}_complete2.pkl')
        files.append(f'cell5_complete.pkl')
        files.append(f'cellall_complete2.pkl')
        files.append(f'all_complete.pkl')

    completed_cluster_idxs = []
    completed_cluster_pts = []
    for file in files:
        print(f'adding clusters found in {file}')
        with open(file,'rb') as f:
            # dict s.t. {cluster_id: [list_of_pts]}
            cell_completed = pickle.load(f)
        breakpoint()

        pts_and_orders =[x for x in cell_completed.values()]
        pts =[[x[0] for x  in y] for y in pts_and_orders]
        
        # non_overlap_grid = grid[cell_num]
        all_idcs = len([idc for idc,_ in cell_completed.items()])
        pts_per_cluster = [len(pts) for (idc,pts) in cell_completed.items()]
        
        in_idcs = [idc for idc in cell_completed.keys() if idc not in completed_cluster_idxs]
        k_cluster_pts= [pts for idc, pts in cell_completed.items() if idc in in_idcs]
        # completed_cluster_keys = [x for x in completed.keys()]
        # in_idcs = filter_list_to_region(k_cluster_pts, non_overlap_grid)

        # breakpoint()
        num_new_clusters = len(in_idcs)
        pts_per_new_cluster = [len(pts) for pts in k_cluster_pts]
        num_new_pts = sum(pts_per_new_cluster)
        completed_cluster_idxs.extend(in_idcs)
        completed_cluster_pts.extend(k_cluster_pts)

        print(f'{all_idcs} total clusters found in grid')
        print(f'{pts_per_cluster} pts in each cluster')

        print(f'{num_new_clusters} new, complete clusters found in {file}')
        print(f'{pts_per_new_cluster} pts in each new cluster')
        print(f'{num_new_pts} points categorized in grid')
        # added_k_ids = len(in_idcs)
        # print(f'{added_k_ids} complete clusters from grid cell {cell_num}')
    return completed_cluster_idxs , completed_cluster_pts

# def save_pt_to_labelorder_map():

        
def extend_seed_clusters(clusters_and_idxs:list[tuple],
                            src_pcd,
                            file_label,
                            k=200,
                            max_distance=.3,
                            cycles= 150,
                            save_every = 10,
                            cutoff_order=60,
                            draw_progress = True,
                            debug = True,
                            recalc_match_set:bool=True):
    """
        Takes a list of tuples defining clusters (e.g.[(label,cluster_pcd)...])
        and a source pcd
    """
    idcs_to_run = [idc for idc, cluster in clusters_and_idxs]
    cluster_pts = [(idc, arr(cluster.points)) for idc,cluster in clusters_and_idxs]

    high_c_pt_assns = defaultdict(lambda:-1) 
    curr_pts = [[]]*len(clusters_and_idxs)
    for idc, cluster_pt_list in cluster_pts:
        curr_pts[idc] = cluster_pt_list
        for pt in cluster_pt_list:
            high_c_pt_assns[tuple(pt)] = (idc,0)

    src_pts = arr(src_pcd.points)
    print('creating KD search tree')
    # src_tree = sps.KDTree(src_pts)
    # with open(f'new_kd_search_tree.pkl','wb') as f: pickle.dump(src_tree,f)
    # o3d.io.write_point_cloud('new_kd_search_tree.pcd',src_tree)
    src_tree = load('new_kd_search_tree.pkl')
    src_pts = src_tree.data
    num_pts =  len(src_pts)
    complete = []
    cutoff_order = cutoff_order
    max_orders = defaultdict(int)
    iters = save_every
    # recreate = False
    override_cutoff_order=[]
    all_nbrs= []
    print(f'running knn for idcs {idcs_to_run}')
    breakpoint()
    for cycle_num in range(cycles):
        print('start iter')
        if iters<=0:
            iters =save_every
            tree_pts = defaultdict(list)
            src_pcd_pts = arr(src_pcd.points)
            try:
                for pt,tup in high_c_pt_assns.items(): 
                    if isinstance(tup,int):
                        idc,order = tup,0
                    else:
                        idc,order = tup
                    tree_pts[idc].append((pt,order))
                print('created to_save. Pickling...')
                base_file = f'{file_label}_w_order'
                complete = list([tuple((idc,pt_order_list)) for idc,pt_order_list in  tree_pts.items() if idc in complete])
 
                save(base_file +'_complete.pkl', complete)
                ## save in progress
                # all = list([tuple((idc,pt_order_list)) for idc,pt_order_list in  tree_pts.items()])
                # save(base_file +'_in_progress.pkl', all)

                # num_assigned_pts = sum([len(x[1]) for x in all])
                    
            except Exception as e:
                breakpoint()
                print('error saving')

            # if recalc_match_set:
                # categorized_pts = arr(itertools.chain.from_iterable(tree_pts.values()))
                # src_pts = np.setdiff1d(src_pts, categorized_pts)
                # src_pts = arr(src_pts)
                # src_tree = sps.KDTree(src_pts)
                # src_pcd = src_pcd.select_by_index(all_nbrs,invert=True)
                # src_pcd_pts = arr(src_pcd.points)
                # src_tree = sps.KDTree(src_pcd_pts)
                # src_pts = src_tree.data
                # num_pts =  len(src_pts)
                # all_nbrs= []

            if draw_progress:
                print('starting draw')
                labels = list([x for x in range(len(tree_pts))])
                pts = list([[y[0] for y in x] for x in tree_pts.values()])
                
                tree_pcds = create_one_or_many_pcds(pts = pts, labels = labels)
                # maxzs = [max(arr(pt_list)[:,2]) for pt_list in pts]
                ## for coloring based on a given condition
                # for idc, pcd in enumerate(tree_pcds):
                #     if max_orders[idc]>10:
                #         pcd.paint_uniform_color([1,0,0])
                draw(tree_pcds)
                breakpoint()
            del tree_pts
            del tree_pcds
            tree_pcds=None

        iters=iters-1
        print(f'querying {cycle_num}')
        for idx, cluster in clusters_and_idxs:
            
            if idx not in complete:               
                if len(curr_pts[idx])>0:
                    # get neighbors in the vicinity of the cluster
                    dists,nbrs = src_tree.query(curr_pts[idx],
                                                k=k,
                                                distance_upper_bound= max_distance)
                    nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
                    nbr_pts = [nbr_pt for nbr_pt in src_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                    order = cycle_num
                    if len(nbr_pts)>0:
                        for nbr_pt in nbr_pts: high_c_pt_assns[tuple(nbr_pt)] = (idx,order)
                        if max_orders[idx]< order: max_orders[idx] = order
                        curr_pts[idx] = nbr_pts
                        all_nbrs.extend(nbrs)
                        if len(curr_pts[idx])<5:
                            complete.append(idx)
                            print(f'{idx} added to complete')

                if len(curr_pts[idx])==0:
                    complete.append(idx)
                    print(f'{idx} added to complete')
    
    print('finish!')
    tree_pts = defaultdict(list)
    try:
        for pt,tup in high_c_pt_assns.items(): 
            idc,order = tup
            tree_pts[idc].append((pt,order))
        print('created to_save. Pickling...')
        base_file = f'{file_label}_w_order'
        complete = list([tuple((idc,pt_order_list)) for idc,pt_order_list in  tree_pts.items() if idc in complete])
        save(base_file +'_complete.pkl', complete)
    except Exception as e:
        breakpoint()
        print('error saving')

def generate_grid(min_bnd,
                  max_bnd,
                    grid_xyz_cnt =  arr((2,3,1)),
                    overlap_ratio = 1/7
                  ):
    """Divides the region defined by the provide range 
        into a grid with a defined # of divisions of the x,y and z dimensions

    Args:
        min_bnd: the 'bottom-left' of the region (x,y,z)
        max_bnd:  the 'top-right' of the region (x,y,z)
        grid_xyz_cnt: the number of divisions desired in the x, y and z dimensions
    """
    col_lwh = max_bnd -min_bnd
    grid_xyz_cnt = arr((2,3,1)) # x,y,z
    grid_lines = [np.linspace(0,num,1) for num in grid_xyz_cnt]
    grid_lwh = col_lwh/grid_xyz_cnt
    ll_mults =  [np.linspace(0,num,num+1) for num in grid_xyz_cnt]
    llv = [minb + dim*mult for dim, mult,minb in zip(grid_lwh,ll_mults,min_bnd)]

    grid = arr([[[llv[0][0],llv[1][0]],[llv[0][1],llv[1][1]]],[[llv[0][1],llv[1][0]],[llv[0][2],llv[1][1]]],    
                [[llv[0][0],llv[1][1]],[llv[0][1],llv[1][2]]],[[llv[0][1],llv[1][1]],[llv[0][2],llv[1][2]]],    
                [[llv[0][0],llv[1][2]],[llv[0][1],llv[1][3]]],[[llv[0][1],llv[1][2]],[llv[0][2],llv[1][3]]]])
    ## We want a bit of overlap since nearby clusters sometime contest for points 
    overlap = grid_lwh*overlap_ratio    
    safe_grid = [[[ll[0]-overlap[0],ll[1]-overlap[1]],[ur[0]+overlap[0], ur[1]+overlap[1]]] for ll, ur in grid]
    return safe_grid

def id_trunk_bases(pcd =None, 
            exclude_boundaries = None,
            load_from_file = False
):
    """Splits input cloud into a 'low' and 'high' component.
        Identifies clusters in the low cloud that likely correspond to 
    """
    print('getting low (slightly above ground) cross section')
    if load_from_file:
        lowc = o3d.io.read_point_cloud('new_low_cloud_all_16-18pct.pcd')
        highc= o3d.io.read_point_cloud('new_collective_highc_18plus.pcd')
        label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl')
    else:
        lowc, lowc_ids_from_col = crop_by_percentile(pcd, 16,18)
        lowc = clean_cloud(lowc)
        o3d.io.write_point_cloud('new_low_cloud_all_16-18pct.pcd',lowc)

        print('getting "high" portion of cloud')
        highc, highc_ids_from_col = crop_by_percentile(pcd, 18,100)
        o3d.io.write_point_cloud('new_collective_highc_18plus.pcd',highc)
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
        with open('new_skio_labels_low_16-18_cluster_pt5-20.pkl','wb') as f: 
            pickle.dump(label_to_clusters,f)
    return lowc, highc, label_to_clusters

def build_trees_knn(exclude_boundaries=[],load_from_file=False):
    
#### Reading in data and preping clusters
    # collective = o3d.io.read_point_cloud('collective.pcd')
    pcd = o3d.io.read_point_cloud('partitioning_search_area_collective.pcd')
    
    ## Identify and clean up cross section clusters 
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

def read_point_shift():
    shift=load('')

def homog_colors(pcd):
    colors = arr(pcd.colors)
    pts=arr(pcd.points)
    white_idxs = [idc for idc,color in enumerate(colors) if sum(color)>2.7]
    white_pts = [pts[idc] for idc in white_idxs]

    non_white_tree = pcd.select_by_index(white_idxs, invert=True)
    tree = sps.KDTree(arr(non_white_tree.points))
    white_nbrs = tree.query([white_pts],30)
    avg_neighbor_color = np.mean(colors[ white_nbrs[1]][0],axis=1)
    for white_idx,avg_color in zip(white_idxs,avg_neighbor_color):  colors[white_idx] = avg_color
    pcd.colors =  o3d.utility.Vector3dVector(colors)
    draw(pcd)
    
def remove_color_pts(pcd,
                        color_lambda = lambda x: sum(x)>2.7,
                        invert=False):
    colors = arr(pcd.colors)
    ids = [idc for idc, color in enumerate(colors) if color_lambda(color)]
    new_pcd = pcd.select_by_index(ids, invert = invert)
    return new_pcd

def get_green_surfaces(pcd,invert=False):
    green_pcd= remove_color_pts(pcd,    lambda rgb: rgb[1]>rgb[0] and rgb[1]>rgb[2] and 0.5<(rgb[0]/rgb[2])<2, invert)
    return green_pcd

def mute_colors(pcd):
    colors = arr(pcd.colors)
    rnd_colors = [[round(a,2) for a in col] for col in colors]
    pcd.colors= o3d.utility.Vector3dVector(rnd_colors)
    draw(pcd)

def contract(in_pcd,shift):
    pts=arr(in_pcd.points)
    shifted=[(pt[0]+shift[0],pt[1]+shift[1],pt[2]+shift[2]) for pt, shift in zip(pts,shift)]
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(arr(in_pcd.colors))
    pcd.points = o3d.utility.Vector3dVector(shifted)
    return  pcd

def identify_epiphytes(pcd, shift):
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

    


if __name__ =="__main__":
    # 0 is parts of many trees
    # 1 is two trees
    # 2 is v horizontal, great detail on epiphite clumps
    # 3 has sfc. 
    # 6 is only a trunk
    # 7s seed seems to have been a vine 
    # 8 straight, tall
    # 9 3 trees, good detail
    # 10 extraordinary detail
    # 11 large but incomplete, possibly cluster splitting
    # 13 great trunk detail, multi stem
    # 14 not a trunk151, 0
    # 107, 1
    # 108, 2
    # 109, 3
    # 110, 4
    # 111, 5
    # 112, 6
    # 113, 7
    # 114, 8
    # 116, 9
    # 132, 10
    # 133, 11
    # 134, 12
    # 135, 13
    # 136, 14
    # 180, 15
    # 189, 16
    # 190, 17
    # 191, 18
    # 193, 19
    # 194, 20

    for i in range(18):
        if i not in [4,15,16,17]:
            print(f'cluster {i}')
            orig = o3d.io.read_point_cloud(f'rf_cluster{i}_orig_detail.pcd')
            #shift = load('rf_cluster5_point_shift.pkl')
            draw(orig)
            breakpoint()
    identify_epiphytes(orig,shift)
    breakpoint()
    ########### Notes ##############
    ###
    ### - completed_cluster_idxs
    ###     - ids of the root clusters used to generate completed trees
    ### - collective 
    ###     - is the full skio scan 
    ### - new_skio_labels_low_16-18_cluster_pt5-20
    ###    - Source pcd for the trunk seed clusters in new_lowc_lbl_to_clusters_pt3_20
    ### -  new_lowc_lbl_to_clusters_pt3_20.pkl
    ###     - Seed clusters fom ...low_16-18_cluster_pt5-20 used to generate trees
    ### - new_lowc_refined_clusters
    ###     - a subset of the above clusters that represent our trees of interest
    ### - new_seeds_w_order_in_progress.pkl
    ###     - [(labelID,[(pt_order, pt),...]),... ] 
    ###     - mappings for points to labels for clusters built off of seed clusters in new_skio_labels_low_16-18_cluster_pt5-20
    ### - rf_cluster{i}_point_shift
    ###     - a list of shifts for each point in the corr. cluster after 1 round of lapacian transform
    ###     - orig = o3d.io.read_point_cloud(f'rf_cluster{i}_orig_detail.pcd')
    ###     - test = orig.voxel_down_sample(voxel_size=.05)
    ### 
    ### 
    ### 
    ###
    # ################################

    # # pcd = pcds[0]
    # # pcd = o3d.io.read_point_cloud('rf_cluster0.pcd')
    # # orig = o3d.io.read_point_cloud('rf_cluster5_orig_detail.pcd')
    # test = orig.voxel_down_sample(voxel_size=.05)
    # # # test = test.uniform_down_sample(4)
    # green = get_green_surfaces(test)
    # not_green = get_green_surfaces(test,True)
    # draw(test)
    # draw(not_green)
    # draw(green)

    # test = test.uniform_down_sample(4)
    # orig = recover_original_details([pcds[0]],save_file=f'rf_cluster{4}')
    # orig = o3d.io.read_point_cloud('rf_cluster9_orig_detail.pcd')
    # draw(orig)
    for i in [13,17]:
        pcd = pcds_from_extend_seed_file('new_seeds_w_order_in_progress.pkl',pcd_index=i)
        recover_original_details(pcd,save_file=f'rf_cluster{i}')
        # orig = o3d.io.read_point_cloud(f'rf_cluster{i}_orig_detail.pcd')
        # test = orig.voxel_down_sample(voxel_size=.05)
        # test = remove_color_pts(test,invert = True)
        # skel_res = extract_skeleton(lowc_pcd, max_iter = 1, debug=True,cmag_save_file=f'rf_cluster{i}')
        # contracted, total_point_shift, shift_by_step = skel_res
        # breakpoint()
        # save(f'rf_cluster{i}_shift.pkl',shift_by_step)
    breakpoint()
        
    green = get_green_surfaces(test)
    not_green = get_green_surfaces(test,True)
    draw(test)
    draw(not_green)
    draw(green)
   
    z_mag = np.array([x[2] for x in total_point_shift])
    z_cutoff = np.percentile(z_mag,80)
    print(f'{z_cutoff=}')
    high_z_idxs = np.where(z_mag>z_cutoff)[0]
    high_z_pcd = test.select_by_index(high_z_idxs)
    draw([high_z_pcd])
    
    c_mag = np.array([np.linalg.norm(x) for x in total_point_shift])
    color_continuous_map(test,c_mag)
    high_c_idxs = np.where(c_mag>np.percentile(c_mag,60))[0]
    high_c_pcd = test.select_by_index(high_c_idxs)
    low_c_pcd = test.select_by_index(high_c_idxs,invert=True)
    draw([high_c_pcd])
    draw([low_c_pcd])

    green = get_green_surfaces(ow_c_pcd)
    not_green = get_green_surfaces(ow_c_pcd,True)
    draw(ow_c_pcd)
    draw(not_green)
    draw(green)

    # contracted_pcd= o3d.read_point_cloud('rf_cluster0_contracted_pcd')
    # contraction_mag_processing(pcd, contracted_pcd, file='rf_cluster0_c_mag.pkl' )

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20))
    # stat_down.normalize_normals()
    mesh, densities = map_density(pcd)
    # collective= o3d.io.read_point_cloud('data/in_process/collective.pcd')
    # exclude_boundaries = [[ (77,350, 0),(100, 374,5.7)],  [(0, 350, 0), (70, 374, 5.7)]  ]
    exclude_regions = {
            'building1' : [ (77,350,0),(100, 374,5.7)], 
            'building2' : [(0, 350), (70, 374)],
            'far_front_yard':[ (0,400),(400, 500)],
            'far_rside_brush':[ (140,0),(190, 500)],
            'lside_brush':[ (0,0),(77, 500)],
            'far_back_brush': [ (0,0),(200, 325)]
    }
    # exclude_boundaries = exclude_regions.values()
    exclude_boundaries=[]
    pcd = o3d.io.read_point_cloud('partitioning_search_area_collective.pcd')
    lowc,highc, label_to_clusters = id_trunk_bases(pcd,  
                                                    exclude_boundaries, 
                                                    load_from_file=True)
    # label_to_clusters = cluster_plus(lowc,eps=.3, min_points=20,return_pcds=False)
    # label_idls= label_to_clusters.values()
    # clusters = [(idc,lowc.select_by_index(idls)) for idc, idls in enumerate(label_idls)]
    # breakpoint()
    # o3d.io.write_point_cloud('new_lowc_clustered_pt3_20.pcd',pcd)
    # del pcd
    # highc_tree = sps.KDTree(np.asarray(highc.points()))
    # with open(f'new_lowc_ref_clusters.pkl','wb') as f: pickle.dump(refined_clusters,f)
    
    breakpoint()
    # take out cluster 3
    # extend_seed_clusters(clusters,highc,'new_seeds')


    breakpoint()
    # build_trees_knn(load_from_file=True)

    # file = "cell3_complete2.pkl"
    file = "completed_cluster_idxs.pkl"
    # load_clusters_get_details()
    

    
    draw(pcd)
    # highlight and draw 
    # building_two = zoom_pcd([ (77,350,0),(100, 374,5.7)], collective)
    # draw(building_two)
    # far_front_yard = zoom_pcd([ (0,400),(400, 500)], pcd)
    # far_front_yard.paint_uniform_color([1,0,0])

    # far_rside_brush = zoom_pcd([ (140,0),(190, 500)], collective)
    # far_rside_brush.paint_uniform_color([1,0,0])
    
    # lside_brush = zoom_pcd([ (0,0),(77, 500)], collective)
    # lside_brush.paint_uniform_color([1,0,0])

    # far_back_brush = zoom_pcd([ (0,0),(200, 325)], collective)
    # far_back_brush.paint_uniform_color([1,0,0])

    # #test.paint_uniform_color([1,0,0])
    # draw([pcd,far_front_yard,far_rside_brush,far_back_brush])
    breakpoint()

    # lowc,highc,labels = id_trunk_bases(None,  exclude_boundaries)


    clusters = create_one_or_many_pcds(pts)