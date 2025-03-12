
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
from viz.viz_utils import iter_draw, draw

def step_through_and_evaluate_clusters():
    # file = "cell3_complete2.pkl"
    file = "completed_cluster_idxs.pkl"

    ########### Notes ##############
    ###
    ### - completed_cluster_idxs
    ###     - ids of the root clusters used to generate completed trees
    ### - collective 
    ###     - is the full skio scan 
    ### - 
    ### 
    ### 
    ### 
    ### 
    ### 
    ###
    # ################################

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


def loop_over_pcd_parts(file_prefix = 'data/input/SKIO/part_skio_raffai',
                    return_pcd_list =False,
                    return_single_pcd = False,
                    return_list = None,
                    return_lambda = None):
    """Used to read in the skio scan. reads in the gridded sections 

    Args:
        file_prefix (str, optional): _description_. Defaults to 'data/input/SKIO/part_skio_raffai'.
        return_pcd_list (bool, optional): _description_. Defaults to False.
        return_single_pcd (bool, optional): _description_. Defaults to False.
        return_list (_type_, optional): _description_. Defaults to None.
        return_lambda (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if return_lambda: return_list = []
    else:
        if not return_single_pcd and not return_pcd_list:
            return None, None, None 
    extents = {}
    pcds = []
    pts = []
    colors = []
    contains_region= []
    base = 20000000

    for i in range(40):
        num = base*(i+1) 
        file = f'{file_prefix}_{num}.pcd'
        pcd = read_point_cloud(file)
        if return_lambda:
            return_list.append(return_lambda(pcd))
        if return_pcd_list:
            pcds.append(pcd)
        elif return_single_pcd:
            pts.extend(list(arr(pcd.points)))   
            colors.extend(list(arr(pcd.colors)))
    if return_single_pcd:
        collective = o3d.geometry.PointCloud()
        collective.points = o3d.utility.Vector3dVector(pts)
        collective.colors = o3d.utility.Vector3dVector(colors)   
        pcds.append(pcd)     
    return return_list, contains_region, pcds

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
    breakpoint()
    labels = [x for x in  tree_clusters.keys()]
    pts =[x for x in tree_clusters.values()]

    # pts = pts[:3]
    # labels= labels[:3]

    cluster_pcds = create_one_or_many_pcds(pts, labels = labels)
    recover_original_detail(cluster_pcds, file_num_base = 20000000)
    breakpoint()


def load_completed(cell_to_run_id,
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

from string import Template 
def recover_original_detail(cluster_pcds,
                            file_prefix = Template('data/input/SKIO/part_skio_raffai_$idc.pcd'),
                            #= 'data/input/SKIO/part_skio_raffai',
                            save_result = False,
                            save_file = 'orig_dets',
                            file_num_base = None,
                            file_num_iters = 39):
    """
        Reversing the initial voxelization (which was done to increase algo speed)
        Functions by (for each cluster) limiting parent pcd to just points withing the
            vicinity of the search cluster then performs a KNN search 
            to find neighobrs of points in the cluster
    """
    detailed_pcds = []
    # defining files in which initial details are stored
    files = []
    if file_num_base:
        for idc in range(file_num_iters):
            files.append(file_prefix.substitute(idc = idc).replace(' ','_') )
    else:
        files = [file_prefix]
    pt_branch_assns = defaultdict(list)
    # iterating a bnd_box for each pcd for which
    #    we want to recover the orig details
    for idb, cluster_pcd in enumerate(cluster_pcds):
        bnd_box = cluster_pcd.get_oriented_bounding_box() 
        vicinity_pts = []
        v_colors = []
        skel_ids = []
        # Limiting the search field to points in the general
        #   vicinity of the non-detailed pcd
        for file in files:
            print(f'checking file {file}')
            pcd = read_point_cloud(file)
            pts = arr(pcd.points)
            if len(pts)>0:
                cols = arr(pcd.colors)
                all_pts_vect = o3d.utility.Vector3dVector(pts)
                vicinity_pt_ids = bnd_box.get_point_indices_within_bounding_box(all_pts_vect) 
                if len(vicinity_pt_ids)>0:
                    v_pt_values = pts[vicinity_pt_ids]
                    colors = cols[vicinity_pt_ids]
                    print(f'adding {len(vicinity_pt_ids)} out of {len(pts)}')
                    vicinity_pts.extend(v_pt_values)
                    v_colors.extend(colors)
            else:
                print(f'No points found for {file}')
        try:
            print('Building pcd from points in vicinity')

            query_pts = arr(cluster_pcd.points)
            whole_tree = sps.KDTree(vicinity_pts)
            print('Finding neighbors in vicinity') 
            dists,nbrs = whole_tree.query(query_pts, k=750, distance_upper_bound= .3) 

            nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x!= len(vicinity_pts)]
            pt_branch_assns[idb] = arr(vicinity_pt_ids)[nbrs]

            # if idb%5==0 and idb>5:
            complete = list([tuple((idb,nbrs)) for idb,nbrs in  pt_branch_assns.items()])
            save(f'skeletor_branch{idb}_complete.pkl', complete)
        except Exception as e:
            breakpoint()
            print(f'error {e} getting clouds')
    return detailed_pcds

def update(file, to_write):
    curr = load(file)
    curr.extend(to_write)
    with open(file,'wb') as f:
        pickle.dump(curr,f)

def normalize_to_origin(pcd):
    print(f'original: {pcd.get_max_bound()=}, {pcd.get_min_bound()=}')
    center =pcd.get_min_bound()+((pcd.get_max_bound() -pcd.get_min_bound())/2)
    pcd.translate(-center)
    print(f'translated: {pcd.get_max_bound()=}, {pcd.get_min_bound()=}')
    return pcd

def save(file, to_write):
    with open(file,'wb') as f:
        pickle.dump(to_write,f)

def load(file):
    with open(file,'rb') as f:
        ret = pickle.load(f)
    return ret
        
def extend_seed_clusters(clusters_and_idxs:list[tuple],
                            src_pcd,
                            file_label,
                            k=200,
                            max_distance=.3,
                            cycles= 150,
                            save_every = 30,
                            draw_every = 10,
                            draw_progress = True,
                            debug = True):
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
    src_tree = sps.KDTree(src_pts)
    o3d.io.write_point_cloud('new_kd_search_tree.pcd',src_tree)
    # src_tree =o3d.read_point_cloud('new_kd_search_tree.pcd')
    src_pts = src_tree.data
    num_pts =  len(src_pts)
    complete = []
    cutoff_order = 10
    max_orders = defaultdict(int)
    iters = save_every
    # recreate = False
    print(f'running knn for idcs {idcs_to_run}')
    for cycle_num in range(cycles):
        print('start iter')
        if iters<=0:
            iters =save_every
            tree_pts = defaultdict(list)
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
            except Exception as e:
                breakpoint()
                print('error saving')
            

            # if recreate:
                # categorized_pts = arr(itertools.chain.from_iterable(tree_pts.values()))
                # src_pts = np.setdiff1d(src_pts, categorized_pts)
                # src_pts = arr(src_pts)
                # src_tree = sps.KDTree(src_pts)

            if draw_progress:
                print('starting draw')
                labels = list([x for x in range(len(tree_pts))])
                pts = list([[y[0] for y in x] for x in tree_pts.values()])
                tree_pcds = create_one_or_many_pcds(pts = pts, labels = labels)
                # if debug: breakpoint()
                # draw(tree_pcds)

                for idc, pcd in enumerate(tree_pcds):
                    if max_orders[idc]>10:
                        pcd.paint_uniform_color([1,0,0])
                draw(tree_pcds)
    
            del tree_pcds

        iters=iters-1
        print(f'querying {cycle_num}')
        for idx, cluster in clusters_and_idxs:
            if idx not in complete:
                
                if len(curr_pts[idx])>0:
                    dists,nbrs = src_tree.query(curr_pts[idx],k=k,distance_upper_bound= max_distance) #max(cluster_extent))
                    nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
                    nbr_pts = [nbr_pt for nbr_pt in src_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                    if len(nbr_pts)>0:
                        label_to_cluster = cluster_plus(nbr_pts,eps = .11, min_points=20,from_pts=True)
                        counts = [len(x) for x in label_to_cluster.values()]
                        order = len(arr(counts)[arr(counts)>20]) 
                        if order<=cutoff_order:
                            for nbr_pt in nbr_pts:
                               high_c_pt_assns[tuple(nbr_pt)] = (idx,order)
                            if max_orders[idx]< order:
                                max_orders[idx] = order
                            curr_pts[idx] = nbr_pts
                        else:
                            curr_pts[idx] = []
                    # print(f'{num_new_nbrs} new nbrs for cluster idx')
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


if __name__ =="__main__":
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
    label_to_clusters = load('new_lowc_ref_clusters.pkl')
    label_idls= label_to_clusters
    clusters = [(idc,lowc.select_by_index(idls)) for idc, idls in enumerate(label_idls)]
    # draw(arr(clusters)[:,1])
    # refined_clusters_idls = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194]
    # refined_clusters_labels = [label_to_clusters[idc] for idc in refined_clusters_idls]
    # refined_clusters= [clusters[idc] for idc in refined_clusters_idls]
    # other_clusters= [cluster for idc,cluster in clusters if idc not in [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194]]
    breakpoint()
    # o3d.io.write_point_cloud('new_lowc_clustered_pt3_20.pcd',pcd)
    # del pcd
    # highc_tree = sps.KDTree(np.asarray(highc.points()))
    # with open(f'new_lowc_ref_clusters.pkl','wb') as f: pickle.dump(refined_clusters,f)
    
    extend_seed_clusters(clusters,highc,'new_seeds')


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