
import open3d as o3d
import scipy.spatial as sps
import numpy as np
from numpy import asarray as arr
import pickle
import itertools

from collections import defaultdict

import open3d as o3d
import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

from geometry.zoom_and_filter import filter_to_region_pcds, zoom_pcd
from set_config import config, log
from utils.math_utils import (
    generate_grid
)
from geometry.point_cloud_processing import crop_by_percentile ( 
    clean_cloud,
    create_one_or_many_pcds,
    crop_by_percentile,
    cluster_plus
)
# from geometry.point_cloud_processing import ( 
# )
from utils.io import save,load
from viz.viz_utils import draw
from viz.color import *

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
        log.info(f'Pickling {file}...')
        save(file, pt_lists)
    return pt_lists, tree_pcds


def extend_seed_clusters(clusters_and_idxs:list[tuple],
                            src_pcd,
                            file_label,
                            k=200,
                            max_distance=.3,
                            cycles= 150,
                            save_every = 10,
                            draw_every = 10,
                            order_cutoff = None):
    """
        Takes a list of tuples defining clusters (e.g.[(label,cluster_pcd)...])
        and a source pcd
    """
    cluster_pts = [(idc, arr(cluster.points)) for idc,cluster in clusters_and_idxs]


    # create dict to track point labels
    num_nbr_clusters=[]
    high_c_pt_assns = defaultdict(lambda:-1) 
    curr_pts = [[]]*len(clusters_and_idxs)
    curr_nbrs = [[]]*len(clusters_and_idxs)
    all_nbrs = [[]]*len(clusters_and_idxs)
    idc_to_label =  defaultdict() 
    label_to_idc =  defaultdict() 
    for idc, (label, cluster_pt_list) in enumerate(cluster_pts):
        num_nbr_clusters.append([])
        curr_pts[idc] = cluster_pt_list
        # label pts already in clusters
        for pt in cluster_pt_list: 
            high_c_pt_assns[tuple(pt)] = idc
            # BElow is how this is done if we also send order through
            # high_c_pt_assns[tuple(pt)] = (idc,0)
        idc_to_label[idc] = label
        label_to_idc[label] = idc

    # try: 
    #     src_tree = load(f'{file_label}_kd_search_tree.pkl',dir ='')
    # except Exception as e:
    #     log.info('creating KD search tree')
    src_pts = arr(src_pcd.points)
    src_tree = sps.KDTree(src_pts)
        # with open(f'{file_label}_kd_search_tree.pkl','wb') as f: pickle.dump(src_tree,f)
    
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
        log.info('start iter')
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

        if end_early:
            break
        save_iters=save_iters-1
        draw_iters=draw_iters-1
        log.info(f'querying {cycle_num}')
        for idc, (label, cluster) in enumerate(clusters_and_idxs):
            idx = label_to_idc[label]
            if idx not in complete:               
                if len(curr_pts[idx])>0:
                    # get neighbors in the vicinity of the cluster
                    dists,nbrs = src_tree.query(curr_pts[idx],
                                                k=k,
                                                distance_upper_bound= max_distance)
                    nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
                    nbr_pts = [nbr_pt for nbr_pt in src_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                    num_clusters=0
                    if cycle_num>0 and order_cutoff:
                        nbr_pcd = src_pcd.select_by_index(nbrs)
                        if 1==0: #idx == 10:
                            res = cluster_plus(nbr_pcd, eps=.15, min_points=20,return_pcds=True,from_points=False, draw_result=False)
                            num_clusters = len([x for x in res])
                            print(f' {cycle_num=}, {num_clusters=}, ')
                            draw(res[0])
                        else:
                            res = cluster_plus(nbr_pcd, eps=.15, min_points=20,return_pcds=False,from_points=False, draw_result=False)
                            num_clusters = len([x for x in res])
                            print(f' {cycle_num=}, {num_clusters=}, ')
                            # breakpoint()
                            num_nbr_clusters[idc].append(num_clusters)
                    # if check_overlap:
                    #     nbr_set = set(nbrs)
                    #     # overlap = chain.from_iterable([nbr_set.intersection(nbr_list) for nbr_list in all_nbrs])
                    #     overlaps = [(idnl,nbr_set.intersection(nbr_list)) for idnl,nbr_list in enumerate(all_nbrs)]
                    #     overlaps = [x for x in overlaps if len(x)>0]
                    #     if len(overlaps)>0: 
                    #         pt_lists, tree_pcds = labeled_pts_to_lists(high_c_pt_assns,idc_to_label,draw_cycle=True)
                    #     for overlap in overlaps:
                    #         overlap_pcd = src_pcd.select_by_index(overlap)
                    #         overlap_pcd.paint_uniform_color([1,0,1])
                    #         draw([overlap_pcd]+tree_pcds)
                    #         breakpoint()
                    #         log.info('breakpoint in overlap draw')
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
                            log.info(f'{idx} added to complete')
                    if num_clusters>(order_cutoff or 0):
                        complete.append(idx)
                        curr_pts[idx] = []
                        curr_nbrs[idx] = []
                else:
                    num_nbr_clusters[idc].append(0)
                if len(curr_pts[idx])==0:
                    complete.append(idx)
                    log.info(f'{idx} added to complete')
    
    log.info('finish!')
    try:
        pt_lists, tree_pcds = labeled_pts_to_lists(high_c_pt_assns,
                                                        idc_to_label,
                                                        file,
                                                        draw_cycle=True,
                                                        save_cycle=True)
        breakpoint()
        i,k = 0,35
        ax = plt.subplot()
        for idc, series in enumerate(num_nbr_clusters[i:k]): ax.plot(range(len(series)),series)
        plt.show()
        draw(tree_pcds[i:k])
        # plt.axvline(x=avg_dist, color='r', linestyle='--')
        # plt.axvline(x=avg_dist*2, color='r', linestyle='--')
        plt.show()
        return tree_pcds
    except Exception as e:
        breakpoint()
        log.info(f'error saving {e}')
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
    log.info('getting low (slightly above ground) cross section')
    low_lb,low_ub = low_percentiles[0],low_percentiles[1]
    hi_lb,hi_ub = high_percentiles[0],high_percentiles[1]
    lowc, lowc_ids_from_col = crop_by_percentile(pcd,low_lb,low_ub)
    lowc = clean_cloud(lowc)

    log.info('getting "high" portion of cloud')
    highc, highc_ids_from_col = crop_by_percentile(pcd,hi_lb,hi_ub)
    draw(highc)

    log.info('Removing buildings')
    for region in exclude_boundaries:
        test = zoom_pcd( region, test, reverse=True)
        highc = zoom_pcd(region,  highc,reverse=True)

    # with open('skio_labels_low_16-18_cluster_pt5-20.pkl','rb') as f: 
    #         labels = pickle.load(f)
    log.info('clustering')
    try:
        label_to_clusters = cluster_plus(lowc,eps=.3, min_points=20,return_pcds=False)
    except Exception as e:
        log.info(f'error {e} when clustering ')
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
    #     completed_cluster_idxs = load('completed_cluster_idxs.pkl',dir='')

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
    log.info('preparing KDtree')   
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

def pcds_from_extend_seed_file(file,pcd_idxs=[]):
    with open(file,'rb') as f:
        # dict s.t. {cluster_id: [list_of_pts]}
        cell_completed = dict(pickle.load(f))
    log.info('read in source file, creating pcds...')
    if 'w_order' in file:
        pts_and_orders =[x for x in cell_completed.values()]
        pts =[[x[0] for x  in y] for y in pts_and_orders]
    else:
        pts =[x for x in cell_completed.values()]
        pts =[[x for x  in y] for y in pts]
    labels = [x for x in cell_completed.keys()]
    if pcd_idxs!=[]:
        new_pts,new_labels = [],[]
        for pcd_idx in pcd_idxs:
            new_pts.append(pts[pcd_idx])
            new_labels.append(labels[pcd_idx])
        pts = new_pts
    cluster_pcds = create_one_or_many_pcds(pts)

    return cluster_pcds
        
def read_point_shift():
    shift=load('')

def inspect_others():
    
    # # reading in results from extend
    # rf_ext_pcds = pcds_from_extend_seed_file('new_seeds_w_order_in_progress.pkl')
    oth_ext_pcds = pcds_from_extend_seed_file('other_clusters_w_order_in_process.pkl')
    # root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    # Source seed cluster id to ext in 'other_clusters_w_order_in_process.pkl'
    oth_seed_to_ext = {40: 0, 41: 1, 44: 2, 45: 3, 48: 4, 49: 5, 51: 6, 52: 7, 54: 8, 62: 9, 63: 10, 64: 11, 65: 12, 66: 13, 77: 14, 78: 15, 79: 16, 85: 17, 86: 18, 88: 19, 90: 20, 121: 21, 122: 22, 123: 23, 125: 24, 126: 25, 128: 26, 144: 27, 146: 28, 147: 29, 149: 30, 150: 31, 152: 32, 153: 33, 156: 34, 157: 35, 181: 36, 182: 37}

    for pcd,seed_id_tup in zip(oth_ext_pcds,oth_seed_to_ext.items()):
        log.info(f'{seed_id_tup}')
        seed,idc = seed_id_tup
        draw(pcd)
        breakpoint()
            # draw(oth_ext_pcds[13])
    breakpoint()
    seeds = []
    clusters = []
    
def run_extend():
    # exclude_regions = {
    #         'building1' : [ (77,350,0),(100, 374,5.7)], 
    #         'building2' : [(0, 350), (70, 374)],
    #         'far_front_yard':[ (0,400),(400, 500)],
    #         'far_rside_brush':[ (140,0),(190, 500)],
    #         'lside_brush':[ (0,0),(77, 50 all_ids = rf_seeds + final_int_ids0)],
    #         'far_back_brush': [ (0,0),(200, 325)]
    # }
    [ (77,350,0),(100, 374,5.7)]
    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    all_ids = rf_seeds + final_int_ids
    all_ids.sort()

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

    # # #       and the clusters fed into extend seed clusters
    lowc  = read_pcd('new_low_cloud_all_16-18pct.pcd')
    # highc = read_pcd('new_collective_highc_18plus.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl',dir='')
    empty_pcd= o3d.geometry.PointCloud() 
    # # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
    c_trunk_clusters = create_one_or_many_pcds([arr(lc.points) for lc in trunk_clusters],labels=[k for k,v in label_to_clusters.items()])

    
    # rf_colored = arr(c_trunk_clusters)[rf_seeds]  
    # other_colored = arr(c_trunk_clusters)[~arr(refined_clusters)]
    
    other_clusters = [(idc,cluster) if idc in final_int_ids else (idc,empty_pcd) for idc,cluster in enumerate(c_trunk_clusters) ]
    # extend_seed_clusters(clusters_and_idxs = other_clusters,
    #                         src_pcd=highc,
    #                         file_label='other_clusters',cycles=250,save_every=20,draw_every=160, max_distance=.1)

if __name__ =="__main__":
    # in_ids = [1,2,3,4,5,6,7,8]
    # rf_ext_pcds = pcds_from_extend_seed_file('data/skio/exts/new_seeds_w_order_in_progress.pkl',[in_ids])

    # # in_seed_ids = [107,108,109,111,112,113,114,115,116,133,151]
    # oth_ext_pcds = pcds_from_extend_seed_file('data/skio/exts/other_clusters_w_order_in_process.pkl',[12])
    # pcd107 = pcds_from_extend_seed_file('data/skio/exts/rf_cluster_107_rebuild_in_process.pkl')
    # draw(pcd107)

    test  = read_pcd('data/skio/results/skio/full_ext_seed135_rf11_orig_detail.pcd')
    pcd = test.uniform_down_sample(15)
    from ray_casting import project_pcd
    breakpoint()
    mesh = project_pcd(pcd,.1,True)
    # # in_ids = [12]
    # # in_seed_ids = [33]
    # ## 12/33 and 5/113 share a pcd but are two trees

    lowc  = read_pcd('data/skio/inputs/new_low_cloud_all_16-18pct.pcd')


    final_113 = o3d.io.read_point_cloud('data/skio/results/skio/final_ext_seed113_rf5_orig_detail.pcd')
    # final_113 = final_113.uniform_down_sample(10)
    # final_33 = oth_ext_pcds.uniform_down_sample(10)
    # from reconstruction import get_neighbors_kdtree
    # nbrs_in_113, _, analouge_idxs = get_neighbors_kdtree(oth_ext_pcds[0],final_113,k=200, dist=.1)
    # # draw(nbrs_in_113)
    # not113 = oth_ext_pcds[0].select_by_index(analouge_idxs, invert=True)
    # draw(not113)
    # breakpoint()

    test = read__point_cloud('data/skio/results/skio/final_ext_not113_seed33_oc5_orig_detail.pcd')
    
    # collective  = read_pcd('data/skio/inputs/partitioning_search_area_collective.pcd')
  
    # draw(rf_ext_pcds+oth_ext_pcds)
    # breakpoint()
    
    new_113 = read_pcd('data/skio/results/skio/full_ext_seed113_rf5_orig_detail.pcd')
    # hue_pcds,no_hue_pcds =segment_hues(new_113,113,draw_gif=True, save_gif=False)
    # stripped = clean_cloud(hue_pcds[5])
    # seed ,_ = crop_by_percentile(stripped,0,4)
    # # draw(seed)
    # clusters = [(1,seed)]
    # tree_pcds = extend_seed_clusters(clusters,stripped, '113_split', k=50, max_distance=.1, cycles= 500, save_every = 200, draw_every = 200)
    # branch_nbrs, _, analouge_idxs = get_neighbors_kdtree(new_113,tree_pcds[0],k=800, dist=2)
    # non_113 = new_113.select_by_index(analouge_idxs,invert=True)
    # test = non_113.uniform_down_sample(10)
    # res = cluster_plus(test, eps=.5, min_points=1000,return_pcds=True,from_points=False, draw_result=True)
    breakpoint()


    # axes = o3d.geometry.create_mesh_coordinate_frame()
    # exclude_regions = {
    #         'building1' : [ (77,350,0),(100, 374,5.7)], 
    #         'building2' : [(0, 350), (70, 374)],
    #         'far_front_yard':[ (0,400),(400, 500)],
    #         'far_rside_brush':[ (140,0),(190, 500)],
    #         'lside_brush':[ (0,0),(77, 50 all_ids = rf_seeds + final_int_ids0)],
    #         'far_back_brush': [ (0,0),(200, 325)]
    # }
    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    all_ids = rf_seeds + final_int_ids
    all_ids.sort()

    seed_to_exts = { 
                    #### 17 and 19 are inconcequentially small,
                    ####  seed 110 is super small
                    ##  148,180  doesnt match to any rf
                    152: 0  # untested 51 root
                    ,151: 9  # untested 51 root
                    ,107: 1  # ext 1 is two trees
                    ,108: 1  #   and so correlates to seeds 107 and 108
                    ,109: 2
                    ,111: 3
                    ,112: 4  
                    ,113: 5 
                    ,114: 6
                    ,115: 7  # this is a good seed but the ext only goes up the trunk a couple feet
                    ,116: 8  # this is a good ext, but the seed is spanish moss
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
    unmatched_seeds = [seed for seed in rf_seeds if not seed_to_exts.get(seed)]
    #[151, 110, 113, 132, 148, 180, 189, 193]

    matched_rf_ext = [x for x in seed_to_exts.values()]#[0 ,1 ,1 ,2,3,4 ,6,7 ,8 ,9 ,10,11,12,13,14,15,16,18]

    collective =  read_pcd('data/input/collective.pcd')
    root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    breakpoint()
    seed_to_root_map = {seed_id: root_pcd for seed_id,root_pcd in zip(all_ids,root_pcds)}
    seed_to_root_id = {seed: idc for idc,seed in enumerate(all_ids)}
    unmatched_roots = [seed_to_root_id[seed] for seed in seed_to_exts.keys()]


    # # load all seed clusters 
    lowc  = read_pcd('new_low_cloud_all_16-18pct.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
    # # highc = read_pcd('new_collective_highc_18plus.pcd')
    # empty_pcd= o3d.geometry.PointCloud() 
    # # # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')

    all_seeds = {seed: 
                    (trunk_clusters[seed] ,
                      seed_to_root_map[seed], seed_to_root_id[seed]) for seed in rf_seeds}

    unmatched_rf_ext_idx = [idx for idx,pcd in enumerate(rf_ext_pcds) if idx not in [0 ,1 ,1 ,2,3,4 ,6,7 ,8 ,9 ,10,11,12,13,14,15,16,18] ]
    unmatched_rf_ext_pcds = [pcd for idx,pcd in enumerate(rf_ext_pcds) if idx not in [0 ,1 ,1 ,2,3,4 ,6,7 ,8 ,9 ,10,11,12,13,14,15,16,18] ]

    breakpoint()
