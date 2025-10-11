
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
import tensorflow as tf

from geometry.zoom import filter_to_region_pcds, zoom_pcd
from viz.ray_casting import project_pcd
from set_config import config, log
from math_utils.general import (
    generate_grid
)
from geometry.point_cloud_processing import ( 
    clean_cloud,
    create_one_or_many_pcds,
    crop_by_percentile,
    cluster_plus,
    join_pcds
)
# from geometry.point_cloud_processing import ( 
# )
from utils.io import save,load
from viz.viz_utils import draw
from viz.color import *
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

def labeled_pts_to_lists(labeled_pts,
                            idc_to_label_map={},
                            file='',
                            draw_cycle=False,
                            save_cycle=False
                         ):
    tree_pts = defaultdict(list)
    for pt,idc in labeled_pts.items(): #tree_pts[idc_to_label_map.get(idc,idc)].append(pt)
        label = idc_to_label_map.get(idc,idc)
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
                            tb_every = 10,
                            order_cutoff = None,
                            exclude_pcd = None,
                            exclude_pts = None):
    """
        Takes a list of tuples defining clusters (e.g.[(label,cluster_pcd)...])
        and a source pcd
    """
    cluster_pts = [(idc, arr(cluster.points)) for idc,cluster in clusters_and_idxs]

    logdir = "src/logs/ext_seeds_by_color"
    writer = tf.summary.create_file_writer(logdir)
    # no_hue_pcds = [x for x in no_hue_pcds if x is not None]
    # target = no_hue_pcds[len(no_hue_pcds)-1]
    # with writer.as_default(): 
    #     for step in range(0,len(hue_pcds)):
    #         summary.add_3d(f'no_hue', to_dict_batch([no_hue_pcds[step]]), step=step+1, logdir=logdir)

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
        
    cvar = arr([x for x in range(len(clusters_and_idxs))])
    cluster_colors = plt.get_cmap('plasma')((cvar - cvar.min()) / (cvar.max() - cvar.min()))
    cluster_colors = cluster_colors[:, :3]


    try: 
        src_tree = load(f'{file_label}_kd_search_tree.pkl',dir ='')
    except Exception as e:
        log.info('creating KD search tree')
        if exclude_pts is None:
            if exclude_pcd is not None:
                exclude_pts = arr(exclude_pcd.points)
        
        src_pts = arr(src_pcd.points)
        src_tree = sps.KDTree(src_pts)
            # with open(f'{file_label}_kd_search_tree.pkl','wb') as f: pickle.dump(src_tree,f)
        dists,nbrs = src_tree.query(exclude_pts,
                                k=k,
                                distance_upper_bound= max_distance)
        num_pts = len(src_pts)
        nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
        nbr_pts = [nbr_pt for nbr_pt in src_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
        
        src_pcd = src_pcd.select_by_index(nbrs,invert=True)
        src_pts = arr(src_pcd.points)
        src_tree = sps.KDTree(src_pts)
        with open(f'{file_label}_kd_search_tree.pkl','wb') as f: pickle.dump(src_tree,f)
    colors = arr(src_pcd.colors)
    src_pts = src_tree.data

    file = f'{file_label}_in_process.pkl'
    num_pts =  len(src_pts)
    complete = []
    max_orders = defaultdict(int)
    save_iters = save_every
    draw_iters = draw_every
    tb_iters = tb_every
    end_early =False
    check_overlap = True
    # all_nbrs= []
    step = 1
    cylce_pts = []
    with writer.as_default(): 
        cluster_summary = to_dict_batch([src_pcd])
        summary.add_3d(f'ext_file', cluster_summary, step=step, logdir=logdir)
        for cycle_num in range(cycles):
            log.info('start iter')
            draw_cycle,save_cycle = draw_iters<=0, save_iters<=0 
            tensorboard_cycle = tb_iters<=0
            if tensorboard_cycle:
                step+=1
                for idl, nbr_list in enumerate(all_nbrs):
                    colors[nbr_list] = cluster_colors[idl]
                src_pcd.colors = o3d.utility.Vector3dVector(colors)
                cluster_summary = to_dict_batch([src_pcd])
                cluster_summary['vertex_positions'] = 0
                cluster_summary['vertex_normals'] = 0
                summary.add_3d(f'ext_file', cluster_summary, step=step, logdir=logdir)
                tb_iters = tb_every
            if draw_cycle or save_cycle:
                pt_lists, tree_pcds = labeled_pts_to_lists(high_c_pt_assns,
                                                            idc_to_label,
                                                            file,
                                                            draw_cycle,
                                                            save_cycle=False)
                # test = join_pcds([x for x in tree_pcds if x is not None])[0]
                # summary.add_3d(f'ext_file', to_dict_batch([test]), step=step, logdir=logdir)
                # del test
                if save_cycle: save_iters = save_every
                if draw_cycle: 
                    draw_iters = draw_every
                    draw(tree_pcds)
                cylce_pts = []

            if end_early:
                break
            save_iters=save_iters-1
            draw_iters=draw_iters-1
            tb_iters=tb_iters-1
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
                            
                            cylce_pts.extend(nbr_pts)
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
        return tree_pcds, all_nbrs
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
    col_min = collective.get_min_bound()
    col_max = collective.get_max_bound()
    # col_min = arr([ 34.05799866, 286.28399658, -21.90399933])
    # col_max = arr([189.47099304, 458.29800415,  40.64099884])
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

def pcds_from_extend_seed_file(file,pcd_idxs=[],return_pcds=True):
    with open(file,'rb') as f:
        # dict s.t. {cluster_id: [list_of_pts]}
        cell_completed = dict(pickle.load(f))
    labels = []
    orders = []
    log.info('read in source file, creating pcds...')
    if 'w_order' in file:
        pts_and_orders =[x for x in cell_completed.values()]
        pts =[[x[0] for x  in y] for y in pts_and_orders]
        orders = [x[1] for x in pts_and_orders]
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

    if return_pcds:
        cluster_pcds = create_one_or_many_pcds(pts)
        return cluster_pcds
    else:
        return pts, labels,orders
    
    
if __name__ =="__main__":
    pass
    # from geometry.reconstruction import get_neighbors_kdtree
    # from geometry.skeletonize import extract_skeleton
    
    # nn108 = cluster_roots_w_order_in_process('data/skio/results/skio/108_fin2_in_process.pkl')
    # # draw(nn108[0])

    # not_in108, _, ni_idxs = get_neighbors_kdtree(nn108_down,nnew107,k=200, dist=1)
    # newnew108 = new108.select_by_index(ni_idxs,invert=True)

    # low, _ = crop_by_percentile(old107,0,10)
    # draw(low)
    
    # test = zoom_pcd([[0,0,0],[1,1,1]],low)
    # clusters = cluster_plus(low, eps=.15, min_points=200,return_pcds=True,from_points=False, draw_result=True)
    
    # ### Gettting skeleton
    # # for seed,pcd in [(108,new108),(107,new107)]:
    # #     voxed_down = pcd.voxel_down_sample(voxel_size=.05)
    # #     uni_down = voxed_down.uniform_down_sample(3)
    # #     clean_pcd = remove_color_pts(uni_down,invert = True)
    # #     file = f'iso_seed{seed}'
    # #     skel_res = extract_skeleton(clean_pcd, max_iter = 3, cmag_save_file=file)
    # #     res.append(skel_res)

    # collective =  read_pcd('data/input/collective.pcd')
    # root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    # breakpoint()
    # seed_to_root_map = {seed_id: root_pcd for seed_id,root_pcd in zip(all_ids,root_pcds)}
    # seed_to_root_id = {seed: idc for idc,seed in enumerate(all_ids)}
    # unmatched_roots = [seed_to_root_id[seed] for seed in seed_to_exts.keys()]

