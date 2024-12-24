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

# from geometry.skeletonize import extract_skeleton, extract_topology
# from utils.fit import cluster_DBSCAN, fit_shape_RANSAC, kmeans
# from utils.fit import choose_and_cluster, cluster_DBSCAN, fit_shape_RANSAC, kmeans
# from utils.lib_integration import find_neighbors_in_ball
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
    get_ball_mesh
)
from viz.viz_utils import iter_draw, draw

def loop_over_pcd_parts(file_prefix = 'data/input/SKIO/part_skio_raffai',
                    return_pcd_list =False,
                    return_single_pcd = False,
                    return_list = None,
                    return_lambda = None):
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
    if not isinstance(pts[0],list) and not isinstance(pts[0],np.array):
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(in_pts)
    pcd.colors = o3d.utility.Vector3dVector(in_colors)
    return pcd

def zoom(zoom_region, #=[(97.67433, 118.449), (342.1023, 362.86)],
        pts,
        colors = None,
        reverse=False):

    low_bnd = arr(zoom_region)[0,:]
    up_bnd =arr(zoom_region)[1,:]
    if isinstance(pts, list): pts = arr(pts)
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

def filter_to_region(ids_and_pts,
                        zoom_region):
    # Limiting to just the clusters near cluster[19]
    # factor = 10 # @40 reduces clusters 50%, @20 we get a solid 20 or so clusters in the
    # center_id = 126
    # zoom_region = [(clusters[center_id].get_max_bound()[0]+factor,    
    #                 clusters[center_id].get_min_bound()[0]-factor),     
    #                 (clusters[center_id].get_max_bound()[1]+factor,    
    #                 clusters[center_id].get_min_bound()[1]-factor)] # y max/min
    # zoom_region = [(),()]
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
    new_idcs = filter_to_region(pts,zoom_region)
    new_clusters = [(idc, cluster) for idc, cluster in clusters if idc in new_idcs]
    return  new_clusters

def load_clusters_get_details():
    # For loading results of KNN loop
    pfile = 'cluster126_fact10_0to50.pkl'
    with open(pfile,'rb') as f:
        tree_clusters = dict(pickle.load(f))
    labels = [x for x in  tree_clusters.keys()]
    pts =[x for x in tree_clusters.values()]

    # pts = pts[:3]
    # labels= labels[:3]

    cluster_pcds = create_one_or_many_pcds(pts, labels = labels)
    recover_original_detail(cluster_pcds)
    breakpoint()

def id_trunk_bases(pcd =None, 
            exclude_boundaries = None,
            load_from_file = True
):
    """Splits input cloud into a 'low' and 'high' component.
        Identifies clusters in the low cloud that likely correspond to 
    """
    print('getting low (slightly above ground) cross section')
    if load_from_file:
        lowc = o3d.io.read_point_cloud('low_cloud_all_16-18pct.pcd')
    else:
        lowc, lowc_ids_from_col = get_low_cloud(pcd, 16,18)
        lowc = clean_cloud(lowc)
        o3d.io.write_point_cloud('low_cloud_all_16-18pct.pcd',lowc)

    print('getting "high" portion of cloud')
    if load_from_file:
        highc= o3d.io.read_point_cloud('collective_highc_18plus.pcd')
    else:
        highc, highc_ids_from_col = get_low_cloud(pcd, 18,100)
        o3d.io.write_point_cloud('collective_highc_18plus.pcd',highc)
        draw(lowc)

    # midc, _ = get_low_cloud(collectivme, 16,30)
    
    print('Removing buildings')
    for region in exclude_boundaries:
        lowc = zoom_pcd( region, lowc, reverse=True)
        highc = zoom_pcd(region,  highc,reverse=True)

    print('clustering')
    if load_from_file:
        highc= o3d.io.read_point_cloud('collective_highc_18plus.pcd')
        with open('skio_labels_low_16-18_cluster_pt5-20.pkl','rb') as f: 
            labels = pickle.load(f)
    else:
        labels, colors = cluster_and_color(lowc,eps=.5, min_points=20)
        with open('skio_labels_low_16-18_cluster_pt5-20.pkl','wb') as f: 
            pickle.dump(labels,f)

    return lowc, highc, labels

def load_completed(cell_to_run_id):
    # Loading completed clusters and their 
    #  found nbr points from last run
    with open(f'all_complete.pkl','rb') as f: 
        completed = dict(pickle.load(f))

    completed_cluster_idxs = [idc for idc in completed.keys()]
    completed_cluster_pts = [pts for pts in completed.values()]
    grid=np.asarray([[[ 34.05799866, 286.28399658],
        [111.76449585, 343.6219991 ]],

       [[111.76449585, 286.28399658],
        [189.47099304, 343.6219991 ]],

       [[ 34.05799866, 343.6219991 ],
        [111.76449585, 400.96000163]],

       [[111.76449585, 343.6219991 ],
        [189.47099304, 400.96000163]],

       [[ 34.05799866, 400.96000163],
        [111.76449585, 458.29800415]],

       [[111.76449585, 400.96000163],
        [189.47099304, 458.29800415]]])

    files = []
    for cell_num in range(cell_to_run_id):
        files.append(f'cell{cell_num}_complete2.pkl')

    files.append(f'cell5_complete.pkl')
    for file in files:
        print(f'adding clusters found in {file}')
        with open(file,'rb') as f:
            # dict s.t. {cluster_id: [list_of_pts]}
            cell_completed = dict(pickle.load(f))
        
        # non_overlap_grid = grid[cell_num]
        all_idcs = len([idc for idc,_ in cell_completed.items()])
        pts_per_cluster = [len(pts) for (idc,pts) in cell_completed.items()]
        
        in_idcs = [idc for idc in cell_completed.keys() if idc not in completed_cluster_idxs]
        k_cluster_pts= [pts for idc, pts in cell_completed.items() if idc in in_idcs]
        # completed_cluster_keys = [x for x in completed.keys()]
        # in_idcs = filter_to_region(k_cluster_pts, non_overlap_grid)

        # breakpoint()
        num_new_clusters = len(in_idcs)
        pts_per_new_cluster = [len(pts) for pts in k_cluster_pts]
        num_new_pts = sum(pts_per_new_cluster)
        completed_cluster_idxs.extend(in_idcs)
        completed_cluster_pts.extend(k_cluster_pts)

        print(f'{all_idcs} total clusters found in grid')
        print(f'{pts_per_cluster} pts in each cluster')

        print(f'{num_new_clusters} new, complete clusters found in grid')
        print(f'{pts_per_new_cluster} pts in each new cluster')
        print(f'{num_new_pts} points categorized in grid')
        # added_k_ids = len(in_idcs)
        # print(f'{added_k_ids} complete clusters from grid cell {cell_num}')
    breakpoint()    
    return completed_cluster_idxs , completed_cluster_pts

def recover_original_detail(cluster_pcds):
    bnd_boxes = [pcd.get_oriented_bounding_box() for pcd in cluster_pcds]
    # alpha = 0.03
    # print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()

    # with open():

    base = 20000000
    file_prefix = 'data/input/SKIO/part_skio_raffai'
    for idb, bnd_box in enumerate(bnd_boxes):
        if idb>0:
            # contained_pts = []
            # all_colors = []
            # for i in range(39):
            #     num = base*(i+1) 
            #     file = f'{file_prefix}_{num}.pcd'
            #     print(f'checking file {file}')
            #     pcd = read_point_cloud(file)
            #     pts = arr(pcd.points)
            #     if len(pts)>0:
            #         cols = arr(pcd.colors)
            #         pts_vect = o3d.utility.Vector3dVector(pts)
            #         pt_ids = bnd_box.get_point_indices_within_bounding_box(pts_vect) 
            #         if len(pt_ids)>0:
            #             pt_values = pts[pt_ids]
            #             colors = cols[pt_ids]
            #             print(f'adding {len(pt_ids)} out of {len(pts)}')
            #             contained_pts.extend(pt_values)
            #             all_colors.extend(colors)
            #     else:
            #         print(f'No in region points found for {file}')
                    
            try:
                file = f'whole_clus/cluster_{idb}_all_points.pcd'
                print(f'writing pcd file {file}')
                test = o3d.io.read_point_cloud(file)
                contained_pts = arr(test.points)
                all_colors = arr(test.colors)

                # Reversing the initial voxelization done
                #   to make the dataset manageable 
                # KNN search each cluster against nearby pts in
                #   the original scan. Drastically increase detail

                query_pts = arr(cluster_pcds[idb].points)
                whole_tree = sps.KDTree(contained_pts)
                dists,nbrs = whole_tree.query(query_pts, k=750, distance_upper_bound= .2) 

                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x!= len(contained_pts)]
                print(f'{len(nbrs)} nbrs found for cluster {idb}')
                # for nbr_pt in nbr_pts:
                #     high_c_pt_assns[tuple(nbr_pt)] = idx

                # final_pts = np.append(arr(contained_pts)[nbrs])#,cluster_pcds[idb].points)
                # final_colors = np.append(arr(all_colors)[nbrs])#,cluster_pcds[idb].colors)
                final_pts =    arr(contained_pts)[nbrs]
                final_colors = arr(all_colors)[nbrs]
                wpcd = o3d.geometry.PointCloud()
                wpcd.points = o3d.utility.Vector3dVector(final_pts)
                wpcd.colors = o3d.utility.Vector3dVector(final_colors)  
                draw(wpcd)

                out_pts = np.delete(contained_pts,nbrs ,axis =0)
                out_pcd = o3d.geometry.PointCloud()
                out_pcd.points = o3d.utility.Vector3dVector(out_pts)
                wpcd.paint_uniform_color([0,0,1])                    
                out_pcd.paint_uniform_color([1,0,0])                
                draw([wpcd,out_pcd])

                breakpoint()
                wpcd.colors = o3d.utility.Vector3dVector(final_colors) 
                o3d.io.write_point_cloud(file, wpcd)

                breakpoint()

            except Exception as e:
                breakpoint()
                print(f'error {e} getting clouds')

def save(file, to_write):
    with open(file,'wb') as f:
        pickle.dump(to_write,f)

def load(file):
    with open(file,'rb') as f:
        ret = pickle.load(f)

if __name__ =="__main__":
    # test = compare_complete_to_in_progress(6)

#### Reading in data and preping clusters
    # o3d.io.write_point_cloud('collective.pcd',collective)
    # collective = o3d.io.read_point_cloud('collective.pcd')

    ## Identify and clean up cross section clusters 
    ## Filter out those unlikley trunk canidates (too small, too large)
    exclude_boundaries = [[ (77,350, 0),(100, 374,5.7)], # grainery
                            [(0, 350, 0), (70, 374, 5.7)] # big house 
                            ]
    lowc,highc,labels = id_trunk_bases(None, #collective, 
                                        exclude_boundaries)
    # clusters = filter_pcd_list(clusters)
    ## Define clusters based on labels 
    unique_vals, counts = np.unique(labels, return_counts=True)
    label_idls= [ np.where(labels ==val)[0] for val in unique_vals]
    clusters = [(idc,lowc.select_by_index(idls)) for idc, idls in enumerate(label_idls)]
   
#### Reading data from previous runs to exclude from run
    rerun_cluster_selection = False 
    if rerun_cluster_selection:
        completed_cluster_idxs , completed_cluster_pts = load_completed(3)
        save('completed_cluster_idxs.pkl',completed_cluster_idxs)
    else:
        completed_cluster_idxs = load('completed_cluster_idxs.pkl')
    nk_clusters= [(idc, cluster) for idc, cluster in clusters if idc not in completed_cluster_idxs]

####  Dividing clouds into smaller 
    from itertools import product
    # col_min = collective.get_min_bound()
    # col_max = collective.get_max_bound()
    col_min = arr([ 34.05799866, 286.28399658, -21.90399933])
    col_max = arr([189.47099304, 458.29800415,  40.64099884])
    col_lwh = col_max -col_min
    col_grid_num = arr((2,3,1)) # x,y,z
    grid_lines = [np.linspace(0,num,1) for num in col_grid_num]
    grid_lwh = col_lwh/col_grid_num
    ll_mults =  [np.linspace(0,num,num+1) for num in col_grid_num]
    llv = [minb + dim*mult for dim, mult,minb in zip(grid_lwh,ll_mults,col_min)]

    grid = arr([[[llv[0][0],llv[1][0]],[llv[0][1],llv[1][1]]],[[llv[0][1],llv[1][0]],[llv[0][2],llv[1][1]]],    
                [[llv[0][0],llv[1][1]],[llv[0][1],llv[1][2]]],[[llv[0][1],llv[1][1]],[llv[0][2],llv[1][2]]],    
                [[llv[0][0],llv[1][2]],[llv[0][1],llv[1][3]]],[[llv[0][1],llv[1][2]],[llv[0][2],llv[1][3]]]])
    ## w#e want a bit of overlap since nearby clusters sometime contest for points 
    overlap = grid_lwh/7    
    safe_grid = [[[ll[0]-overlap[0],ll[1]-overlap[1]],[ur[0]+overlap[0], ur[1]+overlap[1]]] for ll, ur in grid]
    ## Using the grid above to divide the cluster pcd list 
    ##  into easier to process parts 
    cell_clusters = []
    for idc, cell in enumerate(safe_grid):
        ids_and_clusters = filter_to_region_pcds(nk_clusters, cell)
        cell_clusters.append(ids_and_clusters)
    
####  KD tree with no prev. complete clusters 
    print('preparing KDtree')   
    if rerun_cluster_selection:
        # KD Tree reduction given completed clusters 
        highc_pts = [tuple(x) for x in highc.points]
        categorized_pts = [tuple(x) for x in itertools.chain.from_iterable(completed_cluster_pts)]
        highc_pts = set(highc_pts)-set(categorized_pts)
        highc_pts = arr([x for x in highc_pts])
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.asarray(highc_pts))
        # draw(pcd)
        # o3d.io.write_point_cloud('highc_incomplete.pcd',pcd)
        # del pcd
        highc_tree = sps.KDTree(np.asarray(highc_pts))
        with open(f'highc_KDTree.pkl','wb') as f: pickle.dump(highc_tree,f)
    else:
        with open('highc_KDTree.pkl','rb') as f: # all, cell0 and cell1 completed removed 
            highc_tree = pickle.load(f)
    


####  Running KNN algo to build trees
    # cell_to_run_id = 3
    # grid_wo_overlap =  grid[cell_to_run_id]
    for cell_to_run_id in [3,4,2]:
        ## Prepping algo input
        clusters_to_run = cell_clusters[cell_to_run_id]
        idcs_to_run = [idc for idc, cluster in clusters_to_run]
        cluster_pts = [(idc, arr(cluster.points)) for idc,cluster in clusters_to_run]

        high_c_pt_assns = defaultdict(lambda:-1) 
        curr_pts = [[]]*len(unique_vals)
        for idc, cluster_pt_list in cluster_pts:
            curr_pts[idc] = cluster_pt_list
            for pt in cluster_pt_list:
                high_c_pt_assns[tuple(pt)] = idc

        highc_pts = highc_tree.data
        num_pts =  len(highc_pts)
        complete = []

        iters = 30
        cycles = 200
        recreate = False
        draw_progress = False
        print(f'running knn for idcs {idcs_to_run}')
        for i in range(cycles):
            print('start iter')
            if iters<=0:
                iters =30
                tree_pts = defaultdict(list)
                for pt,idc in high_c_pt_assns.items(): 
                    tree_pts[idc].append(pt)
                print('created to_save. Pickling...')
                base_file = f'cell{cell_to_run_id}'
                to_write = list([tuple((k,pt_list)) for k,pt_list in tree_pts.items()])
                complete = list([tuple((idc,pt_list)) for idc,pt_list in to_write if idc in complete])
                save(base_file +'_clusters_in_progress2.pkl', to_write)
                save(base_file +'_complete2.pkl', complete)
                del tree_pts

                # if recreate:
                    # categorized_pts = arr(itertools.chain.from_iterable(tree_pts.values()))
                    # highc_pts = np.setdiff1d(highc_pts, categorized_pts)
                    # highc_pts = arr(highc_pts)
                    # highc_tree = sps.KDTree(highc_pts)

                if draw_progress:
                    print('starting draw')
                    labels = list([x for x in range(len(tree_pts))])
                    pts = list([x for x in tree_pts.values()])
                    tree_pcds = create_one_or_many_pcds(pts = pts, labels = labels)
                    draw(tree_pcds)
                    del tree_pcds

            iters=iters-1
            print(f'querying {i}')
            for idx, cluster in clusters_to_run:
                if idx not in complete:

                    dists,nbrs = highc_tree.query(curr_pts[idx],k=750,distance_upper_bound= .3) #max(cluster_extent))
                    nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
                    nbr_pts = [nbr_pt for nbr_pt in highc_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                    for nbr_pt in nbr_pts:
                        high_c_pt_assns[tuple(nbr_pt)] = idx
                    curr_pts[idx] = nbr_pts

                if len(curr_pts[idx])==0:
                    complete.append(idx)
                    print(f'{idx} added to complete')
        
        # run_again =False
        # cycles = 20
        # print('finish!')

        tree_pts = defaultdict(list)
        for pt,idc in high_c_pt_assns.items():  tree_pts[idc].append(pt)
        with open(f'cell{cell_to_run_id}_clusters_in_progress2.pkl','wb') as f: 
            to_write = list([tuple((k,pt_list)) for k,pt_list in tree_pts.items()])
            pickle.dump(to_write, f)

        with open(f'cell{cell_to_run_id}_complete2.pkl','wb') as f:
            complete =list([tuple((k,pt_list)) for k,pt_list in tree_pts.items() if k in complete])
            pickle.dump(complete,f)
            # ids_and_clusters = filter_to_region_pcds(completed, grid_wo_overlap)
            # cell_clusters.append(ids_and_clusters)