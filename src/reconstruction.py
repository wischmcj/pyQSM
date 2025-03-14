
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
from utils.io import save,load


from string import Template 

def get_pcd(pts,colors):
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(arr(pts))   
    pcd.colors = o3d.utility.Vector3dVector(arr(colors))
    return pcd

def recover_original_details(cluster_pcds,
                            file_prefix = Template('data/input/SKIO/part_skio_raffai_$idc.pcd'),
                            #= 'data/input/SKIO/part_skio_raffai',
                            save_result = False,
                            save_file = 'orig_dets',
                            file_num_base = 20000000,
                            file_num_iters = 39,
                            starting_num = 0):
    """
        Reversing the initial voxelization (which was done to increase algo speed)
        Functions by (for each cluster) limiting parent pcd to just points within the
            vicinity of the search cluster then performs a KNN search 
            to find neighobrs of points in the cluster
    """
    # defining files in which initial details are stored
    files = []
    if file_num_base:
        for idc in range(file_num_iters):
            if idc*file_num_base >= starting_num:
                files.append(file_prefix.substitute(idc =  idc*file_num_base ).replace(' ','_') )
    else:
        files = [file_prefix]
    pt_branch_assns = defaultdict(list)
    # iterating a bnd_box for each pcd for which
    #    we want to recover the orig details
    for idb, cluster_pcd in enumerate(cluster_pcds):
        # max_bnd = cluster_pcd.get_max_bound()
        # min_bnd = cluster_pcd.get_min_bound()
        bnd_box = cluster_pcd.get_oriented_bounding_box() 
        vicinity_pts = []
        v_colors = []
        skel_ids = []
        cnt=0
        # Limiting the search field to points in the general
        #   vicinity of the non-detailed pcd
        file_bounds=[]
        for file in files:
            print(f'checking file {file}')
            pcd = read_point_cloud(file)
            # section_max_bnd = pcd.get_max_bound()
            # section_min_bnd = pcd.get_min_bound()
            # file_bounds.append((section_min_bnd,section_max_bnd))
            # x_overlap = (section_min_bnd[0]>max_bnd[0] or section_max_bnd[0]<min_bnd[0])
            # y_overlap = (section_min_bnd[1]>max_bnd[1] or section_max_bnd[1]<min_bnd[1])
            pts = arr(pcd.points)
            if len(pts)>0: # and (x_overlap or y_overlap):
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
                    print(f'No points in vicinity from fole {file}')
            else:
                print(f'No points found in {file}')
                # del pcd
                # del vicinity_pt_ids
                # del pts
                # del all_pts_vect
            if len(vicinity_pts)>5000000:
                cnt=cnt+1
                print('Building pcd in parts due to a large volume of vicinity points')
                query_pts = arr(cluster_pcd.points)
                whole_tree = sps.KDTree(vicinity_pts)
                print('Finding neighbors in vicinity') 
                dists,nbrs = whole_tree.query(query_pts, k=500, distance_upper_bound= .3)
                del whole_tree
                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(vicinity_pts) ]
                try:
                    nbr_pts = arr(vicinity_pts)[nbrs]
                except Exception as e:
                    nbrs = [x for x in nbrs if x< len(vicinity_pts)-1 ]
                    nbr_pts = arr(vicinity_pts)[nbrs]
                    print(f'error {e} when getting neighbors in vicinity')
                nbr_colors = arr(v_colors)[nbrs]
                detailed_pcd = o3d.geometry.PointCloud()
                detailed_pcd.points = o3d.utility.Vector3dVector(nbr_pts)                
                # detailed_pcd.points = o3d.utility.Vector3dVector(nbr_pts)
                detailed_pcd.colors = o3d.utility.Vector3dVector(nbr_colors)  
                write_point_cloud(f'{save_file}_orig_detail{cnt}.pcd', detailed_pcd)
                del detailed_pcd
                vicinity_pts = []
                v_colors = []
                skel_ids = []
        if len(vicinity_pts)>0:
            try:
                print('Building pcd from points in vicinity')
                query_pts = arr(cluster_pcd.points)
                whole_tree = sps.KDTree(vicinity_pts)
                print('Finding neighbors in vicinity') 
                dists,nbrs = whole_tree.query(query_pts, k=400, distance_upper_bound= .3)
                del whole_tree
                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(vicinity_pts) ]
                try:
                    nbr_pts = arr(vicinity_pts)[nbrs]
                except Exception as e:
                    nbrs = [x for x in nbrs if x< len(vicinity_pts)-1 ]
                    nbr_pts = arr(vicinity_pts)[nbrs]
                    print(f'error {e} when getting neighbors in vicinity')
                nbr_colors = arr(v_colors)[nbrs]
                detailed_pcd = o3d.geometry.PointCloud()
                detailed_pcd.points = o3d.utility.Vector3dVector(nbr_pts)
                detailed_pcd.colors = o3d.utility.Vector3dVector(nbr_colors)  
                write_point_cloud(f'{save_file}_orig_detail.pcd', detailed_pcd)
                # detailed_pcds.append(detailed_pcd)
                del detailed_pcd
                # pt_branch_assns[idb] = arr(vicinity_pt_ids)[nbrs]
                
                # if idb%5==0 and idb>5:
                # complete = list([tuple((idb,nbrs)) for idb,nbrs in  pt_branch_assns.items()])
                # save(f'skeletor_branch{idb}_complete.pkl', complete) 
                # save(f'{save_file}_{idb}_orig_detail.pkl', complete)
            except Exception as e:
                print(f'error {e} getting clouds')
    # return detailed_pcd

def get_neighbors_kdtree(src_pcd,pcd, dist=0.05, k=750):
    query_pts = arr(pcd.points)
    src_pts = arr(src_pcd.points)
    whole_tree = sps.KDTree(src_pts)
    print('Finding neighbors in vicinity') 
    dists,nbrs = whole_tree.query(query_pts, k=k, distance_upper_bound= dist) 
    print('concatenating neighbors') 

    chained_nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(src_pts)]
    # nbr_pts = arr(vicinity_pts)[nbrs]
    print('selecting unique neighbors') 
    uniques = np.unique(chained_nbrs)
    print('building pcd') 
    pts = src_pts[uniques]
    colors = arr(src_pcd.colors)[uniques]

    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pts)   
    pcd.colors = o3d.utility.Vector3dVector(colors)    
    # if idb%5==0 and idb>5:
    # complete = list([tuple((idb,nbrs)) for idb,nbrs in  pt_branch_assns.items()])
    # save(f'{save_file}_{idb}_orig_detail.pkl', complete)

    ## the pcd of neighobrs of 'pcd' in src_pcd, a list of closest neighbors for each pt in pcd
    return  pcd, nbrs