

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
                            save_file_base = 'orig_dets',
                            file_num_base = 2000000,
                            file_num_iters = 41,
                            starting_num = 0,
                            scale=1.1,
                            num_nbrs = 200,max_distance = .4,chunk_size =10000000):
    """
        Reversing the initial voxelization (which was done to increase algo speed)
        Functions by (for each cluster) limiting parent pcd to just points within the
            vicinity of the search cluster then performs a KNN search 
            to find neighobrs of points in the cluster
    """
    file_bounds = dict(load('skio_bounds.pkl'))
    # defining files in which initial details are stored
    files = []
    if file_num_base:
        for idc in range(file_num_iters):
            if idc*file_num_base >= starting_num:
                files.append(file_prefix.substitute(idc =  idc*file_num_base ).replace(' ','_') )
    else:
        files = [file_prefix]
    pt_branch_assns = defaultdict(list)
    files_written = []
    pcds = []
    # iterating a bnd_box for each pcd for which
    #    we want to recover the orig details
    for idb, cluster_pcd in enumerate(cluster_pcds):
        cluster_max = cluster_pcd.get_max_bound()
        cluster_min = cluster_pcd.get_min_bound()
        bnd_box = cluster_pcd.get_oriented_bounding_box() 
        bnd_box.color = [1,0,0]
        # draw([cluster_pcd,bnd_box])
        # bnd_box.scale(scale,center = bnd_box.center)
        vicinity_pts = []
        v_colors = []
        skel_ids = []
        cnt=0
        # Limiting the search field to points in the general
        #   vicinity of the non-detailed pcd\
        for file in files:
            bounds = file_bounds.get(file,[cluster_min,cluster_max])
            # file_min,file_max = bounds[0], bounds[1]
            print(f'checking file {file}')
            # l_overlap = all([a>b for a,b in zip(cluster_min,file_min)]) and all([a<b for a,b in zip(cluster_min,file_max)])
            # r_overlap = all([a>b for a,b in zip(cluster_max,file_min)]) and all([a<b for a,b in zip(cluster_max,file_max)])
            l_overlap=True
            r_overlap=True
            pcd = read_point_cloud(file)
            # draw([cluster_pcd,bnd_box,pcd])
            if l_overlap or r_overlap: 
                # pcd = read_point_cloud(file)
                # x_overlap = (any([a<b for a,b in zip(minb,min_bnd)]) and 1==1)
                # y_overlap = (bounds[1]>max_bnd[1] or section_max_bnd[1]<min_bnd[1])
                pts = arr(pcd.points)
                if len(pts)>0: # and (x_overlap or y_overlap):
                    cols = arr(pcd.colors)
                    all_pts_vect = o3d.utility.Vector3dVector(pts)
                    vicinity_pt_ids = bnd_box.get_point_indices_within_bounding_box(all_pts_vect) 
                    v_pt_values = pts[vicinity_pt_ids]
                    pts_gr_zero = np.where(v_pt_values[:,2]>0)[0]
                    vicinity_pt_ids = arr(vicinity_pt_ids)[pts_gr_zero]
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
            else:
                print(f'No overlap between {bounds} and cluster with {cluster_min=}, {cluster_max=}')
                # del pcd
                # del vicinity_pt_ids
                # del pts
                # del all_pts_vect
            if len(vicinity_pts)> chunk_size:
                print(f'Building pcd in parts due to a large volume of vicinity points: {len(vicinity_pts)}')
                all_vicinity_pts = vicinity_pts
                all_vicinity_colors = v_colors
                for chunk in range(int(((len(vicinity_pts)-len(vicinity_pts)%chunk_size)/chunk_size )+1)):
                    cnt=cnt+1
                    end= (chunk+1)*chunk_size
                    if end>=len(all_vicinity_pts): end= len(all_vicinity_pts)-1
                    start =chunk*chunk_size
                    vicinity_pts = all_vicinity_pts[start:end]
                    v_colors = all_vicinity_colors[start:end]
                    print(f'range {start} to {end}, out of {len(vicinity_pts)}, {chunk=} {cnt=}')
                    if len(vicinity_pts)>0:
                        detailed_pcd = o3d.geometry.PointCloud()
                        detailed_pcd.points = o3d.utility.Vector3dVector(vicinity_pts)
                        detailed_pcd.colors = o3d.utility.Vector3dVector(v_colors) 
                        # draw(detailed_pcd)
                        # draw([detailed_pcd,cluster_pcd])
                        # draw([detailed_pcd,cluster_pcd,bnd_box])
                        query_pts = arr(cluster_pcd.points)
                        whole_tree = sps.KDTree(vicinity_pts)
                        print('Finding neighbors in vicinity') 
                        dists,nbrs = whole_tree.query(query_pts, k=num_nbrs, distance_upper_bound= max_distance)
                        print(f'{len(nbrs)} nbrs found') 
                        nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(vicinity_pts) ]
                        print(f'{len(nbrs)} valid nbrs found') 
                        print('extracting nbr pts') 

                        nbr_pts = arr(vicinity_pts)[nbrs]
                        detailed_pcd = o3d.geometry.PointCloud()
                        detailed_pcd.points = o3d.utility.Vector3dVector(arr(nbr_pts))
                        print('extracting nbr colors') 
                        nbr_colors = arr(v_colors)[nbrs]                
                        # detailed_pcd.points = o3d.utility.Vector3dVector(nbr_pts)
                        detailed_pcd.colors = o3d.utility.Vector3dVector(arr(nbr_colors)) 

                        print(f'adding pcd {detailed_pcd}')
                        pcds.append(detailed_pcd)
                        del whole_tree
                        try:
                            save_file = f'{save_file_base}_orig_detail{cnt}.pcd'
                            write_point_cloud(save_file, detailed_pcd)
                            files_written.append(save_file)
                        except Exception as e:
                            breakpoint()
                            print(f'error writing pcd {e}')

                        del detailed_pcd
                        
                        vicinity_pts = []
                        v_colors = []
        # save('skio_bounds.pkl', file_bounds)
        if len(vicinity_pts)>0:
            try:
                # detailed_pcd = o3d.geometry.PointCloud()
                # detailed_pcd.points = o3d.utility.Vector3dVector(vicinity_pts)
                # detailed_pcd.colors = o3d.utility.Vector3dVector(v_colors) 
                # draw(detailed_pcd)
                # draw([detailed_pcd,cluster_pcd])
                # draw([detailed_pcd,cluster_pcd,bnd_box])
                print('Building pcd from points in vicinity')
                query_pts = arr(cluster_pcd.points)
                whole_tree = sps.KDTree(vicinity_pts)
                print('Finding neighbors in vicinity') 
                dists,nbrs = whole_tree.query(query_pts, k=num_nbrs, distance_upper_bound= max_distance)
                
                print(f'{len(nbrs)} nbrs found')
                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(vicinity_pts) ]
                print(f'{len(nbrs)} valid nbrs found') 
                if len(nbrs)>0:
                    try:
                        nbr_pts = arr(vicinity_pts)[nbrs]
                    except Exception as e:
                        nbrs = [x for x in nbrs if x< len(vicinity_pts)-1 ]
                        nbr_pts = arr(vicinity_pts)[nbrs]
                        print(f'error {e} when getting neighbors in vicinity')
                    nbr_colors = arr(v_colors)[nbrs]
                    detailed_pcd = o3d.geometry.PointCloud()
                    detailed_pcd.points = o3d.utility.Vector3dVector(arr(nbr_pts))
                    detailed_pcd.colors = o3d.utility.Vector3dVector(arr(nbr_colors) )
                    print(f'adding pcd {detailed_pcd}')
                    # pcds.append(detailed_pcd)
                    del whole_tree
                try:
                    print(f'writing pcd {detailed_pcd}')
                    save_file = f'{save_file_base}_orig_detail.pcd' if cnt==0 else f'{save_file_base}_orig_detail{cnt}.pcd'
                    print(f'writing to {save_file}')
                    write_point_cloud(save_file, detailed_pcd)
                    print(f'wrote to {save_file}')
                    files_written.append(save_file)
                except Exception as e:
                    breakpoint()
                    print(f'error writing pcd {e}')
                # detailed_pcds.append(detailed_pcd)
                # pt_branch_assns[idb] = arr(vicinity_pt_ids)[nbrs]
                
                # if idb%5==0 and idb>5:
                # complete = list([tuple((idb,nbrs)) for idb,nbrs in  pt_branch_assns.items()])
                # save(f'skeletor_branch{idb}_complete.pkl', complete) 
                # save(f'{save_file}_{idb}_orig_detail.pkl', complete)
            except Exception as e:
                breakpoint()
                print(f'error finding pts in final vicinity pcd {e}')
    # if len(files_written)>0:
    #     print(f'aggregating files written')
    #     pcds = []
    #     for save_file in files_written:
    #         try:
    #             pcds.append(read_point_cloud(save_file))
    #         except Exception as e:
    #             print(f'couldnt read in {save_file}: {e}')
    #     pcd = o3d.geometry.PointCloud()
    #     pts = [arr(pcd.points) for pcd in pcds]
    #     colors = [arr(pcd.colors) for pcd in pcds]
    #     pcd.points = o3d.utility.Vector3dVector(pts)
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    #     try:
    #         write_point_cloud(f'{save_file_base}_orig_detail.pcd', pcd)
    #     except Exception as e:
    #         breakpoint()
    #         print(f'error writing pcd {e}')
    #     return pcd
    # else:
    return pcds
        

    # return detailed_pcd

def get_neighbors_kdtree(src_pcd,pcd, dist=0.05, k=750, return_pcd = True):
    query_pts = arr(pcd.points)
    src_pts = arr(src_pcd.points)
    whole_tree = sps.KDTree(src_pts)
    print('Finding neighbors in vicinity') 
    dists,nbrs = whole_tree.query(query_pts, k=k, distance_upper_bound= dist) 
    print('concatenating neighbors') 

    if return_pcd:
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
        return pcd, nbrs
    else:
        return dists,nbrs
    