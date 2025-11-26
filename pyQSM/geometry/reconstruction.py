

import open3d as o3d
import scipy.spatial as sps
import numpy as np
from numpy import asarray as arr

import itertools

from collections import defaultdict
import open3d as o3d
import numpy as np
from numpy import asarray as arr

from open3d.io import read_point_cloud, write_point_cloud

from utils.io import save,load
from set_config import log


from string import Template 

def recover_original_details(cluster_pcds,
                            file_prefix = Template('data/input/SKIO/part_skio_raffai_$idc.pcd'),
                            #= 'data/input/SKIO/part_skio_raffai',
                            save_result = False,
                            save_file_base = 'orig_dets',
                            file_num_base = 20000000,
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
        cnt=0
        files_to_check = []
        any_gtr = lambda x,y: any([a>b for a,b in zip(x,y)])
        for file, (file_min,file_max) in file_bounds:
            l_overlap = any_gtr(cluster_min,file_min) or any_gtr(file_max,cluster_min)
            r_overlap = any_gtr(cluster_max,file_min) or any_gtr(file_max,cluster_max) 
            has_pts_gr_zero = file_max[2]>0.1
            if (l_overlap or r_overlap) and has_pts_gr_zero:
                files_to_check.append(file)
            else:
                log.warning(f'Excluding file: {file}')

        # Limiting the search field to points in the general
        #   vicinity of the non-detailed pcd\
        for file in files_to_check:
            # bounds = file_bounds.get(file,[cluster_min,cluster_max])
            print(f'checking file {file}')
            if (l_overlap or r_overlap) and len(pts_gr_zero)>0: 
                pcd = read_point_cloud(file)
                pts = arr(pcd.points)
                if len(pts)>0: # and (x_overlap or y_overlap):
                    cols = arr(pcd.colors)
                    all_pts_vect = o3d.utility.Vector3dVector(pts)
                    vicinity_pt_ids = bnd_box.get_point_indices_within_bounding_box(all_pts_vect) 
                    v_pt_values = pts[vicinity_pt_ids]
                    pts_gr_zero = np.where(v_pt_values[:,2]>0.1)[0]
                    vicinity_pt_ids = arr(vicinity_pt_ids)[pts_gr_zero]
                    if len(vicinity_pt_ids)>0:
                        v_pt_values = pts[vicinity_pt_ids]
                        colors = cols[vicinity_pt_ids]
                        print(f'adding {len(vicinity_pt_ids)} out of {len(pts)}')
                        vicinity_pts.extend(v_pt_values)
                        v_colors.extend(colors)
                    else:
                        print(f'No points in vicinity from {file=}')
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

def get_neighbors_kdtree(src_pcd, query_pcd=None,query_pts=None, kd_tree = None, dist=0.05, k=500, return_pcd = True):
    
    if query_pcd: query_pts = arr(query_pcd.points)
    src_pts = arr(src_pcd.points)
    if not kd_tree:
        kd_tree = sps.KDTree(src_pts)
    print('Finding neighbors in vicinity') 
    dists,nbrs = kd_tree.query(query_pts, k=k, distance_upper_bound= dist) 

    if return_pcd:
        print('concatenating neighbors') 
        chained_nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(src_pts)]
        if len(chained_nbrs)==0:
            return None, None, None
        # nbr_pts = arr(vicinity_pts)[nbrs]
        print('selecting unique neighbors') 
        uniques = np.unique(chained_nbrs)
        print('building pcd') 
        try:
            pts = src_pts[uniques]
        except Exception as e:
            print(f'error {e} when selecting unique neighbors')
            return None, None, None
        colors = arr(src_pcd.colors)[uniques]

        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(pts)   
        pcd.colors = o3d.utility.Vector3dVector(colors)    
        return pcd, nbrs, chained_nbrs
    else:
        return dists,nbrs
    

def overlap_voxel_grid(src_pts, comp_voxel_grid = None, source_pcd = None, invert = False):
    if comp_voxel_grid is None:
        comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source_pcd,voxel_size=0.2)

    log.info('querying voxel grid')
    queries = src_pts
    in_occupied_voxel_mask= comp_voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    num_in_occupied_voxel = np.sum(in_occupied_voxel_mask)
    log.info(f'{num_in_occupied_voxel} points in occupied voxels')
    if num_in_occupied_voxel == 0:
        return []
    if invert:
        not_in_occupied_voxel_mask = np.ones_like(in_occupied_voxel_mask, dtype=bool)
        not_in_occupied_voxel_mask[in_occupied_voxel_mask] = False
        uniques = np.where(not_in_occupied_voxel_mask)[0]
    # else:
    uniques = np.where(in_occupied_voxel_mask)[0]

    return uniques

def get_nbrs_voxel_grid(comp_pcd, 
                        comp_file_name,
                        tile_dir, 
                        tile_pattern,
                        invert=False,
                        out_folder='detail',
                        out_file_prefix='detail_feats',
                        ):
    from glob import glob
    import os
    log.info('creating voxel grid')
    comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(comp_pcd,voxel_size=0.1)
    log.info('voxel grid created')
    nbr_ids = defaultdict(list)
    all_data =  defaultdict(list)
    
    # dodraw = True
    files = glob(f'{tile_dir}/{tile_pattern}')
    pcd=None
    for file in files:
        file_name = file.split('/')[-1].replace('.pcd','').replace('.npz','')
        log.info(f'processing {file_name}')
        if '.pcd' in file:
            pcd = o3d.io.read_point_cloud(file)
            data = {'points': np.array(pcd.points), 'colors': np.array(pcd.colors)}
            data_keys = data.keys()
        else:
            data = np.load(file)
            data_keys = data.files
            log.info(f'{data_keys=}')

        # Determine if the boundaries of pcd and comp_pcd intersect at all
        # Compute bounding boxes for both point clouds and check for intersection
        pcd_min = np.min(data['points'], axis=0)
        pcd_max = np.max(data['points'], axis=0)
        comp_min = np.min(np.asarray(comp_pcd.points), axis=0)
        comp_max = np.max(np.asarray(comp_pcd.points), axis=0)
        # Intersection exists if, on all axes, the max of the lower bounds <= min of the upper bounds
        intersect = np.all((pcd_max >= comp_min) & (comp_max >= pcd_min))
        log.info(f'Bounding box intersection: {intersect}')
        if not intersect:
            log.info("Bounding boxes do not intersect, skipping file.")
            continue
        else:
            log.info("Bounding boxes intersect, processing file.")

        nbr_dir = f'{tile_dir}/color_int_tree_nbrs/{file_name}'
        uniques = overlap_voxel_grid(data['points'], comp_voxel_grid, invert=invert)

        if not os.path.exists(nbr_dir):
            os.makedirs(nbr_dir)
        np.savez_compressed(f'{nbr_dir}/{out_file_prefix}_{comp_file_name}.npz', nbrs=uniques)
        log.info('filtering data to neighbors') 
        filtered_data = {zfile:data[zfile][uniques] for zfile in data_keys}
        for datum_name, datum in filtered_data.items():
            if len(datum.shape) == 1:
                all_data[datum_name].append(datum)#[:,np.newaxis])
            else:
                all_data[datum_name].append(datum)

    log.info('Saving all data')
    for datum_name, datum in all_data.items():
        if len(datum[0].shape) == 1:
            all_data[datum_name] = np.hstack(datum)
        else:
            all_data[datum_name] = np.vstack(datum)


    np.savez_compressed(f'{out_folder}/{comp_file_name}.npz', **all_data)
    return all_data
