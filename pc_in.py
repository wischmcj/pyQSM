# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np
import copy
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from open3d.visualization import draw_geometries
from open3d.io import read_point_cloud, write_point_cloud
from open3d.geometry import TriangleMesh

from itertools import chain 
from collections import defaultdict

from src.fit import cluster_DBSCAN, fit_shape, kmeans
from src.point_cloud_processing import crop, get_percentile, orientation_from_norms
from src.viz import iter_draw
from src.utils import get_center, get_radius, get_angles, rotation_matrix_from_arr, unit_vector, get_k_smallest, get_lowest_points


skeletor ="/code/code/Research/lidar/converted_pcs/skeletor.pts"
s27 ="/code/code/Research/lidar/converted_pcs/Secrest27_05.pts"
s32 ="/code/code/Research/lidar/converted_pcs/Secrest32_06.pts"

s27d = "s32_downsample_0.04.pcd"
s32d = "s32_downsample_0.04.pcd"

config= {

    'whole_voxel_size':  .02 
    ,'neighbors':6
    ,'ratio':4
    ,'iters': 3
    ,'voxel_size':  None
    # stem
    ,'angle_cutoff':  20
    ,'stem_voxel_size':  None #.04
    ,'post_id_stat_down':  True
    ,'stem_neighbors':10
    ,'stem_ratio':2
    ,'stem_iters':  3 
    # trunk 
    ,'num_lowest':  2000
    ,'trunk_neighbors':  10
    ,'trunk_ratio':  0.25
    #DBSCAN
    ,'epsilon':  0.1
    ,'min_neighbors':  10
    #sphere
    ,'min_sphere_radius':  0.01
    ,'max_radius':  1.5
    ,'radius_multiplier':  1.75
    ,'dist':  .07
    ,'bad_fit_radius_factor':  2.5
    ,'min_contained_points':  8
}

def draw(pcds,raw = True, **kwargs):
    if raw: 
         draw_geometries(pcds)
    else:
        draw_geometries(pcds, mesh_show_wireframe=True, zoom=0.7,front=[0,2,0], lookat=[3,-3,4], up=[0,-1,1 ],**kwargs)


def filter_by_norm(pcd, angle_thresh=10):
    norms = np.asarray(pcd.normals) 
    angles = np.apply_along_axis(get_angles,1,norms)
    angles = np.degrees(angles)
    stem_idxs = np.where((angles>-angle_thresh) & (angles<angle_thresh))[0]
    stem_cloud = pcd.select_by_index(stem_idxs)
    return stem_cloud


def clean_cloud(pcd, voxels = None,
                neighbors=20, ratio=2.0,
                iters=3):
    """Reduces the number of points in the point cloud via 
        voxel downsampling. Reducing noise via statistical outlier removal.
    """
    if voxels:
        print("Downsample the point cloud with voxels")
        print(f"orig {pcd}")
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.04)
        print(f"downed {voxel_down_pcd}")
    else: 
        voxel_down_pcd = pcd

    print("Statistical oulier removal")
    for i in range(iters):
        neighbors= neighbors*1.5
        ratio = ratio/1.5
        _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=.1)
        voxel_down_pcd = voxel_down_pcd.select_by_index(ind)
    return voxel_down_pcd


def highlight_inliers(pcd, inlier_idxs, color = [1.0, 0, 0], draw =False):
    inlier_cloud = pcd.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color(color)
    if draw:
        outlier_cloud = pcd.select_by_index(inlier_idxs, invert=True)
        draw([inlier_cloud, outlier_cloud])
    return inlier_cloud

def get_neighbors_in_tree(sub_pcd_pts, full_tree, radius):
    trunk_tree = KDTree(sub_pcd_pts)
    pairs = trunk_tree.query_ball_tree(full_tree, r=radius)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    return neighbors


def evaluate_axis(pcd):

    # Get normals and align to Z axis
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.1, max_nn=20))
    pcd.normalize_normals()
    norms = np.array(pcd.normals)
    axis_guess = orientation_from_norms(norms,  samples = 100, max_iter = 1000 )
    return axis_guess

def z_align_and_fit(pcd, axis_guess, **kwargs):
    R_to_z = rotation_matrix_from_arr(unit_vector(axis_guess),[0,0,1])
    R_from_z = rotation_matrix_from_arr([0,0,1],unit_vector(axis_guess))
    #align with z-axis
    pcd_r = copy.deepcopy(pcd)
    pcd_r.rotate(R_to_z)
    #approx via circle
    mesh, _, inliers, fit_radius, _ = fit_shape(pcd = pcd_r,**kwargs)
    # Rotate mesh and pcd back to original orientation'
    # pcd.rotate(R_from_z)
    if mesh is None:
        # draw([pcd])
        return mesh, _, inliers, fit_radius , _
    mesh_pts = mesh.sample_points_uniformly(1000)
    mesh_pts.paint_uniform_color([0,1.0,0])
    mesh_pts.rotate(R_from_z) 
    draw([mesh_pts,pcd])
    return mesh, _, inliers, fit_radius, _ 

def ball_based_search(base_pts, points_to_search,
                            points_idxs, radius = None,
                            center = None):
    if not center:
        center = get_center(base_pts)
    if not radius:
        radius = get_radius(base_pts)*config['radius_multiplier']

    if radius < config['min_sphere_radius']:
        radius = config['min_sphere_radius']
    if radius > config['max_radius']:
        radius = config['max_radius']
    print(f'{radius=}')

    center = [center[0],center[1],max(base_pts[:,2])]#- (radius/4)])

    full_tree=KDTree(points_to_search)
    neighbors = full_tree.query_ball_point(center, r=radius)
    res=[]
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(center_points[:,0], center_points[:,1], center_points[:,2], 'r')
    for results in neighbors:
        new_points = np.setdiff1d(results,points_idxs)
        res.extend(new_points)
    #     nearby_points = points_to_search[new_points]
    #     ax.plot(nearby_points[:,0], nearby_points[:,1], nearby_points[:,2], 'o')
    sphere = TriangleMesh.create_sphere(radius = radius)
    sphere.translate(center)
    # sphere_pts=sphere.sample_points_uniformly(500)
    # sphere_pts.paint_uniform_color([0,1,0])
    # ax.plot(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], 'o')
    # plt.show()
    return sphere, res

def choose_and_cluster(new_neighbors, main_pts,
                      cluster_type):
    returned_clusters = []
    try:
        nn_points = main_pts[new_neighbors]
    except Exception as e:
        breakpoint()
        print(f'error in choose_and_cluster {e}')
    if cluster_type =='kmeans':
        # in these cases we expect the previous branch 
        #     has split into several new branches. Kmeans is
        #     better at characterizing this structure
        print('clustering via kmeans')
        labels, returned_clusters = kmeans(nn_points,1)
        # labels = [idx for idx,_ in enumerate(returned_clusters)]
        # ax = plt.figure().add_subplot(projection='3d')
        # for cluster in returned_clusters: ax.scatter(nn_points[cluster][:,0], nn_points[cluster][:,1], nn_points[cluster][:,2], 'r')
        # plt.show()    
    if  cluster_type !='kmeans' or len(returned_clusters)<2:
        print('clustering via DBSCAN')
        labels, returned_clusters, noise = cluster_DBSCAN(new_neighbors,
                                                             nn_points, 
                                                             eps = config['epsilon'],
                                                             min_pts = config['min_neighbors'])


def sphere_step(curr_pts, last_radius, main_pcd,
                curr_pts_idxs, branch_order, branch_num,
                total_found, run=0,branches = [[]]
                ,id_to_num = defaultdict(int)
                ,cyls = []
                ,cyl_details=[]
                ,spheres = []):
    """
    1. Takes in points from a neighbor cluster found in previous run
         set found
    2. Approximates a center and radius for a cylinder bounding those points
        - stored this to a list representing the QSM
    3. Approimates a circle to surround the neighbors of the cluster points
        - 

    Returns:
        _description_
    """
    if branches == [[]]:
        branches[0].append(total_found)

    main_pts = np.asarray(main_pcd.points)  
    
    
    cyl_mesh= None
    fit_radius =0
    # try to fit a cylinder to previous neighbors via a 2D circle
    #    only keep cyl_mesh if fit radius is reasonable
    prev_neighbor_height = np.min(curr_pts[:,2])
    cyl_mesh, fit_pcd, inliers, fit_radius, axis = (
        fit_shape(pts= curr_pts, shape = 'cylinder',threshold=0.04, 
                    lower_bound=prev_neighbor_height,
                    max_radius=last_radius*config['radius_multiplier']))
    
    good_fit_found = (cyl_mesh is not None and 
                        fit_radius<config['bad_fit_radius_factor']*last_radius)

    if good_fit_found:
        cyls.append(cyl_mesh.sample_points_uniformly(500))
        center = get_center(curr_pts)
        cyl_details.append({'center':center,
                            'axis': axis,
                            'height': prev_neighbor_height,
                            'radius':fit_radius})

    # get all points within radius of the former neighbor
    #    group's center point.
    # Exclude any that have already been found
    sphere, new_neighbors = ball_based_search(curr_pts, 
                                                main_pts, 
                                                total_found)
    spheres.append(sphere)
    clusters = []
    if len(new_neighbors)>0:
        cluster_type = 'kmeans' if not good_fit_found else 'DBSCAN'
        # Cluster new neighbors to finpd potential branches 
        clusters = choose_and_cluster(np.asarray(new_neighbors), main_pts, cluster_type)
    if (clusters == [] or
        len(new_neighbors)<config['min_contained_points']):
        # print(f'Zero or few new neighbors found')
        # if last_radius*2<config['max_radius']:
        #     sphere_step(curr_pts, last_radius*2
        #                             ,main_pcd
        #                             ,curr_pts_idxs
        #                             ,branch_order
        #                             ,branch_num
        #                             ,total_found
        #                             ,run+1
        #                             ,branches
        #                             ,id_to_num
        #                             ,cyls
        #                             ,cyl_details
        #                             ,spheres)
        # else:
        #     print(f'radius too large to retry')
        #     mask = []
        #     for val in total_found:
        #         if val in curr_pts_idxs:
        #             mask.append(False)
        #         else:
        #             mask.append(True)
        #     total_found = np.asarray(total_found)[mask]
        #     return []
        return []
    else:
        if len(clusters[0])>2:
            try:
                test = iter_draw(clusters[1],main_pcd)
            except Exception as e:
                print(f'error in iterdraw {e}')

    for label, cluster_idxs in clusters:  
        # done here so all idxs are added to total_found
        #  prior to doing any more fitting, finding
        total_found.extend(cluster_idxs)

    returned_idxs = []
    returned_pts = []
    for label, cluster_idxs in clusters:
        cluster_branch = branch_order
        if label!=0:
            cluster_branch+=1
            branches.append([])

        branch_id = branch_num+cluster_branch
        print(f"{branch_id=}, {cluster_branch=}")
        
        cluster_dict = {cluster_idx:branch_id for cluster_idx in cluster_idxs}
        id_to_num.update(cluster_dict)
        branches[cluster_branch].extend(cluster_idxs)
        
        # curr_plus_cluster = np.concatenate([curr_pts_idxs, cluster_idxs]) 
        cluster_pcd_pts = np.asarray(main_pcd.points)[cluster_idxs]
        # center = get_center(sub_pcd_pts)
        cluster_radius = get_radius(cluster_pcd_pts)
        print(f'{cluster_radius=}')
        if cluster_radius <  config['min_sphere_radius']:
            cluster_radius = config['min_sphere_radius']
        if cluster_radius >  config['max_radius']:
            cluster_radius = config['max_radius']
        if cluster_radius< last_radius/2:
            cluster_radius = last_radius/2

        if len(cyls)%10 == 0:
            try: 
                test = main_pcd.select_by_index(total_found)
                draw([test])
                for cyl_pts in cyls: cyl_pts.paint_uniform_color([0,1,0])
                draw([test]+cyls)
                sphere_pts= [sph.sample_points_uniformly(500) for sph in spheres]
                for sphere_pt in sphere_pts: sphere_pt.paint_uniform_color([0,0,1])
                draw([test]+sphere_pts)

                not_found = main_pcd.select_by_index(returned_idxs)
                test.paint_uniform_color([0,1,0])
                not_found.paint_uniform_color([1,0,0])
                draw([test, not_found])
            except Exception as e:
                print(f'error in checkpoint_draw {e}')
            breakpoint()
        fit_remaining_branch = sphere_step(cluster_pcd_pts, cluster_radius, main_pcd,
                                    cluster_idxs, cluster_branch, 
                                    branch_num,
                                    total_found, run+1,
                                    branches
                                    ,id_to_num
                                    ,cyls
                                    ,cyl_details
                                    ,spheres)
        if fit_remaining_branch == []:
            print(f'returned ids {cluster_idxs}')
            returned_idxs.extend(cluster_idxs)
            returned_pts.extend(cluster_idxs)
        branch_num+=1 
    print('reached end of function, returning')
    print('')
    return branches, id_to_num, cyls, cyl_details



def find_normal(a, norms):
    for norm in norms[1:]:
        if np.dot(a,norm)<0.01:
           return norm

def find_low_order_branches():
    
    # ***********************
    # IDEAS FOR CLEANING RESULTS
    # Mutliple iterations of statistical outlier removal
    #    start with latge std ratio, and low num neighbors(4?)
    # Larger Voxel size for Stem Filtering - more representitive normals
    # Fit plane to ground, remove ground before finiding lowest

    # ***********************

    ## starting with trunk we check for neighbors within sphere radius
    ## then cluster those neighbors with dist being the minimum distance between
    ##   two clusters

    # Reading in cloud and smooth
    # pcd = read_point_cloud('27_pt02.pcd',print_progress=True)
    # pcd = pcd.voxel_down_sample(voxel_size=config['whole_voxel_size'])
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # write_point_cloud("27_pt02.pcd",pcd)
    
    # print('read in cloud')
    # stat_down=pcd
    # stat_down = clean_cloud(pcd, 
    #                         voxels=voxel_size, 
    #                         neighbors=neighbors, 
    #                         ratio=ratio,
    #                         iters = iters)
    
    stat_down = read_point_cloud("27_vox_pt02_sta_6-4-3.pcd") 
    print('cleaned cloud')
    stat_down_pts = np.asarray(stat_down.points)
    stat_down_cropped_idxs=crop(stat_down_pts,minz=np.min(stat_down_pts[:,2])+.5)
    stat_down = stat_down.select_by_index(stat_down_cropped_idxs)

    # voxel_down_pcd = stat_down.voxel_down_sample(voxel_size=0.04)
    stat_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    stem_cloud = filter_by_norm(stat_down,config['angle_cutoff'])
    print('IDd stem_cloud')
    
    if config['stem_voxel_size']:
        stem_cloud = stem_cloud.voxel_down_sample(voxel_size=config['stem_voxel_size'])
    if config['post_id_stat_down']:
        _, ind = stem_cloud.remove_statistical_outlier(nb_neighbors=config['stem_neighbors'],
                                                       std_ratio=config['stem_ratio'])
        stem_cloud = stem_cloud.select_by_index(ind)
        # stem_cloud =clean_cloud(stem_cloud, 
        #                     voxels=stem_voxel_size, 
        #                     neighbors=stem_neighbors, 
        #                     ratio=stem_ratio,
        #                     iters = stem_iters)
    # draw([stem_cloud])
    print('Cleaned stem_cloud')
    stat_down_pts = np.asarray(stat_down.points)
    not_so_low_idxs,_ = get_percentile(stat_down_pts, 0,.25)
    low_cloud = stat_down.select_by_index(not_so_low_idxs)
    low_cloud_pts = np.asarray(low_cloud.points)

    print('IDd trunk')
    # stem_tree = KDTree(stem_cloud.points)
    # neighbors = get_neighbors_in_tree(np.asarray(trunk_pcd.points),stem_tree,.1)
    # trunk_pts = np.asarray(stem_cloud.points)[neighbors]
    
    # neighbors = get_neighbors_in_tree(np.asarray(low_cloud.points),stat_down_tree,.1)
    sphere, neighbors = ball_based_search(low_cloud_pts, stat_down_pts, not_so_low_idxs)
    new_neighbors = np.setdiff1d(neighbors, not_so_low_idxs)
    total_found = np.concatenate([not_so_low_idxs,new_neighbors])

    nn_pcd = stat_down.select_by_index(new_neighbors)
    nn_pts = np.asarray(stat_down.points)[new_neighbors]
    mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape(pcd = nn_pcd, shape = 'circle')


    res, id_to_num, cyls, cyl_details = sphere_step(nn_pts, fit_radius, stat_down 
                                                        ,inliers, branch_order=0
                                                        ,branch_num=0
                                                        ,total_found=list(total_found)
                                                    )
    breakpoint()
    iter_draw(res[0], stem_cloud)

    branch_index = defaultdict(list)
    for  idx, branch_num in id_to_num.items(): 
        branch_index[branch_num].append(idx)
    try:
        iter_draw(list(branch_index.values()),stem_cloud )
    except Exception as e:  
        print('error'+ e)

    
    # converting ids from low_cloud stat reduction
    #   To ids in stem_cloud    
    # stem_pts  = list(map(tuple,np.asarray(stem_cloud.points)))
    # trunk_pts = list(map(tuple,np.asarray(trunk_pcd.points)))
    # trunk_stem_idxs = []
    # for idx, val in enumerate(stem_pts):
    #     if val in trunk_pts:
    #         trunk_stem_idxs.append(idx)

    # just_trunk = stem_cloud.select_by_index(trunk_stem_idxs)
    # draw([trunk_pcd])


    #draw([stem_cloud]+pcds)

    ## KDTree neighbor finding 
    # stem_pts = np.asarray(pcd.points)
    # full_tree = KDTree(stem_pts)
    # connd = full_tree.query(trunk_points, 50)
    # pairs = full_tree.query_pairs(r=.02)
    # trunk_connected = trunk_tree.sparse_distance_matrix(full_tree, max_distance=radius)
    # neighbors_idx =np.array(list(set(chain.from_iterable(connd[1]))))

    # tc_idxs = np.array(list(set(chain.from_iterable(trunk_connected))))
    # trunk_connected_cloud = stem_cloud.select_by_index(neighbors_idx)  
    #draw([trunk_connected_cloud])

    ## removing non dense areas of points
    # mesh = map_density(stem_cloud)
    # mesh, densities = TriangleMesh.create_from_point_cloud_poisson(stem_cloud, depth=4)
    # densities = np.asarray(densities)
    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    #draw([mesh])
    breakpoint()

    # TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))


if __name__ == "__main__":
    find_low_order_branches()

    # pcd =  read_point_cloud("27_vox_pt02_sta_6-4-3.pcd") 
    # pcd =  read_point_cloud("stem_cloud.pcd") 
    # labels = np.array( pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # draw([pcd])
