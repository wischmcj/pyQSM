# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np
import copy
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.spatial as sps 
# import scipy.cluster as spc 
# import matplotlib.colors as mcolors
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import calinski_harabasz_score
# from sklearn.metrics import davies_bouldin_score
# from sklearn.cluster import DBSCAN
# import pyransac3d as pyrsc
from open3d.visualization import draw_geometries as draw

from itertools import chain 
from collections import defaultdict

from fit import fit_shape, cluster_neighbors, kmeans
from vectors_shapes import orientation_from_norms
from viz import iter_draw
from utils import get_center, get_radius, get_angles, rotation_matrix_from_arr, unit_vector, get_k_smallest, get_lowest_points


skeletor ="/code/code/Research/lidar/converted_pcs/skeletor.pts"
s27 ="/code/code/Research/lidar/converted_pcs/Secrest27_05.pts"
s32 ="/code/code/Research/lidar/converted_pcs/Secrest32_06.pts"

s27d = "s32_downsample_0.04.pcd"
s32d = "s32_downsample_0.04.pcd"

# [points[i, 0], points[j, 0],points[k,0]], [points[i, 1], points[j, 1],points[k,1]], [points[i, 2], points[j, 2],points[k,2]]

# def read_xyz(filename):
#     """Reads an XYZ point cloud file.

#     Args:
#         filename (str): The path to the XYZ file.

#     Returns:
#         numpy.ndarray: A Nx3 array of point coordinates (x, y, z).
#     """


#     points = []
#     with open(filename, 'rb') as f:
#         lines = []
#         i=0
#         for line in f:
#             if i != 0:
#                 try:
#                     line = line.decode('unicode_escape')
#                     lines.append([x for x in line.split(' ')])
                    
#                     # attr, x, y, z = line.split()
#                     # points.append(list(map(float,[x, y, z])))
#                 except Exception as e:
#                     breakpoint()
#                     print(e)
#                     print(line)
#             i+=1

#     return lines
    
def map_density(pcd, remove_outliers=True):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    if remove_outliers:
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    # o3d.visualization.draw_geometries([density_mesh])
    return density_mesh

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
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

def get_neighbors_in_tree(sub_pcd_pts, full_tree, radius):
    trunk_tree = sps.KDTree(sub_pcd_pts)
    pairs = trunk_tree.query_ball_tree(full_tree, r=radius)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    return neighbors

def get_neighbors_by_point(points, pts, radius):
    full_tree=sps.KDTree(pts)
    pairs = full_tree.query_ball_point(points, r=radius*1)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], 'r')
    if isinstance(neighbors[0],np.int64):
        neighbors = [neighbors]
    for results in neighbors:
        nearby_points = pts[results]
        plt.plot(nearby_points[:,0], nearby_points[:,1], nearby_points[:,2], 'o')
    plt.show()
    breakpoint()
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


def sphere_step(sub_pcd_pts, radius, main_pcd,
                curr_neighbors, branch_order, branch_num,
                total_found, run=0,branches = [[]],
                # new_branches = [[]],
                min_sphere_radius = 0.1
                ,max_radius = 0.5
                ,radius_multiplier = 2
                ,dist =.07
                ,id_to_num = defaultdict(int)
                ,cyls = []
                ,cyl_details=[]):
    if branches == [[]]:
        branches[0].append(total_found)

    main_pts = np.asarray(main_pcd.points)  
    full_tree = sps.KDTree(main_pts)

    
    # get all points within radius of a current neighbor
    #   exclude any that have already been found
    # print('trying to find neighbors')
    # new_neighbors = get_neighbors_in_tree(sub_pcd_pts, full_tree,radius)
    center = get_center(sub_pcd_pts)
    get_neighbors_by_point(np.array([center]), main_pts, radius)
    breakpoint()
    print(f"found {len(new_neighbors)} new neighbors")
    new_neighbors = np.setdiff1d(new_neighbors, curr_neighbors)
    print(f"found {len(new_neighbors)} not in current")
    new_neighbors = np.setdiff1d(new_neighbors, np.array(total_found)) 
    print(f"found {len(new_neighbors)} not in total_found")
    
    if len(new_neighbors) == 0: 
        print(f'no new neighbors found')
        return []

    inliers_list=[]
    # in_cyl_neighbors = []

    prev_neighbor_height = np.max(np.asarray(sub_pcd_pts)[:,2])
    nn_pts = main_pts[new_neighbors]
    try:
        mesh, _, inliers, fit_radius, _ = fit_shape(pts= nn_pts, shape = 'circle',threshold=0.02, 
                                                    lower_bound=prev_neighbor_height,
                                                    max_radius=radius*1.5)
    except Exception as e: 
        print(f'fit_shape error {e}')
        mesh, inliers, fit_radius = None, None, None   
    
    if mesh is None: # or fit_radius>=max_radius:
        mesh=None
        to_cluster = new_neighbors
    if fit_radius >= radius*1.5:
        orig_in_cyl_neighbors= new_neighbors[inliers]
        orig_in_cyl_points = main_pts[orig_in_cyl_neighbors]
        res = kmeans(orig_in_cyl_points, 2)
        breakpoint()
    else:
        in_cyl_neighbors = new_neighbors[inliers]
        to_cluster =  np.setdiff1d(new_neighbors, np.array( new_neighbors[inliers])) 
        inliers_list.append(inliers)
        cyls.append(mesh)
        cyl_details.append((inliers, nn_pts, fit_radius))
        # neighbor_cloud = main_pcd.select_by_index(in_cyl_neighbors)     
        # mesh_pts = mesh.sample_points_uniformly()
        # mesh_pts.paint_uniform_color([1.0,0,0])
        # draw([neighbor_cloud,mesh_pts])
        # else:
        #     in_cyl_points= main_pts[to_cluster]
        #     labels, returned_clusters, noise = cluster_neighbors(in_cyl_neighbors, in_cyl_points, dist =dist)

    if len(to_cluster) != 0: 
        nn_points= main_pts[to_cluster]
        labels, returned_clusters, noise = cluster_neighbors(to_cluster, nn_points, dist =dist)
        if mesh!=None:
            labels = [(x+len(inliers_list) if x!= -1 else x) for x in list(labels)]
        elif labels == {-1}:
            branch_num+=1
            print(f'only noise found, returning')
            return []
    else:
        labels = []
        returned_clusters = []
    # sub_pcd.paint_uniform_color([0,0,0])

    if mesh!=None:
        addl_labels = list(range(len(inliers_list)))
        labels = addl_labels + labels
        returned_clusters=[in_cyl_neighbors]+returned_clusters

    # try:
    #     test = iter_draw(returned_clusters,main_pcd)
    # except Exception as e:
    #     print(f'error in iterdraw {e}')


    
    clusters = [x for x in zip(labels,returned_clusters) if x[0] != -1 and len(x[1])>4]
    # sets_of_neighbors.append(clusters)
    # noise = [x for x in zip(labels,returned_clusters) if x[0] == -1]
    for label, cluster_idxs in clusters:  
        total_found.extend(cluster_idxs)

    print(f'iterating over {len(clusters)} clusters')
    for label, cluster_idxs in clusters:
        cluster_branch = branch_order
        if label!=0:
            cluster_branch+=1
            branches.append([])
            # breakpoint()
        if label >2:
            break
        branch_id = branch_num+cluster_branch
        print(f"{branch_id=}, {cluster_branch=}")
        cluster_dict = {cluster_idx:branch_id for cluster_idx in cluster_idxs}
        id_to_num.update(cluster_dict)
        branches[cluster_branch].extend(cluster_idxs)
        
        curr_plus_cluster = np.concatenate([curr_neighbors, cluster_idxs]) 
        cluster_pcd_pts = np.asarray(main_pcd.points)[cluster_idxs]
        # cluster_cloud = main_pcd.select_by_index(np.asarray(cluster_idxs))
        # o3d.visualization.draw_geometries([cluster_cloud]) 
        if fit_radius:
            new_radius = fit_radius*radius_multiplier
        else:
            new_radius = radius

        if new_radius < min_sphere_radius:
            new_radius = min_sphere_radius
        if new_radius > max_radius:
            new_radius = max_radius
        print(f"{new_radius=}, fit_radius: {fit_radius}")
        # print(f"""len(curr_plus_cluster): {len(curr_plus_cluster)}, 
        #             len(cluster_idxs): {len(cluster_idxs)}, 
        #             len(curr_neighbors): {len(curr_neighbors)} 
        #             len(new_neighbors): {len(new_neighbors)}""")
        # main_less_foud = main_pcd.select_by_index(curr_plus_cluster,invert=True)
        # main_less_points = np.asarray(main_less_foud.points)
        # o3d.visualization.draw_geometries([main_less_foud])
        
        if len(cyls)%10 == 0:
            try:
                test = main_pcd.select_by_index(total_found)
                o3d.visualization.draw_geometries([test])
                o3d.visualization.draw_geometries(cyls)
                cyl_pts = [cyl.sample_points_uniformly(500) for cyl in cyls]
                for pcd in cyl_pts: pcd.paint_uniform_color([0,1,0])
                draw([test]+cyl_pts)
            except Exception as e:
                print(f'error in iterdraw {e}')
            breakpoint()
        sphere_step(cluster_pcd_pts, new_radius, main_pcd,
                                    curr_plus_cluster, cluster_branch, 
                                    branch_num,
                                    total_found, run+1,
                                    branches
                                    # ,new_branches[label]
                                    ,min_sphere_radius
                                    ,max_radius 
                                    ,radius_multiplier
                                    ,dist
                                    ,id_to_num
                                    ,cyls
                                    ,cyl_details)
        branch_num+=1 
    print('reached end of function, returning')
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
    
    # pcd clean
    whole_voxel_size = .02 
    neighbors=6
    ratio=4
    iters= 3
    voxel_size = None
    # stem
    angle_cutoff = 20
    stem_voxel_size = None #.04
    post_id_stat_down = True
    stem_neighbors=10
    stem_ratio=2
    stem_iters = 3 
    # trunk 
    num_lowest = 2000
    trunk_neighbors = 10
    trunk_ratio = 0.25

    #sphere
    min_sphere_radius = 0.01
    max_radius = .17
    radius_multiplier = 1.11
    dist = .07

    ## starting with trunk we check for neighbors within sphere radius
    ## then cluster those neighbors with dist being the minimum distance between
    ##   two clusters

    # Reading in cloud and smooth
    # pcd = o3d.io.read_point_cloud('27_pt02.pcd',print_progress=True)
    # pcd = pcd.voxel_down_sample(voxel_size=whole_voxel_size)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # o3d.io.write_point_cloud("27_pt02.pcd",pcd)
    
    # print('read in cloud')
    # stat_down=pcd
    # stat_down = clean_cloud(pcd, 
    #                         voxels=voxel_size, 
    #                         neighbors=neighbors, 
    #                         ratio=ratio,
    #                         iters = iters)
    
    stat_down = o3d.io.read_point_cloud("27_vox_pt02_sta_6-4-3.pcd") 
    print('cleaned cloud')
    # voxel_down_pcd = stat_down.voxel_down_sample(voxel_size=0.04)
    stat_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    stem_cloud = filter_by_norm(stat_down,angle_cutoff)
    print('IDd stem_cloud')
    
    if stem_voxel_size:
        stem_cloud = stem_cloud.voxel_down_sample(voxel_size=stem_voxel_size)
    if post_id_stat_down:
        _, ind = stem_cloud.remove_statistical_outlier(nb_neighbors=stem_neighbors,
                                                       std_ratio=stem_ratio)
        stem_cloud = stem_cloud.select_by_index(ind)
        # stem_cloud =clean_cloud(stem_cloud, 
        #                     voxels=stem_voxel_size, 
        #                     neighbors=stem_neighbors, 
        #                     ratio=stem_ratio,
        #                     iters = stem_iters)
    # draw([stem_cloud])
    print('Cleaned stem_cloud')

    # Finding the trunk of the tree, eliminating outliers
    mins, min_idxs = get_lowest_points(stem_cloud,num_lowest)
    too_low_pctile = np.percentile(mins, 20)
    not_so_low_idxs = min_idxs[np.where(mins>too_low_pctile)]
    low_cloud = stem_cloud.select_by_index(not_so_low_idxs)
    # test = stem_cloud.select_by_index(low_cloud)
    # o3d.visualization.draw_geometries([low_cloud])
    # draw([low_cloud])

    # low_cloud = low_cloud.voxel_down_sample(voxel_size=0.04)
    # _, ind = low_cloud.remove_statistical_outlier(nb_neighbors=5,std_ratio=4)
    # clean_low_cloud = low_cloud.select_by_index(ind)
    # _, ind = low_cloud.remove_statistical_outlier(nb_neighbors=10,std_ratio=2)
    # clean_low_cloud = low_cloud.select_by_index(ind)
    
    mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape(pcd = low_cloud, shape = 'circle')
    # draw([trunk_pcd])
    # low_cloud.paint_uniform_color([0,0,1.0])
    # draw([low_cloud,mesh_pts])

    print('IDd trunk')
    stem_tree = sps.KDTree(stem_cloud.points)
    neighbors = get_neighbors_in_tree(np.asarray(trunk_pcd.points),stem_tree,.1)
    trunk_pts = np.asarray(stem_cloud.points)[neighbors]
    # just_trunk = stem_cloud.select_by_index(inliers)
    # draw([just_trunk])

    # just_trunk.paint_uniform_color([1.0,0,0])
    # pts = np.asarray(just_trunk.points)
    # radius = get_radius(trunk_pts)
    # draw([stem_cloud.select_by_index(neighbors)])

    total_found = np.concatenate([neighbors, not_so_low_idxs])
    test = stem_cloud.select_by_index(total_found)
    # o3d.visualization.draw_geometries([test])
    res, id_to_num, cyls, cyl_details = sphere_step(trunk_pts, fit_radius, stem_cloud 
                        ,neighbors, branch_order=0
                        ,branch_num=0
                        ,total_found=list(total_found)
                        ,min_sphere_radius=min_sphere_radius
                        ,max_radius=max_radius
                        ,radius_multiplier=radius_multiplier
                        ,dist=dist
                    )
    breakpoint()
    iter_draw(res, stem_cloud)

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


    # o3d.visualization.draw_geometries([stem_cloud]+pcds)

    ## KDTree neighbor finding 
    # stem_pts = np.asarray(pcd.points)
    # full_tree = sps.KDTree(stem_pts)
    # connd = full_tree.query(trunk_points, 50)
    # pairs = full_tree.query_pairs(r=.02)
    # trunk_connected = trunk_tree.sparse_distance_matrix(full_tree, max_distance=radius)
    # neighbors_idx =np.array(list(set(chain.from_iterable(connd[1]))))

    # tc_idxs = np.array(list(set(chain.from_iterable(trunk_connected))))
    # trunk_connected_cloud = stem_cloud.select_by_index(neighbors_idx)  
    # o3d.visualization.draw_geometries([trunk_connected_cloud])

    ## removing non dense areas of points
    # mesh = map_density(stem_cloud)
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(stem_cloud, depth=4)
    # densities = np.asarray(densities)
    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    # o3d.visualization.draw_geometries([mesh])
    breakpoint()

    # o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))


if __name__ == "__main__":
    find_low_order_branches()

    # pcd =  o3d.io.read_point_cloud("27_vox_pt02_sta_6-4-3.pcd") 
    # pcd =  o3d.io.read_point_cloud("stem_cloud.pcd") 
    # labels = np.array( pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # draw([pcd])
