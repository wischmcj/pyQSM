# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np

import open3d as o3d
import matplotlib.pyplot as plt
import scipy.spatial as sps 
import scipy.cluster as spc 
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
from open3d.visualization import draw_geometries as draw

from itertools import chain 
from collections import defaultdict

from fit import fit_cylinder
from vectors_shapes import get_shape
from viz import iter_draw

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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_from_xy(v1):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v2 = [v1[0],v1[1],0]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_angles(tup,radians=False):
    a=tup[0]
    b=tup[1]
    c=tup[2]
    denom = np.sqrt(a**2 +b**2)
    if denom !=0:
        radians = np.arctan(c/np.sqrt(a**2 + b**2))
        if radians:
            return radians
        else:
            return np.degrees(radians)
    else:
        return 0
    
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

def get_k_smallest(arr,k):
    idx = np.argpartition(arr, k)
    return arr[idx[:k]], idx[:k]

def get_lowest_points(pcd,k):
    pts = np.asarray(pcd.points)
    z_vals = pts[:,2]
    k_mins, mins_idxs = get_k_smallest(z_vals,k)
    return k_mins, mins_idxs

def get_center(points, center_type = 'centroid'):
    if len(points[0]) !=3:
        breakpoint()
        print('not 3 points')
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    if center_type == 'centroid':
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid
    if center_type == 'middle':
        middle = middle(x), middle(y), middle(z)
        return middle

def get_radius(points, center_type = 'centroid'):
    center = get_center(points, center_type)
    xy_pts = points[:,:2]
    xy_center = center[:2]
    r = np.average([np.sqrt(np.sum((xy_pt - xy_center)**2)) for xy_pt in xy_pts])
    return r

def get_sphere(pts, 
                center = None, radius = None):
    new_center = center or get_center(pts)
    new_radius = radius or get_radius(pts)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=new_radius)
    sphere.translate(new_center)
    sphere_pts = sphere.sample_points_uniformly()
    sphere_pts.paint_uniform_color([0,1.0,0])
    return sphere_pts


def highlight_inliers(pcd, inlier_idxs, color = [1.0, 0, 0], draw =False):
    inlier_cloud = pcd.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color(color)
    if draw:
        outlier_cloud = pcd.select_by_index(inlier_idxs, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

def get_neighbors_in_tree(sub_pcd, full_tree, radius):
    trunk_tree = sps.KDTree(sub_pcd.points)
    pairs = trunk_tree.query_ball_tree(full_tree, r=radius)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    # neighbors.extend(sub_pcd_idxs)
    return neighbors

def spc_cluster_neighbors(points,current_clusters ):
    """
    https://www.comet.com/site/blog/how-to-evaluate-clustering-models-in-python/
    """
    silhouette_score_co = 0.3
    ch_max = 50
    clusters_to_try = [current_clusters-1, 
                        current_clusters, 
                        current_clusters+1]
    results = []
    codes,book = None,None
    for num in clusters_to_try:
        if num>0:
            codes, book = spc.vq.kmeans2(points, num)
            cluster_sizes = np.bincount(book)
            if num ==1:
                results.append(((codes,book,num),[.5,0,0]))
            else:
                sh_score =  silhouette_score(points,book)
                ch_score = calinski_harabasz_score(points,book)
                db_score = davies_bouldin_score(points,book)
                print(f'''num clusters: {num}, sizes: {cluster_sizes}, 
                            sh_score: {sh_score}, ch_score: {ch_score}, db_score: {db_score}''') 
                results.append(((codes,book,num),[sh_score,ch_score,db_score]))
    ret = results[0]
    for res in results:
        # if res[1][1] >= ch_max:
        #     pass
        if res[1][2]<.7 and res[1][0] >= silhouette_score_co:
            ret = res
    return ret[0]

def cluster_neighbors(pts_idxs, points,dist=.3, min_samples=5):
    clustering = DBSCAN(eps=dist, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    idxs = []
    noise = []
    # converting ids for 'points' to ids
    #   in main_pcd stored in pts_idxs
    for k in unique_labels:
        class_member_mask = labels == k
        if k == -1:
            idx_bool = class_member_mask & (core_samples_mask==False)
            pt_idxs = np.where((np.array(idx_bool)==True))
            noise = pts_idxs[pt_idxs]
        else:
            idx_bool = class_member_mask & core_samples_mask
            pt_idxs = np.where((np.array(idx_bool)==True))
            neighbor_idxs = pts_idxs[pt_idxs]
            idxs.append(neighbor_idxs)

    return unique_labels, idxs, noise

def cluster_and_draw(sub_pcd,new_neighbors,nn_points,total_found,main_pcd):
    labels, cluster_idxs, noise = cluster_neighbors(new_neighbors, nn_points,dist =.07)
    sub_pcd.paint_uniform_color([0,0,0])
    geos =[sub_pcd]
    # if ((current_clusters > 1 and -1 not in labels)
    #     or (current_clusters > 2  and -1 in labels)):
    if len(labels) <6:
        colors = [[0,1,0],[0,0,1],[1,0,0],[0,1,1],[1,0,1],[1,1,0]]
    else:
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
    i=0
    geos_label = ['']
    if labels == {-1}:
        cluster_idxs.append(noise)
    for label, idxs in zip(labels,cluster_idxs):
        total_found.extend(idxs)
        cluster_cloud = main_pcd.select_by_index(np.asarray(idxs))    
        col= colors[i]
        cluster_cloud.paint_uniform_color(mcolors.to_rgb(col))
        if len(idxs)!=0:
            geos.append(cluster_cloud)
            geos_label.append(label)
            i+=1
    o3d.visualization.draw_geometries(geos)

def sphere_step(sub_pcd, radius, main_pcd,
                curr_neighbors, branch_order, branch_num,
                total_found, run=0,branches = [[]],
                # new_branches = [[]],
                min_sphere_radius = 0.1
                ,max_radius = 0.5
                ,radius_multiplier = 2
                ,dist =.07
                ,id_to_num = defaultdict(int)):

    main_pts = np.asarray(main_pcd.points)  
    full_tree = sps.KDTree(main_pts)

    # get all points within radius of a current neighbor
    #   exclude any that have already been found
    # print('trying to find neighbors')
    new_neighbors = get_neighbors_in_tree(sub_pcd, full_tree,radius/2)
    # print(f"found {len(new_neighbors)} new neighbors")
    new_neighbors = np.setdiff1d(new_neighbors, curr_neighbors)
    # print(f"found {len(new_neighbors)} not in current")
    new_neighbors = np.setdiff1d(new_neighbors, np.array(total_found)) 
    # print(f"found {len(new_neighbors)} not in total_found")
    if len(new_neighbors) == 0: 
        return []


    # nn_pcd = main_pcd.select_by_index(new_neighbors)
    nn_pts = main_pts[new_neighbors]
    mesh, _, inliers = fit_cylinder(pts = nn_pts,threshold=0.08)
    
    neighbor_cloud = main_pcd.select_by_index(np.asarray(new_neighbors))    
    mesh_pts = mesh.sample_points_uniformly()
    mesh_pts.paint_uniform_color([1.0,0,0])
    draw([neighbor_cloud,mesh_pts])
    # trunk_pcd
    #next step needed to translate idxs in nn_pcd to idxs in main_pcd
    # in_cyl_neighbors = get_neighbors_in_tree(trunk_pcd,full_tree,radius/4)
    in_cyl_neighbors = new_neighbors[inliers]
    to_cluster =  np.setdiff1d(new_neighbors, np.array(in_cyl_neighbors)) 

    labels = []
    cluster_idxs = []
    if len(to_cluster) != 0: 
        nn_points= main_pts[to_cluster]
        labels, cluster_idxs, noise = cluster_neighbors(to_cluster, nn_points, dist =dist)
        labels = [(x+1 if x!= -1 else x) for x in list(labels)]
    labels.append(0)
    cluster_idxs.append(in_cyl_neighbors)

    sub_pcd.paint_uniform_color([0,0,0])
    geos =[sub_pcd]
    # if ((current_clusters > 1 and -1 not in labels)
    #     or (current_clusters > 2  and -1 in labels)):
    if len(labels) <6:
        colors = [[0,1,0],[0,0,1],[1,0,0],[0,1,1],[1,0,1],[1,1,0]]
    else:
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
    i=0
    geos_label = ['']
    if labels == {-1}:
        branch_num+=1
        return []
    for label, idxs in zip(labels,cluster_idxs):
        total_found.extend(idxs)
        cluster_cloud = main_pcd.select_by_index(np.asarray(idxs))    
        col= colors[i]
        cluster_cloud.paint_uniform_color(mcolors.to_rgb(col))
        if len(idxs)!=0:
            geos.append(cluster_cloud)
            geos_label.append(label)
            i+=1
    
    o3d.visualization.draw_geometries(geos)
    
    # points = main_pts[new_neighbors]
    # sphere_pts = get_sphere(points)
    # sphere_pts.paint_uniform_color([0,0,0])

    clusters = [x for x in zip(labels,cluster_idxs) if x[0] != -1 and len(x[1])>0]
    # sets_of_neighbors.append(clusters)
    noise = [x for x in zip(labels,cluster_idxs) if x[0] == -1]

    # is_first_run = len(branches.keys())==0
    # first_run = (branches == [[]])

    
    for label, cluster_idxs in clusters:
        cluster_branch = branch_order
        if label!=0:
            cluster_branch+=1
            branches.append([])
            # breakpoint()
        branch_id = branch_num+cluster_branch
        print(f"{branch_id=}, {cluster_branch=}")
        cluster_dict = {cluster_idx:branch_id for cluster_idx in cluster_idxs}
        id_to_num.update(cluster_dict)
        branches[cluster_branch].extend(cluster_idxs)
        
        curr_plus_cluster = np.concatenate([curr_neighbors, cluster_idxs]) 
        cluster_cloud = main_pcd.select_by_index(np.asarray(cluster_idxs))
        # o3d.visualization.draw_geometries([cluster_cloud]) 

        new_points = main_pts[cluster_idxs]
        new_radius = get_radius(new_points)*radius_multiplier
        if new_radius < min_sphere_radius:
            new_radius = min_sphere_radius
        if new_radius > max_radius:
            new_radius = max_radius
        print(f"""len(curr_plus_cluster): {len(curr_plus_cluster)}, len(cluster_idxs): {len(cluster_idxs)}, len(curr_neighbors): {len(curr_neighbors)} len(new_neighbors): {len(new_neighbors)}""")
        # main_less_foud = main_pcd.select_by_index(curr_plus_cluster,invert=True)
        # main_less_points = np.asarray(main_less_foud.points)
        # o3d.visualization.draw_geometries([main_less_foud])
        
        # if run%100 == 0:
        #     test = main_pcd.select_by_index(total_found)
        #     o3d.visualization.draw_geometries([test])
            # test = main_pcd.select_by_index(branches[0])
            # o3d.visualization.draw_geometries([test])
            # breakpoint()
        sphere_step(cluster_cloud, new_radius, main_pcd,
                                    curr_plus_cluster, cluster_branch, 
                                    branch_num,
                                    total_found, run+1,
                                    branches
                                    # ,new_branches[label]
                                    ,min_sphere_radius
                                    ,max_radius 
                                    ,radius_multiplier
                                    ,dist,
                                     id_to_num )
        branch_num+=1 
    return branches, id_to_num

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
    angle_cutoff = 15
    stem_voxel_size = None
    post_id_stat_down = True
    stem_neighbors=10
    stem_ratio=4
    # trunk 
    num_lowest = 3000
    trunk_neighbors = 10
    trunk_ratio = 0.25

    #sphere
    min_sphere_radius = 0.1
    max_radius = 1.4
    radius_multiplier = 2
    dist = .07

    ## starting with trunk we check for neighbors within sphere radius
    ## then cluster those neighbors with dist being the minimum distance between
    ##   two clusters

    # Reading in cloud and smooth
    pcd = o3d.io.read_point_cloud('27_pt02.pcd',print_progress=True)
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
    # draw([stem_cloud])
    print('Cleaned stem_cloud')

    # Finding the trunk of the tree, eliminating outliers
    _, min_idxs = get_lowest_points(stem_cloud,num_lowest)
    low_cloud = stem_cloud.select_by_index(min_idxs)
    # draw([low_cloud])

    low_cloud = low_cloud.voxel_down_sample(voxel_size=0.04)
    _, ind = low_cloud.remove_statistical_outlier(nb_neighbors=10,std_ratio=2)
    clean_low_cloud = low_cloud.select_by_index(ind)
    # draw([test])
    
    print('Fitting trunk cyl')
    mesh, trunk_pcd, inliers = fit_cylinder(pcd = clean_low_cloud)

    print('IDd trunk')
    stem_tree = sps.KDTree(stem_cloud.points)
    neighbors = get_neighbors_in_tree(trunk_pcd,stem_tree,.1)
    just_trunk = stem_cloud.select_by_index(neighbors)

    just_trunk.paint_uniform_color([1.0,0,0])
    pts = np.asarray(just_trunk.points)
    radius = get_radius(pts)

    res, id_to_num = sphere_step(just_trunk, radius, stem_cloud 
                        ,neighbors, branch_order=0
                        ,branch_num=0
                        ,total_found=list(min_idxs)
                        ,min_sphere_radius=min_sphere_radius
                        ,max_radius=max_radius
                        ,radius_multiplier=radius_multiplier
                        ,dist=dist
                    )
    iter_draw(res, stem_cloud)

    branch_index = defaultdict(list)
    for  idx, branch_num in id_to_num.items(): 
        branch_index[branch_num].append(idx)
    iter_draw(res, list(branch_index.values()))

    breakpoint()

    
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