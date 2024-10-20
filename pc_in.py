# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np

import open3d as o3d
import matplotlib.pyplot as plt
import scipy.spatial as sps 
from itertools import chain 

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
    
def highlight_inliers(pcd, inlier_idxs, color = [1.0, 0, 0], draw =False):
    inlier_cloud = pcd.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color(color)
    if draw:
        outlier_cloud = pcd.select_by_index(inlier_idxs, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

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



def clean_cloud(pcd, neighbors=20, ratio=2.0):
    """Reduces the number of points in the point cloud via 
        voxel downsampling. Reducing noise via statistical outlier removal.
    """
    print("Downsample the point cloud with voxels")
    print(f"orig {pcd}")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.04)
    print(f"downed {voxel_down_pcd}")

    print("Statistical oulier removal")
    _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=neighbors,std_ratio=ratio)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)
    return inlier_cloud

def get_k_smallest(arr,k):
    idx = np.argpartition(arr, k)
    return arr[idx[:k]], idx[:k]

def get_lowest_points(pcd,k):
    pts = np.asarray(pcd.points)
    z_vals = pts[:,2]
    k_mins, mins_idxs = get_k_smallest(z_vals,k)
    return k_mins, mins_idxs

def get_neighbors_in_tree(sub_pcd, sub_pcd_idxs ,main_pcd,radius):
    trunk_tree = sps.KDTree(sub_pcd.points)
    non_trunk_tree = sps.KDTree(main_pcd.points)
    pairs = trunk_tree.query_ball_tree(non_trunk_tree, r=radius)
    neighbors = list(set(chain.from_iterable(pairs)))
    neighbors.extend(sub_pcd_idxs)
    # neighbor_cloud = main_pcd.select_by_index(neighbors)

    # neighbor_cloud.paint_uniform_color([0,0,1.0])
    # sub_pcd.paint_uniform_color([1.0,0,0])
    # o3d.visualization.draw_geometries([neighbor_cloud,sub_pcd])
    return neighbors

def climb_neighbors(sub_pcd, sub_pcd_idxs, main_pcd,radius):
    for i in range(10):
        neighbors = get_neighbors_in_tree(sub_pcd, sub_pcd_idxs ,main_pcd,radius)
        print(f'started with {len(sub_pcd_idxs)} now have {len(neighbors)}')
        sub_pcd_idxs = neighbors
        neighbor_cloud = stem_cloud.select_by_index(neighbors)   
        neighbor_cloud.paint_uniform_color([0,0,1.0])
        sub_pcd.paint_uniform_color([1.0,0,0])
        # len([n for n in neighbors if n in min_idxs])
        o3d.visualization.draw_geometries([neighbor_cloud,just_trunk])
        breakpoint()
        sub_pcd = neighbor_cloud
    return neighbors


pcd = o3d.io.read_point_cloud("s27_downsample_0.04.pcd",print_progress=True)

pcd = clean_cloud(pcd)
# o3d.visualization.draw_geometries([pcd])#, point_show_normal=True)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

neighbors = 20
ratio = 2.0
stem_cloud = filter_by_norm(pcd,10)
# _, ind = stem_cloud.remove_statistical_outlier(nb_neighbors=neighbors,std_ratio=ratio)
# inlier_cloud = stem_cloud.select_by_index(ind)
# o3d.visualization.draw_geometries([stem_cloud])#, point_show_normal=True)

k=1000
mins, min_idxs = get_lowest_points(stem_cloud,k)
low_cloud = highlight_inliers(stem_cloud, min_idxs, draw = False)
just_trunk = clean_cloud(low_cloud)
# o3d.visualization.draw_geometries([just_trunk])

idxs = climb_neighbors(just_trunk,min_idxs,stem_cloud,0.1)
breakpoint()
# mesh = map_density(stem_cloud)
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(stem_cloud, depth=4)
# densities = np.asarray(densities)
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# mesh.remove_vertices_by_mask(vertices_to_remove)
# o3d.visualization.draw_geometries([mesh])
breakpoint()

o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))