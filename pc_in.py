# with open("/code/code/Research/lidar/Secrest_27-5.xyb", mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

import numpy as np

import open3d as o3d
import matplotlib.pyplot as plt
import scipy.spatial as sps 
import scipy.cluster as spc 
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


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

def get_sphere(pts):
    new_center = get_center(pts)
    new_radius = get_radius(pts)*1.25
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

def cluster_neighbors(points,current_clusters ):
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



def climb_neighbors(sub_pcd, sub_pcd_idxs, main_pcd,
                        initial_radius = 0.2):
    neighbors = 20
    ratio = 0.25
    min_sphere_radius = 0.1
    epsilon_radius = 0.25
    current_clusters = 1
    radius = initial_radius
    
    main_pts = np.asarray(main_pcd.points)
    full_tree = sps.KDTree(main_pts)
    sub_pcd.paint_uniform_color([1.0,0,0])

    curr_neighbors = sub_pcd_idxs
    sets_of_neighbors = [curr_neighbors]
    breakpt = 5
    for i in range(100):

        new_neighbors = get_neighbors_in_tree(sub_pcd, full_tree,radius)
        new_neighbors = np.setdiff1d(new_neighbors, curr_neighbors)
        curr_neighbors = np.concatenate([curr_neighbors, new_neighbors])
        neighbor_cloud = main_pcd.select_by_index(curr_neighbors)   
        _, ind = neighbor_cloud.remove_statistical_outlier(nb_neighbors=neighbors,std_ratio=ratio)
        neighbor_cloud = highlight_inliers(neighbor_cloud,ind,draw=False, color = [0,1.0,0])

        nn_locations = main_pts[new_neighbors]  

        codes,book,num = cluster_neighbors(nn_locations,current_clusters)
        current_clusters = num
        if num > 1:
            breakpoint()
            c1_idxs = new_neighbors[np.where(book==1)[0]]
            c0_idxs = new_neighbors[np.where(book!=1)[0]]

            c1_cloud = main_pcd.select_by_index(np.asarray(c1_idxs))    
            c0_cloud = main_pcd.select_by_index(np.asarray(c0_idxs))  
            c1_cloud.paint_uniform_color([0,0,1.0])
            c0_cloud.paint_uniform_color([1.0,0,0])  
            o3d.visualization.draw_geometries([c1_cloud,c0_cloud])
        else:
            sets_of_neighbors.append(new_neighbors)
        
        # points = main_pts[new_neighbors]
        # sphere_pts = get_sphere(points)

        # neighbor_cloud.paint_uniform_color([0,0,1.0])
        # sub_pcd.paint_uniform_color([1.0,0,0])
        # len([n for n in neighbors if n in min_idxs])

        if i % breakpt == 0:
            points = main_pts[new_neighbors]
            sphere_pts = get_sphere(points)
            neighbor_cloud.paint_uniform_color([0,0,1.0])
            sub_pcd.paint_uniform_color([1.0,0,0])
            o3d.visualization.draw_geometries([neighbor_cloud,sub_pcd,sphere_pts])
            breakpoint()

        sub_pcd = neighbor_cloud
    return neighbors

def get_ball_mesh(pcd):
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(just_trunk, o3d.utility.DoubleVector(radii))
    return rec_mesh


pcd = o3d.io.read_point_cloud("s27_downsample_0.04.pcd",print_progress=True)

pcd = clean_cloud(pcd)
# o3d.visualization.draw_geometries([pcd])#, point_show_normal=True)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

stem_cloud = filter_by_norm(pcd,10)
# _, ind = stem_cloud.remove_statistical_outlier(nb_neighbors=neighbors,std_ratio=ratio)
# inlier_cloud = stem_cloud.select_by_index(ind)
# o3d.visualization.draw_geometries([stem_cloud])#, point_show_normal=True)

k=1000
mins, min_idxs = get_lowest_points(stem_cloud,k)
low_cloud = highlight_inliers(stem_cloud, min_idxs, draw = False)

neighbors = 20
ratio = 0.25
_, ind = low_cloud.remove_statistical_outlier(nb_neighbors=neighbors,std_ratio=ratio)
just_trunk = highlight_inliers(low_cloud,ind,draw=False, color = [1.0,0,0])
trunk_points = np.asarray(just_trunk.points)
# sphere_pts=get_sphere(trunk_points)
# o3d.visualization.draw_geometries([just_trunk,sphere_pts])


radius = get_radius(trunk_points)

idxs = climb_neighbors(just_trunk,min_idxs,stem_cloud,radius)
breakpoint()


# mesh = map_density(stem_cloud)
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(stem_cloud, depth=4)
# densities = np.asarray(densities)
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# mesh.remove_vertices_by_mask(vertices_to_remove)
# o3d.visualization.draw_geometries([mesh])
breakpoint()

o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))