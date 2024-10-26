import open3d as o3d
import numpy as np
import scipy.spatial as sps

from src.utils import get_angles, get_center, get_radius, rotation_matrix_from_arr, unit_vector, poprow

# def map_density(pcd, remove_outliers=True):
#     mesh, densities = TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
#     densities = np.asarray(densities)
#     if remove_outliers:
#         vertices_to_remove = densities < np.quantile(densities, 0.01)
#         mesh.remove_vertices_by_mask(vertices_to_remove)
#     density_colors = plt.get_cmap('plasma')(
#         (densities - densities.min()) / (densities.max() - densities.min()))
#     density_colors = density_colors[:, :3]
#     density_mesh = TriangleMesh()
#     density_mesh.vertices = mesh.vertices
#     density_mesh.triangles = mesh.triangles
#     density_mesh.triangle_normals = mesh.triangle_normals
#     density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
#     #draw([density_mesh])
#     return density_mesh

def get_percentile(pts,low,high):
    z_vals = pts[:,2]


    lower  = np.percentile(z_vals, low)
    upper  = np.percentile(z_vals, high)
    all_idxs =  np.where(z_vals)
    too_low_idxs = np.where(z_vals<=lower)
    too_high_idxs = np.where(z_vals>=upper)
    not_too_low_idxs = np.setdiff1d(all_idxs ,too_low_idxs)
    select_idxs = np.setdiff1d(not_too_low_idxs ,too_high_idxs)
    vals =  z_vals[select_idxs]
    return select_idxs, vals 

def crop(pts, 
         minx = None, maxx = None, 
         miny = None, maxy = None, 
         minz = None, maxz = None):
    x_vals = pts[:,0]
    y_vals = pts[:,1]
    z_vals = pts[:,2]
    to_remove = []
    all_idxs = [idx for idx,_ in enumerate(pts)]
    for min_val,max_val,pt_vals in [(minx, maxx, x_vals),
                                    (miny, maxy, y_vals),
                                    (minz, maxz, z_vals)]:
        if min_val:
            to_remove.append(np.where(pt_vals<=min_val))
        if max_val:
            to_remove.append(np.where(pt_vals>=max_val))

    select_idxs = np.setdiff1d(all_idxs, to_remove)
    return select_idxs

def orientation_from_norms(norms, 
                            samples = 10,
                            max_iter = 100):
    """Attempts to find the orientation of a cylindrical point cloud
    given the normals of the points. Attempts to find <samples> number
    of vectors that are orthogonal to the normals and then averages
    the third ortogonal vector (the cylinder axis) to estimate orientation.
    """
    sum_of_vectors = [0,0,0]    
    found=0 
    iter_num=0
    while found<samples and iter_num<max_iter and len(norms)>1:
        iter_num+=1
        rand_id = np.random.randint(len(norms)-1)
        norms, vect = poprow(norms,rand_id)
        dot_products = abs(np.dot(norms, vect))
        most_normal_val = min(dot_products)
        if most_normal_val <=.001:
            idx_of_normal = np.where(dot_products == most_normal_val)[0][0]
            most_normal = norms[idx_of_normal]
            approx_axis = np.cross(unit_vector(vect), 
                                unit_vector(most_normal))
            sum_of_vectors+=approx_axis
            found+=1
    print(f'found {found} in {iter_num} iterations')
    axis_guess = np.asarray(sum_of_vectors)/found
    return axis_guess

def hull_to_mesh(voxel_down_pcd, type = 'ConvexHull'):

    mesh = o3d.geometry.TriangleMesh
    three_dv = o3d.utility.Vector3dVector
    three_di = o3d.utility.Vector3iVector
    
    points = np.asarray(voxel_down_pcd.points)
    if type != 'ConvexHull':
        test = sps.Delaunay(points)
    else:
        test = sps.ConvexHull(points)
    verts = three_dv(points)
    tris =three_di(np.array(test.simplices[:,0:3]))
    mesh = o3d.geometry.TriangleMesh(verts, tris)
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def filter_by_norm(pcd, angle_thresh=10):
    norms = np.asarray(pcd.normals) 
    angles = np.apply_along_axis(get_angles,1,norms)
    angles = np.degrees(angles)
    stem_idxs = np.where((angles>-angle_thresh) & (angles<angle_thresh))[0]
    stem_cloud = pcd.select_by_index(stem_idxs)
    return stem_cloud


def get_ball_mesh(pcd):
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = (o3d.geometry.TriangleMesh.
                    create_from_point_cloud_ball_pivoting(pcd,
                                                        o3d.utility.DoubleVector(radii)))
    return rec_mesh

def get_shape(pts, 
              shape = 'sphere', 
              as_pts = True,
              rotate = 'axis',
                **kwargs):
    if not kwargs.get('center'):
        kwargs['center'] = get_center(pts)
    if not kwargs.get('radius'):
        kwargs['radius'] = get_radius(pts)
    
    if shape == 'sphere':
        shape = o3d.geometry.TriangleMesh.create_sphere(radius=kwargs['radius'])
    elif shape == 'cylinder':
        try: 
            shape = o3d.geometry.TriangleMesh.create_cylinder(radius=kwargs['radius'],
                                                          height=kwargs['height'])
        except Exception as e:
            breakpoint()
            print(f'error getting cylinder {e}')
    
    # print(f'Starting Translation/Rotation')
    
    if as_pts:
        shape = shape.sample_points_uniformly()
        shape.paint_uniform_color([0,1.0,0])


    shape.translate(kwargs['center'])
    arr = kwargs.get('axis')
    if arr is not None:
        vector = unit_vector(arr)       
        print(f'rotate vector {arr}')
        if rotate == 'axis':   
            R = shape.get_rotation_matrix_from_axis_angle(kwargs['axis'])
        else:
            R = rotation_matrix_from_arr([0,0,1],vector)
        shape.rotate(R, center=kwargs['center'])
    elif rotate == 'axis':
        print('no axis given for rotation, not rotating')
        return shape

    return shape


# def sphere_step(sub_pcd_pts, radius, main_pcd,
#                 curr_neighbors, branch_order, branch_num,
#                 total_found, run=0,branches = [[]],
#                 # new_branches = [[]],
#                 min_sphere_radius = 0.1
#                 ,max_radius = 0.5
#                 ,radius_multiplier = 2
#                 ,dist =.07
#                 ,id_to_num = defaultdict(int)
#                 ,cyls = []
#                 ,cyl_details=[]):
#     if branches == [[]]:
#         branches[0].append(total_found)

#     main_pts = np.asarray(main_pcd.points)  
#     full_tree = sps.KDTree(main_pts)

    
#     # get all points within radius of a current neighbor
#     #   exclude any that have already been found
#     # print('trying to find neighbors')
#     # new_neighbors = get_neighbors_in_tree(sub_pcd_pts, full_tree,radius)
#     center = get_center(sub_pcd_pts)
#     get_neighbors_by_point(np.array([center]), main_pts, radius)
#     breakpoint()
#     print(f"found {len(new_neighbors)} new neighbors")
#     new_neighbors = np.setdiff1d(new_neighbors, curr_neighbors)
#     print(f"found {len(new_neighbors)} not in current")
#     new_neighbors = np.setdiff1d(new_neighbors, np.array(total_found)) 
#     print(f"found {len(new_neighbors)} not in total_found")
    
#     if len(new_neighbors) == 0: 
#         print(f'no new neighbors found')
#         return []

#     inliers_list=[]
#     # in_cyl_neighbors = []

#     prev_neighbor_height = np.max(np.asarray(sub_pcd_pts)[:,2])
#     nn_pts = main_pts[new_neighbors]
#     try:
#         mesh, _, inliers, fit_radius, _ = fit_shape(pts= nn_pts, shape = 'circle',threshold=0.02, 
#                                                     lower_bound=prev_neighbor_height,
#                                                     max_radius=radius*1.5)
#     except Exception as e: 
#         print(f'fit_shape error {e}')
#         mesh, inliers, fit_radius = None, None, None   
    
#     if mesh is None: # or fit_radius>=max_radius:
#         mesh=None
#         to_cluster = new_neighbors
#     if fit_radius >= radius*1.5:
#         orig_in_cyl_neighbors= new_neighbors[inliers]
#         orig_in_cyl_points = main_pts[orig_in_cyl_neighbors]
#         res = kmeans(orig_in_cyl_points, 2)
#         breakpoint()
#     else:
#         in_cyl_neighbors = new_neighbors[inliers]
#         to_cluster =  np.setdiff1d(new_neighbors, np.array( new_neighbors[inliers])) 
#         inliers_list.append(inliers)
#         cyls.append(mesh)
#         cyl_details.append((inliers, nn_pts, fit_radius))
#         # neighbor_cloud = main_pcd.select_by_index(in_cyl_neighbors)     
#         # mesh_pts = mesh.sample_points_uniformly()
#         # mesh_pts.paint_uniform_color([1.0,0,0])
#         # draw([neighbor_cloud,mesh_pts])
#         # else:
#         #     in_cyl_points= main_pts[to_cluster]
#         #     labels, returned_clusters, noise = cluster_neighbors(in_cyl_neighbors, in_cyl_points, dist =dist)

#     if len(to_cluster) != 0: 
#         nn_points= main_pts[to_cluster]
#         labels, returned_clusters, noise = cluster_neighbors(to_cluster, nn_points, dist =dist)
#         if mesh!=None:
#             labels = [(x+len(inliers_list) if x!= -1 else x) for x in list(labels)]
#         elif labels == {-1}:
#             branch_num+=1
#             print(f'only noise found, returning')
#             return []
#     else:
#         labels = []
#         returned_clusters = []
#     # sub_pcd.paint_uniform_color([0,0,0])

#     if mesh!=None:
#         addl_labels = list(range(len(inliers_list)))
#         labels = addl_labels + labels
#         returned_clusters=[in_cyl_neighbors]+returned_clusters

#     # try:
#     #     test = iter_draw(returned_clusters,main_pcd)
#     # except Exception as e:
#     #     print(f'error in iterdraw {e}')


    
#     clusters = [x for x in zip(labels,returned_clusters) if x[0] != -1 and len(x[1])>4]
#     # sets_of_neighbors.append(clusters)
#     # noise = [x for x in zip(labels,returned_clusters) if x[0] == -1]
#     for label, cluster_idxs in clusters:  
#         total_found.extend(cluster_idxs)

#     print(f'iterating over {len(clusters)} clusters')
#     for label, cluster_idxs in clusters:
#         cluster_branch = branch_order
#         if label!=0:
#             cluster_branch+=1
#             branches.append([])
#             # breakpoint()
#         if label >2:
#             break
#         branch_id = branch_num+cluster_branch
#         print(f"{branch_id=}, {cluster_branch=}")
#         cluster_dict = {cluster_idx:branch_id for cluster_idx in cluster_idxs}
#         id_to_num.update(cluster_dict)
#         branches[cluster_branch].extend(cluster_idxs)
        
#         curr_plus_cluster = np.concatenate([curr_neighbors, cluster_idxs]) 
#         cluster_pcd_pts = np.asarray(main_pcd.points)[cluster_idxs]
#         # cluster_cloud = main_pcd.select_by_index(np.asarray(cluster_idxs))
#         # o3d.visualization.draw_geometries([cluster_cloud]) 
#         if fit_radius:
#             new_radius = fit_radius*radius_multiplier
#         else:
#             new_radius = radius

#         if new_radius < min_sphere_radius:
#             new_radius = min_sphere_radius
#         if new_radius > max_radius:
#             new_radius = max_radius
#         print(f"{new_radius=}, fit_radius: {fit_radius}")
#         # print(f"""len(curr_plus_cluster): {len(curr_plus_cluster)}, 
#         #             len(cluster_idxs): {len(cluster_idxs)}, 
#         #             len(curr_neighbors): {len(curr_neighbors)} 
#         #             len(new_neighbors): {len(new_neighbors)}""")
#         # main_less_foud = main_pcd.select_by_index(curr_plus_cluster,invert=True)
#         # main_less_points = np.asarray(main_less_foud.points)
#         # o3d.visualization.draw_geometries([main_less_foud])
        
#         if len(cyls)%10 == 0:
#             try:
#                 test = main_pcd.select_by_index(total_found)
#                 o3d.visualization.draw_geometries([test])
#                 o3d.visualization.draw_geometries(cyls)
#                 cyl_pts = [cyl.sample_points_uniformly(500) for cyl in cyls]
#                 for pcd in cyl_pts: pcd.paint_uniform_color([0,1,0])
#                 draw([test]+cyl_pts)
#             except Exception as e:
#                 print(f'error in iterdraw {e}')
#             breakpoint()
#         sphere_step(cluster_pcd_pts, new_radius, main_pcd,
#                                     curr_plus_cluster, cluster_branch, 
#                                     branch_num,
#                                     total_found, run+1,
#                                     branches
#                                     # ,new_branches[label]
#                                     ,min_sphere_radius
#                                     ,max_radius 
#                                     ,radius_multiplier
#                                     ,dist
#                                     ,id_to_num
#                                     ,cyls
#                                     ,cyl_details)
#         branch_num+=1 
#     print('reached end of function, returning')
#     return branches, id_to_num, cyls, cyl_details
