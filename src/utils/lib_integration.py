from itertools import chain
import open3d as o3d
import numpy as np
import scipy.spatial as sps
from matplotlib import pyplot as plt, patches

from set_config import config, log
from utils.math_utils import get_center, get_radius, get_percentile
from viz.viz_utils import iter_draw, draw

## Numpy

def pts_to_cloud(points:np.ndarray, colors = None):
    '''
    Convert a numpy array to an open3d point cloud. Just for convenience to avoid converting it every single time.
    Assigns blue color uniformly to the point cloud.

    :param points: Nx3 array with xyz location of points
    :return: a blue open3d.geometry.PointCloud()
    '''
    if not colors:
        colors = [[0, 0, 1] for i in range(points.shape[0])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

## SciPy spatial 

def sps_hull_to_mesh(voxel_down_pcd, type="ConvexHull"):
    mesh = o3d.geometry.TriangleMesh
    three_dv = o3d.utility.Vector3dVector
    three_di = o3d.utility.Vector3iVector

    points = np.asarray(voxel_down_pcd.points)
    if type != "ConvexHull":
        test = sps.Delaunay(points)
    else:
        test = sps.ConvexHull(points)
    verts = three_dv(points)
    tris = three_di(np.array(test.simplices[:, 0:3]))
    mesh = o3d.geometry.TriangleMesh(verts, tris)
    # o3d.visualization.draw_geometries([mesh])
    return mesh

### KDTrees

def get_neighbors_in_tree(sub_pcd_pts, full_tree, radius):
    '''Returns indicies n_idx of points in full_tree.data such that 
        distance(full_tree.data[n_idx],q_pt)<=radius for q_pt in sub_pcd_pts'''
    trunk_tree = sps.KDTree(sub_pcd_pts)
    pairs = full_tree.query_ball_tree(trunk_tree, r=radius)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    return neighbors

def find_neighbors_in_ball(
    base_pts, points_to_search, 
    points_idxs, radius=None, center=None,
    use_top = None, #(85,100)
    draw_results = False
):
    """
       Defines a centroid for a set of base points,
        finds all points within the search set falling
         within a sphere of a given radius centered on the centroid
    """
    if use_top:
        top_pt_idxs,_ = get_percentile(base_pts,use_top[0],use_top[1])
        top_pts = base_pts[top_pt_idxs]
        if not center:
            center = get_center(top_pts)
            center = [center[0], center[1], max(base_pts[:, 2])] 
        if not radius:
            radius = get_radius(top_pts) * config['sphere']["radius_multiplier"]
    else:
        if not center:
            center = get_center(base_pts)
        if not radius:
            radius = get_radius(base_pts) * config['sphere']["radius_multiplier"]

    if radius < config['sphere']["min_radius"]:
        radius = config['sphere']["min_radius"]
    if radius > config['sphere']["max_radius"]:
        radius = config['sphere']["max_radius"]
    log.info(f" Finding nbrs in ball w/ {radius=}, {center=}")

    full_tree = sps.KDTree(points_to_search)
    neighbors = full_tree.query_ball_point(center, r=radius)
    res = []
    if draw_results:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(base_pts[:,0], base_pts[:,1], base_pts[:,2], 'r')
    for results in neighbors:
        new_points = np.setdiff1d(results, points_idxs)
        res.extend(new_points)
        if draw_results:
            nearby_points = points_to_search[new_points]
            ax.plot(nearby_points[:,0], nearby_points[:,1], nearby_points[:,2], 'o')
    sphere =  o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    if draw_results:
        # sphere_pts=sphere.sample_points_uniformly(500)
        # sphere_pts.paint_uniform_color([0,1,0])
        # sphere_pts = np.array(sphere_pts.points)
        # ax.plot(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], 'o')
        # plt.show()
        test = o3d.geometry.PointCloud()
        test.points = o3d.utility.Vector3dVector(base_pts)
        draw([sphere, test])
    return sphere, neighbors, center, radius

## Matplotlib

def plot_squares(extents= None,
                    lls_urs = None):
    if not lls_urs and not extents:
        raise ValueError('No range input provided ')
    fig, ax = plt.subplots()
    if not lls_urs:
        bounds = [((x_min,y_min),x_max-x_min, y_max-y_min ) for ((x_min, y_min,_), ( x_max, y_max,_)) in extents.values() ]
    else: 
        bounds = lls_urs
    g_x_min = col_min[0]
    g_x_max = col_max[0]
    g_y_min = col_min[1]
    g_y_max = col_max[1]
    plt.xlim(g_x_min - 1, g_x_max + 2)
    plt.ylim(g_y_min - 1, g_y_max + 2)
    for ll, ur in safe_grid: ax.add_patch(patches.Rectangle(ll, ur[0] - ll[0], ur[1] - ll[1], linewidth=1, edgecolor='black', facecolor='none'))
    plt.show()