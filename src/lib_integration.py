from itertools import chain
import open3d as o3d
import numpy as np
import scipy.spatial as sps
from matplotlib import pyplot as plt

from utils import get_center, get_radius, config

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
    trunk_tree = sps.KDTree(sub_pcd_pts)
    pairs = trunk_tree.query_ball_tree(full_tree, r=radius)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    return neighbors

def find_neighbors_in_ball(
    base_pts, points_to_search, 
    points_idxs, radius=None, center=None,
    draw = False
):
    """
       Defines a centroid for a set of base points,
        finds all points within the search set falling
         within a sphere of a given radius centered on the centroid
    """
    if not center:
        center = get_center(base_pts)
    if not radius:
        radius = get_radius(base_pts) * config["radius_multiplier"]

    if radius < config["min_sphere_radius"]:
        radius = config["min_sphere_radius"]
    if radius > config["max_radius"]:
        radius = config["max_radius"]
    print(f"{radius=}")

    center = [center[0], center[1], max(base_pts[:, 2])]  # - (radius/4)])

    full_tree = sps.KDTree(points_to_search)
    neighbors = full_tree.query_ball_point(center, r=radius)
    res = []
    if draw:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(base_pts[:,0], base_pts[:,1], base_pts[:,2], 'r')
    for results in neighbors:
        new_points = np.setdiff1d(results, points_idxs)
        res.extend(new_points)
        if draw:
            nearby_points = points_to_search[new_points]
            ax.plot(nearby_points[:,0], nearby_points[:,1], nearby_points[:,2], 'o')
    sphere =  o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    if draw:
        sphere_pts=sphere.sample_points_uniformly(500)
        sphere_pts.paint_uniform_color([0,1,0])
        ax.plot(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], 'o')
        plt.show()
    return sphere, res