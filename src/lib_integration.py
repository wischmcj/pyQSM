import open3d as o3d
import numpy as np
import scipy.spatial as sps

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