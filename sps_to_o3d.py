import open3d as o3d
import numpy as np
import scipy.spatial as sps

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