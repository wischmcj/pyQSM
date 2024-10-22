import open3d as o3d
import numpy as np
import scipy.spatial as sps

from utils import get_angles, get_center, get_radius, rotation_matrix_from_arr, unit_vector

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
                **kwargs):
    if not kwargs.get('center'):
        kwargs['center'] = get_center(pts)
    if not kwargs.get('radius'):
        kwargs['radius'] = get_radius(pts)
    
    if shape == 'sphere':
        shape = o3d.geometry.TriangleMesh.create_sphere(center=kwargs['center'], 
                                                        radius=kwargs['radius'])
    elif shape == 'cylinder':
        try: 
            shape = o3d.geometry.TriangleMesh.create_cylinder(radius=kwargs['radius'],
                                                          height=kwargs['height'])
        except Exception as e:
            breakpoint()
            print(f'error getting cylinder {e}')
    
    print(f'Starting Translation/Rotation')
    
    shape.translate(kwargs['center'])
    arr = kwargs.get('axis')
    if arr is not None:
        vector = unit_vector(arr)
        R = rotation_matrix_from_arr([0,0,1],vector)
        print(f'{vector=}')
        shape.rotate(R, center=kwargs['center'])

    if as_pts:
        shape_pts = shape.sample_points_uniformly()
        shape_pts.paint_uniform_color([0,1.0,0])
        return shape_pts
    else:
        return shape

