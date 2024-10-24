import open3d as o3d
import numpy as np
import scipy.spatial as sps

from utils import get_angles, get_center, get_radius, rotation_matrix_from_arr, unit_vector, poprow


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

