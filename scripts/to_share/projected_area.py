
import numpy as np
import pyvista as pv
import open3d as o3d
from numpy import asarray as arr
from open3d.io import read_point_cloud, write_point_cloud


def project_pcd(point_cloud = None, pts = None):
    # num_points = 100
    # rng = np.random.default_rng(seed=0)  # Seed rng for reproducibility
    # point_cloud = rng.random((num_points, 3))
    if point_cloud:
        pts = arr(point_cloud.points)
    # if not isinstance(np.array,pts):
    points = arr(pts)
    # Define a plane
    origin = [0, 0, 0]
    normal = [0, 0, 1]
    plane = pv.Plane(center=origin, direction=normal)


    def project_points_to_plane(points, plane_origin, plane_normal):
        """Project points to a plane."""
        vec = points - plane_origin
        dist = np.dot(vec, plane_normal)
        return points - np.outer(dist, plane_normal)


    projected_points = project_points_to_plane(points, origin, normal)

    # Create a polydata object with projected points
    polydata = pv.PolyData(projected_points)

    # Mesh using delaunay_2d and pyvista
    mesh = polydata.delaunay_2d()
    plane_vis = pv.Plane(
        center=origin,
        direction=normal,
        i_size=.5,
        j_size=.5,
        i_resolution=10,
        j_resolution=10,
    )

    # plot it
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True, color='white', opacity=0.5, label='Tessellated mesh')
    pl.add_mesh(    pv.PolyData(points),    color='red',    render_points_as_spheres=True,    point_size=1,    label='Points to project',)
    pl.add_mesh(plane_vis, color='blue', opacity=0.1, label='Projection Plane')
    pl.add_legend()
    pl.show()
    
    geo = mesh.extract_geometry()
    pl = pv.Plotter()
    pl.add_mesh(geo)
    pl.show()
    print(geo.area)    
    breakpoint()
    return mesh


inputs = 'data/skeletor/inputs'
src_pcd = read_point_cloud(f'{inputs}/skeletor_clean.pcd')
ds20 = src_pcd.uniform_down_sample(20)
pts = arr(ds20.points)
project_pcd(pts = pts)

dtrunk = read_point_cloud(f'{inputs}/skeletor_dtrunk.pcd')
dtrunk10 = dtrunk.uniform_down_sample(10)
pts = arr(dtrunk.points)
breakpoint()
project_pcd(pts = pts)