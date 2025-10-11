from copy import deepcopy
import numpy as np
from numpy import asarray as arr
from glob import glob
import re
import random

from open3d.t.geometry import RaycastingScene as rcs
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd
import open3d as o3d

from collections import defaultdict

import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt

from set_config import config, log
from geometry.reconstruction import get_neighbors_kdtree
from math_utils.general import (
    get_center,
    generate_grid
)
from math_utils.fit import kmeans,cluster_DBSCAN
from geometry.skeletonize import extract_skeleton, extract_topology
from geometry.point_cloud_processing import ( filter_by_norm,
    clean_cloud,
    crop, get_shape,
    orientation_from_norms,
    filter_by_norm,
    get_ball_mesh,
    crop_by_percentile,
    cluster_plus
)
from geometry.mesh_processing import ( 
    check_properties
)
from utils.io import load, load_line_set,save_line_set
from viz.viz_utils import color_continuous_map, draw, rotating_compare_gif
from viz.color import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors   
import cv2
from math import floor
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv

from viz.color import remove_color_pts, get_green_surfaces

from geometry.mesh_processing import get_surface_clusters
from geometry.reconstruction import recover_original_details

from viz.viz_utils import color_continuous_map

import pyvista as pv

import open3d.core as o3c

pinhole_config = { 'fov_deg':60,'center':[-3,-.25,-3],
                    'eye':[10, 10, 20],'up':[0, 0, 1],
                    'width_px':640,'height_px':480,}
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
# working
# pinhole_config = { 'fov_deg': 90, 'center': tmesh.get_center(), 'eye': [-2,-2,15], 'up': [0, -1, 1], 'width_px':1280, 'height_px':960,}
origin = [0,0,0]

def get_points_inside_mesh(radius,height,start,end,pcd,idxs):
    center = (start+end)/2
    mesh = o3d.t.geometry.TriangleMesh.create_cylinder(radius=radius,height=height)
    mesh.translate(center)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    all_pts = arr(pcd.points)
    query_pts = all_pts[idxs]
    tpts =  o3c.Tensor(query_pts, o3c.float32)
    # Create a scene and add the triangle mesh

    scene = rcs()
    _ = scene.add_triangles(mesh)
    # scene.add_points(pcd)
    # scene.compute_occupancy_grid()
    occ = scene.compute_occupancy(tpts)
    # breakpoint()
    return occ


def project_pcd(point_cloud = None, 
                pts = None,
                alpha=.1,
                plot=True,
                name='',
                seed='default',
                screen_shots = [[20,-60,30],[20,-60,60],[-20,-60,-10],[-20,60,30],[-20,60,60]]):
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
    # plane = pv.Plane(center=origin, direction=normal)


    def project_points_to_plane(points, plane_origin, plane_normal):
        """Project points to a plane."""
        vec = points - plane_origin
        dist = np.dot(vec, plane_normal)
        return points - np.outer(dist, plane_normal)

    log.info(f'Projecting points')

    projected_points = project_points_to_plane(points, origin, normal)

    # Create a polydata object with projected points
    polydata = pv.PolyData(projected_points)

    log.info(f'Getting 2D hull')

    # Mesh using delaunay_2d and pyvista
    mesh = polydata.delaunay_2d(alpha=alpha)
    log.info(f'Plotting...')
    # plane_vis = pv.Plane(center=origin,direction=normal,i_size=.5,j_size=.5,i_resolution=10,j_resolution=10,)
    if plot:
        for pos in screen_shots:
            pl = pv.Plotter(off_screen=True)
            pl.add_mesh(mesh)
            pl.add_mesh( points,    color='red',    
                        render_points_as_spheres=True,    
                        point_size=2,    label='Points to project',)
            # pl.add_mesh(plane_vis, color='blue', opacity=0.1, label='Projection Plane')
            pl.camera.position = (polydata.center[0]+pos[0],polydata.center[1]+pos[1],polydata.center[2]+pos[2])
            pl.camera.focal_point = polydata.center
            file = f'data/skio/projection/{seed}_{name}_{pos[0]}_{pos[1]}_{pos[2]}.png'
            pl.show(screenshot =file)
            log.info(f'saved {file}')

        proj = mesh.extract_geometry()
        # Screen Shotting Geometry
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(proj)
        pl.camera.position = (proj.center[0]+15,proj.center[1],proj.center[2]+50)
        pl.camera.focal_point = proj.center
        pl.show(screenshot =f'data/skio/projection/{seed}_{name}_shape.png')
        # pl.show()
    # mesh.save(f'data/skio/projection/{{seed}_{name}_{pos[0]}_{pos[1]}_{pos[2]}.ply')
    return mesh


def sparse_cast_w_intersections(mesh):
    # Create scene and add the monkey model.
    # d = o3d.data.MonkeyModel()
    # mesh = o3d.t.io.read_triangle_mesh(d.path)
    scene = rcs()
    mesh_id = scene.add_triangles(mesh)

    # Create a grid of rays covering the bounding box
    bb_min = mesh.vertex['positions'].min(dim=0).numpy()
    bb_max = mesh.vertex['positions'].max(dim=0).numpy()
    x,y = np.linspace(bb_min, bb_max, num=10)[:,:2].T
    xv, yv = np.meshgrid(x,y)
    orig = np.stack([xv, yv, np.full_like(xv, bb_min[2]-1)], axis=-1).reshape(-1,3)
    dest = orig + np.full(orig.shape, (0,0,2+bb_max[2]-bb_min[2]),dtype=np.float32)
    rays = np.concatenate([orig, dest-orig], axis=-1).astype(np.float32)

    # Compute the ray intersections.
    lx = scene.list_intersections(rays)
    lx = {k:v.numpy() for k,v in lx.items()}

    # Calculate intersection coordinates using the primitive uvs and the mesh
    v = mesh.vertex['positions'].numpy()
    t = mesh.triangle['indices'].numpy()
    tidx = lx['primitive_ids']
    uv = lx['primitive_uvs']
    w = 1 - np.sum(uv, axis=1)
    c_arr = \
    v[t[tidx, 1].flatten(), :] * uv[:, 0][:, None] + \
    v[t[tidx, 2].flatten(), :] * uv[:, 1][:, None] + \
    v[t[tidx, 0].flatten(), :] * w[:, None]

    # Calculate intersection coordinates using ray_ids
    # c_arr = rays[lx['ray_ids']][:,:3] + rays[lx['ray_ids']][:,3:]*lx['t_hit'][...,None]

    # Visualize the rays and intersections.
    lines = o3d.t.geometry.LineSet()
    lines.point.positions = np.hstack([orig,dest]).reshape(-1,3)
    lines.line.indices = np.arange(lines.point.positions.shape[0]).reshape(-1,2)
    lines.line.colors = np.full((lines.line.indices.shape[0],3), (1,0,0))
    x = o3d.t.geometry.PointCloud(positions=c_arr)
    o3d.visualization.draw([mesh, lines, x], point_size=8)
    return lines,x

def birdseye(mesh):
    center =mesh.get_center()
    ub = mesh.get_max_bound().numpy()
    lb = mesh.get_min_bound().numpy()
    diff = ub-lb 
    half_diff = (diff/2)
    mid = ub-half_diff
    birds_eye = [float(mid[0]),float(mid[1]),float(ub[2]+half_diff[2])]
    return birds_eye
    # eye = [-3.480174,-0.662891,10]

def project_to_image(mesh):
    # scene = rcs()
    # scene.add_triangles(cube)
    # scene.add_triangles(torus)
    # _ = scene.add_triangles(sphere)  
    # rays = rcs.create_rays_pinhole( **pinhole_config)

    # coords = o3d.t.geometry.TriangleMesh.create_coordinate_frame()
    # _ = scene.add_triangles(coords) 

    # rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    # tmesh.rotate(rot_90_x,center = tmesh.get_center())
    
    scene = rcs()  
    scene.add_triangles(mesh)
    rays = rcs.create_rays_scene = rcs()  
    scene.add_triangles(mesh)
    rays = rcs.create_rays_pinhole(**pinhole_config)
    ans = scene.cast_rays(rays)
    intersecting_rays = ans['t_hit'].isfinite()
    hits = rays[intersecting_rays]
    points = hits[:,:3] + hits[:,3:]*ans['t_hit'][intersecting_rays].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points)
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    pcd_out = pcd.to_legacy()
    rays = rcs.create_rays_pinhole(**pinhole_config)
    ans = scene.cast_rays(rays)
    plt.imshow(ans['t_hit'].numpy())
    plt.show()
    pcd1 = raycast_to_pcd( **pinhole_config)
    breakpoint()

def mri(mesh=None,rcs_in = None):
    if rcs_in:
        scene = rcs_in
    elif mesh:
        scene = rcs()
        _ = scene.add_triangles(mesh)
    else:
        raise ValueError('No mesh or rcs provided for mri')
    min_bound = mesh.vertex.positions.min(0).numpy()
    max_bound = mesh.vertex.positions.max(0).numpy()
    N = 256
    query_points = np.random.uniform(low=min_bound, high=max_bound, size=[N, 3]).astype(np.float32)
    # Compute the signed distance for N random points
    signed_distance = scene.compute_signed_distance(query_points)
    xyz_range = np.linspace(min_bound, max_bound, num=64)
    # query_points is a [32,32,32,3] array ..
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
    # signed distance is a [32,32,32] array
    signed_distance = scene.compute_signed_distance(query_points)
    # We can visualize a slice of the distance field directly with matplotlib
    for i in range(32):
        plt.imshow(signed_distance.numpy()[:, :, i*2])
        plt.show()
    breakpoint()

def cast_rays(tmesh, 
                surf_2d:bool = False,
                img:bool = False,
                pinhole_config = pinhole_config):
    # [-3,-.25,-3]
    breakpoint()
    log.info('starting cast rays')
    center = tmesh.get_center().numpy()
    eye = [center[0],center[1],center[2]+10]
    pinhole_config = { 'fov_deg': 90,  'center': tmesh.get_center(),   'eye': list(eye),    'up': [0, -1, 1],    'width_px':640*2, 'height_px':475*2,}
    pinhole_config['up'] = [0, 1, -1]
    log.info('creating pinhole')
    # tmesh.rotate(rot_90_x,center = tmesh.get_center())
    scene = rcs()  
    scene.add_triangles(tmesh)
    rays = rcs.create_rays_pinhole(**pinhole_config)
    log.info('casting rays')
    ans = scene.cast_rays(rays)
    intersecting_rays = ans['t_hit'].isfinite()
    breakpoint()
    if img:
        plt.imshow(ans['t_hit'].numpy())
        plt.show()
    if surf_2d:
        log.info('getting surface area')
        hit_triangle_ids = ans['primitive_ids'][intersecting_rays].numpy()
        lmesh = tmesh.to_legacy()
        hit_tris = arr(lmesh.triangles)[ans['primitive_ids'][intersecting_rays].numpy()]
        triangles = np.unique(hit_tris,axis=0)
        hit_vert_ids = np.unique(hit_tris)
        hit_mesh =lmesh.select_by_index(hit_vert_ids)
        sa_3d = hit_mesh.get_surface_area()
        # o3d.visualization.draw_geometries([hit_mesh], mesh_show_back_face=True)
        # o3d.io.write_triangle_mesh('data/skeletor/results/hit_mesh_33.pcd',hit_mesh)

        mesh_2d = deepcopy(hit_mesh)
        hit_verticies = arr(mesh_2d.vertices)
        hvs_2d = [(x,y,0) for x,y,z in hit_verticies]
        mesh_2d.vertices = o3d.utility.Vector3dVector(hvs_2d)
        sa_2d = mesh_2d.get_surface_area()
        o3d.visualization.draw_geometries([mesh_2d], mesh_show_back_face=True)
        breakpoint()

    # tcoords = o3d.t.geometry.TriangleMesh.create_coordinate_frame()
    # tcoords.translate([5,0,0])
    # tmesh.rotate(rot_90_x,center = tmesh.get_center())
    # draw([tcoords.to_legacy(),tmesh.to_legacy()])

    # pcd = raycast_to_pcd(tmesh,pinhole_config)
    # draw([tcoords.to_legacy(),pcd])
    # breakpoint()
    return hit_mesh

def raycast_to_pcd(mesh, pinhole_config):
    scene = rcs()  
    scene.add_triangles(mesh)
    rays = rcs.create_rays_pinhole(**pinhole_config)
    ans = scene.cast_rays(rays)
    intersecting_rays = ans['t_hit'].isfinite()
    hits = rays[intersecting_rays]
    points = hits[:,:3] + hits[:,3:]*ans['t_hit'][intersecting_rays].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points)
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    pcd_out = pcd.to_legacy()
    color_continuous_map(pcd_out,ans['t_hit'][intersecting_rays].numpy())
    draw(pcd_out)
    breakpoint()
    o3d.visualization.draw_geometries([pcd.to_legacy()],front=[0.5, 0.86, 0.125],lookat=[0.23, 0.5, 2], up=[-0.63, 0.45, -0.63],zoom=0.7)
    return pcd, pcd_out
    
def get_intersected(mesh,ans):
    intersecting_rays = ans['t_hit'].isfinite()
    hits = rays[intersecting_rays]
    crossed = mesh.select_by_index(ans['primitive_ids'][intersecting_rays])
