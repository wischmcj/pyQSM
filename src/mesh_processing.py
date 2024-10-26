import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

def define_conn_comps(mesh, max_num_comps):
    """
    Sourced from open3d recipes. Identifies portions
    of a mesh that form a continuous surface.

    Args:
        mesh: o3d.geometry.TriangleMesh

    Returns:
        o3d.geometry.TriangleMesh: 
    """
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    vert = np.asarray(mesh.vertices)
    min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
    for _ in range(max_num_comps):
        cube = o3d.geometry.TriangleMesh.create_box()
        cube.scale(0.005, center=cube.get_center())
        cube.translate(
            (
                np.random.uniform(min_vert[0], max_vert[0]),
                np.random.uniform(min_vert[1], max_vert[1]),
                np.random.uniform(min_vert[2], max_vert[2]),
            ),
            relative=False,
        )
        mesh += cube
    mesh.compute_vertex_normals()
    return mesh

def get_surface_clusters(mesh,
                       top_n_clusters=10,
                       min_cluster_area=0,
                       max_cluster_area=100): 
    """
        Identifies the connected components of a mesh,
            clusters them by proximity and filters for
            component groups matching the specified criteria.
    """
    mesh = define_conn_comps(mesh)
    # cluster index per triangle, 
    #   number of triangles per cluster, 
    #   surface area per cluster
    (triangle_clusters, cluster_n_triangles, cluster_area ) =  (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)
    if top_n_clusters:
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
    if max_cluster_area:
        triangles_to_remove = cluster_area[triangle_clusters] < max_cluster_area
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
    if min_cluster_area:
        cluster_area[triangle_clusters] > min_cluster_area
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def map_density(pcd,depth=10, outlier_quantile = .01, remove_outliers=True):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    if remove_outliers:
        vertices_to_remove = densities < np.quantile(densities, outlier_quantile)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]

    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    #draw([density_mesh])
    return density_mesh, densities