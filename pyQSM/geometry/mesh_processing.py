
from set_config import config
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

from numpy import array as arr


def edges_to_lineset(mesh,edges, color):

    """
    Convert a set of mesh edges to an Open3D LineSet for visualization.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh whose edges are to be visualized.
    edges : iterable of tuple of int
        Each tuple contains two vertex indices representing an edge.
    color : tuple of float
        RGB color for the lines, each value in [0, 1].
    """
    points = []
    lines = []
    verts = arr(mesh.vertices)
    i=0
    for u,v in edges:
        new_u_id, new_v_id = i, i+1
        u_pt = verts[u]
        v_pt = verts[v]
        points.extend([u_pt,v_pt])
        lines.append([new_u_id, new_v_id])
        i=i+2
    # points = [vert]
    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
    )
    colors = arr([color for _ in range(len(edges))])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def check_properties(mesh, draw_result=False):
    """
    Calculate and print various geometric and topological properties of a mesh.

    Properties checked include:
        - Edge manifoldness (with and without boundary edges)
        - Vertex manifoldness
        - Self-intersection
        - Watertightness
        - Orientability

    Optionally visualizes the mesh and highlights non-manifold edges.

    """
    
    print(f"Checking mesh properties")
    print(f"Computing normals")
    mesh.compute_vertex_normals()

    print(f"Evaluating edges and vertices")
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    import open3d.examples as o3dex
    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(edges_to_lineset(mesh , edges, (1, 0, 0)))
        num_non_manifold = len(edges)
        print(f"{num_non_manifold=}")
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(edges_to_lineset(mesh, edges, (0, 1, 0)))
        num_non_manifold_boundary = len(edges)
        print(f"{num_non_manifold_boundary=}")
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
        verts_non_manifold = len(verts)
        print(f"{verts_non_manifold=}")
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(edges_to_lineset(mesh, edges, (1, 0, 1)))
        self_intersecting_edges = len(edges)
        print(f"{self_intersecting_edges=}")
    if draw_result:
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)
        for geom in geoms:
            o3d.visualization.draw_geometries([geom], mesh_show_back_face=True)
    return geoms

def subdivide_mesh(mesh, max_num_comps):
    """

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

def cluster_and_remove_triangles(mesh ):
    triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 200
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_0])
    return mesh

def get_surface_clusters(mesh,
                       top_n_clusters=10,
                       min_cluster_area=None,
                       max_cluster_area=None): 
    """
        Identifies the connected components of a mesh,
            clusters them by proximity and filters for
            component groups matching the specified criteria.
    """
    # mesh = define_conn_comps(mesh,max_num_comps=10)
    # cluster index per triangle, 
    #   number of triangles per cluster, 
    #   surface area per cluster
    (triangle_clusters, cluster_n_triangles, cluster_area ) =  (mesh.cluster_connected_triangles())
    print(f'{cluster_n_triangles=}')
    print(f'{cluster_area}')
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)
    out_mesh = copy.deepcopy(mesh)
    if top_n_clusters:
        largest_inds = np.argpartition(cluster_n_triangles, -top_n_clusters)[-top_n_clusters:]
        largest_ns = cluster_n_triangles[largest_inds]
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min(largest_ns)
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        # out_mesh.remove_triangles_by_mask(~triangles_to_remove)
    if max_cluster_area:
        triangles_to_remove = cluster_area[triangle_clusters] < max_cluster_area
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        out_mesh.remove_triangles_by_mask(~triangles_to_remove)
    if min_cluster_area:
        cluster_area[triangle_clusters] > min_cluster_area
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        out_mesh.remove_triangles_by_mask(~triangles_to_remove)
    return mesh_0,out_mesh, triangle_clusters

def map_density(pcd,depth=10, outlier_quantile = .01, remove_outliers=False
                 ,normals_radius   = config['stem']["normals_radius"]
                ,normals_nn       = config['stem']["normals_nn"]  
                ,normals_smoothing_nn       = config['stem']["normals_smoothing_nn"]     ):
    print('creating mesh from pcd')
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_radius, max_nn=normals_nn))
    pcd.orient_normals_consistent_tangent_plane(normals_smoothing_nn)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    if remove_outliers: 
        vertices_to_remove = densities < np.quantile(densities, outlier_quantile)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        densities = densities[~vertices_to_remove]
    density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    # draw([density_mesh])
    return density_mesh, densities
