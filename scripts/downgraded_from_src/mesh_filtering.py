
from numpy.random.mtrand import laplace
import open3d as o3d
import numpy as np


def average_filtering(mesh_in):
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of average mesh filter after 1 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of average mesh filter after 5 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])
    return mesh_out


def laplace_filtering(mesh_in):
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of Laplace mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of Laplace mesh filter after 50 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])
    return mesh_out


def taubin_filtering(mesh_in):
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of Taubin mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of Taubin mesh filter after 100 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])
    return mesh_out


# if __name__ == "__main__":
#     average_filtering()
#     laplace_filtering()
#     taubin_filtering()