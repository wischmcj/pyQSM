import open3d as o3d
import numpy as np
import scipy.spatial as sps 

s27d = "s32_downsample_0.04.pcd"

def display_inlier_outlier(cloud, ind):

    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    del cloud
    print("Painting outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    breakpoint()
    print("Showing outliers (red) and inliers (gray): ")
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def rotate(pcd):

    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))

    pcd.rotate(R, center=(0, 0, 0))

    o3d.visualization.draw([pcd])


def clean_cloud(pcd):
    """Reduces the number of points in the point cloud via 
        voxel downsampling. Reducing noise via statistical outlier removal.
    """
    # print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd
    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.04)
    # del pcd
    # o3d.visualization.draw_geometries([voxel_down_pcd])
    print("Statistical oulier removal")
    _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)
    return inlier_cloud


# def make_triangle_mesh():
#     mesh = open3d.geometry.TriangleMesh()
#     np_vertices = np.array([[-1, 2, 0],
#                             [1, 2, 0],
#                             [0, 0, 0],
#                             [2, 0, 0]])
#     np_triangles = np.array([[0, 2, 1],
#                             [1, 2, 3]]).astype(np.int32)
#     mesh.vertices = open3d.Vector3dVector(np_vertices)

#     # From numpy to Open3D
#     mesh.triangles = open3d.Vector3iVector(np_triangles)

#     # From Open3D to numpy
#     np_triangles = np.asarray(mesh.triangles)

if __name__ == "__main__":
    s27 ="/code/code/Research/lidar/converted_pcs/Secrest27_05.pts"
    pcd = o3d.io.read_point_cloud(s27, format='xyz',print_progress=True)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.04)
    del pcd
    # voxel_down_pcd = clean_cloud(voxel_down_pcd)
    # voxel_down_pcd = o3d.io.read_point_cloud(s27, print_progress=True)
    # o3d.io.write_point_cloud(s27d,voxel_down_pcd)
    # o3d.visualization.draw_geometries([voxel_down_pcd])
    o3d.visualization.draw_geometries([voxel_down_pcd],point_show_normal=True)
    breakpoint()


    