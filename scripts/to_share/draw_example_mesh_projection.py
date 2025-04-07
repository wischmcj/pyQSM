import open3d as o3d
import numpy as np
from open3d.io import read_point_cloud
from geometry.surf_recon import meshfix
from ray_casting import cast_rays


def generate_comparison(pcds, meshes):
    pcd_list = [read_point_cloud(pcd) for pcd in pcds]
    mesh_list = [o3d.io.read_triangle_mesh(mesh) for mesh in meshes]
    o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)
    
    mesh_volumes = [mesh.to_legacy().get_volume() for mesh in mesh_list]
    mesh_sas = [mesh.to_legacy().get_surface_area() for mesh in mesh_list]
    raise NotImplementedError

if __name__ == "__main__":
    inputs = 'data/skeletor/inputs'
    src_pcd = read_point_cloud(f'{inputs}/skeletor_clean.pcd')
    branch_mesh = o3d.io.read_triangle_mesh('data/skeletor/branches_mesh_alphapt3.ply')
    # triangle_clusters, cluster_n_triangles, cluster_area = (branch_mesh.cluster_connected_triangles())
    # tc_idxs = [list(np.where(triangle_clusters == val)[0]) for val in np.unique(triangle_clusters)]
    tc_colors = branch_mesh.vertex_colors
    unique_colors = np.unique(tc_colors)
    tc_idxs  = [list(np.where(tc_colors == val)[0]) for val in unique_colors] 
    tc_meshes = [branch_mesh.select_by_index(idxs) for idxs in tc_idxs]
    test_mesh = tc_meshes[33]
    o3d.visualization.draw_geometries([branch_mesh], mesh_show_back_face=False,mesh_show_wireframe=False)
    o3d.visualization.draw_geometries([tc_meshes[33]], mesh_show_back_face=False,mesh_show_wireframe=True)


    test_mesh = cast_rays(test_mesh, surf_2d = True)
    fixed_mesh = meshfix(test_mesh,'repair')
    # t_branch_mesh = o3d.t.geometry.TriangleMesh.from_legacy(test)

    # fixed_mesh = o3d.io.read_triangle_mesh('data/skeletor/meshes/fixed_mesh_33.ply')
    rc_pcd = read_point_cloud('data/skeletor/meshes/mesh_pcd_33.pcd')
    rc_mesh = o3d.io.read_triangle_mesh('data/skeletor/meshes/hit_mesh_33.ply')
    rc_2d_mesh = o3d.io.read_triangle_mesh('data/skeletor/meshes/hit_mesh_2d_33.ply')
    
    # o3d.visualization.draw_geometries([fixed_mesh.to_legacy()], mesh_show_back_face=False,mesh_show_wireframe=True)
    
   
    o3d.visualization.draw_geometries([rc_pcd])
    o3d.visualization.draw_geometries([rc_mesh], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([rc_2d_mesh], mesh_show_back_face=True)
    final_iso = read_point_cloud('final_branch_cc_iso_pt9_top100_mdpt_k50')
    fixed_mesh_volume = fixed_mesh.to_legacy().get_volume()
    fixed_mesh_sa = fixed_mesh.to_legacy().get_surface_area()
    exposed_surface_area = rc_mesh.get_surface_area()
    projected_area = rc_2d_mesh.get_surface_area()
    print(f'{fixed_mesh_volume=}')
    print(f'{exposed_surface_area=}')
    print(f'{projected_area=}')
    
    cols = ['', 'Fixed Mesh', 'Exposed Surface', '2d Projection']
    rows = [['Volume',fixed_mesh_volume,0,0],['Surface Area',fixed_mesh_sa,exposed_surface_area,projected_area]]

    from prettytable import PrettyTable
    x = PrettyTable(cols)
    for row in rows: x.add_row(row)
    breakpoint()