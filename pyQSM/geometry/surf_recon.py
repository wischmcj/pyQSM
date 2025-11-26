
import open3d as o3d
import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd


from set_config import  log
from geometry.mesh_processing import check_properties, get_surface_clusters
from viz.plotting import plot_dist_dist
from viz.viz_utils import draw


def deform_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
    static_pos = []
    for id in static_ids: static_pos.append(vertices[id])
    handle_ids = [2490]
    handle_pos = [vertices[2490] + np.array((-40, -40, -40))]
    constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm: mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids,constraint_pos,max_iter=50)
    draw(mesh_prime)

def pytmesh_to_mesh(tin,src_mesh):
    v,f = tin.return_arrays()
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.triangles = o3d.utility.Vector3iVector(arr(f))
    new_mesh.vertices =  o3d.utility.Vector3dVector(arr(v))
    new_mesh.vertex_colors = src_mesh.vertex_colors
    o3d.visualization.draw_geometries([new_mesh], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([new_mesh])
    return new_mesh

def meshfix(mesh, algo = 'pyt'):
    import pymeshfix as mf
    import numpy as np
    import pyvista as pv

    if algo == 'repair':
        mfix = mf.MeshFix(arr(mesh.vertices), arr(mesh.triangles))
        mfix.repair()
        pv.plot(mfix.mesh)
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.triangles = o3d.utility.Vector3iVector(arr(mfix.f))
        new_mesh.vertices =  o3d.utility.Vector3dVector(arr(mfix.v))
    if algo == 'pyt':
        # log.info(f'Checking initial properties of mesh')
        # check_properties(mesh)
        tin = mf.PyTMesh()
        tin.load_array(arr(mesh.vertices), arr(mesh.triangles)) # or read arrays from memory
        holes = tin.fill_small_boundaries()
        new_mesh = pytmesh_to_mesh(tin,mesh)
        print(f'{holes}')
        # check_properties(mesh)
        log.info('getting surface clusters')
        mesh10,mesh10_out,triangle_clusters = get_surface_clusters(new_mesh,top_n_clusters=10)
        mesh5,mesh5_out,_ = get_surface_clusters(new_mesh,top_n_clusters=5)
        draw(mesh10_out)
        draw(mesh5_out)
        log.info('getting significant components defined by surface clusters')
        vals, cnts = np.unique(triangle_clusters, return_counts=True)
        sig_comps = [triangle_clusters!=val for val,cnt in zip(vals,cnts) if cnt>50]
        # [len(triangle_clusters[triangle_clusters==val]) for val,cnt in zip(vals,cnts) if cnt>50]
        for mask in sig_comps: #draw([new_mesh.remove_triangles_by_mask(mask)])
            val_mesh = new_mesh.remove_triangles_by_mask(mask) #.select_by_index(val_list)
            draw(val_mesh)

        # log.info('joining closest components')
        # tin.join_closest_components()
        new_mesh = pytmesh_to_mesh(tin,mesh)
        # check_properties(mesh)
        
        print('There are {:d} boundaries'.format(tin.boundaries()))
        # tin.clean(max_iters=5, inner_loops=1)
        holes = tin.fill_small_boundaries()
        new_mesh = pytmesh_to_mesh(tin,mesh)
        print(f'holes')
        log.info('checking properties of final mesh')
        check_properties(mesh)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(new_mesh)
    return tmesh

def pivot_ball_mesh(pcd, 
                    radii_factors = [0.1,0.2,0.3,0.4,0.5,0.7,1,1.2,1.5,1.7,2],
                    plot_distribution=False):
    log.info(f'Computing KNN distance')
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f'{avg_dist=}')
    if plot_distribution:
        plot_dist_dist(pcd,distances)
    dist_radii = [f*avg_dist for f in radii_factors]
    # radii = [avg_dist*(i/10) for i in range(1,20)]
    # radii = [0.2*avg_dist,0.3*avg_dist,0.5*avg_dist,0.7*avg_dist,0.9*avg_dist,avg_dist, 1.2*avg_dist,1.5*avg_dist,1.7*avg_dist, 2*avg_dist,2.5*avg_dist,3*avg_dist] 
    # 0.1*avg_dist,0.3*avg_dist
    radii_lists = [dist_radii,
                    # [.05,.06,.07,.08,.09,.1],
                #    [.1,.13,.15,.17,.2],
                #    [.05,.06,.07,.08,.09,.1],
                #    np.logspace(.4, 0.05, num=6)
                   ]
    log.info(f'Estimating normals')
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist*3, max_nn=20))
    pcd.orient_normals_consistent_tangent_plane(100)
    log.info(f'Creating mesh(s)')
    for radii in radii_lists:
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        rec_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([rec_mesh])
        # o3d.visualization.draw_geometries([rec_mesh,pcd])
    return rec_mesh


# def pivot_ball_mesh(pcd, 
#                     radii_factors = [0.1,0.2,0.3,0.4,0.5,0.7,1,1.2,1.5,1.7,2],
#                     plot_distribution=False):
#     log.info(f'Computing KNN distance')
#     distances = pcd.compute_nearest_neighbor_distance()
#     avg_dist = np.mean(distances)
#     print(f'{avg_dist=}')
#     if plot_distribution:
#         log.info(f'plotting KNN distance distrobution')
#         bins = np.histogram_bin_edges(distances, bins=300)
#         cum_sum=[ len([x for x in distances if lb>x]) for lb in bins]
#         ax = plt.subplot()
#         ax.scatter(bins,cum_sum)
#         plt.axvline(x=avg_dist, color='r', linestyle='--')
#         plt.axvline(x=avg_dist*2, color='r', linestyle='--')
#         plt.show()
#     dist_radii = [f*avg_dist for f in radii_factors]
#     # radii = [avg_dist*(i/10) for i in range(1,20)]
#     # radii = [0.2*avg_dist,0.3*avg_dist,0.5*avg_dist,0.7*avg_dist,0.9*avg_dist,avg_dist, 1.2*avg_dist,1.5*avg_dist,1.7*avg_dist, 2*avg_dist,2.5*avg_dist,3*avg_dist] 
#     # 0.1*avg_dist,0.3*avg_dist
#     radii_lists = [dist_radii,
#                     # [.05,.06,.07,.08,.09,.1],
#                 #    [.1,.13,.15,.17,.2],
#                 #    [.05,.06,.07,.08,.09,.1],
#                 #    np.logspace(.4, 0.05, num=6)
#                    ]
#     log.info(f'Estimating normals')
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist*3, max_nn=20))
#     pcd.orient_normals_consistent_tangent_plane(100)
#     log.info(f'Creating mesh(s)')
#     for radii in radii_lists:
#         rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
#         rec_mesh.compute_vertex_normals()
#         o3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True)
#         o3d.visualization.draw_geometries([rec_mesh])
#         # o3d.visualization.draw_geometries([rec_mesh,pcd])
#     return rec_mesh

def get_mesh(pcd,lowc,target):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.3, max_nn=20))
    pcd.orient_normals_consistent_tangent_plane(100)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # for alpha in np.logspace(np.log10(1), np.log10(0.5*avg_dist), num=6):
    # # print(f"alpha={alpha:.3f}")
    alpha = .3
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    draw(mesh)
    breakpoint()
    radii = [.3]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    rec_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([rec_mesh])
    breakpoint()
    # # mesh.paint_uniform_color([1,0,1])
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([mesh])
  # rec_mesh =pivot_ball_mesh(lowc.voxel_down_sample(.1)) 
    # rec_mesh =pivot_ball_mesh(lowc,[0.2,0.5,0.7,.9,1,1.2,1.5,1.7,2]) 
    # rec_mesh.compute_vertex_normals()
    # tmesh = meshfix(rec_mesh,algo= 'repair')
    # mesh = tmesh.to_legacy()
    # mesh0 = get_surface_clusters(mesh,top_n_clusters=None)

    rec_mesh =pivot_ball_mesh(pcd,[0.2,0.5,0.7,.9,1,1.2,1.5,1.7,2]) 
    rec_mesh.compute_vertex_normals()
    tmesh = meshfix(rec_mesh,algo= 'repair')
    mesh = tmesh.to_legacy()
    mesh0 = get_surface_clusters(mesh,top_n_clusters=None)

    breakpoint()

    rec_mesh = pivot_ball_mesh(pcd.uniform_down_sample(10),True)
    pivot_ball_mesh(lowc)   # close but sparse
    pivot_ball_mesh(target) # sparse, only trunk cover
    radii_factors = [0.2,0.5,0.7,1,1.5, 2]
    pivot_ball_mesh(lowc,radii_factors)
    pivot_ball_mesh(lowc.voxel_down_sample(.1),radii_factors)
    pivot_ball_mesh(target,radii_factors)
    pivot_ball_mesh(target.voxel_down_sample(.1),plot_distribution=True)
    breakpoint()
    rec_mesh =pivot_ball_mesh(lowc.voxel_down_sample(.1)) 
    tmesh = meshfix(rec_mesh)
    # mesh.rotate(rot_90_x,center = mesh.get_center())
    # mesh.rotate(rot_90_x,center = mesh.get_center())
    # mesh.rotate(rot_90_x,center = mesh.get_center())
    # raycast(tmesh)
    # mri(tmesh)
    # cast_rays(tmesh)
    breakpoint()

    # breakpoint()
    rec_mesh = pivot_ball_mesh(pcd.uniform_down_sample(10),True)
    rec_mesh = pivot_ball_mesh(lowc)
    rec_mesh = pivot_ball_mesh(target,True)
    test = target.uniform_down_sample(.1)
    rec_mesh = pivot_ball_mesh(test,True)
    
    # mesh_geoms = check_properties(f'{seed}_mesh',mesh)


def radius_search(pcd, radius):
    print("Loading pointcloud ...")
    # sample_pcd_data = o3d.data.PCDPointCloud()
    # pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
    print(
        "Find the neighbors of 50000th point with distance less than 0.2, and painting them green ..."
    )
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[50000], radius)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    o3d.visualization.draw([pcd])
    return k,idx


def knn_search():
    print("Loading pointcloud ...")
    sample_pcd = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the 2000 nearest neighbors of 50000th point, and painting them red ..."
    )
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[50000], 2000)
    np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])


if __name__=="__main__":
    get_mesh()