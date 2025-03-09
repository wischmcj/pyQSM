
# max_cluster = o3d.io.read_point_cloud("skeletor_trunk.pcd")
# points = np.asarray(max_cluster.points)
# # Build point cloud Laplacian
# L, M = robust_laplacian.point_cloud_laplacian(points,1e-10)

# # (or for a mesh)
# # L, M = robust_laplacian.mesh_laplacian(verts, faces)

# # Compute some eigenvectors
# n_eig = 10
# evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

# # Visualize
# ps.init()
# ps_cloud = ps.register_point_cloud("my cloud", points)
# for i in range(n_eig): 
#     ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)
# breakpoint()
# ps.show()

