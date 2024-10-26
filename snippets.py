# testing the reversibility of rotation
# arr = unit_vector([-.2,.4,-.5])
# low_cloud = stem_cloud.select_by_index(not_so_low_idxs)

# rot = low_cloud
# rot.rotate(rotation_matrix_from_arr([0,0,1],arr))
# low_cloud = stem_cloud.select_by_index(not_so_low_idxs)
# # mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape(pcd = rot, shape = 'cylinder')
# draw([rot,low_cloud])

# rot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.1, max_nn=20))
# rot.normalize_normals()
# norms = np.array(rot.normals)

# axis_guess = orientation_from_norms(norms,  samples = 10, max_iter = 100 )
# R = rotation_matrix_from_arr(unit_vector(axis_guess),[0,0,1])
# rot.rotate(R)
# rot.paint_uniform_color([0,1.0,0])
# draw([low_cloud,rot])
# draw([rot])
