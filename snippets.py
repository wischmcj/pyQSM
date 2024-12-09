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



###### Iterating over clusters 

# i =19
    # cluster         = clusters[i][i]
    # cluster_ids     = clustered_ids[i]
    # cluster_extent  = cluster_extents[i]
    # center          = centers[i]
    # curr_pts        = cluster_pts[i]

    # other_ids = cluster_ids_in_col[0:i]
    # other_ids.extend(cluster_ids_in_col[i+1:])
    # other_ids = list(chain.from_iterable(other_ids))
    # nearby_ids = np.where(cluster_extents)
### REcreate kdtree
# if recreate:
#             breakpoint()
#             # Remove already assigned points from our tree
#             print('creating new tree')
#             # if we want to recreate tree each cycle'
#             tree_data = arr(highc_tree.data)
#             new_tree_data = np.delete(tree_data,traversed,axis=0)
#             highc_tree = sps.KDTree(new_tree_data)
#             traversed = []
#             print('created new tree')


### In points and out point comparison
# out_pts = np.delete(pts,pt_ids ,axis =0)
# in_pcd = o3d.geometry.PointCloud()
# in_pcd.points = o3d.utility.Vector3dVector(pt_values)
# out_pcd = o3d.geometry.PointCloud()
# out_pcd.points = o3d.utility.Vector3dVector(out_pts)
# in_pcd.paint_uniform_color([0,0,1])                    
# out_pcd.paint_uniform_color([1,0,0])
# bnd_box.paint_uniform_color([0,1,0])                   
# draw([in_pcd,out_pcd, bnd_box])

                        # breakpoint()

# Bounding boxes by XY only 

    # def within(extent, containing):
    #     all([
    #             all([pt[0][i]<bnds[0][i] for i in range(3) for pt,bnds in zip(extent,containing)]),
    #             all([pt[1][i]>bnds[1][i] for i in range(3) for pt,bnds in zip(extent,containing)])
    #         ])

    # def in_box(point,
    #             extent):
    #     x_min,x_max = extent[0]
    #     y_min,y_max = extent[1]
    #     if (( (point[0]>x_min and point[0]<x_max) and 
    #              (point[1]>y_min and point[1]<y_max))):
    #         return True
    #     return False
    # part_files = [[]]*len(bnd_boxes)
    # box_range = [(box.get_min_bound(),box.get_max_bound()) for cluster in clusters]
    # x_min,x_max,y_min,y_max = 
    # box_minmax =[((minv[0],minv[1],minv[2]),(maxv[0],maxv[1], maxv[2]))    for minv, maxv in box_range]
    # files = [[ file for file, extent in extents.items()
    #             if any(( (extent[0][idx]>bnds[0][idx] 
    #                         or extent[1][idx]<bnds[0][idx]   for idx in range(3))) ] 
    #             for bnds in box_range]
