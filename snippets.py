
# reversibility of rotation
    arr = unit_vector([-.2,.4,-.5])
    low_cloud = stem_cloud.select_by_index(not_so_low_idxs)

    rot = low_cloud
    rot.rotate(rotation_matrix_from_arr([0,0,1],arr))
    low_cloud = stem_cloud.select_by_index(not_so_low_idxs)
    # mesh, trunk_pcd, inliers, fit_radius, axis = fit_shape(pcd = rot, shape = 'cylinder')
    draw([rot,low_cloud])

    rot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.1, max_nn=20))
    rot.normalize_normals()
    norms = np.array(rot.normals)
    
    axis_guess = orientation_from_norms(norms,  samples = 10, max_iter = 100 )
    R = rotation_matrix_from_arr(unit_vector(axis_guess),[0,0,1])
    rot.rotate(R)
    rot.paint_uniform_color([0,1.0,0])
    draw([low_cloud,rot])
    draw([rot])

## Clustering when bad cyl fit is found 

    #     orig_in_cyl_neighbors= new_neighbors[inliers]
    #     orig_in_cyl_points = main_pts[orig_in_cyl_neighbors]
    #     if len(orig_in_cyl_points)==0:
    #         breakpoint()
    #         print('no points')
    #     # It may be the case we have two or more cylinders in the inliers set cluster
    #     labels, returned_clusters, _ = cluster_neighbors(orig_in_cyl_neighbors, 
    #                                                         orig_in_cyl_points,
    #                                                         dist =dist)
    #     test = iter_draw(returned_clusters,main_pcd)
    #     if len(returned_clusters)>1:
    #         for label, cluster_idxs in zip(labels,returned_clusters):
    #             # cluster_cloud = main_pcd.select_by_index(cluster_idxs)
    #             cluster_pts = main_pts[cluster_idxs]
    #             total_found.extend(cluster_idxs)
    #             mesh, _, inliers, fit_radius, _ = fit_shape(pts= cluster_pts, 
    #                                                         shape = 'circle',
    #                                                         threshold=0.04,  
    #                                                         lower_bound=prev_neighbor_height, 
    #                                                         max_radius=radius*1.5)
    #             cyls.append(mesh)
    #             cyl_details.append((inliers, nn_pts, fit_radius))
    #             in_cyl_neighbors.append(new_neighbors[inliers])
    #             inliers_list.append(inliers)
    #         to_cluster =  np.setdiff1d(new_neighbors, np.array(chain.from_iterable(in_cyl_neighbors))) 
    #         breakpoint()
        # else:
        #     to_cluster = new_neighbors

        # elif mesh is not None or fit_radius>=max_radius:
            # #try to fit just the first cluster found
            # orig_in_cyl_neighbors= new_neighbors[inliers]
            # orig_in_cyl_points = main_pts[orig_in_cyl_neighbors]
            # if len(orig_in_cyl_points)==0:
            #     breakpoint()
            #     print('no points')
            # # It may be the case we have two or more cylinders in the inliers set cluster
            # nn_points= main_pts[new_neighbors]
            # labels, returned_clusters, noise = cluster_neighbors(new_neighbors, nn_points, dist =dist*1.2)
            # for cluster_idxs in returned_clusters:
            #     # cluster_pts = np.asarray(main_pcd.points)[cluster_idxs]
            #     cluster_pcd = main_pcd.select_by_index(cluster_idxs)
            #     axis_guess = evaluate_axis(cluster_pcd)
            #     mesh, _, inliers, fit_radius, _ = z_align_and_fit(cluster_pcd,axis_guess,
            #                                                       shape = 'circle',threshold=0.04, 
            #                                             lower_bound=prev_neighbor_height,
            #                                             max_radius=radius*1.5)
            #     if (mesh is not None and 
            #             fit_radius < max_radius):
            #         in_cyl_neighbors = new_neighbors[inliers]
            #         inliers_list.append(inliers)
            #         cyls.append(mesh)
            #         cyl_details.append((inliers, nn_pts, fit_radius))
            #         to_cluster =  np.setdiff1d(new_neighbors, np.array(in_cyl_neighbors))
            #         break
            # if mesh is None:  
            #     print(f'fit_radius {fit_radius} is greater than max_radius {max_radius}')
            # mesh=None
            # to_cluster = new_neighbors