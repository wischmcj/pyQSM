
    skeletor_whole_trunk =f"skel_iso_trunk.pcd"
    iso_trunk = o3d.io.read_point_cloud(f"skel_iso_trunk.pcd") # Full resolution
    iso_trunk= iso_trunk.uniform_down_sample(4) 

    iso_not_trunk = o3d.io.read_point_cloud(f"skel_iso_nottrunk.pcd") # low res
    skeletor = o3d.io.read_point_cloud(f"skeletor_cleaned.pcd") # full res
    print('attempting_to_read skeletor')

    # obb = iso_trunk.get_oriented_bounding_box()
    # skel_pts = o3d.utility.Vector3dVector(arr(skeletor.points))
    # in_pt_ids = obb.get_point_indices_within_bounding_box(skel_pts) 
    # in_pt_pcd = skeletor.select_by_index(in_pt_ids)
    # in_pts = arr(skeletor.points)[in_pt_ids]
    # query_pts = arr(iso_trunk.points) 
    # full_res_trunk_tree = sps.KDTree(arr(in_pts))
    # _,close_nbrs = full_res_trunk_tree.query(query_pts, k=200, distance_upper_bound= .1) 
    # close_nbrs = [x for x in set(chain.from_iterable(close_nbrs)) if x!= len(in_pts)]
    # trunk_in_skel = skeletor.select_by_index(in_pt_ids[close_nbrs])
    # draw(trunk_in_skel)
    # skel_no_stem = skeletor.select_by_index(arr(in_pt_ids)[close_nbrs], invert=True)
    # o3d.io.write_point_cloud('skel_no_stem.pcd',skel_no_stem)
    skel_no_stem = o3d.io.read_point_cloud('skel_no_stem.pcd')
    

    # query_pts = arr(iso_trunk.points) 
    # full_res_trunk_tree = sps.KDTree(arr(skel_no_stem.points))
    # print('Finding neighbors in vicinity') 
    # dists,nbrs = full_res_trunk_tree.query(query_pts, k=200, distance_upper_bound= .3) 
    # nbrs = [x for x in set(chain.from_iterable(nbrs)) if x!= len(in_pts)]
    # bo1 = skel_no_stem.select_by_index(nbrs)
    # bo1_clean = crop_by_percentile(bo1,57,100)
    # o3d.io.write_point_cloud('skeletor_bo1_clusters.pcd',bo1_clean)
    bo1_clean = o3d.io.read_point_cloud('skeletor_bo1_clusters.pcd')
    # print('removing points already present in stem pcd') 
    # non_stem = [pt for pt in arr(bo1.points) if pt not in query_pts]
    # non_stem_pcd = o3d.geometry.PointCloud()
    # non_stem_pcd.points = o3d.utility.Vector3dVector(non_stem)
    # draw(non_stem_pcd)

    skel_remaining = o3d.io.read_point_cloud('skel_remaining.pcd')
    # skel_remaining = skel_no_stem.select_by_index(nbrs,invert=True)
    bo1_clean = clean_cloud(bo1_clean)

    query_pts = arr(bo1_clean.points) 
    full_res_trunk_tree = sps.KDTree(arr(skel_remaining.points))
    print('Finding neighbors in vicinity') 
    dists,nbrs = full_res_trunk_tree.query(query_pts, k=200, distance_upper_bound= .3) 
    nbrs = [x for x in set(chain.from_iterable(nbrs)) if x!= len(arr(skel_remaining.points))]
    bo1_2 = skel_remaining.select_by_index(nbrs)
    draw(bo1_2)
    draw([bo1_2, iso_trunk])
    breakpoint()
    # bo1_2_clean = crop_by_percentile(bo1,57,100)
    # o3d.io.write_point_cloud('skeletor_bo1_2_clusters.pcd',bo1_2_clean)

    skel_remaining_2 =  skel_remaining.select_by_index(nbrs,invert=True)
    bo1_2_clean = clean_cloud(bo1_2)

    query_pts = arr(bo1_2_clean.points) 
    full_res_trunk_tree = sps.KDTree(arr(skel_remaining_2.points))
    print('Finding neighbors in vicinity') 
    dists,nbrs = full_res_trunk_tree.query(query_pts, k=200, distance_upper_bound= .3) 
    nbrs = [x for x in set(chain.from_iterable(nbrs)) if x!= len(arr(skel_remaining_2.points))]
    bo1_3 = skel_remaining_2.select_by_index(nbrs)
    draw(bo1_3)
    draw([bo1_3, iso_trunk])

    # o3d.io.write_point_cloud('skeletor_bo1_3_clusters.pcd',bo1_3)
    bo1_3 =o3d.io.read_point_cloud('skeletor_bo1_3_clusters.pcd')
    # bo1_3_clean = clean_cloud(bo1_3)
    # skel_remaining_3 = skel_remaining_2.select_by_index(nbrs,invert=True)
    # o3d.io.write_point_cloud('skel_remaining_3.pcd',skel_remaining_3)
    skel_remaining_3 = o3d.io.read_point_cloud('skel_remaining_3.pcd')
    bo1_3_clean = clean_cloud(bo1_3)

    labels, colors= cluster_plus(bo1_3_clean,eps = .11, min_points=20)
    draw(bo1_3_clean)

    unique_vals, counts = np.unique(labels, return_counts=True)
    print(unique_vals)
    print(counts)
    label_idls= [ np.where(labels ==val)[0] for val in unique_vals if val!=-1]
    clusters = [(idc,bo1_3_clean.select_by_index(idls)) for idc, idls in enumerate(label_idls)]
    # draw(arr(clusters)[:,1])
    # draw(list(arr(clusters)[:,1]) +[iso_trunk])
    # o3d.io.write_point_cloud('skeletor_bo0_clusters.pcd',bo1)
    
    lrg_clusters = extend_seed_clusters(clusters, skel_remaining_3, 
    file_label= 'skel', k=300, max_distance=.1, cycles= 250,  
    save_every = 20, draw_progress = True, debug = True)