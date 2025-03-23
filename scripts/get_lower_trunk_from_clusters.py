 lowc  = o3d.io.read_point_cloud('new_low_cloud_all_16-18pct.pcd')
    # highc = o3d.io.read_point_cloud('new_collective_highc_18plus.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl')
    empty_pcd= o3d.geometry.PointCloud() 
    # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
    c_trunk_clusters = create_one_or_many_pcds([arr(lc.points) for lc in trunk_clusters],labels=[k for k,v in label_to_clusters.items()])
    # draw(vlow)
    # breakpoint()
    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    # other_clusters = [(idc,cluster) if idc in final_int_ids else (idc,empty_pcd) for idc,cluster in enumerate(c_trunk_clusters) ]
    # extend_seed_clusters(clusters_and_idxs = other_clusters,
    #                         src_pcd=highc,
    #                         file_label='other_clusters',cycles=150,save_every=20, max_distance=.1)
    all_ids = rf_seeds + final_int_ids 
    pcd = o3d.io.read_point_cloud('partitioning_search_area_collective.pcd')
    vlow, _ = crop_by_percentile(pcd, 4,17)
    all_clusters = [(idc,empty_pcd) if idc not in all_ids else (idc, cluster) for idc,cluster in enumerate(c_trunk_clusters) ]
    extend_seed_clusters(clusters_and_idxs = all_clusters,
                            src_pcd=vlow,
                            file_label='cluster_roots',cycles=20,save_every=10,draw_every=3)