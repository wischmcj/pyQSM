
def step_through_and_evaluate_clusters():
    # file = "cell3_complete2.pkl"
    file = "completed_cluster_idxs.pkl"

    idxs, pts = load_completed(0, [file])
    # lowc = o3d.io.read_point_cloud('low_cloud_all_16-18pct.pcd')
    # highc= o3d.io.read_point_cloud('collective_highc_18plus.pcd')
    collective= o3d.io.read_point_cloud('collective.pcd')
    clusters = create_one_or_many_pcds(pts)

    breakpoint()
    # draw(clusters)
    good_clusters = [] 
    multi_clusters = [] 
    partial_clusters = []
    bad_clusters = []
    factor = 5
    for idc, cluster in zip(idxs,clusters):
        is_good = 0
        is_multi = 0
        partial = 0
        region = [( cluster.get_min_bound()[0]-factor,
                         cluster.get_min_bound()[1]-factor),     
                        (cluster.get_max_bound()[0]+factor,
                        cluster.get_max_bound()[1]+factor)]
        zoomed_col = zoom_pcd(region, collective)
        # cluster.paint_uniform_color([1,0,0])
        draw([zoomed_col,cluster])

        # tree = sps.KDTree(np.asarray(arr(zoomed_col.points)))
        # dists,nbrs = tree.query(curr_pts[idx],k=750,distance_upper_bound= .3) #max(cluster_extent))
        # nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=num_pts]
        # nbr_pts = [nbr_pt for nbr_pt in highc_tree.data[nbrs]]      
        breakpoint()
        if is_good ==1:
            good_clusters.append((file,idc))
        elif is_multi==1:
            multi_clusters.append((file,idc))
        elif partial==1:
            partial_clusters.append((file,idc))
        else: 
            bad_clusters.append((file,idc))
    try:
        update('good_clusters.pkl',good_clusters)
        update('partial_clusters.pkl',partial_clusters)
        update('bad_clusters.pkl',bad_clusters)
        update('multi_clusters.pkl',multi_clusters)
    except Exception as e:
        breakpoint()
        print('failed writing to files')