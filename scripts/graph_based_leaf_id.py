 
    
    # breakpoint()
    from viz.color import color_distribution
    from utils.lib_integration import get_pairs
    # source_pcd = read_point_cloud(f'{inputs}/skeletor_clean.pcd')
    trim = read_point_cloud(f'{inputs}/trim/skeletor_full_ds2.pcd')
    # source_pcd = read_point_cloud(f'{inputs}/skeletor_clean_ds2.pcd') 
    # file = 'data/skeletor/skel_clean_ext_full.pkl'   
    # pcds= pcds_from_extend_seed_file(file)
    # # branches = []
    # # leaves = []
    # # low_idx_lists= []
    # # # degree_lists = [] 
    # for idp, pcd in enumerate(pcds):

    
    plot_dist_dist(pcd)
    distances = pcd.compute_nearest_neighbor_distance()
    pts = arr(pcd.points)
    detailed_branch, _, analouge_idxs = get_neighbors_kdtree(trim, query_pts=pts,k=200, dist=.1)
    print(f'Orig Size {len(pts)}')
    print(f'Detailed Size {len(detailed_branch.points)}')
    if idp<2:
        draw(detailed_branch)
    write_point_cloud(f'data/skeletor/branch_and_leaves/detailed_branch{idp}.pcd',pcd)
    # local_density = []  
    degrees,cnts,line_set= get_pairs(query_pts=arr(pcd.points),radius=.3)
    # breakpoint()
    # density_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr(pcd.points)))
    # color_continuous_map(density_pcd,arr(degrees))
    # draw(density_pcd)
    lowd_idxs = np.where(degrees<np.percentile(degrees,30))[0]
    branch = pcd.select_by_index(lowd_idxs)
    # leafs = pcd.select_by_index(lowd_idxs,invert=True)
    branches.append(branch)
    # leaves.append(leafs)
    low_idx_lists.append(lowd_idxs)
    # draw(branch)
    # draw(leafs)
    print(f'finished {idp}')
    draw(branch)
    draw(leaves)

    colored_jleaves, _, analouge_idxs = get_neighbors_kdtree(trim, jleaves, dist=.01)
    colored_leaves, _, analouge_idxs = get_neighbors_kdtree(colored_pcd, leaves, dist=.01)
    colored_branch, _, analouge_idxs = get_neighbors_kdtree(colored_pcd, branch, dist=.01)
    # colored_branch = colored_pcd.select_by_index(lowd_idxs,)
    
    draw(colored_leaves)
    draw(colored_branch)
    # breakpoint()
    leaf_colors = arr(colored_leaves.colors)
    all_colors = arr(colored_branch.colors)
    try:
        _, hsv_fulls = color_distribution(leaf_colors,all_colors,cutoff=.1,space='',sc_func =lambda sc: sc)
    except Exception as e:
        breakpoint()
        print('err')
    breakpoint()
    # orig_detail = source_pcd.select_by_index(analouge_idxs)
    # draw(orig_detail)
    ds_pcd = pcd.uniform_down_sample(3)
    draw(ds_pcd)
    res = get_mesh(pcd,ds_pcd,pcd)
    breakpoint()
    jleaves = join_pcds(leaves)[0]
    jbranches= join_pcds(branches)[0]
    clusters = cluster_plus(jbranches,eps=.2, min_points=50,return_pcds=True,from_points=False)
    test = read_point_cloud('branches_first5_pt3_25pct.pcd')
    breakpoint()
    write_point_cloud('leaves_first5_pt3_25pct.pcd',jleaves)
    breakpoint()
    dtrunk= read_point_cloud(f'{inputs}/skeletor_dtrunk.pcd')
    joined = read_point_cloud('branches_pt3_30pct.pcd')
    
    import rustworkx as rx
    conn_comps_list = []
    final_list = []
    for radius in [.06,.07,.08,.09]:
        degrees,cnts,line_set= get_pairs(query_pts=arr(joined.points),radius=radius, return_line_set=True)
        # draw(line_set)
        g = rx.PyGraph()
        g.add_nodes_from([idp for idp,_ in enumerate(arr(line_set.points))])
        g.add_edges_from_no_data([tuple(pair) for pair in arr(line_set.lines)])
        conn_comps = arr([x for x in sorted(rx.connected_components(g), key=len, reverse=True)])
        print(len(conn_comps))
        print([len(x) for x in conn_comps])

        large_comp_pt_ids = list(chain.from_iterable([x for x in conn_comps[0:500] if len(x)>100]))
        candidate_comps = list(chain.from_iterable([x for x in conn_comps[0:100]]))
        new_pts = arr(line_set.points)
        new_lines = g.edges(candidate_comps)
        new_line_set = o3d.geometry.LineSet()
        new_line_set.points = o3d.utility.Vector3dVector(new_pts)
        new_line_set.lines = o3d.utility.Vector2iVector(new_lines)
        draw(new_line_set)
        branches, _, analouge_idxs = get_neighbors_kdtree(trim, query_pts=new_pts,k=200, dist=.01)
        draw(branches)
        # from viz.color import segment_hues
        # hue_pcds,no_hue_pcds =segment_hues(branches,'test',draw_gif=False, save_gif=False)
        # res = plt.hist(arr(hsv_fulls[0])[:,0],nbins,facecolor=rgb)

        branch_nbrs, _, analouge_idxs = get_neighbors_kdtree(branches,dtrunk,k=500, dist=2)
        draw(branch_nbrs)
        obb = dtrunk.get_minimal_oriented_bounding_box()
        obb.color = (1, 0, 0)
        obb.scale(1.2,center=obb.get_center())
        obb.translate((0,0,-6),relative = True)
        draw([obb,branches])
        test = obb.get_point_indices_within_bounding_box( o3d.utility.Vector3dVector(arr(branches.points)) )
        nbrs = branches.select_by_index(test)
        draw(nbrs)
        test = cluster_plus(arr(nbrs.points),eps=.1)
        clusters_and_idxs = [(idc, t) for idc, t in enumerate(test)]
        tree_pcds = extend_seed_clusters(clusters_and_idxs, branches, 'skeletor4', k=50, max_distance=.1, cycles= 200, save_every = 50, draw_every = 50)
        draw(branch_nbrs)
        
        
        conn_comps_list.append(conn_comps)
        final_list.append(branches)