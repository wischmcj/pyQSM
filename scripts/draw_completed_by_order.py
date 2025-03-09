with open('skel_w_order_complete.pkl','rb') as f:
        # dict s.t. {cluster_id: [list_of_pts]}
        skel_completed = dict(pickle.load(f))
    pts = [[y[0] for y in x] for idx, x in enumerate(skel_completed.values()) if idx = 8]
    labels =[arr([y[1] for y in x]) for x in skel_completed.values()] 
    pcds = create_one_or_many_pcds(pts)
    # pcd = create_one_or_many_pcds(pts, single_pcd = True)[0]
    # contracted, total_point_shift, shift_by_step = extract_skeleton(pcd, max_iter = 1, debug=True)
    
    branches = []
    leaves = []
    for idx, pcd_label_list in enumerate(zip(pcds,labels)):
        cvar = label_list    
        prev = 0

        print(f'{np.max(label_list)}')
        for idl, label in enumerate(label_list):
            diff = label - prev                
            cvar[idl] = diff
        avg_diff = np.mean([x for x in cvar if x >0])
        print(f'{avg_diff=}')

        colors = plt.get_cmap('plasma')((cvar - cvar.min()) / (cvar.max() - cvar.min()))
        colors = colors[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # cutoff = np.percentile(cvar,20)
        high_idxs = np.where(cvar<avg_diff)[0]
        new_pcd= pcd.select_by_index(high_idxs)
        branches.append(new_pcd)
        new_pcd= pcd.select_by_index(high_idxs, invert= True)
        leaves.append(new_pcd)
    breakpoint()
    # draw(pcds)
    # draw(branches)
    # draw(leaves)
    branch_pts = list([list(arr(branch.points)) for branch in branches])
    all_pts = []
    for pts in branch_pts: all_pts.extend(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)    