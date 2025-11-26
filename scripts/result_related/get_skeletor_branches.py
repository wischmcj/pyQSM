
    # write_pcd('./TreeLearn/data/collective_clean.pcd',clean_full)
    # breakpoint()
    # full = read_pcd('data/skeletor/inputs/skeletor_full_ds2.pcd')
    # trunk = read_pcd('data/skeletor/inputs/skeletor_trunk_isolated.pcd')
    # trunk = trunk.uniform_down_sample(6)
    # nbr_pcd = read_pcd('data/skeletor/exts/skel_branch_0_orig_detail_ds2.pcd')
    # import pickle 
    # with open('data/skeletor/seeds/skeletor_final_branch_seeds.pkl','rb') as f:
    #     labeled_cluster_pcds = pickle.load(f)
    # clusters = []
    # for idc, pt_list in labeled_cluster_pcds:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(arr(pt_list))
    #     clusters.append((idc,pcd))

    # res = extend_seed_clusters(clusters,full,file_label='skel_tb_test',cycles=125,save_every=60,draw_every=200,tb_every=2,exclude_pcd=trunk)
    