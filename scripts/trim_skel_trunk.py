
    stem_trunk = o3d.io.read_point_cloud("skeletor_stem20.pcd")
    stem_trunk = stem_trunk.uniform_down_sample(4)
    # topo = load_topo('skel_stem20_moll1e-6')
    # draw([topo,trunk])
    # breakpoint()

    whole_tree = o3d.io.read_point_cloud("skeletor_cleaned.pcd")
    whole_tree = whole_tree.uniform_down_sample(4)
    # draw(whole_tree)
    # draw(stem_trunk)
    
    #############
    ### COMMON CENTER FOR SKELETOR
    ### trunk.get_max_bound()=array([-135.15299988,  192.19450378,   27.20366669]), 
    ### trunk.get_min_bound()=array([-146.81143188,  184.83717346,   -1.80327272])
    ### center = arr([-140.98221588,  188.51583862,   12.70019698])
    #############

    trunk = o3d.io.read_point_cloud("skel_stem_from_contraction.pcd")
    center =trunk.get_min_bound()+((trunk.get_max_bound() -trunk.get_min_bound())/2)
    trunk.translate(-center)
    print(f'{trunk.get_max_bound()=}, {trunk.get_min_bound()=}')
    obb = trunk.get_oriented_bounding_box()
    obb.color = (1, 0, 0)
    # trunk.transform(arr([[.8,0,0,0],[0,.8,0,0],[0,0,1,0],[0,0,0,1]]))

    # limit up to lowest_branch
    branch_grp1,tt_trunk_idxs = crop_by_percentile(trunk,53,90)
    print(f'selecting from branch_grp')
    removed = trunk.select_by_index(tt_trunk_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch_grp1,removed])

    branch_limited_x,blx_idxs = crop_by_percentile(branch_grp1,18,70,axis = 0)
    # blx_trunk_idxs = tt_trunk_idxs[blx_idxs]
    print(f'selecting from branch_grp')
    removed = branch_grp1.select_by_index(blx_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch_limited_x,removed])

    branch_limited,bl_idxs = crop_by_percentile(branch_limited_x,13,87,axis = 1)
    bl_trunk_idxs = tt_trunk_idxs[bl_idxs]
    print(f'selecting from branch_limited_x')
    removed = branch_limited_x.select_by_index(bl_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch_limited,removed])

    branch_limited_xy,blxy_idxs = crop_by_percentile(branch_limited,0,95,axis = [0,1])
    # blxy_trunk_idxs = tt_trunk_idxs[blxy_idxs]
    print(f'selecting from branch_limited_x')
    removed = branch_limited.select_by_index(blxy_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch_limited_xy,removed])


    branch_limited_xmy,blxmy_idxs = crop_by_percentile(branch_limited_xy,3,99,axis = [0,1], invert = True)
    # blxy_trunk_idxs = tt_trunk_idxs[blxy_idxs]
    print(f'selecting from branch_limited_x')
    removed = branch_limited_xy.select_by_index(blxmy_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch_limited_xmy,removed])
    
    breakpoint()
#################################################33


    # limit up to lowest_branch
    branch_grp2,tt_trunk_idxs = crop_by_percentile(trunk,90,93)
    print(f'selecting from branch_grp')
    draw(branch_grp2)

    branch2_limited_x,blx_idxs = crop_by_percentile(branch_grp2,6,100,axis = 0)
    # blx_trunk_idxs = tt_trunk_idxs[blx_idxs]
    print(f'selecting from branch_grp')
    removed = branch_grp2.select_by_index(blx_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch2_limited_x,removed])

    branch2_limited,bl2_idxs = crop_by_percentile(branch2_limited_x,10,100,axis = [0,1])
    print(f'selecting from branch_limited_x')
    removed = branch2_limited_x.select_by_index(bl2_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    draw([branch2_limited,removed])


    lower_trunk,lt_idxs = crop_by_percentile(trunk,0,53)
    stem_pts = np.concatenate([arr(lower_trunk.points),
                                arr(branch2_limited.points),
                                arr(branch_limited_xmy.points), ])
    iso_trunk = o3d.geometry.PointCloud()
    iso_trunk.points = o3d.utility.Vector3dVector(stem_pts)
    o3d.io.write_point_cloud(f"skel_iso_trunk.pcd", iso_trunk)
    
