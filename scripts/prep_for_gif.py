  
    
    ## Finding point to rotate around
    base_center = get_center(np.asarray(trunk.points),center_type = "bottom")
    mn = trunk.get_min_bound()
    centroid = mn+((mx-mn)/2)
    base = (base_center[0], base_center[1], mn[2])
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sp.paint_uniform_color([1,0,0])
    sp.translate(base)

    for pcd in [trunk,contracted,skeleton,topology]:
        pcd.translate(np.array([-x for x in base_center ]))
        pcd.rotate(rot_90_x)
        pcd.rotate(rot_90_x)
        pcd.rotate(rot_90_x)

    # draw([ttrunk,trunk])
    # draw([contracted,sp])
    # draw([skeleton])
    # draw([topology])
    
    gif_center = get_center(np.asarray(trunk.points),center_type = "centroid")
    eps=config['trunk']['cluster_eps']
    min_points=config['trunk']['cluster_nn']
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))

    animate_contracted_pcd(trunk,topology, point_size=4, rot_center = gif_center, steps=360,  transient_period= 40, save = True,  file_name = '_ste,20_whole_and_topo')
    
    # animate_contracted_pcd(pcd_colored,  topology, point_size=3,  
    #                        rot_center = gif_center,  steps=360, save = True, file_name ='_min_w_topo') 
    # animate_contracted_pcd(trunk.voxel_down_sample(.05),  topology,  point_size=3,  rot_center = gif_center,   steps=360, save = True, file_name = '_min_w_topo' 
    