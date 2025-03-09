# breakpoint()
    ## Finding point to rotate around
    base_center = get_center(np.asarray(trunk.points),center_type = "bottom")
    # mn = trunk.get_min_bound()
    # centroid = mn+((mx-mn)/2)
    # base = (base_center[0], base_center[1], mn[2])
    # sp = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # sp.paint_uniform_color([1,0,0])
    # sp.translate(base)

    
    # for eapcd in [pcd, trunk,contracted,skeleton,topology]:
    #     eapcd.translate(np.array([-x for x in base_center ]))
    #     eapcd.rotate(rot_90_x)
    #     eapcd.rotate(rot_90_x)
    #     eapcd.rotate(rot_90_x)

    # draw([ttrunk,trunk])
    # draw([contracted,sp])
    # draw([skeleton])
    # draw([topology])
    
    
    # gif_center = get_center(np.asarray(trunk.points),center_type = "centroid")
    # eps=config['trunk']['cluster_eps']
    # min_points=config['trunk']['cluster_nn']
    # labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
    
    # breakpoint()
    # animate_contracted_pcd(trunk,topology, point_size=4,rot_center = gif_center, steps=360, transient_period= 40, save = True, file_name = '_super_whole_and_topo')
    
    # animate_contracted_pcd(pcd_colored,  topology, point_size=3,  
    #                        rot_center = gif_center,  steps=360, save = True, file_name ='_min_w_topo') 
    # animate_contracted_pcd(trunk.voxel_down_sample(.05),  topology,  point_size=3,  rot_center = gif_center,   steps=360, save = True, file_name = '_min_w_topo' 
    
    ####
    #    there is a bijection trunk to contracted - Same ids
    #    there is a bijection contracted to skeleton_points - same ids
    #    there is a surgection skeleton_points to skeleton, skeleton_graph
    #       ****Ids wise skeleton points = contracted.points = trunk.points
    #    there is a surjection skeleton to topology graph
    #    there is a surjection (albeit, of low order) topology_graph to topologogy
    ####


    points = np.asarray(topology.points)
    lines = np.asarray(topology.lines)
    line_lengths = np.linalg.norm(points[lines[:, 0]] - points[lines[:, 1]], axis=1)
    edges = topology_graph.edges(data=True)
    edge_to_orig = {tuple((x[0],x[1])):x[2].get('data') for x in edges}
    all_verticies = list(chain.from_iterable([x[2].get('data',[]) for x in edges]))
    
    orig_verticies = [x for x in edge_to_orig.values()]
    most_contracted_idx = np.argmax([len(x) for x in orig_verticies if x is not None])
    most_contracted_list = orig_verticies[most_contracted_idx]

    print("Mean:", np.mean(line_lengths))
    print("Median:", np.median(line_lengths))
    print("Standard Deviation:", np.std(line_lengths))
    colors = [[0,0,0]]*len(line_lengths)
    long_line_idxs = np.where(line_lengths>np.percentile(line_lengths,60))[0]
    for idx in long_line_idxs: colors[idx] = [1,0,0]
    topology.colors = o3d.utility.Vector3dVector(colors)

    contraction_dist = np.linalg.norm(total_point_shift, axis=1)
    print("Max Contraction:", np.max(contraction_dist))
    print("Mean Contraction:", np.mean(contraction_dist))
    print("Median Contraction:", np.median(contraction_dist))
    print("Standard Deviation Contraction:", np.std(contraction_dist))
    contracted_to_orig = {v:k for k,v in orig_to_contracted.items()} # mapping is a bijection

    # import pickle as pk
    # skeletor_hyper_qsm_state = [trunk, skeleton, skeleton_graph, contracted, topology, topology_graph, total_point_shift,]
    # with open('skeletor_hyper_qsm_state.pkl', 'wb') as f: pk.dump(skeletor_hyper_qsm_state, f)
    # with open('skeletor_hyper_qsm_state.pkl', 'rb') as f: skeletor_hyper_qsm_state = pk.load(f)
    # trunk, skeleton, skeleton_graph, contracted, topology, topology_graph, total_point_shift = skeletor_hyper_qsm_state
   
    # For point with idp in topology.points, 
    #   point = skeleton_points[contracted_to_orig[idp]]
    #
    # For point w/ idp in pcd, and idc in (0,1,2),
    #  absolute difference of pcd.points[idp][idc]
    #  and skeleton_points[idp][idc] is total_point_shift[idp][idc]
    test = False
    cyls = []
    cyl_objects = []
    # pcd_pts_contained = []
    # pcd_points = np.array(pcd_colored.points)
    for idx, line in enumerate(lines):
        from skspatial.objects import Cylinder
        start = points[line[0]]
        end = points[line[1]]
        vector = end-start
        line_dist = np.linalg.norm(vector)
        uvector = unit_vector(vector)    

        orig_verticies = edge_to_orig[tuple(line)] 
        contraction_dists = contraction_dist[orig_verticies]
        
        cyl_radius= np.mean(contraction_dists)/2
        cyl  = Cylinder.from_points(start,end,cyl_radius)
        cyl_pts = cyl.to_points(n_angles=30, n_along_axis = 70).round(3).unique()
        cyl_pcd = o3d.geometry.PointCloud()
        cyl_pcd.points = o3d.utility.Vector3dVector(cyl_pts)
        cyls.append(cyl_pcd)
        cyl_objects.append(cyl)
        
        if test:
            breakpoint()
            print('test')
        if idx %20 == 0:    
            print(f'finished iteration {idx}')
        

        # for idp, point in enumerate(pcd_points): 
        #     if cyl.is_point_within(point):
        #         pcd_pts_contained.append(point)
        #         pcd_points.pop(idp)
        # if idx %10 == 0:
        #     log.info(f'finished iteration {idx}')
        #     draw(cyls)
        #     breakpoint()
        #     print('checkin')
    # contained_pcd = o3d.geometry.PointCloud()
    # contained_pcd.points = o3d.utility.Vector3dVector(pcd_pts_contained)
    pts = []
    for cyl in cyls: 
        pts.extend(np.array(cyl.points))
    all_cyl_pcd= o3d.geometry.PointCloud()
    all_cyl_pcd.points = o3d.utility.Vector3dVector(pts)
    draw([all_cyl_pcd,pcd])
    breakpoint()
    # labels = np.array( trunk.cluster_dbscan(eps=0.01, min_points=15, print_progress=True)) 
    labels = np.array( kmeans(np.asarray(trunk.points),6))
    max_label = labels.max()
    log.info(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    trunk.colors = o3d.utility.Vector3dVector(colors[:, :3])
    draw(trunk)

    # gif_center = get_center(np.asarray(trunk.points),center_type = "bottom")
    # animate_contracted_pcd( all_cyl_pcd,trunk.voxel_down_sample(.05),  point_size=3, rot_center = gif_center, steps=360, save = True, file_name = '_proto_qsm_trunk',transient_period = 30)
    # animate_contracted_pcd( all_cyl_pcd,topology,  point_size=3, rot_center = gif_center, steps=360, save = True, file_name = '_proto_qsm_topo',transient_period = 30)
    
    breakpoint()


# max_cluster = o3d.io.read_point_cloud("skeletor_trunk.pcd")
# points = np.asarray(max_cluster.points)
# # Build point cloud Laplacian
# L, M = robust_laplacian.point_cloud_laplacian(points,1e-10)

# # (or for a mesh)
# # L, M = robust_laplacian.mesh_laplacian(verts, faces)

# # Compute some eigenvectors
# n_eig = 10
# evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

# # Visualize
# ps.init()
# ps_cloud = ps.register_point_cloud("my cloud", points)
# for i in range(n_eig): 
#     ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)
# breakpoint()
# ps.show()