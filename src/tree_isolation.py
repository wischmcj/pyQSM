import open3d as o3d
import scipy.spatial as sps
import numpy as np
import np.asarray as arr

import itertools

def find_extents(file_prefix = 'data/input/SKIO/part_skio_raffai',
                    return_pcd_list =False,
                    return_single_pcd = False):
    extents = {}
    pcds = []
    pts = []
    colors = []
    contains_region= []
    base = 20000000

    factor=5
    x_min, x_max= 85.13-factor, 139.06+factor
    y_min, y_max= 338.26-factor, 379.921+factor

    for i in range(40):
        num = base*(i+1) 
        file = f'{file_prefix}_{num}.pcd'
        pcd = read_point_cloud(file)
        print(f"{num}: ({extent[0]}, {extent[1]})")
        if return_pcd_list:
            pcds.append(pcd)
        elif return_single_pcd:
            pts.extend(list(arr(pcd.points)))   
            colors.extend(list(arr(pcd.colors)))
    if return_single_pcd:
        collective = o3d.geometry.PointCloud()
        collective.points = o3d.utility.Vector3dVector(pts)
        collective.colors = o3d.utility.Vector3dVector(colors)   
        pcds.append(pcd)     
    return extents, contains_region, pcds

def zoom(pcd,
        zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):

    collective = pcd
    low_pts = arr(collective.points)
    low_colors = arr(collective.colors)
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       and pt[1]>zoom_region[1][0] and  pt[1]<zoom_region[1][1])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in region]))
    pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in region]))
    draw(pcd)
    return pcd

def bin_colors(zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):
    # binning colors in pcd, finding most common
    # round(2) reduces 350k to ~47k colors
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) 
                if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       
                    and pt[1]>zoom_region[1][0] 
                    and  pt[1]<zoom_region[1][1])]

    cols = [','.join([f'{round(y,1)}' for y in x[1]]) for x in region]
    d_cols, cnts = np.unique(cols, return_counts=True)
    print(len(d_cols))
    print((max(cnts),min(cnts)))
    # removing points of any color other than the most common
    most_common = d_cols[np.where(cnts)]
    most_common_rgb = [tuple((float(num) for num in col.split(','))) for col in most_common]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) in most_common_rgb ]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) != tuple((1.0,1.0,1.0)) ]
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in color_limited]))
    limited_pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in color_limited]))
    draw(limited_pcd)

def load_file_or_files(files = ['cluster126_fact10_0to50.pkl'],
                        from_pickle = True,
                        single_pcd = False):
    import pickle
    contain_region = []    

    if from_pickle:
        # For loading results of KNN loop
        with open(pfile,'rb') as f:
            tree_clusters = dict(pickle.load(f))
        
    pcds = []
    tree_pts= []
    tree_color=[]
    labels = np.asarray([x for x in tree_clusters.keys()])
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # for pcd, color in zip(tree_pcds, colors): pcd.paint_uniform_color(color[:3])
    for pts, color in zip(tree_clusters.values(), colors): 
        if single_pcd:
            tree_pts.extend(pts)
            tree_color.extend(color[:3]*len(pts))
        else:
            cols = [color[:3]]*len(pts)
            print(f'creating pcd with {len(pts)} points')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            pcds.append(pcd)
    if single_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tree_pts)
        pcd.colors = o3d.utility.Vector3dVector([x for x in tree_color])
        pcds.append(pcd)
    return pcds

def recover_original_detail(cluster_pcds):
    pcds = load_clusters_pickle()
    bnd_boxes = [pcd.get_oriented_bounding_box() for pcd in pcds]
    
    base = 20000000
    file_prefix = 'data/input/SKIO/part_skio_raffai'
    for idb, bnd_box in enumerate(bnd_boxes):
        if idb>0:
            contained_pts = []
            all_colors = []
            for i in range(39):
                num = base*(i+1) 
                file = f'{file_prefix}_{num}.pcd'
                print(f'checking file {file}')
                pcd = read_point_cloud(file)
                pts = arr(pcd.points)
                if len(pts)>0:
                    cols = arr(pcd.colors)
                    pts_vect = o3d.utility.Vector3dVector(pts)
                    pt_ids = bnd_box.get_point_indices_within_bounding_box(pts_vect) 
                    if len(pt_ids)>0:
                        pt_values = pts[pt_ids]
                        colors = cols[pt_ids]
                        print(f'adding {len(pt_ids)} out of {len(pts)}')
                        contained_pts.extend(pt_values)
                        all_colors.extend(colors)
            try:
                file = f'whole_clus/cluster_{idb}_all_points.pcd'
                print(f'writing pcd file {file}')
                
                # Reversing the initial voxelization done
                #   to make the dataset manageable 
                # KNN search each cluster against nearby pts in
                #   the original scan. Drastically increase detail
                cluster_tree = sps.KDTree(arr(pcds[idb].points))
                whole_tree = sps.KDTree(contained_pts)
                # dists,nbrs = whole_tree.query(query_pts, k=750, distance_upper_bound= .2) 
                nbrs = cluster_tree.query_ball_tree(whole_tree, r= .3) 
                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x!= len(contained_pts)]
                print(f'{len(nbrs)} nbrs found for cluster {idx}')
                # for nbr_pt in nbr_pts:
                #     high_c_pt_assns[tuple(nbr_pt)] = idx


                # final_pts = np.append(arr(contained_pts)[nbrs])#,pcds[idb].points)
                # final_colors = np.append(arr(all_colors)[nbrs])#,pcds[idb].colors)
                final_pts =    arr(contained_pts)[nbrs]
                final_colors = arr(all_colors)[nbrs]
                wpcd = o3d.geometry.PointCloud()
                wpcd.points = o3d.utility.Vector3dVector(final_pts)
                wpcd.colors = o3d.utility.Vector3dVector(final_colors)  
                o3d.io.write_point_cloud(file, wpcd)

                draw(wpcd)
                out_pts = np.delete(contained_pts,nbrs ,axis =0)
                out_pcd = o3d.geometry.PointCloud()
                out_pcd.points = o3d.utility.Vector3dVector(out_pts)
                wpcd.paint_uniform_color([0,0,1])                    
                out_pcd.paint_uniform_color([1,0,0])                
                draw([wpcd,out_pcd])

                breakpoint()

            except Exception as e:
                breakpoint()
                print(f'error {e} getting clouds')

if __name__ =="__main__":
    #collective = load_file_or_files()
    # o3d.io.write_point_cloud('collective.pcd',collective)
    collective = o3d.io.read_point_cloud('collective.pcd')
    

    print('getting slightly above ground cross section')
    # lowc, lowc_ids_from_col = get_low_cloud(collective, 16,18)
    # lowc = clean_cloud(lowc)
    # highc, highc_ids_from_col = get_low_cloud(collective, 18,100)
    # o3d.io.write_point_cloud('low_cloud_all_16-18pct.pcd',lowc)
    # o3d.io.write_point_cloud('collective_highc_18plus.pcd',highc)
    # draw(lowc)
    lowc= o3d.io.read_point_cloud('low_cloud_all_16-18pct.pcd')
    highc= o3d.io.read_point_cloud('collective_highc_18plus.pcd')


    print('clustering')
    # Current settings, close trees being combined, need less neighbors, smaller eps
    # labels = np.array( lowc.cluster_dbscan(eps=.5, min_points=20, print_progress=True))   
    # with open('skio_labels_low_16-18_cluster_pt5-20.pkl','wb') as f:
    #     pickle.dump(labels,f)
    
    with open('skio_labels_low_16-18_cluster_pt5-20.pkl','rb') as f:
        labels = pickle.load(f)

    # Define subpcds implied by labels
    max_label = labels.max()
    unique_vals, counts = np.unique(labels, return_counts=True)
    label_idls = [ np.where(labels ==val)[0] for val in unique_vals]
    clusters = [lowc.select_by_index(idls) for idls in label_idls]
    cluster_centers = [get_center(arr(x.points)) for x in clusters]


# Filtering down the cluster list to reduce computation needed 
    
    ## Filtering out unlikley trunk canidates 
    # cluster_sizes = np.array([len(x) for x in label_idls])
    # large_cutoff = np.percentile(clu0ster_sizes,85)
    # large_clusters  = np.where(cluster_sizes> large_cutoff)[0]
    # draw(clusters)

    # for idc in large_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])
    # draw(clusters)
    # small_cutoff = np.percentile(cluster_sizes,30)
    # small_clusters  = np.where(cluster_sizes< large_cutoff)[0]
    # draw(clusters)
    # for idc in small_clusters: clusters[int(idc)].paint_uniform_color([1,0,0])
    # draw(clusters)

    # Limiting to just the clusters near cluster[19]
    factor = 10 # @40 reduces clusters 50%, @20 we get a solid 20 or so clusters in the
    center_id = 126
    zoom_region = [(clusters[center_id].get_max_bound()[0]+factor,    
                    clusters[center_id].get_min_bound()[0]-factor),     
                    (clusters[center_id].get_max_bound()[1]+factor,    
                    clusters[center_id].get_min_bound()[1]-factor)] # y max/min
    new_clusters = [cluster for cluster in clusters  if (cluster.get_max_bound()[0]<zoom_region[0][0] and              cluster.get_max_bound()[1]<zoom_region[1][0] and              cluster.get_min_bound()[0]>zoom_region[0][1] and              cluster.get_min_bound()[1]>zoom_region[1][1])]
    new_cluster_pts = [arr(cluster.points) for cluster in new_clusters]
    ## color and draw these local clusters
    clustered_pts = list(chain.from_iterable(new_cluster_pts))
    labels = arr(range(len(clustered_pts)))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    for color,cluster in zip(colors,new_clusters): cluster.paint_uniform_color(color[:3])
    draw(new_clusters)

    # Reclustering the local points
    #  for manual runs, reducing multi tree clusters
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr(clustered_pts))
    labels = np.array( pcd.cluster_dbscan(eps=.25, min_points=20, print_progress=True))   
    max_label = labels.max()
    # visualize the labels
    log.info(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    first = colors[0]
    colors[0] = colors[-1]
    colors[-1]=first
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # draw([pcd])
    breakpoint()



    print('preparing KDtree')
    highc_pts = arr(highc.points)
    # highc_tree = sps.KDTree(highc_pts)
    # with open('cluster126_fact100_iter50.pkl','wb') as f:  pickle.dump(tree_pts,f)
    # with open('entire_50iter_assns.pkl','rb') as f: high_c_pt_assns = pickle.load(f)
    with open('highc_KDTree.pkl','rb') as f:
        highc_tree = pickle.load(f)
    highc_pts = highc_tree.data
    
    minz = min(arr(lowc.points)[:,2])



    unique_vals, counts = np.unique(labels, return_counts=True)
    label_idls= [ np.where(labels ==val)[0] for val in unique_vals]
    clusters = [pcd.select_by_index(idls) for idls in label_idls if len(idls)>200]
    # cluster_ids_in_col = [arr(lowc_ids_from_col)[idls] for idls in label_idls]
    cluster_extents = [cluster.get_max_bound()-cluster.get_min_bound() 
                                for cluster in clusters]
    cluster_pts = [arr(cluster.points) for cluster in clusters]
    # centers = [get_center(arr(x.points)) for x in clusters]


    

    # clustered_ids = list(chain.from_iterable(cluster_ids_in_col))
    # mask = np.ones(col_pts.shape[0], dtype=bool)
    # mask[clustered_ids] = False
    # not_yet_traversed = np.where(mask)[0] # ids of non-clustered
    # not_yet_traversed = [idx for idx,_ in enumerate(highc_pts)]
        
    traversed = []
    tree_pts = [list(arr(x.points)) for x in clusters]
    curr_pts = cluster_pts
    high_c_pt_assns = defaultdict(lambda:-1) 

    ########## Notes for continued Runs #############
    # Run more that 100 iters, there are looong trees to be built
    # 

    #####################################
    iters = 50
    recreate = False
    for i in range(100):
        print('start iter')
        if iters<=0:
            iters =50
            tree_pts = defaultdict(list)
            for k,v in high_c_pt_assns.items(): tree_pts[v].append(k)
            with open(f'cluster126_fact10_0to50.pkl','wb') as f:
                to_write = list([tuple((k,pt_list)) for k,pt_list in tree_pts.items()])
                pickle.dump(to_write,f)

            # for j in [(0,50),(50,100),(100,150),(150,200), (200,281)]:
            #     try:
            #         with open(f'cluster126_fact10_{j[0]}to{j[1]}.pkl','wb') as f:  
            #             to_write = list([tuple((k,pt_list)) for k,pt_list in tree_pts.items()])
            #             pickle.dump(to_write,f)
            #     except Exception as e:
            #         breakpoint()
            #         print(f'error {e}')
            if i %30==0:
                tree_pcds= []
                labels = arr(range(len(tree_pts)))
                max_label = labels.max()
                colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                # for pcd, color in zip(tree_pcds, colors): pcd.paint_uniform_color(color[:3])
                for pts, color in zip(tree_pts.values(), colors):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.paint_uniform_color(color[:3])
                    tree_pcds.append(o3d.geometry.PointCloud())
                draw(tree_pcds)
                test = deepcopy(tree_pcds)
                test.extend(clusters)
                for c in clusters:c.paint_uniform_color([1,0,0])
                draw(test)
                breakpoint()

        iters=iters-1
        for idx, (cluster, cluster_extent) in enumerate(zip(clusters, 
                                                                    cluster_extents)):
            if len(curr_pts[idx])>0:
                print(f'querying {i}')
                dists,nbrs = highc_tree.query(curr_pts[idx],k=750,distance_upper_bound= .25) #max(cluster_extent))
                print(f'reducting nbrs {idx}')

                nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x !=40807492]
                print(f'{len(nbrs)} nbrs found for cluster {idx}')
                # Note: distances are all rather similar -> uniform distribution of collective
                nbr_pts = [nbr_pt for nbr_pt in highc_tree.data[nbrs] if high_c_pt_assns[tuple(nbr_pt)]==-1]
                for nbr_pt in nbr_pts:
                    high_c_pt_assns[tuple(nbr_pt)] = idx

                curr_pts[idx] = nbr_pts # note how these are overwritten each cycle
                # tree_pts[idx].extend(curr_pts[idx])
                # tree_idxs[idx].extend(nbrs)
                traversed.extend(nbrs)
            else:
                print(f'no more new neighbors for cluster {idx}')
                curr_pts[idx] = []
            # if idx == 19:
            #     print('draw nbrs')
            #     tree_pcd = highc.select_by_index(d_nbrs)
            #     draw([tree_pcd, cluster])
            #     breakpoint()
    # breakpoint()
    print('finish!')