

def get_canopy_metrics_from_skio():

    # breakpoint()
    # # pcds = pcds_from_extend_seed_file('data/skeletor/exts/skel_clean_ext_full.pkl',[1])
    # # breakpoint()
    # # from string import Template
    # # recover_original_details(pcds,file_prefix = Template('/media/penguaman/backupSpace/pyqsm_data/data/skeletor_20250419/inputs/trim/skeletor_full_$idc.pcd',file_base_num=1,file_num_iters=12))
    # # nbr_pcd, _, nbr_idxs = get_neighbors_kdtree(full,query_pcd=pcds[0],k=200, dist=.15)
    # # write_pcd('data/skeletor/exts/skel_branch_0_orig_detail_ds2.pcd',nbr_pcd)

    # read_results = loop_over_files(draw_hues, kwargs=[{'get_shifts':False}],requested_pcds=[nbr_pcd])

    # breakpoint()

    # read_results = loop_over_files(get_pcd_projections, kwargs={},requested_seeds=[107,108])

    # breakpoint()
    # # breakpoint()
    # inputs = [(1,1)]
    # kwargs_in = [dict({'contraction':x,'attraction':y}) for x,y in inputs]
    # # res = loop_over_files(get_shift, kwargs=kwargs_in,skip_seeds=[133,151,113,107,108,189,033]) 

    # read_results = loop_over_files(read_shift_results, kwargs=kwargs_in,skip_seeds=[113,133,151,107,108,189])

    # # Example of using read_shift_results to load the results of get_shift
    # # This will read the results for all seeds with the specified parameters
    # # read_results = loop_over_files(read_shift_results, kwargs=[{'contraction': 8, 'attraction': 2}])
    # # print(f"Loaded results for {len(read_results)} seeds")
    # # for seed, result in enumerate(read_results):
    # #     if result:
    # #         print(f"Seed {seed}: Found {len(result)} components")
    # #         if 'contracted' in result:
    # #             print(f"  - Contracted point cloud with {len(result['contracted'].points)} points")
    # #         if 'lines' in result:
    # #             print(f"  - Line set with {len(result['lines'])} lines")

    # # res = loop_over_files(evaluate_shifts, kwargs=kwargs_in) 

    # # res = loop_over_files(identify_epiphytes, skip_seeds=[113,107,108,189])
    #                     #   requested_seeds=['33', 107,108,109,111,112,113,114,115,133,116]) 
    # breakpoint()
    # lowc  = o3d.io.read_point_cloud('data/skio/inputs/new_low_cloud_all_16-18pct.pcd')
    # draw([lowc]+res)
    # # loop_over_files(identify_epiphytes, skip_seeds = [189,151,113, 33,191,137,134,135])
    # ###Goods
    # #113, 136**
    # ##Needs Work
    # # 135, alpha shape looks really skinny and sparse 
    # #       seems like this tree was far away from the scanner and one side is missing
    # # 191, tighter alpha value, trunk bases are from geometry.reconstruction import get_neighbors_kdtree
    # from geometry.skeletonize import extract_skeleton
    
    # nn108 = cluster_roots_w_order_in_process('data/skio/results/skio/108_fin2_in_process.pkl')
    # # draw(nn108[0])

    # not_in108, _, ni_idxs = get_neighbors_kdtree(nn108_down,nnew107,k=200, dist=1)
    # newnew108 = new108.select_by_index(ni_idxs,invert=True)

    # low, _ = crop_by_percentile(old107,0,10)
    # draw(low)
    
    # test = zoom_pcd([[0,0,0],[1,1,1]],low)
    # clusters = cluster_plus(low, eps=.15, min_points=200,return_pcds=True,from_points=False, draw_result=True)
    
    # ### Gettting skeleton
    # # for seed,pcd in [(108,new108),(107,new107)]:
    # #     voxed_down = pcd.voxel_down_sample(voxel_size=.05)
    # #     uni_down = voxed_down.uniform_down_sample(3)
    # #     clean_pcd = remove_color_pts(uni_down,invert = True)
    # #     file = f'iso_seed{seed}'
    # #     skel_res = extract_skeleton(clean_pcd, max_iter = 3, cmag_save_file=file)
    # #     res.append(skel_res)

    # collective =  read_pcd('data/input/collective.pcd')
    # root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    # breakpoint()
    # seed_to_root_map = {seed_id: root_pcd for seed_id,root_pcd in zip(all_ids,root_pcds)}
    # seed_to_root_id = {seed: idc for idc,seed in enumerate(all_ids)}
    # unmatched_roots = [seed_to_root_id[seed] for seed in seed_to_exts.keys()]
    pass
     
def inspect_others():
    
    # # reading in results from extend
    # rf_ext_pcds = pcds_from_extend_seed_file('new_seeds_w_order_in_progress.pkl')
    oth_ext_pcds = pcds_from_extend_seed_file('other_clusters_w_order_in_process.pkl')
    # root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    # Source seed cluster id to ext in 'other_clusters_w_order_in_process.pkl'
    oth_seed_to_ext = {40: 0, 41: 1, 44: 2, 45: 3, 48: 4, 49: 5, 51: 6, 52: 7, 54: 8, 62: 9, 63: 10, 64: 11, 65: 12, 66: 13, 77: 14, 78: 15, 79: 16, 85: 17, 86: 18, 88: 19, 90: 20, 121: 21, 122: 22, 123: 23, 125: 24, 126: 25, 128: 26, 144: 27, 146: 28, 147: 29, 149: 30, 150: 31, 152: 32, 153: 33, 156: 34, 157: 35, 181: 36, 182: 37}

    for pcd,seed_id_tup in zip(oth_ext_pcds,oth_seed_to_ext.items()):
        log.info(f'{seed_id_tup}')
        seed,idc = seed_id_tup
        draw(pcd)
        breakpoint()
            # draw(oth_ext_pcds[13])
    breakpoint()
    seeds = []
    clusters = []
    

def run_extend():
    """The ids and num hard coded below are for the SKIO datas
    """
    # exclude_regions = {
    #         'building1' : [ (77,350,0),(100, 374,5.7)], 
    #         'building2' : [(0, 350), (70, 374)],
    #         'far_front_yard':[ (0,400),(400, 500)],
    #         'far_rside_brush':[ (140,0),(190, 500)],
    #         'lside_brush':[ (0,0),(77, 50 all_ids = rf_seeds + final_int_ids0)],
    #         'far_back_brush': [ (0,0),(200, 325)]
    # }
    [ (77,350,0),(100, 374,5.7)]
    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    all_ids = rf_seeds + final_int_ids
    all_ids.sort()

    seed_to_exts = {152: 0 # untested 51 root
                    ,107: 1# ext 1 is two trees
                    ,108: 1#   and so correlates to seeds 107 and 108
                    ,109: 2
                    ,111: 3
                    ,112: 4                    
                    #(113: 5)
                    ,114: 6
                    ,115: 7 # this is a good seed but the ext only goes up the trunk a couple feet
                    ,116: 8 # this is a good ext, but the seed is spanish moss
                    ,133: 9  # This is a group of 3 trees
                    ,134: 10 
                    ,135: 11
                    ,136: 12 # This seed only goes halfway around the trunk
                    ,137: 13
                    ,138: 14 # seed is vine, cluster contains a branch of some tree
                    ,190: 15
                    ,191: 16 # ***needs to be rerun, is cutoff by the edge of the selected area
                    #(193, 17), # is a mismatch
                    ,194: 18  # ***needs to be rerun, is cutoff by the edge of the selected area
    } 
    exts_to_seed = {v:k for k,v in seed_to_exts.items()}
    unmatched_clusters = [151,110,113,180,189,193,132,148]
    unmatched_exts = [17] # 17 is the bottom of some spanish moss

    #to_pass_back through
    pass_again = [107,108]

    # # #       and the clusters fed into extend seed clusters
    lowc  = read_pcd('new_low_cloud_all_16-18pct.pcd')
    # highc = read_pcd('new_collective_highc_18plus.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl',dir='')
    empty_pcd= o3d.geometry.PointCloud() 
    # # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
    c_trunk_clusters = create_one_or_many_pcds([arr(lc.points) for lc in trunk_clusters],labels=[k for k,v in label_to_clusters.items()])

    
    # rf_colored = arr(c_trunk_clusters)[rf_seeds]  
    # other_colored = arr(c_trunk_clusters)[~arr(refined_clusters)]
    
    other_clusters = [(idc,cluster) if idc in final_int_ids else (idc,empty_pcd) for idc,cluster in enumerate(c_trunk_clusters) ]
    # extend_seed_clusters(clusters_and_idxs = other_clusters,
    #                         src_pcd=highc,
    #                         file_label='other_clusters',cycles=250,save_every=20,draw_every=160, max_distance=.1)

if __name__ =="__main__":
    from geometry.reconstruction import get_neighbors_kdtree
    from geometry.skeletonize import extract_skeleton
    
    # in_ids = [1,2,3,4,5,6,7,8]
    # rf_ext_pcds = pcds_from_extend_seed_file('data/skio/exts/new_seeds_w_order_in_progress.pkl',in_ids)
    # # in_seed_ids = [107,108,109,111,112,113,114,115,116,133,151]
    # oth_ext_pcds = pcds_from_extend_seed_file('data/skio/exts/other_clusters_w_order_in_process.pkl',[12])
    # pcd107 = pcds_from_extend_seed_file('data/skio/exts/rf_cluster_107_rebuild_in_process.pkl')[0]
    # pcd108 = pcds_from_extend_seed_file('data/skio/results/skio/108_split_in_process.pkl')[0]
 
    # collective  = read_pcd('data/skio/inputs/partitioning_search_area_collective.pcd')
    # low ,_ = crop_by_percentile(collective,5,20)

    # ext108  = read_pcd('data/skio/ext_detail/full_ext_seed108_rf1_orig_detail.pcd')

    new107 = read_pcd('data/skio/ext_detail/full_ext_seed107_rf1_orig_detail.pcd')
    # new108 = read_pcd('data/skio/ext_detail/full_ext_seed108_rf1_orig_detail.pcd')
    # old108 = read_pcd('data/skio/ext_detail/full_ext_seed108_rf1_orig_detail_old.pcd')
    nn108 = cluster_roots_w_order_in_process('data/skio/results/skio/108_fin2_in_process.pkl')
    draw(nn108[0])
    breakpoint()
    not_in108, _, ni_idxs = get_neighbors_kdtree(nn108_down,nnew107,k=200, dist=1)
    newnew108 = new108.select_by_index(ni_idxs,invert=True)

    old107 = read_pcd('data/skio/ext_detail/old_pcds/full_ext_seed107_rf1_orig_detail_old.pcd')
    draw(old107)
    low, _ = crop_by_percentile(old107,0,10)
    draw(low)
    print(low.get_max_bound(),low.get_min_bound())
    breakpoint()
    test = zoom_pcd([[0,0,0],[1,1,1]],low)
    clusters = cluster_plus(low, eps=.15, min_points=200,return_pcds=True,from_points=False, draw_result=True)
    

    # with open('data/skio/pepi_shift/seed108_rf1_voxpt05_uni3_shift.pkl','rb') as f: old_shift108 = pickle.load(f)
    # with open('data/skio/pepi_shift/seed107_rf1_voxpt05_uni3_shift.pkl','rb') as f: old_shift107 = pickle.load(f)
    with open('data/skio/results/skio/new_seed108_rf_voxpt05_uni3_shift_shift.pkl','rb') as f: shift108 = pickle.load(f)
    res = []
    for seed,pcd in [(108,new108),(107,new107)]:
        voxed_down = pcd.voxel_down_sample(voxel_size=.05)
        uni_down = voxed_down.uniform_down_sample(3)
        clean_pcd = remove_color_pts(uni_down,invert = True)
        file = f'iso_seed{seed}'
        skel_res = extract_skeleton(clean_pcd, max_iter = 3, cmag_save_file=file)
        res.append(skel_res)
        
    lowc  = read_pcd('data/skio/inputs/new_low_cloud_all_16-18pct.pcd')

    # axes = o3d.geometry.create_mesh_coordinate_frame()
    # exclude_regions = {
    #         'building1' : [ (77,350,0),(100, 374,5.7)], 
    #         'building2' : [(0, 350), (70, 374)],
    #         'far_front_yard':[ (0,400),(400, 500)],
    #         'far_rside_brush':[ (140,0),(190, 500)],
    #         'lside_brush':[ (0,0),(77, 50 all_ids = rf_seeds + final_int_ids0)],
    #         'far_back_brush': [ (0,0),(200, 325)]
    # }
    rf_seeds = [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
    final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    all_ids = rf_seeds + final_int_ids
    all_ids.sort()

    seed_to_exts = { 
                    #### 17 and 19 are inconcequentially small,
                    ####  seed 110 is super small
                    ##  148,180  doesnt match to any rf
                    152: 0  # untested 51 root
                    ,151: 9  # untested 51 root
                    ,107: 1  # ext 1 is two trees
                    ,108: 1  #   and so correlates to seeds 107 and 108
                    ,109: 2
                    ,111: 3
                    ,112: 4  
                    ,113: 5 
                    ,114: 6
                    ,115: 7  # this is a good seed but the ext only goes up the trunk a couple feet
                    ,116: 8  # this is a good ext, but the seed is spanish moss
                    ,133: 9  # This is a group of 3 trees
                    ,134: 10 
                    ,135: 11
                    ,136: 12 # This seed only goes halfway around the trunk
                    ,137: 13
                    ,138: 14 # seed is vine, cluster contains a branch of some tree
                    ,190: 15
                    ,191: 16 # ***needs to be rerun, is cutoff by the edge of the selected area
                    #(193, 17), # is a mismatch
                    ,194: 18  # ***needs to be rerun, is cutoff by the edge of the selected area
    }
    unmatched_seeds = [seed for seed in rf_seeds if not seed_to_exts.get(seed)]
    #[151, 110, 113, 132, 148, 180, 189, 193]

    matched_rf_ext = [x for x in seed_to_exts.values()]#[0 ,1 ,1 ,2,3,4 ,6,7 ,8 ,9 ,10,11,12,13,14,15,16,18]

    collective =  read_pcd('data/input/collective.pcd')
    root_pcds = pcds_from_extend_seed_file('cluster_roots_w_order_in_process.pkl')
    breakpoint()
    seed_to_root_map = {seed_id: root_pcd for seed_id,root_pcd in zip(all_ids,root_pcds)}
    seed_to_root_id = {seed: idc for idc,seed in enumerate(all_ids)}
    unmatched_roots = [seed_to_root_id[seed] for seed in seed_to_exts.keys()]


    # # load all seed clusters 
    lowc  = read_pcd('new_low_cloud_all_16-18pct.pcd')
    label_to_clusters = load('new_skio_labels_low_16-18_cluster_pt5-20.pkl')
    trunk_clusters = [lowc.select_by_index(x) for x in label_to_clusters.values()]
    # # highc = read_pcd('new_collective_highc_18plus.pcd')
    # empty_pcd= o3d.geometry.PointCloud() 
    # # # # label_to_clusters = load('new_lowc_lbl_to_clusters_pt3_20.pkl')

    all_seeds = {seed: 
                    (trunk_clusters[seed] ,
                      seed_to_root_map[seed], seed_to_root_id[seed]) for seed in rf_seeds}

    unmatched_rf_ext_idx = [idx for idx,pcd in enumerate(rf_ext_pcds) if idx not in [0 ,1 ,1 ,2,3,4 ,6,7 ,8 ,9 ,10,11,12,13,14,15,16,18] ]
    unmatched_rf_ext_pcds = [pcd for idx,pcd in enumerate(rf_ext_pcds) if idx not in [0 ,1 ,1 ,2,3,4 ,6,7 ,8 ,9 ,10,11,12,13,14,15,16,18] ]

    breakpoint()
