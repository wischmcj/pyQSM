    
 ########### Notes ##############
    ### TO-DO: 
    ###   - extend cluster 189 within full collective set
    ###
    ### - new_low_cloud_all_16-18pct.pcd - pcd for original cluster set
    ### - new_collective_highc_18plus.pcd - pcds for original search set 
    ### - completed_cluster_idxs
    ###     - ids of the root clusters used to generate completed trees
    ### - collective 
    ###     - is the full skio scan 
    ##  - new_skio_labels_low_16-18_cluster_pt5-20 ?
    ### - new_low_cloud_all_16-18pct.pcd
    ###    - Source pcd for the trunk seed clusters in new_lowc_lbl_to_clusters_pt3_20
    ### -  new_lowc_lbl_to_clusters_pt3_20.pkl
    ###     - Seed clusters fom ...new_low_cloud_all_16-18pct used to generate trees
    ### - new_lowc_refined_clusters
    ###     - a subset of the above clusters that represent our trees of interest
    ### - new_seeds_w_order_in_progress.pkl
    ###     - [(labelID,[(pt_order, pt),...]),... ] 
    ###     - mappings for points to labels for clusters built off of seed clusters in new_skio_labels_low_16-18_cluster_pt5-20
    ### - rf_cluster{i}_point_shift
    ###     - a list of shifts for each point in the corr. cluster after 1 round of lapacian transform
    ###     - orig = o3d.io.read_point_cloud(f'rf_cluster{i}_orig_detail.pcd')
    ###     - test = orig.voxel_down_sample(voxel_size=.05)
    ### 
    ### 
    ### 
    ###
    # ################################
 ## REbuild/root match notes 
    #[need shift rerun]
    #[133,151,113,132] [191,193,194, 189]
    # First get details for :
    # 189 matches with rf_ext_pcds[16], has 3 roots ids 50,51,58
    # 193 matches with rf_ext_pcds[18]

# seed='189',pcd_size=43307685,pcd_file='data/results/full_skio_iso/full_ext_seed189_rf16_orig_detail.pcd'
# seed='151',pcd_size=22178596,pcd_file='data/results/full_skio_iso/full_ext_seed151_rf9_orig_detail.pcd'
# seed='113',pcd_size=10239951,pcd_file='data/results/full_skio_iso/full_ext_seed113_rf5_orig_detail.pcd'
# seed='133',pcd_size=22178596,pcd_file='data/results/full_skio_iso/full_ext_seed133_rf9_orig_detail.pcd'
# seed='191',pcd_size=43307685,pcd_file='data/results/full_skio_iso/full_ext_seed191_rf16_orig_detail.pcd'
# seed='137',pcd_size=22515019,pcd_file='data/results/full_skio_iso/full_ext_seed137_rf13_orig_detail.pcd'
# seed='134',pcd_size=1455432,pcd_file='data/results/full_skio_iso/full_ext_seed134_rf10_orig_detail.pcd'
# seed='109',pcd_size=5265854,pcd_file='data/results/full_skio_iso/full_ext_seed109_rf2_orig_detail.pcd'
# seed='136',pcd_size=9842424,pcd_file='data/results/full_skio_iso/full_ext_seed136_rf12_orig_detail.pcd'
# seed='135',pcd_size=4267104,pcd_file='data/results/full_skio_iso/full_ext_seed135_rf11_orig_detail.pcd'
# seed='111',pcd_size=4492182,pcd_file='data/results/full_skio_iso/full_ext_seed111_rf3_orig_detail.pcd'
# seed='138',pcd_size=2059828,pcd_file='data/results/full_skio_iso/full_ext_seed138_rf14_orig_detail.pcd'
# seed='112',pcd_size=3204362,pcd_file='data/results/full_skio_iso/full_ext_seed112_rf4_orig_detail.pcd'
# seed='116',pcd_size=12840014,pcd_file='data/results/full_skio_iso/full_ext_seed116_rf8_orig_detail.pcd'
# seed='107',pcd_size=19161968,pcd_file='data/results/full_skio_iso/full_ext_seed107_rf1_orig_detail_old.pcd'
# seed='193',pcd_size=5767768,pcd_file='data/results/full_skio_iso/full_ext_seed193_194_rf18_orig_detail.pcd'
# seed='114',pcd_size=345768,pcd_file='data/results/full_skio_iso/full_ext_seed114_rf6_orig_detail.pcd'
# seed='190',pcd_size=25388369,pcd_file='data/results/full_skio_iso/full_ext_seed190_rf15_orig_detail.pcd'
# seed='108',pcd_size=19132331,pcd_file='data/results/full_skio_iso/full_ext_seed108_rf1_orig_detail.pcd'
# seed='132',pcd_size=22178596,pcd_file='data/results/full_skio_iso/full_ext_seed132_rf9_orig_detail.pcd'
# seed='115',pcd_size=3828494,pcd_file='data/results/full_skio_iso/full_ext_seed115_rf7_orig_detail.pcd'


    label_idls = load('new_lowc_ref_clusters.pkl')
    # label_to_clusters = [labels for idc, labels in enumerate(label_to_clusters) if idc not in [3,10,17,18,19]]
    label_idls= label_to_clusters
    clusters = [(idc,lowc.select_by_index(i     dls)) for idc, idls in enumerate(label_idls)]
    # draw(arr(clusters)[:,1])
    refined_clusters_idls = [151,107,108,109,110,111,112,113,114,
                                #115,
                                116,132,133,134,135,136,
                                #137,138,148,
                                180,189,190,191,193,194]
    refined_clusters_idls = [151,107,108,109,110,111,112,113,114,116,132,133,134,135,136,180,189,190,191,193,194]
    # refined_clusters_labels = [label_to_clusters[idc] for idc in refined_clusters_idls]
    # refined_clusters= [clusters[idc] for idc in refined_clusters_idls]
    empty_pcd = o3d.geometry.PointCloud()
    other_clusters= [empty_pcd if idc in [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
                              else cluster for idc,cluster in clusters ]

    # # trunk clusters are the clusters originally fed into 'extend_seed_clusters'
    # # rf_cluster{num}_orig_detail are the result of 'extend_seed_clusters', w/ details from orig scan read in

    ### Other Seed clusters (other than refined)
    ## Looped through seeds 10 at a time, marked those that looked of interest for extension
    ### Looped through the interesting ranges  then 5 at a time to future limit data set
    ## Came up with the followng ranges interesting_intervals =  [(40, 45), (45, 50), (50, 55), (60, 65), (65, 70), (75, 80), (85, 90), (125, 130), (120, 130), (140, 150), (150, 160), (180, 190)]
    ## these imply the folowing seed ids final_int_ids = [40, 41, 44, 45, 48, 49, 51, 52, 54, 62, 63, 64, 65, 66, 77, 78, 79, 85, 86, 88, 90, 121, 122, 123, 125, 126, 128, 144, 146, 147, 149, 150, 152, 153, 156, 157, 181, 182]
    ## these were then run through ext to get the pcds defined by other_clusters_w_order_in_process.pkl


    # Source seed cluster id to ext in 'other_clusters_w_order_in_process.pkl'
    oth_seed_to_ext = {40: 0, 41: 1, 44: 2, 45: 3, 48: 4, 49: 5, 51: 6, 52: 7, 54: 8, 62: 9, 63: 10,
      64: 11, 65: 12, 66: 13, 77: 14, 78: 15, 79: 16, 85: 17, 86: 18, 88: 19, 90: 20,
        121: 21, 122: 22, 123: 23, 125: 24, 126: 25, 128: 26, 144: 27, 146: 28, 147: 29,
          149: 30, 150: 31, 152: 32, 153: 33, 156: 34, 157: 35, 181: 36, 182: 37}
    seed_to_exts = {107: 1# ext 1 is two trees
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
    # 0 is parts of many trees
    # 1 is two trees
    # 2 is v horizontal, great detail on epiphite clumps
    # 3 has sfc. 
    # 6 is only a trunk
    # 7s seed seems to have been a vine 
    # 8 straight, tall
    # 9 3 trees, good detail
    # 10 extraordinary detail
    # 11 large but incomplete, possibly cluster splitting
    # 13 great trunk detail, multi stem
    # 14 not a trunk151, 0
    # 15 is three large chunks of several trees
    # 17 is nothing at all really
    seed_to_cluster=[(107, 1), (108, 1), 
                    (109, 2), 
                    #(110, ), 
                    (111, 3), (112, 4),(113, 5), 
                      (114, 6), 
                      #(116, 7), appears to be no rf 7
                      # cluster 8 seems to not correlate to any listed seed
                      (132, 7), (133, 9), (134, 10), 
                    (135, 11), (136, 12), 
                    #(180, 13), 13 seems to no correlated with any cluster
                    # 14,15, 17 has not orig detail
                    #(189, 14), (190, 15), (193, 17),
                    (191, 16),
                (194, 18)]
    unmatched = [(110, ), (116, 7), (180, 13),(189, 14), (190, 15), (193, 17),]
# Manual eval of refined clusters 
    ### good, 1 --> good but with extra branches from others
    ### good, 2 --> good but with multiple full trees
    # seed_id, eval, is_multi
    # 0 ,'partial',1
    # 1 ,'good',2
    # 2 ,'good',0
    # 3 ,'good',0
    # 4 ,'good',0
    # 5 ,'good',1
    # 6 ,'partial',0
    # 7 ,'partial',0
    # 8 ,'good',1
    # 9 ,'good',1
    # 10,'good',2
    # 11,'partial',0
    # 12,'good',0
    # 13,'good',1
    # 14,'bad',0
    # 15,'partial',0
    # 16,'good',1
    # 17,'bad',0
    # 18,'partial',0
    # 19,'bad',0
# Manual eval others of interest 
# seed_id, eval, is_multi
    # 40, 'bad',0
    # 41, 'good',1
    # 44, 'good',0
    # 45, 'bad',0
    # 48, 'bad',0
    # 49, 'partial',1
    # 51, 'bad',1
    # 52, 'good', 0
    # 54, 'bad',0
    # 62, 'partial',0
    # 63 , 'good',0
    # 64 , 'good',0
    # 65, 'good',1
    # 66, 'bad',1
    # 77, 'bad',0
    # 78, 'bad',1
    # 79, 'good',1
    # 85, 'bad',0
    # 86, 'bad',0
    # 88, 'partial',0
    # 90, 'partial',1
    # 121, 'bad',0
    # 122, 'bad',1
    # 123, 'good',1
    # 125, 'bad'.0
    # 126, 'bad'.0
    # 128, 'partial',1
    # 144, 'bad',0
    # 146, 'bad',0
    # 147, 'good',1
    # 149, 'partial',0
    # 150, 'partial',0
    # 152, 'partial',1
    # 153, 'partial',0
    # 156, 'partial',1
    # 157, 'bad',0
    # 181, 'partial',0 
    # 182, 'good',1
#seed clusters 
# k_cluster_107 107
# k_cluster_108 108
# k_cluster_109 109
# k_cluster_110 110
# k_cluster_111 111
# k_cluster_112 112
# k_cluster_113 113
# k_cluster_114 114
# k_cluster_115 
# k_cluster_116 116
# k_cluster_132 132
# k_cluster_133 133
# k_cluster_134 134
# k_cluster_135 135
# k_cluster_136 136
# k_cluster_137 
# k_cluster_138 
# k_cluster_148 
# k_cluster_151
# k_cluster_180  
# k_cluster_189 189
# k_cluster_190 190
# k_cluster_191 191
# k_cluster_193 193
# k_cluster_194 194



    
    label_idls = load('new_lowc_ref_clusters.pkl')
    # label_to_clusters = [labels for idc, labels in enumerate(label_to_clusters) if idc not in [3,10,17,18,19]]
    label_idls= label_to_clusters
    clusters = [(idc,lowc.select_by_index(i     dls)) for idc, idls in enumerate(label_idls)]
    # draw(arr(clusters)[:,1])
    refined_clusters_idls = [151,107,108,109,110,111,112,113,114,
                                #115,
                                116,132,133,134,135,136,
                                #137,138,148,
                                180,189,190,191,193,194]
    refined_clusters_idls = [151,107,108,109,110,111,112,113,114,116,132,133,134,135,136,180,189,190,191,193,194]
    # refined_clusters_labels = [label_to_clusters[idc] for idc in refined_clusters_idls]
    # refined_clusters= [clusters[idc] for idc in refined_clusters_idls]
    empty_pcd = o3d.geometry.PointCloud()
    other_clusters= [empty_pcd if idc in [151,107,108,109,110,111,112,113,114,115,116,132,133,134,135,136,137,138,148,180,189,190,191,193,194] 
                              else cluster for idc,cluster in clusters ]

 ########### Notes ##############
    ###
    ### - completed_cluster_idxs
    ###     - ids of the root clusters used to generate completed trees
    ### - collective 
    ###     - is the full skio scan 
    ### - new_skio_labels_low_16-18_cluster_pt5-20
    ###    - Source pcd for the trunk seed clusters in new_lowc_lbl_to_clusters_pt3_20
    ### -  new_lowc_lbl_to_clusters_pt3_20.pkl
    ###     - Seed clusters fom ...low_16-18_cluster_pt5-20 used to generate trees
    ### - new_lowc_refined_clusters
    ###     - a subset of the above clusters that represent our trees of interest
    ### - new_seeds_w_order_in_progress.pkl
    ###     - [(labelID,[(pt_order, pt),...]),... ] 
    ###     - mappings for points to labels for clusters built off of seed clusters in new_skio_labels_low_16-18_cluster_pt5-20
    ### - rf_cluster{i}_point_shift
    ###     - a list of shifts for each point in the corr. cluster after 1 round of lapacian transform
    ###     - orig = o3d.io.read_point_cloud(f'rf_cluster{i}_orig_detail.pcd')
    ###     - test = orig.voxel_down_sample(voxel_size=.05)
    ### 
    ### 
    ### 
    ###
    # ################################

    # 0 is parts of many trees
    # 1 is two trees
    # 2 is v horizontal, great detail on epiphite clumps
    # 3 has sfc. 
    # 6 is only a trunk
    # 7s seed seems to have been a vine 
    # 8 straight, tall
    # 9 3 trees, good detail
    # 10 extraordinary detail
    # 11 large but incomplete, possibly cluster splitting
    # 13 great trunk detail, multi stem
    # 14 not a trunk151, 0
    # 15 is three large chunks of several trees
    # 17 is nothing at all really
    seed_to_cluster=[(107, 1), (108, 1), 
                    (109, 2), 
                    #(110, ), 
                    (111, 3), (112, 4),(113, 5), 
                      (114, 6), 
                      #(116, 7), appears to be no rf 7
                      # cluster 8 seems to not correlate to any listed seed
                      (132, 7), (133, 9), (134, 10), 
                    (135, 11), (136, 12), 
                    #(180, 13), 13 seems to no correlated with any cluster
                    # 14,15, 17 has not orig detail
                    #(189, 14), (190, 15), (193, 17),
                    (191, 16),
                (194, 18)]
    unmatched = [(110, ), (116, 7), (180, 13),(189, 14), (190, 15), (193, 17),]
#seed clusters 
# k_cluster_107 107
# k_cluster_108 108
# k_cluster_109 109
# k_cluster_110 110
# k_cluster_111 111
# k_cluster_112 112
# k_cluster_113 113
# k_cluster_114 114
# k_cluster_115 
# k_cluster_116 116
# k_cluster_132 132
# k_cluster_133 133
# k_cluster_134 134
# k_cluster_135 135
# k_cluster_136 136
# k_cluster_137 
# k_cluster_138 
# k_cluster_148 
# k_cluster_151
# k_cluster_180  
# k_cluster_189 189
# k_cluster_190 190
# k_cluster_191 191
# k_cluster_193 193
# k_cluster_194 194