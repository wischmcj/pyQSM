    
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