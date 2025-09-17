
########### Notes ##############
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

new_seeds_w_order_in_progress
    - the extensions of the prime candidate seeds (rf or refined seeds)
other_clusters_w_order_in_process
 - the extensions of the non-prime candidate seeds 