
            ## The below was/is used to combine extended seed files (root files)
            ## with their existing extensions, and to rebuild partial exts as needed
            ## It proceeds as follows:
            ##   0. User sets seed and rf_ext to run. Multiple provided will combine them into a single ext
            ##   1. Manual input determines if extended seed needs to be rebuild 
            ##      - i.e. if it is missing points
            ##   2a. If so, the root is extened against the full set of points
            ##   2b. If not, then the existing orig_detail file is read in 
            ##   3. Original detail is then recovered for either:
            #       - The new_ext (if step 2a was taken)
            #       - Or just the existing root_pcd (if step 2b was taken)
            #    4. All original details files are read in and joined to from the final detailed ext

            #input
            rf_ext_id = [rf_ext_id]
            seeds = [seed]
            #setup
            roots = [(my_seed,seed_to_root_map[my_seed]) for my_seed in seeds]
            root_pcd_ids = [seed_to_root_id[my_seed] for my_seed in seeds]
            roots_pcds = [seed_to_root_map[my_seed] for my_seed in seeds]
            seed_pcd = trunk_clusters[seed]
            rf_pcd = rf_ext_pcds[rf_ext_id]

            breakpoint()
            ## If partial ext, rebuild from larger source pcd
            src_type = ''
            if partial: # if the pcd is cutoff at a border
                pcd = read_pcd('collective.pcd')
                highc = crop_by_percentile(pcd,20,100)[0]
                rf_new = extend_seed_clusters(voxed_down,highc,f'rf_cluster_root{seed}_rebuild',cycles=50,save_every=10,draw_every=10,max_distance=.05,k=300)
                # new_new_rf = extend_seed_clusters([(seed,new_rf)],low,f'rf_cluster_{seed}_plus_down',cycles=12,save_every=3,draw_every=3,max_distance=.2,k=50) 
                draw(rf_new)
                get_details_pcds,src_type = rf_new,'rf_new'
            else: # if the seed and cluster look good
                rf_pcd = pcds_from_extend_seed_file(f'rf_cluster_root{seed}_rebuild_in_process.pkl')
                get_details_pcds,src_type = roots_pcds,'roots_pcds'

            roots_details = []
            for root,root_id in zip(get_details_pcds,root_pcd_ids):
                try:
                    recover_original_details(rf_new[0],save_file=f'rf_cluster_seed{seed}_rf{11}',chunk_size=10000000)
                except Exception as e:
                    log.info(f'This fails but writes the file I need: {e}')
                roots_details.append(read_pcd(f'rf_cluster_seed{seed}_root{root_id}_orig_detail'))
            draw(roots_details)

            if src_type == 'roots_pcds':
                finals = join_pcds([roots_details,rf_pcd])[0]
            elif src_type == 'rf_new':
                finals = roots_details
            draw(finals)

            write_point_cloud(f'data/results/skio/full_ext_seed{'_'.join(seeds)}_rf{11}_orig_detail.pcd',finals)
        