
import tensorflow as tf




def get_and_label_neighbors(comp_pcd, base_dir, nbr_label, non_nbr_label = 'wood'):

    # comp_kd_tree = sps.KDTree(arr(comp_pcd.points))

    glob_pattern = f'{base_dir}/inputs/skio_parts/*'
    files = glob(glob_pattern)
    logdir = "/media/penguaman/backupSpace/lidar_sync/pyqsm/tensor_board/get_epi_details"
    #Tensor Board prep
    names_to_labels  = {'unknown':0,f'orig_{nbr_label}':1, f'found_{nbr_label}':2, non_nbr_label:3}
    writer = tf.summary.create_file_writer(logdir)
    comp_points = np.array(comp_pcd.points)
    comp_labels = [names_to_labels[f'orig_{nbr_label}']]*len(comp_pcd.points) 
    combined_summary = {'vertex_positions':np.vstack(comp_points,np.array(pcd.points)), 
                        }

    def update_tb(case_name, summary, pcd_labels, step):
        if step >0: 
            summary['vertex_positions'] = 0
            summary['vertex_scores'] = 0
            summary['vertex_features'] = 0
        summary['vertex_labels'] = np.vstack(comp_labels,pcd_labels)            
        summary.add_3d(case_name, summary, step=step, logdir=logdir)
        step = step + 1
    
    epi_ids = defaultdict(list)
    finished = []
    # finished = ['660000000.pcd','400000000.pcd','800000000.pcd','560000000.pcd']
    files = [file for file in files if not any(file_name in file for file_name in finished)]
    with writer.as_default(): 
        for file in files:
            #  = {'vertex_positions': np.array(comp_pcd.points), 'vertex_labels': np.zeros(len(comp_pcd.points)), 'vertex_scores': np.zeros((len(comp_pcd.points), 3)), 'vertex_features': np.zeros((len(comp_pcd.points), 3))}
            # summary.add_3d('get_epi_details', summary_dict, step=step, logdir=logdir)
            try:
                #identify neighbors
                file_name = file.split('/')[-1].replace('.pcd','')
                case_name = f'{file_name}_get_{nbr_label}_details'
                pcd = read_pcd(file)
                log.info(f'getting neighbors for {file_name}')
                nbrs_pcd, nbrs, chained_nbrs = get_neighbors_kdtree(pcd, comp_pcd, return_pcd=True)
                if nbrs_pcd is None:
                    log.info(f'no nbrs found for {file_name}')
                    epi_ids[file_name] = []
                    continue
                uniques = np.unique(chained_nbrs)
                o3d.io.write_point_cloud(f'{base_dir}/detail/{file_name}_nbrs.pcd', nbrs_pcd)
                non_matched = pcd.select_by_index(uniques, invert=True)
                o3d.io.write_point_cloud(f'{base_dir}/not_epis/{file_name}_non_matched.pcd', non_matched)

                # Add run to Tensor Board (done at the end in case there are no neighbors)
                ## Initial
                step=0
                vertices = np.vstack(comp_points,np.array(pcd.points))
                combined_summary['vertex_positions'] = vertices
                combined_summary['vertex_scores'] = np.zeros_like(vertices)
                combined_summary['vertex_features'] = np.zeros_like(vertices)

                pcd_labels = np.zeros(len(pcd.points))
                pcd_labels = np.vstack(comp_labels,pcd_labels)
                update_tb(case_name, combined_summary, pcd_labels, step)
                log.info(f'{step=}')
            
                # Add labels for epiphytes
                pcd_labels[uniques] = names_to_labels[f'found_{nbr_label}']
                update_tb(case_name, combined_summary, pcd_labels, step)
                log.info(f'{step=}')

                # Update labels for 'wood' (e.g. whatever is left over)
                pcd_labels = np.full_like(pcd_labels, names_to_labels[non_nbr_label])
                pcd_labels[uniques] = names_to_labels[f'found_{nbr_label}']
                update_tb(case_name, combined_summary, pcd_labels, step)
                log.info(f'{step=}')

                # Save epiphyte nbrs for future use 
                epi_ids[file_name] = uniques
            except Exception as e:
                log.info(f'error {e} when getting neighbors in tile {file_name}')

    np.savez_compressed(f'{base_dir}/epis/epis_ids_by_tile.npz', **epi_ids)
    # breakpoint()


def identify_epiphytes_tb(file_content, save_gif=False, out_path = 'data/results/gif/'):
    logdir = "/media/penguaman/backupSpace/lidar_sync/pyqsm/tensor_board/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    assert shift_one is not None # No shifts for this seed; Ensure you pass get_shifts=True to read_pcds_and_feats
    run_name = f'{seed}_id_epi_'

    epi_file_dir = f'/media/penguaman/backupSpace/lidar_sync/pyqsm/epis/'
    epi_file_name = f'seed{seed}_epis.pcd'
    step=0
    try:
        with writer.as_default():
            log.info('Calculating/drawing contraction')
            step+=1
            summary.add_3d(run_name, to_dict_batch([clean_pcd]), step=step, logdir=logdir)
            orig_colors = deepcopy(arr(clean_pcd.colors))
            try:
                highc, lowc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif)
            except Exception as e:
                log.info(f'error drawing shift for {seed}: {e}')
                # breakpoint()
                log.info(f'error drawing shift for {seed}: {e}')
                
            clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
            lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
            highc = clean_pcd.select_by_index(highc_idxs, invert=False)
            step+=1
            summary.add_3d(run_name, to_dict_batch([lowc]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([highc]), step=step, logdir=logdir)
            draw([clean_pcd])
            draw([highc])
            draw([lowc])
            # breakpoint()
            high_shift = shift_one[highc_idxs]
            z_mag = np.array([x[2] for x in high_shift])
            leaves_idxs, leaves, epis = split_on_percentile(highc,z_mag,60, color_on_percentile=True)
            epis_colored  = highc.select_by_index(leaves_idxs, invert=True)
            o3d.io.write_point_cloud(f'{epi_file_dir}/{epi_file_name}', epis_colored)
            # epis_idxs, epis, leaves = split_on_percentile(highc,z_mag,60, comp=lambda x,y:x<y, color_on_percentile=True)
            pcd_no_epi = join_pcds([highc,leaves])[0]
            step+=1
            summary.add_3d(run_name, to_dict_batch([epis]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([epis_colored]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([pcd_no_epi]), step=step, logdir=logdir)
            # breakpoint()
            cvar = np.array([np.linalg.norm(x) for x in shift_one])
            proped_cmag = propogate_shift(pcd,clean_pcd,cvar)
            color_continuous_map(pcd,proped_cmag)
            draw([pcd])
            # breakpoint()
            log.info('Orienting, extracting hues')
            # center_and_rotate(lowc) 
            hue_pcds,no_hue_pcds =segment_hues(lowc,seed,hues=['white','blues','pink'],draw_gif=False, save_gif=save_gif)
            no_hue_pcds = [x for x in no_hue_pcds if x is not None]
            target = no_hue_pcds[len(no_hue_pcds)-1]
            # draw(target)
            # epis_hue_pcds,epis_no_hue_pcds =segment_hues(epis,seed,hues=['white','blues','pink'],draw_gif=False, save_gif=save_gif)
            # epis_no_hue_pcds = [x for x in epis_no_hue_pcds if x is not None]
            # stripped_epis = epis_no_hue_pcds[len(epis_no_hue_pcds)-1]

            step+=1
            # summary.add_3d('epis', to_dict_batch([stripped_epis]), step=step, logdir=logdir)
            summary.add_3d('id_epi_low', to_dict_batch([target]), step=step, logdir=logdir)
            # summary.add_3d('removed', to_dict_batch([hue_pcds[len(hue_pcds)-1]]), step=step, logdir=logdir)
    except Exception as e:
        log.info(f'error getting epiphytes for {seed}: {e}')
    return []