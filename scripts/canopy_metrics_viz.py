
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from pyQSM.viz.color import segment_hues

import tensorflow as tf
from open3d.visualization.tensorboard_plugin import summary

def draw_hues(file_content,**kwargs):
    seed, pcd, _, _ = file_content
    # clean_pcd = get_downsample(pcd=pcd,normalize=False)
    logdir = "src/logs/hues"
    writer = tf.summary.create_file_writer(logdir)
    hue_pcds,no_hue_pcds =segment_hues(pcd,seed,draw_gif=False, save_gif=False)
    # no_hue_pcds = [x for x in no_hue_pcds if x is not None]
    # target = no_hue_pcds[len(no_hue_pcds)-1]
    with writer.as_default(): 
        summary.add_3d(f'hue_pcd', to_dict_batch([pcd]), step=0, logdir=logdir)
        for step in range(0,len(hue_pcds)):
            summary.add_3d(f'hue_pcd', to_dict_batch([hue_pcds[step]]), step=step+1, logdir=logdir)
            summary.add_3d(f'no_hue', to_dict_batch([no_hue_pcds[step]]), step=step+1, logdir=logdir)


def clean_topo(topo):
    lines = arr(topo.lines)
    pts = arr(topo.points)
    len_lines = [np.linalg.norm(pts[l[0]]-pts[l[1]]) for l in lines]
    lines_w_len = [(l,llen) for l,llen in sorted(zip(lines,len_lines),key=lambda x: x[1])]
    long_lines = np.where(np.array(len_lines)>2*np.percentile(len_lines,90))[0]

    new_lines = arr([l for i,l in enumerate(lines) if i not in long_lines])
    new_pts = pts[new_lines[:,0]] + pts[new_lines[:,0]]
    new_pt_ids = {tuple(pt):idx for idx,pt in enumerate(new_pts)}
    pt_ids = {idx:tuple(pt) for idx,pt in enumerate(pts)}
    new_lines = [[new_pt_ids[tuple(pt_ids[i])] for i in l] for l in new_lines]
    # lines = lines[:len(lines)-1]
    breakpoint()
    topo_new = o3d.geometry.LineSet()
    topo_new.lines = o3d.utility.Vector2iVector(lines)
    topo_new.points = o3d.utility.Vector3dVector(pts)
    return topo_new

def evaluate_shifts(file_content, 
                    contraction=config['skeletonize']['init_contraction'],
                    attraction=config['skeletonize']['init_attraction']):
    seed, pcd, clean_pcd, shift = file_content
    file_base = f'skels3/skel_{str(contraction).replace('.','pt')}_{str(attraction).replace('.','pt')}_seed{seed}' #_vox{vox}_ds{ds}'                    
    contracted = contract(clean_pcd,shift)
    draw(contracted)
    topo = load_line_set(file_base)
    draw(topo)
    breakpoint()
    return shift


def read_shift_results(file_content, contraction=1, attraction=1, vox=0, ds=0):
    """
    Reads the results of get_shift from the skels3 directory.
    
    Args:
        file_content: Tuple containing (seed, pcd, clean_pcd, shift_one)
        contraction: Contraction factor used in get_shift
        attraction: Attraction factor used in get_shift
        vox: Voxel size used in get_shift
        ds: Downsample factor used in get_shift
        
    Returns:
        Dictionary containing the loaded results:
        - 'contracted': The contracted point cloud
        - 'total_shift': The total shift applied to the point cloud
        - 'lines': The line set representing the skeleton
        - 'points': The points of the skeleton
    """
    seed, pcd, clean_pcd, shift_one = file_content
    cmag = np.array([np.linalg.norm(x) for x in shift_one])
    highc_idxs = np.where(cmag>np.percentile(cmag,70))[0]
    test = clean_pcd.select_by_index(highc_idxs, invert=True)
    file_base = f'skel_{str(contraction).replace(".","pt")}_{str(attraction).replace(".","pt")}_seed{seed}_vox{vox}_ds{ds}'
    
    shift_path = f'skels3/{file_base}_total_shift.pkl'
    contracted_path = f'data/skio/results/skio/skels3/{file_base}_contracted.pcd'
    lines_path = f'skels3/{file_base}'

    cyl_objects=f'data/skio/results/skio/cyls/{file_base}_cyls.pkl'
    contained_idxs_file=f'data/skio/results/skio/cyls/{file_base}_contained_idxs.pkl'
    topo_res_file=f'data/skio/results/skio/cyls/{file_base}_topo_graph.pkl'

    results = {}
    import os
    
    # Load the contracted point cloud
    try:
        results['total_shift'] = load(shift_path)
        new107 = read_pcd('data/skio/ext_detail/new_107.pcd')
        new108 = read_pcd('data/skio/ext_detail/new_108.pcd')
        results['contracted'] = read_pcd(contracted_path)
        results['topo'] = load_line_set(lines_path)
    except Exception as e:
        log.info(f'Error loading contracted point cloud for skels3/{file_base}: {e}')
        breakpoint()
    
    try:
        with open(cyl_objects,'rb') as f: results['cyl_objects'] = pickle.load(f)
        with open(contained_idxs_file,'rb') as f: results['contained_idxs'] = pickle.load(f)
        with open(topo_res_file,'rb') as f: results['topo_graph'] = pickle.load(f)
    except Exception as e:
        log.info(f'Error loading qsm_data for skels3/{file_base}: {e}')
        breakpoint()

    topo = results['topo']
    total_shift = results['total_shift']        
    contracted = results['contracted']

    clean_c,inds = contracted.remove_statistical_outlier(nb_neighbors=20, std_ratio=.95)  
    new_shift = total_shift[inds]
    new_cmag = cmag[inds]



    orig = contract(contracted,total_shift,invert=True)
    draw(orig)
    draw(contracted)
    draw(topo)
    
    colored_clean_c = color_continuous_map(clean_c,new_cmag)
    draw(colored_clean_c)
    from geometry.skeletonize import skeleton_to_QSM
    topology=extract_topology(clean_c)
    topo=topology[0]
    topology_graph = topology[1]
    draw(topo)
    breakpoint()    
    # topo = clean_topo(topo)
    all_cyl_pcd, cyls, cyl_objects , radii = skeleton_to_QSM(topo,topology_graph,new_shift)
    breakpoint()    
    draw(all_cyl_pcd)
    draw(all_cyl_pcd)

    edges = topology_graph.edges(data=True)
    contained_idxs = [x[2].get('data') for x in edges]
    edge_to_orig = {tuple((x[0],x[1])):x[2].get('data') for x in edges}

    import pickle 
    
    cyl_objects=f'data/skio/results/skio/cyls/{file_base}_cyls.pkl'
    contained_idxs_file=f'data/skio/results/skio/cyls/{file_base}_contained_idxs.pkl'
    topo_res_file=f'data/skio/results/skio/cyls/{file_base}_topo_graph.pkl'
    with open(cyl_objects,'wb') as f: pickle.dump(cyl_objects,f)
    with open(contained_idxs_file,'wb') as f: pickle.dump(contained_idxs,f)
    with open(topo_res_file,'wb') as f: pickle.dump(topology_graph,f)

    breakpoint()
    
    return results


def identify_epiphytes(file_content, save_gif=False, out_path = 'data/results/gif/'):
    logdir = "src/logs/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    step=0
    with writer.as_default():
        seed, pcd, clean_pcd, shift_one = file_content
        log.info('Calculating/drawing contraction')
        step+=1
        summary.add_3d('id_epi_low', to_dict_batch([clean_pcd]), step=step, logdir=logdir)
        summary.add_3d('id_epi_high', to_dict_batch([clean_pcd]), step=step, logdir=logdir)
        
        orig_colors = deepcopy(arr(clean_pcd.colors))
        highc, lowc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif)
        clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
        lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
        highc = clean_pcd.select_by_index(highc_idxs, invert=False)
        step+=1
        summary.add_3d('id_epi_low', to_dict_batch([lowc]), step=step, logdir=logdir)
        summary.add_3d('removed', to_dict_batch([highc]), step=step, logdir=logdir)
        # draw(lowc)
        # draw(highc)

        high_shift = shift_one[highc_idxs]
        z_mag = np.array([x[2] for x in high_shift])
        leaves_idxs, leaves, epis = color_on_percentile(highc,z_mag,60)
        pcd_no_epi = join_pcds([highc,leaves])
        step+=1
        summary.add_3d('removed', to_dict_batch([epis]), step=step, logdir=logdir)
        summary.add_3d('id_epi_low', to_dict_batch([pcd_no_epi]), step=step, logdir=logdir)
        # draw(leaves)
        # draw(epis)
        # breakpoint()
        # log.info('Extrapoloating contraction to original pcd')
        # proped_cmag = propogate_shift(pcd,clean_pcd,shift_one)

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
        summary.add_3d('removed', to_dict_batch([hue_pcds[len(hue_pcds)-1]]), step=step, logdir=logdir)

    return []
    # log.info('creating alpha shapes')

    # metrics = {}
    # # get_mesh(pcd,lowc,target)
    # to_project = [('whole',pcd),('lowc',lowc),('highc',highc),('target',target),('leaves',leaves),('epis',epis)]
    # for name, tp_pcd in to_project:
    #     try:
    #         mesh = project_pcd(tp_pcd,.1,name = name,seed=f'{seed}_{name}_pcd')
    #         metrics[name] = {'pcd_max': tp_pcd.get_max_bound(), 'pcd_min': tp_pcd.get_min_bound(), 'mesh': mesh, 'mesh_area': mesh.area }
    #     except Exception as e:
    #         print(f'error creating {name} mesh for {seed}: {e}')

    # log.info(f'finished seed {seed}')
    # log.info(f'{seed=}, {metrics=}')
    # o3d.visualization.draw_geometries([test], mesh_show_back_face=True)
    ######Ordered small to large leads to more,smaller triangles and increased coverage
    # return metrics 
    # mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
  