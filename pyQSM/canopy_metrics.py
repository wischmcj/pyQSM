from copy import deepcopy
from glob import glob
import re

from tree_isolation import extend_seed_clusters, pcds_from_extend_seed_file
from utils.io import save
            
import open3d as o3d
import numpy as np
from numpy import asarray as arr
import pyvista as pv
import pc_skeletor as pcs

import tensorflow as tf
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

from set_config import config, log
from geometry.general import center_and_rotate
from geometry.reconstruction import get_neighbors_kdtree
from geometry.skeletonize import extract_skeleton, extract_topology
from geometry.point_cloud_processing import (
    clean_cloud,
    join_pcds,
    join_pcd_files
)
from utils.lib_integration import get_pairs
from utils.io import load, load_line_set,save_line_set, create_table
from viz.ray_casting import project_pcd
from viz.viz_utils import color_continuous_map, draw, rotating_compare_gif
from viz.color import (
    remove_color_pts, 
    get_green_surfaces,
    split_on_percentile,
    segment_hues,
    saturate_colors
)


def list_if(x):
    if isinstance(x,list):
        return x
    else:
        return [x]

color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,
               'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,
               'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,
               'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,
               'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,
               'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,
               'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])


def propogate_shift(pcd,clean_pcd,shift):
    """
        extrapolating point shift to the more detailed pcd
    """
    # voxed_down = pcd.voxel_down_sample(voxel_size=.01)
    c_mag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs = np.where(c_mag>np.percentile(c_mag,40))[0]
    highc = clean_pcd.select_by_index(highc_idxs)
    c_mag_high = c_mag[highc_idxs]
    _, nbrs = get_neighbors_kdtree(highc,pcd, k=50)
    nbrs = [[x for x in nbr_list if x<len(c_mag_high)] for nbr_list in nbrs]
    proped_c_mag = np.zeros(len(nbrs))
    for idnl, nbr_list in enumerate(nbrs): 
        if len(nbr_list)>0: 
            proped_c_mag[idnl] = np.mean(arr(c_mag_high[nbr_list]))
    # proped_c_mag = [np.mean(arr(c_mag_high[nbr_list])) for nbr_list in nbrs]
    # color_continuous_map(pcd,arr(proped_c_mag))
    # draw(pcd)
    np.mean(arr(proped_c_mag)[np.where(arr(proped_c_mag)>0)[0]])
    highc_proped_idxs = np.where(arr(proped_c_mag)>0)[0]
    test = pcd.select_by_index(highc_proped_idxs, invert = True)
    draw(test)
    return proped_c_mag
    
def draw_shift(pcd,
                seed,
                shift,
                down_sample=False,draw_results=False, save_gif=False, out_path = None,
                on_frames=25, off_frames=25, addnl_frame_duration=.01, point_size=5):
    out_path = out_path or f'data/results/gif/'
    clean_pcd = pcd
    if down_sample: clean_pcd = get_downsample(pcd=clean_pcd, normalize=False)
    c_mag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs, highc,lowc = split_on_percentile(clean_pcd,c_mag,70, color_on_percentile=True)
    # center_and_rotate(lowc)
    # center_and_rotate(highc)

    log.info('preping contraction/coloring')
    if draw_results:
        log.info('giffing')
        lowc.rotate(rot_90_x,center =  clean_pcd.get_center())
        lowc.rotate(rot_90_x,center =  clean_pcd.get_center())
        lowc.rotate(rot_90_x,center =  clean_pcd.get_center())
        highc.rotate(rot_90_x,center = clean_pcd.get_center())
        highc.rotate(rot_90_x,center = clean_pcd.get_center())
        highc.rotate(rot_90_x,center = clean_pcd.get_center())
        gif_kwargs = {'on_frames': on_frames, 'off_frames': off_frames, 
                        'addnl_frame_duration':addnl_frame_duration, 'point_size':point_size,
                        'save':save_gif, 'out_path':out_path, 'rot_center':clean_pcd.get_center(),
                         'sub_dir':f'{seed}_draw_shift' }
        rotating_compare_gif(highc,constant_pcd_in=lowc,**gif_kwargs)
    return highc,lowc,highc_idxs

    # # nowhite_custom =remove_color_pts(pcd, lambda x: sum(x)>2.3,invert=True)
    # draw(highc_detail

def get_pcd_projections(file_content=None, pcd=None, seed='', save_gif=False, out_path = 'data/results/gif/'):
    if file_content:
        seed, pcd, clean_pcd, shift_one = file_content
    down = pcd.uniform_down_sample(2)
    # draw(down)

    if shift_one is not None :
        if clean_pcd is None:
            clean_pcd = get_downsample(pcd=clean_pcd, normalize=False)
        c_mag = np.array([np.linalg.norm(x) for x in shift_one])
        highc_idxs, highc,lowc = split_on_percentile(clean_pcd,c_mag,70, color_on_percentile=True)
        draw(lowc)

    make_mesh=True
    metrics = {}
    if make_mesh:
        print(f'Creating projection for {seed}')
        mesh = project_pcd(down,.1,plot=True, seed=seed)
        geo = mesh.extract_geometry()
        metrics[seed] ={'pcd_max': pcd.get_max_bound(),
                    'pcd_min': pcd.get_min_bound(),
                    'mesh': mesh,
                    'mesh_area': mesh.area
                    }
        print(metrics)
    return metrics

def get_pepi_shift(file_content, iters=20):
    seed, pcd, clean_pcd, shift_one = file_content
    file = f'new_seed{seed}_rf_voxpt05_uni3_shift'
    skel_res = extract_skeleton(clean_pcd, max_iter = iters, cmag_save_file=file)
    pass

def get_shift(file_content,
              initial_shift = True, contraction=6, attraction=2, iters=20, 
              debug=False, vox=None, ds=None, use_scs = True):
    """
        Orig. run with contraction_factor 3, attraction .6, max contraction 2080 max attraction 
        Determines what files (e.g. information) is missing for the case passed and 
            calculates what is needed 
    """
    seed, pcd, clean_pcd, shift_one = file_content
    trunk = None
    pcd_branch = None
    file_base = f'skels3/skel_{str(contraction).replace('.','pt')}_{str(attraction).replace('.','pt')}_seed{seed}_vox{vox or 0}_ds{ds or 0}'
    log.info(f'getting shift for {seed}')
    if initial_shift:
        cmag = np.array([np.linalg.norm(x) for x in shift_one])
        highc_idxs = np.where(cmag>np.percentile(cmag,70))[0]
        test = clean_pcd.select_by_index(highc_idxs, invert=True)
    else:
        test = clean_pcd
    if vox: test = test.voxel_down_sample(voxel_size=vox)
    if ds: test = test.uniform_down_sample(ds)
    if not use_scs:
        skel_res = extract_skeleton(test, max_iter = iters, debug=debug, cmag_save_file=file_base, contraction_factor=contraction, attraction_factor=attraction)
    else:
        try:
            # lbc = pcs.LBC(point_cloud=test, filter_nb_neighbors = config['skeletonize']['n_neighbors'], max_iteration_steps= config['skeletonize']['max_iter'], debug = False, termination_ratio=config['skeletonize']['termination_ratio'], step_wise_contraction_amplification = config['skeletonize']['init_contraction'], max_contraction = config['skeletonize']['max_contraction'], max_attraction = config['skeletonize']['max_attraction'])
            lbc = pcs.LBC(point_cloud=test,
                     filter_nb_neighbors = config['skeletonize']['n_neighbors'],
                     max_iteration_steps=20,
                     debug = False,
                     down_sample = 0.0001,
                     termination_ratio=config['skeletonize']['termination_ratio'],
                     step_wise_contraction_amplification = config['skeletonize']['init_contraction'],
                     max_contraction = config['skeletonize']['max_contraction'],
                     max_attraction = config['skeletonize']['max_attraction'])
            lbc.extract_skeleton()
            # Debug/Visualization
            # lbc.visualize()
            contracted = lbc.contracted_point_cloud
            lbc_pcd = lbc.pcd
            total_shift = arr(lbc_pcd.points)-arr(contracted.points)
            save(f'{file_base}_total_shift.pkl',total_shift)
            write_pcd(f'data/skio/results/skio/{file_base}_contracted.pcd',contracted)

            topo=extract_topology(lbc.contracted_point_cloud)
            save_line_set(topo[0],file_base)
            import pickle 
            try:
                with open(f'data/skio/results/skio/{file_base}_topo_graph.pkl','rb') as f:
                    pickle.dump(topo[1],f)
            except Exception as e:
                log.info(f'error saving topo {e}')

        except Exception as e:
            log.info(f'error getting lbc {e}')
    return lbc, topo


def contract(in_pcd,shift, invert=False):
    "Translates the points in the "
    pts=arr(in_pcd.points)
    if not invert:
        shifted=[(pt[0]-shift[0],pt[1]-shift[1],pt[2]-shift[2]) for pt, shift in zip(pts,shift)]
    else:
        shifted=[(pt[0]+shift[0],pt[1]+shift[1],pt[2]+shift[2]) for pt, shift in zip(pts,shift)]
    contracted = o3d.geometry.PointCloud()
    contracted.colors = o3d.utility.Vector3dVector(arr(in_pcd.colors))
    contracted.points = o3d.utility.Vector3dVector(shifted)
    return contracted

def get_downsample(file = None, pcd = None, normalize = False):
    if file: pcd = read_pcd(file)
    log.info('Down sampling pcd')
    voxed_down = pcd.voxel_down_sample(voxel_size=.05)
    uni_down = voxed_down.uniform_down_sample(3)
    clean_pcd = remove_color_pts(uni_down,invert = True)
    if normalize: _ = center_and_rotate(clean_pcd)
    return clean_pcd


def reduce_bloom(file_content, **kwargs):
    seed, pcd, clean_pcd, shift_one = file_content
    logdir = "src/logs/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    params = [(lambda sc: sc + (1-sc)/3, 1, '33 inc, 1x'), 
                (lambda sc: sc + (1-sc)/2, 1, '50 inc, 1x'),
                (lambda sc: sc + (1-sc)/3, 1.5, '33 inc, 1.5x'), 
                (lambda sc: sc + (1-sc)/2, 1.5, '50 inc, 1.5x'),
                (lambda sc: sc + (1-sc)/3, .5, '33 inc, .5x'), 
                (lambda sc: sc + (1-sc)/2, .5, '50 inc, .5x')
                ]

    with writer.as_default():
        for sat_func, sat_cutoff, case_name in params:
            sat_pcd, sat_orig_colors = saturate_colors(pcd, cutoff=sat_cutoff, sc_func=sat_func)
            step+=1
            summary.add_3d('sat_test', to_dict_batch([sat_pcd]), step=step, logdir=logdir)
    breakpoint()


def identify_epiphytes(file_content, save_gif=False, out_path = 'data/results/gif/'):
    logdir = "src/logs/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    seed, pcd, clean_pcd, shift_one = file_content
    assert shift_one is not None # No shifts for this seed; Ensure you pass get_shifts=True to file_info_to_pcds
    run_name = f'{seed}_id_epi'
    step=0
    try:
        with writer.as_default():
            log.info('Calculating/drawing contraction')
            step+=1
            summary.add_3d(run_name, to_dict_batch([clean_pcd]), step=step, logdir=logdir)
            orig_colors = deepcopy(arr(clean_pcd.colors))
            highc, lowc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif)
            clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
            lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
            highc = clean_pcd.select_by_index(highc_idxs, invert=False)
            step+=1
            summary.add_3d(run_name, to_dict_batch([lowc]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([highc]), step=step, logdir=logdir)

            high_shift = shift_one[highc_idxs]
            z_mag = np.array([x[2] for x in high_shift])
            leaves_idxs, leaves, epis = split_on_percentile(highc,z_mag,60, color_on_percentile=True)
            
            high_shift = shift_one[highc_idxs]
            z_mag = np.array([x[2] for x in high_shift])
            leaves_idxs, leaves, epis = color_on_percentile(highc,z_mag,60)
            # epis_idxs, epis, leaves = color_on_percentile(highc,z_mag,60, comp=lambda x,y:x<y)
            pcd_no_epi = join_pcds([highc,leaves])[0]
            step+=1
            summary.add_3d(run_name, to_dict_batch([epis]), step=step, logdir=logdir)
            step+=1
            summary.add_3d(run_name, to_dict_batch([pcd_no_epi]), step=step, logdir=logdir)
           
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
    except Exception as e:
        breakpoint()
        log.info(f'error getting epiphytes for {seed}: {e}')
    return []

def file_info_to_pcds(file_info,
                        normalize = False,
                        get_shifts = True,
                        get_clean_pcd =True,
                        get_contracted = False,
                        topo_data = False,
                        qsm_data = False,
                        base_dir = '/media/penguaman/writable/SyncedBackup/Research/projects/skio/py_qsm',
                        **kwarg_dict):
    # base_dir = 'data/skio'
    detail_ext_dir = f'{base_dir}/ext_detail/'
    shift_dir = f'{base_dir}/pepi_shift/'
    addnl_skel_dir = f'{base_dir}/results/skio/skels2/'
    topo_dir = f'{base_dir}/results/'

    seed, (pcd_file, shift_file_one,shift_file_two) = file_info
    log.info('')
    log.info(f' {shift_file_one=},{shift_file_two=},{pcd_file=}')
    log.info('loading shifts')
    shift_one = None
    if get_shifts:
        try:
            shift_one = load(shift_file_one, root_dir=shift_dir)
        except Exception as e:
            shift_one = None
            print(f'Error getting shift for seed {seed}: {e}')
    if get_contracted:
        try:
            contracted = read_pcd(f'{pcd_file}', root_dir=addnl_skel_dir)
        except Exception as e:
            print(f'Error getting contracted for seed {seed}: {e}')
    if topo_data:
        try:
            topo = load_line_set(f'{pcd_file}', root_dir=topo_dir)
        except Exception as e:
            print(f'Error getting topo for seed {seed}: {e}')
    if qsm_data:
        try:
            qsm_data = load(f'{pcd_file}', root_dir=detail_ext_dir)   
        except Exception as e:
            print(f'Error getting qsm_data for seed {seed}: {e}')
    
    log.info('loading pcd')
    pcd = read_pcd(f'{detail_ext_dir}/{pcd_file}')
    log.info('downsampling/coloring pcd')
    clean_pcd = None
    if get_clean_pcd:   
        clean_pcd = get_downsample(pcd=pcd,normalize=normalize)
    return seed, pcd, clean_pcd, shift_one

def get_seed_id_from_file(file, seed_pat = re.compile('.*seed([0-9]{1,3}).*')):
    return re.match(seed_pat,file).groups(1)[0]

def loop_over_files(func,args = [], kwargs =[],
                    requested_pcds=[],
                    requested_seeds=[],
                    skip_seeds = [],
                    base_dir = '/media/penguaman/writable/SyncedBackup/Research/projects/skio/py_qsm',
                    detail_ext_folder = 'ext_detail',
                    shift_folder = 'pepi_shift'
                    ):
    # reads in the files from the indicated directories
    if not requested_pcds:
        # Get files present in pipeline directories
        detail_ext_dir = f'{base_dir}/{detail_ext_folder}/'
        shift_dir = f'{base_dir}/{shift_folder}/'
        detail_files = glob('*detail*',root_dir=detail_ext_dir)
        shift_one_files = glob('*shift*',root_dir=shift_dir)
        # Get files by seed id 
        seed_to_shift = {get_seed_id_from_file(file):file for file in shift_one_files}
        seed_to_detail = {get_seed_id_from_file(file):file for file in detail_files}
        seed_to_content = {seed:(detail,seed_to_shift.get(seed)) for seed,detail in seed_to_detail.items()}
        
    else:
        seed_to_content = {seed:(seed,pcd,None,None) for seed,pcd in enumerate(requested_pcds)}
    print(detail_files)
    print(seed_to_content)
                    
    if args ==[]: args = ['']*len(kwargs)
    inputs = [(arg,kwarg) for arg,kwarg in zip(list_if(args),list_if(kwargs))]

    results = []
    for file_info in seed_to_content.items():
        try:
            seed, file_content = file_info
            log.info(f'processing seed {seed}')
            if ((requested_seeds==[] or int(seed) in requested_seeds)
                and int(seed) not in skip_seeds):
                if not requested_pcds:
                    file_content = file_info_to_pcds(file_info,detail_ext_dir=detail_ext_dir,shift_dir=shift_dir,**kwargs)
                breakpoint()
                if len(inputs) == 0:
                    result  = func(file_content)
                for arg_tup, kwarg_dict in inputs:
                    result  = func(file_content,*arg_tup,**kwarg_dict)
                    results.append(result)
        except Exception as e:
            log.info(f'error with {seed}: {e}')
    # test = [file_info_to_pcds(file_info) for file_info in seed_to_content.items()]
    log.info(results)
    breakpoint()
    myTable = create_table(results)
    print(myTable)
    breakpoint()    
    print('dont finish yet')
    return results

if __name__ =="__main__":

    base_dir = '/media/penguaman/writable/SyncedBackup/Research/projects/skio/py_qsm'
    detail_ext_dir = f'{base_dir}/ext_detail/'
    shift_dir = f'{base_dir}/pepi_shift/'
    addnl_skel_dir = f'{base_dir}/results/skio/skels2/'
    loop_over_files( reduce_bloom, #identify_epiphytes,]=
                    kwargs={'save_gif':True},
                    base_dir=base_dir,
                    )
    breakpoint()


    # import time 
    # from geometry.zoom import filter_to_region_pcds, zoom_pcd
    # import laspy
    # mv_drive='/media/penguaman/TOSHIBA EXT/tls_lidar/MonteVerde'
    # # file = 'CR-ET6-Crop.las'
    # file = 'EpiphytusTV4.pts'
    # # mv_drive = 'data/epip/inputs'
    # # file = 'cleaned_ds10_epip.pcd'
    # las = laspy.read(f'{mv_drive}/{file}')
    # # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(arr(las.xyz))
    # # pcd.colors = o3d.utility.Vector3dVector(arr(np.stack([las.red,las.green,las.blue],axis=1)/255))
    # try:
    #     las.write(f'/{file.replace('.pts','.las')}')
    # except Exception as e:
    #     log.info(f'error writing las {e}')
    # breakpoint()
    
    # # mv_drive='data/epip/inputs/' pcd.colors = o3d.utility.Vector3dVector(arr(np.stack([las.red,las.green,las.blue],axis=1)/255))
    # # file = 'EpiphytusTV4.pts'
    # full = read_pcd(f'{mv_drive}/{file}')
    # # cuts out unneeded area
    # print('read',time.time())
    # full = full.uniform_down_sample(10)
    # print('downsampled',time.time())
    # # full_z = zoom_pcd([[0,120,-25],[70,200,11]],full) # original used 
    # breakpoint()

    # full_z = zoom_pcd([[0,120,-25],[40,165,11]],full)
    # print('zoomed',time.time())
    # clean_full2 = clean_cloud(full_z)
    # full_znw =  remove_color_pts(clean_full2, lambda x: sum(x)>2.7,invert=True)
    # print('removed colors',time.time())
    # write_pcd('/media/penguaman/code/ActualCode/Research/pyQSM/data/epip/inputs/clean_twice_ds10_epip.pcd',clean_full2)

    # breakpoint()
    # write_pcd('/media/penguaman/code/ActualCode/Research/pyQSM/data/epip/inputs/epi_zoomed.pcd',full)
    
    # pcd = zoom_pcd([[0,120,-25],[70,200,11]],clean_full2)

    
    # write_pcd('./TreeLearn/data/collective_clean.pcd',clean_full)
    # breakpoint()
    # full = read_pcd('data/skeletor/inputs/skeletor_full_ds2.pcd')
    # trunk = read_pcd('data/skeletor/inputs/skeletor_trunk_isolated.pcd')
    # trunk = trunk.uniform_down_sample(6)
    # nbr_pcd = read_pcd('data/skeletor/exts/skel_branch_0_orig_detail_ds2.pcd')
    # import pickle 
    # with open('data/skeletor/seeds/skeletor_final_branch_seeds.pkl','rb') as f:
    #     labeled_cluster_pcds = pickle.load(f)
    # clusters = []
    # for idc, pt_list in labeled_cluster_pcds:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(arr(pt_list))
    #     clusters.append((idc,pcd))

    # res = extend_seed_clusters(clusters,full,file_label='skel_tb_test',cycles=125,save_every=60,draw_every=200,tb_every=2,exclude_pcd=trunk)
    