from copy import deepcopy
from glob import glob
import re
import random

from collections import defaultdict

import open3d as o3d
import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

from geometry.surf_reconstruction import get_mesh
from set_config import config, log
from reconstruction import get_neighbors_kdtree
from utils.math_utils import (
    get_center,
    generate_grid
)
from utils.fit import kmeans,cluster_DBSCAN
from geometry.skeletonize import extract_skeleton, extract_topology
from geometry.point_cloud_processing import ( filter_by_norm,
    clean_cloud,
    crop, get_shape,
    orientation_from_norms,
    filter_by_norm,
    get_ball_mesh,
    crop_by_percentile,
    cluster_plus
)
from geometry.mesh_processing import ( 
    check_properties
)
from utils.io import load, load_line_set,save_line_set
from viz.viz_utils import color_continuous_map, draw, rotating_compare_gif

from viz.color import (
    remove_color_pts, 
    get_green_surfaces,
    color_on_percentile,
    color_distribution,
    segment_hues,
    saturate_colors
)

from geometry.mesh_processing import get_surface_clusters
from reconstruction import recover_original_details
from ray_casting import sparse_cast_w_intersections, project_to_image,mri,cast_rays

color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,
               'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,
               'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,
               'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,
               'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,
               'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,
               'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])

def create_alpha_shapes(file_content, save_gif=False, out_path = 'data/results/gif/'):
    seed, pcd, clean_pcd, shift_one = file_content
    log.info('Calculating/drawing contraction')
    orig_colors = arr(clean_pcd.colors)
    lowc, highc, highc_idxs = draw_shift(clean_pcd,seed,shift_one,save_gif=save_gif)
    clean_pcd.colors = o3d.utility.Vector3dVector(orig_colors)
    lowc = clean_pcd.select_by_index(highc_idxs, invert=True)
    log.info('Extrapoloating contraction to original pcd')
    # proped_cmag = propogate_shift(pcd,clean_pcd,shift_one)

    log.info('Orienting, extracting hues')
    center_and_rotate(lowc) 
    hue_pcds,no_hue_pcds =segment_hues(lowc,seed,hues=['white','blues'],draw_gif=True, save_gif=save_gif)
    no_hue_pcds = [x for x in no_hue_pcds if x is not None]
    target = no_hue_pcds[len(no_hue_pcds)-1]
    # draw(target)
    log.info('creating alpha shapes')

    get_mesh(pcd,lowc,target)
    
  

    log.info('attempting alpha shape by pivoting ball')
    # o3d.visualization.draw_geometries([test], mesh_show_back_face=True)
    ######Ordered small to large leads to more,smaller triangles and increased coverage
    breakpoint()
    # mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
  
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
        if len(nbr_list)>0: proped_c_mag[idnl] = np.mean(arr(c_mag_high[nbr_list]))
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
                down_sample=False,draw_results=True, save_gif=False, out_path = None,
                on_frames=25, off_frames=25, addnl_frame_duration=.01, point_size=5):
    out_path = out_path or f'data/results/gif/'
    clean_pcd = pcd
    if down_sample: clean_pcd = get_downsample(pcd=clean_pcd, normalize=False)
    c_mag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70)
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

def loop_over_files(func,requested_seeds=[],skip_seeds = []):
    dir = 'data/results/skio/skels2/'
    seed_pat = re.compile('.*seed([0-9]{1,3}).*')
    detail_files = glob('*detail*',root_dir='data/results/skio/')
    shift_two_files = glob('*shift*',root_dir=dir)
    shift_one_files = glob('*shift*',root_dir='data/results/skio/')
    
    seed_to_detail = {re.match(seed_pat,file).groups(1)[0]:file for file in detail_files}
    # for seed,file in seed_to_detail.items(): get_shift(read_pcd(file),seed)
    seed_to_shift_one = {re.match(seed_pat,file).groups(1)[0]:file for file in shift_one_files}
    seed_to_shift_two = {re.match(seed_pat,file).groups(1)[0]:file for file in shift_two_files}
    seed_to_files = [(seed,(seed_to_detail.get(seed),seed_to_shift_one.get(seed),seed_to_shift_two.get(seed)))  for seed in seed_to_detail.keys() ]
    seed_to_content = {seed:(detail,shift_one,shift_two) for seed,(detail,shift_one,shift_two) in seed_to_files}

    for file_info in seed_to_content.items():
        seed, _ = file_info
        log.info(f'processing seed {seed}')
        if ((requested_seeds==[] or int(seed) in requested_seeds)
            and int(seed) not in skip_seeds):
            file_content = file_info_to_pcds(file_info)
            # try:
            func(file_content)
            # except Exception as e:
                # breakpoint()
                # log.info(f'error with {seed},{e}')

def file_info_to_pcds(file_info,normalize = False ):
    seed, (pcd_file, shift_file_one,shift_file_two) = file_info
    log.info('')
    log.info(f' {shift_file_one=},{shift_file_two=},{pcd_file=}')
    log.info('loading shifts')
    shift_one = load(shift_file_one)
    # shift_two_total = load(shift_file_two,dir)
    # shift_two = shift_two_total[0]
    log.info('loading pcd')
    pcd = read_pcd(f'data/results/skio/{pcd_file}')
    log.info('downsampling/coloring pcd')
    clean_pcd = get_downsample(pcd=pcd,normalize=normalize)
    return seed, pcd, clean_pcd, shift_one

def get_epiphyes(file_info,dir):
    pcd, clean_pcd, shift_one =file_info_to_pcds(file_info)

    orig_color = arr(clean_pcd.colors)
    c_mag = np.array([np.linalg.norm(x) for x in shift_one])
    low_idxs = np.where(c_mag<np.percentile(c_mag,70))
    lowc = clean_pcd.select_by_index(low_idxs)
    highc = clean_pcd.select_by_index(low_idxs,invert=True)
    low_color = arr(lowc.color)
    hi_color  = arr(highc.color)
    # highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70) 
    
    targets = [highc, lowc]
    color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
    for target in targets:        
        corrected_colors, sc = color_distribution(arr(target.colors),cutoff=1,sc_func =lambda sc: sc + (1-sc)/3)
        target.colors = o3d.utility.Vector3dVector(corrected_colors)
        lowc = target
    
    target = highc
    # draw(target)
    corrected_colors, sc = color_distribution(arr(target.colors),cutoff=1,sc_func =lambda sc: sc + (1-sc)/3)
    target.colors = o3d.utility.Vector3dVector(corrected_colors)
    highc = target

    color_compare(arr(lowc.colors),arr(highc.colors),color_conds=color_conds,in_rgb = False)
    #hue_pcds,no_hue_pcds =segment_hues(clean_pcd)

    # clean_pcd.colors = o3d.utility.Vector3dVector(orig_color)
    # lowc.colors = o3d.utility.Vector3dVector(low_color)
    # highc.colors = o3d.utility.Vector3dVector(hi_color)
    # for idh, hue in enumerate(hues): draw(no_hue_pcds[idh+1])
    # for idh, hue in enumerate(hues): draw(hue_pcds[idh+1])

    # trunk = no_hue_pcds[len(hues)]
    # moss = hue_pcds[len(hues)]
    # trunk_and_moss = join_pcds([trunk,moss])
    # draw(trunk_and_moss)
    # draw(trunk)
    # draw(moss)
    breakpoint()
    # masked = lowc.select_by_index(color_dict[color]['ids'],invert=True)

def draw_two_shifts(file_info,dir):
    seed, (pcd_file, shift_file_one,shift_file_two) = file_info
    pcd, clean_pcd, shift_one = file_info_to_pcds(file_info)
    shift_two_total = load(shift_file_two,dir)
    shift_two = shift_two_total[0]
    log.info('loading pcd')
    pcd = read_pcd(f'data/results/skio/{pcd_file}')
    log.info('downsampling/coloring pcd')
    
    clean_pcd = get_downsample(pcd=pcd)
    corrected_colors, sc = color_distribution(clean_pcd)
    draw(clean_pcd)
    color_continuous_map(clean_pcd,sc)
    draw(clean_pcd)


    c_mag = np.array([np.linalg.norm(x) for x in shift_one])
    highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70) 

    
    draw(lowc)

    clean_lowc = get_downsample(pcd=lowc)
    c_mag_two = np.array([np.linalg.norm(x) for x in shift_two])
    highc_idxs2, highc2,lowc2 = color_on_percentile(clean_lowc,c_mag_two,70) 
    draw(lowc2)
    # breakpoint()
    
    log.info('creating skeletons')  
    skeleton_one = contract(clean_pcd,shift_one)
    skeleton_two = contract(clean_lowc,shift_two)
    skeleton_two_full = contract(clean_lowc,np.sum(shift_two_total,axis=0))
    draw(skeleton_two_full)
    draw_skel(clean_pcd,seed,shift_one, point_size=2,down_sample=False)

    # extracting topologies
    topo= extract_topology(skeleton_one)
    topo= extract_topology(skeleton_two)
    topo= extract_topology(skeleton_two_full)
    draw([topo[0],lowc])
    draw(highc)
    breakpoint()

    log.info('loading shift')                
    get_shift(lowc,seed,iters=15,debug=False)    

def get_canopy_coverage_area(pcd):
    down_sample = pcd.uniform_down_sample(20)
    clean_pcd = clean_cloud(down_sample,iters = 1)
    draw(clean_pcd)
    upcd = o3d.geometry.sample_points_uniformly(clean_pcd,100000)
    mesh = pivot_ball_mesh(clean_pcd,[1.2,1.5,1.7,2,2.2,2.5,2.7,3])

def get_shift(pcd, seed,
              contraction,
              attraction,
              iters=15, 
              debug=True):
    """
        Orig. run with contraction_factor 3, attraction .6, max contraction 2080 max attraction 1024
    """
    file_base = f'{str(contraction).replace('.','pt')}_{str(attraction).replace('.1','pt')}/seed{seed}_two_voxpt05_uni3'
    log.info(f'getting shift for {seed}')
    orig = pcd
    test = orig.voxel_down_sample(voxel_size=.05)
    test = test.uniform_down_sample(3)
    test = remove_color_pts(test,invert = True)
    ratio = len(arr(test.points))/len(arr(orig.points))
    log.info(f'final ratio {ratio}')
    skel_res = extract_skeleton(test, max_iter = iters, debug=debug, cmag_save_file=file_base,
                                    contraction_factor=contraction,
                                    attraction_factor=attraction)
    # contracted, total_point_shift, shift_by_step = skel_res
    # breakpoint()
    try:
        topo = extract_topology(skel_res[0])
        save_line_set(topo[0],file_base)
    except Exception as e:
        log.info(f'error getting topo {e}')
    return skel_res, topo
    # save(f'shifts_{file}.pkl', (total_point_shift,shift_by_step))

def center_and_rotate(pcd, center=None):
    center = pcd.get_center() if center is None else center
    rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    pcd.translate(np.array([-x for x in center ]))
    pcd.rotate(rot_90_x)
    pcd.rotate(rot_90_x)
    pcd.rotate(rot_90_x)
    return center

def contract(in_pcd,shift):
    "Translates the points in the "
    pts=arr(in_pcd.points)
    shifted=[(pt[0]-shift[0],pt[1]-shift[1],pt[2]-shift[2]) for pt, shift in zip(pts,shift)]
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

def draw_skel(pcd, 
                seed,
                shift=[], 
                down_sample=True,
                skeleton = None,
                save_gif=False, 
                out_path = None,
                on_frames=25,
                off_frames=25,
                addnl_frame_duration=.05,
                point_size=5):
    out_path = out_path or f'data/results/gif/{seed}'
    clean_pcd = pcd
    if down_sample: clean_pcd = get_downsample(pcd=pcd, normalize=False)
    if not skeleton:
        skeleton = contract(clean_pcd,shift)
    draw(skeleton)
    # center_and_rotate(skeleton)
    # center_and_rotate(clean_pcd)
    # _ = center_and_rotate(skeleton) #Rotation makes contraction harder
    gif_kwargs = {'on_frames': on_frames,'off_frames': off_frames, 'addnl_frame_duration':addnl_frame_duration,'point_size':point_size,'save':save_gif,'out_path':out_path}
    # rotating_compare_gif(skeleton,clean_pcd, **gif_kwargs)
    # rotating_compare_gif(contracted,clean_pcd, point_size=4, output=out_path,save = save_gif, on_frames = 50,off_frames=50, addnl_frame_duration=.03)
    return skeleton

def identify_epiphytes(file, pcd, shift):

    green = get_green_surfaces(pcd)
    not_green = get_green_surfaces(pcd,True)
    # draw(lowc_pcd)
    draw(green)
    draw(not_green)


    c_mag = np.array([np.linalg.norm(x) for x in shift])
    z_mag = np.array([x[2] for x in shift])

    highc_idxs, highc, lowc = color_on_percentile(pcd,c_mag,70)

    z_cutoff = np.percentile(z_mag,80)
    log.info(f'{z_cutoff=}')
    low_idxs = np.where(z_mag<=z_cutoff)[0]
    lowc = clean_pcd.select_by_index(low_idxs)
    ztrimmed_shift = shift[low_idxs]
    ztrimmed_cmag = c_mag[low_idxs]
    draw(lowc)
    highc_idxs, highc, lowc = color_on_percentile(lowc,c_mag,70)


    # color_continuous_map(test,c_mag)
    highc_idxs = np.where(c_mag>np.percentile(c_mag,70))[0]
    highc_pcd = test.select_by_index(highc_idxs)
    lowc_pcd = test.select_by_index(highc_idxs,invert=True)
    draw([lowc_pcd])
    draw([highc_pcd])

    breakpoint()
    
    lowc_detail = get_neighbors_kdtree(pcd,lowc_pcd)
    draw(lowc_detail)
    breakpoint()

def assess_skeletons():
    file_types = ['lines'
                    ,'points'
                    #  ,'shift'
                    ,'tpshift']
    seed_pat = re.compile('.*seed([0-9]{1,3}).*')
    factors_pat = re.compile('([0-9]*\\.?[0-9]*_[0-9]*\\.?[0-9]*).*')
    root_dir = 'data/results/skio/3_1pt1/'

    files = defaultdict(list)
    # for ftype in file_types:
    points_files = glob('*_points.pkl',root_dir=root_dir)
    for f in points_files:
        try:
            if '135' not in f:
                factors = re.match(factors_pat,f).groups(1)[0]
                seed = re.match(seed_pat,f).groups(1)[0]
                base_name = f.replace('_points.pkl','')
                tp_shift = load(f'{base_name}_tpshift.pkl',dir = root_dir)
                total_shift = np.sum(arr(tp_shift),axis=0)
                # topo = load_line_set(base_name,root_dir)
                detail_file = glob(f'data/results/skio/full_ext_seed{seed}*')[0]
                pcd = read_pcd(detail_file)
                
                clean_pcd = get_downsample(pcd=pcd, normalize=True)
                skeleton = contract(clean_pcd,tp_shift[0])
                topo= extract_topology(skeleton)
                breakpoint()
                # draw(topo)
                log.info(f)
                # detail_file = glob(f'data/results/skio/{factors}_{base_name}_orig_detail.pcd')
                draw_skel(pcd,seed,tp_shift[0])#,skeleton=topo)
                draw_shift(pcd,seed,tp_shift[0])
                breakpoint()
        except Exception as e:
            log.info(f'error running seed {seed}: {e}. Skipping...')

def run_skeleton_cases():
    seed = 135
    file = 'data/results/skio/full_ext_seed135_rf11_orig_detail.pcd'
    test = read_pcd(file)
    config_list = [#(1,1), # lvl 5 monkey puzzle, has vertical lines for moss
                   (2,1),   # monkey puzzle, not enough contraction
                   (4,1),  #
                   (1,1),  #
                   (1,2),  # lvl6 monkey puzzle, has vertical lines for moss
                   (1,4),  #
                   (1,1),  #
                   (4,3),  # lvl 3 monkey puzzle,
                   (4,2),  #
                   (2,.5), #
                   (3,.8),  # lvl4 monkey puzzle, has vertical lines for moss
                   (.5,2), # lvl6 monkey puzzle, has vertical lines for moss
                   (.8,3), #
                   (5,6),  #
                   (6,5),] #
    res = []
    for cont, att in config_list:
        try:
            shift_res = get_shift(test,135,iters=15,
                                  contraction = cont, attraction = att,
                                  debug=True)
            res.append(shift_res)
        except Exception as e:
            log.info(f'error getting shift for {seed}:{e}')

if __name__ =="__main__":
    # plot_color()
    # breakpoint()
    # file = 'data/results/skio/full_ext_seed111_rf3_orig_detail.pcd'
    # test = read_pcd(file)
    # data = [arr(test.points)[:,2]]
    # data = []
    # # data =[]
    # new_colors,sc = color_distribution(arr(test.colors),elev=40, azim=110, roll=0, 
    #                                              space='hsv',min_s=.2,sat_correction=2 )
    # breakpoint()
    
    # highc_idxs, highs,lows = color_on_percentile(test, sc, 30)
    # draw(test)
    # draw(highs)
    # draw(lows)

    # breakpoint()
    # loop_over_files(project2d)
    # loop_over_files(project_to_rgbd)
    loop_over_files(create_alpha_shapes, skip_seeds = [189,151,113,
                                                       133,191,137,134,135])
    ###Goods
    #113, 136**
    ##Needs Work
    # 135, alpha shape looks really skinny and sparse 
    #       seems like this tree was far away from the scanner and one side is missing
    # 191, tighter alpha value, trunk bases are one solid object
    # 134, color removal removes too much
    ### Bads
    #133
    breakpoint()
    # shift_res = get_shift(test,135,iters=11)#,debug=True)
    # breakpoint()
    # tp_shift = load(f'3_1.1_seed135_two_voxpt05_uni3_shift',dir = 'data/results/skio/')
    # shift = load(f'4_2_seed135_two_voxpt05_uni3_shift',dir = 'data/results/skio/')
    # draw_shift(test,seed,np.sum(tp_shift,axis=0))
    # breakpoint()
    # assess_skeletons()
    # breakpoint()