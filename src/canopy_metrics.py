import open3d as o3d
import numpy as np
from numpy import asarray as arr
from glob import glob
import re
import random

from collections import defaultdict

import open3d as o3d
import numpy as np
from numpy import asarray as arr

import matplotlib.pyplot as plt
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

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
from utils.io import load, load_line_set,save_line_set
from viz.viz_utils import color_continuous_map, draw, rotating_compare_gif
from viz.color import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors   
import cv2
from math import floor
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv

from viz.color import remove_color_pts, get_green_surfaces

from geometry.point_cloud_processing import join_pcds

class Projector:
    def __init__(self, cloud) -> None:
        self.cloud = cloud
        self.points = np.asarray(cloud.points)
        self.colors = np.asarray(cloud.colors)
        self.n = len(self.points)


    # intri 3x3, extr 4x4
    def project_to_rgbd(self,
                        width,
                        height,
                        intrinsic,
                        extrinsic,
                        depth_scale,
                        depth_max
                        ):
        depth = np.zeros((height, width, 1), dtype=np.uint16)
        color = np.zeros((height, width, 3), dtype=np.uint8)

        # The commented code here is vectorized but is missing the filtering at the end where projected points are
        # outside image bounds and depth bounds.
        # world_points = np.asarray(self.points).transpose()
        # world_points = np.append(world_points, np.ones((1, world_points.shape[1])), axis=0)
        # points_in_ccs = np.matmul(extrinsic, world_points)[:3]
        # points_in_ccs = points_in_ccs / points_in_ccs[2, :]
        # projected_points = np.matmul(intrinsic, points_in_ccs)
        # projected_points = projected_points.astype(np.int16)

        for i in range(0, self.n):
            point4d = np.append(self.points[i], 1)
            new_point4d = np.matmul(extrinsic, point4d)
            point3d = new_point4d[:-1]
            zc = point3d[2]
            new_point3d = np.matmul(intrinsic, point3d)
            new_point3d = new_point3d/new_point3d[2]
            u = int(round(new_point3d[0]))
            v = int(round(new_point3d[1]))

            # Fixed u, v checks. u should be checked for width
            if (u < 0 or u > width - 1 or v < 0 or v > height - 1 or zc <= 0 or zc > depth_max):
                continue

            d = zc * depth_scale
            depth[v, u ] = d
            color[v, u, :] = self.colors[i] * 255

        im_color = o3d.geometry.Image(color)
        im_depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000.0, depth_trunc=depth_max, convert_rgb_to_intensity=False)
        return rgbd

def project_to_rgbd(file_info,dir):
    seed, (pcd_file, shift_file_one,shift_file_two) = file_info
    orig_pcd, pcd, shift_one = file_info_to_pcds(file_info)
    # Use unequal length, width and height. 

    scene = o3d.visualization.Visualizer()
    scene.create_window()
    scene.add_geometry(pcd)
    scene.update_renderer()
    scene.poll_events()
    scene.run()
    view_control = scene.get_view_control()
    cam = view_control.convert_to_pinhole_camera_parameters()
    scene.destroy_window()

    p = Projector(pcd)
    rgbd = p.project_to_rgbd(cam.intrinsic.width,
                    cam.intrinsic.height,
                    cam.intrinsic.intrinsic_matrix,
                    cam.extrinsic,
                    1000,
                    10)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=cam.intrinsic, extrinsic=cam.extrinsic, project_valid_depth_only=True)
    scene = o3d.visualization.Visualizer()
    scene.create_window()
    scene.add_geometry(pcd1)
    scene.run()


def color_on_percentile(pcd,
                        val_list,
                        pctile,
                        comp=lambda x,y:x>y ):
    if len(pcd.points)!= len(val_list):
        msg = f'length of val list does not match size of pcd'
        log.error(f'length of val list does not match size of pcd')
        raise ValueError(msg)

    color_continuous_map(pcd,val_list)
    val = np.percentile(val_list,pctile)
    highc_idxs = np.where(comp(val_list,val))[0]
    highc_pcd = pcd.select_by_index(highc_idxs,invert=False)
    lowc_pcd = pcd.select_by_index(highc_idxs,invert=True)
    return highc_idxs, highc_pcd,lowc_pcd

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

def draw_shift(pcd, 
                seed,
                shift,
                down_sample=True,
                draw_results=True, 
                save_gif=False, 
                out_path = None,
                on_frames=25,
                off_frames=25,
                addnl_frame_duration=.05,
                point_size=5):
    out_path = out_path or f'data/results/gif/{seed}'
    clean_pcd = pcd
    if down_sample: clean_pcd = get_downsample(pcd=pcd, normalize=False)
    c_mag = np.array([np.linalg.norm(x) for x in shift])
    highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70)

    log.info('preping contraction/coloring')
    if draw_results:
        log.info('giffing')
        gif_kwargs = {'on_frames': on_frames, 'off_frames': off_frames, 
                        'addnl_frame_duration':addnl_frame_duration, 'point_size':point_size,
                        'save':save_gif, 'out_path':out_path}
        # rotating_compare_gif(contracted,clean_pcd, output=out_path, output=out_path,save = save_gif,on_frames = 15,off_frames=3, addnl_frame_duration=.05)
        rotating_compare_gif(highc,lowc, output=out_path,**gif_kwargs)

    #extrapolating point shift to the more detailed pcd
    # # voxed_down = pcd.voxel_down_sample(voxel_size=.01)
    # pcd, nbrs = get_neighbors_kdtree(highc_pcd,orig,k=50)
    
    # highc_pts = [arr(highc_pcd.points)]
    # orig_shifts = [np.mean(c_mag[nbr_list]) for nbr_list in nbrs]
    return highc,lowc

    # # nowhite_custom =remove_color_pts(pcd, lambda x: sum(x)>2.3,invert=True)
    # draw(highc_detail

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

def draw_two_shifts(file_info,dir):
    seed, (pcd_file, shift_file_one,shift_file_two) = file_info
    pcd, clean_pcd, shift_one = file_info_to_pcds(file_info)
    shift_two_total = load(shift_file_two,dir)
    shift_two = shift_two_total[0]
    log.info('loading pcd')
    pcd = read_pcd(f'data/results/skio/{pcd_file}')
    log.info('downsampling/coloring pcd')
    
    clean_pcd = get_downsample(pcd=pcd)
    corrected_colors, sc = color_dist2(clean_pcd)
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

def loop_over_files(func,requested_seeds=[]):
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

    for files in seed_to_content.items():
        seed, (pcd_file, shift_file_one,shift_file_two) = files
        log.info(f'processing seed {seed}')
        if requested_seeds==[] or int(seed) in requested_seeds:
            try:
                func(files, dir)
            except Exception as e:
                breakpoint()
                log.info(f'error with {seed},{e}')

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
    breakpoint()
    return pcd, clean_pcd, shift_one

def project2d(file_info,dir):
    pcd, clean_pcd, shift_one =file_info_to_pcds(file_info)
    
    clean_pcd = get_downsample(pcd=pcd)
    orig_color = arr(clean_pcd.colors)
    c_mag = np.array([np.linalg.norm(x) for x in shift_one])
    highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70) 
    low_color = orig_color[np.where(c_mag<np.percentile(c_mag,70))]
    hi_color = orig_color[np.where(c_mag>=np.percentile(c_mag,70))]
    low_c_mag = c_mag[np.where(c_mag<np.percentile(c_mag,70))]
    tree,not_tree = highc,lowc 
    # draw([lowc,highc])
    

    # target.colors = o3d.utility.Vector3dVector(low_color)
    color_conds = {
        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5
        ,'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3
        ,'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4
        ,'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3
        ,'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2
        ,'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5
        ,'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3
    }
    # reds = lambda tup: tup[0]<.5 and tup[1]<=.5
    # green_purple = lambda tup: tup[0]>=.5 and tup[0]<=.75
    #greys = lambda tup: tup[2]<.25 
    # greens = lambda tup: tup[0]<.5 and tup[0]>1/6
    # greens = lambda tup: tup[0]<.45 and tup[0]>1/6
    # yellow/red = lambda tup: tup[0]<1/6
    # breakpoint()
    clean_pcd.colors = o3d.utility.Vector3dVector(orig_color)
    lowc.colors = o3d.utility.Vector3dVector(low_color)
    highc.colors = o3d.utility.Vector3dVector(hi_color)

    target = lowc
    # draw(target)
    corrected_colors, sc = color_dist2(arr(target.colors),cutoff=1,sc_func =lambda sc: sc + (1-sc)/3)
    target.colors = o3d.utility.Vector3dVector(corrected_colors)
    lowc = target
    
    target = highc
    # draw(target)
    corrected_colors, sc = color_dist2(arr(target.colors),cutoff=1,sc_func =lambda sc: sc + (1-sc)/3)
    target.colors = o3d.utility.Vector3dVector(corrected_colors)
    highc = target

    color_compare(arr(lowc.colors),arr(highc.colors),color_conds=color_conds,in_rgb = False)

    
    # downed =  #pcd.uniform_down_sample(2)
    # targets = [lowc,highc] # clean_cloud(downed)
    # for target in targets:
    #     draw(target)
    #     orig_colors = arr(target.colors)
    #     corrected_colors, sc = color_dist2(arr(target.colors),cutoff=1,sc_func =lambda sc: sc + (1-sc)/3)
    #     target.colors = o3d.utility.Vector3dVector(corrected_colors)
    #     hsv = arr(rgb_to_hsv(corrected_colors)) #tup[0]<.5 or tup[0]>5/6 or

    #     hues = ['white','blues','pink','red_yellow', 'greens']
    #     hue_pcds = [None]*(len(hues)+2)
    #     no_hue_pcds = [None]*(len(hues)+2)
    #     hue_idxs = [None]*(len(hues)+1)
    #     no_hue_pcds[0] = target
    #     hue_pcds[0] =target
    #     for idh, hue in enumerate(hues): hue_pcds[idh+1],no_hue_pcds[idh+1],hue_idxs[idh] = get_color_by_hue(no_hue_pcds[idh], color_conds[hue])
    #     sizes = [len(arr(x.points)) for x in hue_pcds[1:]]
    #     hue_by_idhs = {hue: idhs for hue,idhs in zip(hues,hue_idxs)}

    # for idh, hue in enumerate(hues): draw(no_hue_pcds[idh+1])
    # for idh, hue in enumerate(hues): draw(hue_pcds[idh+1])
    
    # trunk = no_hue_pcds[len(hues)]
    # moss = hue_pcds[len(hues)]
    # trunk_and_moss = join_pcds([trunk,moss])
    # draw(trunk_and_moss)
    # draw(trunk)
    # draw(moss)

    # for color,color_details in color_dict.items(): draw(lowc.select_by_index(color_details['ids'],invert=True))
    
    breakpoint()
    # masked = lowc.select_by_index(color_dict[color]['ids'],invert=True)
    # draw(lowc)
    # draw(masked)

    # masked = lowc.select_by_index(color_dict[color]['ids'])
    # color_dict = isolate_color(low_color)

def color_compare(in_colors_one, in_colors_two,color_conds, cutoff=.01,elev=40, azim=110, roll=0, 
                space='none',min_s=.2,sat_correction=2,
                sc_func =lambda sc: sc + (1-sc)/2, in_rgb = False ):
    hsv_one = arr(rgb_to_hsv(in_colors_one))
    hsv_two = arr(rgb_to_hsv(in_colors_two))
    if in_rgb:
        hsv_one=in_colors_one
        hsv_two=in_colors_two

    rands = np.random.sample(len(in_colors_one))
    in_colors_one = arr(in_colors_one)[rands<cutoff]
    hsv_one = arr(hsv_one)[rands<cutoff]

    rands = np.random.sample(len(in_colors_two))
    in_colors_two = arr(in_colors_two)[rands<cutoff]
    hsv_two = arr(hsv_two)[rands<cutoff]
   
    # hsv_two[:,2] = hsv_two
    
    hco,sco,vco = zip(*hsv_one)
    hct,sct,vct = zip(*hsv_two)

    # breakpoint()
    fig = plt.figure(figsize=(12, 9))
    # row=row+1
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    # breakpoint()
    axis.scatter(hco, sco, vco, facecolors=in_colors_one, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(elev=elev, azim=azim, roll=roll)

    axis = fig.add_subplot(1, 2,2, projection="3d")
    axis.scatter(hct, sct, vct, facecolors=in_colors_two, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(elev=elev, azim=azim, roll=roll)
    plt.show()

def get_color_by_hue(pcd,condition):
    target = pcd
    orig_colors = arr(target.colors)
    hsv = arr(rgb_to_hsv(arr(target.colors))) #tup[0]<.5 or tup[0]>5/6 or 
    hue_idxs,in_vals  = [x for x in zip(*[(idt,tup) for idt, tup in enumerate(hsv) if condition(tup) ])]
    target.colors = o3d.utility.Vector3dVector(orig_colors)
    hue = target.select_by_index(hue_idxs,invert=False)
    no_hue = target.select_by_index(hue_idxs,invert=True)
    # draw(hue)
    # draw(no_hue)
    return hue, no_hue, hue_idxs

def isolate_color(in_colors,icolor='white',get_all=True, std='hsv'):
    color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],'white': [[180, 18, 255], [0, 0, 231]],'red1': [[180, 255, 255], [159, 50, 70]],'red2': [[9, 255, 255], [0, 50, 70]],'green': [[89, 255, 255], [36, 50, 70]],'blue': [[128, 255, 255], [90, 50, 70]],'yellow': [[35, 255, 255], [25, 50, 70]],'purple': [[158, 255, 255], [129, 50, 70]],'orange': [[24, 255, 255], [10, 50, 70]],'gray': [[180, 18, 230], [0, 0, 40]]}        
    if std == 'hsv':
        hsv = arr(rgb_to_hsv(in_colors))
        color_dict_HSV ={'white':[[2/3,2/3,1],[0,0,0]], 'black': [[1/6,1,1],[0,.75,.75]],'red': [[.7,1,1],[0,.5,0.5]],
                          'green': [[.5,1,1],[1/6,.5,.5]],'blue': [[5/6,1,1],[.5,.5,.5]],}
        # color_dict_HSV= {col:arr(rgb_to_hsv(arr(vals)/255)) for col,vals in color_dict_HSV.items()}
        # color_dict_HSV ={'white': [[0,0,1],[0,0,0.75]] ,'silver': [[0,0,0.75],[0,0,0.5]] ,'gray': [[0,0,0.5],[0,0,0]] ,
                            # 'black': [[0,0,0],[0,1,1]] ,'red': [[0,1,1],[0,.5,0.5]] ,'maroon': [[0,1,0.5],[1/6,1,1]] ,'yellow': [[1/6,1,1],[1/6,1,0.5]] ,'olive': [[1/6,1,0.5],[1/3,1,1]] ,
        # 'lime': [[1/3,1,1],[1/3,1,0.5]] ,'green': [[1/3,1,0.5],[0.5,1,1]] ,
        # 'aqua': [[0.5,1,1],[0.5,1,0.5]] ,'teal': [[0.5,1,0.5],[2/3,1,1]] ,
        # 'blue': [[2/3,1,1],[2/3,1,0.5]] ,'navy': [[2/3,1,0.5],[5/6,1,1]] ,'fuchsia': [[5/6,1,1],[5/6,1,0.5]] ,'purple': [[5/6,1,0.5],[100,1,1]]}
        # color_dict_HSV = {col:[[vals[0][0],1,1],[vals[1][0],0,0]] for col,vals in color_dict_HSV.items()}
    
    else:
        hsv = arr(rgb_to_hsv(in_colors))
        hsv = in_colors*255

    if icolor not in [x for x in color_dict_HSV.keys()]: 
        raise ValueError(f'Color {icolor} not found in ranges, use one of {clist}')
    clist=[x for x in color_dict_HSV.keys()]
    ret={}
    in_range=  lambda tup,ub,lb: all([lbv<val<ubv for val,lbv,ubv in zip(tup,ub,lb)])
    test = []
    for color,(ub,lb) in color_dict_HSV.items(): #test.append({color:{'ids':idt,'cols':arr(tup)} for idt,tup in zip(*[(idt,tup) for idt, tup in enumerate(hsv) if in_range(tup,lb,ub)])})
        if get_all or color ==icolor: 
            in_hsv = [x for x in zip(*[(idt,tup) for idt, tup in enumerate(hsv) if in_range(tup,lb,ub)])]
            if len(in_hsv)>0:
                in_ids,in_vals = in_hsv
                color_details = {color: {'ids':in_ids,'cols':arr(in_vals)}}
                ret.update(color_details)
    for mydict in test: ret.update(mydict)
    # new_hsv.append(idc) 
    # new_hsv= hsv[np.logical_and(hsv<upper,hsv>lower)]
    # rgb = arr(hsv_to_rgb(new_hsv))
    log.info({col:len(x['ids']) for col, x in ret.items()})
    return ret

    non_white_idxs = remove_color(low_color)
    
    # center_and_rotate(clean_pcd)
    # gif_kwargs = {'on_frames': on_frames,'off_frames': off_frames, 'addnl_frame_duration':addnl_frame_duration,'point_size':point_size,'save':save_gif,'out_path':out_path}
    # rotating_compare_gif(skeleton,clean_pcd, **gif_kwargs)
    
    breakpoint()
    
def plot_color(file_info,dir):
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
    
    clean_pcd = get_downsample(pcd=pcd)
    orig_color = arr(clean_pcd.colors)
    c_mag = np.array([np.linalg.norm(x) for x in shift_one])
    highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70) 
    low_color = orig_color[np.where(c_mag<np.percentile(c_mag,70))]
    low_c_mag = c_mag[np.where(c_mag<np.percentile(c_mag,70))]
    tree,not_tree = highc,lowc 

    # corrected_colors, sc = color_dist2(arr(clean_pcd.colors))
    # color_continuous_map(clean_pcd,sc)
    draw(clean_pcd)
    # adjust saturation
    corrected_colors, sc = color_dist2(arr(lowc.colors),cutoff=.01)
    # color_continuous_map(lowc,sc)
    draw(lowc)
    # breakpoint()
    cutoff = .2
    # in_colors = arr(lowc.colors)
    lowc.color = o3d.utility.Vector3dVector(corrected_colors)
    in_colors = arr(low_color)
    low_pts = arr(lowc.points)
    rands = np.random.sample(len(in_colors))
    new_colors = arr(in_colors)[rands<cutoff]
    new_c_mag = arr(low_c_mag)[rands<cutoff]
    new_low_pts = arr(low_pts)[rands<cutoff]
    sc= arr(sc)[rands<cutoff]
    # corrected_rgb =  arr(in_colors)[rands<cutoff]

    fig = plt.figure(figsize=(12, 9))
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    axis.view_init(elev=15, azim=90, roll=0)
    axis.scatter(new_c_mag, sc*1.4, new_low_pts[:,2], facecolors=new_colors, marker=".",s=.1)
    plt.show()
    # axis.scatter(c_mag, sc, np.zeros_like(sc), facecolors=new_colors, marker=".",s=.1)


    breakpoint()
    breakpoint()

    # log.info('loading shift')                
    # get_shift(lowc,seed,iters=15,debug=False)   
    #  
# def cluster_colors():
#     kmeans(arr([(x,y) for x,y,_ in hsv_colors]),3)

def color_dist2(in_colors,cutoff=.01,elev=40, azim=110, roll=0, 
                space='none',min_s=.2,sat_correction=2,sc_func =lambda sc: sc + (1-sc)/2):
    data = []
    hsv = arr(rgb_to_hsv(in_colors))
    hc,sc,vc = zip(*hsv)
    sc = arr(sc)
    vc = arr(vc)
    # low_saturation_idxs = np.where(sc<min_s)[0]
    # sc[sc<min_s] = sc[sc<min_s]*sat_correction
    ret_sc = sc_func(sc)
    # vc =   sc_func(vc)
    # sc = sc*.6
    corrected_rgb_full = arr(hsv_to_rgb([x for x in zip(hc,ret_sc,vc)]))
    
#    15     lower_blue = np.array([110,50,50])
#    16     upper_blue = np.array([130,255,255])
#    17 
#    18     # Threshold the HSV image to get only blue colors
#    19     mask = cv2.inRange(hsv, lower_blue, upper_blue)

    rands = np.random.sample(len(in_colors))
    in_colors = arr(in_colors)[rands<cutoff]
    for sids,series in enumerate(data):
        data[sids] = arr(series)[rands<cutoff]
    corrected_rgb =  arr(corrected_rgb_full)[rands<cutoff]
    # osc = arr(osc)[rands<cutoff]
    ## RGB
    if space=='rgb':
        pixel_colors = in_colors
        r, g, b = zip(*in_colors)
        # r, g, b = cv2.split(pixel_colors)
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.scatter(r, g, b, facecolors=in_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()

    # HSV 
    if space=='hsv':
        hsv = arr(rgb_to_hsv(in_colors))
        # hsv = hsv[hsv[:,1]>min_s]
        hc,sc,vc = zip(*hsv)
        import math
        # sc[sc<.5] = sc[sc<.5]*1.5
        rows = 1+ math.ceil(len(data)/2)
        series = [vc]
        series.extend(data)
        # breakpoint()
        fig = plt.figure(figsize=(12, 9))
        for row in range(rows):
            # row=row+1
            axis = fig.add_subplot(rows, 2, int((row+1)), projection="3d")
            z = series[row]
            # breakpoint()
            axis.scatter(hc, sc, z, facecolors=in_colors, marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Saturation")
            axis.set_zlabel("Value")
            axis.view_init(elev=elev, azim=azim, roll=roll)

            axis = fig.add_subplot(rows, 2, int(row+2), projection="3d")
            axis.scatter(hc, sc, z, facecolors=corrected_rgb, marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Saturation")
            axis.set_zlabel("Value")
            axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()
    return corrected_rgb_full,ret_sc

if __name__ =="__main__":
    # plot_color()
    # breakpoint()
    # file = 'data/results/skio/full_ext_seed111_rf3_orig_detail.pcd'
    # test = read_pcd(file)
    # data = [arr(test.points)[:,2]]
    # data = []
    # # data =[]
    # new_colors,sc = color_dist2(arr(test.colors),elev=40, azim=110, roll=0, 
    #                                              space='hsv',min_s=.2,sat_correction=2 )
    # breakpoint()
    
    # highc_idxs, highs,lows = color_on_percentile(test, sc, 30)
    # draw(test)
    # draw(highs)
    # draw(lows)

    # breakpoint()
    # loop_over_files(project2d)
    loop_over_files(project_to_rgbd)
    # run_skeleton_cases()
    # seed = 135
    
    breakpoint()
    # shift_res = get_shift(test,135,iters=11)#,debug=True)
    # breakpoint()
    # tp_shift = load(f'3_1.1_seed135_two_voxpt05_uni3_shift',dir = 'data/results/skio/')
    # shift = load(f'4_2_seed135_two_voxpt05_uni3_shift',dir = 'data/results/skio/')
    # draw_shift(test,seed,np.sum(tp_shift,axis=0))
    # breakpoint()
    # assess_skeletons()
    # breakpoint()