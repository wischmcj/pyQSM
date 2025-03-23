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


# # Use unequal length, width and height. 
# points = np.random.rand(30000, 3) * [1,2,3]
# colors = np.random.rand(30000, 3)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# scene = o3d.visualization.Visualizer()
# scene.create_window()
# scene.add_geometry(pcd)
# scene.update_renderer()
# scene.poll_events()
# scene.run()
# view_control = scene.get_view_control()
# cam = view_control.convert_to_pinhole_camera_parameters()
# scene.destroy_window()

# p = Projector(pcd)
# rgbd = p.project_to_rgbd(cam.intrinsic.width,
#                   cam.intrinsic.height,
#                   cam.intrinsic.intrinsic_matrix,
#                   cam.extrinsic,
#                   1000,
#                   10)
# pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=cam.intrinsic, extrinsic=cam.extrinsic, project_valid_depth_only=True)
# scene = o3d.visualization.Visualizer()
# scene.create_window()
# scene.add_geometry(pcd1)
# scene.run()


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
        draw(highc)
        draw(lowc)
        log.info('giffing')
        gif_kwargs = {'on_frames': on_frames,
                        'off_frames': off_frames, 
                        'addnl_frame_duration':addnl_frame_duration,
                        'point_size':point_size,
                        'save':save_gif,
                        'out_path':out_path}
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

def draw_two_shifts(file_info,dir):
    seed, (pcd_file, shift_file_one,shift_file_two) = file_info
    log.info('')
    log.info(f' {shift_file_one=},{shift_file_two=},{pcd_file=}')
    log.info('loading shifts')
    shift_one = load(shift_file_one)
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
    draw(skeleton_one)
    skeleton_two = contract(clean_lowc,shift_two)
    draw(skeleton_two)
    skeleton_two_full = contract(clean_lowc,np.sum(shift_two_total,axis=0))
    draw(skeleton_two_full)
    draw_skel(clean_pcd,seed,shift_one, point_size=2,down_sample=False)
    draw_skel(clean_lowc,seed,shift_two,point_size=2,down_sample=False)47

    extracting topologies
    topo= extract_topology(skeleton_one)
    topo= extract_topology(skeleton_two)
    topo= extract_topology(skeleton_two_full)
    draw([topo[0],lowc])
    draw(highc)
    breakpoint()

    log.info('loading shift')                
    get_shift(lowc,seed,iters=15,debug=False)    

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

if __name__ =="__main__":
    # assess_skeletons()
    # breakpoint()
    file = 'data/results/skio/full_ext_seed111_rf3_orig_detail.pcd'
    test = read_pcd(file)
    new_colors,sc = color_dist2(arr(test.colors),cutoff=.01,elev=40, azim=110, roll=0, 
                                                 space='none',min_s=.2,sat_correction=2 )
    
    # highc_idxs, highs,lows = color_on_percentile(test, sc, 30)
    # draw(test)
    # draw(highs)
    # draw(lows)

    # breakpoint()
    loop_over_files(draw_saturation)
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