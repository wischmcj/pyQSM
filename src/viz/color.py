from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np
from numpy import array as arr, mean
import scipy.spatial as sps
import  open3d as o3d
from open3d.geometry import 

from canopy_metrics import get_downsample
from utils.io import load
from viz.viz_utils import color_continuous_map, draw
from set_config import log, config
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

def homog_colors(pcd):
    colors = arr(pcd.colors)
    pts=arr(pcd.points)
    white_idxs = [idc for idc,color in enumerate(colors) if sum(color)>2.7]
    white_pts = [pts[idc] for idc in white_idxs]

    non_white_tree = pcd.select_by_index(white_idxs, invert=True)
    tree = sps.KDTree(arr(non_white_tree.points))
    white_nbrs = tree.query([white_pts],30)
    avg_neighbor_color = mean(colors[ white_nbrs[1]][0],axis=1)
    for white_idx,avg_color in zip(white_idxs,avg_neighbor_color):  colors[white_idx] = avg_color
    pcd.colors =  o3d.utility.Vector3dVector(colors)
    draw(pcd)
    
def remove_color_pts(pcd,
                        color_lambda = lambda x: sum(x)>2.7,
                        invert=False):
    colors = arr(pcd.colors)
    ids = [idc for idc, color in enumerate(colors) if color_lambda(color)]
    new_pcd = pcd.select_by_index(ids, invert = invert)
    return new_pcd

def get_green_surfaces(pcd,invert=False):
    green_pcd= remove_color_pts(pcd,    lambda rgb: rgb[1]>rgb[0] and rgb[1]>rgb[2] and 0.5<(rgb[0]/rgb[2])<2, invert)
    return green_pcd

def mute_colors(pcd):
    colors = arr(pcd.colors)
    rnd_colors = [[round(a,2) for a in col] for col in colors]
    pcd.colors= o3d.utility.Vector3dVector(rnd_colors)
    draw(pcd)

def bin_colors(zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):
    # binning colors in pcd, finding most common
    # round(2) reduces 350k to ~47k colors
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) 
                if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       
                    and pt[1]>zoom_region[1][0] 
                    and  pt[1]<zoom_region[1][1])]

    cols = [','.join([f'{round(y,1)}' for y in x[1]]) for x in region]
    d_cols, cnts = np.unique(cols, return_counts=True)
    print(len(d_cols))
    print((max(cnts),min(cnts)))
    # removing points of any color other than the most common
    most_common = d_cols[np.where(cnts)]
    most_common_rgb = [tuple((float(num) for num in col.split(','))) for col in most_common]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) in most_common_rgb ]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) != tuple((1.0,1.0,1.0)) ]
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in color_limited]))
    limited_pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in color_limited]))
    draw(limited_pcd)


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
    c_mag = np.array([np.linalg.norm(x) for x in shift_one])
    highc_idxs, highc,lowc = color_on_percentile(clean_pcd,c_mag,70) 

    tree,not_tree = highc,lowc 

    corrected_colors, sc = color_dist2(arr(clean_pcd.colors))
    color_continuous_map(clean_pcd,sc)
    draw(clean_pcd)

    corrected_colors, sc = color_dist2(arr(lowc.colors))
    color_continuous_map(lowc,sc)
    draw(lowc)

    in_colors
    rands = np.random.sample(len(in_colors))
    in_colors = arr(in_colors)[rands<cutoff]
    corrected_rgb =  arr(corrected_rgb_full)[rands<cutoff]
    fig = plt.figure(figsize=(12, 9))
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    axis.scatter(c_mag, sc, np.zeros_like(sc), facecolors=in_colors, marker=".")


    breakpoint()
    breakpoint()

    # log.info('loading shift')                
    # get_shift(lowc,seed,iters=15,debug=False)   
    #  
# def cluster_colors():
#     kmeans(arr([(x,y) for x,y,_ in hsv_colors]),3)

def color_dist2(in_colors, cutoff=.01,elev=40, azim=110, roll=0, 
                                                 space='none',min_s=.2,sat_correction=2 ):

    hsv = arr(rgb_to_hsv(in_colors))
    hc,sc,vc = zip(*hsv)
    sc = arr(sc)
    # low_saturation_idxs = np.where(sc<min_s)[0]
    # sc[sc<min_s] = sc[sc<min_s]*sat_correction
    sc = sc + (1-sc)/2
    corrected_rgb_full = arr(hsv_to_rgb([x for x in zip(hc,sc,vc)]))
    

    rands = np.random.sample(len(in_colors))
    in_colors = arr(in_colors)[rands<cutoff]
    corrected_rgb =  arr(corrected_rgb_full)[rands<cutoff]

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
        
        # sc[sc<.5] = sc[sc<.5]*1.5
        fig = plt.figure(figsize=(12, 9))
        axis = fig.add_subplot(1, 2, 1, projection="3d")
        # breakpoint()
        axis.scatter(hc, sc, vc, facecolors=in_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        axis.view_init(elev=elev, azim=azim, roll=roll)

        axis = fig.add_subplot(1, 2,figsize=(12, 9))
        axis = fig.add_subplot(1, 2, 1, projection="3d")
        axis.scatter(hc, sc, vc, facecolors=corrected_rgb, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()
    return corrected_rgb_full,sc
